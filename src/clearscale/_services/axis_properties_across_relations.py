"""Reconciliation and validation of axis properties across coordinate systems."""

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

from clearscale._axis_values import AxisKey, OrderedAxes, Unit
from clearscale.types import AxisKeyT
from clearscale._services.units import is_same_unit, unit_conversion_factor

from clearscale._services.axis_links import IDENTITY_TOLERANCE, AxisLinks
from clearscale._spatial_relations import SpatialRelation
from clearscale._transforms._base import OmeZarrAxes, OmeZarrAxis

OmeZarrAxesParam = Union[Literal["infer"], OmeZarrAxes, Mapping[AxisKeyT, OmeZarrAxis]]


def normalize_ome_zarr_axes_param(ome_zarr_axes: Optional[OmeZarrAxesParam], keys: OrderedAxes) -> OmeZarrAxes:
    """Return OmeZarrAxes with `keys`. Blank if param None, default properties if param 'infer'.
    Preserve ome_zarr_axes instance if no changes."""
    if isinstance(ome_zarr_axes, str):
        if ome_zarr_axes != "infer":
            raise ValueError(
                f"ome_zarr_axes must be 'infer', None, or {{axis_key : ome_zarr.Axis}}, not {ome_zarr_axes!r}"
            )
        return OmeZarrAxes.fromkeys(keys).with_types_inferred()
    elif ome_zarr_axes is None:
        return OmeZarrAxes.fromkeys(keys)
    elif isinstance(ome_zarr_axes, OmeZarrAxes):  # preserve instance
        return ome_zarr_axes.with_axes(keys)
    else:
        return OmeZarrAxes(ome_zarr_axes).with_axes(keys)


def reconcile_axis_prop_params(
    target_axes: Tuple[AxisKey, ...],
    links: AxisLinks,
    source_ome_axes: OmeZarrAxes,
    provided_ome_axes: Optional[OmeZarrAxesParam],
    provided_unit: Optional[Union[Unit, Mapping[AxisKeyT, str]]],
) -> OmeZarrAxes:
    """
    Three-way merge of axis properties between
    - carryover from source_ome_axes into the target_axes keys according to `links`,
    - provided_ome_axes and
    - provided_unit
    as required by any method that creates new coordinate systems and hence accepts the properties defining it.

    provided_ome_axes and provided_unit must match for axes where both are non-blank.
    Between provided_ome_axes and carryover from source_ome_axes:
    Any provided fields override carryover. In "infer" mode, infer axis types for inserted axes only.
    """
    unit = Unit.empty(target_axes) if provided_unit is None else Unit(provided_unit)
    parsed_ome_axes = normalize_ome_zarr_axes_param(provided_ome_axes, target_axes)

    conflicts = parsed_ome_axes.conflicts_with_unit(unit)
    if conflicts:
        raise ValueError(
            f"Conflicting unit between `unit` and `ome_zarr_axes`: "
            f"{ {a: f'{slf!r} vs {oth!r}' for a, (slf, oth) in conflicts.items()} }. Only specify unit once."
        )

    carried_ome_axes = source_ome_axes.with_axes_carried_over(target_axes, links)
    if provided_ome_axes == "infer":
        result = carried_ome_axes
        inserted_axes = {axis for axis, link in zip(target_axes, links.links) if not link.sources}
        if inserted_axes:
            inserted_ome_axes = parsed_ome_axes.without_axes_except(inserted_axes)
            result = result.with_blanks_filled_from(inserted_ome_axes)
    else:
        result = parsed_ome_axes.with_axes(target_axes).with_blanks_filled_from(carried_ome_axes)

    return result.with_unit_merged(unit)


def merge_ome_zarr_axes(axes_list: Sequence[OmeZarrAxes], *, sources: str, conflict_hint: str) -> OmeZarrAxes:
    """Merge properties from multiple sources, rejecting any mismatches across the list.
    Appropriate when multiple axis definitions flow into one common coordinate system, i.e.
    Scale validation in Multiscale, or hub-and-spokes graphs in Scene."""
    axes = list(axes_list[0].keys())
    merged: Dict[AxisKey, Dict[str, Any]] = {a: {} for a in axes}
    for field_name in ("unit", "type", "discrete", "long_name"):
        for a in axes:
            values = list(
                dict.fromkeys(v for ome_axes in axes_list if (v := getattr(ome_axes[a], field_name)) not in (None, ""))
            )
            if field_name == "unit":
                # Normalise different spellings so "nm" and "nanometer" merge fine.
                # First spelling wins.
                values = [v for i, v in enumerate(values) if not any(is_same_unit(v, w) for w in values[:i])]
            if len(values) > 1:
                raise ValueError(
                    f"Conflicting {field_name!r} for axis {a!r} across {sources}: {set(values)!r}. {conflict_hint}"
                )
            merged[a][field_name] = next(iter(values), None)
    return OmeZarrAxes([(a, OmeZarrAxis(name=a, **merged[a])) for a in axes])


def _unit_mismatch(source_unit: str, target_unit: str, scale: float) -> Optional[str]:
    """Why a target axis in `target_unit` cannot be `scale` times its source axis in `source_unit` (up to the sign,
    which a flip changes), if it can't.

    - If the axis is not scaled, the units must be the same.
    - If it is, and the units differ, `scale` must be the factor that converts one to the other.
    - If the units are the same, any scale is possible: Relating an image to an atlas can include real scaling.
    - Units that are not known or cannot be converted (space to time) cannot be checked."""
    if abs(abs(scale) - 1) <= IDENTITY_TOLERANCE:
        return None if is_same_unit(source_unit, target_unit) else ""
    if is_same_unit(source_unit, target_unit):
        return None
    try:
        factor = unit_conversion_factor(source_unit, target_unit)
    except ValueError:
        return None
    if math.isclose(abs(scale), factor, rel_tol=1e-9):
        return None
    return f"scaled by {abs(scale):g}, but {source_unit} to {target_unit} is {factor:g}"


def _type_mismatch(source_type: str, target_type: str, scale: float) -> Optional[str]:
    """Scaling an axis does not change what type of axis it is, so the types must be the same whatever the scale."""
    return None if source_type == target_type else ""


@dataclass(frozen=True)
class _AxisPropertyValidation:
    """Validation policy for a single property of OmeZarrAxis"""

    field_name: str
    mismatch: Callable[[str, str, float], Optional[str]]
    """(property_value_source, property_value_target, scaling_factor)
    Both property values are non-empty. Return None if no mismatch, otherwise mismatch reason (or empty string)."""
    plural: str  # for error messages
    consequence: str  # for error messages


_EXACT_VALIDATION_POLICIES = (
    _AxisPropertyValidation(
        field_name="unit",  # OmeZarrAxis.unit
        mismatch=_unit_mismatch,
        plural="units",
        consequence=(
            "but no relation converts between them. clearscale does not convert units. State the conversion explicitly, "
            "using an `Affine` (e.g. `Affine.from_unit_conversion(source_unit, target_unit)`) in the relation, or use "
            "the same unit for both. If the Affine also scales for other reasons, leave one of the units unspecified."
        ),
    ),
    _AxisPropertyValidation(
        field_name="type",  # OmeZarrAxis.type
        mismatch=_type_mismatch,
        plural="types",
        consequence=(
            "but the relation keeps these axes as they are, and an axis cannot change its type. "
            "Use the same type for both, or leave one of them unspecified."
        ),
    ),
)


def require_consistent_axis_properties(
    source_axes: OmeZarrAxes,
    target_axes: OmeZarrAxes,
    relation: SpatialRelation,
    *,
    source_label: str,
    target_label: str,
) -> None:
    """
    Raise if `relation` can't be a true statement because source and target axis properties do not line up with the
    axis linkage implied by the relation.

    Validates source and target unit:
    * When the target is exactly that source, unscaled (see `AxisLink.exact`): Unit must be same.
    * When the target is exactly that source, scaled: Unit must be same, or scaling must be the conversion factor.
    * Converting unit and simultaneously scaling by another factor is rejected.
    If x is in micrometer in one space, and nanometer in another, then there *must* be a scaling relation between them.
    `Affine` is the only SpatialRelation that can express scaling.
    The user has to state the scaling conversion explicitly as an `Affine`, rather than e.g. clearscale auto-inferring it.

    Validates axis type: When the target axis is exactly one source, they must have the same type.

    Blank values mean "unspecified" (as in OME-Zarr), so they never conflict.
    Axes present on only one side (inserts, drops) never conflict.
    """
    source_keys = tuple(source_axes)
    links = relation._axis_links(source_keys)
    assert links.target_axes == tuple(
        target_axes
    ), f"Received links for {links.target_axes!r}, not {tuple(target_axes)!r}."

    for policy in _EXACT_VALIDATION_POLICIES:
        conflicts: List[str] = []
        for target, link in zip(links.target_axes, links.links):
            if link.exact is None:
                continue
            source, scale = link.exact
            source_value = getattr(source_axes[source], policy.field_name)
            target_value = getattr(target_axes[target], policy.field_name)
            if not (source_value and target_value):
                continue
            reason = policy.mismatch(source_value, target_value, scale)
            if reason is not None:
                conflicts.append(
                    f"{target!r}: {source_value!r} vs {target_value!r}" + (f" ({reason})" if reason else "")
                )
        if conflicts:
            raise ValueError(
                f"The {policy.plural} of {source_label} and {target_label} differ ({', '.join(conflicts)}), "
                f"{policy.consequence}"
            )
