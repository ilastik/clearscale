"""Reconciliation and validation of axis properties across coordinate systems."""

from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Tuple, Union

from clearscale._axis_values import AxisKey, OrderedAxes, Unit
from clearscale.types import AxisKeyT
from clearscale._services.units import is_same_unit

from clearscale._affines import Affine
from clearscale._services.axis_links import AxisLinks
from clearscale._spatial_relations import SpatialRelation, SpatialRelationSequence
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


def require_matching_units_or_affine_relation(
    source_axes: OmeZarrAxes,
    target_axes: OmeZarrAxes,
    relation: Optional[SpatialRelation],
    *,
    source_label: str,
    target_label: str,
) -> None:
    """
    Raise if `relation` can't be a true statement because two non-blank units on a shared axis differ.

    Without an `Affine` in `relation`, the connection between the two spaces is the identity along every shared axis,
    which is false if the units differ.
    If x is in micrometer in one space, and nanometer in another, then there *must* be a scaling relation between them.
    `Affine` is the only SpatialRelation that can express scaling.
    The user has to state the scaling conversion explicitly as an `Affine`, rather than e.g. clearscale auto-inferring it.

    Whether the Affine is numerically correct for the conversion is not validated - any Affine bypasses the
    check for unit mismatch.

    Blank units mean "unspecified" (as in OME-Zarr), so they never conflict. Axes present on only one side never conflict.
    """

    def _contains_affine(relation: Optional[SpatialRelation]) -> bool:
        if isinstance(relation, Affine):
            return True
        if isinstance(relation, SpatialRelationSequence):
            return any(_contains_affine(r) for r in relation)
        return False

    if _contains_affine(relation):
        return
    implicit_target_axes = source_axes
    if relation:
        links = relation._axis_links(source_axes)
        implicit_target_axes = source_axes.with_axes_carried_over(tuple(target_axes), links)
    conflicts = implicit_target_axes.conflicts_with_unit(target_axes.get_unit())
    if not conflicts:
        return
    details = ", ".join(f"{axis!r}: {src!r} vs {tgt!r}" for axis, (src, tgt) in conflicts.items())
    raise ValueError(
        f"The units of {source_label} and {target_label} differ ({details}), but no relation converts between them. "
        "clearscale does not convert units. State the conversion explicitly, using an `Affine` "
        "(e.g. `Affine.from_unit_conversion(source_unit, target_unit)`) in the relation, or use the same unit for both."
    )
