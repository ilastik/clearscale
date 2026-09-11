import math
import string
import warnings
from abc import ABC
from collections import OrderedDict, defaultdict
from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import (
    Optional,
    TypeVar,
    Mapping,
    Generic,
    Union,
    Sequence,
    Set,
    Callable,
    Iterable,
    List,
    Tuple,
    Literal,
    Dict,
    Any,
    Hashable,
    Iterator,
    ItemsView,
    KeysView,
    ValuesView,
    Collection,
)

from clearscale._axis_values import (
    Shape,
    Factor,
    PixelSize,
    Unit,
    Translation,
    PixelOffset,
    ShapeLike,
    FactorLike,
    Axes,
    RoundingMethod,
    OrderedAxes,
    AxisKey,
    AxisKeyT,
    _axis_in,
)
from clearscale._errors import NoSuchCoordinateSystemError
from clearscale._spatial_relations import SpatialRelation
from clearscale._transforms import (
    CoordinateSystemName,
    CoordinateSystem,
    TransformGraph,
    NodeRef,
    IdentityTransform,
    TransformGraphNode,
    PRE_TRANSFORMS_VERSIONS,
    Transform,
    relation_chain_target_axes,
    relations_to_transform,
    ScaleTransform,
    TransformSequence,
    ProjectAxisTransform,
    TranslationTransform,
    MapAxisTransform,
    OmeZarrAxes,
    OmeZarrAxis,
)
from clearscale._services import ome_zarr, precomputed

ScaleKey = str
ValueType = TypeVar("ValueType", Shape, Factor, "Scale")
AxisValuesType = TypeVar("AxisValuesType", Shape, Factor)
_ScaleMappingSelf = TypeVar("_ScaleMappingSelf", bound="_ScaleMapping[Any]")
_ScaledAxisValuesSelf = TypeVar("_ScaledAxisValuesSelf", bound="_ScaledAxisValues[Any]")
DEFAULT_NAME_PATTERN = "s{}"

TranslationShiftFunction = Callable[["Scale", "Scale"], "Translation"]
"""
base_scale: the reference scale being transformed from
target_scale: the new scale being created (with 0 translation)
Returns: target_scale's translation
"""
OmeZarrAxesParam = Union[Literal["infer"], OmeZarrAxes, Mapping[AxisKeyT, OmeZarrAxis]]


class DuplicatePolicy(str, Enum):
    ERROR = "error"
    """Raise if two scale keys have the same shape or factor value."""
    KEEP_ALL = "keep_all"
    """No deduplication. The same shape or factor value can appear multiple times under different scale keys."""
    # Both KEEP_ policies are identical when generating blueprints, because the scale keys are rewritten consecutively anyway.
    KEEP_FIRST = "keep_first"
    """Deduplicate: For each set of scale keys with the same shape or factor value, keep only the first scale key."""
    KEEP_LAST = "keep_last"
    """Deduplicate: For each set of scale keys with the same shape or factor value, keep only the last scale key."""


def _normalize_ome_zarr_axes_param(ome_zarr_axes: Optional[OmeZarrAxesParam], keys: OrderedAxes) -> OmeZarrAxes:
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


@dataclass(frozen=True, slots=True, init=False)
class Scale:
    shape: Shape
    pixel_size: PixelSize
    unit: Unit
    translation: Translation
    ome_zarr_axes: OmeZarrAxes

    @classmethod
    def from_lists(
        cls,
        keys: Optional[OrderedAxes] = None,
        shape: Optional[Sequence[int]] = None,
        pixel_size: Optional[Sequence[float]] = None,
        unit: Optional[Sequence[str]] = None,
        translation: Optional[Sequence[float]] = None,
        ome_zarr_axes: Optional[Union[Literal["infer"], Sequence[Union[Mapping[str, Any], OmeZarrAxis]]]] = None,
    ) -> "Scale":
        """
        Convenience constructor for axes given as parallel sequences rather than {axis: value} mappings.
        Enables construction of Scales
        * with just axis keys: `Scale.from_lists("zyx")`, `Scale.from_lists("zyx", ome_zarr_axes="infer")`
        * directly from OME-Zarr json: `Scale.from_lists(json["axes"])`

        `keys` is optional if `ome_zarr_axes` is given as a sequence of dicts/ome_zarr.Axis that all
        carry a `name` (or "name" key) - in that case, axis keys are derived from them. Otherwise, `keys` is
        mandatory. Any other argument left as None takes that type's usual default (e.g. shape=None means
        every axis is a singleton).

        `ome_zarr_axes` entries may be either raw OME-Zarr axis dicts (using OME-Zarr's own key
        spelling: "type", "unit", "discrete", "longName", "name") or `ome_zarr.Axis` instances directly.
        """
        parsed_ome_zarr_axes: Optional[List[OmeZarrAxis]]
        if ome_zarr_axes is None or isinstance(ome_zarr_axes, str):
            parsed_ome_zarr_axes = None
        else:
            parsed_ome_zarr_axes = [
                entry if isinstance(entry, OmeZarrAxis) else OmeZarrAxis.from_ome_zarr(entry) for entry in ome_zarr_axes
            ]

        if keys is None:
            if parsed_ome_zarr_axes is None:
                raise ValueError(
                    "Scale.from_lists requires `keys`, unless `ome_zarr_axes` is given as a sequence of "
                    "dicts/ome_zarr.Axis that all specify a name."
                )
            missing_name = [i for i, ax in enumerate(parsed_ome_zarr_axes) if ax.name is None]
            if missing_name:
                raise ValueError(
                    "Scale.from_lists cannot derive `keys` from `ome_zarr_axes`: entries at index "
                    f"{missing_name} have no name. Either provide `keys`, or a name for every entry."
                )
            keys = [str(ax.name) for ax in parsed_ome_zarr_axes]

        keys = list(keys)
        if not keys:
            raise ValueError("Scale.from_lists requires at least one axis key.")
        if len(set(keys)) != len(keys):
            raise ValueError(f"Scale.from_lists requires unique axis keys. Received: {keys!r}")

        Scale._require_matching_length(shape, keys, "shape")
        Scale._require_matching_length(pixel_size, keys, "pixel_size")
        Scale._require_matching_length(unit, keys, "unit")
        Scale._require_matching_length(translation, keys, "translation")

        resolved_ome_zarr_axes: Union[None, Literal["infer"], OmeZarrAxes]
        if parsed_ome_zarr_axes is None:
            # Let __init__ handle != "infer"; instance check to satisfy pyright
            resolved_ome_zarr_axes = ome_zarr_axes if isinstance(ome_zarr_axes, str) else None
        else:
            Scale._require_matching_length(parsed_ome_zarr_axes, keys, "ome_zarr_axes")
            resolved_ome_zarr_axes = OmeZarrAxes(zip(keys, parsed_ome_zarr_axes))

        return cls(
            shape=Shape.all_singletons(keys) if shape is None else Shape(zip(keys, shape)),
            pixel_size=None if pixel_size is None else PixelSize(zip(keys, pixel_size)),
            unit=None if unit is None else Unit(zip(keys, unit)),
            translation=None if translation is None else Translation(zip(keys, translation)),
            ome_zarr_axes=resolved_ome_zarr_axes,
        )

    def __init__(
        self,
        shape: ShapeLike,
        pixel_size: Optional[Union[PixelSize, Mapping[AxisKeyT, float]]] = None,
        unit: Optional[Union[Unit, Mapping[AxisKeyT, str]]] = None,
        translation: Optional[Union[Translation, Mapping[AxisKeyT, float]]] = None,
        ome_zarr_axes: Optional[OmeZarrAxesParam] = None,
    ):
        shape = Shape(shape)
        pixel_size = PixelSize.fromkeys(shape) if pixel_size is None else PixelSize(pixel_size)
        translation = Translation.fromkeys(shape) if translation is None else Translation(translation)
        assert translation is not None, "for pyright"

        parsed_unit = unit if isinstance(unit, Unit) or unit is None else Unit(unit)
        keys = shape.keys()
        parsed_ome_axes = _normalize_ome_zarr_axes_param(ome_zarr_axes, keys)

        if parsed_unit is not None:
            conflicts = parsed_ome_axes.conflicts_with_unit(parsed_unit)
            if conflicts:
                raise ValueError(
                    f"Conflicting unit between `unit` and `ome_zarr_axes`: "
                    f"{ {a: f'{slf!r} vs {oth!r}' for a, (slf, oth) in conflicts.items()} }. Only specify unit once."
                )
            parsed_ome_axes = parsed_ome_axes.with_unit_merged(parsed_unit.with_axes(keys))

        merged_unit = parsed_ome_axes.get_unit()
        final_unit = parsed_unit if parsed_unit == merged_unit else merged_unit  # preserve instance where possible
        assert final_unit is not None, "for pyright"

        if not (shape.keys() == pixel_size.keys() == final_unit.keys() == translation.keys() == parsed_ome_axes.keys()):
            raise ValueError(
                f"Tried to set up invalid scale: Axiskeys differ "
                f"(shape={list(shape.keys())}, pixel_size={list(pixel_size.keys())}, "
                f"translation={list(translation.keys())}, unit={list(final_unit.keys())}, "
                f"ome_zarr_axes={list(parsed_ome_axes.keys())})"
            )
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "pixel_size", pixel_size)
        object.__setattr__(self, "unit", final_unit)
        object.__setattr__(self, "translation", translation)
        object.__setattr__(self, "ome_zarr_axes", parsed_ome_axes)

    def with_axes(self, axes: OrderedAxes, *, infer_inserted_types: bool = False) -> "Scale":
        """Build a Scale with all properties produced by their respective `.with_axes`.
        infer_types: If True, infer OME-Zarr axis types *only for newly inserted axes*.
        If you want to infer for all axes, use `ome_zarr_axes='infer'` during Scale construction."""
        if not axes:
            raise ValueError(f"Cannot create empty {self.__class__.__name__}. Attempted reorder to: {axes!r}")
        new_ome_zarr_axes = self.ome_zarr_axes.with_axes(axes, infer_inserted_types=infer_inserted_types)
        return Scale(
            shape=self.shape.with_axes(axes),
            pixel_size=self.pixel_size.with_axes(axes),
            unit=self.unit.with_axes(axes),
            translation=self.translation.with_axes(axes),
            ome_zarr_axes=new_ome_zarr_axes,
        )

    def has_physical_meta(self):
        return not self.unit.is_default() or not self.pixel_size.is_default()

    def to_display_string(self, name=""):
        shape = ", ".join(f"{axis}: {size}" for axis, size in self.shape.items())
        name_and_shape = f'"{name}" ({shape})' if name else f"{shape}"
        pixel_size = ""
        if self.has_physical_meta():
            axis_strings = []
            for axis in self.shape.keys():
                if axis == "c":
                    continue
                pixel_size = self.pixel_size[axis]
                unit = ""
                if self.unit[axis]:
                    unit = f" {self.unit[axis]}"
                elif axis != "t":
                    unit = " px"
                axis_strings.append(f"{axis}: {pixel_size:g}{unit}")
            pixel_size = " at pixel size: " + ", ".join(axis_strings)
        return f"{name_and_shape}{pixel_size}"

    @staticmethod
    def _require_matching_length(values: Optional[Sequence[Any]], reference: Sequence[Any], param_name: str) -> None:
        if values is not None and len(values) != len(reference):
            raise ValueError(
                f"Scale.from_lists: '{param_name}' has length {len(values)}, "
                f"expected {len(reference)} to match keys={reference!r}"
            )


class _ScaleMapping(ABC, ABCMapping[ScaleKey, ValueType], Generic[ValueType]):
    """Common base class for Multiscale, BlueprintShapes and BlueprintFactors"""

    def __init__(self, *args, **kwargs):
        self._mapping = OrderedDict(*args, **kwargs)
        if not self._mapping:
            raise ValueError(f"Cannot instantiate empty {self.__class__.__name__}")
        if any(v is None for v in self._mapping.values()):
            raise ValueError(f"None values not allowed. Received: {list(self._mapping.values())}")

    def __repr__(self):
        map_substr = self._mapping.__repr__()[len(type(self._mapping).__name__) :]
        return str(self.__class__.__name__) + map_substr

    def __getitem__(self, key: ScaleKey) -> ValueType:
        if key not in self:
            raise KeyError(f"No such scale: '{key}' (available: {list(self.keys())})")
        return self._mapping[key]

    def __contains__(self, key: object) -> bool:
        return key in self._mapping

    def __iter__(self) -> Iterator[ScaleKey]:
        return iter(self._mapping)

    def __len__(self):
        return len(self._mapping)

    def keys(self) -> KeysView[ScaleKey]:
        return self._mapping.keys()

    def values(self) -> ValuesView[ValueType]:
        return self._mapping.values()

    def first_value(self) -> ValueType:
        return next(iter(self.values()))

    def items(self) -> ItemsView[ScaleKey, ValueType]:
        return self._mapping.items()

    def __hash__(self):
        return hash(tuple(self._mapping.items()))

    def __eq__(self, other):
        if isinstance(other, _ScaleMapping):
            return self._mapping == other._mapping
        if isinstance(other, ABCMapping):
            return self._mapping == other
        return False

    def copy(self: _ScaleMappingSelf) -> _ScaleMappingSelf:
        return self.__class__(self._mapping)

    def filter_items(self: _ScaleMappingSelf, keep_func: Callable[[ScaleKey, ValueType], bool]) -> _ScaleMappingSelf:
        items = [(k, v) for k, v in self.items() if keep_func(k, v)]
        return self.__class__(items)

    def with_keys(
        self: _ScaleMappingSelf,
        keys_pattern_or_func: Union[Sequence[ScaleKey], str, Callable[[int, ScaleKey, ValueType], ScaleKey]],
    ) -> _ScaleMappingSelf:
        """
        Assign new scale keys using one of:
        - a sequence of new scale keys (one per current scale, unique)
        - a string format pattern with placeholder for the scale's int index
        - a function that takes the int index, the old scale key, and the Scale object, and returns a new key
        """
        if isinstance(keys_pattern_or_func, str):
            pattern = keys_pattern_or_func
            fields = [field_name for _, field_name, _, _ in string.Formatter().parse(pattern) if field_name is not None]
            if fields != [""]:
                raise ValueError(
                    "Name pattern must contain exactly one anonymous placeholder for the scale index, "
                    f"e.g. 's{{}}' or 's{{:03d}}'. Received: '{pattern}'"
                )
            items = [(keys_pattern_or_func.format(i), v) for i, v in enumerate(self.values())]
            return self.__class__(items)
        elif callable(keys_pattern_or_func):
            new_keys = self._generate_and_validate_new_keys(keys_pattern_or_func)
            return self.__class__(zip(new_keys, self.values()))
        else:
            new_keys = keys_pattern_or_func
            if len(new_keys) != len(self):
                raise ValueError(
                    f"Must provide a new key for every current key: {list(self.keys())}. Received: {new_keys}"
                )
            if not self._all_unique(new_keys):
                raise ValueError(f"All new scale keys must be unique. Received: {new_keys}")
            return self.__class__(zip(new_keys, self.values()))

    def drop_before(self: _ScaleMappingSelf, key: ScaleKey, inclusive=False) -> _ScaleMappingSelf:
        keys = list(self.keys())
        if key not in keys:
            raise KeyError(f"No such scale: '{key}' (available: {keys})")

        start_idx = keys.index(key)
        if inclusive:
            start_idx += 1

        items = [(k, v) for k, v in self.items() if k in keys[start_idx:]]
        return self.__class__(items)

    def _generate_and_validate_new_keys(self, keys_pattern_or_func: Callable) -> List[ScaleKey]:
        new_keys = []
        for i, (key, value) in enumerate(self.items()):
            try:
                new_key = keys_pattern_or_func(i, key, value)
            except TypeError as e:
                if "positional argument" in str(e):
                    raise TypeError(
                        "Key-generating function must accept scale's integer index, "
                        "the old scale key, and the corresponding value object, e.g.: "
                        "lambda i, old_key, factor: f\"scale{i}-{factor['x']}\""
                    ) from e
                raise e
            new_keys.append(new_key)
        if not self._all_unique(new_keys):
            raise ValueError(f"All new scale keys must be unique. Generated: {new_keys}")
        return new_keys

    @staticmethod
    def _all_unique(things: Sequence[Hashable]) -> bool:
        seen = set()
        for item in things:
            if item in seen:
                return False
            seen.add(item)
        return True


class _ScaledAxisValues(_ScaleMapping[AxisValuesType], Generic[AxisValuesType]):
    """Base class for BlueprintShapes and BlueprintFactors"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._mapping:
            return

        if len(set(self._mapping.keys())) != len(self._mapping.keys()):
            raise ValueError(f"Scale keys must be unique. Received: {list(self._mapping.keys())}")

        axes = next(iter(self._mapping.values())).keys()
        for k, v in self._mapping.items():
            if v.keys() != axes:
                raise ValueError(
                    f"All values must have the same axes. (Expected {axes}, received {v.keys()} for key '{k}')"
                )

    def to_dict(self) -> OrderedDict[str, OrderedDict[AxisKey, Union[int, float]]]:
        return OrderedDict([(scale_key, OrderedDict(axis_values)) for scale_key, axis_values in self.items()])

    def _with_values(self: _ScaledAxisValuesSelf, values: Sequence[AxisValuesType]) -> _ScaledAxisValuesSelf:
        return self.__class__(zip(self.keys(), values))

    def _with_values_by_axes(
        self: _ScaledAxisValuesSelf,
        other: Union[ShapeLike, Mapping[ScaleKey, ShapeLike], FactorLike, Mapping[ScaleKey, FactorLike]],
        *,
        only_axes: Optional[Axes] = None,
    ) -> _ScaledAxisValuesSelf:
        if not isinstance(other, ABCMapping):
            raise TypeError(f"Pass {{axis: value}} or {{scale_key: {{axis: value}}}} mapping. Received: {other!r}")
        if not other or (only_axes is not None and not only_axes):
            return self
        is_other_nested = isinstance(next(iter(other.values())), ABCMapping)
        if is_other_nested:
            new_values = [
                axis_values.with_values(other[scale_key], only=only_axes) if scale_key in other else axis_values
                for scale_key, axis_values in self.items()
            ]
            return self._with_values(new_values)
        # Broadcast across scales
        return self._with_values([axis_values.with_values(other, only=only_axes) for axis_values in self.values()])

    def with_axes(self: _ScaledAxisValuesSelf, axes: OrderedAxes) -> _ScaledAxisValuesSelf:
        return self._with_values([value.with_axes(axes) for value in self.values()])

    @staticmethod
    def _require_positive_resampling_step(step: Union[int, float]):
        if step <= 0:
            raise ValueError(f"Cannot resample by a negative step size (received: {step})")

    @staticmethod
    def _limit_crossed(value: Union[int, float], limit: Union[int, float], step: Union[int, float]) -> bool:
        """Whether `value` has reached or passed `limit` in the direction implied by `step`."""
        return value <= limit if step > 1 else value >= limit

    @staticmethod
    def _validate_scaling_limit(
        base_shape: Shape, scaled_axes: Axes, limit: ShapeLike, step: Union[int, float], *, require_full_coverage: bool
    ) -> None:
        applicable_axes = [a for a in limit if a in base_shape and _axis_in(a, scaled_axes)]
        if not applicable_axes:
            raise ValueError(
                f"Cannot apply a limit if none of its axes ({list(limit.keys())}) "
                f"are scaled axes present in scaled_axes ({list(scaled_axes)})."
            )
        if require_full_coverage and step < 1 and set(scaled_axes) != set(applicable_axes):
            raise ValueError("When upscaling, limit_all must cover all scaled axes.")
        for axis in applicable_axes:
            if step > 1 and limit[axis] > base_shape[axis]:
                raise ValueError(
                    f"limit already exceeded by base_shape along '{axis}' while downsampling "
                    f"(limit={limit[axis]}, base={base_shape[axis]}). No scale could ever be generated."
                )
            if step < 1 and limit[axis] < base_shape[axis]:
                raise ValueError(
                    f"limit already exceeded by base_shape along '{axis}' while upsampling "
                    f"(limit={limit[axis]}, base={base_shape[axis]}). No scale could ever be generated."
                )

    @staticmethod
    def _require_same_axis_sets(pixel_size: PixelSize, scaled_axes: Sequence[AxisKey]) -> None:
        missing = [a for a in scaled_axes if a not in pixel_size]
        if missing:
            raise ValueError(f"pixel_size must provide values for all scaled_axes. Missing: {missing}")

    @classmethod
    def _resolve_scaled_axes_and_limits(
        cls,
        *,
        step: Union[int, float],
        base_shape: Shape,
        scaled_axes: Optional[Axes],
        limit_all: Optional[ShapeLike],
        limit_any: Optional[ShapeLike],
    ) -> Optional[Tuple[List[AxisKey], Shape, Optional[Shape]]]:
        """
        Shared setup/validation for .uniform_steps and .adaptive_steps on BlueprintShapes and BlueprintFactors.
        Returns `(scaled_axes, limit_all_shape, limit_any_shape)`, resolved and validated against `base_shape`.
        Returns None for single-scale blueprint cases (step == 1 or none of `scaled_axes` present in `base_shape`).
        """
        cls._require_positive_resampling_step(step)
        if step == 1:
            return None

        if scaled_axes is None:
            scaled_axes = base_shape.keys()
        resolved_scaled_axes = [a for a in scaled_axes if a in base_shape]
        if not resolved_scaled_axes:
            return None

        limit_all = limit_all or base_shape.with_ones(resolved_scaled_axes)
        cls._validate_scaling_limit(base_shape, resolved_scaled_axes, limit_all, step, require_full_coverage=True)
        limit_all_shape = Shape(limit_all).without_axes_except(base_shape)

        limit_any_shape: Optional[Shape] = None
        if limit_any is not None:
            limit_any_shape = Shape(limit_any)
            cls._validate_scaling_limit(
                base_shape,
                resolved_scaled_axes,
                limit_any_shape,
                step,
                require_full_coverage=False,
            )
            limit_any_shape = limit_any_shape.without_axes_except(base_shape)

        return resolved_scaled_axes, limit_all_shape, limit_any_shape

    @classmethod
    def _generate_adaptive_scales(
        cls,
        *,
        step: Union[int, float],
        base_shape: Shape,
        pixel_size: PixelSize,
        rounding: RoundingMethod,
        scaled_axes: Sequence[AxisKey],
        limit_all: Shape,
        limit_any: Optional[Shape],
        max_levels: int,
        name_pattern: str,
    ) -> List[Tuple[ScaleKey, Factor, Shape]]:
        """
        Shared engine for BlueprintShapes.adaptive_steps and BlueprintFactors.adaptive_steps.
        Returns (scale_key, scale factor, projected shape) for each generated level.
        """
        accum_factor = Factor.identity(scaled_axes)
        results: List[Tuple[ScaleKey, Factor, Shape]] = []
        for i in range(max_levels):
            scaled_shape = base_shape.scaled_by(accum_factor, rounding=rounding)
            key = name_pattern.format(i)
            results.append((key, accum_factor.with_axes(base_shape), scaled_shape))

            all_crossed = all(cls._limit_crossed(scaled_shape[a], limit_all[a], step) for a in scaled_axes)
            any_crossed = limit_any is not None and any(
                cls._limit_crossed(scaled_shape[a], limit_any[a], step) for a in limit_any
            )
            if all_crossed or any_crossed:
                break

            effective_pixel_size = pixel_size * accum_factor
            reference = max(effective_pixel_size.values()) if step > 1 else min(effective_pixel_size.values())
            lagging = [a for a in scaled_axes if not math.isclose(effective_pixel_size[a], reference, rel_tol=1e-9)]

            axes_to_scale_this_round = scaled_axes

            if lagging:
                current_anisotropy = max(effective_pixel_size.values()) / min(effective_pixel_size.values())
                next_effective_pixel_size = PixelSize(effective_pixel_size) * Factor.uniform(lagging, step)
                next_anisotropy = max(next_effective_pixel_size.values()) / min(next_effective_pixel_size.values())

                if next_anisotropy < current_anisotropy:
                    axes_to_scale_this_round = lagging

            accum_factor *= Factor.uniform(axes_to_scale_this_round, step).with_axes(scaled_axes)

        if step < 1:
            # Same reasoning as uniform_steps: per-axis step counts only ever grow, so upscaling
            # (step < 1) generates smallest-first. Reverse to match required output order.
            results.reverse()

        return results

    @staticmethod
    def _resolve_duplicates(
        raw_items: Iterable[Tuple[ScaleKey, Union[Shape, Factor]]], on_duplicate: DuplicatePolicy
    ) -> List[Tuple[ScaleKey, Union[Shape, Factor]]]:
        """
        Ensure raw_items contains no duplicate values. Resolve duplicates according to on_duplicate:
        "error": Raise error if there are duplicates.
        "keep_all": Skip (return raw_items as list)
        "keep_first": Keep the first key seen with any particular value.
        "keep_last": Keep the last key seen with any particular value.
        """
        raw_items = list(raw_items)
        if on_duplicate == DuplicatePolicy.KEEP_ALL:
            return raw_items

        by_value = defaultdict(list)
        for k, v in raw_items:
            by_value[tuple(v.items())].append(k)
        duplicates = {tuple(ks): v for v, ks in by_value.items() if len(ks) > 1}

        if duplicates and on_duplicate == DuplicatePolicy.ERROR:
            raise ValueError(f"Duplicate values not allowed. Collisions: {duplicates}")

        pop_keys: List[ScaleKey] = []
        for dup_keys in duplicates:
            if on_duplicate == DuplicatePolicy.KEEP_FIRST:
                keep = dup_keys[0]
            elif on_duplicate == DuplicatePolicy.KEEP_LAST:
                keep = dup_keys[-1]
            else:
                raise AssertionError(f"Invalid duplicate scale policy: {on_duplicate!r}")

            pop_keys.extend(k for k in dup_keys if k != keep)

        return [(k, v) for k, v in raw_items if k not in pop_keys]


class BlueprintShapes(_ScaledAxisValues[Shape]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self._mapping.items():
            self._mapping[k] = Shape(v)

    @classmethod
    def from_multiscale(cls, multiscale: "Multiscale") -> "BlueprintShapes":
        return cls([(key, scale.shape) for key, scale in multiscale.items()])

    @classmethod
    def from_multiscale_rescaled(
        cls,
        multiscale: "Multiscale",
        *,
        target_shape: ShapeLike,
        rounding: RoundingMethod,
        source_key: Optional[ScaleKey] = None,
        scaled_axes: Optional[Axes] = None,
    ) -> "BlueprintShapes":
        """
        Build a blueprint rescaling shapes from `multiscale`
        such that the shape at `source_key` matches `target_shape`.
        If no `source_key`, `target_shape` becomes the blueprint's base shape.
        All other shapes are rescaled from `target_shape` according to their relative factor to `source_key`
        """
        resolved_source_key = source_key if source_key is not None else next(iter(multiscale.keys()))
        source_shape = multiscale[resolved_source_key].shape

        factors = BlueprintFactors.from_multiscale(multiscale, reference=source_shape)
        if scaled_axes:
            factors = factors.with_identity_except(scaled_axes)

        return factors.to_shapes(reference=target_shape, rounding=rounding)

    @classmethod
    def uniform_steps(
        cls,
        *,
        step: Union[int, float],
        base_shape: Shape,
        rounding: RoundingMethod,
        limit_all: Optional[ShapeLike] = None,
        limit_any: Optional[ShapeLike] = None,
        scaled_axes: Optional[Axes] = None,
        max_levels=42,
        name_pattern=DEFAULT_NAME_PATTERN,
        on_duplicate=DuplicatePolicy.KEEP_FIRST,
    ) -> "BlueprintShapes":
        """
        Generate a blueprint where each scale is a `step` downsampling of the previous scale.
        Applies scaling uniformly to all axes until they become singleton.

        When step < 1 (generating upscales), the result is still ordered largest-to-smallest, with the
        raw shape at the end of the blueprint.

        Generation stops (inclusive of the triggering scale) when either:
        - `limit_all`: every scaled axis has crossed its value (<= when downsampling, >= when
          upsampling). Defaults to singleton (1) on `scaled_axes` if not given. This is the
          primary target shape.
        - `limit_any`: any single axis has crossed its value. Use this as a floor/ceiling
          guard on individual axes (e.g. "never let z go below 4"), independent of `limit_all`.
        Both may reference the same or different axes; whichever condition is met first wins.
        Raises if `base_shape` already crosses a limit.

        If `step` is fractional, `rounding` is applied to round the generated shapes.
        If this generates duplicate shapes, they are deduplicated according to `on_duplicate`.
        """
        resolved = cls._resolve_scaled_axes_and_limits(
            step=step, base_shape=base_shape, scaled_axes=scaled_axes, limit_all=limit_all, limit_any=limit_any
        )
        if resolved is None:
            return cls({name_pattern.format(0): base_shape})
        scaled_axes, limit_all, limit_any_shape = resolved

        scales_items = []
        for i in range(0, max_levels):
            scale_key = name_pattern.format(i)
            scale_factor = step**i
            scaling = Factor.uniform(base_shape, scale_factor).with_identity_except(scaled_axes)
            scaled_shape = base_shape.scaled_by(scaling, rounding=rounding)
            scales_items.append((scale_key, scaled_shape))

            all_crossed = all(cls._limit_crossed(scaled_shape[axis], limit_all[axis], step) for axis in scaled_axes)
            any_crossed = limit_any_shape is not None and any(
                cls._limit_crossed(scaled_shape[axis], limit_any_shape[axis], step) for axis in limit_any_shape
            )
            if all_crossed or any_crossed:
                break

        if step < 1:
            scales_items.reverse()  # Ensure scales are largest-to-smallest

        scales_items = cls._resolve_duplicates(scales_items, on_duplicate)
        bp = cls(scales_items)
        return bp.with_keys(name_pattern)

    @classmethod
    def downscale_powers_of_2_xyz(
        cls,
        *,
        base_shape: Shape,
        rounding: RoundingMethod,
        limit_all: Optional[ShapeLike] = None,
        limit_any: Optional[ShapeLike] = None,
        max_levels: int = 42,
        name_pattern=DEFAULT_NAME_PATTERN,
        on_duplicate=DuplicatePolicy.KEEP_FIRST,
    ):
        """
        Generate a blueprint where each scale is a 2x downsampling of the previous scale.
        Apply scaling uniformly to axes "z", "y" and "x".
        Error if base_shape has none of these axes.

        See BlueprintShapes.uniform_steps for `limit_all`/`limit_any`/`max_levels`.
        """
        xyz = [a for a in base_shape if str(a).lower() in ("x", "y", "z")]
        return cls.uniform_steps(
            step=2,
            scaled_axes=xyz,
            base_shape=base_shape,
            rounding=rounding,
            limit_all=limit_all,
            limit_any=limit_any,
            max_levels=max_levels,
            name_pattern=name_pattern,
            on_duplicate=on_duplicate,
        )

    @classmethod
    def adaptive_steps(
        cls,
        *,
        step: Union[int, float],
        base_shape: Shape,
        pixel_size: PixelSize,
        rounding: RoundingMethod,
        limit_all: Optional[ShapeLike] = None,
        limit_any: Optional[ShapeLike] = None,
        scaled_axes: Optional[Axes] = None,
        max_levels=42,
        name_pattern=DEFAULT_NAME_PATTERN,
        on_duplicate=DuplicatePolicy.KEEP_FIRST,
    ) -> "BlueprintShapes":
        """
        Generate a blueprint that first converges scaled axes toward isotropic physical pixel
        size, then steps all of them together by `step`, like `uniform_steps`.

        Most combinations of `step` and `pixel_size` will never converge to exact isotropy, so the initial convergence
        stops when the pixel size becomes as isotropic as the requested `step` permits, continuing with uniform steps.

        Equivalent to `uniform_steps` when `pixel_size` is already isotropic along `scaled_axes`.

        When step < 1 (generating upscales), the result is still ordered largest-to-smallest, with the
        raw shape at the end of the blueprint.

        See BlueprintShapes.uniform_steps for `limit_all`/`limit_any`/`max_levels`.
        """
        resolved = cls._resolve_scaled_axes_and_limits(
            step=step, base_shape=base_shape, scaled_axes=scaled_axes, limit_all=limit_all, limit_any=limit_any
        )
        if resolved is None:
            return cls({name_pattern.format(0): base_shape})
        scaled_axes, limit_all, limit_any_shape = resolved
        cls._require_same_axis_sets(pixel_size, scaled_axes)

        generated = cls._generate_adaptive_scales(
            step=step,
            base_shape=base_shape,
            pixel_size=pixel_size,
            rounding=rounding,
            scaled_axes=scaled_axes,
            limit_all=limit_all,
            limit_any=limit_any_shape,
            max_levels=max_levels,
            name_pattern=name_pattern,
        )
        scales_items = [(key, shape) for key, _factor, shape in generated]
        scales_items = cls._resolve_duplicates(scales_items, on_duplicate)
        bp = cls(scales_items)
        return bp.with_keys(name_pattern)

    @classmethod
    def adaptive_powers_of_2_xyz(
        cls,
        *,
        base_shape: Shape,
        pixel_size: PixelSize,
        rounding: RoundingMethod,
        limit_all: Optional[ShapeLike] = None,
        limit_any: Optional[ShapeLike] = None,
        max_levels: int = 42,
        name_pattern=DEFAULT_NAME_PATTERN,
        on_duplicate=DuplicatePolicy.KEEP_FIRST,
    ) -> "BlueprintShapes":
        """
        Generate a blueprint where each scale is a 2x downsampling of the previous scale.
        Apply scaling to axes "z", "y" and "x", first converging toward isotropic physical pixel size,
        then stepping all of them together like `downscale_powers_of_2_xyz`.
        Error if base_shape has none of these axes.

        Equivalent to `downscale_powers_of_2_xyz` when `pixel_size` is already isotropic.

        See BlueprintShapes.uniform_steps for `limit_all`/`limit_any`/`max_levels`.
        """
        xyz = [a for a in base_shape if str(a).lower() in ("x", "y", "z")]
        return cls.adaptive_steps(
            step=2,
            scaled_axes=xyz,
            base_shape=base_shape,
            pixel_size=pixel_size,
            rounding=rounding,
            limit_all=limit_all,
            limit_any=limit_any,
            max_levels=max_levels,
            name_pattern=name_pattern,
            on_duplicate=on_duplicate,
        )

    def axes(self) -> Iterable[AxisKey]:
        return self.first_value().keys()

    def scaled_axes(self) -> Tuple[AxisKey, ...]:
        """Axes where shapes differ across scales."""
        if len(self) < 2:
            return ()

        shapes = list(self.values())
        first_shape = shapes[0]
        scaled = []

        for axis in first_shape.keys():
            first_value = first_shape[axis]
            if any(shape[axis] != first_value for shape in shapes[1:]):
                scaled.append(axis)

        return tuple(scaled)

    def with_sizes(
        self, other: Union[ShapeLike, Mapping[ScaleKey, ShapeLike]], *, only_axes: Optional[Axes] = None
    ) -> "BlueprintShapes":
        """
        Replace shape values in this blueprint.
        Provide either
        - a blueprint-like mapping {scale_key: {axis: size}}
          to replace the values for those axes at that scale
        - a shape-like mapping {axis: size}
          to replace the values for those axes in *all* scales.
        Optionally limit the replacing to `only` axes (and ignore other axes in the provided mapping).
        """
        return super()._with_values_by_axes(other, only_axes=only_axes)

    def to_factors(self, reference: Optional[Shape] = None) -> "BlueprintFactors":
        if reference is None:
            reference = self.first_value()
        factors = [Shape(reference).scaling_to(scale_shape) for scale_shape in self.values()]
        return BlueprintFactors(zip(self.keys(), factors))

    def apply_to_scale(
        self, base: Scale, *, translation_shift_func: Optional[TranslationShiftFunction] = None
    ) -> "Multiscale":
        if list(self.first_value().keys()) != list(base.shape.keys()):
            raise ValueError(
                f"Cannot apply blueprint with axes {list(self.first_value().keys())} "
                f"to base scale with axes {list(base.shape.keys())}. "
                "Axes must match exactly. Maybe blueprint.with_axes(base.shape) first?"
            )

        scales = []
        for scale_key, target_shape in self.items():
            factor = base.shape.scaling_to(target_shape)
            new_pixel_size = base.pixel_size.scaled_by(factor)

            if translation_shift_func is not None:
                target_scale_pre_shift = Scale(
                    shape=target_shape, pixel_size=new_pixel_size, unit=base.unit, translation=base.translation
                )
                shift = self._compute_and_validate_shift(translation_shift_func, base, target_scale_pre_shift)
                new_translation = base.translation + shift
            else:
                new_translation = base.translation

            scales.append(
                (
                    scale_key,
                    Scale(
                        shape=target_shape,
                        pixel_size=new_pixel_size,
                        unit=base.unit,
                        translation=new_translation,
                        ome_zarr_axes=base.ome_zarr_axes,
                    ),
                )
            )

        return Multiscale(scales)

    @staticmethod
    def _compute_and_validate_shift(translation_shift_func, base, target_scale_pre_shift):
        try:
            shift = translation_shift_func(base, target_scale_pre_shift)
        except TypeError as e:
            if "argument" in str(e):
                raise TypeError(
                    "translation_shift_func must accept two positional arguments (base and target scale). "
                    "See clearscale.half_pixel_shift for an example implementation."
                ) from e
            raise e
        if not isinstance(shift, Translation):
            raise TypeError(
                f"translation_shift_func must return a Translation, got {type(shift).__name__}. "
                "See clearscale.half_pixel_shift for an example implementation."
            )
        if list(shift.keys()) != list(target_scale_pre_shift.shape.keys()):
            raise ValueError(
                f"translation_shift_func returned Translation with axes {list(shift.keys())}, "
                f"but target scale has axes {list(target_scale_pre_shift.shape.keys())}."
            )
        return shift


class BlueprintFactors(_ScaledAxisValues[Factor]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self._mapping.items():
            self._mapping[k] = Factor(v)

    @classmethod
    def from_shapes(cls, shapes: Mapping[ScaleKey, ShapeLike], reference: Shape) -> "BlueprintFactors":
        return BlueprintShapes(shapes).to_factors(reference)

    @classmethod
    def from_multiscale(cls, multiscale: "Multiscale", reference: Shape) -> "BlueprintFactors":
        return BlueprintShapes.from_multiscale(multiscale).to_factors(reference)

    @classmethod
    def uniform_steps(
        cls,
        *,
        step: Union[int, float],
        base_shape: Shape,
        rounding: RoundingMethod,
        limit_all: Optional[ShapeLike] = None,
        limit_any: Optional[ShapeLike] = None,
        scaled_axes: Optional[Axes] = None,
        max_levels=42,
        name_pattern=DEFAULT_NAME_PATTERN,
        on_duplicate=DuplicatePolicy.KEEP_FIRST,
    ) -> "BlueprintFactors":
        """
        Generate a blueprint where each scale's factor is an exact power of `step`
        (step**0, step**1, ...).

        `rounding` is used only to project each candidate factor to a
        shape, for limit-checking and for detecting when consecutive levels collapse to the same
        shape. It does not affect the returned factor values.

        When step < 1 (generating upscales), the result is still ordered such that the generated scaled
        shapes will be largest-to-smallest, with factor=1 at the end of the blueprint.

        See BlueprintShapes.uniform_steps for `limit_all`/`limit_any`/`max_levels`.
        The limits apply in the same way here, to the *projected* shape at the respective factor.
        """
        resolved = cls._resolve_scaled_axes_and_limits(
            step=step, base_shape=base_shape, scaled_axes=scaled_axes, limit_all=limit_all, limit_any=limit_any
        )
        if resolved is None:
            return cls({name_pattern.format(0): Factor.identity(base_shape)})
        scaled_axes, limit_all, limit_any_shape = resolved

        factor_items = []
        shape_items = []
        for i in range(0, max_levels):
            scale_key = name_pattern.format(i)
            factor = Factor.uniform(base_shape, step**i).with_identity_except(scaled_axes)
            projected_shape = base_shape.scaled_by(factor, rounding=rounding)

            factor_items.append((scale_key, factor))
            shape_items.append((scale_key, projected_shape))

            all_crossed = all(cls._limit_crossed(projected_shape[axis], limit_all[axis], step) for axis in scaled_axes)
            any_crossed = limit_any_shape is not None and any(
                cls._limit_crossed(projected_shape[axis], limit_any_shape[axis], step) for axis in limit_any_shape
            )
            if all_crossed or any_crossed:
                break

        if step < 1:
            # Ensure scales are largest-to-smallest
            factor_items.reverse()
            shape_items.reverse()

        kept_keys = {k for k, _ in cls._resolve_duplicates(shape_items, on_duplicate)}
        factor_items = [(k, f) for k, f in factor_items if k in kept_keys]

        bp = cls(factor_items)
        return bp.with_keys(name_pattern)

    @classmethod
    def downscale_powers_of_2_xyz(
        cls,
        *,
        base_shape: Shape,
        rounding: RoundingMethod,
        limit_all: Optional[ShapeLike] = None,
        limit_any: Optional[ShapeLike] = None,
        max_levels: int = 42,
        name_pattern=DEFAULT_NAME_PATTERN,
        on_duplicate=DuplicatePolicy.KEEP_FIRST,
    ) -> "BlueprintFactors":
        """
        Generate a blueprint where each scale factor is successive powers of 2 (1, 2, 4, 8...) along "x", "y" and "z".
        Error if base_shape has none of these axes.

        See BlueprintShapes.uniform_steps for `limit_all`/`limit_any`/`max_levels`.
        The limits apply in the same way here, to the *projected* shape at the respective factor.
        """
        xyz = [a for a in base_shape if str(a).lower() in ("x", "y", "z")]
        return cls.uniform_steps(
            step=2,
            scaled_axes=xyz,
            base_shape=base_shape,
            rounding=rounding,
            limit_all=limit_all,
            limit_any=limit_any,
            max_levels=max_levels,
            name_pattern=name_pattern,
            on_duplicate=on_duplicate,
        )

    @classmethod
    def adaptive_steps(
        cls,
        *,
        step: Union[int, float],
        base_shape: Shape,
        pixel_size: PixelSize,
        rounding: RoundingMethod,
        limit_all: Optional[ShapeLike] = None,
        limit_any: Optional[ShapeLike] = None,
        scaled_axes: Optional[Axes] = None,
        max_levels=42,
        name_pattern=DEFAULT_NAME_PATTERN,
        on_duplicate=DuplicatePolicy.KEEP_FIRST,
    ) -> "BlueprintFactors":
        """Like BlueprintShapes.adaptive_steps, generate factors as powers of `step` along `scaled_axes`, first
        converging `pixel_size` to isotropy and then continuing as uniform steps. Returns each level's per-axis factor
        (step ** per-axis step count).

        See BlueprintShapes.uniform_steps for `limit_all`/`limit_any`/`max_levels`.
        `rounding` is used only to project shapes for limit-checking and to detect duplicates internally.
        The returned factor values are exact powers of `step`."""
        resolved = cls._resolve_scaled_axes_and_limits(
            step=step, base_shape=base_shape, scaled_axes=scaled_axes, limit_all=limit_all, limit_any=limit_any
        )
        if resolved is None:
            return cls({name_pattern.format(0): Factor.identity(base_shape)})
        scaled_axes, limit_all, limit_any_shape = resolved
        cls._require_same_axis_sets(pixel_size, scaled_axes)

        generated = cls._generate_adaptive_scales(
            step=step,
            base_shape=base_shape,
            pixel_size=pixel_size,
            rounding=rounding,
            scaled_axes=scaled_axes,
            limit_all=limit_all,
            limit_any=limit_any_shape,
            max_levels=max_levels,
            name_pattern=name_pattern,
        )

        factor_items = [(key, factor) for key, factor, _shape in generated]
        shape_items = [(key, shape) for key, _step_counts, shape in generated]  # dedup by projected shape

        kept_keys = {k for k, _ in cls._resolve_duplicates(shape_items, on_duplicate)}
        factor_items = [(k, f) for k, f in factor_items if k in kept_keys]

        bp = cls(factor_items)
        return bp.with_keys(name_pattern)

    @classmethod
    def adaptive_powers_of_2_xyz(
        cls,
        *,
        base_shape: Shape,
        pixel_size: PixelSize,
        rounding: RoundingMethod,
        limit_all: Optional[ShapeLike] = None,
        limit_any: Optional[ShapeLike] = None,
        max_levels: int = 42,
        name_pattern=DEFAULT_NAME_PATTERN,
        on_duplicate=DuplicatePolicy.KEEP_FIRST,
    ) -> "BlueprintFactors":
        """Like BlueprintShapes.adaptive_powers_of_2_xyz: Generate shape-halving factors (1, 2, 4, 8...) along "x", "y" and "z",
        first converging `pixel_size` to isotropy and then continuing as uniform steps. Returns each level's per-axis factor
        (step ** per-axis step count).

        See BlueprintShapes.uniform_steps for `limit_all`/`limit_any`/`max_levels`.
        `rounding` is used only to project shapes for limit-checking and duplicate detection internally.
        The returned factor values are exact powers of `step`."""
        xyz = [a for a in base_shape if str(a).lower() in ("x", "y", "z")]
        return cls.adaptive_steps(
            step=2,
            scaled_axes=xyz,
            base_shape=base_shape,
            pixel_size=pixel_size,
            rounding=rounding,
            limit_all=limit_all,
            limit_any=limit_any,
            max_levels=max_levels,
            name_pattern=name_pattern,
            on_duplicate=on_duplicate,
        )

    def axes(self) -> Iterable[AxisKey]:
        return self.first_value().keys()

    @property
    def scaled_axes(self) -> Tuple[AxisKey, ...]:
        """Axes where any factor is not 1.0."""
        if len(self) < 2:
            return ()

        scaled: Set[AxisKey] = set()
        for factor in self.values():
            scaled.update(axis for axis, value in factor.items() if value != 1.0)

        all_axes = next(iter(self.values())).keys()
        return tuple(axis for axis in all_axes if axis in scaled)

    def with_factors(
        self, other: Union[FactorLike, Mapping[ScaleKey, FactorLike]], *, only_axes: Optional[Axes] = None
    ) -> "BlueprintFactors":
        """
        Replace factor values in this blueprint.
        Provide either
        - a blueprint-like mapping {scale_key: {axis: factor}}
          to replace the values for those axes at that scale
        - a shape-like mapping {axis: factor}
          to replace the values for those axes in *all* scales.
        Optionally limit the replacing to `only` axes (and ignore other axes in the provided mapping).
        """
        return super()._with_values_by_axes(other, only_axes=only_axes)

    def to_shapes(self, reference: ShapeLike, *, rounding: RoundingMethod) -> BlueprintShapes:
        ref = Shape(reference)
        shapes = [ref.scaled_by(scale_factor, rounding=rounding) for scale_factor in self.values()]
        return BlueprintShapes(zip(self.keys(), shapes))

    def apply_to_scale(
        self,
        scale: Scale,
        *,
        rounding: RoundingMethod,
        translation_shift_func: Optional[TranslationShiftFunction] = None,
    ) -> "Multiscale":
        shapes = self.to_shapes(scale.shape, rounding=rounding)
        return shapes.apply_to_scale(scale, translation_shift_func=translation_shift_func)

    def with_identity(self, axes: Axes) -> "BlueprintFactors":
        return self._with_values([factor.with_identity(axes) for factor in self.values()])

    def with_identity_except(self, axes: Axes) -> "BlueprintFactors":
        return self._with_values([factor.with_identity_except(axes) for factor in self.values()])


def _random_multiscale_name(seed: int | str | None = None) -> str:
    from clearscale._services.animal_names import generate_random_animal_name

    return generate_random_animal_name(seed)


class Multiscale(_ScaleMapping[Scale], TransformGraphNode):
    _transform_graph: TransformGraph
    """Transform graph that by default consists only of one isolated node: _intrinsic_ref.
    The graph can acquire additional CoordinateSystems by .from_ome_zarr, or by derivation from a Multiscale that 
    acquired them from_ome_zarr.
    Transforms in Multiscale graphs may only source/target `NodeRef[CoordinateSystem]`, and the paths from 
    self._intrinsic_ref to each satellite coordinate system must be a single Transform."""
    _intrinsic_ref: NodeRef[CoordinateSystem]
    """The system in which the Scales' shape, pixel size, translation etc. are correct."""
    _zero_scale_axes_by_key: Mapping[str, Tuple[AxisKey, ...]]
    """Dataset scale axes that were read as 0.0 from loaded meta; kept for as-read round-trip."""
    _legacy_convention_global_t_scale: Optional[float]
    """Special-case for a legacy OME-Zarr convention where time step size (t-scale) is written as a global scale 
    transform. Serializes like a `ScaleTransform.bound(source=self._intrinsic_ref, target=synthetic_external)`, 
    but is redacted from self._transform_graph and instead stored on every scale as `self[].pixel_size['t']`."""
    has_shapes: bool
    """If False, this indicates the Multiscale was generated with fake (all-singleton) shapes."""
    ome: ome_zarr.MultiscaleProperties
    """Additional props specific to OME-Zarr, customizable by user (mutable!)."""

    def __init__(
        self,
        *args,
        ome: Optional[ome_zarr.MultiscaleProperties] = None,
        _transform_graph: Optional[TransformGraph] = None,
        _intrinsic_ref: Optional[NodeRef[CoordinateSystem]] = None,
        _zero_scale_axes_by_key: Optional[Mapping[str, Tuple[AxisKey, ...]]] = None,
        _legacy_convention_global_t_scale=None,
        has_shapes=True,
        **kwargs,
    ):
        """
        Multiscales can be constructed from a `scale_key : Scale` mapping, but this should be avoided.
        Multiscale objects should reflect either metadata read from a file (`.from_ome_zarr`, `.from_precomputed`),
        or expand a single Scale according to a scaling blueprint (`.from_shapes`, `.from_factors`).
        """
        super().__init__(*args, **kwargs)
        for key, scale in self._mapping.items():
            if scale.shape.keys() != self.axes():
                raise ValueError(
                    f"All Scales must have identical axes. Scale at '{key}' has {list(scale.shape.keys())} != {list(self.axes())}"
                )

        if _intrinsic_ref is None:
            # No internals supplied - "From scratch" construction with enforcement of multiscale format requirements
            if _transform_graph:
                raise AssertionError("Must specify _intrinsic_ref when _transform_graph is given.")
            canonical_axes = self._merge_ome_zarr_axes(list(self._mapping.values()))
            self._require_ome_zarr_permitted_type_order(canonical_axes)
            self._require_non_increasing_shape_order(list(self._mapping.items()))
            canonical_unit = Unit([(a, ax.unit or "") for a, ax in canonical_axes.items()])
            # Dedup: every Scale shares the same canonical Unit/OmeZarrAxes instances.
            self._mapping = OrderedDict(
                (
                    key,
                    Scale(
                        shape=s.shape,
                        pixel_size=s.pixel_size,
                        unit=canonical_unit,
                        translation=s.translation,
                        ome_zarr_axes=canonical_axes,
                    ),
                )
                for key, s in self._mapping.items()
            )
            intrinsic_sys = CoordinateSystem(canonical_axes)
            sys_ref = intrinsic_sys.as_ref(_random_multiscale_name())
            self._transform_graph = TransformGraph.single_isolated_system(sys_ref)
            self._intrinsic_ref = sys_ref
        else:
            # Internal construction without format requirement enforcement.
            # Leniency for loading existing, strictly speaking invalid, datasets through from_ome_zarr / from_precomputed.
            transform_graph = _transform_graph or TransformGraph.single_isolated_system(_intrinsic_ref)
            if _intrinsic_ref not in transform_graph.all_system_refs:
                raise AssertionError("_intrinsic_ref must be inside _transform_graph")
            self._transform_graph = transform_graph
            self._intrinsic_ref = _intrinsic_ref
        zero_scale_axes_by_key = {}
        if _zero_scale_axes_by_key:
            available_axes = set(self.axes())
            for key, axes in _zero_scale_axes_by_key.items():
                if key not in self:
                    continue
                kept_axes = tuple(axis for axis in axes if axis in available_axes)
                if kept_axes:
                    zero_scale_axes_by_key[key] = kept_axes
        self._zero_scale_axes_by_key = zero_scale_axes_by_key
        self._legacy_convention_global_t_scale = _legacy_convention_global_t_scale
        self.has_shapes = has_shapes
        self.ome = ome if isinstance(ome, ome_zarr.MultiscaleProperties) else ome_zarr.MultiscaleProperties()

    def __eq__(self, other):
        return _ScaleMapping.__eq__(self, other)

    def __hash__(self):
        return _ScaleMapping.__hash__(self)

    def copy(self) -> "Multiscale":
        raise NotImplementedError(
            "Shallow copying of Multiscale is likely to cause surprising behavior. "
            "Please use the `copy` module and consider using `copy.deepcopy()` to avoid mutating the original."
        )

    @classmethod
    def from_single(
        cls,
        scale: Scale,
        *,
        scale_key: Optional[str] = "s0",
        blueprint: Union[BlueprintShapes, BlueprintFactors, None] = None,
        translation_shift_func: Optional[TranslationShiftFunction] = None,
        rounding: Optional[RoundingMethod] = None,
    ):
        """
        Expand the provided Scale into a new Multiscale.

        If nothing else is provided, this is equivalent to `Multiscale({"s0": scale})`

        Optionally provide a `blueprint` to specify how the data described by `scale` was/will be scaled, and compute matching metadata.
        Related:
          - `translation_shift_func`: Function that computes your scaling method's translation shift (see translation shift docs)
          - `rounding`: Your scaling method's rounding behavior. Only required when `blueprint` is `BlueprintFactors`.

        If you already have an existing Multiscale, and `scale` is one of its values, consider `Multiscale.derive` instead.
        """
        if not isinstance(scale_key, str):
            raise TypeError(
                f"Scale keys must be relative paths to each scale's data within the zarr group. Received: {scale_key!r}"
            )
        bp = blueprint if blueprint is not None else BlueprintShapes({scale_key: scale.shape})
        if isinstance(bp, BlueprintFactors):
            if rounding is None:
                raise ValueError(
                    "Must specify how your scaling method rounds when factors unevenly divide shape. E.g. `rounding='floor'`"
                )
            return bp.apply_to_scale(scale, translation_shift_func=translation_shift_func, rounding=rounding)
        return bp.apply_to_scale(scale, translation_shift_func=translation_shift_func)

    @staticmethod
    def from_shapes(
        blueprint: Mapping[ScaleKey, ShapeLike],
        *,
        base: Optional[Scale] = None,
        translation_shift_func: Optional[TranslationShiftFunction] = None,
    ):
        bp = BlueprintShapes(blueprint)
        base = base or Scale(shape=bp.first_value())
        return bp.apply_to_scale(base, translation_shift_func=translation_shift_func)

    @staticmethod
    def from_factors(blueprint: BlueprintFactors, base: Scale, *, rounding: RoundingMethod):
        return blueprint.apply_to_scale(base, rounding=rounding)

    @classmethod
    def from_ome_zarr(
        cls,
        multiscale_dict: ome_zarr.OME_ZARR_MULTISCALE,
        *,
        shape_source: ome_zarr.ShapeSource,
    ):
        ome_zarr.require_dataset_paths(multiscale_dict)
        get_shape = ome_zarr.normalize_shape_source_to_callable(shape_source, multiscale_dict)
        global_meta = global_t_scale = None
        try:
            transform_graph, intrinsic_system_ref = ome_zarr.extract_multiscale_graph(multiscale_dict)
        except ValueError:
            intrinsic_system_name = _random_multiscale_name()
            transform_graph, intrinsic_system_ref, global_meta = ome_zarr.multiscale_graph_from_legacy(
                multiscale_dict, name=intrinsic_system_name
            )
            if global_meta is not None:
                global_t_scale, _ = global_meta
        assert intrinsic_system_ref.owner, "dev error: must reference intrinsic system"
        axis_keys = list(intrinsic_system_ref.owner.axes())
        unit = intrinsic_system_ref.owner.get_unit()
        datasets = multiscale_dict["datasets"]
        base_shape = None
        scales_items = []
        zero_scale_axes_by_key = {}
        for dataset in datasets:
            scale_key = dataset["path"]
            scale_shape = Shape(zip(axis_keys, get_shape(scale_key)))
            if base_shape is None:
                base_shape = scale_shape
            relative_scale_factor = base_shape.scaling_to(scale_shape)
            relative_scale_pixel_size = PixelSize.identity(axis_keys).scaled_by(relative_scale_factor)
            dataset_transforms = dataset.get("coordinateTransformations")
            scale_pixel_size, scale_translation, zero_scale_axes = ome_zarr.scale_meta_from_dataset_transforms(
                axis_keys, global_meta, relative_scale_pixel_size, dataset_transforms
            )
            if zero_scale_axes:
                zero_scale_axes_by_key[scale_key] = zero_scale_axes
            scales_items.append(
                (
                    scale_key,
                    Scale(
                        shape=scale_shape,
                        pixel_size=scale_pixel_size,
                        translation=scale_translation,
                        unit=unit,
                        ome_zarr_axes=OmeZarrAxes(intrinsic_system_ref.owner),
                    ),
                )
            )

        return cls(
            scales_items,
            _transform_graph=transform_graph,
            _intrinsic_ref=intrinsic_system_ref,
            _zero_scale_axes_by_key=zero_scale_axes_by_key,
            _legacy_convention_global_t_scale=global_t_scale,
            has_shapes=shape_source != "singletons",
            ome=ome_zarr.MultiscaleProperties.from_ome_zarr(multiscale_dict),
        )

    @classmethod
    def from_precomputed(cls, info_dict: precomputed.INFO_DICT):
        precomputed.validate_info_dict(info_dict)
        scales_list = info_dict["scales"]
        num_channels = info_dict.get("num_channels", 1)
        axis_keys = ["c", "z", "y", "x"]  # Precomputed is always czyx (x varies fastest)
        unit = Unit(zip(axis_keys, ["", "nm", "nm", "nm"]))
        ome_zarr_axes = OmeZarrAxes(
            [
                ("c", OmeZarrAxis(name="c", type="channel", discrete=True)),
                ("z", OmeZarrAxis(name="z", type="space", discrete=False)),
                ("y", OmeZarrAxis(name="y", type="space", discrete=False)),
                ("x", OmeZarrAxis(name="x", type="space", discrete=False)),
            ]
        )

        scales_items = []
        zero_scale_axes_by_key = {}
        for scale_dict in scales_list:
            scale_key = scale_dict["key"]

            size = scale_dict["size"]
            if len(size) != 3:
                raise ValueError(f"Scale {scale_key!r} must have 'size' as [x, y, z]")
            shape = Shape(zip(axis_keys, [num_channels] + list(reversed(size))))

            resolution = scale_dict["resolution"]
            if len(resolution) != 3:
                raise ValueError(f"Scale {scale_key!r} must have 'resolution' as [x, y, z]")
            zero_axes = precomputed.zero_resolution_axes(resolution, "zyx")
            if zero_axes:
                zero_scale_axes_by_key[scale_key] = zero_axes
            pixel_size = precomputed.pixel_size_from_resolution(resolution, axis_keys)

            voxel_offset = scale_dict.get("voxel_offset", [0, 0, 0])
            if len(voxel_offset) != 3:
                warnings.warn(f"Scale {scale_key!r} has invalid voxel_offset. Using [0, 0, 0].")
                voxel_offset = [0, 0, 0]
            offset = PixelOffset(zip(axis_keys, [0] + list(reversed(voxel_offset))))
            translation = offset.to_physical(pixel_size)

            scale = Scale(shape, pixel_size, unit, translation, ome_zarr_axes)
            scales_items.append((scale_key, scale))

        return cls(scales_items, _zero_scale_axes_by_key=zero_scale_axes_by_key)

    def axes(self) -> OrderedAxes:
        return self.first_value().shape.keys()

    @property
    def unit(self) -> Unit:
        return self._intrinsic_ref.owner.get_unit()

    @property
    def ome_zarr_axes(self) -> OmeZarrAxes:
        return OmeZarrAxes(self._intrinsic_ref.owner)

    def coordinate_system_ome_zarr_axes(self, name: str) -> OmeZarrAxes:
        """Return axis properties of the coordinate system named `name`"""
        refs = self._transform_graph.all_system_refs
        matching = [ref for ref in refs if ref.name == name and ref is not self._intrinsic_ref]
        if not matching:
            available = [ref.name for ref in refs]
            raise ValueError(f"No coordinate system named {name!r} in {available}")
        assert len(matching) == 1, "names should be unique in the graph"
        return OmeZarrAxes(matching[0].owner)

    def scaled_axes(self) -> Tuple[AxisKey, ...]:
        """Axes where pixel_sizes differ across scales."""
        if len(self) < 2:
            return ()

        pixel_sizes = list(scale.pixel_size for scale in self.values())
        first_pixel_size = pixel_sizes[0]
        scaled = []

        for axis in first_pixel_size.keys():
            first_value = first_pixel_size[axis]
            if any(pixel_size[axis] != first_value for pixel_size in pixel_sizes[1:]):
                scaled.append(axis)

        return tuple(scaled)

    @cached_property
    def coordinate_systems(self) -> Tuple[str, ...]:
        """Spaces, other than the multiscale's own, into which it can be transformed."""
        return tuple(ref.name for ref in self._transform_graph.all_system_refs if ref is not self._intrinsic_ref)

    @cached_property
    def keys_by_shape(self) -> Mapping[Shape, Tuple[ScaleKey, ...]]:
        grouped = defaultdict(list)
        for key, scale in self.items():
            grouped[scale.shape].append(key)
        return {shape: tuple(keys) for shape, keys in grouped.items()}

    def with_coordinate_system(
        self,
        name: str,
        *,
        reached_by: Union[SpatialRelation, Sequence[SpatialRelation], None] = None,
        unit: Optional[Union[Unit, Mapping[AxisKeyT, str]]] = None,
        ome_zarr_axes: Optional[OmeZarrAxesParam] = None,
    ) -> "Multiscale":
        def _reconcile_axis_prop_params(
            target_axes: Tuple[AxisKey, ...],
            source_ome_axes: OmeZarrAxes,
            provided_ome_axes: Optional[OmeZarrAxesParam],
            unit: Optional[Union[Unit, Mapping[AxisKeyT, str]]],
        ) -> OmeZarrAxes:
            unit = Unit.empty(target_axes) if unit is None else Unit(unit)
            parsed_ome_axes = _normalize_ome_zarr_axes_param(provided_ome_axes, target_axes)

            conflicts = parsed_ome_axes.conflicts_with_unit(unit)
            if conflicts:
                raise ValueError(
                    f"Conflicting unit between `unit` and `ome_zarr_axes`: "
                    f"{ {a: f'{slf!r} vs {oth!r}' for a, (slf, oth) in conflicts.items()} }. Only specify unit once."
                )

            if provided_ome_axes == "infer":
                authoritative = source_ome_axes.with_axes(target_axes)
                # Drop inferred properties for already existing axes
                extra_properties = parsed_ome_axes.without_axes_except(set(target_axes) - set(source_ome_axes.keys()))
            else:
                authoritative = parsed_ome_axes.with_axes(target_axes)
                extra_properties = source_ome_axes

            result = authoritative.with_blanks_filled_from(extra_properties)

            return result.with_unit_merged(unit)

        relations = (
            [] if reached_by is None else [reached_by] if isinstance(reached_by, SpatialRelation) else list(reached_by)
        )
        if not all(isinstance(r, SpatialRelation) for r in relations):
            raise ValueError(
                "How the new coordinate system is reached must be expressed using SpatialRelations "
                'like `Factor`, `Translation` or AxisRearrangementTo("zyx").'
            )

        if name in self.coordinate_systems:
            raise ValueError(f"Coordinate system name {name!r} already exists on this Multiscale.")

        source_axes = tuple(self.axes())
        target_axes = relation_chain_target_axes(relations, source_axes) if relations else source_axes
        target_ome_axes = _reconcile_axis_prop_params(target_axes, self.ome_zarr_axes, ome_zarr_axes, unit)

        target_ref = CoordinateSystem(target_ome_axes).as_ref(name)
        transform = (relations_to_transform(relations, source_axes) if relations else IdentityTransform()).bound(
            source=self._intrinsic_ref, target=target_ref
        )

        new_graph = TransformGraph(
            transforms=self._transform_graph.transforms + (transform,),
            system_refs=self._transform_graph.system_refs,
        )
        return Multiscale(
            self.items(),
            _transform_graph=new_graph,
            _intrinsic_ref=self._intrinsic_ref,
            _zero_scale_axes_by_key=self._zero_scale_axes_by_key,
            _legacy_convention_global_t_scale=self._legacy_convention_global_t_scale,
        )

    def as_derived_from(
        self, other: "Multiscale", *, by: Union[SpatialRelation, Sequence[SpatialRelation], None] = None
    ) -> "Multiscale":
        """
        Transfer the spatial context, axis properties, and serialization convention from `other`.
        Optionally specify *how* `self` was derived from `other` using `by=Factor(...)` or other SpatialRelations.
        """
        relations: List[SpatialRelation] = [] if by is None else [by] if isinstance(by, SpatialRelation) else list(by)
        if not all(isinstance(r, SpatialRelation) for r in relations):
            raise ValueError(
                "The derivation relationship must be expressed using SpatialRelations like Factor,"
                'Translation or AxisRearrangementTo("zyx").'
            )
        source_axes = tuple(other.axes())
        target_axes = tuple(self.axes())

        if not relations:
            if source_axes != target_axes:
                raise ValueError(
                    f"Cannot transfer coordinate systems from source with axes {source_axes!r} to Multiscale "
                    f"with axes {target_axes!r}. Use `by` to specify how the new axes were obtained. "
                    f"E.g. `ProjectionTo(result_axes)` for inserted or dropped axes, `PermutationTo(result_axes)` "
                    f"for reordering, or a list of both."
                )
        else:
            derived_axes = relation_chain_target_axes(relations, source_axes)
            if derived_axes != target_axes:
                raise ValueError(
                    f"Incompatible derivation: Provided relation chain would produce axes {derived_axes!r} "
                    f"from {source_axes!r}, but this Multiscale has {target_axes!r}. {by=!r}"
                )

        merged_ome_axes = self.ome_zarr_axes.with_blanks_filled_from(other.ome_zarr_axes)
        if merged_ome_axes == self.ome_zarr_axes:
            intrinsic_ref = self._intrinsic_ref
            transform_graph = self._transform_graph
            mapping_items = list(self.items())
        else:
            self._require_ome_zarr_permitted_type_order(merged_ome_axes)  # should be unnecessary but might as well
            merged_unit = merged_ome_axes.get_unit()
            intrinsic_ref = CoordinateSystem(merged_ome_axes).as_ref(self._intrinsic_ref.name)
            transform_graph = TransformGraph(
                transforms=tuple(
                    self._replace_transform_ref(t, self._intrinsic_ref, intrinsic_ref)
                    for t in self._transform_graph.transforms
                ),
                system_refs=tuple(
                    intrinsic_ref if r == self._intrinsic_ref else r for r in self._transform_graph.system_refs
                ),
            )
            mapping_items = [
                (
                    key,
                    Scale(
                        shape=s.shape,
                        pixel_size=s.pixel_size,
                        unit=merged_unit,
                        translation=s.translation,
                        ome_zarr_axes=merged_ome_axes,
                    ),
                )
                for key, s in self.items()
            ]

        transferred_global_t_scale = None
        if other._legacy_convention_global_t_scale:
            # Transfer the fact that we use the convention; but only if self can even be expressed using it
            # (i.e. has t, and isn't scaled across t). The actual value stored must be self's own t-scale.
            if "t" in self.axes() and "t" not in self.scaled_axes():
                transferred_global_t_scale = self.first_value().pixel_size["t"]
        unchanged_t_scale = transferred_global_t_scale == self._legacy_convention_global_t_scale

        existing_refs = list(transform_graph.all_system_refs)
        other_unique = self._find_or_make_unique_ref(
            other._intrinsic_ref.owner, other._intrinsic_ref.name, existing_refs
        )
        derivation = (relations_to_transform(relations, source_axes) if relations else IdentityTransform()).bound(
            source=other_unique, target=intrinsic_ref
        )
        existing_refs.append(other_unique)

        to_merge_list: List[Transform] = []
        for satellite in other._transform_graph.all_system_refs:
            resolved_ref = self._find_or_make_unique_ref(satellite.owner, satellite.name, existing_refs)
            to_port = (
                t
                for t in other._transform_graph.transforms
                if {satellite, other._intrinsic_ref} == {t.source, t.target}
            )
            for t in to_port:
                # derivation is natively `self <--deriv-- other`
                if t.target is satellite and derivation.is_invertible:
                    # `self <--deriv--(>) other --t--> satellite` becomes `self --(deriv^-1)--t--> satellite` (preferred final direction)
                    edge = (
                        TransformSequence((derivation.inverted(), t))
                        .canonicalized()
                        .bound(source=intrinsic_ref, target=resolved_ref)
                    )
                elif t.source is satellite:
                    # `self <--deriv-- other <--t-- satellite` becomes `self <--deriv--t-- satellite` (prefer not inverting)
                    edge = (
                        TransformSequence((t, derivation))
                        .canonicalized()
                        .bound(source=resolved_ref, target=intrinsic_ref)
                    )
                elif t.is_invertible:
                    # `self <--deriv-- other (<)--t--> satellite` becomes `self <--deriv--(t^-1)-- satellite` (last resort)
                    edge = (
                        TransformSequence((t.inverted(), derivation))
                        .canonicalized()
                        .bound(source=resolved_ref, target=intrinsic_ref)
                    )
                else:
                    # `self <--deriv-- other --t--> satellite`
                    # Can only be resolved if drops in deriv and t are a subset of each other. Edge case, skip.
                    if relations:
                        warnings.warn(
                            f"Cannot carry over a connection to {satellite.name!r}: Both `by` and the transformation "
                            "from the source to it lose information (e.g. by dropping axes)."
                        )
                    continue

                if edge not in transform_graph.transforms + tuple(to_merge_list):
                    existing_refs.append(resolved_ref)
                    to_merge_list.append(edge)

        if relations and derivation not in transform_graph.transforms + tuple(to_merge_list):
            to_merge_list.append(derivation)
        if not to_merge_list and unchanged_t_scale and intrinsic_ref == self._intrinsic_ref:
            return self
        to_merge = tuple(to_merge_list)

        new_graph = TransformGraph(
            transforms=transform_graph.transforms + to_merge,
            system_refs=transform_graph.system_refs,
        )
        return Multiscale(
            mapping_items,
            _transform_graph=new_graph,
            _intrinsic_ref=intrinsic_ref,
            _zero_scale_axes_by_key=self._zero_scale_axes_by_key,
            _legacy_convention_global_t_scale=transferred_global_t_scale,
        )

    def derive(
        self,
        base_key: ScaleKey,
        *,
        blueprint: Union[BlueprintShapes, BlueprintFactors, None] = None,
        derived_by: Union[SpatialRelation, Sequence[SpatialRelation], None] = None,
        translation_shift_func: Optional[TranslationShiftFunction] = None,
        rounding: Optional[RoundingMethod] = None,
    ) -> "Multiscale":
        """
        Derive a new Multiscale from one of this Multiscale's own scale levels.

        Deriving transfers `.coordinate_systems` from this Multiscale to the new one where possible,
        and keeps the new one within the nifti-zarr convention, if this one is.

        `blueprint`: How to expand the Scale at `base_key` into the derived Multiscale's levels.
        Omit to extract the respective Scale as an independent Multiscale with just that one entry.
        Related:
          - `translation_shift_func`: Function that computes your scaling method's translation shift (see translation shift docs)
          - `rounding`: Your scaling method's rounding behavior. Only required when `blueprint` is `BlueprintFactors`.

        `derived_by`: The relation(s) describing how the new Multiscale was derived from the Scale at `base_key`.
        For example, provide a `Translation` if the derived Multiscale is shifted from its parent's origin, or
        an `AxisRearrangementTo` if axes were dropped/inserted/reordered.
        The derived Multiscale will retain the relation as a reference to an external coordinate system.
        """
        new_ms = Multiscale.from_single(
            self[base_key],
            scale_key=base_key,
            blueprint=blueprint,
            translation_shift_func=translation_shift_func,
            rounding=rounding,
        )
        return new_ms.as_derived_from(self, by=derived_by)

    # Ignore narrowing of `version: str` to Literal (nicer to be explicit)
    def to_ome_zarr(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        version: Literal["0.4", "0.5", "0.6.rc0"],
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        if version not in ome_zarr.SUPPORTED_OME_ZARR_VERSIONS_WRITE:
            raise ValueError("Cannot write OME-Zarr versions other than 0.4, 0.5 and 0.6.rc0.")
        ome_zarr.validate_multiscale(self)
        result: Dict[str, Any] = {"version": version, "datasets": []}
        if self.ome:
            result.update(self.ome.to_ome_zarr())

        if name:
            result["name"] = name

        # Modern: Multiscale is graph + datasets
        if version not in PRE_TRANSFORMS_VERSIONS:
            result.update(self._transform_graph.to_ome_zarr(version=version))
            for key, scale in self.items():
                dataset = ome_zarr.build_dataset_dict(
                    version,
                    key,
                    scale.pixel_size,
                    scale.translation,
                    self._intrinsic_ref,
                    self._zero_scale_axes_by_key.get(key, ()),
                )
                result["datasets"].append(dataset)
            return result

        # Legacy: Determine axes, coordinateTransformations, and datasets, and handle legacy t-scale convention
        assert self._intrinsic_ref.owner, "dev error: must always have intrinsic"
        intrinsic_system_dict = self._intrinsic_ref.owner.to_ome_zarr(name="", version=version)
        result["axes"] = intrinsic_system_dict["axes"]

        multiscale_transforms = self._find_legacy_compatible_coordinate_system()
        do_apply_global_t = False
        if self._legacy_convention_global_t_scale:
            can_apply_global_t_convention = "t" in self.axes() and "t" not in self.scaled_axes()
            # Decision: Convention overrides external system in the legacy case.
            # See `test_multiscale_to_ome_zarr_t_scale_convention_overrides_compatible_coordinate_system`
            if can_apply_global_t_convention:
                do_apply_global_t = True
                multiscale_transforms = None

        if not do_apply_global_t:
            # "Normal" case, no special treatment.
            if multiscale_transforms:
                result["coordinateTransformations"] = multiscale_transforms.to_legacy_ome_zarr()
            for key, scale in self.items():
                dataset = ome_zarr.build_dataset_dict(
                    version,
                    key,
                    scale.pixel_size,
                    scale.translation,
                    serialized_zero_scale_axes=self._zero_scale_axes_by_key.get(key, ()),
                )
                result["datasets"].append(dataset)
            return result

        # Legacy convention where pixel size along t is written as a global scale transform
        axes = list(self.axes())
        assert do_apply_global_t, "implied by branch above"
        pixel_size_t_values = set(scale.pixel_size["t"] for scale in self.values())
        assert len(pixel_size_t_values) == 1, "non-scaling across t implied by do_apply_global_t condition"
        global_t_scale = next(iter(pixel_size_t_values))
        global_pixel_size = PixelSize(t=global_t_scale).with_axes(axes)
        global_scale_transform = ScaleTransform.from_pixel_size(global_pixel_size)
        multiscale_transforms = ome_zarr.MultiscaleTransforms((global_scale_transform,))
        result["coordinateTransformations"] = multiscale_transforms.to_legacy_ome_zarr()
        for key, scale in self.items():
            dataset_scale = scale.pixel_size
            dataset_translation = scale.translation
            if global_t_scale:
                assert dataset_scale["t"] == global_t_scale, "doesn't conform, should've been in the earlier branch"
                # scale.pixel_size["t"] is written in global transforms in this case
                dataset_scale = scale.pixel_size.with_identity("t")
                # and the t-translation must be un-scaled by the factor we've just divided out
                dataset_translation = scale.translation * Factor(t=global_t_scale).inverted()
            dataset = ome_zarr.build_dataset_dict(
                version,
                key,
                dataset_scale,
                dataset_translation,
                serialized_zero_scale_axes=self._zero_scale_axes_by_key.get(key, ()),
            )
            result["datasets"].append(dataset)
        return result

    def as_ref(self, name: CoordinateSystemName) -> NodeRef["Multiscale"]:
        """For Multiscale, making a ref means selecting one of their coordinate systems by name."""
        if name not in (ref.name for ref in self._transform_graph.all_system_refs):
            raise NoSuchCoordinateSystemError(name)
        return NodeRef(name=str(name), owner=self)

    @staticmethod
    def _merge_ome_zarr_axes(scales: Sequence["Scale"]) -> OmeZarrAxes:
        axes = list(scales[0].ome_zarr_axes.keys())
        merged: Dict[AxisKey, Dict[str, Any]] = {a: {} for a in axes}
        for field_name in ("unit", "type", "discrete", "long_name"):
            for a in axes:
                values = {v for s in scales if (v := getattr(s.ome_zarr_axes[a], field_name)) not in (None, "")}
                if len(values) > 1:
                    raise ValueError(
                        f"Conflicting {field_name!r} for axis {a!r} across this Multiscale's Scales: {values!r}. "
                        "Only the first Scale providing a value gets to define it; all others must agree or be blank."
                    )
                merged[a][field_name] = next(iter(values), None)
        return OmeZarrAxes([(a, OmeZarrAxis(name=a, **merged[a])) for a in axes])

    @staticmethod
    def _require_ome_zarr_permitted_type_order(axes: "OmeZarrAxes") -> None:
        """As long as the only format we can serialize to is OME-Zarr, we should enforce the order as early as possible"""
        type_order_ranks = {"time": 0, "channel": 1, None: 99999}
        ranks = [type_order_ranks.get(ax.type, 99999) for a, ax in axes.items()]
        if ranks != sorted(ranks):
            raise ValueError(
                f"When OME-Zarr axis types are specified, axes must be ordered time-channel-others. "
                f"(Reorder using `.with_axes` first?) Received: {axes!r}"
            )

    @staticmethod
    def _require_non_increasing_shape_order(items: Sequence[Tuple[ScaleKey, Scale]]):
        """
        Like _require_ome_zarr_permitted_type_order, this exists to prevent creating Multiscales that will not be
        valid in any serialized format.
        All existing multiscale formats require scales to be ordered from largest to smallest (shape).
        """
        for (prev_key, prev_scale), (key, scale) in zip(items, items[1:]):
            shape = scale.shape
            prev_shape = prev_scale.shape
            increased = [a for a in shape if shape[a] > prev_shape[a]]
            if increased:
                raise ValueError(
                    f"Multiscales must be ordered from largest to smallest. Scale shape increases along "
                    f"{increased} between {prev_key!r} and {key!r} ({prev_shape} -> {shape})."
                )

    def _get_interface_transform(self):
        """Allows a scene to traverse into this subgraph"""
        return IdentityTransform(source=self._intrinsic_ref, target=self.as_ref(self._intrinsic_ref.name))

    @staticmethod
    def _replace_transform_ref(t: Transform, old: NodeRef, new: NodeRef) -> Transform:
        return t.bound(source=new if t.source == old else t.source, target=new if t.target == old else t.target)

    def _find_or_make_unique_ref(
        self, owner: CoordinateSystem, requested_name: str, existing_refs: List[NodeRef]
    ) -> NodeRef:
        """Reuse an existing ref if `owner` is already known under any name; otherwise mint one
        under `requested_name`, disambiguated only if that name is already taken by a *different* owner."""
        for ref in existing_refs:
            if ref.owner is owner:
                return ref
        existing_names = {ref.name for ref in existing_refs}
        return owner.as_ref(self._make_unique_name(requested_name, existing_names))

    @staticmethod
    def _make_unique_name(name: str, exclude: Collection[str]) -> str:
        if name not in exclude:
            return name
        i = 1
        while (candidate := f"{name}-{i}") in exclude:
            i += 1
        return candidate

    def _find_legacy_compatible_coordinate_system(self) -> Optional["ome_zarr.MultiscaleTransforms"]:
        """
        Inspect graph to find MultiscaleTransforms for multiscale_dict["coordinateTransformations"].
        Preferably bound directly on this Multiscale's own intrinsic system (exact, lossless).
        Falls back to one reachable via inserts-only ProjectAxisTransform path,
        in which case the MultiscaleTransforms can be widened to self's axes with identity values.
        This padding is a deliberate approximation: the legacy format cannot express the ProjectAxisTransform
        itself, but MultiscaleTransforms along undefined axes are implicitly identities.
        """

        def _is_raw_or_transferred_legacy(t: Transform):
            try:
                ome_zarr.MultiscaleTransforms.from_transforms((t,))
                return True
            except ValueError:
                pass
            if not isinstance(t, TransformSequence):
                return False
            # `.as_derived_from` can produce any TransformSequence with these at the end/front
            if (
                isinstance(t, ome_zarr.MultiscaleTransforms)
                or (isinstance(t.transforms[-1], ome_zarr.MultiscaleTransforms))
                or (isinstance(t.transforms[0], ome_zarr.InvertedMultiscaleTransforms))
            ):
                return True
            scale_found = translation_found = False
            for i, nested in enumerate(t.transforms):
                compatible_types = (
                    IdentityTransform,
                    ProjectAxisTransform,
                    MapAxisTransform,
                    ScaleTransform,
                    TranslationTransform,
                )
                if not isinstance(nested, compatible_types):
                    return False  # sequence with something we can't handle
                if isinstance(nested, ScaleTransform):
                    # scale at start or at end-1 (or end if translation missing)
                    if scale_found or i not in (0, len(t.transforms), len(t.transforms) - 1):
                        # sequence with multiple scales or scale in middle -- would need to track axes
                        return False
                    scale_found = True
                if isinstance(nested, TranslationTransform):
                    # translation at 1, 0 if scale missing, or at end (inverted or just axesTo-then-translate)
                    if translation_found or i not in (0, 1, len(t.transforms)):
                        # see above
                        return False
                    translation_found = True
            return scale_found or translation_found

        candidates = [t for t in self._transform_graph.transforms if _is_raw_or_transferred_legacy(t)]

        for t in candidates:
            if t.source == self._intrinsic_ref and isinstance(t, ome_zarr.MultiscaleTransforms):
                return t
        for t in candidates:
            if isinstance(t, ome_zarr.MultiscaleTransforms):
                return t
            if isinstance(t, ome_zarr.InvertedMultiscaleTransforms):
                return t.inverted()
        for t in candidates:
            if not isinstance(t, TransformSequence):
                continue
            assert self._intrinsic_ref in {t.source, t.target}, "should never have two-hops in a Multiscale graph"
            if isinstance(t.transforms[-1], ome_zarr.MultiscaleTransforms) and self._is_pure_axis_reshape(
                t.transforms[:-1]
            ):
                ms_t = t.transforms[-1]
                assert isinstance(ms_t, ome_zarr.MultiscaleTransforms) and isinstance(ms_t.source, NodeRef)
                return self._reshaped_multiscale_transforms(
                    ms_t, source_axes=tuple(ms_t.source.owner.axes()), target_axes=tuple(self.axes())
                )
            elif isinstance(t.transforms[0], ome_zarr.InvertedMultiscaleTransforms) and self._is_pure_axis_reshape(
                t.transforms[1:]
            ):
                ms_t = t.transforms[0].inverted()
                assert isinstance(ms_t, ome_zarr.MultiscaleTransforms) and isinstance(ms_t.source, NodeRef)
                return self._reshaped_multiscale_transforms(
                    ms_t, source_axes=tuple(ms_t.source.owner.axes()), target_axes=tuple(self.axes())
                )
        for t in candidates:
            try:
                if t.source == self._intrinsic_ref:
                    return ome_zarr.MultiscaleTransforms.from_transforms((t,))
                assert t.target == self._intrinsic_ref, f"all transforms must connect to self, {t!r}"
                return ome_zarr.MultiscaleTransforms.from_transforms((t.inverted(),))
            except ValueError:
                continue
        for t in candidates:
            # (scale) (+ translation) (+ structural) or (structural) (+ scale) (+ translation)
            if not isinstance(t, TransformSequence):
                continue
            children = list(t.transforms)
            scale = translation = None
            scale_i = translation_i = None
            for i, c in enumerate(children):
                if isinstance(c, ScaleTransform):
                    scale_i = i
                    scale = children.pop(i)
                if isinstance(c, TranslationTransform):
                    translation_i = i
                    translation = children.pop(i)
            if not self._is_pure_axis_reshape(tuple(children)):
                continue  # Should already be ensured anyway
            if t.source == self._intrinsic_ref:
                if translation_i is not None and scale_i is not None and scale_i > translation_i:
                    # self --translation--scale--> external
                    # The inverse of this sequence would be a sequence that *undoes* the original.
                    # Here we would instead need the same translation expressed in `external`'s scale:
                    # self --scale--(translation*scale)--> external
                    # Flipped, not inverse. Rather weird thing to do. There are reasons OME-Zarr is always (scale[, translation]).
                    continue
                is_physical_then_structural = translation_i == 0 if scale is None else scale_i == 0
                assert isinstance(t.target, NodeRef), "all transforms in graph must be bound and resolved"
                ms_t_source_axes = tuple(self.axes()) if is_physical_then_structural else tuple(t.target.owner.axes())
                identity_scale = ScaleTransform((1.0,) * len(ms_t_source_axes))
                scale = scale if scale is not None else identity_scale
                ms_children = (scale,) if translation is None else (scale, translation)
                ms_t = ome_zarr.MultiscaleTransforms(ms_children)
                return self._reshaped_multiscale_transforms(
                    ms_t, source_axes=ms_t_source_axes, target_axes=tuple(self.axes())
                )
            assert t.target == self._intrinsic_ref, "intrinsic must be either source or target"
            # Inverted: external --t--> self; presumably stored this way because the sequence is non-invertible.
            # Legacy transforms express self --global_transforms--> external, so the physical component of t must be inverted
            if translation_i is not None and scale_i is not None and scale_i < translation_i:
                # external --translation--scale--> self ; see above
                continue
            last = len(t.transforms)
            is_physical_then_structural = translation_i == last if scale is None else scale_i == last
            assert isinstance(t.source, NodeRef), "all transforms in graph must be bound and resolved"
            ms_t_source_axes = tuple(t.source.owner.axes()) if is_physical_then_structural else tuple(self.axes())
            identity_scale = ScaleTransform((1.0,) * len(ms_t_source_axes))
            scale = scale if scale is not None else identity_scale
            ms_children = (scale,) if translation is None else (translation, scale)
            ms_t = ome_zarr.InvertedMultiscaleTransforms(ms_children).inverted()
            return self._reshaped_multiscale_transforms(
                ms_t, source_axes=ms_t_source_axes, target_axes=tuple(self.axes())
            )
        return None

    @staticmethod
    def _is_pure_axis_reshape(path: Tuple["Transform", ...]) -> bool:
        """True if `path` is pure axis bookkeeping (drop/insert/reorder).
        For legacy OME-Zarr, in this case a subsequent (scale[+translation]) can absorb the axis
        rearrangement (_reshaped_multiscale_transforms), even if the format cannot express the
        rearrangement itself."""
        if len(path) < 1:
            return True
        collapsed = TransformSequence(path).canonicalized() if len(path) > 1 else path[0]
        structural_types = (IdentityTransform, ProjectAxisTransform, MapAxisTransform)
        if isinstance(collapsed, TransformSequence):
            return all(isinstance(child, structural_types) for child in collapsed.transforms)
        return isinstance(collapsed, structural_types)

    @staticmethod
    def _reshaped_multiscale_transforms(
        t: ome_zarr.MultiscaleTransforms, *, source_axes: Tuple[AxisKey, ...], target_axes: Tuple[AxisKey, ...]
    ) -> "ome_zarr.MultiscaleTransforms":
        scale = ome_zarr.scale_to_pixel_size_with_normalized_zeros(t.scale_transform.scale, source_axes).with_axes(
            target_axes
        )
        widened_scale = ScaleTransform(scale=scale.to_tuple())
        if t.translation_transform is None:
            return ome_zarr.MultiscaleTransforms(transforms=(widened_scale,))
        translation = t.translation_transform.to_translation(source_axes).with_axes(target_axes)
        return ome_zarr.MultiscaleTransforms(
            transforms=(widened_scale, TranslationTransform.from_translation(translation))
        )
