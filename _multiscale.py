from abc import ABC
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from enum import StrEnum
from typing import Optional, TypeVar, Mapping, Generic, Union, Sequence, Callable, Iterable, List, Tuple

from lazyflow.utility.io_util.clearscale import Shape, Factor, Spacing, Unit
from lazyflow.utility.io_util.clearscale._axis_values import ShapeLike, Axes, RoundingMethod, OrderedAxes, _AxisValues

ScaleKey = TypeVar("ScaleKey", bound=str)
ValueType = TypeVar("ValueType", Shape, Factor, "Scale")
AxisValuesType = TypeVar("AxisValuesType", Shape, Factor)
_Self = TypeVar("_Self", bound="ScaleMapping[Any, Any]")
DEFAULT_NAME_PATTERN = "s{}"


class _DuplicatePolicy(StrEnum):
    ERROR = "error"
    KEEP = "keep"
    KEEP_FIRST = "keep_first"
    KEEP_LAST = "keep_last"


@dataclass(frozen=True, slots=True)
class Scale:
    shape: Shape
    spacing: Optional[Spacing] = None
    unit: Optional[Unit] = None

    def __post_init__(self):
        object.__setattr__(self, "shape", Shape(self.shape))
        if self.spacing is None:
            object.__setattr__(self, "spacing", Spacing.fromkeys(self.shape))
        else:
            object.__setattr__(self, "spacing", Spacing(self.spacing))
        if self.unit is None:
            object.__setattr__(self, "unit", Unit.fromkeys(self.shape))
        else:
            object.__setattr__(self, "unit", Unit(self.unit))
        if self.shape.keys() != self.spacing.keys() or self.shape.keys() != self.unit.keys():
            raise ValueError(
                f"Tried to set up invalid scale: Axiskeys differ "
                f"(shape={self.shape.keys()}, spacing={self.spacing.keys()}, unit={self.unit.keys()})"
            )

    def has_pixel_size(self):
        return not self.unit.is_default() or not self.spacing.is_default()

    def to_display_string(self, name=""):
        shape = ", ".join(f"{axis}: {size}" for axis, size in self.shape.items())
        name_and_shape = f'"{name}" ({shape})' if name else f"{shape}"
        pixel_size = ""
        if self.has_pixel_size():
            axis_strings = []
            for axis in self.shape.keys():
                if axis == "c":
                    continue
                spacing = self.spacing[axis]
                unit = ""
                if self.unit[axis]:
                    unit = f" {self.unit[axis]}"
                elif axis != "t":
                    unit = " px"
                axis_strings.append(f"{axis}: {spacing:g}{unit}")
            pixel_size = " at pixel size: " + ", ".join(axis_strings)
        return f"{name_and_shape}{pixel_size}"


class _ScaleMapping(ABC, Mapping[ScaleKey, ValueType], Generic[ScaleKey, ValueType]):
    """Common base class for Multiscale, BlueprintShapes and BlueprintFactors"""

    def __init__(self, *args, **kwargs):
        self._mapping = OrderedDict(*args, **kwargs)
        if any(v is None for v in self._mapping.values()):
            raise ValueError(f"None values not allowed. Received: {kwargs}")

    def __repr__(self):
        map_substr = self._mapping.__repr__()[len(type(self._mapping).__name__) :]
        return str(self.__class__.__name__) + map_substr

    def __getitem__(self, key: ScaleKey) -> ValueType:
        if key not in self:
            raise KeyError(f"No such scale: '{key}' (available: {list(self.keys())})")
        return self._mapping[key]

    def __contains__(self, key: ScaleKey):
        return key in self._mapping

    def __iter__(self):
        return iter(self._mapping)

    def __len__(self):
        return len(self._mapping)

    def keys(self):
        return self._mapping.keys()

    def values(self):
        return self._mapping.values()

    def items(self):
        return self._mapping.items()

    def __eq__(self, other):
        if isinstance(other, _ScaleMapping):
            return self._mapping == other._mapping
        if isinstance(other, OrderedDict) or isinstance(other, dict):
            return self._mapping == other
        return False

    def copy(self):
        return self.__class__(self._mapping)

    def filter_items(self, predicate: Callable[[ScaleKey, ValueType], bool]) -> _Self:
        items = [(k, v) for k, v in self.items() if predicate(k, v)]
        return self.__class__(items)

    def rekey(self, name_pattern=DEFAULT_NAME_PATTERN) -> _Self:
        self._validate_name_pattern(name_pattern)
        items = [(name_pattern.format(i), v) for i, v in enumerate(self.values())]
        return self.__class__(items)

    def drop_before(self, key: ScaleKey, inclusive=False) -> _Self:
        keys = list(self.keys())
        if key not in keys:
            raise KeyError(f"No such scale: '{key}' (available: {keys})")

        start_idx = keys.index(key)
        if inclusive:
            start_idx += 1

        items = [(k, v) for k, v in self.items() if k in keys[start_idx:]]
        return self.__class__(items)

    @staticmethod
    def _validate_name_pattern(pattern: str):
        if pattern.format(0) == pattern:
            raise ValueError(
                f"Name pattern must contain exactly one placeholder for scale index (received: '{pattern}')"
            )


class _ScaledAxisValues(_ScaleMapping[str, AxisValuesType], Generic[AxisValuesType]):
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

    def _with_values(self, values: Sequence[_AxisValues]) -> _Self:
        return self.__class__(zip(self.keys(), values))

    def reorder(self, axes: OrderedAxes) -> _Self:
        return self._with_values([value.reorder(axes) for value in self.values()])

    @staticmethod
    def _resolve_duplicates(
        raw_items: Iterable[Tuple[ScaleKey, _AxisValues]], on_duplicate: _DuplicatePolicy, on_duplicate_prefer
    ) -> List[Tuple[ScaleKey, _AxisValues]]:
        """
        Ensure raw_items contains no duplicate values. Resolve duplicates according to on_duplicate:
        "error": Raise error if there are duplicates.
        "keep_first": Keep the first key seen with any particular value.
        "keep_last": Keep the last key seen with any particular value.
        The two "keep" policies can be combined with `on_duplicate_prefer`.
        In this case, if the `on_duplicate_prefer` key is involved in a duplication, it has priority over first/last.
        """
        raw_items = list(raw_items)
        if on_duplicate == _DuplicatePolicy.KEEP:
            return raw_items

        by_value = defaultdict(list)
        for k, v in raw_items:
            by_value[tuple(v.items())].append(k)
        duplicates = {tuple(ks): v for v, ks in by_value.items() if len(ks) > 1}

        if duplicates and on_duplicate == _DuplicatePolicy.ERROR:
            raise ValueError(f"Duplicate values not allowed. Collisions: {duplicates}")

        pop_keys = []
        for dup_keys in duplicates:
            if on_duplicate_prefer is not None and on_duplicate_prefer in dup_keys:
                keep = on_duplicate_prefer
            elif on_duplicate == _DuplicatePolicy.KEEP_FIRST:
                keep = dup_keys[0]
            elif on_duplicate == _DuplicatePolicy.KEEP_LAST:
                keep = dup_keys[-1]
            else:
                raise AssertionError(f"Invalid duplicate scale policy: '{on_duplicate}'")

            pop_keys.extend(k for k in dup_keys if k != keep)

        return [(k, v) for k, v in raw_items if k not in pop_keys]


class BlueprintShapes(_ScaledAxisValues[Shape]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self._mapping.items():
            self._mapping[k] = Shape(v)

    @classmethod
    def from_multiscale(
        cls, multiscale: "Multiscale", reference: ScaleKey, exclude_reference=False
    ) -> "BlueprintShapes":
        raise NotImplementedError()

    @classmethod
    def uniform_steps(
        cls,
        *,
        step: Union[int, float],
        base_shape: Shape,
        rounding: RoundingMethod,
        shape_limit: Optional[ShapeLike] = None,
        only: Optional[Axes] = None,
        max_levels: Optional[int] = 42,
        name_pattern=DEFAULT_NAME_PATTERN,
        on_duplicate=_DuplicatePolicy.KEEP_FIRST,
        on_duplicate_prefer: ScaleKey = None,
    ) -> "BlueprintShapes":
        """Generate Blueprint where each scale is a `step` downsampling of the previous scale.
        Applies scaling uniformly to all axes until they become singleton."""
        cls._validate_name_pattern(name_pattern)
        cls._validate_resampling_step(step)
        if step == 1:
            return cls({name_pattern.format(0): base_shape})
        if not shape_limit:
            shape_limit = Shape.all_singletons(base_shape)
        if not only:
            only = base_shape.keys()
        else:
            only = [a for a in only if a in base_shape]

        shape_limit = Shape(shape_limit).reorder([a for a in shape_limit if a in base_shape])
        cls._validate_shape_limit(base_shape, only, shape_limit, step)

        scales_items = []
        for i in range(0, max_levels):
            scale_key = name_pattern.format(i)
            scale_factor = step**i
            scaling = Factor.uniform(base_shape, scale_factor).with_identity_except(only)
            scaled_shape = base_shape.scale_by(scaling, rounding=rounding)
            scales_items.append((scale_key, scaled_shape))
            if (step > 1 and all(scaled_shape[axis] <= shape_limit[axis] for axis in shape_limit)) or (
                step < 1 and all(scaled_shape[axis] >= shape_limit[axis] for axis in shape_limit)
            ):
                break
        scales_items = cls._resolve_duplicates(scales_items, on_duplicate, on_duplicate_prefer)
        bp = cls(scales_items)
        return bp.rekey(name_pattern)

    def to_factors(self, reference: ShapeLike) -> "BlueprintFactors":
        factors = [Shape(reference).scaling_to(scale_shape) for scale_shape in self.values()]
        return BlueprintFactors(zip(self.keys(), factors))

    def with_sizes(self, other: ShapeLike, axes: Axes):
        return self._with_values([shape.with_values(other, axes) for shape in self.values()])

    @staticmethod
    def _validate_resampling_step(step: Union[int, float]):
        if step <= 0:
            raise ValueError(f"Cannot downsample by a negative step size (received: {step})")

    @staticmethod
    def _validate_shape_limit(base_shape: Shape, only: Axes, shape_limit: Shape, step: Union[int, float]):
        for axis in only:
            if axis not in shape_limit:
                continue
            if step > 1 and shape_limit[axis] > base_shape[axis]:
                raise ValueError(f"Cannot limit downsampling to a shape larger than the base (along {axis}).")
            if step < 1 and shape_limit[axis] < base_shape[axis]:
                raise ValueError(f"Cannot limit upsampling to a shape smaller than the base (along {axis}).")


class BlueprintFactors(_ScaledAxisValues[Factor]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self._mapping.items():
            self._mapping[k] = Factor(v)

    @classmethod
    def from_shapes(cls, shapes: Mapping[ScaleKey, ShapeLike], reference: ShapeLike):
        return BlueprintShapes(shapes).to_factors(reference)

    @classmethod
    def from_multiscale(cls, multiscale: "Multiscale", reference: ScaleKey, exclude_reference=False) -> _Self:
        raise NotImplementedError()

    def to_shapes(self, reference: ShapeLike, *, rounding: RoundingMethod) -> BlueprintShapes:
        ref = Shape(reference)
        shapes = [ref.scale_by(scale_factor, rounding=rounding) for scale_factor in self.values()]
        return BlueprintShapes(zip(self.keys(), shapes))

    def with_identity(self, axes: Axes):
        return self._with_values([factor.with_identity(axes) for factor in self.values()])


class Multiscale(_ScaleMapping[str, Scale]): ...
