from abc import ABC
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from enum import StrEnum
from functools import wraps
from typing import (
    Optional,
    TypeVar,
    Mapping,
    Generic,
    Union,
    Sequence,
    Callable,
    Iterable,
    List,
    Tuple,
    Literal,
    Dict,
    Any,
)

from lazyflow.utility.io_util.clearscale import Shape, Factor, Spacing, Unit, Translation, _ome_zarr
from lazyflow.utility.io_util.clearscale._axis_values import (
    ShapeLike,
    Axes,
    RoundingMethod,
    OrderedAxes,
    _AxisValues,
    AxisKey,
)

ScaleKey = TypeVar("ScaleKey", bound=str)
ValueType = TypeVar("ValueType", Shape, Factor, "Scale")
AxisValuesType = TypeVar("AxisValuesType", Shape, Factor)
_Self = TypeVar("_Self", bound="ScaleMapping[Any, Any]")
DEFAULT_NAME_PATTERN = "s{}"

TranslationShiftFunction = Callable[["Scale", "Scale"], "Translation"]
"""
base_scale: the reference scale being transformed from
target_scale: the new scale being created (with 0 translation)
Returns: target_scale's translation
"""


class _DuplicatePolicy(StrEnum):
    ERROR = "error"
    KEEP_ALL = "keep_all"
    KEEP_FIRST = "keep_first"
    KEEP_LAST = "keep_last"


def half_pixel_shift(base: "Scale", target: "Scale") -> "Translation":
    """Apply half-pixel shift in each downsampled axis."""
    if list(base.spacing.keys()) != list(target.spacing.keys()):
        raise ValueError("Axis mismatch. Cannot compute half-pixel shift between unrelated Scales.")
    shift_items = []
    for axis, target_spacing in target.spacing.items():
        base_spacing = base.spacing[axis]
        if target_spacing > base_spacing:
            # Downsampled - apply half pixel shift
            shift_items.append((axis, 0.5 * (target_spacing - base_spacing)))
        else:
            shift_items.append((axis, 0.0))
    return Translation(shift_items)


@dataclass(frozen=True, slots=True)
class Scale:
    shape: Shape
    spacing: Optional[Spacing] = None
    unit: Optional[Unit] = None
    translation: Optional[Translation] = None

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
        if self.translation is None:
            object.__setattr__(self, "translation", Translation.fromkeys(self.shape))
        else:
            object.__setattr__(self, "translation", Translation(self.translation))
        if (
            self.shape.keys() != self.spacing.keys()
            or self.shape.keys() != self.unit.keys()
            or self.shape.keys() != self.translation.keys()
        ):
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

    def first_value(self) -> ValueType:
        return next(iter(self.values()))

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

    def to_dict(self) -> OrderedDict[ScaleKey, OrderedDict[AxisKey, Union[int, float]]]:
        return OrderedDict([(scale_key, OrderedDict(axis_values)) for scale_key, axis_values in self.items()])

    def _with_values(self, values: Sequence[_AxisValues]) -> _Self:
        return self.__class__(zip(self.keys(), values))

    def with_order(self, axes: OrderedAxes) -> _Self:
        return self._with_values([value.with_order(axes) for value in self.values()])

    @staticmethod
    def _resolve_duplicates(
        raw_items: Iterable[Tuple[ScaleKey, _AxisValues]], on_duplicate: _DuplicatePolicy, on_duplicate_prefer
    ) -> List[Tuple[ScaleKey, _AxisValues]]:
        """
        Ensure raw_items contains no duplicate values. Resolve duplicates according to on_duplicate:
        "error": Raise error if there are duplicates.
        "keep_all": Skip (return raw_items as list)
        "keep_first": Keep the first key seen with any particular value.
        "keep_last": Keep the last key seen with any particular value.
        The two "keep" policies can be combined with `on_duplicate_prefer`.
        In this case, if the `on_duplicate_prefer` key is involved in a duplication, it has priority over first/last.
        """
        raw_items = list(raw_items)
        if on_duplicate == _DuplicatePolicy.KEEP_ALL:
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
    def from_multiscale(cls, multiscale: "Multiscale") -> "BlueprintShapes":
        return cls([(key, scale.shape) for key, scale in multiscale.items()])

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

        shape_limit = Shape(shape_limit).with_order([a for a in shape_limit if a in base_shape])
        cls._validate_shape_limit(base_shape, only, shape_limit, step)

        scales_items = []
        for i in range(0, max_levels):
            scale_key = name_pattern.format(i)
            scale_factor = step**i
            scaling = Factor.uniform(base_shape, scale_factor).with_identity_except(only)
            scaled_shape = base_shape.scaled_by(scaling, rounding=rounding)
            scales_items.append((scale_key, scaled_shape))
            if (step > 1 and all(scaled_shape[axis] <= shape_limit[axis] for axis in shape_limit)) or (
                step < 1 and all(scaled_shape[axis] >= shape_limit[axis] for axis in shape_limit)
            ):
                break
        scales_items = cls._resolve_duplicates(scales_items, on_duplicate, on_duplicate_prefer)
        bp = cls(scales_items)
        return bp.rekey(name_pattern)

    @classmethod
    def downscale_powers_of_2_xyz(
        cls,
        *,
        base_shape: Shape,
        rounding: RoundingMethod,
        shape_limit: Optional[ShapeLike] = None,
        max_levels: int = 42,
        name_pattern=DEFAULT_NAME_PATTERN,
        on_duplicate=_DuplicatePolicy.KEEP_FIRST,
        on_duplicate_prefer: ScaleKey = None,
    ):
        return cls.uniform_steps(
            step=2,
            only="xyz",
            base_shape=base_shape,
            rounding=rounding,
            shape_limit=shape_limit,
            max_levels=max_levels,
            name_pattern=name_pattern,
            on_duplicate=on_duplicate,
            on_duplicate_prefer=on_duplicate_prefer,
        )

    def scaled_axes(self) -> tuple[AxisKey, ...]:
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

    def to_factors(self, reference: Shape) -> "BlueprintFactors":
        factors = [Shape(reference).scaling_to(scale_shape) for scale_shape in self.values()]
        return BlueprintFactors(zip(self.keys(), factors))

    def apply_to_scale(
        self, base: Scale, translation_shift_func: Optional[TranslationShiftFunction] = None
    ) -> "Multiscale":
        if list(self.first_value().keys()) != list(base.shape.keys()):
            raise ValueError(
                f"Cannot apply blueprint with axes {list(self.first_value().keys())} "
                f"to base scale with axes {list(base.shape.keys())}. "
                "Axes must match exactly. Maybe blueprint.with_order(base.shape) first?"
            )

        scales = []
        for scale_key, target_shape in self.items():
            factor = base.shape.scaling_to(target_shape)
            new_spacing = base.spacing.scaled_by(factor)

            if translation_shift_func is not None:
                target_scale_pre_shift = Scale(
                    shape=target_shape, spacing=new_spacing, unit=base.unit, translation=base.translation
                )
                shift = self._compute_and_validate_shift(translation_shift_func, base, target_scale_pre_shift)
                new_translation = base.translation + shift
            else:
                new_translation = base.translation

            scales.append(
                (scale_key, Scale(shape=target_shape, spacing=new_spacing, unit=base.unit, translation=new_translation))
            )

        return Multiscale(scales)

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
    def from_shapes(cls, shapes: Mapping[ScaleKey, ShapeLike], reference: Shape):
        return BlueprintShapes(shapes).to_factors(reference)

    @classmethod
    def from_multiscale(cls, multiscale: "Multiscale", reference: Shape) -> _Self:
        return BlueprintShapes.from_multiscale(multiscale).to_factors(reference)

    @property
    def scaled_axes(self) -> tuple[AxisKey, ...]:
        """Axes where any factor is not 1.0."""
        if len(self) < 2:
            return ()

        scaled = set()
        for factor in self.values():
            scaled.update(axis for axis, value in factor.items() if value != 1.0)

        all_axes = next(iter(self.values())).keys()
        return tuple(axis for axis in all_axes if axis in scaled)

    def to_shapes(self, reference: ShapeLike, *, rounding: RoundingMethod) -> BlueprintShapes:
        ref = Shape(reference)
        shapes = [ref.scaled_by(scale_factor, rounding=rounding) for scale_factor in self.values()]
        return BlueprintShapes(zip(self.keys(), shapes))

    def apply_to_scale(self):
        return NotImplementedError()

    def with_identity(self, axes: Axes):
        return self._with_values([factor.with_identity(axes) for factor in self.values()])


class Multiscale(_ScaleMapping[str, Scale]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, scale in self._mapping.items():
            if scale.shape.keys() != self.axes():
                raise ValueError(
                    f"All Scales must have identical axes. Scale at '{key}' has {list(scale.shape.keys())}"
                )

    @staticmethod
    @wraps(BlueprintShapes.apply_to_scale)
    def from_shapes(blueprint: BlueprintShapes, base: Scale, *args, **kwargs):
        return blueprint.apply_to_scale(base, *args, **kwargs)

    @staticmethod
    @wraps(BlueprintFactors.apply_to_scale)
    def from_factors(blueprint: BlueprintFactors, base: Scale, *args, **kwargs):
        return blueprint.apply_to_scale(base, *args, **kwargs)

    def axes(self):
        return self.first_value().shape.keys()

    def scaled_axes(self) -> tuple[AxisKey, ...]:
        """Axes where spacings differ across scales."""
        if len(self) < 2:
            return ()

        spacings = list(scale.spacing for scale in self.values())
        first_spacing = spacings[0]
        scaled = []

        for axis in first_spacing.keys():
            first_value = first_spacing[axis]
            if any(spacing[axis] != first_value for spacing in spacings[1:]):
                scaled.append(axis)

        return tuple(scaled)

    def to_ome_zarr(
        self,
        *,
        version: Literal["0.4", "0.5"],
        axis_types: Union[None, Literal["infer"], Mapping[str, Literal["space", "time", "channel"]]] = None,
    ) -> Dict[str, Any]:
        _ome_zarr.validate_multiscale(self)

        first_scale = self.first_value()
        axes = list(first_scale.shape.keys())

        ome_axes = _ome_zarr.build_axis_dicts(axes, first_scale.unit, axis_types)

        result = {"version": version, "axes": ome_axes, "datasets": []}

        # If single-scale, do not include global (multiscale) coordinateTransformations
        scaled_axes = self.scaled_axes() if len(self) > 1 else tuple(axes)
        global_scale = first_scale.spacing.with_identity(scaled_axes)
        global_translation = first_scale.translation if len(self) > 1 else Translation.identity(axes)

        global_transforms = _ome_zarr.build_multiscale_transforms(global_scale, global_translation)
        if global_transforms:
            result["coordinateTransformations"] = global_transforms

        for key, scale in self.items():
            dataset_scale = scale.spacing.with_identity_except(scaled_axes)
            dataset_translation = scale.translation - global_translation

            dataset = _ome_zarr.build_dataset_dict(key, dataset_scale, dataset_translation)
            result["datasets"].append(dataset)

        return result
