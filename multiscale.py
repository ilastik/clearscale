from abc import ABC
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, TypeVar, Mapping, Generic, Union

from lazyflow.utility.io_util.clearscale import Shape, Factor, Spacing, Unit
from lazyflow.utility.io_util.clearscale.tagged_values import ShapeLike, Axes, RoundingMethod

ScaleKey = TypeVar("ScaleKey", bound=str)
ValueType = TypeVar("ValueType", Shape, Factor, "Scale")
_Self = TypeVar("_Self", bound="ScaleMapping[Any, Any]")


@dataclass(frozen=True, slots=True)
class Scale:
    shape: Shape
    spacing: Optional[Spacing] = None
    unit: Optional[Unit] = None

    def __post_init__(self):
        object.__setattr__(self, "shape", Shape(self.shape))
        if self.spacing is None:
            object.__setattr__(self, "spacing", Spacing.fromkeys(self.shape.keys()))
        else:
            object.__setattr__(self, "spacing", Spacing(self.spacing))
        if self.unit is None:
            object.__setattr__(self, "unit", Unit.fromkeys(self.shape.keys()))
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


class ScaleMapping(ABC, Mapping[ScaleKey, ValueType], Generic[ScaleKey, ValueType]):
    def __init__(self, *args, **kwargs):
        self._mapping = OrderedDict(*args, **kwargs)
        if any(v is None for v in self._mapping.values()):
            raise ValueError(f"None values not allowed. Received: {kwargs}")

    def __repr__(self):
        map_substr = self._mapping.__repr__()[len(type(self._mapping).__name__) :]
        return str(type(self).__name__) + map_substr

    def __getitem__(self, key: ScaleKey):
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
        if isinstance(other, ScaleMapping):
            return self._mapping == other._mapping
        if isinstance(other, OrderedDict) or isinstance(other, dict):
            return self._mapping == other
        return False

    def copy(self):
        return type(self)(self._mapping)


class BlueprintShapes(ScaleMapping[Shape]):
    @classmethod
    def from_multiscale(cls, multiscale: "Multiscale", reference: ScaleKey, exclude_reference=False) -> _Self:
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
        name_pattern="s{}",
    ):
        """Generate Blueprint where each scale is a `step` downsampling of the previous scale.
        Applies scaling uniformly to all axes until they become singleton."""
        cls._validate_name_pattern(name_pattern)
        cls._validate_resampling_step(step)
        if step == 1:
            return cls({name_pattern.format(0): base_shape})
        if not shape_limit:
            shape_limit = Shape.all_singletons(base_shape)
        if not only:
            only = shape_limit.keys()
        else:
            only = [a for a in only if a in shape_limit]

        shape_limit = Shape(shape_limit).reorder(base_shape)
        cls._validate_shape_limit(base_shape, only, shape_limit, step)

        scales_items = []
        for i in range(0, max_levels):
            scale_key = name_pattern.format(i)
            scale_factor = step**i
            scaling = Factor.uniform(base_shape, scale_factor).to_identity_except(only)
            scaled_shape = base_shape.scale_by(scaling, rounding=rounding)
            scales_items.append((scale_key, scaled_shape))
            if (step > 1 and all(scaled_shape[axis] <= shape_limit[axis] for axis in only)) or (
                step < 1 and all(scaled_shape[axis] >= shape_limit[axis] for axis in only)
            ):
                break
        return cls(scales_items)

    def to_factors(self, reference_shape: Shape) -> "BlueprintFactors":
        raise NotImplementedError()

    @staticmethod
    def _validate_name_pattern(pattern: str):
        if pattern.format(0) == pattern:
            raise ValueError(
                "Name pattern must contain exactly one placeholder for scale index (received: '{}')".format(pattern)
            )

    @staticmethod
    def _validate_resampling_step(step: Union[int, float]):
        if step <= 0:
            raise ValueError("Cannot downsample by a negative step size (received: {})".format(step))

    @staticmethod
    def _validate_shape_limit(base_shape: Shape, only: Axes, shape_limit: Shape, step: Union[int, float]):
        for axis in only:
            if step > 1 and shape_limit[axis] > base_shape[axis]:
                raise ValueError("Cannot limit downsampling to a shape larger than the base (along {}).".format(axis))
            if step < 1 and shape_limit[axis] < base_shape[axis]:
                raise ValueError("Cannot limit upsampling to a shape smaller than the base (along {}).".format(axis))


class BlueprintFactors(ScaleMapping[Factor]):
    @classmethod
    def from_multiscale(cls, multiscale: "Multiscale", reference: ScaleKey, exclude_reference=False) -> _Self:
        raise NotImplementedError()

    def to_shapes(self, reference_shape: Shape, rounding="round") -> BlueprintShapes:
        raise NotImplementedError()


class Multiscale(ScaleMapping[Scale]): ...
