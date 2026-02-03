import math
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import (
    Mapping,
    Generic,
    TypeVar,
    Union,
    Container,
    Sequence,
    Callable,
    Optional,
    List,
    Literal,
)

AxisKey = TypeVar("AxisKey", bound=str)
ValueType = TypeVar("ValueType", int, float, str)
_Self = TypeVar("_Self", bound="TaggedValues[Any, Any]")
Axes = Union[Container[AxisKey], str]
OrderedAxes = Sequence[AxisKey]
FactorLike = Union["Factor", "Shape", Mapping[AxisKey, int], Mapping[AxisKey, float]]
ShapeLike = Union["Shape", Mapping[AxisKey, int]]
RoundingMethod = Union[Literal["ceil"], Literal["floor"], Literal["round"], Callable[[float], int]]


class _AxisValues(ABC, Mapping[AxisKey, ValueType], Generic[AxisKey, ValueType]):
    """
    Base class for "tagged dictionaries" that map axis keys to values (like shape, resolution, unit).
    Instantiation and usage of the subclasses should work pretty much like usual dicts, but with
    guaranteed (and explicit) order of the elements and immutability.

    Currently just wraps an OrderedDict.
    Custom classes are used instead of plain OrderedDicts to:
    - hide the internal implementation from the consumer in case this might need to change in the future
    - enforce immutability
    - control which parts of the OrderedDict API are exposed
    """

    @property
    @abstractmethod
    def _default(self):
        """Subclasses should provide the Class._default property.
        Default value for axes where no value is provided."""
        ...

    def __init__(self, *args, **kwargs):
        self._mapping = OrderedDict(*args, **kwargs)
        if not self._mapping:
            raise ValueError(f"Empty {self.__class__.__name__}. Received: {args=}, {kwargs=}")
        if any(v is None for v in self._mapping.values()):
            raise ValueError(f"None values not allowed. Received: {list(self._mapping.values())}")

    def __repr__(self):
        map_substr = self._mapping.__repr__()[len(type(self._mapping).__name__) :]
        return str(self.__class__.__name__) + map_substr

    def __getitem__(self, key: AxisKey):
        return self._mapping[key]

    def __contains__(self, key: AxisKey):
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
        if isinstance(other, _AxisValues):
            return self._mapping == other._mapping
        if isinstance(other, OrderedDict) or isinstance(other, dict):
            return self._mapping == other
        return False

    def copy(self):
        return self.__class__(self._mapping)

    @classmethod
    def fromkeys(cls, keys: Sequence[AxisKey]) -> _Self:
        return cls(zip(keys, [cls._default] * len(keys)))

    def is_default(self) -> bool:
        """Check if all values in this metadata are the default value."""
        return self == self.__class__.fromkeys(self)

    def with_order(self, axes: Sequence[AxisKey]) -> _Self:
        """
        Reorder to `axes`.

        Inserts this type's default value for target axes that this instance doesn't have yet.

        Equivalent to pandas.DataFrame.reindex(axes, fill_value=self._default).
        """
        reordered_items = [(a, self[a] if a in self else self._default) for a in axes]
        return self.__class__(reordered_items)

    def with_default(self, axes: Axes) -> _Self:
        """
        Reset the values for `axes` to the type's default value, keeping the rest unchanged.

        Equivalent to pandas.DataFrame.mask(self in axes, other=self._default).
        """
        reset_items = [(a, self._default if a in axes else self[a]) for a in self]
        return self.__class__(reset_items)

    def with_default_except(self, axes: Axes) -> _Self:
        """
        Keep the values for `axes` and reset the remaining values to the type's default value.

        Equivalent to pandas.DataFrame.where(self in axes, other=self._default).
        """
        keep_items = [(a, self[a] if a in axes else self._default) for a in self]
        return self.__class__(keep_items)

    def with_values(self, other: Mapping[AxisKey, ValueType], axes: Axes):
        if not axes:
            return self.__class__(self)
        replaced_items = []
        for a in self:
            new_value = self[a]
            if a in axes and a in other and other[a] is not None:
                new_value = other[a]
            if type(self[a]) != type(new_value):
                if isinstance(new_value, int) and isinstance(self[a], float):
                    new_value = float(new_value)
                else:
                    raise TypeError(
                        f"During attempted merge: Cannot cast '{new_value}' of type "
                        f"{type(new_value).__name__} to {type(self[a]).__name__}"
                    )
            replaced_items.append((a, new_value))

        return self.__class__(replaced_items)


class _AxisFloats(_AxisValues[AxisKey, float], ABC):
    """
    Base for Scaling, Resolution and Translation. Ensures values are floats.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for axis, value in self._mapping.items():
            if isinstance(value, int):
                self._mapping[axis] = float(value)
                continue
            if not isinstance(value, float):
                raise TypeError(f"All values must be float. Got {type(value).__name__} for axis '{axis}'.")


class Factor(_AxisFloats):
    """
    Describes relative scaling factors from some shape to another.
    The values are in units of "scaled pixels per raw pixel".
    This makes them divisors for the original shape.
    """

    _default = 1.0

    def with_order(self, axes: OrderedAxes) -> _Self:
        """
        Reorder to `axes`.

        Inserts 1.0 for target axes that this Scaling doesn't have yet.

        Examples:
            >>> res = Factor(z=0.25, y=120., x=120., t=0.1)
            >>> res.with_order("tczyx")
            Scaling(t=0.1, c=1.0, z=0.25, y=120.0, x=120.0)
        """
        return super().with_order(axes)

    @classmethod
    def identity(cls, axes: Sequence[AxisKey]) -> "Factor":
        """Create a new identity Scaling (1.0 along all axes) with `axes`."""
        return super().fromkeys(axes)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for axis, value in self._mapping.items():
            if value == 0:  # 0 is nonsense both for scaling factors and spacing
                raise ValueError(f"Scaling factor cannot be 0 (got 0 for axis '{axis}').")

    @classmethod
    def uniform(cls, axes: Sequence[AxisKey], factor: float) -> "Factor":
        """Create a new Scaling with `axes` and all values being `factor`."""
        return Factor(zip(axes, [factor] * len(axes)))

    def magnitude(self):
        return math.prod(self.values())

    def is_identity(self) -> bool:
        """True if this Factor is the identity scaling (1.0 along all axes)."""
        return super().is_default()

    def is_downscaling(self) -> bool:
        """True if the product across all axes is greater than 1.
        Note: Factors act as divisors for shape (e.g. 1024 / 2 = 512)."""
        return math.prod(self.values()) > 1

    def is_upscaling(self) -> bool:
        """True if the product across all axes is lesser than 1.
        Note: Factors act as divisors for shape (e.g. 1024 / 0.5 = 2048)."""
        return math.prod(self.values()) < 1

    def with_identity(self, axes: Axes) -> _Self:
        """Reset the values for `axes` to 1.0."""
        return super().with_default(axes)

    def with_identity_except(self, axes: Axes) -> _Self:
        """Reset the values for all axes except `axes` to 1.0."""
        return super().with_default_except(axes)

    def to_physical(self, base: "Spacing") -> "Spacing":
        """
        Convert relative scaling factor to absolute physical spacing.
        Identical to base.scaled_by(self).
        """
        return Spacing(base).scaled_by(self)


class Spacing(_AxisFloats):
    """
    Describes absolute scaling factors, i.e. physical pixel size.
    The values are in "units (e.g. nanometer) per pixel".
    """

    _default = 1.0

    @classmethod
    def from_vigra(cls, axistags: "vigra.AxisTags") -> _Self:
        vigra_default_resolution = 0.0
        axes = []
        resolutions = []
        for tag in axistags:
            axes.append(tag.key)
            resolutions.append(tag.resolution if tag.resolution != vigra_default_resolution else cls._default)
        return cls(zip(axes, resolutions))

    def is_identity(self) -> bool:
        """True if this Spacing is the unit spacing (1.0 along all axes)."""
        return super().is_default()

    def with_identity(self, axes: Axes) -> _Self:
        """Reset the values for `axes` to 1.0."""
        return super().with_default(axes)

    def with_identity_except(self, axes: Axes) -> _Self:
        """Reset the values for all axes except `axes` to 1.0."""
        return super().with_default_except(axes)

    def to_vigra(self, axistags: Optional["vigra.AxisTags"]) -> "vigra.AxisTags":
        """
        Creates or modifies vigra.AxisTags with the values in this Scaling/Resolution.
        :param axistags: (Optional) Existing AxisTags to modify.
            Only transfers values for axes shared between the AxisTags and this Scaling/Resolution.
        :return: Returns the newly created or the modified AxisTags.
        """
        try:
            import vigra
        except ImportError as e:
            raise ImportError(
                'This function requires the package "vigra". '
                "Please install it using e.g. `conda install -c conda-forge vigra`"
            ) from e
        tags = axistags if axistags else vigra.defaultAxistags("".join(self.keys()))
        for tag in tags:
            if tag.key in self and self[tag.key] != self._default:
                tags.setResolution(tag.key, self[tag.key])
        return tags

    def scaled_by(self, factor: Union[Factor, Mapping[AxisKey, float], float]) -> "Spacing":
        """
        Scale this Spacing by factor to obtain a scaled Spacing.
        This is an axis-wise operation:
        - Missing axes in `factor` default to 1.0 (no change)
        - Passing a scalar (float/int) applies it uniformly to all axes
        - Extra axes in `factor` are rejected
        Note if passing scalar: factor 2.0 means double spacing = half resolution.
        """
        if isinstance(factor, float) or isinstance(factor, int):
            factor = Factor.uniform(self, factor)
        elif not isinstance(factor, Factor):
            factor = Factor(factor)
        base_axes = set(self.keys())
        factor_axes = set(factor.keys())
        invalid_axes = factor_axes - base_axes
        if invalid_axes:
            raise ValueError(
                f"Attempted to scale axes with no base spacing: "
                f"{sorted(invalid_axes)} not present in {sorted(base_axes)}"
            )
        reordered = factor.with_order(self)
        scaled_items = [(a, reordered[a] * self[a]) for a in self]
        return Spacing(scaled_items)


class Translation(_AxisFloats):
    """Describes a shift in physical units."""

    _default = 0.0

    def with_order(self, axes: Sequence[AxisKey]) -> "Translation":
        """
        Reorder to `axes`.

        Inserts 0.0 for target axes that this Translation doesn't have yet.

        Examples:
            >>> translate = Translation(y=0.5, x=0.5, t=0.3)
            >>> translate.with_order("tczyx")
            Translation(t=0.3, c=0.0, z=0.0, y=0.5, x=0.5)
        """
        return super().with_order(axes)

    @classmethod
    def identity(cls, axes: Sequence[AxisKey]) -> "Translation":
        """Create a new identity Translation (0.0 along all axes) with `axes`."""
        return super().fromkeys(axes)

    def is_identity(self) -> bool:
        """True if this Translation is the identity translation (0.0 along all axes)."""
        return super().is_default()

    def __add__(self, other: "Translation") -> "Translation":
        if not isinstance(other, Translation):
            return NotImplemented
        if list(self) != list(other):
            raise ValueError(f"Incompatible axes/order: {list(self)} vs {list(other)}")
        return Translation((a, self[a] + other[a]) for a in self)

    def __sub__(self, other: "Translation") -> "Translation":
        if not isinstance(other, Translation):
            return NotImplemented
        if list(self) != list(other):
            raise ValueError(f"Incompatible axes/order: {list(self)} vs {list(other)}")
        return Translation((a, self[a] - other[a]) for a in self)


class Unit(_AxisValues[AxisKey, str]):
    """
    Describes the physical units of the image's axes.
    Together with a Resolution, this can form a complete pixel size description.
    Example: {"t": "seconds", "x": "nm, "y": "nm, "c": ""}.
    """

    _default = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for axis, value in self._mapping.items():
            if not isinstance(value, str):
                raise TypeError(f"All values must be strings. Got {type(value).__name__} for axis '{axis}'.")

    def with_order(self, axes: Sequence[AxisKey]) -> _Self:
        """
        Reorder to `axes`.

        Inserts "" for target axes that this Unit doesn't have yet.

        Examples:
            >>> unit = Unit(y="nm", x="nm", t="sec")
            >>> unit.with_order("tczyx")
            Unit(t="sec", c="", z="", ="nm", x="nm")
        """
        return super().with_order(axes)

    @classmethod
    def empty(cls, axes: Sequence[AxisKey]) -> "Unit":
        """Create a new Unit with `axes` and empty string values."""
        return super().fromkeys(axes)


class Offset(_AxisValues[AxisKey, int]):
    """
    Describes the number of pixels of distance from some reference point to another.
    """

    _default = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for axis, value in self._mapping.items():
            if not isinstance(value, int):
                raise TypeError(f"All values must be integer. Got {type(value).__name__} for axis '{axis}'.")

    def with_order(self, axes: Sequence[AxisKey]) -> "Offset":
        """
        Reorder to `axes`.

        Inserts 0 for target axes that this Offset doesn't have yet.

        Examples:
            >>> crop_offset = Offset(y=15, x=37, t=23)
            >>> crop_offset.with_order("tczyx")
            Offset(t=23, c=0, z=0, y=15, x=37)
        """
        return super().with_order(axes)

    def to_physical(self, spacing: Union[Spacing, Mapping[AxisKey, float]]) -> Translation:
        """
        Multiply with `resolution` to obtain this Offset as a Translation in physical units.
        """
        items_in_physical_units = [(a, self[a] * spacing[a]) for a in self]
        return Translation(items_in_physical_units)


class Shape(_AxisValues[AxisKey, int]):
    """
    Describes the number of pixels in the image.
    """

    _default = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for axis, value in self._mapping.items():
            if not isinstance(value, int):
                raise TypeError(f"All values must be integer. Got {type(value).__name__} for axis '{axis}'.")
            if value < 1:
                raise ValueError(f"Shape cannot be lower than 1 (got {value} for axis '{axis}').")

    @classmethod
    def all_singletons(cls, axes: OrderedAxes):
        return super().fromkeys(axes)

    def with_order(self, axes: OrderedAxes) -> _Self:
        """
        Reorder to `axes`.

        Inserts 1 for target axes that this Shape doesn't have yet.

        Examples:
            >>> shape = Shape(y=256, x=256, t=23)
            >>> shape.with_order("tczyx")
            Shape(t=23, c=1, z=1, y=256, x=256)
        """
        return super().with_order(axes)

    def with_ones(self, axes: Axes) -> _Self:
        """Reset the values for `axes` to 1."""
        return super().with_default(axes)

    def matches(self, other: ShapeLike, *, only: Optional[Axes] = None) -> bool:
        """Permissive value matching.
        True if shapes are equal in all shared axes, optionally further constrained to `only`."""
        shared = set(self.keys()) & set(other.keys())
        if only:
            shared &= set(only)
        return all(self[axis] == other[axis] for axis in shared)

    def non_singleton_axes(self, axes: Optional[Axes] = None) -> List[AxisKey]:
        """Return axes along which this Shape is singleton (value is 1).

        :param axes: (Optional) Return subset of `axes` along which this Shape is singleton."""
        axes = axes if axes is not None else self.keys()
        return [a for a in axes if a in self and self[a] != self._default]

    def scaled_by(self, factor: Union[Factor, Mapping[AxisKey, float]], *, rounding: RoundingMethod) -> "Shape":
        """
        Returns the Shape of this image when scaled by `factor`.

        :param factor:
        :param rounding: Specify how the scaling implementation used to scale the image data treats uneven cases.
            Allowed: "ceil", "floor", or a function that takes a float and returns an int.
            Example: When scaling 11 pixels by factor 2, does your method produce 5 or 6 pixels?
            Common examples:
             - skimage.transform.rescale: `rounding=round`.
             - skimage.transform.downscale_local_mean: `rounding="ceil"`.
        """
        if rounding == "ceil":
            rounding = math.ceil
        elif rounding == "floor":
            rounding = int
        elif rounding == "round":
            rounding = round
        factor = Factor(factor).with_order(self)

        def _rescale_size(s: int, f: float) -> int:
            """
            Rescale a single dimension of a shape.
            Floor-round to match behavior of OpResize, and ensure minimum size is 1.
            """
            return max(rounding(s / f), self._default)

        scaled_shape = self.__class__([(a, _rescale_size(size, factor[a])) for a, size in self.items()])
        return scaled_shape

    def scaling_to(self, resized: "Shape") -> "Factor":
        """
        Returns the Scaling factors of this Shape that would produce the `resized` shape.
        """
        if list(self) != list(resized):
            raise ValueError(
                "Original and resized shapes must have identical axes in identical order. "
                f"Original axes: {list(self)}; target axes: {list(resized)}"
            )
        # In multiscale image context, scaling "factors" are technically divisors for the shape
        # (factor 2.0 means half the shape).
        return Factor((a, self[a] / resized[a]) for a in self)
