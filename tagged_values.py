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
    overload,
    Callable,
    Tuple,
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
RoundingMethod = Union[Literal["ceil"], Literal["floor"], Callable[[float], int]]


class AxisValues(ABC, Mapping[AxisKey, ValueType], Generic[AxisKey, ValueType]):
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
        if any(v is None for v in self._mapping.values()):
            raise ValueError(f"None values not allowed. Received: {kwargs}")

    def __repr__(self):
        map_substr = self._mapping.__repr__()[len(type(self._mapping).__name__) :]
        return str(type(self).__name__) + map_substr

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
        if isinstance(other, AxisValues):
            return self._mapping == other._mapping
        if isinstance(other, OrderedDict) or isinstance(other, dict):
            return self._mapping == other
        return False

    def copy(self):
        return type(self)(self._mapping)

    @classmethod
    def fromkeys(cls, keys: Sequence[AxisKey]) -> _Self:
        return cls(zip(keys, [cls._default] * len(keys)))

    def is_default(self) -> bool:
        """Check if all values in this metadata are the default value."""
        return self == type(self).fromkeys(self)

    def reorder(self, axes: Sequence[AxisKey]) -> _Self:
        """
        Reorder to `axes`.

        Inserts this type's default value for target axes that this instance doesn't have yet.

        Equivalent to pandas.DataFrame.reindex(axes, fill_value=self._default).
        """
        reordered_items = [(a, self[a] if a in self else self._default) for a in axes]
        return type(self)(reordered_items)

    def reset(self, axes: Axes) -> _Self:
        """
        Reset the values for `axes` to the type's default value, keeping the rest unchanged.

        Equivalent to pandas.DataFrame.mask(self in axes, other=self._default).
        """
        reset_items = [(a, self._default if a in axes else self[a]) for a in self]
        return type(self)(reset_items)

    def reset_except(self, axes: Axes) -> _Self:
        """
        Keep the values for `axes` and reset the remaining values to the type's default value.

        Equivalent to pandas.DataFrame.where(self in axes, other=self._default).
        """
        keep_items = [(a, self[a] if a in axes else self._default) for a in self]
        return type(self)(keep_items)

    @overload
    def partition(self, predicate: Callable[[AxisKey], bool]) -> Tuple[_Self, _Self]: ...

    @overload
    def partition(self, include_axes: Axes) -> Tuple[_Self, _Self]: ...

    def partition(self, first: Union[Axes, Callable[[AxisKey], bool]]) -> Tuple[_Self, _Self]:
        """Split this metadata into two, with values for `first` axes in the first instance, the rest in the second.

        :param first: Either a set of axis keys (str, list, tuple, ...) or a predicate function (Callable).
            - If axis keys, values for given axes go into the first instance.
            - If a predicate, values for axes where `predicate(axis)` is True go into the first instance.

        :return: A (first, rest) tuple of two instances with the same axes, but complementary values.

        Example:
            >>> resolution = Spacing(x=0.1, y=0.1, z=0.5, t=1.0)
            >>> by_predicate = resolution.partition(lambda a: a in ["x", "y"])
            (Resolution(x=0.1, y=0.1, z=1.0, t=1.0), Resolution(x=1.0, y=1.0, z=0.5, t=1.0))
            >>> by_axes = resolution.partition(["x", "y"])
            (Resolution(x=0.1, y=0.1, z=1.0, t=1.0), Resolution(x=1.0, y=1.0, z=0.5, t=1.0))
            >>> by_axes == by_predicate
            True
        """
        if callable(first):
            predicate = first
            keep_axes = [key for key in self if (predicate(key))]
        else:
            keep_axes = first
        return self.reset_except(keep_axes), self.reset(keep_axes)

    def merge(
        self, other: Mapping[AxisKey, ValueType], *, only: Optional[Axes] = None, force: Optional[Axes] = None
    ) -> _Self:
        """Merge values from `other` into this instance.

        Args:
            other: Another instance of this metadata type.
            only: (Optional) Axes to limit the merge to.
            force: (Optional) Axes to replace with the other value even if not default in this instance.
                `only` takes precedence. If `only` is given, axes in `force` are ignored if they are not in `only`.

        Returns:
            A copy of this instance with potentially some replaced values.

        Raises:
            TypeError if a value to be copied from other has a different type than this instance,
            and it can't be cast unambiguously.

        Examples:
            >>> raw = Shape(x=128, y=128, c=3)
            >>> feature_chunk = Shape(x=32, y=32, c=25)
            >>> feature_image = raw.reset("c").merge(feature_chunk)
            >>> feature_image
            Shape(x=128, y=128, c=25)

            >>> shape = Shape(x=128, y=128, c=3)
            >>> shape.merge()
            >>> feature_image = raw.reset("c").merge(feature_chunk)
            >>> feature_image
            Shape(x=128, y=128, c=25)
        """
        only = only if only is not None else self.keys()
        force = force or []
        replaced_items = []
        for a in self:
            new_value = self[a]
            if (a in force or self[a] == self._default) and a in only and a in other and other[a] is not None:
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

        return type(self)(replaced_items)

    def overwrite_with(self, other: Mapping[AxisKey, ValueType], axes: Axes):
        return self.merge(other, only=axes, force=axes)


class AxisFloats(AxisValues[AxisKey, float], ABC):
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


class Factor(AxisFloats):
    """
    Describes relative scaling factors from some shape to another.
    The values are in units of "scaled pixels per raw pixel".
    This makes them divisors for the original shape.

    Examples:
        >>> raw = Shape(x=128, y=128, c=3)
        >>> scaled = Shape(x=32, y=32, c=3)
        >>> scaling = raw.scaling_to(scaled)
        >>> scaling
        Scaling(x=4.0, y=4.0, c=1.0)

        >>> raw = Shape(x=128, y=128, c=2)
        >>> scaled = Shape(x=32, y=32, c=4)
        >>> scaling = raw.scaling_to(scaled, fixed="c")
        >>> scaling
        Scaling(x=4.0, y=4.0, c=1.0)
    """

    _default = 1.0

    def reorder(self, axes: Sequence[AxisKey]) -> _Self:
        """
        Reorder to `axes`.

        Inserts 1.0 for target axes that this Scaling doesn't have yet.

        Examples:
            >>> res = Factor(z=0.25, y=120., x=120., t=0.1)
            >>> res.reorder("tczyx")
            Scaling(t=0.1, c=1.0, z=0.25, y=120.0, x=120.0)
        """
        return super().reorder(axes)

    @classmethod
    def identity(cls, axes: Sequence[AxisKey]) -> "Factor":
        """Create a new identity Scaling (1.0 along all axes) with `axes`."""
        return super().fromkeys(axes)

    def is_identity(self) -> bool:
        """True if this Scaling is the unit scaling (1.0 along all axes)."""
        return super().is_default()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for axis, value in self._mapping.items():
            if value == 0:  # 0 is nonsense both for scaling factors and resolution
                self._mapping[axis] = self._default

    @classmethod
    def uniform(cls, axes: Sequence[AxisKey], factor: float) -> "Factor":
        """Create a new Scaling with `axes` and all values being `factor`."""
        return Factor(zip(axes, [factor] * len(axes)))

    def with_ones(self, axes: Axes) -> _Self:
        """Reset the values for `axes` to 1.0."""
        return super().reset(axes)

    def with_ones_except(self, axes: Axes) -> _Self:
        """Reset the values for all axes except `axes` to 1.0."""
        return super().reset_except(axes)

    def to_physical(self, base_resolution: Union["Spacing", Mapping[AxisKey, float]]) -> "Spacing":
        """
        Multiply with `base_resolution` to obtain the resolution at this Scaling level.
        """
        items_in_physical_units = [(a, self[a] * base_resolution[a]) for a in self]
        return Spacing(items_in_physical_units)


class Spacing(Factor):
    """
    Describes absolute scaling factors, i.e. physical pixel size.
    The values are in "units (e.g. nanometer) per pixel".
    """

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

    @classmethod
    def from_vigra(cls, axistags: "vigra.AxisTags") -> _Self:
        vigra_default_resolution = 0.0
        axes = []
        resolutions = []
        for tag in axistags:
            axes.append(tag.key)
            resolutions.append(tag.resolution if tag.resolution != vigra_default_resolution else cls._default)
        return cls(zip(axes, resolutions))


class Translation(AxisFloats):
    _default = 0.0

    def reorder(self, axes: Sequence[AxisKey]) -> "Translation":
        """
        Reorder to `axes`.

        Inserts 0.0 for target axes that this Translation doesn't have yet.

        Examples:
            >>> translate = Translation(y=0.5, x=0.5, t=0.3)
            >>> translate.reorder("tczyx")
            Translation(t=0.3, c=0.0, z=0.0, y=0.5, x=0.5)
        """
        return super().reorder(axes)

    @classmethod
    def identity(cls, axes: Sequence[AxisKey]) -> "Translation":
        """Create a new identity Translation (0.0 along all axes) with `axes`."""
        return super().fromkeys(axes)

    def is_identity(self) -> bool:
        """True if this Translation is the identity translation (0.0 along all axes)."""
        return super().is_default()

    def __add__(self, other: Union["Translation", Mapping[AxisKey, float]]) -> "Translation":
        sum_items = [(a, self[a] + other[a]) for a in self]
        return Translation(sum_items)


class Unit(AxisValues[AxisKey, str]):
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

    def reorder(self, axes: Sequence[AxisKey]) -> _Self:
        """
        Reorder to `axes`.

        Inserts "" for target axes that this Unit doesn't have yet.

        Examples:
            >>> unit = Unit(y="nm", x="nm", t="sec")
            >>> unit.reorder("tczyx")
            Unit(t="sec", c="", z="", ="nm", x="nm")
        """
        return super().reorder(axes)


class Offset(AxisValues[AxisKey, int]):
    """
    Describes the number of pixels of distance from some reference point to another.
    """

    _default = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for axis, value in self._mapping.items():
            if not isinstance(value, int):
                raise TypeError(f"All values must be integer. Got {type(value).__name__} for axis '{axis}'.")

    def reorder(self, axes: Sequence[AxisKey]) -> "Offset":
        """
        Reorder to `axes`.

        Inserts 0 for target axes that this Offset doesn't have yet.

        Examples:
            >>> crop_offset = Offset(y=15, x=37, t=23)
            >>> crop_offset.reorder("tczyx")
            Offset(t=23, c=0, z=0, y=15, x=37)
        """
        return super().reorder(axes)

    def to_physical(self, resolution: Union[Spacing, Mapping[AxisKey, float]]) -> Translation:
        """
        Multiply with `resolution` to obtain the translation in physical units that this Offset represents.
        """
        items_in_physical_units = [(a, self[a] * resolution[a]) for a in self]
        return Translation(items_in_physical_units)


class Shape(AxisValues[AxisKey, int]):
    """
    Describes the number of pixels in the image.
    """

    _default = 1

    def reorder(self, axes: Sequence[AxisKey]) -> _Self:
        """
        Reorder to `axes`.

        Inserts 1 for target axes that this Shape doesn't have yet.

        Examples:
            >>> shape = Shape(y=256, x=256, t=23)
            >>> shape.reorder("tczyx")
            Shape(t=23, c=1, z=1, y=256, x=256)
        """
        return super().reorder(axes)

    def with_ones(self, axes: Axes) -> _Self:
        """Reset the values for `axes` to 1."""
        return super().reset(axes)

    def singletons(self, axes: Optional[Axes] = None) -> List[int]:
        """Return axes along which this Shape is singleton (value is 1).

        :param axes: (Optional) Return subset of `axes` along which this Shape is singleton."""
        axes = axes if axes is not None else self.keys()
        return [a for a in axes if a in self and self[a] != self._default]

    def scale_by(self, factors: Union[Factor, Mapping[AxisKey, float]], *, rounding: RoundingMethod) -> "Shape":
        """
        Returns the Shape of this image when scaled by `factors`.

        :param factors:
        :param rounding: Function used to round fractional outputs of the scaling. This function
            should mirror the behavior of the scaling implementation used to scale the image's data. By default,
            f_round is simply a cast to int, i.e. floor-rounding. This matches e.g. skimage.transform.rescale.
        """
        if rounding == "ceil":
            rounding = math.ceil
        elif rounding == "floor":
            rounding = int

        def _rescale_size(size: int, factor: float) -> int:
            """
            Rescale a single dimension of a shape.
            Floor-round to match behavior of OpResize, and ensure minimum size is 1.
            """
            return max(rounding(size / factor), self._default)

        scaled_shape = type(self)([(a, _rescale_size(size, factors[a])) for a, size in self.items()])
        return scaled_shape

    def scaling_to(self, resized: ShapeLike, fixed: Optional[Axes] = None) -> "Factor":
        """
        Returns the Scaling factors of this Shape that would produce the `resized` shape.
        """
        common_axes = [a for a in resized if a in self]
        extra_axes = set(self.keys()) ^ set(resized.keys())
        if extra_axes:
            raise ValueError(
                "Original and resized shapes must have the same axes. "
                f"Original axes: {list(self.keys())}; target axes {list(resized.keys())}"
            )
        scale_values = [resized[a] for a in common_axes]
        base_values = [self[a] for a in common_axes]
        # This scale's scaling relative to base_shape.
        # Scaling "factors" are technically divisors for the shape (factor 2.0 means half the shape).
        scaling = Factor((a, base / s) for a, s, base in zip(common_axes, scale_values, base_values))
        if fixed:
            return scaling.with_ones(fixed)
        else:
            return scaling
