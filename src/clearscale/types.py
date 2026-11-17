from typing import (
    Hashable,
    Protocol,
    Iterator,
    TypeVar,
    Callable,
    Literal,
    Mapping,
    MutableMapping,
    Any,
    Sequence,
    Union,
    TYPE_CHECKING,
    TypeAlias,
)

if TYPE_CHECKING:
    from clearscale._axis_values import Factor, PixelSize, Shape, Translation

CoordinateSystemName: TypeAlias = str
"""The name of a coordinate system, unique within the Multiscale or Scene that has it."""
ScaleKey: TypeAlias = str
"""Identifies one scale of a Multiscale or Blueprint: the path of its array, relative to the zarr group that holds it."""
AxisKey = Hashable
"""Any hashable object whose `str()` is exactly the axis name, such as "x".

That string is what gets written to OME-Zarr as the axis name, so axes of one Multiscale need distinct strings.
Plain strings are the usual choice.
Other objects work if they define `__str__` accordingly (dataclass default does not)`.
"""
AxisKeyT = TypeVar("AxisKeyT", bound=AxisKey)
AxisKeyT_co = TypeVar("AxisKeyT_co", bound=AxisKey, covariant=True)
Json: TypeAlias = Union[str, int, float, bool, Mapping[str, "Json"], Sequence["Json"], None]
"""JSON-shaped values. This annotation does not validate encoder compatibility, finite numbers, or cycles."""


class _Ordered(Protocol[AxisKeyT_co]):
    """Defined order (unlike Iterable, Collection), but not necessarily indexable (unlike Sequence).

    To match e.g. strings, but also odict_keys and of course _AxisMapping.
    """

    def __iter__(self) -> Iterator[AxisKeyT_co]: ...
    def __len__(self) -> int: ...


OrderedAxes = _Ordered[AxisKey]
ShapeLike = Union["Shape[AxisKeyT]", Mapping[AxisKeyT, int]]
FactorLike = Union["Factor[AxisKeyT]", Mapping[AxisKeyT, float]]
RoundingMethod: TypeAlias = Literal["ceil", "floor", "round", "round_half_up", "error_on_round"]
"""How the scaling method rounds output shape when applying the factor to the input shape would result in a fractional shape.

- `"ceil"`: Round toward positive infinity.
- `"floor"`: Round toward negative infinity.
- `"round"`: Round to the nearest integer, with ties going to the even integer (Python's `round`).
- `"round_half_up"`: Round to the nearest integer, with ties going to the larger integer (positive infinity).
- `"error_on_round"`: Require an integer-valued input; raise `ValueError` if rounding would be needed.
"""
TranslatingMethod: TypeAlias = Literal["half_pixel_space_preservation", "discrete_bin_center", "first_value_decimation"]
"""How the scaling method shifts the position of the first pixel in space.

See characterization docs for detail."""
PixelSizingMethod = Literal["shape_ratio", "corner_ratio", "exact_factor"]
"""How the scaling method used spaces output pixels.

Options for both BlueprintShapes and BlueprintFactors:

- shape_ratio: output_spacing = input_spacing * input_shape / output_shape
- corner_ratio: output_spacing = input_spacing * (input_shape - 1) / (output_shape - 1)

Option only for BlueprintFactors:

- exact_factor: output_spacing = input_spacing * factor
"""


class HasShape(Protocol):
    @property
    def shape(self) -> Sequence[int]: ...


class HasPixelGrid(Protocol):
    """Anything that describes where an array's pixels lie in physical space, like a [Scale][clearscale.Scale].

    Physical coordinate of pixel index `i` is `i * pixel_size + translation`.
    """

    @property
    def pixel_size(self) -> "PixelSize": ...
    @property
    def translation(self) -> "Translation": ...


ShapeValue = Union[Sequence[int], ShapeLike, HasShape]


class ShapeSourceMap(Protocol):
    def __getitem__(self, path: str, /) -> ShapeValue: ...


ShapeSource = Union[Literal["singletons"], Callable[[str], ShapeValue], ShapeSourceMap, Mapping[str, ShapeValue]]
"""Lets clearscale know how to obtain a zarr's array shape in this Python environment.

Options:

- "singletons": Skip obtaining shapes and use placeholder all-singleton shapes (like `Shape(x=1, y=1, z=1)`).
    This sets `Multiscale.has_shapes = False` as a convenience indicator.
- Callable: A function that takes a relative path that *should* point to an array, and retrieves its shape tuple.
    Example: zarr.open_array
- Map: A dict-like that can be indexed to retrieve shapes like `{ <relative path> : <shape tuple or array> }`.
    Example: zarr.Group or fsspec.FSMap
"""


class ZarrGroup(ShapeSourceMap, Protocol):
    """Matches e.g. zarr.Group (zarr-python) or z5py.Group."""

    @property
    def attrs(self) -> Mapping[str, Any]: ...


class MutableZarrGroup(Protocol):
    """A group whose `attrs` can be modified, e.g. zarr.Group (zarr-python) with mode='w'."""

    @property
    def attrs(self) -> MutableMapping[str, Any]: ...


__all__ = [
    "CoordinateSystemName",
    "ScaleKey",
    "AxisKey",
    "AxisKeyT",
    "AxisKeyT_co",
    "Json",
    "OrderedAxes",
    "ShapeLike",
    "FactorLike",
    "RoundingMethod",
    "TranslatingMethod",
    "PixelSizingMethod",
    "HasShape",
    "HasPixelGrid",
    "ShapeValue",
    "ShapeSourceMap",
    "ShapeSource",
    "ZarrGroup",
    "MutableZarrGroup",
]
