from typing import (
    Hashable,
    Protocol,
    Iterator,
    TypeVar,
    Callable,
    Literal,
    Mapping,
    Any,
    Sequence,
    Union,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from clearscale._axis_values import Shape, Translation
    from clearscale._multiscale import Scale

AxisKey = Hashable
AxisKeyT = TypeVar("AxisKeyT", bound=AxisKey)
AxisKeyT_co = TypeVar("AxisKeyT_co", bound=AxisKey, covariant=True)


class _Ordered(Protocol[AxisKeyT_co]):
    """Defined order (unlike Iterable, Collection), but not necessarily indexable (unlike Sequence).
    To match e.g. strings, but also odict_keys and of course _AxisMapping"""

    def __iter__(self) -> Iterator[AxisKeyT_co]: ...
    def __len__(self) -> int: ...


OrderedAxes = _Ordered[AxisKey]
ShapeLike = Union["Shape", Mapping[AxisKeyT, int]]
TranslationShiftFunction = Callable[["Scale", "Scale"], "Translation"]
"""
base_scale: the reference scale being transformed from
target_scale: the new scale being created (with 0 translation)
Returns: target_scale's translation
"""
PixelSizingMethod = Literal["shape_ratio", "corner_ratio", "exact_factor"]
"""
How the scaling method used spaces output pixels.
Options for both BlueprintShapes and BlueprintFactors:
* shape_ratio: output_spacing = input_spacing * input_shape / output_shape
* corner_ratio: output_spacing = input_spacing * (input_shape - 1) / (output_shape - 1)
Option only for BlueprintFactors:
* exact_factor: output_spacing = input_spacing * factor
"""


class HasShape(Protocol):
    @property
    def shape(self) -> Sequence[int]: ...


ShapeValue = Union[Sequence[int], ShapeLike, HasShape]


class ShapeSourceMap(Protocol):
    def __getitem__(self, path: str, /) -> ShapeValue: ...


ShapeSource = Union[Literal["singletons"], Callable[[str], ShapeValue], ShapeSourceMap, Mapping[str, ShapeValue]]
"""
Lets clearscale know how to obtain a zarr's array shape in this Python environment.
Options:
- "singletons": Skip obtaining shapes and use placeholder all-singleton shapes (like `Shape(x=1, y=1, z=1)`)
    This sets `Multiscale.has_shapes = False` as a convenience indicator.
- Callable: A function that takes a relative path that *should* point to an array, and retrieves its shape tuple
    Example: zarr.open_array
- Map: A dict-like that can be indexed to retrieve shapes like `{ <relative path> : <shape tuple or array> }`
    Example: zarr.Group or fsspec.FSMap
"""


class ZarrGroup(ShapeSourceMap, Protocol):
    """Matches e.g. zarr.Group (zarr-python) or z5py.Group."""

    @property
    def attrs(self) -> Mapping[str, Any]: ...
