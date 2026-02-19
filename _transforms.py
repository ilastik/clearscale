import enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Optional, List, Tuple, Dict, Mapping, Iterable, Sequence, TYPE_CHECKING, TypeVar

from lazyflow.utility.io_util.clearscale._axis_values import _AxisMapping, AxisKey, OrderedAxes, Spacing, Translation

if TYPE_CHECKING:
    try:
        from typing import Self  # py 3.11+
    except ImportError:
        try:
            from typing_extensions import Self  # py 3.10 + optional dep
        except ImportError:
            _Self = TypeVar("_Self")
            Self = _Self

RelativePath = str  # RFC-5: scene["coordinateTransformations"][]["input"]["path"]
CoordinateSystemName = str  # str from ["input"]["name"]
CoordinateSystemKey = Tuple[Optional[RelativePath], CoordinateSystemName]


class Continuity(enum.Enum):
    Discrete = enum.auto()
    Continuous = enum.auto()


@dataclass(frozen=True, slots=True)
class AxisSemantics:
    continuity: Optional[Continuity] = None
    _ome_zarr_type: Optional[str] = None
    _ome_zarr_unit: Optional[str] = None


class CoordinateSystem(_AxisMapping[AxisKey, AxisSemantics]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def without_semantics(cls, axes: OrderedAxes) -> "CoordinateSystem":
        return cls([(a, AxisSemantics()) for a in axes])


@dataclass(frozen=True, slots=True)
class Transform(ABC):
    raw: Optional[CoordinateSystemKey] = field(
        default=None, kw_only=True
    )  # default required; optional when nested in sequence or bijection (and will be None by default in clearscale when nested)
    derived: Optional[CoordinateSystemKey] = field(default=None, kw_only=True)
    raw_axes: Optional[OrderedAxes] = field(
        default=None, kw_only=True
    )  # default None; required when nested in byDimension
    derived_axes: Optional[OrderedAxes] = field(default=None, kw_only=True)

    @property
    @abstractmethod
    def is_invertible(self) -> bool: ...
    def bind(self, raw: CoordinateSystemKey, derived: CoordinateSystemKey) -> "Self":
        # binding required to use the Transform in a TransformGraph
        return replace(self, raw=raw, derived=derived)

    def unbind(self) -> "Self":
        # for nesting inside a TransformSequence or BijectionTransform
        return replace(self, raw=None, derived=None)

    @property
    def is_bound(self) -> bool:
        return self.raw is not None and self.derived is not None

    # Import methods: These handle normalizing common image processing packages' conventions for
    # computing/providing transforms to OME-Zarr's convention.
    # They're only applicable for certain subclasses, so should go there
    # def from_skimage(self):
    #     # Method must know how skimage stores transforms and determine which
    #     # coordinate system is the .input and which the .output when the user passes an skimage transform object
    #     ...
    # def from_itk(self):
    #     # E.g. probably need to do A_omezarr = np.linalg.inv(A_ITK_homogeneous) for ITK affines
    #     ... # from_simpleitk, from_scipy, from_antspy...


@dataclass(frozen=True, slots=True)
class IdentityTransform(Transform):
    ome_zarr_path: Optional[str] = (
        None  # ome-zarr rfc-5: transform["path"] (e.g. for displacement); required for e.g. displacement
    )

    @property
    def is_invertible(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class ScaleTransform(Transform):
    spacing: Spacing
    ome_zarr_path: Optional[str] = None

    @property
    def is_invertible(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class TranslationTransform(Transform):
    translation: Translation
    ome_zarr_path: Optional[str] = None

    @property
    def is_invertible(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class AffineTransform(Transform):
    m: List[List]
    ome_zarr_path: Optional[str] = None

    @property
    def is_invertible(self) -> bool:
        return False  # TODO: Figure out how to determine this :)


@dataclass(frozen=True, slots=True)
class TransformSequence(Transform):
    _children: Tuple[Transform]

    def __init__(self, transforms: Sequence["Transform"]):
        if not transforms:
            raise ValueError("Cannot construct empty TransformSequence")
        if any(not isinstance(t, Transform) for t in transforms):
            raise ValueError("All children must be Transform instances")
        super().__init__(raw=transforms[0].raw, derived=transforms[-1].derived)

    def __iter__(self):
        return iter(self._children)

    def __len__(self):
        return len(self._children)

    @property
    def is_invertible(self) -> bool:
        return all(t.is_invertible for t in self._children)


@dataclass(frozen=True, slots=True)
class OmeZarrDatasetTransforms(TransformSequence):
    # Has exactly one ScaleTransformation and one TranslationTransformation, in that order.
    def from_coordinate_transformations(self, ct: Dict):
        # TODO: move current implementation?
        # or maybe this belongs in _ome_zarr.py
        raise NotImplementedError()


@dataclass(slots=True)
class _TransformGraph:
    coordinate_systems: Mapping[CoordinateSystemKey, CoordinateSystem]
    transforms: Iterable[Transform]

    def __post_init__(self):
        object.__setattr__(self, "coordinate_systems", MappingProxyType(dict(self.coordinate_systems)))
        object.__setattr__(self, "transforms", tuple(self.transforms))

    def path_between(
        self,
        source: CoordinateSystemKey,
        target: CoordinateSystemKey,
        allow_inverse=True,
        validate_rfc5_connectedness=False,
    ) -> list[Transform]:
        if source == target:
            return [IdentityTransform(source, target)]

        # Adjacency
        graph = defaultdict(list)
        for t in self.transforms:
            graph[t.raw].append((t.derived, t, False))  # (dest, transform, is_inverse)
            if validate_rfc5_connectedness or (allow_inverse and t.is_invertible):
                graph[t.derived].append((t.raw, t, True))

        # BFS
        path = []
        queue = deque([(source, path)])
        visited = {source}
        while queue:
            node, path = queue.popleft()
            if node == target:
                break
            for neighbor, transform, is_inverse in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [transform.inverse() if is_inverse else transform]
                    queue.append((neighbor, new_path))
        return path
