import enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import (
    Optional,
    List,
    Tuple,
    Dict,
    Mapping,
    Iterable,
    Sequence,
    TYPE_CHECKING,
    TypeVar,
    Union,
    Generic,
    Literal,
    Any,
    Collection,
)

from lazyflow.utility.io_util.clearscale._axis_values import _AxisMapping, AxisKey, OrderedAxes, Spacing, Translation

if TYPE_CHECKING:
    from lazyflow.utility.io_util.clearscale._multiscale import Multiscale
    from lazyflow.utility.io_util.clearscale._scene import MultiscalesByPath, CoordinateSystemEndpoint

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
SystemKey = TypeVar("SystemKey")


class ValidationError(ValueError):
    pass


class CoordinateDomain(enum.Enum):
    Categorical = enum.auto()
    Discrete = enum.auto()
    Continuous = enum.auto()


class ValueDomain(enum.Enum):
    # Not sure this has any value - does it map directly to concepts from image processing packages?
    Real = enum.auto()
    OrderedInteger = enum.auto()
    SequentialInteger = enum.auto()
    UnorderedInteger = enum.auto()


@dataclass(frozen=True, slots=True)
class AxisSemantics:
    coordinate_domain: Optional[CoordinateDomain] = None
    value_domain: Optional[ValueDomain] = None
    _ome_zarr_type: Optional[str] = None
    _ome_zarr_unit: Optional[str] = None
    _ome_zarr_long_name: Optional[str] = None


class CoordinateSystem(_AxisMapping[AxisKey, AxisSemantics]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_ome_zarr(cls, system_dict):
        semantics_by_axis = []
        for axis_dict in system_dict["axes"]:
            coordinates = CoordinateDomain.Discrete if axis_dict.get("discrete") == "true" else None
            oz_type = axis_dict.get("type")
            values = ValueDomain.UnorderedInteger if oz_type == "channel" else None
            semantics_by_axis.append(
                (
                    axis_dict["name"],
                    AxisSemantics(
                        coordinate_domain=coordinates,
                        value_domain=values,
                        _ome_zarr_type=oz_type,
                        _ome_zarr_unit=axis_dict.get("unit"),
                        _ome_zarr_long_name=axis_dict.get("longName"),
                    ),
                )
            )
        return cls(semantics_by_axis)

    @classmethod
    def without_semantics(cls, axes: OrderedAxes) -> "CoordinateSystem":
        return cls([(a, AxisSemantics()) for a in axes])


@dataclass(frozen=True, slots=True)
class _UnresolvedCoordinateSystemReference:
    name: Optional[CoordinateSystemName]
    path: Optional[RelativePath]


PathsByMultiscale = Mapping["Multiscale", RelativePath]


@dataclass(frozen=True, slots=True)
class Transform(ABC, Generic[SystemKey]):
    source: Optional[SystemKey] = field(
        default=None, kw_only=True
    )  # default required; optional when nested in sequence or bijection (and will be None by default in clearscale when nested)
    target: Optional[SystemKey] = field(default=None, kw_only=True)
    source_axes: Optional[OrderedAxes] = field(
        default=None, kw_only=True
    )  # default None; required when nested in byDimension
    target_axes: Optional[OrderedAxes] = field(default=None, kw_only=True)
    payload: Union[RelativePath, Any] = field(default=None, kw_only=True)

    @property
    @abstractmethod
    def is_invertible(self) -> bool: ...
    def bind(self, source: "Multiscale", target: "Multiscale") -> "Self":
        # binding required to use the Transform in a TransformGraph
        return replace(self, source=source, target=target)

    def unbind(self) -> "Self":
        # for nesting inside a TransformSequence or BijectionTransform
        return replace(self, source=None, target=None)

    @property
    def is_bound(self) -> bool:
        return self.source is not None and self.target is not None

    @property
    def has_unresolved_endpoint(self) -> bool:
        return isinstance(self.source, _UnresolvedCoordinateSystemReference) or isinstance(
            self.target, _UnresolvedCoordinateSystemReference
        )

    @classmethod
    def from_ome_zarr(cls, ome_dict: Dict) -> "Transform[CoordinateSystemEndpoint]":
        endpoints = {"input": None, "output": None}
        for side in endpoints.keys():
            ref = ome_dict.get(side, {})
            path = ref.get("path")
            name = ref.get("name")
            if path is not None:
                endpoints[side] = _UnresolvedCoordinateSystemReference(name, path)
            elif name is not None:
                endpoints[side] = name

        t_type = ome_dict.get("type")
        # OME-Zarr uses coordinate semantics for in-out, i.e.:
        # the transform says how to get the output coordinates of a point given its input coordinates.
        # Image processing tools usually work exactly the other way round: Their transforms say
        # what coordinates in the input to look for given some output coordinates they want to resample.
        source = endpoints["input"]
        target = endpoints["output"]

        if t_type == "identity":
            return IdentityTransform(source=source, target=target)
        elif t_type == "scale":
            return ScaleTransform(source=source, target=target, payload=ome_dict["scale"])
        elif t_type == "translation":
            return TranslationTransform(source=source, target=target, payload=ome_dict["translation"])
        elif t_type == "sequence":
            return TransformSequence(transforms=[cls.from_ome_zarr(td) for td in ome_dict["transformations"]])
        else:
            raise ValueError(f"Unknown transform type: {t_type}")

    def to_ome_zarr(self, paths_by_multiscale: PathsByMultiscale) -> Dict[Literal["input", "output"], Dict]:
        if isinstance(self.source, CoordinateSystemName):
            input_name = self.source
            input_path = None
        elif isinstance(self.source, Multiscale):
            input_name = self.source.aligned_system
            input_path = paths_by_multiscale.get(self.source)
        else:
            raise ValueError("wat? using a wrong type for transform endpoint")
        # TODO: actually make dict (this method should be an internal helper and to_ome_zarr is on each subclass)
        raise NotImplementedError()

    def resolved_with(self, multiscales: MultiscalesByPath) -> Tuple["Transform", Dict[Multiscale, RelativePath]]:
        if not self.has_unresolved_endpoint:
            return self, {}
        used_multiscales: Dict[Multiscale, RelativePath] = {}
        new_source = self.source
        if isinstance(self.source, _UnresolvedCoordinateSystemReference):
            maybe_match = self._extract_multiscale_for_ref(self.source, multiscales)
            if maybe_match is not None:
                new_source = maybe_match[0]
                used_multiscales[new_source] = maybe_match[1]
        new_target = self.target
        if isinstance(self.target, _UnresolvedCoordinateSystemReference):
            maybe_match = self._extract_multiscale_for_ref(self.target, multiscales)
            if maybe_match is not None:
                new_target = maybe_match[0]
                used_multiscales[new_target] = maybe_match[1]
        return replace(self, source=new_source, target=new_target), used_multiscales

    @staticmethod
    def _extract_multiscale_for_ref(
        ref: _UnresolvedCoordinateSystemReference, multiscales: MultiscalesByPath
    ) -> Optional[Tuple[Multiscale, RelativePath]]:
        if ref.path not in multiscales:
            return None
        try:
            # TODO: settle method name and implement (whereby calling with name == self.aligned_system returns self?)
            return multiscales[ref.path].aligned_with(ref.name)
        except ValueError as e:
            raise ValueError(
                f"Multiscale provided for path '{ref.path}' does not have a coordinate system '{ref.name}' as "
                f"expected for transform."
            ) from e

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
        super().__init__(source=transforms[0].source, target=transforms[-1].target)

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


@dataclass(frozen=True, slots=True)
class _TransformGraph(Generic[SystemKey]):
    coordinate_systems: Mapping[SystemKey, CoordinateSystem]
    transforms: Iterable[Transform[SystemKey]]

    def __post_init__(self):
        if any(not key for key in self.coordinate_systems):
            raise ValueError(
                f"All coordinate systems must have a name or reference. Received: {list(self.coordinate_systems.keys())}"
            )
        object.__setattr__(self, "coordinate_systems", MappingProxyType(dict(self.coordinate_systems)))
        object.__setattr__(self, "transforms", tuple(self.transforms))

    def path_between(
        self,
        source: SystemKey,
        target: SystemKey,
        allow_inverse=True,
        validate_rfc5_connectedness=False,
    ) -> list[Transform]:
        if source == target:
            return [IdentityTransform(source=source, target=target)]

        # Adjacency
        graph = defaultdict(list)
        for t in self.transforms:
            graph[t.source].append((t.target, t, False))  # (dest, transform, is_inverse)
            if validate_rfc5_connectedness or (allow_inverse and t.is_invertible):
                graph[t.target].append((t.source, t, True))

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


_NamingTransformGraph = _TransformGraph[CoordinateSystemName]
_ReferencingTransformGraph = _TransformGraph[CoordinateSystemEndpoint]
