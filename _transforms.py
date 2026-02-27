import enum
import functools
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
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
    Literal,
    Any,
    FrozenSet,
    Set,
)

from lazyflow.utility.io_util.clearscale._axis_values import _AxisMapping, AxisKey, OrderedAxes, Spacing, Translation

if TYPE_CHECKING:
    from lazyflow.utility.io_util.clearscale._multiscale import Multiscale

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
NodesByPath = Mapping[RelativePath, "TransformGraphNode"]
PathsByNode = Mapping["TransformGraphNode", RelativePath]


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


class TransformGraphNode(ABC):
    """Mixin for classes that can act as an endpoint for a Transform (i.e. a node in a _TransformGraph)"""

    @abstractmethod
    def as_ref(self, name) -> "CoordinateSystemRef": ...


@dataclass(frozen=True)
class CoordinateSystemRef:
    # Refs are how we deal with the fact that nodes can be of different types (Multiscale, CoordinateSystem),
    # or absent entirely (_UnresolvedRef), and node referencing works via object identity and/or name.
    name: CoordinateSystemName
    owner: Optional[TransformGraphNode]
    """The Multiscale or CoordinateSystem that produced this, for identity. None only in _UnresolvedRef"""

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.name == other.name and self.owner is other.owner

    def to_ome_zarr(self, path: Optional[RelativePath] = None) -> Union[str, Dict[Literal["name", "path"], str]]:
        if path is None:
            return self.name
        return {"name": self.name, "path": path}


@dataclass(frozen=True, slots=True)
class _UnresolvedRef(CoordinateSystemRef):
    """Degenerate placeholder reference.
    Enables round-trip serialization and graph traversal without fully resolved scene metadata."""

    path: Optional[RelativePath] = None
    owner: Optional[TransformGraphNode] = field(default=None, init=False)

    def __post_init__(self):
        if not self.name and not self.path:
            raise ValueError("_UnresolvedRef requires at least one of: name, path")

    def to_ome_zarr(self, _=None) -> Dict[Literal["name", "path"], str]:
        d = {}
        if self.path:
            d["path"] = self.path
        if self.name:
            d["name"] = self.name
        return d


class CoordinateSystem(_AxisMapping[AxisKey, AxisSemantics], TransformGraphNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __hash__(self):
        """(See __eq__)"""
        return id(self)

    def __eq__(self, other):
        """Identity-based equality and hash.
        Even content-identical coordinate systems are not necessarily the same system.
        For example, most JPEGs have content-identical coordinate systems (x, y, color), but there is no
        relationship between the coordinate systems of two different JPEG scans of paper."""
        return self is other  # even content-identical coordinate systems may not be the same system

    def as_ref(self, name: CoordinateSystemName):
        return CoordinateSystemRef(name, self)

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
class Transform(ABC):
    source: Optional[CoordinateSystemRef] = field(
        default=None, kw_only=True
    )  # default required; optional when nested in sequence or bijection (and will be None by default in clearscale when nested)
    target: Optional[CoordinateSystemRef] = field(default=None, kw_only=True)
    source_axes: Optional[OrderedAxes] = field(
        default=None, kw_only=True
    )  # default None; required when nested in byDimension
    target_axes: Optional[OrderedAxes] = field(default=None, kw_only=True)
    payload: Union[RelativePath, Any] = field(default=None, kw_only=True)

    @property
    @abstractmethod
    def is_invertible(self) -> bool: ...
    @property
    @abstractmethod
    def inverse(self) -> Optional["Self"]: ...

    @property
    def is_bound(self) -> bool:
        return self.source is not None and self.target is not None

    def bind(self, source: Union[CoordinateSystemRef], target: Union[CoordinateSystemRef]) -> "Self":
        # binding required to use the Transform in a TransformGraph
        return replace(self, source=source, target=target)

    def unbind(self) -> "Self":
        # for nesting inside a TransformSequence or BijectionTransform
        return replace(self, source=None, target=None)

    @property
    def has_unresolved_endpoint(self) -> bool:
        return isinstance(self.source, _UnresolvedRef) or isinstance(self.target, _UnresolvedRef)

    @classmethod
    def from_ome_zarr(cls, ome_dict: Dict) -> "Transform":
        endpoints = {"input": None, "output": None}
        for side in endpoints.keys():
            ref = ome_dict.get(side, {})
            path = ref.get("path")
            name = ref.get("name")
            if path or name:
                endpoints[side] = _UnresolvedRef(path=path, name=name)
        if bool(endpoints["input"]) != bool(endpoints["output"]):
            raise ValueError(f"Invalid transform (in/out must either both be undefined or both defined): {ome_dict}")

        t_type = ome_dict.get("type")
        source = endpoints["input"]
        target = endpoints["output"]

        if t_type == "identity":
            return IdentityTransform(source=source, target=target)
        elif t_type == "sequence":
            return TransformSequence(transforms=[cls.from_ome_zarr(td) for td in ome_dict["transformations"]])
        else:
            raise ValueError(f"Unknown transform type: {t_type}")

    def _get_ome_zarr_inout(
        self, paths_by_node: Optional[PathsByNode] = None
    ) -> Dict[Literal["input", "output"], Dict]:
        # TODO: Use in subclass.to_ome_zarr
        # Generate the "input" and "output" of the ome-zarr transform - other fields added per-subclass
        paths_by_node = paths_by_node or {}
        input_dict = {}
        output_dict = {}
        for ms, path in paths_by_node.items():
            if self.source.owner is ms:
                input_dict = self.source.to_ome_zarr(path)
            if self.target.owner is ms:
                output_dict = self.target.to_ome_zarr(path)
        input_dict = input_dict or self.source.to_ome_zarr()
        output_dict = output_dict or self.target.to_ome_zarr()
        return {"input": input_dict, "output": output_dict}

    def with_resolved(
        self, path_nodes: Optional[NodesByPath], *, named_refs: Optional[Set[CoordinateSystemRef]]
    ) -> "Self":
        """
        Resolving name-only references takes a lot of care: coordinate system names are not unique,
        and there is no robust way to determine if a system with the expected name is actually the specific
        system the transform referenced.
        Using this (or wrapping its usage) with named_refs should be avoided, or force explicit intention.
        """
        if not self.has_unresolved_endpoint or (not path_nodes and not named_refs):
            return self
        path_nodes = path_nodes or {}
        named_refs = named_refs or {}
        new_source = self._resolve_ref(self.source, path_nodes, named_refs)
        new_target = self._resolve_ref(self.target, path_nodes, named_refs)
        if new_source is self.source and new_target is self.target:
            return self
        return replace(self, source=new_source, target=new_target)

    @staticmethod
    def _resolve_ref(
        ref: CoordinateSystemRef, path_nodes: NodesByPath, named_refs: Set[CoordinateSystemRef]
    ) -> CoordinateSystemRef:
        if not isinstance(ref, _UnresolvedRef):
            return ref
        if ref.path:
            new_node = path_nodes.get(ref.path)
            if new_node is not None:
                return new_node.as_ref(ref.name)
        if ref.name:
            name_matches = [other for other in named_refs if other.name == ref.name]
            if len(name_matches) > 1:
                raise ValueError(
                    f"Cannot resolve transform: Received multiple coordinate systems named '{ref.name}': "
                    ", ".join([r.name for r in named_refs])
                )
            elif name_matches:
                return name_matches[0]
        return ref

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
    @property
    def is_invertible(self) -> bool:
        return True

    @property
    def inverse(self) -> "IdentityTransform":
        return replace(self, source=self.target, target=self.source)


@dataclass(frozen=True, slots=True)
class ScaleTransform(Transform):
    spacing: Spacing
    ome_zarr_path: Optional[str] = (
        None  # ome-zarr rfc-5: transform["path"] (e.g. for displacement); required for e.g. displacement
    )

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
class _TransformGraph:
    # Graph was originally supposed to primarily just be Mapping[CoordinateSystemName, CoordinateSystem]
    # with Graph._transforms acting as a hidden store for .path_between
    # But this doesn't work out: coordinate systems won't be uniquely named across scenes. Multiple multiscales
    # will likely have systems named "physical". So (Multiscale, CoordinateSystemName) as a combined key?
    # This isn't sufficient: Transform source/target need to be lazy because Scene cannot know all
    # Multiscales when first loading OME-Zarr scene metadata.
    # So there needs to be an "unresolved multiscale" node key.
    # Plus, both multiscales and scenes are allowed to define coordinate systems that act as source/target for
    # transforms defined elsewhere. In this case, we only know a system name, and the fact that it doesn't appear
    # in any transform the current multiscale/scene is parsing.
    # So there needs to be a name-only "unresolved system" node key.
    transforms: Iterable[Transform]
    isolated_system_refs: Optional[FrozenSet[CoordinateSystemRef]] = None
    # isolated_system_refs should only be used to deal with ome-zarr multiscales or scenes that define
    # coordinate systems without defining any transforms that reference them.
    # _TransformGraph effectively contains one true graph defined by transform edges;
    # which may be disjunct and may contain unresolved nodes; plus a set of known-unconnected coordinate system nodes.

    @functools.cached_property
    def coordinate_system_refs(self) -> FrozenSet[CoordinateSystemRef]:
        return frozenset(ref for ref in self.node_refs if isinstance(ref.owner, CoordinateSystem))

    @functools.cached_property
    def multiscales(self) -> FrozenSet["Multiscale"]:
        from ._multiscale import Multiscale  # kinda unneeded (can't make a graph before importing Multiscale)

        return frozenset(ref.owner for ref in self.node_refs if isinstance(ref.owner, Multiscale))

    @functools.cached_property
    def node_refs(self) -> FrozenSet[CoordinateSystemRef]:
        refs = set()
        for t in self.transforms:
            refs.add(t.source)
            refs.add(t.target)
        return frozenset(refs)

    def __post_init__(self):
        bad = [t for t in self.transforms if t.source is None or t.target is None]
        if bad:
            raise ValueError(f"Graph transforms must have bound endpoints: {bad}")
        object.__setattr__(self, "transforms", tuple(self.transforms))

    def path_between(
        self,
        source: CoordinateSystemRef,
        target: CoordinateSystemRef,
        allow_inverse=True,
        validate_rfc5_connectedness=False,
    ) -> Optional[list[Transform]]:
        if source == target:
            return [IdentityTransform(source=source, target=target)]

        # Adjacency - could be worth caching for performance
        graph = defaultdict(list)
        for t in self.transforms:
            graph[t.source].append((t.target, t, False))  # (dest, transform, is_inverse)
            if validate_rfc5_connectedness or (allow_inverse and t.is_invertible):
                graph[t.target].append((t.source, t, True))

        # BFS tracking (predecessor, transform) instead of copying paths
        visited = {source: None}  # node -> (predecessor, transform)
        queue = deque([source])
        while queue:
            node = queue.popleft()
            if node == target:
                break
            for neighbor, transform, is_inverse in graph[node]:
                if neighbor not in visited:
                    visited[neighbor] = (node, transform.inverse() if is_inverse else transform)
                    queue.append(neighbor)

        # Reconstruct path
        path = []
        node = target
        while visited[node] is not None:
            predecessor, transform = visited[node]
            path.append(transform)
            node = predecessor
        path.reverse()
        return path
