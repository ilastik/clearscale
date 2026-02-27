import enum
import functools
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from typing import (
    Optional,
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

PRE_TRANSFORMS_VERSIONS = ("0.1", "0.2", "0.3", "0.4", "0.5")


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

    def to_ome_zarr(self) -> Dict:
        axis_dict = {}
        if self._ome_zarr_type:
            axis_dict["type"] = self._ome_zarr_type
        if self._ome_zarr_unit:
            axis_dict["unit"] = self._ome_zarr_unit
        if self._ome_zarr_long_name:
            axis_dict["longName"] = self._ome_zarr_long_name
        if self.coordinate_domain == CoordinateDomain.Discrete:
            axis_dict["discrete"] = True
        return axis_dict


class TransformGraphNode(ABC):
    """Mixin for classes that can act as an endpoint for a Transform (i.e. a node in a _TransformGraph)"""

    @abstractmethod
    def as_ref(self, name) -> "CoordinateSystemRef": ...

    @abstractmethod
    def to_ome_zarr(self, name) -> Dict: ...


@dataclass(frozen=True)
class CoordinateSystemRef:
    # Refs are how we deal with the fact that nodes can be of different types (Multiscale, CoordinateSystem),
    # or absent entirely (_UnresolvedRef), and node referencing works via object identity and/or name.
    name: CoordinateSystemName
    owner: Optional[TransformGraphNode]
    """The Multiscale or CoordinateSystem that produced this, for identity. None only in _UnresolvedRef"""

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.name == other.name and self.owner is other.owner

    def to_ome_zarr(self, for_scene: bool, path: Optional[RelativePath] = None) -> Union[CoordinateSystemName, Dict]:
        if not for_scene:
            return self.name
        if path:
            return {"name": self.name, "path": path}
        return {"name": self.name}


@dataclass(frozen=True, slots=True)
class _UnresolvedRef(CoordinateSystemRef):
    """Degenerate placeholder reference.
    Enables round-trip serialization and graph traversal without fully resolved scene metadata."""

    path: Optional[RelativePath] = None
    owner: Optional[TransformGraphNode] = field(default=None, init=False)

    def __post_init__(self):
        if not self.name and not self.path:
            raise ValueError("_UnresolvedRef requires at least one of: name, path")

    def to_ome_zarr(self, for_scene: bool, _=None) -> Union[CoordinateSystemName, Dict]:
        assert for_scene or not self.path, "Unresolved refs with path only allowed inside Scenes"
        if not for_scene:
            return self.name
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

    def to_ome_zarr(self, name: CoordinateSystemName) -> Dict:
        if not name:
            raise ValueError("Cannot store coordinate system without name.")
        axis_dicts = [sem.to_ome_zarr() for sem in self.values()]
        return {"name": name, "axes": axis_dicts}

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
    def inverted(self) -> Optional["Self"]: ...

    @property
    def is_bound(self) -> bool:
        return self.source is not None and self.target is not None

    def bind(self, source: CoordinateSystemRef, target: CoordinateSystemRef) -> "Self":
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
            raise ValueError(f"Invalid transform (in/out must either both be undefined or both defined): {ome_dict!r}")

        t_type = ome_dict.get("type")
        source = endpoints["input"]
        target = endpoints["output"]

        if t_type == "identity":
            return IdentityTransform(source=source, target=target)
        elif t_type == "sequence":
            return TransformSequence(transforms=[cls.from_ome_zarr(td) for td in ome_dict["transformations"]])
        else:
            raise ValueError(f"Unknown transform type: {t_type!r}")

    @abstractmethod
    def to_ome_zarr(self, version: str, *, for_scene: bool, paths_by_node: Optional[PathsByNode] = None) -> Dict: ...

    """Should use _get_ome_zarr_inout for the shared fields and add the transform-specific ones."""

    def _get_ome_zarr_inout(
        self, version: str, for_scene: bool, paths_by_node: Optional[PathsByNode]
    ) -> Dict[Literal["input", "output"], Dict]:
        if version in PRE_TRANSFORMS_VERSIONS:
            return {}
        if for_scene and not self.is_bound:
            raise ValueError("Scene transforms must be bound for OME-Zarr. Use .bind(source, target)")
        paths_by_node = paths_by_node or {}
        input_dict = {}
        output_dict = {}
        for ms, path in paths_by_node.items():
            if self.source.owner is ms:
                input_dict = self.source.to_ome_zarr(for_scene, path)
            if self.target.owner is ms:
                output_dict = self.target.to_ome_zarr(for_scene, path)
            if input_dict and output_dict:
                break
        input_dict = input_dict or self.source.to_ome_zarr(for_scene)
        output_dict = output_dict or self.target.to_ome_zarr(for_scene)
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
    def inverted(self) -> "IdentityTransform":
        return replace(self, source=self.target, target=self.source)

    def to_ome_zarr(self, version: str, *, for_scene: bool, paths_by_node: Optional[PathsByNode] = None) -> Dict:
        return {"type": "identity", **self._get_ome_zarr_inout(version, for_scene, paths_by_node)}


@dataclass(frozen=True, slots=True)
class ScaleTransform(Transform):
    spacing: Spacing
    ome_zarr_path: Optional[str] = None

    @property
    def is_invertible(self) -> bool:
        return all(v for v in self.spacing.values())  # Not invertible with 0 values

    @property
    def inverted(self) -> "ScaleTransform":
        spacing_inverted = Spacing([(a, 1 / v) for a, v in self.spacing.items()])
        return replace(self, spacing=spacing_inverted, ome_zarr_path=None)

    def to_ome_zarr(self, version: str, *, for_scene: bool, paths_by_node: Optional[PathsByNode] = None) -> Dict:
        payload_dict = {"path": self.ome_zarr_path} if self.ome_zarr_path else {"scale": list(self.spacing.values())}
        return {"type": "scale", **payload_dict, **self._get_ome_zarr_inout(version, for_scene, paths_by_node)}


@dataclass(frozen=True, slots=True)
class TranslationTransform(Transform):
    translation: Translation
    ome_zarr_path: Optional[str] = None

    @property
    def is_invertible(self) -> bool:
        return True

    @property
    def inverted(self) -> "TranslationTransform":
        translation_inverted = Translation([(a, -v) for a, v in self.translation.items()])
        return replace(self, translation=translation_inverted, ome_zarr_path=None)

    def to_ome_zarr(self, version: str, *, for_scene: bool, paths_by_node: Optional[PathsByNode] = None) -> Dict:
        payload_dict = (
            {"path": self.ome_zarr_path} if self.ome_zarr_path else {"translation": list(self.translation.values())}
        )
        return {"type": "translation", **payload_dict, **self._get_ome_zarr_inout(version, for_scene, paths_by_node)}


class TransformSequence(Transform):
    _children: Tuple[Transform]

    def __init__(self, transforms: Sequence["Transform"]):
        if not transforms:
            raise ValueError("Cannot make empty TransformSequence.")
        if any(not isinstance(t, Transform) for t in transforms):
            raise ValueError("All children must be Transform instances.")
        if len(transforms) > 1:
            for i, (a, b) in enumerate(zip(transforms, transforms[1:])):
                if a.target is not None and b.source is not None and a.target != b.source:
                    raise ValueError(f"Transform chain broken at position {i}→{i+1}: {a.target!r} != {b.source!r}")
        super().__init__(source=transforms[0].source, target=transforms[-1].target)
        object.__setattr__(self, "_children", tuple(transforms))

    def __hash__(self):
        return hash(self._children)

    def __eq__(self, other):
        return isinstance(other, TransformSequence) and self._children == other._children

    def __iter__(self):
        return iter(self._children)

    def __len__(self):
        return len(self._children)

    @property
    def is_invertible(self) -> bool:
        return all(t.is_invertible for t in self._children)

    def inverted(self) -> "TransformSequence":
        if not self.is_invertible:
            raise ValueError("TransformSequence is not invertible: contains non-invertible transform(s).")
        return TransformSequence(list(reversed([t.inverted() for t in self._children])))

    def to_ome_zarr(self, version: str, *, for_scene: bool, paths_by_node: Optional[PathsByNode] = None) -> Dict:
        return {
            "type": "sequence",
            "transformations": [
                t.to_ome_zarr(version, for_scene=for_scene, paths_by_node=paths_by_node) for t in self._children
            ],
            **self._get_ome_zarr_inout(version, for_scene, paths_by_node),
        }


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
                    visited[neighbor] = (node, transform.inverted() if is_inverse else transform)
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
