import enum
import functools
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace, fields
from typing import (
    Optional,
    Tuple,
    Dict,
    Mapping,
    Iterable,
    TYPE_CHECKING,
    TypeVar,
    Union,
    Literal,
    Any,
    FrozenSet,
    Set,
    List,
    Generic,
)

from lazyflow.utility.io_util.clearscale._axis_values import (
    _AxisMapping,
    AxisKey,
    OrderedAxes,
    Spacing,
    Translation,
    Unit,
)

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
NodesByPath = Mapping[RelativePath, "TransformGraphNode"]
PathsByNode = Mapping["TransformGraphNode", RelativePath]
AnyTransformGraphNode = TypeVar("AnyTransformGraphNode", bound="TransformGraphNode")

PRE_TRANSFORMS_VERSIONS = ("0.1", "0.2", "0.3", "0.4", "0.5")


class CoordinateContinuity(enum.StrEnum):
    Categorical = enum.auto()
    Discrete = enum.auto()
    Continuous = enum.auto()


@dataclass(frozen=True, slots=True)
class AxisSemantics:
    coordinate_domain: Optional[CoordinateContinuity] = None
    _ome_zarr_type: Optional[str] = None
    _ome_zarr_unit: Optional[str] = None
    _ome_zarr_long_name: Optional[str] = None

    @classmethod
    def from_ome_zarr(cls, axis_dict: Dict) -> "AxisSemantics":
        coordinates = CoordinateContinuity.Discrete if axis_dict.get("discrete") else None
        return cls(
            coordinate_domain=coordinates,
            _ome_zarr_type=axis_dict.get("type"),
            _ome_zarr_unit=axis_dict.get("unit"),
            _ome_zarr_long_name=axis_dict.get("longName"),
        )

    def __repr__(self):
        items = (f"{f.name}={getattr(self, f.name)!r}" for f in fields(self) if getattr(self, f.name) is not None)
        return f"{self.__class__.__name__}({', '.join(items)})"

    def to_ome_zarr(self, *, name: str) -> Dict:
        axis_dict = {"name": name}
        if self._ome_zarr_type:
            axis_dict["type"] = self._ome_zarr_type
        if self._ome_zarr_unit:
            axis_dict["unit"] = self._ome_zarr_unit
        if self._ome_zarr_long_name:
            axis_dict["longName"] = self._ome_zarr_long_name
        if self.coordinate_domain == CoordinateContinuity.Discrete:
            axis_dict["discrete"] = True
        return axis_dict


class TransformGraphNode(ABC):
    """Mixin for classes that can act as an endpoint for a Transform (i.e. a node in a _TransformGraph)"""

    @abstractmethod
    def axes(self) -> Iterable[AxisKey]: ...

    @abstractmethod
    def as_ref(self, name: CoordinateSystemName) -> "CoordinateSystemRef": ...

    @abstractmethod
    def to_ome_zarr(self, *, name: CoordinateSystemName, version: str) -> Dict: ...


@dataclass(frozen=True)
class CoordinateSystemRef(Generic[AnyTransformGraphNode]):
    # Refs are how we deal with the fact that nodes can be of different types (Multiscale, CoordinateSystem),
    # or absent entirely (_UnresolvedRef), and node referencing works via object identity and/or name.
    name: CoordinateSystemName
    owner: Optional[AnyTransformGraphNode]
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

    def axes(self) -> Iterable[AxisKey]:
        return self.keys()

    def as_ref(self, name: CoordinateSystemName):
        return CoordinateSystemRef(name, self)

    @classmethod
    def without_semantics(cls, axes: OrderedAxes) -> "CoordinateSystem":
        return cls([(a, AxisSemantics()) for a in axes])

    @classmethod
    def from_ome_zarr(cls, system_or_multiscale_dict: Dict):
        axis_dicts = system_or_multiscale_dict.get("axes")
        if not axis_dicts:
            # v0.1 and v0.2 did not have any axis metadata
            return cls.without_semantics(["t", "c", "z", "y", "x"])
        if not isinstance(axis_dicts, list):
            raise ValueError(f"Invalid axis metadata. Received: {system_or_multiscale_dict}")
        if isinstance(axis_dicts[0], str):
            # v0.3 allowed specifying a subset of tczyx, e.g. ["t", "c", "y", "x"]
            return cls.without_semantics(axis_dicts)
        semantics_by_axis = []
        for axis_dict in system_or_multiscale_dict["axes"]:
            if not axis_dict.get("name"):
                raise ValueError(f"Invalid axis metadata: Missing axis name. Received: {system_or_multiscale_dict}")
            semantics_by_axis.append((axis_dict["name"], AxisSemantics.from_ome_zarr(axis_dict)))
        return cls(semantics_by_axis)

    def to_ome_zarr(
        self,
        *,
        name: CoordinateSystemName,
        version="rfc-5",
        axis_types: Union[None, Literal["infer"], Mapping[str, Literal["space", "time", "channel"]]] = None,
        unit: Unit = None,
        long_names: Mapping[AxisKey, str] = None,
        discrete: Mapping[AxisKey, bool] = None,
    ) -> Dict:
        if not name and version not in PRE_TRANSFORMS_VERSIONS:
            raise ValueError(f"Cannot store coordinate system without name in OME-Zarr version {version}.")
        unit = unit or {}
        long_names = long_names or {}
        discrete = discrete or {}
        if not axis_types:
            axis_types = {}
        if axis_types == "infer":
            axis_types = {
                "t": "time",
                "time": "time",
                "timestep": "time",
                "timepoint": "time",
                "c": "channel",
                "ch": "channel",
                "channel": "channel",
                "channels": "channel",
                "z": "space",
                "y": "space",
                "x": "space",
            }
        elif not any(ax in self.axes for ax in axis_types):
            warnings.warn(f"Unexpected axis types provided: Did not find any axis of: {list(axis_types.keys())}")
        axis_dicts = []
        for ax, sem in self.items():
            adict = sem.to_ome_zarr(name=ax)
            if ax in unit and unit[ax]:
                adict["unit"] = unit[ax]
            if ax in axis_types and axis_types[ax]:
                adict["type"] = axis_types[ax]
            if ax in long_names and long_names[ax]:
                adict["longName"] = long_names[ax]
            if ax in discrete and discrete[ax]:
                adict["discrete"] = discrete[ax]
            axis_dicts.append(adict)
        d = {"axes": axis_dicts}
        if name:
            d["name"] = name
        return d

    def get_unit(self) -> Unit:
        return Unit([(a, sem._ome_zarr_unit or "") for a, sem in self.items()])  # noqa


@dataclass(frozen=True, slots=True)
class Transform(ABC):
    """
    Coordinate transformation with OME-Zarr convention for source/target coordinates:
    `source_coords x t = target_coords`
    This convention prioritises *technical simplicity*, not mathematical theory.
    Transforming array indices or slicings to meaningful physical coordinates is simple:
    `[0, 124, 124] x Scale(1, 0.2, 0.2) = [0, 24.8, 24.8]`
    """

    source: Optional[CoordinateSystemRef] = field(default=None, kw_only=True)
    """The transform graph node (coordinate system) whose coordinates this transform acts on"""
    target: Optional[CoordinateSystemRef] = field(default=None, kw_only=True)
    """The transform graph node (coordinate system) whose coordinates this transform produces"""

    @property
    @abstractmethod
    def is_invertible(self) -> bool: ...
    @property
    @abstractmethod
    def inverted(self) -> Optional["Self"]: ...

    @property
    def is_fully_bound(self) -> bool:
        return self.source is not None and self.target is not None

    @property
    def is_fully_unbound(self) -> bool:
        return self.source is None and self.target is None

    def bound(self, source: Optional[CoordinateSystemRef], target: Optional[CoordinateSystemRef]) -> "Self":
        # binding required to use the Transform in a TransformGraph
        return replace(self, source=source, target=target)

    def unbound(self) -> "Self":
        # for nesting inside a TransformSequence or BijectionTransform
        return replace(self, source=None, target=None)

    @property
    def has_unresolved_endpoint(self) -> bool:
        return self.source is None or self.target is None or self.source.owner is None or self.target.owner is None

    @property
    def is_fully_unresolved(self) -> bool:
        return (self.source is None or self.source.owner is None) and (self.target is None or self.target.owner is None)

    @classmethod
    def from_ome_zarr(cls, ome_dict: Dict) -> "Transform":
        t_type = ome_dict.get("type")
        if t_type == "identity":
            source, target = cls._parse_source_and_target(ome_dict)
            return IdentityTransform(source=source, target=target)
        elif t_type == "scale":
            return ScaleTransform.from_ome_zarr(ome_dict)
        elif t_type == "translation":
            return TranslationTransform.from_ome_zarr(ome_dict)
        elif t_type == "sequence":
            return TransformSequence(
                transforms=tuple(Transform.from_ome_zarr(td) for td in ome_dict["transformations"])
            )
        else:
            raise ValueError(f"Unknown transform type: {t_type!r}")

    @staticmethod
    def _parse_source_and_target(ome_dict: Dict):
        endpoints = {"input": None, "output": None}
        for side in endpoints.keys():
            ref = ome_dict.get(side, {})
            path = ref.get("path")
            name = ref.get("name")
            if path or name:
                endpoints[side] = _UnresolvedRef(path=path, name=name)
        if bool(endpoints["input"]) != bool(endpoints["output"]):
            raise ValueError(f"Invalid transform (in/out must either both be undefined or both defined): {ome_dict!r}")
        source = endpoints["input"]
        target = endpoints["output"]
        return source, target

    @abstractmethod
    def to_ome_zarr(self, version: str, *, for_scene: bool, paths_by_node: Optional[PathsByNode] = None) -> Dict:
        """Should use _get_ome_zarr_inout for the shared fields and add the transform-specific ones."""
        pass

    def _get_ome_zarr_inout(
        self, version: str, for_scene: bool, paths_by_node: Optional[PathsByNode]
    ) -> Dict[Literal["input", "output"], Dict]:
        if version in PRE_TRANSFORMS_VERSIONS:
            return {}
        if for_scene and not self.is_fully_bound:
            raise ValueError("OME-Zarr Scene transforms must be `.bound(source, target)`")
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

    @abstractmethod
    def composed_with(self, earlier: "Transform") -> Optional["Transform"]: ...

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

    def composed_with(self, earlier: "Transform") -> Optional["Transform"]:
        if earlier.target is not None and self.source is not None and earlier.target != self.source:
            return None
        return replace(earlier, target=self.target)


@dataclass(frozen=True, slots=True)
class ScaleTransform(Transform):
    _scale: Tuple[float, ...]
    ome_zarr_path: Optional[str] = None

    @classmethod
    def from_spacing(cls, spacing: Spacing):
        return cls(_scale=tuple(spacing.values()))

    @classmethod
    def from_ome_zarr(cls, ome_dict: Dict) -> "ScaleTransform":
        source, target = cls._parse_source_and_target(ome_dict)
        return cls(
            _scale=tuple(ome_dict.get("scale") or []),
            ome_zarr_path=ome_dict.get("path"),
            source=source,
            target=target,
        )

    def to_spacing(self, axes: Optional[Iterable[AxisKey]] = None) -> Spacing:
        if not self._scale:
            raise ValueError("Cannot derive Spacing: Values not set.")
        final_axes = axes or self._axes()
        return Spacing(zip(final_axes, self._scale))

    def _axes(self) -> Iterable[AxisKey]:
        """Must be kept in sync with TranslationTransform._axes"""
        # TODO: Move to base class?
        if self.is_fully_unbound:
            raise ValueError("Missing axes: Bind to coordinate systems or multiscales first to define.")
        if self.is_fully_unresolved:
            raise ValueError(
                "Missing axes: Resolve at least one multiscale first to define. "
                f"Source: {self.source}, Target: {self.target}"
            )
        return self.source.owner.axes() if self.source else self.target.owner.axes()

    @property
    def is_invertible(self) -> bool:
        return all(v for v in self._scale)  # Not invertible with 0 values

    def inverted(self) -> "ScaleTransform":
        scale_inverted = tuple(1 / v for v in self._scale)
        return replace(self, _scale=scale_inverted, ome_zarr_path=None)

    def to_ome_zarr(self, version: str, *, for_scene: bool, paths_by_node: Optional[PathsByNode] = None) -> Dict:
        payload_dict = {"path": self.ome_zarr_path} if self.ome_zarr_path else {"scale": list(self._scale)}
        return {"type": "scale", **payload_dict, **self._get_ome_zarr_inout(version, for_scene, paths_by_node)}

    def composed_with(self, earlier: "Transform") -> Optional["Transform"]:
        if not isinstance(earlier, ScaleTransform):
            return None
        return replace(self, _scale=tuple(a * b for a, b in zip(self._scale, earlier._scale)), ome_zarr_path=None)


@dataclass(frozen=True, slots=True)
class TranslationTransform(Transform):
    _translation: Tuple[float, ...]
    ome_zarr_path: Optional[str] = None

    @classmethod
    def from_translation(cls, translation: Translation):
        return cls(_translation=tuple(translation.values()))

    @classmethod
    def from_ome_zarr(cls, ome_dict: Dict) -> "TranslationTransform":
        source, target = cls._parse_source_and_target(ome_dict)
        return cls(
            _translation=tuple(ome_dict.get("translation") or []),
            ome_zarr_path=ome_dict.get("path"),
            source=source,
            target=target,
        )

    def to_translation(self, axes: Optional[Iterable[AxisKey]] = None) -> Translation:
        if not self._translation:
            raise ValueError("Cannot derive Translation: Values not set")
        final_axes = axes or self._axes()
        return Translation(zip(final_axes, self._translation))

    def _axes(self):
        """Must be kept in sync with ScaleTransform._axes"""
        if self.is_fully_unbound:
            raise ValueError("Missing axes: Bind to coordinate systems or multiscales first to define.")
        if self.is_fully_unresolved:
            raise ValueError(
                "Missing axes: Resolve at least one multiscale first to define. "
                f"Source: {self.source}, Target: {self.target}"
            )
        return self.source.owner.axes() if self.source.owner else self.target.owner.axes()

    @property
    def is_invertible(self) -> bool:
        return True

    def inverted(self) -> "TranslationTransform":
        translation_inverted = tuple(-v for v in self._translation)
        return replace(self, _translation=translation_inverted, ome_zarr_path=None)

    def to_ome_zarr(self, version: str, *, for_scene: bool, paths_by_node: Optional[PathsByNode] = None) -> Dict:
        payload_dict = {"path": self.ome_zarr_path} if self.ome_zarr_path else {"translation": list(self._translation)}
        return {"type": "translation", **payload_dict, **self._get_ome_zarr_inout(version, for_scene, paths_by_node)}

    def composed_with(self, earlier: "Transform") -> Optional["Transform"]:
        if not isinstance(earlier, TranslationTransform):
            return None
        return replace(
            self, _translation=tuple(a + b for a, b in zip(self._translation, earlier._translation)), ome_zarr_path=None
        )


@dataclass(frozen=True, slots=True)
class TransformSequence(Transform):
    transforms: Tuple[Transform, ...] = field(default=())

    def __post_init__(self):
        if not self.transforms:
            raise ValueError("Cannot make empty TransformSequence.")
        if any(not isinstance(t, Transform) for t in self.transforms):
            raise ValueError("All children must be Transform instances.")
        for i, (a, b) in enumerate(zip(self.transforms, self.transforms[1:])):
            if a.target is not None and b.source is not None and a.target != b.source:
                raise ValueError(f"Transform chain broken at {i}→{i+1}: {a.target!r} != {b.source!r}")
        # Infer source/target from children if not explicitly provided
        inferred_source = self.transforms[0].source
        inferred_target = self.transforms[-1].target
        if self.source is None and inferred_source is not None:
            object.__setattr__(self, "source", inferred_source)
        if self.target is None and inferred_target is not None:
            object.__setattr__(self, "target", inferred_target)

    def __hash__(self):
        return hash(self.transforms)

    def __eq__(self, other):
        return isinstance(other, TransformSequence) and self.transforms == other.transforms

    def __iter__(self):
        return iter(self.transforms)

    def __len__(self):
        return len(self.transforms)

    def __getitem__(self, item):
        return self.transforms[item]

    def bound(self, source: Optional[CoordinateSystemRef], target: Optional[CoordinateSystemRef]) -> "Self":
        # Override from base: Sequence needs to update endpoint transforms
        if len(self.transforms) == 1:
            first = self.transforms[0].bound(source=source, target=target)
            new_transforms = (first,)
        else:
            first = self.transforms[0].bound(source=source, target=self.transforms[0].target)
            last = self.transforms[-1].bound(source=self.transforms[-1].source, target=target)
            new_transforms = (first,) + self.transforms[1:-1] + (last,)
        return replace(self, source=source, target=target, transforms=new_transforms)

    @property
    def is_invertible(self) -> bool:
        return all(t.is_invertible for t in self.transforms)

    def inverted(self) -> "TransformSequence":
        if not self.is_invertible:
            raise ValueError("TransformSequence is not invertible: contains non-invertible transform(s).")
        return TransformSequence(tuple(reversed([t.inverted() for t in self.transforms])))

    def to_ome_zarr(
        self, version: str, *, for_scene: bool, paths_by_node: Optional[PathsByNode] = None
    ) -> Union[Dict, List]:
        if version in PRE_TRANSFORMS_VERSIONS:
            return [t.to_ome_zarr(version, for_scene=for_scene, paths_by_node=paths_by_node) for t in self.transforms]
        else:
            return {
                "type": "sequence",
                "transformations": [
                    t.to_ome_zarr(version, for_scene=for_scene, paths_by_node=paths_by_node) for t in self.transforms
                ],
                **self._get_ome_zarr_inout(version, for_scene, paths_by_node),
            }

    def composed_with(self, earlier: "Transform") -> Optional["Transform"]:
        # TODO: compatibility check if self or earlier is bound, and/or axis compatibility
        # See if maybe that check ends up working out identical across Transform subclasses
        # and move to the base class if so
        if isinstance(earlier, TransformSequence):
            return replace(earlier, transforms=tuple(earlier.transforms + self.transforms))
        return replace(self, transforms=(earlier,) + self.transforms)

    def collapsed(self, *, raise_uncollapsed: bool = False) -> "Transform | TransformSequence":
        result: list[Transform] = [self.transforms[0]]

        for current in self.transforms[1:]:
            previous = result[-1]
            merged = current.composed_with(previous)
            if merged is not None:
                result[-1] = merged
            elif raise_uncollapsed:
                raise ValueError(f"Cannot collapse {type(previous).__name__} followed by {type(current).__name__}")
            else:
                result.append(current)

        if len(result) == 1:
            return result[0]
        return replace(self, transforms=tuple(result))


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
    transforms: Iterable[Transform]  # This could be ~15k entries in prod
    isolated_system_refs: Optional[FrozenSet[CoordinateSystemRef]] = None
    unresolved_transforms: Optional[Iterable[Transform]] = None
    # isolated_system_refs and unresolved_transforms are used to deal with ome-zarr multiscales
    # or scenes that define coordinate systems without defining any transforms that reference them;
    # and respectively with transforms whose source/target metadata have not been loaded into memory yet.
    # `unresolved_transforms` should be a SUBSET of `transforms`. It is implemented as a parameter
    # rather than a cached_property only because it is more efficient for Scene.from_ome_zarr to build it
    # as it iterates the metadata.

    def __bool__(self):
        return bool(self.transforms) or bool(self.isolated_system_refs)

    @functools.cached_property
    def all_system_refs(self) -> FrozenSet[CoordinateSystemRef]:
        return self.connected_system_refs | self.isolated_system_refs

    @functools.cached_property
    def connected_system_refs(self) -> FrozenSet[CoordinateSystemRef]:
        return frozenset(ref for ref in self.node_refs if isinstance(ref.owner, CoordinateSystem))

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

    @classmethod
    def single_isolated_system(cls, sys_ref: CoordinateSystemRef[CoordinateSystem]):
        return cls([], isolated_system_refs=frozenset([sys_ref]))

    @classmethod
    def from_ome_zarr(cls, transform_dicts: List[Dict], system_dicts: List[Dict]):
        named_systems: Set[CoordinateSystemRef] = set()
        seen_names = set()
        for system_dict in system_dicts:
            system = CoordinateSystem.from_ome_zarr(system_dict)
            name: CoordinateSystemName = system_dict.get("name")
            if not name:
                raise ValueError(f"Invalid metadata: Coordinate system has no name. Received: {system_dict}")
            if name in seen_names:
                raise ValueError(
                    f'Invalid metadata: Multiple coordinate systems named "{name}". Received: {system_dict}'
                )
            named_systems.add(system.as_ref(name))
            seen_names.add(name)
        unresolved_transforms: List[Transform] = []
        all_transforms: List[Transform] = []
        isolated_systems = set(named_systems)
        for transform_dict in transform_dicts:
            t = Transform.from_ome_zarr(transform_dict).with_resolved(None, named_refs=named_systems)
            if not t.is_fully_bound:
                raise ValueError(
                    f'Transform input and output must have "path", "name" or both. Received: {transform_dict}'
                )
            all_transforms.append(t)
            isolated_systems.discard(t.source)
            isolated_systems.discard(t.target)
            if t.has_unresolved_endpoint:
                unresolved_transforms.append(t)
        graph = _TransformGraph(
            all_transforms,
            unresolved_transforms=frozenset(unresolved_transforms),
            isolated_system_refs=frozenset(isolated_systems),
        )
        return graph

    def to_ome_zarr(
        self, version="rfc-5", paths_by_node: Optional[PathsByNode] = None
    ) -> Dict[Literal["coordinateTransformations", "coordinateSystems"], List[Dict]]:
        if version != "rfc-5":
            warnings.warn(
                f"Unsupported OME-Zarr version {version!r}. "
                f"This method only targets RFC-5 as of 03/2026. Metadata may be invalid."
            )
        systems = [ref.owner.to_ome_zarr(name=ref.name, version=version) for ref in self.all_system_refs]
        transforms = [t.to_ome_zarr(version, for_scene=True, paths_by_node=paths_by_node) for t in self.transforms]
        d: Dict[Literal["coordinateTransformations", "coordinateSystems"], List[Dict]] = {}
        if systems:
            d["coordinateSystems"] = systems
        if transforms:
            d["coordinateTransformations"] = transforms
        return d

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
