import functools
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, Mapping, NamedTuple, Union, Tuple, Literal, FrozenSet, TypeGuard
from typing import Optional, List

from clearscale._errors import MismatchingMultiscaleError
from clearscale._axis_values import Unit
from clearscale._multiscale import (
    Multiscale,
    OmeZarrAxesParam,
    _merge_ome_zarr_axes,
    _reconcile_axis_prop_params,
)
from clearscale._axis_values import AxisKey
from clearscale._spatial_relations import RelationsParam, SpatialRelation, normalize_relations_param
from clearscale.types import AxisKeyT
from clearscale._transforms import (
    IdentityTransform,
    OmeZarrAxes,
    relation_to_transform_canonic,
    RelativePath,
    CoordinateSystem,
    CoordinateSystemName,
    FileRef,
    NodeRef,
    _UnresolvedRef,
    AnyRef,
    Transform,
    TransformGraph,
)
from clearscale._transforms._base import _is_owner_coordinate_system

MultiscalesByPath = Mapping[RelativePath, Multiscale]
MultiscaleSystemKey = Union[Multiscale, Tuple[Multiscale, CoordinateSystemName]]
"""Selects one of a Multiscale's coordinate systems: A plain Multiscale selects its own (intrinsic) space,
`(multiscale, name)` selects one of the other coordinate systems it knows."""
UserFacingCoordinateSystemKey = Union[
    CoordinateSystemName,
    MultiscaleSystemKey,
    Dict[Literal["path", "name"], Union[RelativePath, CoordinateSystemName]],
]
Node = Union[Multiscale, NodeRef[CoordinateSystem]]


class Connection(NamedTuple):
    """One Multiscale's relation to the coordinate system shared by a Scene's Multiscales,
    for `Scene.from_shared_coordinate_system`. Plain `(source, to_shared)` tuples work as well."""

    source: MultiscaleSystemKey
    """The Multiscale (its own space), or `(multiscale, name)` for one of the other coordinate systems it knows."""
    to_shared: RelationsParam
    """How the shared coordinate system is reached from `source`, as for `Multiscale.with_coordinate_system`:
    `source --to_shared--> shared`. `None` means `source` is the same space as the shared one."""


@dataclass(frozen=True)
class Scene:
    _internal_graph: TransformGraph
    _multiscale_paths: MultiscalesByPath
    """Helper property to round-trip paths: Scene.from_ome_zarr().with_resolved().to_ome_zarr()."""

    def __post_init__(self):
        if not isinstance(self._multiscale_paths, MappingABC):
            raise TypeError(
                f"_multiscale_paths must be a mapping like {{path: Multiscale}}. "
                f"Received: {self._multiscale_paths!r}"
            )
        paths = dict(self._multiscale_paths)
        object.__setattr__(self, "_multiscale_paths", paths)

    def __eq__(self, other):
        """Structural equality: Same coordinate systems and transforms between the same (equal) Multiscales.
        Ignores which instances the graph refers to, and the helper `_multiscale_paths` (like Multiscale
        ignores `.ome`), so that `Scene.from_ome_zarr(scene.to_ome_zarr(...)).with_resolved(...) == scene`."""
        if not isinstance(other, Scene):
            return NotImplemented
        return self._structural_signature() == other._structural_signature()

    def __hash__(self):
        return hash(self._structural_signature())

    def _structural_signature(self):
        # Multiscales name their own (intrinsic) system randomly, which says nothing about the space they describe
        intrinsic_refs = {
            ref.owner._as_ref_to_intrinsic(): "_intrinsic"
            for ref in self._internal_graph.node_refs
            if isinstance(ref, NodeRef) and isinstance(ref.owner, Multiscale)
        }
        return self._internal_graph.structural_signature(rename=intrinsic_refs)

    @property
    def is_fully_resolved(self) -> bool:
        return len(self._internal_graph.unresolved_transforms) == 0

    @functools.cached_property
    def unresolved_paths(self) -> List[RelativePath]:
        paths = []  # Order might matter to the consumer
        seen_paths = set()
        for t in self._internal_graph.unresolved_transforms:
            for endpoint in (t.source, t.target):
                if not isinstance(endpoint, _UnresolvedRef):
                    continue
                if endpoint.file is None or endpoint.file.path in seen_paths:
                    continue
                paths.append(endpoint.file.path)
                seen_paths.add(endpoint.file.path)
        return paths

    @functools.cached_property
    def _resolved_multiscales(self) -> FrozenSet[NodeRef[Multiscale]]:
        def _is_owner_multiscale(ref: Optional[AnyRef]) -> TypeGuard[NodeRef[Multiscale]]:
            return isinstance(ref, NodeRef) and isinstance(ref.owner, Multiscale)

        return frozenset(
            endpoint
            for t in self._internal_graph.transforms
            for endpoint in (t.source, t.target)
            if _is_owner_multiscale(endpoint)
        )

    @functools.cached_property
    def _full_graph(self):
        all_transforms = list(self._internal_graph.transforms)
        for ms_ref in self._resolved_multiscales:
            ms = ms_ref.owner
            ms_transforms = []
            for child_t in ms._transform_graph.transforms:
                if not _is_owner_coordinate_system(child_t.source) or not _is_owner_coordinate_system(child_t.target):
                    # Scenes can only reference CoordinateSystems within their child Multiscales,
                    # they can't reach the Multiscale's own child (label) Multiscales, if it transforms to any of them.
                    continue
                new_source = ms._as_ref(child_t.source.name)
                new_target = ms._as_ref(child_t.target.name)
                ms_transforms.append(child_t.bound(source=new_source, target=new_target))
            all_transforms.extend(ms_transforms)
        return TransformGraph(all_transforms)

    @classmethod
    def from_shared_coordinate_system(
        cls,
        *,
        coordinate_system: CoordinateSystemName,
        connections: Iterable[Tuple[MultiscaleSystemKey, RelationsParam]],
        unit: Optional[Union[Unit, Mapping[AxisKeyT, str]]] = None,
        ome_zarr_axes: Optional[OmeZarrAxesParam] = "infer",
    ) -> "Scene":
        """
        Relate several Multiscales to one new coordinate system that they share (e.g. tiles to a stage).

        Every connection is exactly what `Multiscale.with_coordinate_system(coordinate_system, reached_by=...)`
        would be for that Multiscale alone: `source --to_shared--> coordinate_system`. The `source` can also be one of
        the other coordinate systems the Multiscale knows, using `(multiscale, name)`.
        The shared system's axes are what the relations produce, so all connections must produce identical axes.
        Its axis properties (`unit`, `ome_zarr_axes`) are determined like in `with_coordinate_system`, from all
        sources together: They may differ in blank properties, but not in contradicting values.

        ```
        Scene.from_shared_coordinate_system(
            coordinate_system="stage",
            connections=[
                Connection(tile1, to_shared=tile1_to_stage),
                Connection(tile2, to_shared=tile2_to_stage),
            ],
        )
        ```
        Note `Translation` says where the *origin moves to* when you apply it, so a tile that sits at position
        `p` on the stage reaches the stage through `Translation(p).inverted()`.
        """
        if not isinstance(coordinate_system, str) or not coordinate_system:
            raise ValueError(f"The shared coordinate system needs a name. Received: {coordinate_system!r}")
        entries = list(connections)
        if not entries:
            raise ValueError("Need at least one connection to a Multiscale.")

        sources: List[NodeRef[Multiscale]] = []
        relations: List[Optional[SpatialRelation]] = []
        target_axes: Optional[Tuple[AxisKey, ...]] = None
        reconciled_axes: List[OmeZarrAxes] = []
        for i, entry in enumerate(entries):
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError(f"connections[{i}] must be a Connection or a (source, to_shared) tuple. Got: {entry!r}")
            key, relations_param = entry
            source = cls._multiscale_key_to_ref(key)
            if source in sources:
                raise ValueError(
                    f"connections[{i}] selects the same coordinate system as connections[{sources.index(source)}]. "
                    "Each Multiscale coordinate system can only be connected once."
                )
            relation = normalize_relations_param(relations_param)
            try:
                reached_axes = relation.target_axes(source.axes) if relation else source.axes
            except ValueError as e:
                raise ValueError(f"connections[{i}] cannot be applied to axes {source.axes}: {e}") from e
            if target_axes is None:
                target_axes = reached_axes
            elif reached_axes != target_axes:
                raise ValueError(
                    "All connections must reach the shared coordinate system with identical axes. "
                    f"connections[0] reaches {target_axes}, but connections[{i}] reaches {reached_axes}."
                )
            sources.append(source)
            relations.append(relation)
            dependencies = relation._axis_dependencies(source.axes) if relation else tuple((a,) for a in source.axes)
            reconciled_axes.append(
                _reconcile_axis_prop_params(
                    reached_axes, dependencies, cls._source_ome_zarr_axes(source), ome_zarr_axes, unit
                )
            )

        shared_axes = _merge_ome_zarr_axes(reconciled_axes, sources="the connected coordinate systems")
        shared = CoordinateSystem(shared_axes)._as_ref(coordinate_system)
        transforms = [
            (relation_to_transform_canonic(relation, source.axes) if relation else IdentityTransform()).bound(
                source=source, target=shared
            )
            for source, relation in zip(sources, relations)
        ]
        return cls(_internal_graph=TransformGraph(transforms, system_refs=(shared,)), _multiscale_paths={})

    @classmethod
    def from_multiscale_pair(
        cls,
        *,
        a: MultiscaleSystemKey,
        b: MultiscaleSystemKey,
        a_to_b: RelationsParam,
    ) -> "Scene":
        """
        Relate two Multiscales to each other, e.g. an image to the atlas it was registered to.

        `a_to_b` says how `b`'s space is reached from `a`'s, the same way as for `b.as_derived_from(a, by=a_to_b)`
        or `a.with_coordinate_system(..., reached_by=a_to_b)`: `a --a_to_b--> b`.
        Spatial relations say what you do with `a` to obtain `b`, e.g. `Translation(p)` for "cropped with its
        origin at p", or `ProjectionTo(axes)` for "dropped or added axes". `None` means the same space.

        Applying `a_to_b` to the axes of `a` must produce the axes of `b`.
        `a` and `b` can also be one of the other coordinate systems of a Multiscale: `(multiscale, name)`.
        To relate two coordinate systems of a single Multiscale, use `Multiscale.with_coordinate_system` instead.
        """
        a_ref = cls._multiscale_key_to_ref(a)
        b_ref = cls._multiscale_key_to_ref(b)
        if a_ref.owner is b_ref.owner:
            raise ValueError(
                "`a` and `b` are the same Multiscale. Spaces of a single Multiscale are related by "
                "`Multiscale.with_coordinate_system`, a Scene relates different Multiscales."
            )
        relation = normalize_relations_param(a_to_b)
        try:
            reached_axes = relation.target_axes(a_ref.axes) if relation else a_ref.axes
        except ValueError as e:
            raise ValueError(f"`a_to_b` cannot be applied to the axes of `a` {a_ref.axes}: {e}") from e
        if reached_axes != b_ref.axes:
            raise ValueError(
                f"Applying `a_to_b` to the axes of `a` {a_ref.axes} produces {reached_axes}, "
                f"but `b` has axes {b_ref.axes}."
            )
        transform = (relation_to_transform_canonic(relation, a_ref.axes) if relation else IdentityTransform()).bound(
            source=a_ref, target=b_ref
        )
        return cls(_internal_graph=TransformGraph([transform]), _multiscale_paths={})

    @classmethod
    def _from_graph_edges(
        cls,
        source_transform_targets: Iterable[Tuple[Node, Transform, Node]],
    ) -> "Scene":
        """
        Basic low-level constructor by directly specifying graph edges (source->transform->target):
        Scene._from_graph_edges([
            (moving_ms1, AffineTransform(...), fixed_ms),
            (moving_ms2, AffineTransform(...), fixed_ms),
        ])
        """

        def node_to_ref(node: Node) -> NodeRef:
            if isinstance(node, Multiscale):
                ref = node._as_ref_to_intrinsic()
            elif isinstance(node, NodeRef) and isinstance(node.owner, CoordinateSystem):
                ref = node
            else:
                raise TypeError(
                    f"Use CoordinateSystem._as_ref(name) to use a CoordinateSystem in a Scene. Received: {node!r}"
                )
            return ref

        transforms = []
        for source_node, transform, target_node in source_transform_targets:
            source = node_to_ref(source_node)
            target = node_to_ref(target_node)
            bound = transform.bound(source=source, target=target)
            transforms.append(bound)
        return cls(_internal_graph=TransformGraph(transforms=transforms), _multiscale_paths={})

    @classmethod
    def from_ome_zarr(cls, scene_attrs: Mapping[str, Any]) -> "Scene":
        # TODO: accept an optional callable get_multiscale_meta;
        #  where the default provided implementation simply chooses
        #  the first entry in the multiscales-array at the path.
        #  Problem: will also need shape_source for Multiscale.from_ome_zarr :)
        transform_dicts = scene_attrs.get("coordinateTransformations", [])
        system_dicts = scene_attrs.get("coordinateSystems", [])
        graph = TransformGraph.from_ome_zarr(transform_dicts, system_dicts)
        return cls(graph, _multiscale_paths={})

    def with_resolved(
        self,
        multiscales_by_path: Optional[MultiscalesByPath] = None,
    ) -> "Scene":
        if not multiscales_by_path or not isinstance(multiscales_by_path, MappingABC):
            return self
        transforms = []
        resolved_paths = {}
        for t in self._internal_graph.transforms:
            try:
                maybe_resolved_t = t.with_resolved(multiscales_by_path)
            except MismatchingMultiscaleError as e:
                raise ValueError(
                    f"Invalid pairing: The Multiscale provide for path {e.path!r} does not seem to be the expected one "
                    f"(expected coordinate system named {e.name!r})."
                ) from None
            transforms.append(maybe_resolved_t)
            resolved_paths.update(self._resolved_multiscale_paths(t, maybe_resolved_t, multiscales_by_path))
        paths = dict(self._multiscale_paths)
        paths.update(resolved_paths)
        graph = replace(self._internal_graph, transforms=tuple(transforms))
        return replace(self, _internal_graph=graph, _multiscale_paths=paths)

    def to_ome_zarr(self, *, version: str, multiscales_by_path: Optional[MultiscalesByPath] = None) -> Dict:
        if version != "0.6":
            raise ValueError(f"Scenes can only be written in OME-Zarr version 0.6. Received: {version}")
        all_paths = dict(self._multiscale_paths)
        if multiscales_by_path is not None:
            if not isinstance(multiscales_by_path, MappingABC):
                raise TypeError(
                    f"multiscales_by_path must be a mapping like {{path: Multiscale}}. Received: {multiscales_by_path!r}"
                )
            cleaned = {k: v for k, v in multiscales_by_path.items() if k not in (None, "")}
            all_paths.update(cleaned)
        return self._internal_graph.to_ome_zarr(version=version, nodes_by_path=all_paths)

    def _transforms_between(
        self, source: UserFacingCoordinateSystemKey, target: UserFacingCoordinateSystemKey, include_children=False
    ) -> Optional[List[Transform]]:
        source_ref = self._get_ref_for_key(source, include_children)
        target_ref = self._get_ref_for_key(target, include_children)
        if source_ref is None or target_ref is None:
            return None
        if include_children:
            return self._full_graph.path_between(source_ref, target_ref)
        return self._internal_graph.path_between(source_ref, target_ref)

    def _get_ref_for_key(self, key: UserFacingCoordinateSystemKey, include_children: bool) -> Optional[AnyRef]:
        if isinstance(key, dict):  # Dict[Literal["path", "name"], Union[RelativePath, CoordinateSystemName]]
            if key["path"] in self._multiscale_paths:
                return self._multiscale_paths[key["path"]]._as_ref(key["name"])
            return _UnresolvedRef(name=key["name"], file=FileRef.from_string(key["path"]))

        if isinstance(key, (tuple, Multiscale)):
            return self._multiscale_key_to_ref(key)

        if isinstance(key, CoordinateSystemName):
            # Purely matching by name could bring up refs to any TransformGraphNode
            # (Multiscale, CoordinateSystem, or placeholder _UnresolvedRef)
            own_systems = self._internal_graph.connected_system_refs
            for ref in own_systems:  # Expected: These CoordinateSystems can only be retrieved by name
                if ref.name == key:
                    return ref
            # Best effort: Maybe the name is still unique among unresolved refs or multiscales
            all_refs = self._full_graph.node_refs if include_children else self._internal_graph.node_refs
            name_matches = [ref for ref in all_refs if ref.name == key]
            if len(name_matches) > 1:
                raise ValueError(
                    f'Cannot retrieve transformations for name "{key}" because it is ambiguous. '
                    "Use a multiscale-name tuple to select multiscales, "
                    "or a {path, name} dict to select unresolved multiscales. "
                    f"Matches: {name_matches}."
                )
            return name_matches[0] if name_matches else None

        raise TypeError(f"Unsupported key type for coordinate system lookup: {key}")

    @staticmethod
    def _source_ome_zarr_axes(ref: NodeRef[Multiscale]) -> OmeZarrAxes:
        """Axis properties of the coordinate system that `ref` selects within its Multiscale"""
        multiscale = ref.owner
        if ref.name == multiscale._intrinsic_ref.name:
            return multiscale.ome_zarr_axes
        return multiscale.coordinate_system_ome_zarr_axes(ref.name)

    @staticmethod
    def _multiscale_key_to_ref(key: MultiscaleSystemKey) -> NodeRef[Multiscale]:
        if isinstance(key, Multiscale):
            return key._as_ref_to_intrinsic()
        if isinstance(key, tuple):
            if not isinstance(key[0], Multiscale) or not (isinstance(key[1], CoordinateSystemName)):
                raise TypeError(f"Coordinate system key must be tuple(multiscale, system_name). Received: {key}")
            return key[0]._as_ref(key[1])
        raise TypeError(f"Expected a Multiscale or a tuple(multiscale, system_name). Received: {key!r}")

    @staticmethod
    def _resolved_multiscale_paths(
        before: Transform, after: Transform, multiscales_by_path: MultiscalesByPath
    ) -> Dict[RelativePath, Multiscale]:
        resolved_paths = {}
        for old_ref, new_ref in ((before.source, after.source), (before.target, after.target)):
            if new_ref is old_ref or not isinstance(old_ref, _UnresolvedRef) or old_ref.file is None:
                continue
            multiscale = multiscales_by_path.get(old_ref.file.path)
            if multiscale is None:
                continue
            assert new_ref is not None, f"Dev error: previously not-None {old_ref!r} became None {new_ref!r}"
            if isinstance(new_ref, NodeRef) and isinstance(new_ref.owner, Multiscale) and new_ref.owner is multiscale:
                resolved_paths[old_ref.file.path] = multiscale
        return resolved_paths
