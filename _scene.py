import functools
import itertools
from dataclasses import dataclass, replace
from typing import Mapping, Dict, Union, FrozenSet, Tuple, Literal
from typing import Optional, List

from lazyflow.utility.io_util.clearscale._multiscale import Multiscale
from ._transforms import (
    RelativePath,
    CoordinateSystemName,
    CoordinateSystemRef,
    _UnresolvedRef,
    CoordinateSystem,
    Transform,
    _TransformGraph,
)

MultiscalesByPath = Mapping[RelativePath, Multiscale]
UserFacingCoordinateSystemKey = Union[
    CoordinateSystemName,
    Multiscale,
    Tuple[Multiscale, CoordinateSystemName],
    Dict[Literal["path", "name"], Union[RelativePath, CoordinateSystemName]],
]


@dataclass(frozen=True, slots=True)
class Scene:
    _resolved_graph: _TransformGraph  # only fully resolved transforms get to go here
    _external_multiscales: Mapping[Multiscale, Optional[RelativePath]]  # remembers paths for to_ome_zarr
    _unresolved_transforms: FrozenSet[
        Transform
    ]  # transforms with either input or output being unresolved reference; needed for ome-zarr scene

    @property
    def is_fully_resolved(self) -> bool:
        return len(self._unresolved_transforms) == 0

    @property
    def unresolved_paths(self) -> List[RelativePath]:
        refs = set()
        for t in self._unresolved_transforms:
            refs.add(t.source)
            refs.add(t.target)
        paths = {ref.path for ref in refs}
        paths.discard(None)
        paths.discard("")
        return list(paths)

    @functools.cached_property
    def _full_graph(self):
        all_transforms = list(self._resolved_graph.transforms) + list(self._unresolved_transforms)
        for ms in self._external_multiscales:
            all_transforms.extend(ms.get_interface_transform())
            all_transforms.extend(ms.transform_graph.transforms)
        return _TransformGraph(all_transforms)

    @functools.cached_property
    def _graph_incl_unresolved(self):
        return _TransformGraph(list(self._resolved_graph.transforms) + list(self._unresolved_transforms))

    @classmethod
    def from_ome_zarr(cls, scene_attrs: Dict, multiscales: Optional[MultiscalesByPath] = None):
        # TODO: optional multiscales parameter could only ever be useful if user pre-parses scene and resolves some multiscales
        # better would be an optional callable get_multiscale_meta; where the default provided implementation simply chooses
        # the first entry in the multiscales-array at the path.
        # Problem: will also need get_shape for Multiscale.from_ome_zarr :)
        isolated_systems: List[CoordinateSystemRef] = []
        seen_names = []
        for system_dict in scene_attrs.get("coordinateSystems", []):
            system = CoordinateSystem.from_ome_zarr(system_dict)
            name: CoordinateSystemName = system_dict.get("name")
            if not name:
                raise ValueError(f"Invalid metadata: Coordinate system has no name. Received: {system_dict}")
            if name in seen_names:
                raise ValueError(
                    f'Invalid metadata: Multiple coordinate systems named "{name}". Received: {scene_attrs}'
                )
            isolated_systems.append(system.as_ref(name))
            seen_names.append(name)

        unresolved_transforms: List[Transform] = []
        resolved_transforms: List[Transform] = []
        for transform_dict in scene_attrs.get("coordinateTransformations", []):
            t = Transform.from_ome_zarr(transform_dict).with_resolved(multiscales, named_refs=set(isolated_systems))
            if not t.is_bound:
                raise ValueError(
                    f'Transform input and output must have "path", "name" or both. Received: {transform_dict}'
                )
            if not t.has_unresolved_endpoint:
                resolved_transforms.append(t)
            else:
                unresolved_transforms.append(t)

        external_multiscales: Dict[Multiscale, RelativePath] = {v: k for k, v in multiscales}
        graph = _TransformGraph(resolved_transforms, isolated_system_refs=frozenset(isolated_systems))
        return cls(graph, external_multiscales, frozenset(unresolved_transforms))

    def with_resolved(
        self,
        multiscales: Optional[MultiscalesByPath],
        *,
        connect_to_child_isolated_systems_and_dangling_transforms=False,
    ) -> "Scene":
        multiscales = multiscales if multiscales else {}
        updated_external: Dict[Multiscale, Optional[RelativePath]] = dict(self._external_multiscales)
        # Invert for quicker lookup.
        # Keeps only the last path if multiple paths are provided for the same Multiscale.
        # Presumably they're all equally valid (pointing to copies I guess?).
        multiscales_inverted: Optional[Dict[Multiscale, RelativePath]] = {v: k for k, v in multiscales.items()}
        all_isolated_systems = set(self._resolved_graph.isolated_system_refs)  # we mainly do lookups
        transforms_to_resolve = list(self._unresolved_transforms)  # we mainly iterate
        if connect_to_child_isolated_systems_and_dangling_transforms:
            # Hopefully this flag name makes it clear enough you're really not supposed to do this.
            all_multiscales = itertools.chain(multiscales.values(), self._resolved_graph.multiscales)
            for ms in all_multiscales:
                all_isolated_systems.update(ms.transform_graph.isolated_system_refs)
                transforms_to_resolve.extend(ms.unresolved_transforms)
        resolved_transforms = list(self._resolved_graph.transforms)
        remaining_unresolved = []
        for t in transforms_to_resolve:
            maybe_resolved_t = t.with_resolved(multiscales, named_refs=all_isolated_systems)
            if maybe_resolved_t.has_unresolved_endpoint:
                remaining_unresolved.append(maybe_resolved_t)
            else:
                resolved_transforms.append(maybe_resolved_t)
            for ref in (maybe_resolved_t.source, maybe_resolved_t.target):
                assert ref is not None, "Should never have unbound refs in scene transforms"
                if not isinstance(ref.owner, Multiscale) or ref.owner in updated_external:
                    continue
                if connect_to_child_isolated_systems_and_dangling_transforms and ref.owner not in multiscales_inverted:
                    # ref.owner was connected through a name-only reference and not originally defined in this Scene
                    continue
                assert (
                    ref.owner in multiscales_inverted
                ), f"If this multiscale wasn't already known and wasn't just provided, then where did it come from? {ref.owner}."
                updated_external[ref.owner] = multiscales_inverted[ref.owner]
        graph = replace(self._resolved_graph, transforms=resolved_transforms)
        return replace(
            self,
            _resolved_graph=graph,
            _external_multiscales=updated_external,
            _unresolved_transforms=frozenset(remaining_unresolved),
        )

    def transforms_between(
        self, source: UserFacingCoordinateSystemKey, target: UserFacingCoordinateSystemKey, include_children=False
    ) -> Optional[List[Transform]]:
        source_ref = self._get_ref_for_key(source, include_children)
        target_ref = self._get_ref_for_key(target, include_children)
        if source_ref is None or target_ref is None:
            return None
        if include_children:
            return self._full_graph.path_between(source_ref, target_ref)
        return self._graph_incl_unresolved.path_between(source_ref, target_ref)

    def _get_ref_for_key(
        self, key: UserFacingCoordinateSystemKey, include_children: bool
    ) -> Optional[CoordinateSystemRef]:
        if isinstance(key, dict):  # Dict[Literal["path", "name"], Union[RelativePath, CoordinateSystemName]]
            return _UnresolvedRef(name=key["name"], path=key["path"])

        if isinstance(key, tuple):  # Tuple[Multiscale, CoordinateSystemName]
            if not isinstance(key[0], Multiscale) or not (isinstance(key[1], CoordinateSystemName)):
                raise TypeError(f"Coordinate system key must be tuple(multiscale, system_name). Received: {key}")
            return key[0].as_ref(key[1])

        if isinstance(key, Multiscale):
            return key.intrinsic_ref  # If there were more than 1 and user cared, they'd give us a tuple

        if isinstance(key, CoordinateSystemName):
            # Purely matching by name could bring up refs to any TransformGraphNode
            # (Multiscale, CoordinateSystem, or None in case of _UnresolvedRef)
            own_systems = self._resolved_graph.coordinate_system_refs
            for ref in own_systems:  # Expected: These CoordinateSystems can only be retrieved by name
                if ref.name == key:
                    return ref
            # Best effort: Maybe the name is still unique among unresolved refs or multiscales
            all_refs = self._full_graph.node_refs if include_children else self._graph_incl_unresolved.node_refs
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
