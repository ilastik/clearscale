import functools
from dataclasses import dataclass, replace
from typing import Mapping, Dict, Union, Tuple, Literal
from typing import Optional, List

from clearscale._multiscale import Multiscale
from clearscale._transforms import (
    RelativePath,
    CoordinateSystemName,
    CoordinateSystemRef,
    _UnresolvedRef,
    CoordinateSystem,
    Transform,
    _TransformGraph,
)

MultiscalesByPath = Mapping[RelativePath, Multiscale]
PathsByMultiscale = Mapping[Multiscale, RelativePath]
UserFacingCoordinateSystemKey = Union[
    CoordinateSystemName,
    Multiscale,
    Tuple[Multiscale, CoordinateSystemName],
    Dict[Literal["path", "name"], Union[RelativePath, CoordinateSystemName]],
]


@dataclass(frozen=True)
class Scene:
    _internal_graph: _TransformGraph
    _external_multiscales: Mapping[Multiscale, Optional[RelativePath]]  # remembers paths for to_ome_zarr

    @property
    def is_fully_resolved(self) -> bool:
        return len(self._internal_graph.unresolved_transforms) == 0

    @property
    def unresolved_paths(self) -> List[RelativePath]:
        refs = set()
        for t in self._internal_graph.unresolved_transforms:
            refs.add(t.source)
            refs.add(t.target)
        paths = {ref.path for ref in refs}
        paths.discard(None)
        paths.discard("")
        return list(paths)

    @functools.cached_property
    def _full_graph(self):
        all_transforms = list(self._internal_graph.transforms)
        for ms in self._external_multiscales:
            all_transforms.extend(ms._get_interface_transform())  # noqa: package-private, not class-private
            all_transforms.extend(ms._transform_graph.transforms)  # noqa: package-private, not class-private
        return _TransformGraph(all_transforms)

    @classmethod
    def from_ome_zarr(cls, scene_attrs: Dict):
        # TODO: accept an optional callable get_multiscale_meta;
        #  where the default provided implementation simply chooses
        #  the first entry in the multiscales-array at the path.
        #  Problem: will also need get_shape for Multiscale.from_ome_zarr :)
        transform_dicts = scene_attrs.get("coordinateTransformations", [])
        system_dicts = scene_attrs.get("coordinateSystems", [])
        graph = _TransformGraph.from_ome_zarr(transform_dicts, system_dicts)
        return cls(graph, _external_multiscales={})

    def with_resolved(
        self,
        multiscales: Optional[MultiscalesByPath] = None,
    ) -> "Scene":
        multiscales = multiscales if multiscales else {}
        if not multiscales:
            return self
        updated_external: Dict[Multiscale, Optional[RelativePath]] = dict(self._external_multiscales)
        # Invert for quicker lookup. Keeps only the last path if multiple for the same Multiscale.
        # Presumably they're all equally valid. Pointing to copies I guess?
        paths_by_multiscale: Optional[Dict[Multiscale, RelativePath]] = {v: k for k, v in multiscales.items()}
        transforms = []
        remaining_unresolved = []
        for t in self._internal_graph.transforms:
            maybe_resolved_t = t.with_resolved(multiscales)
            transforms.append(maybe_resolved_t)
            if not maybe_resolved_t.is_fully_resolved:
                remaining_unresolved.append(maybe_resolved_t)
            if maybe_resolved_t is t:
                continue
            for ref in (maybe_resolved_t.source, maybe_resolved_t.target):
                assert ref is not None, f"Should never have unbound refs in scene transforms {t!r}"
                if not isinstance(ref.owner, Multiscale) or ref.owner in updated_external:
                    continue
                assert (
                    ref.owner in paths_by_multiscale
                ), f"If this multiscale wasn't already known and wasn't just provided, then where did it come from? {ref!r}."
                updated_external[ref.owner] = paths_by_multiscale[ref.owner]
        graph = replace(
            self._internal_graph,
            transforms=tuple(transforms),
            unresolved_transforms=tuple(remaining_unresolved),
        )
        return replace(
            self,
            _internal_graph=graph,
            _external_multiscales=updated_external,
        )

    def extract_unresolved_transforms_by_name_matching(
        self, multiscales: Optional[MultiscalesByPath] = None
    ) -> Tuple["Scene", Dict[Multiscale, Multiscale]]:
        # TODO: Similar to with_resolved. Try to resolve unresolved transforms and isolated systems
        #  from this scene, from all of its external_multiscales, and from all provided multiscales.
        #  Separate function because this explicitly breaks round-trip:
        #  If any new systems or transforms resolve, they are merged into this Scene and "removed"
        #  from the respective Multiscale. Multiscales are immutable, so "removal" means making a new one.
        #  Hence the tuple return: The second value is a mapping of old Multiscales to modified Multiscales.
        raise NotImplementedError()

    def to_ome_zarr(self, version: str = "0.6.dev3", paths: Optional[PathsByMultiscale] = None) -> Dict:
        coordinate_system_dicts = []
        for ref in self._internal_graph.all_system_refs:
            coordinate_system_dicts.append(ref.owner.to_ome_zarr(name=ref.name, version=version))

        all_paths_by_multiscale = {}
        for ms, path in self._external_multiscales.items():
            if path:
                all_paths_by_multiscale[ms] = path
        paths = paths if paths else {}
        for ms, path in paths.items():
            if path:
                all_paths_by_multiscale[ms] = path

        coordinate_transformations_dicts = [
            t.to_ome_zarr(version, for_scene=True, paths_by_node=all_paths_by_multiscale)
            for t in self._internal_graph.transforms
        ]

        result: Dict = {"coordinateTransformations": coordinate_transformations_dicts}
        if coordinate_system_dicts:
            result["coordinateSystems"] = coordinate_system_dicts
        return result

    def transforms_between(
        self, source: UserFacingCoordinateSystemKey, target: UserFacingCoordinateSystemKey, include_children=False
    ) -> Optional[List[Transform]]:
        source_ref = self._get_ref_for_key(source, include_children)
        target_ref = self._get_ref_for_key(target, include_children)
        if source_ref is None or target_ref is None:
            return None
        if include_children:
            return self._full_graph.path_between(source_ref, target_ref)
        return self._internal_graph.path_between(source_ref, target_ref)

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
            # If there were more than 1 and user cared, they'd give us a tuple
            return key._intrinsic_ref  # noqa: package-private, not class-private

        if isinstance(key, CoordinateSystemName):
            # Purely matching by name could bring up refs to any TransformGraphNode
            # (Multiscale, CoordinateSystem, or None in case of _UnresolvedRef)
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
