import functools
from dataclasses import dataclass, replace
from typing import Mapping, Dict, TYPE_CHECKING, Collection, Iterable, Union
from typing import Optional, List

from lazyflow.utility.io_util.clearscale._multiscale import Multiscale

if TYPE_CHECKING:
    from ._transforms import (
        Transform,
        _ReferencingTransformGraph,
        CoordinateSystemName,
        RelativePath,
        CoordinateSystem,
        _UnresolvedCoordinateSystemReference,
    )

MultiscalesByPath = Mapping[RelativePath, Multiscale]
CoordinateSystemsByName = Mapping[CoordinateSystemName, CoordinateSystem]
CoordinateSystemEndpoint = Union[
    CoordinateSystemName,
    Multiscale,
    _UnresolvedCoordinateSystemReference,
]


@dataclass(frozen=True, slots=True)
class Scene:
    _internal_graph: (
        _ReferencingTransformGraph  # only fully resolved systems, multiscales and transforms get to go here
    )
    _external_multiscales: Mapping[Multiscale, Optional[RelativePath]]  # remembers paths for to_ome_zarr
    _unresolved_transforms: List[
        Transform[CoordinateSystemEndpoint]
    ]  # transforms with either input or output being unresolved reference; needed for ome-zarr scene

    @property
    def is_fully_resolved(self) -> bool:
        return len(self._unresolved_transforms) == 0

    @functools.cached_property  # Should be fine as long as mutators create new instances
    def _full_graph(self) -> _ReferencingTransformGraph:
        all_systems = dict(**self._internal_graph.coordinate_systems)
        all_transforms = list(self._internal_graph.transforms) + self._unresolved_transforms
        for ms in self._external_multiscales:
            for name, sys in ms.transform_graph.coordinate_systems.items():
                all_systems[(ms, name)] = sys  # Keyed under (ms, name) for namespacing
            all_transforms.extend(ms.transform_graph.transforms)
        return _ReferencingTransformGraph(all_systems, all_transforms)

    @classmethod
    def from_multiscales(
        cls,
        multiscales: Iterable[Multiscale],  # Enable using Scene without paths outside of ome-zarr context
        *,
        transforms: Collection[Transform],  # input/output hold refs to Multiscales
        multiscales_by_path: Optional[
            MultiscalesByPath
        ] = None,  # overrides multiscales; this makes it possible to create scenes with paths for ome-zarr. path:ms ensures every ms has a path, and is probably the more natural way for a user to provide the information - even if we immediately have to flip it for internal purposes. Allows doing what would otherwise be a second step .with_paths
        scene_systems: Optional[CoordinateSystemsByName] = None,
    ) -> "Scene":
        """Build a Scene from resolved multiscales and transforms between them."""
        transforms = tuple(transforms)
        internal_systems = scene_systems if scene_systems else {}
        if multiscales_by_path:
            multiscales_with_path = {ms: path for path, ms in multiscales_by_path.items()}
        else:
            multiscales_with_path = {ms: None for ms in multiscales}

        # Validate that all transform endpoints are defined
        all_keys = set(internal_systems.keys())
        for ms in multiscales_with_path:
            for name in ms.transform_graph.coordinate_systems:
                all_keys.add((ms, name))  # TODO: just add the multiscale, no more (ms, name) keying
        for t in transforms:
            if t.source not in all_keys:
                raise ValueError(f"Transform source {t.source} not found in any coordinate system")
            if t.target not in all_keys:
                raise ValueError(f"Transform target {t.target} not found in any coordinate system")

        graph = _ReferencingTransformGraph(internal_systems, transforms)
        return cls(graph, multiscales_with_path, [])

    @classmethod
    def from_ome_zarr(cls, scene_attrs: Dict, multiscales: Optional[MultiscalesByPath] = None, strict=True):
        """
        If not strict, drop invalid transforms. This can break graph connectedness,
        which can make the Scene invalid.
        Scene.transforms_between and .to_ome_zarr may error in that case.
        """
        internal_systems: Dict[CoordinateSystemName, CoordinateSystem] = {}
        for system_dict in scene_attrs.get("coordinateSystems", []):
            system = CoordinateSystem.from_ome_zarr(system_dict)
            try:
                name: CoordinateSystemName = system_dict["name"]
            except KeyError as e:
                raise KeyError(f"Invalid metadata: Coordinate system has no name. Received: {system_dict}") from e
            internal_systems[name] = system

        external_multiscales: Dict[Multiscale, RelativePath] = {}
        unresolved_transforms: List[Transform] = []
        resolved_transforms: List[Transform] = []

        for transform_dict in scene_attrs.get("coordinateTransformations", []):
            t = Transform.from_ome_zarr(transform_dict)
            if not t.is_bound:
                raise ValueError(
                    f"Transform input and output must have at least one of 'path' or 'name'. Received: {transform_dict}"
                )
            if multiscales:
                t, used_multiscales = t.resolved_with(multiscales)
                external_multiscales.update(used_multiscales)
            if not t.has_unresolved_endpoint:
                resolved_transforms.append(t)
            else:
                unresolved_transforms.append(t)

        graph = _ReferencingTransformGraph(internal_systems, resolved_transforms)
        return cls(graph, external_multiscales, unresolved_transforms)

    def with_resolved(self, multiscales: MultiscalesByPath) -> "Scene":
        new_external = dict(self._external_multiscales)
        resolved_transforms = list(self._internal_graph.transforms)
        remaining_unresolved = []
        for t in self._unresolved_transforms:
            maybe_resolved_t, used_multiscales = t.resolved_with(multiscales)
            if maybe_resolved_t.has_unresolved_endpoint:
                remaining_unresolved.append(maybe_resolved_t)
            else:
                resolved_transforms.append(maybe_resolved_t)
            for ms, path in used_multiscales.items():
                if ms not in new_external:
                    new_external[ms] = path
        graph = _ReferencingTransformGraph(self._internal_graph.coordinate_systems, resolved_transforms)
        return replace(
            self, _internal_graph=graph, _external_multiscales=new_external, _unresolved_transforms=remaining_unresolved
        )

    def transforms_between(
        self, source: CoordinateSystemEndpoint, target: CoordinateSystemEndpoint
    ) -> Optional[List[Transform]]:
        if not self.is_fully_resolved:
            raise ValueError(
                f"This scene still has unresolved multiscales. Load the multiscales and provide them via scene.with_resolved({{path: multiscale}}). Unresolved paths: {'; '.join(self._unresolved_transforms)}"
            )

        return self._full_graph.path_between(source, target)
