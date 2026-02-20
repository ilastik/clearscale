import functools
from dataclasses import dataclass, replace
from typing import Mapping, Set, Dict, TYPE_CHECKING, Collection, Iterable
from typing import Optional, List

from lazyflow.utility.io_util.clearscale._multiscale import Multiscale

if TYPE_CHECKING:
    from ._transforms import (
        Transform,
        _TransformGraph,
        CoordinateSystemName,
        RelativePath,
        CoordinateSystemKey,
        CoordinateSystem,
    )

MultiscalesByPath = Mapping[RelativePath, Multiscale]
CoordinateSystemsByName = Mapping[CoordinateSystemName, CoordinateSystem]


@dataclass(frozen=True, slots=True)
class Scene:
    _internal_graph: _TransformGraph
    _external_multiscales: Mapping[Multiscale, Optional[RelativePath]]  # each with its own subgraph
    _unresolved_paths: Mapping[RelativePath, Set[CoordinateSystemName]]  # input.path: input.name

    @property
    def is_fully_resolved(self) -> bool:
        # TODO: even if all multiscales are resolved, there could still be transforms with unresolved references
        return len(self._unresolved_paths) == 0

    @functools.cached_property  # Should be fine as long as mutators create new instances
    def _full_graph(self) -> _TransformGraph:
        all_systems = dict(**self._internal_graph.coordinate_systems)
        all_transforms = list(self._internal_graph.transforms)
        for ms in self._external_multiscales:
            for name, sys in ms.transform_graph.coordinate_systems.items():
                all_systems[(ms, name)] = sys  # Keyed under (ms, name) for namespacing
            all_transforms.extend(ms.transform_graph.transforms)
        return _TransformGraph(all_systems, all_transforms)

    @classmethod
    def from_multiscales(
        cls,
        multiscales: Iterable[Multiscale],  # Enable using Scene without paths outside of ome-zarr context
        *,
        transforms: Collection[Transform],  # input/output hold refs to Multiscales
        multiscales_by_path: Optional[
            MultiscalesByPath
        ] = None,  # this is probably the more natural way for a user to provide the information - even if we immediately have to flip it for internal purposes. Allows doing what would otherwise be a second step .with_paths
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
                all_keys.add((ms, name))
        for t in transforms:
            if t.raw not in all_keys:
                raise ValueError(f"Transform raw {t.raw} not found in any coordinate system")
            if t.derived not in all_keys:
                raise ValueError(f"Transform derived {t.derived} not found in any coordinate system")

        graph = _TransformGraph(internal_systems, transforms)
        return cls(graph, multiscales_with_path, {})

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

        if multiscales is None:
            multiscales: MultiscalesByPath = {}
        external_systems: Dict[Multiscale, RelativePath] = {}
        unresolved_paths: Dict[RelativePath, Set[CoordinateSystemName]] = (
            {}
        )  # path -> set of names expected at that path
        edges: List[Transform] = []

        for transform_dict in scene_attrs.get("coordinateTransformations", []):
            has_valid_inout = True  # Used to skip this transform if either its input or output are invalid
            for side in ("input", "output"):
                if not has_valid_inout:
                    continue
                ref = transform_dict.get(side, {})
                path = ref.get("path")
                name = ref.get("name")

                if path is None and name is None:
                    raise ValueError(f"Transform {side} must have at least one of 'path' or 'name'.")

                if name is None:  # technically allowed by spec, but nonsense
                    if strict:
                        raise ValueError(
                            f"Transform {side} does not specify which coordinate system it refers to within "
                            f"multiscale at '{path}'. Sanitize the OME-Zarr metadata. To allow best-guessing"
                            f"the coordinate system (e.g. if you know there is only one), use strict=False. "
                            f"Received: {transform_dict}"
                        )
                    if path not in multiscales:
                        raise ValueError(
                            f"Transform {side} points to path '{path}' but has no 'name'. "
                            "Fetch attrs from the path, use `ms = Multiscale.from_ome_zarr(multiscale_metadata)`, "
                            "and then pass it like `multiscales={path: ms}`. "
                            f"Received: {transform_dict}"
                        )
                    if len(multiscales[path].transform_graph.coordinate_systems) > 1 and strict:
                        raise ValueError(
                            f"Invalid transformation metadata: Multiscale at '{path}' has multiple coordinate systems "
                            f"and this transform does not specify which of them acts as its '{side}'. "
                            f"Received: {transform_dict}"
                        )
                    name = multiscales[
                        path
                    ].aligned_system  # TODO: Inspect ms systems and pick one with type=space annotations over others?

                if path is None:
                    if strict:
                        raise ValueError(
                            f"Invalid transformation metadata: Reference to an undefined '{side}' coordinate system. Received: {transform_dict}"
                        )
                    else:
                        has_valid_inout = False
                elif path not in multiscales:
                    unresolved_paths.setdefault(path, set()).add(
                        name
                    )  # TODO: Ghost entries if input adds an unresolved path, but then output is invalid and causes the transform to be dropped
                else:
                    ms = multiscales[path]
                    if ms in external_systems and external_systems[ms] != path:
                        raise ValueError("Two different paths were provided for the same multiscale.")
                    external_systems[ms] = path
            if has_valid_inout:
                edges.append(Transform.from_ome_zarr(transform_dict))

        graph = _TransformGraph(internal_systems, edges)
        return cls(graph, external_systems, unresolved_paths)

    def with_resolved(self, multiscales: MultiscalesByPath, force=False) -> "Scene":
        new_external = dict(self._external_multiscales)
        new_unresolved = dict(self._unresolved_paths)
        for path, ms in multiscales.items():
            if ms in self._external_multiscales and not force:
                raise ValueError(
                    f"The multiscale at {path} is already resolved in this scene. Use force=True to update it."
                )
            if path not in self._unresolved_paths:
                raise ValueError(f"Not expecting any multiscales at {path}.")
            expected_system_names = self._unresolved_paths[path]
            actual_system_names = set(ms.transform_graph.coordinate_systems.keys())
            if not expected_system_names.issubset(actual_system_names):
                missing = expected_system_names - actual_system_names
                raise ValueError(
                    f"Multiscale at '{path}' is missing expected coordinate systems: {', '.join(missing)}."
                )
            # TODO: Validate that ms actually has required axes? self._unresolved_paths would need to store refs to the transforms, and we could only validate if the other side of the transform has already been resolved
            # TODO: Transforms with edges to the unresolved ms will have keys being _UnresolvedCoordinateSystemReference. We need to track them down and replace them with the real reference.
            new_external[ms] = path
            del new_unresolved[path]
        return replace(self, _external_multiscales=new_external, _unresolved_paths=new_unresolved)

    def transforms_between(self, raw: CoordinateSystemKey, derived: CoordinateSystemKey) -> Optional[List[Transform]]:
        if not self.is_fully_resolved:
            raise ValueError(
                f"This scene still has unresolved multiscales. Load the multiscales and provide them via scene.with_resolved({{path: multiscale}}). Unresolved paths: {'; '.join(self._unresolved_paths.keys())}"
            )

        return self._full_graph.path_between(raw, derived)
