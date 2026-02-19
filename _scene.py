import functools
from collections import deque
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Set, Dict, Union, TYPE_CHECKING, Collection
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


@dataclass(slots=True)
class Scene:
    _internal_graph: _TransformGraph
    _external_multiscales: MultiscalesByPath  # each with its own subgraph
    _unresolved_paths: Mapping[RelativePath, Set[CoordinateSystemName]]  # input.path: input.name

    def __post_init__(self):
        object.__setattr__(self, "_external_multiscales", MappingProxyType(dict(self._external_multiscales)))
        object.__setattr__(self, "_unresolved_paths", MappingProxyType(dict(self._unresolved_paths)))

    @property
    def is_fully_resolved(self) -> bool:
        return len(self._unresolved_paths) == 0

    @property
    @functools.cached_property  # Should be fine as long as mutators create new instances
    def _graph(self) -> _TransformGraph:
        all_systems = dict(self._internal_graph.coordinate_systems)
        all_transforms = list(self._internal_graph.transforms)
        for path, ms in self._external_multiscales.items():
            for (_, name), sys in ms.transform_graph.coordinate_systems.items():
                all_systems[(path, name)] = sys  # Keyed under (path, name) for namespacing
            all_transforms.extend(ms.transform_graph.transforms)
        return _TransformGraph(all_systems, all_transforms)

    @classmethod
    def from_multiscales(
        cls,
        multiscales_by_path: MultiscalesByPath,
        *,
        transforms: Collection[
            Transform
        ],  # TODO: Each Transform must already be namespace-keyed like multiscales_by_path! How can we make that ergonomic?
        scene_systems: Optional[Collection[CoordinateSystem]] = None,
    ) -> "Scene":
        """Build a Scene from resolved multiscales and transforms between them."""
        internal_systems = {(None, sys.name): sys for sys in (scene_systems or [])}

        # Validate that all transform endpoints exist
        all_keys = set(internal_systems.keys())
        for path, ms in multiscales_by_path.items():
            for _, name in ms.transform_graph.coordinate_systems.keys():
                all_keys.add((path, name))

        for t in transforms:
            if t.input_system not in all_keys:
                raise ValueError(f"Transform input {t.input_system} not found in any coordinate system")
            if t.output_system not in all_keys:
                raise ValueError(f"Transform output {t.output_system} not found in any coordinate system")

        graph = _TransformGraph(internal_systems, transforms)
        return cls(graph, multiscales_by_path, {})

    @classmethod
    def from_ome_zarr(cls, scene_attrs: Dict, multiscales: Optional[MultiscalesByPath] = None, strict=True):
        """
        If not strict, drop invalid transforms. This can break graph connectedness,
        which can make the Scene invalid.
        Scene.transforms_between and .to_ome_zarr may error in that case.
        """
        internal_systems: Dict[CoordinateSystemKey, CoordinateSystem] = {}
        for system_dict in scene_attrs.get("coordinateSystems", []):
            system = CoordinateSystem.from_ome_zarr(system_dict)
            try:
                key: CoordinateSystemKey = (None, system_dict["name"])
            except KeyError as e:
                raise KeyError(f"Invalid metadata: Coordinate system has no name. Received: {system_dict}") from e
            internal_systems[key] = system

        if multiscales is None:
            multiscales: MultiscalesByPath = {}
        external_systems: Dict[RelativePath, Multiscale] = {}
        unresolved_paths: Dict[RelativePath, Set[CoordinateSystemName]] = {}  # path -> set of names at that path
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

                if (path, name) in internal_systems:
                    continue
                if path is None:
                    if strict:
                        raise ValueError(
                            f"Invalid transformation metadata: Reference to an undefined '{side}' coordinate system. Received: {transform_dict}"
                        )
                    else:
                        has_valid_inout = False
                elif path not in multiscales:
                    unresolved_paths.setdefault(path, set()).add(name)
                elif path in external_systems and external_systems[path] is not multiscales[path]:
                    raise ValueError("Two different multiscales were provided for the same path.")
                else:
                    external_systems[path] = multiscales[path]
            if has_valid_inout:
                edges.append(Transform.from_ome_zarr(transform_dict))

        graph = _TransformGraph(internal_systems, edges)
        return cls(graph, external_systems, unresolved_paths)

    def with_resolved(self, multiscales: MultiscalesByPath) -> "Scene":
        new_external = self._external_multiscales.copy()
        new_unresolved = self._unresolved_paths.copy()
        for path, ms in multiscales.items():
            if path in self._external_multiscales:
                raise ValueError(
                    f"The multiscale at {path} is already resolved in this scene. Use replace=True to update it."
                )
            if path not in self._unresolved_paths:
                raise ValueError(f"{path} refers to an unknown multiscale.")
            expected_keys = set((None, name) for name in self._unresolved_paths[path])
            actual_keys = set(ms.transform_graph.coordinate_systems.keys())
            if not expected_keys.issubset(actual_keys):
                missing = expected_keys - actual_keys
                raise ValueError(
                    f"Multiscale at '{path}' is missing expected coordinate systems: {[name for _, name in missing]}."
                )
            # TODO: Validate that ms actually has required axes? self._unresolved_paths would need to store refs to the transforms, and we could only validate if the other side of the transform has already been resolved
            new_external[path] = ms
            del new_unresolved[path]
        return self.__class__(self._internal_graph, new_external, new_unresolved)

    def transforms_between(
        self, source: Union[CoordinateSystemKey, Multiscale], target: Union[CoordinateSystemKey, Multiscale]
    ) -> Optional[
        List[Transform]
    ]:  # or maybe return Optional[Transform], but the Transform could be a TransformSequence
        # Individual Scales are always aligned with the multiscale's "intrinsic" coordinate system.
        # Their transforms are not exposed externally.
        if isinstance(source, Multiscale):
            for path, ms in self._external_multiscales.items():
                if ms is source:
                    source_key = (path, source.aligned_system)
                    break
            else:
                raise ValueError(f"Scene contains no multiscale matching source. Received: {source}")
        else:
            source_key = source
        if isinstance(target, Multiscale):
            for path, ms in self._external_multiscales.items():
                if ms is target:
                    target_key = (path, target.aligned_system)
                    break
            else:
                raise ValueError(f"Scene contains no multiscale matching target. Received: {target}")
        else:
            target_key = target

        for key in (source_key, target_key):
            path, name = key
            if path is not None and path in self._unresolved_paths and name in self._unresolved_paths[path]:
                raise ValueError(f"System ({path}, {name}) is unresolved. Provide the metadata via .with_resolved.")

        return self._graph.path_between(source_key, target_key)

    def _validate_connectedness(self) -> None:
        # TODO: LLM VOMIT; HAVEN'T CHECKED
        if not self.is_fully_resolved:
            raise ValueError("Cannot validate connectedness with unresolved references")

        systems = set(self._graph.coordinate_systems.keys())
        if len(systems) <= 1:
            return

        # Build undirected graph (all edges count for connectedness, even non-invertible)
        adjacency: Dict[CoordinateSystemKey, Set[CoordinateSystemKey]] = {s: set() for s in systems}
        for t in self._graph.transforms:
            adjacency[t.input].add(t.output)
            adjacency[t.output].add(t.input)  # Treat as undirected for connectedness

        # BFS from arbitrary start
        start = next(iter(systems))
        reachable = {start}
        queue = deque([start])

        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    queue.append(neighbor)

        if reachable != systems:
            unreachable = systems - reachable
            raise ValueError(
                f"Transformation graph is not fully connected per RFC-5. "
                f"Unreachable coordinate systems: {unreachable}"
            )
