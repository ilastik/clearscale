import pytest

from clearscale import (
    Affine,
    Factor,
    Multiscale,
    PermutationTo,
    ProjectionTo,
    Scale,
    Scene,
    Shape,
    Translation,
    Unit,
    ome_zarr,
)
from clearscale._errors import CannotConvertToSpatialRelationError
from clearscale._spatial_relations import SpatialRelationSequence
from clearscale._transforms import (
    CoordinateSystem,
    TranslationTransform,
    TransformGraph,
    _UnresolvedRef,
    IdentityTransform,
    ScaleTransform,
    AffineTransform,
    CoordinatesTransform,
    FileRef,
    ProjectAxisTransform,
)


def _multiscale(unit=None, **shape: int) -> Multiscale:
    return Multiscale({"s0": Scale(Shape(**shape), unit=unit)})


def test_from_graph_edges_binds_between_multiscales():
    source = _multiscale(z=2, y=3, x=4)
    target = _multiscale(z=2, y=3, x=4)

    scene = Scene._from_graph_edges([(source, ScaleTransform((2, 2, 2)), target)])

    path = scene._transforms_between(source, target)

    assert path
    assert len(path) == 1
    assert path[0] == ScaleTransform((2, 2, 2)).bound(
        source=source._as_ref(source._intrinsic_ref.name),
        target=target._as_ref(target._intrinsic_ref.name),
    )


def test_from_graph_edges_binds_coordinate_system_refs():
    world = CoordinateSystem.fromkeys("zyx")
    image = _multiscale(z=2, y=3, x=4)

    scene = Scene._from_graph_edges(
        [
            (
                image,
                ScaleTransform((2, 2, 2)),
                world._as_ref("world"),
            ),
        ]
    )

    path = scene._transforms_between(image, "world")

    assert path
    assert len(path) == 1
    assert path[0].target == world._as_ref("world")


def test_from_graph_edges_multiple_edges():
    moving = _multiscale(z=2, y=3, x=4)
    fixed = _multiscale(z=2, y=3, x=4)
    world = CoordinateSystem.fromkeys("zyx")

    scene = Scene._from_graph_edges(
        [
            (moving, ScaleTransform((2, 2, 2)), world._as_ref("world")),
            (world._as_ref("world"), AffineTransform(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0))), fixed),
        ]
    )

    path = scene._transforms_between(moving, fixed)

    assert path
    assert len(path) == 2
    scale_unbound = path[0].bound(source=None, target=None)
    assert scale_unbound == ScaleTransform((2, 2, 2))
    affine_unbound = path[1].bound(source=None, target=None)
    assert affine_unbound == AffineTransform(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)))


def test_from_graph_edges_rejects_coordinate_system_and_other_types():
    world = CoordinateSystem.fromkeys("zyx")
    image = _multiscale(z=2, y=3, x=4)

    with pytest.raises(TypeError, match="Use CoordinateSystem._as_ref"):
        Scene._from_graph_edges([("some_string", ScaleTransform((2, 2, 2)), world)])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Use CoordinateSystem._as_ref"):
        Scene._from_graph_edges([(world, ScaleTransform((2, 2, 2)), image)])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Use CoordinateSystem._as_ref"):
        Scene._from_graph_edges([(image, ScaleTransform((2, 2, 2)), world)])  # type: ignore[arg-type]


def test_transforms_between_in_empty_scene():
    scene = Scene._from_graph_edges([])

    assert scene._transforms_between("world", "world") is None
    assert scene._internal_graph.transforms == ()


def test_transforms_between_accepts_path_addressed_unresolved_refs():
    world = CoordinateSystem.fromkeys("yx")._as_ref("world")
    transform = TranslationTransform(
        translation=(1, 2),
        source=_UnresolvedRef(file=FileRef(path="tile_0"), name="physical"),
        target=world,
    )
    scene = Scene(TransformGraph([transform], system_refs=(world,)), _multiscale_paths={})

    result = scene._transforms_between({"path": "tile_0", "name": "physical"}, "world")

    assert result == [transform]


def test_transforms_between_can_include_child_multiscale_graphs():
    multiscale = _multiscale(y=2, x=3).with_coordinate_system("warped")
    # Not the transform created by with_coordinate_system inside multiscale's graph,
    # but its representation when ported into the Scene:
    ms_transform_within_scene = IdentityTransform().bound(
        source=multiscale._as_ref_to_intrinsic(), target=multiscale._as_ref("warped")
    )
    world = CoordinateSystem.fromkeys("yx")._as_ref("world")
    scene_transform = TranslationTransform(
        translation=(10, 20),
        source=multiscale._as_ref_to_intrinsic(),
        target=world,
    )
    scene = Scene(TransformGraph([scene_transform], system_refs=(world,)), _multiscale_paths={})

    without_children = scene._transforms_between((multiscale, "warped"), "world")

    assert without_children is None

    with_children = scene._transforms_between((multiscale, "warped"), "world", include_children=True)

    assert with_children == [
        ms_transform_within_scene.inverted(),
        scene_transform,
    ], "hop1 is stored as ms->warped, hop2 is ms->world, so the path is warped--inv(hop1)->ms--hop2->world"


def test_transforms_between_with_children_through_axis_changing_child_system():
    multiscale = _multiscale(z=2, y=3, x=4).with_coordinate_system("preview", reached_by=ProjectionTo("yx"))
    world = CoordinateSystem.fromkeys("yx")._as_ref("world")
    scene_transform = TranslationTransform(
        translation=(10, 20),
        source=multiscale._as_ref("preview"),
        target=world,
    )
    scene = Scene(TransformGraph([scene_transform], system_refs=(world,)), _multiscale_paths={})

    path = scene._transforms_between((multiscale, "preview"), "world", include_children=True)

    assert path == [scene_transform]

    full_edges = scene._full_graph.transforms  # Porting the child's transform should not trip any errors
    ported_projection = ProjectAxisTransform(
        drops=(0,), source=multiscale._as_ref_to_intrinsic(), target=multiscale._as_ref("preview")
    )
    assert full_edges == (scene_transform, ported_projection)


@pytest.mark.parametrize(
    "key",
    [
        (1, "name"),
        ("not a multiscale", "name"),
        ((_multiscale(y=2, x=3), 5), "name"),  # non-string system name in (ms, name) tuple
    ],
)
def test_transforms_between_rejects_malformed_multiscale_system_key(key):
    scene = Scene._from_graph_edges([])

    with pytest.raises(TypeError, match=r"must be tuple\(multiscale, system_name\)"):
        scene._transforms_between(key, "world")  # type: ignore[arg-type]


class TestMultiscalePaths:
    @staticmethod
    def _two_tile_scene():
        tile_0 = _multiscale(y=2, x=3)
        tile_1 = _multiscale(y=2, x=3)
        scene = Scene._from_graph_edges([(tile_0, TranslationTransform((1, 1)), tile_1)])
        return scene, tile_0, tile_1

    def test_paths_can_be_added_and_retrieved(self):
        scene, tile_0, tile_1 = self._two_tile_scene()

        assert scene.multiscale_paths == {}, "empty by default"
        assert scene.with_multiscale_paths({}) is scene, "passing nothing returns self"

        with_paths = scene.with_multiscale_paths({"tile_0": tile_0, "tile_1": tile_1})

        assert with_paths is not scene, "should not modify original instance"
        assert with_paths.multiscale_paths == {"tile_0": tile_0, "tile_1": tile_1}

    def test_multiscale_paths_is_read_only(self):
        scene, tile_0, _tile_1 = self._two_tile_scene()
        with_paths = scene.with_multiscale_paths({"tile_0": tile_0})

        with pytest.raises(TypeError):
            with_paths.multiscale_paths["other"] = tile_0  # type: ignore[index]

        assert with_paths.multiscale_paths == {"tile_0": tile_0}

    def test_accepts_paths_for_only_some_multiscales(self):
        scene, tile_0, _tile_1 = self._two_tile_scene()

        with_paths = scene.with_multiscale_paths({"tile_0": tile_0})

        assert with_paths.multiscale_paths == {"tile_0": tile_0}

    def test_rejects_path_for_multiscale_outside_graph(self):
        scene, _tile_0, _tile_1 = self._two_tile_scene()
        stray = _multiscale(y=5, x=7)

        with pytest.raises(ValueError, match="'stray' is not part of this Scene's graph"):
            scene.with_multiscale_paths({"stray": stray})

    def test_rejects_path_for_equal_but_not_identical_multiscale(self):
        scene, tile_0, _tile_1 = self._two_tile_scene()
        assert tile_0.copy() == tile_0

        expected_message = "Paths must be given for the specific Multiscale instances passed on construction, not e.g. equal instances created by `Multiscale.copy"
        with pytest.raises(ValueError, match=expected_message):
            scene.with_multiscale_paths({"tile_0": tile_0.copy()})

    def test_rejects_same_instance_at_two_paths(self):
        scene, tile_0, _tile_1 = self._two_tile_scene()

        with pytest.raises(ValueError, match="Received two paths for one instance: 'a' and 'b'"):
            scene.with_multiscale_paths({"a": tile_0, "b": tile_0})

    @pytest.mark.parametrize("path", ["", None, 3])
    def test_rejects_non_string_or_empty_paths(self, path):
        scene, tile_0, _tile_1 = self._two_tile_scene()

        with pytest.raises(ValueError, match="non-empty strings"):
            scene.with_multiscale_paths({path: tile_0})

    def test_rejects_non_multiscale_path_values(self):
        scene, _tile_0, _tile_1 = self._two_tile_scene()

        with pytest.raises(TypeError, match="must map to a Multiscale"):
            scene.with_multiscale_paths({"tile_0": "not a multiscale"})  # type: ignore[arg-type]

    def test_with_resolved_rejects_same_multiscale_for_two_paths(self):
        ms = _multiscale(y=2, x=3)
        object.__setattr__(ms._intrinsic_ref, "name", "physical")
        world = {"name": "world", "axes": [{"name": "y"}, {"name": "x"}]}
        scene = Scene.from_ome_zarr(
            {
                "coordinateSystems": [world],
                "coordinateTransformations": [
                    {
                        "type": "translation",
                        "input": {"path": path, "name": "physical"},
                        "output": {"name": "world"},
                        "translation": [0, 0],
                    }
                    for path in ("tile_0", "tile_1")
                ],
            }
        )

        with pytest.raises(ValueError, match="Received two paths for one instance: 'tile_0' and 'tile_1'"):
            scene.with_resolved({"tile_0": ms, "tile_1": ms})

    def test_with_multiscale_paths_keeps_paths_not_mentioned(self):
        scene, tile_0, tile_1 = self._two_tile_scene()
        scene = scene.with_multiscale_paths({"tile_0": tile_0})
        scene = scene.with_multiscale_paths({"tile_1": tile_1})

        assert scene.multiscale_paths == {"tile_0": tile_0, "tile_1": tile_1}

    def test_with_multiscale_paths_overwrites_existing_path_of_same_instance(self):
        scene, tile_0, tile_1 = self._two_tile_scene()
        scene = scene.with_multiscale_paths({"old": tile_0, "tile_1": tile_1})

        scene = scene.with_multiscale_paths({"new": tile_0})

        assert scene.multiscale_paths == {"tile_1": tile_1, "new": tile_0}, "the old key must not linger"

    def test_with_multiscale_paths_allows_swapping_paths(self):
        scene, tile_0, tile_1 = self._two_tile_scene()
        scene = scene.with_multiscale_paths({"a": tile_0, "b": tile_1})
        swapped = scene.with_multiscale_paths({"a": tile_1, "b": tile_0})

        assert swapped.multiscale_paths == {"a": tile_1, "b": tile_0}

    def test_with_multiscale_paths_allows_moving_into_path_vacated_in_same_call(self):
        scene, tile_0, tile_1 = self._two_tile_scene()
        scene = scene.with_multiscale_paths({"a": tile_0, "b": tile_1})
        moved = scene.with_multiscale_paths({"c": tile_0, "a": tile_1})

        assert moved.multiscale_paths == {"c": tile_0, "a": tile_1}

    def test_with_multiscale_paths_requires_mapping(self):
        scene, tile_0, _tile_1 = self._two_tile_scene()

        with pytest.raises(TypeError, match="must be a mapping"):
            scene.with_multiscale_paths([("tile_0", tile_0)])  # type: ignore[arg-type]

    def test_with_multiscale_paths_rejects_path_of_different_instance(self):
        scene, tile_0, tile_1 = self._two_tile_scene()
        scene = scene.with_multiscale_paths({"tile_0": tile_0})

        with pytest.raises(ValueError, match="Path 'tile_0' is already used by another Multiscale"):
            scene.with_multiscale_paths({"tile_0": tile_1})

        assert scene.multiscale_paths == {"tile_0": tile_0}

    def test_with_multiscale_paths_reports_copy_as_copy_even_if_instance_has_this_path(self):
        scene, tile_0, _tile_1 = self._two_tile_scene()
        scene = scene.with_multiscale_paths({"tile_0": tile_0})

        with pytest.raises(ValueError, match="not part of this Scene's graph"):
            scene.with_multiscale_paths({"tile_0": tile_0.copy()})

    def test_with_multiscale_paths_overrides_paths_recorded_by_with_resolved(self):
        ms = _multiscale(y=2, x=3)
        object.__setattr__(ms._intrinsic_ref, "name", "physical")
        scene = Scene.from_ome_zarr(
            {
                "coordinateSystems": [{"name": "world", "axes": [{"name": "y"}, {"name": "x"}]}],
                "coordinateTransformations": [
                    {
                        "type": "translation",
                        "input": {"path": "tile_0", "name": "physical"},
                        "output": {"name": "world"},
                        "translation": [0, 0],
                    }
                ],
            }
        )
        resolved = scene.with_resolved({"tile_0": ms})
        assert resolved.multiscale_paths == {"tile_0": ms}

        moved = resolved.with_multiscale_paths({"renamed": ms})

        assert moved.multiscale_paths == {"renamed": ms}
        assert moved.is_fully_resolved

    def test_to_ome_zarr_requires_paths_for_all_graph_multiscales(self):
        scene, tile_0, _tile_1 = self._two_tile_scene()

        with pytest.raises(ValueError, match="no path is known for 2 of its Multiscales"):
            scene.to_ome_zarr(version="0.6")

        scene = scene.with_multiscale_paths({"tile_0": tile_0})

        with pytest.raises(ValueError, match="no path is known for 1 of its Multiscales"):
            scene.to_ome_zarr(version="0.6")

    def test_to_ome_zarr_does_not_require_paths_for_coordinate_systems(self):
        world = CoordinateSystem.fromkeys("yx")._as_ref("world")
        other = CoordinateSystem.fromkeys("yx")._as_ref("other")
        scene = Scene._from_graph_edges([(world, TranslationTransform((1, 1)), other)])

        assert scene.to_ome_zarr(version="0.6")["coordinateTransformations"]


class TestMultiscalesProperty:
    def test_lists_multiscales_in_order_of_appearance(self):
        tiles = [_multiscale(y=2, x=2), _multiscale(y=3, x=3), _multiscale(y=4, x=4)]
        stage = CoordinateSystem.fromkeys("yx")._as_ref("stage")

        scene = Scene._from_graph_edges([(tile, IdentityTransform(), stage) for tile in tiles])

        assert len(scene.multiscales) == 3
        assert all(found is tile for found, tile in zip(scene.multiscales, tiles))

    def test_lists_multiscales_that_look_alike_separately(self):
        tiles = [_multiscale(y=2, x=2), _multiscale(y=2, x=2)]
        assert tiles[0] == tiles[1]
        stage = CoordinateSystem.fromkeys("yx")._as_ref("stage")

        scene = Scene._from_graph_edges([(tile, IdentityTransform(), stage) for tile in tiles])

        assert len(scene.multiscales) == 2

    def test_does_not_duplicate_multiply_referenced_instances(self):
        volume = _multiscale(z=2, y=3, x=4).with_coordinate_system("preview", reached_by=ProjectionTo("yx"))
        stage = CoordinateSystem.fromkeys("yx")._as_ref("stage")

        scene = Scene._from_graph_edges(
            [
                (volume._as_ref_to_intrinsic(), ProjectAxisTransform(drops=(0,)), stage),
                (volume._as_ref("preview"), IdentityTransform(), stage),
            ]
        )

        assert len(scene.multiscales) == 1
        assert scene.multiscales[0] is volume

    def test_excludes_unresolved_multiscales(self):
        tile = _multiscale(y=2, x=2)
        stage = CoordinateSystem.fromkeys("yx")._as_ref("stage")
        original = Scene._from_graph_edges([(tile, IdentityTransform(), stage)]).with_multiscale_paths({"tile": tile})
        read_back = Scene.from_ome_zarr(original.to_ome_zarr(version="0.6"))

        assert read_back.multiscales == ()
        assert read_back.with_resolved({"tile": tile}).multiscales == (tile,)


class TestFromSharedCoordinateSystem:
    @staticmethod
    def _system_ome_axes(scene: Scene, name: str):
        system = next(ref for ref in scene._internal_graph.all_system_refs if ref.name == name)
        return system.owner

    def test_connects_each_multiscale_to_the_shared_system(self):
        tile1 = _multiscale(y=2, x=3)
        tile2 = _multiscale(y=2, x=3)

        scene = Scene.from_shared_coordinate_system(
            coordinate_system="stage",
            connections=[
                # Accepts both the Scene.Connection NamedTuple helper and an equivalent bare tuple
                Scene.Connection(tile1, to_shared=Translation(y=1.0, x=2.0)),
                (tile2, Affine.from_factor(Factor(y=0.5, x=0.5))),
            ],
        )

        path1 = scene._transforms_between(tile1, "stage")
        path2 = scene._transforms_between(tile2, "stage")
        assert path1 is not None and path2 is not None
        assert path1[0].unbound() == TranslationTransform((1.0, 2.0)), "relations preserve the coordinate direction"
        assert path2[0].unbound() == ScaleTransform((0.5, 0.5))
        assert path1[0].source == tile1._as_ref_to_intrinsic()
        assert path1[0].target == path2[0].target
        assert len(scene._transforms_between(tile1, tile2) or []) == 2
        assert scene.is_fully_resolved

    def test_none_relation_is_identity(self):
        tile = _multiscale(y=2, x=3)

        scene = Scene.from_shared_coordinate_system(coordinate_system="stage", connections=[(tile, None)])

        path = scene._transforms_between(tile, "stage")
        assert path is not None and path[0].unbound() == IdentityTransform()

    def test_connection_and_plain_tuples_are_equivalent(self):
        tile1 = _multiscale(y=2, x=3)
        tile2 = _multiscale(y=2, x=3)
        relations = [Translation(y=1.0, x=2.0), Translation(y=-1.0, x=0.5)]

        from_named = Scene.from_shared_coordinate_system(
            coordinate_system="stage",
            connections=[Scene.Connection(tile1, relations[0]), Scene.Connection(tile2, relations[1])],
        )
        from_plain = Scene.from_shared_coordinate_system(
            coordinate_system="stage", connections=[(tile1, relations[0]), (tile2, relations[1])]
        )

        assert from_named == from_plain

    def test_accepts_one_shot_iterables(self):
        tile1 = _multiscale(y=2, x=3)
        tile2 = _multiscale(y=2, x=3)
        connections = [(tile1, None), (tile2, Translation(y=1.0, x=2.0))]

        from_generator = Scene.from_shared_coordinate_system(
            coordinate_system="stage", connections=(connection for connection in connections)
        )

        assert from_generator == Scene.from_shared_coordinate_system(coordinate_system="stage", connections=connections)

    def test_axis_properties_are_inherited_from_the_connected_multiscales(self):
        tile1 = _multiscale(unit=Unit(y="micrometer", x="micrometer"), y=2, x=3)
        tile2 = _multiscale(y=2, x=3)  # blank unit, compatible

        scene = Scene.from_shared_coordinate_system(
            coordinate_system="stage", connections=[(tile1, None), (tile2, Translation(y=1.0, x=0.0))]
        )

        assert self._system_ome_axes(scene, "stage") == {
            "y": ome_zarr.Axis(name="y", type="space", unit="micrometer"),
            "x": ome_zarr.Axis(name="x", type="space", unit="micrometer"),
        }

    def test_explicit_unit_overrides_inherited_one(self):
        tile = _multiscale(unit=Unit(y="micrometer", x="micrometer"), y=2, x=3)

        scene = Scene.from_shared_coordinate_system(
            coordinate_system="stage",
            connections=[
                (
                    tile,
                    Affine.from_unit_conversion(
                        source_unit=Unit(y="micrometer", x="micrometer"),
                        target_unit=Unit(y="millimeter", x="millimeter"),
                    ),
                )
            ],
            unit=Unit(y="millimeter", x="millimeter"),
        )

        assert self._system_ome_axes(scene, "stage").get_unit() == Unit(y="millimeter", x="millimeter")

    def test_axes_follow_the_relations(self):
        tile = _multiscale(y=2, x=3)

        scene = Scene.from_shared_coordinate_system(
            coordinate_system="stage", connections=[(tile, [ProjectionTo("zyx"), Translation(z=0.0, y=2.0, x=2.0)])]
        )

        axes = self._system_ome_axes(scene, "stage")
        assert axes.axes == ("z", "y", "x")
        assert axes["z"].type == "space", "inserted axes get their standard type inferred"
        path = scene._transforms_between(tile, "stage")
        assert path is not None and len(path) == 1, "the 2D multiscale can reach the 3D system"

    def test_can_connect_other_coordinate_systems_of_a_multiscale(self):
        volume = _multiscale(z=2, y=3, x=4).with_coordinate_system("preview", reached_by=ProjectionTo("yx"))
        flat = _multiscale(y=3, x=4)

        scene = Scene.from_shared_coordinate_system(
            coordinate_system="stage",
            connections=[((volume, "preview"), Translation(y=1.0, x=1.0)), (flat, None)],
        )

        path = scene._transforms_between((volume, "preview"), "stage")
        assert path is not None and path[0].source == volume._as_ref("preview")
        assert self._system_ome_axes(scene, "stage").axes == ("y", "x")

    def test_same_multiscale_can_connect_via_different_systems(self):
        volume = _multiscale(z=2, y=3, x=4).with_coordinate_system("preview", reached_by=ProjectionTo("yx"))

        scene = Scene.from_shared_coordinate_system(
            coordinate_system="stage",
            connections=[(volume, ProjectionTo("yx")), ((volume, "preview"), None)],
        )

        assert len(scene._internal_graph.transforms) == 2

    def test_rejects_same_coordinate_system_twice(self):
        tile = _multiscale(y=2, x=3)

        with pytest.raises(
            ValueError, match=r"connections\[1\] selects the same coordinate system as connections\[0\]"
        ):
            Scene.from_shared_coordinate_system(coordinate_system="stage", connections=[(tile, None), (tile, None)])

    def test_rejects_connections_reaching_different_axes(self):
        volume = _multiscale(z=2, y=3, x=4)
        flat = _multiscale(y=3, x=4)

        with pytest.raises(ValueError, match=r"connections\[0\] reaches .*z.*connections\[1\] reaches .*y.*x"):
            Scene.from_shared_coordinate_system(coordinate_system="stage", connections=[(volume, None), (flat, None)])

    def test_rejects_conflicting_axis_properties(self):
        um = _multiscale(unit=Unit(y="micrometer", x="micrometer"), y=2, x=3)
        mm = _multiscale(unit=Unit(y="millimeter", x="millimeter"), y=2, x=3)

        with pytest.raises(ValueError, match="Conflicting 'unit'"):
            Scene.from_shared_coordinate_system(coordinate_system="stage", connections=[(um, None), (mm, None)])

    def test_reports_relations_that_cannot_be_applied(self):
        tile = _multiscale(y=2, x=3)

        with pytest.raises(ValueError, match=r"connections\[0\] cannot be applied to axes \('y', 'x'\)"):
            Scene.from_shared_coordinate_system(coordinate_system="stage", connections=[(tile, PermutationTo("zyx"))])

    def test_rejects_unknown_system_name(self):
        tile = _multiscale(y=2, x=3)

        with pytest.raises(ValueError, match="does not specify a coordinate system named: 'nope'"):
            Scene.from_shared_coordinate_system(coordinate_system="stage", connections=[((tile, "nope"), None)])

    @pytest.mark.parametrize("name", ["", None, 5])
    def test_rejects_invalid_shared_system_name(self, name):
        with pytest.raises(ValueError, match="needs a name"):
            Scene.from_shared_coordinate_system(
                coordinate_system=name, connections=[(_multiscale(y=2, x=3), None)]  # type: ignore[arg-type]
            )

    def test_rejects_no_connections(self):
        with pytest.raises(ValueError, match="at least one connection"):
            Scene.from_shared_coordinate_system(coordinate_system="stage", connections=[])

    @pytest.mark.parametrize(
        "entry", [_multiscale(y=2, x=3), (_multiscale(y=2, x=3),), (_multiscale(y=2, x=3), None, None)]
    )
    def test_rejects_malformed_connections(self, entry):
        with pytest.raises(TypeError, match="must be a Scene.Connection or a"):
            Scene.from_shared_coordinate_system(coordinate_system="stage", connections=[entry])  # type: ignore[list-item]


class TestSceneEquality:
    @staticmethod
    def _stitched(*tiles: Multiscale, name: str = "stage", **kwargs) -> Scene:
        return Scene.from_shared_coordinate_system(
            coordinate_system=name,
            connections=[(tile, Translation(y=0.0, x=4.0 * i)) for i, tile in enumerate(tiles)],
            **kwargs,
        )

    def test_scenes_built_alike_from_the_same_multiscales_are_equal(self):
        tiles = [_multiscale(y=2, x=2), _multiscale(y=2, x=2)]

        assert self._stitched(*tiles) == self._stitched(*tiles)
        assert hash(self._stitched(*tiles)) == hash(self._stitched(*tiles))

    def test_equal_multiscales_are_interchangeable_despite_different_intrinsic_names(self):
        first = [_multiscale(y=2, x=2), _multiscale(y=2, x=2)]
        second = [_multiscale(y=2, x=2), _multiscale(y=2, x=2)]
        assert first[0]._intrinsic_ref.name != second[0]._intrinsic_ref.name

        assert self._stitched(*first) == self._stitched(*second)
        assert hash(self._stitched(*first)) == hash(self._stitched(*second))

    def test_differs_by_transform_system_name_and_axis_properties(self):
        tiles = [_multiscale(y=2, x=2), _multiscale(y=2, x=2)]
        scene = self._stitched(*tiles)

        shifted = Scene.from_shared_coordinate_system(
            coordinate_system="stage",
            connections=[(tiles[0], Translation(y=0.0, x=0.0)), (tiles[1], Translation(y=0.0, x=5.0))],
        )
        assert scene != shifted
        assert scene != self._stitched(*tiles, name="world")
        assert scene != self._stitched(*tiles, unit=Unit(y="micrometer", x="micrometer"))

    def test_differs_by_multiscale_content(self):
        assert self._stitched(_multiscale(y=2, x=2)) != self._stitched(_multiscale(y=3, x=2))

    def test_differs_by_number_of_equal_multiscales(self):
        connect = lambda n: Scene.from_shared_coordinate_system(
            coordinate_system="stage", connections=[(_multiscale(y=2, x=2), None) for _ in range(n)]
        )

        assert connect(1) != connect(2)
        assert connect(2) == connect(2)

    def test_interchanging_equal_multiscales_gives_equal_scene(self):
        a, b = _multiscale(y=2, x=2), _multiscale(y=2, x=2)

        assert self._stitched(a, b) == self._stitched(b, a)

    def test_ignores_connection_order(self):
        a, b = _multiscale(y=2, x=2), _multiscale(y=3, x=2)
        connect = lambda *connections: Scene.from_shared_coordinate_system(
            coordinate_system="stage", connections=connections
        )

        assert connect((a, None), (b, Translation(y=1.0, x=0.0))) == connect((b, Translation(y=1.0, x=0.0)), (a, None))

    def test_not_equal_to_other_types(self):
        assert self._stitched(_multiscale(y=2, x=2)) != "stage"

    def test_survives_ome_zarr_round_trip(self):
        tiles = [_multiscale(unit=Unit(y="micrometer", x="micrometer"), y=2, x=2), _multiscale(y=2, x=2)]
        paths = {"a.zarr": tiles[0], "b.zarr": tiles[1]}
        scene = self._stitched(*tiles).with_multiscale_paths(paths)

        read_back = Scene.from_ome_zarr(scene.to_ome_zarr(version="0.6"))

        assert read_back != scene, "unresolved: refs are placeholders for the Multiscales"
        assert read_back.with_resolved(paths) == scene


class TestFromMultiscalePair:
    def test_connects_a_to_b(self):
        raw = _multiscale(y=4, x=4)
        atlas = _multiscale(y=4, x=4)

        scene = Scene.from_multiscale_pair(a=raw, b=atlas, a_to_b=Translation(y=1.0, x=2.0))

        path = scene._transforms_between(raw, atlas)
        assert path is not None and len(path) == 1
        assert path[0].unbound() == TranslationTransform((1.0, 2.0)), "relations preserve the coordinate direction"
        assert path[0].source == raw._as_ref_to_intrinsic()
        assert path[0].target == atlas._as_ref_to_intrinsic()
        inverse = scene._transforms_between(atlas, raw)
        assert inverse is not None and inverse[0].unbound() == TranslationTransform((-1.0, -2.0))
        assert scene.is_fully_resolved

    def test_none_relation_is_identity(self):
        raw = _multiscale(y=4, x=4)
        atlas = _multiscale(y=4, x=4)

        scene = Scene.from_multiscale_pair(a=raw, b=atlas, a_to_b=None)

        path = scene._transforms_between(raw, atlas)
        assert path is not None and path[0].unbound() == IdentityTransform()

    def test_relation_chains_are_applied_in_order(self):
        volume = _multiscale(z=2, y=4, x=4)
        flat = _multiscale(y=2, x=2)

        scene = Scene.from_multiscale_pair(
            a=volume,
            b=flat,
            a_to_b=[ProjectionTo("yx"), Affine.from_factor(Factor(y=0.5, x=0.5)), Translation(y=1.0, x=1.0)],
        )

        path = scene._transforms_between(volume, flat)
        assert path is not None and len(path) == 1

    def test_dropping_axes_cannot_be_traversed_backwards(self):
        volume = _multiscale(z=2, y=4, x=4)
        flat = _multiscale(y=4, x=4)

        scene = Scene.from_multiscale_pair(a=volume, b=flat, a_to_b=ProjectionTo("yx"))

        assert scene._transforms_between(volume, flat) is not None
        assert scene._transforms_between(flat, volume) is None

    def test_inserting_axes_can_be_traversed_backwards(self):
        flat = _multiscale(y=4, x=4)
        volume = _multiscale(z=2, y=4, x=4)

        scene = Scene.from_multiscale_pair(a=flat, b=volume, a_to_b=ProjectionTo("zyx"))

        assert scene._transforms_between(flat, volume) is not None
        assert scene._transforms_between(volume, flat) is not None

    def test_can_relate_other_coordinate_systems_of_the_multiscales(self):
        volume = _multiscale(z=2, y=3, x=4).with_coordinate_system("preview", reached_by=ProjectionTo("yx"))
        flat = _multiscale(y=3, x=4).with_coordinate_system("shifted", reached_by=Translation(y=1.0, x=1.0))

        scene = Scene.from_multiscale_pair(a=(volume, "preview"), b=(flat, "shifted"), a_to_b=None)

        path = scene._transforms_between((volume, "preview"), (flat, "shifted"))
        assert path is not None and len(path) == 1
        assert path[0].source == volume._as_ref("preview") and path[0].target == flat._as_ref("shifted")
        assert scene._transforms_between(volume, flat) is None, "intrinsic systems are not connected"
        children = scene._transforms_between(volume, flat, include_children=True)
        assert children is not None and len(children) == 3, "connected through the multiscales' own graphs"

    def test_rejects_relation_not_producing_axes_of_b(self):
        volume = _multiscale(z=2, y=4, x=4)
        flat = _multiscale(y=4, x=4)

        with pytest.raises(ValueError, match=r"produces \('z', 'y', 'x'\), but `b` has axes \('y', 'x'\)"):
            Scene.from_multiscale_pair(a=volume, b=flat, a_to_b=None)

    def test_reports_relations_that_cannot_be_applied(self):
        with pytest.raises(ValueError, match=r"`a_to_b` cannot be applied to the axes of `a` \('y', 'x'\)"):
            Scene.from_multiscale_pair(
                a=_multiscale(y=2, x=2), b=_multiscale(z=2, y=2, x=2), a_to_b=PermutationTo("zyx")
            )

    def test_rejects_relating_a_multiscale_to_itself(self):
        multiscale = _multiscale(z=2, y=3, x=4).with_coordinate_system("preview", reached_by=ProjectionTo("yx"))

        with pytest.raises(ValueError, match="same Multiscale"):
            Scene.from_multiscale_pair(a=multiscale, b=(multiscale, "preview"), a_to_b=ProjectionTo("yx"))

    def test_accepts_equal_but_distinct_multiscales(self):
        scene = Scene.from_multiscale_pair(a=_multiscale(y=2, x=2), b=_multiscale(y=2, x=2), a_to_b=None)

        assert len(scene._internal_graph.transforms) == 1

    def test_rejects_unknown_system_name_and_malformed_keys(self):
        tile = _multiscale(y=2, x=2)

        with pytest.raises(ValueError, match="does not specify a coordinate system named: 'nope'"):
            Scene.from_multiscale_pair(a=(tile, "nope"), b=_multiscale(y=2, x=2), a_to_b=None)
        with pytest.raises(TypeError, match="Expected a Multiscale"):
            Scene.from_multiscale_pair(a="stage", b=tile, a_to_b=None)  # type: ignore[arg-type]

    def test_equality_depends_on_direction_and_relation(self):
        a, b = _multiscale(y=2, x=2), _multiscale(y=3, x=3)
        scene = Scene.from_multiscale_pair(a=a, b=b, a_to_b=None)

        assert scene == Scene.from_multiscale_pair(a=a, b=b, a_to_b=None)
        assert scene != Scene.from_multiscale_pair(a=b, b=a, a_to_b=None)
        assert scene != Scene.from_multiscale_pair(a=a, b=b, a_to_b=Translation(y=1.0, x=0.0))


class TestRelationBetween:
    @pytest.mark.parametrize(
        ("relation", "a_axes", "b_axes"),
        [
            pytest.param(Affine.from_factor(Factor(y=0.5, x=0.25)), dict(y=8, x=8), dict(y=4, x=2), id="scale"),
            pytest.param(Translation(y=1.5, x=-2.0), dict(y=4, x=4), dict(y=4, x=4), id="translation"),
            pytest.param(PermutationTo("xy"), dict(y=4, x=4), dict(x=4, y=4), id="permutation"),
            pytest.param(ProjectionTo("yx"), dict(z=2, y=4, x=4), dict(y=4, x=4), id="drop axis"),
            pytest.param(ProjectionTo("zyx"), dict(y=4, x=4), dict(z=1, y=4, x=4), id="insert axis"),
        ],
    )
    def test_returns_the_relation_given_to_from_multiscale_pair(self, relation, a_axes, b_axes):
        a, b = _multiscale(**a_axes), _multiscale(**b_axes)
        scene = Scene.from_multiscale_pair(a=a, b=b, a_to_b=relation)

        assert scene.relation_from(a, to=b) == relation

    def test_reverse_direction_is_the_inverse_relation(self):
        a, b = _multiscale(y=8, x=8), _multiscale(y=4, x=4)
        scene = Scene.from_multiscale_pair(
            a=a, b=b, a_to_b=[Affine.from_factor(Factor(y=0.5, x=0.5)), Translation(y=1.0, x=3.0)]
        )

        assert scene.relation_from(a, to=b) == SpatialRelationSequence(
            (Affine.from_factor(Factor(y=0.5, x=0.5)), Translation(y=1.0, x=3.0))
        )
        # Canonical form always puts the scaling first, so the sequence inverted, in canonical form,
        # has the Translation rescaled
        assert scene.relation_from(b, to=a) == SpatialRelationSequence(
            (Affine.from_factor(Factor(y=2.0, x=2.0)), Translation(y=-2.0, x=-6.0))
        )

    def test_returns_relation_from_shared_coordinate_system(self):
        tile1, tile2 = _multiscale(y=2, x=2), _multiscale(y=2, x=2)
        scene = Scene.from_shared_coordinate_system(
            coordinate_system="stage", connections=[(tile1, None), (tile2, Translation(y=0.0, x=-4.0))]
        )

        assert scene.relation_from(tile2, to="stage") == Translation(y=0.0, x=-4.0)
        assert scene.relation_from("stage", to=tile2) == Translation(y=0.0, x=4.0)
        assert scene.relation_from(tile1, to=tile2) == Translation(y=0.0, x=4.0)

    def test_can_select_other_coordinate_systems_of_multiscales(self):
        volume = _multiscale(z=2, y=4, x=4).with_coordinate_system("preview", reached_by=ProjectionTo("yx"))
        flat = _multiscale(y=4, x=4)
        scene = Scene.from_multiscale_pair(a=(volume, "preview"), b=flat, a_to_b=Translation(y=1.0, x=1.0))

        assert scene.relation_from((volume, "preview"), to=flat) == Translation(y=1.0, x=1.0)
        assert scene.relation_from(volume, to=flat) == SpatialRelationSequence(
            (ProjectionTo("yx"), Translation(y=1.0, x=1.0))
        ), "should follow connections within child Multiscales"

    def test_returns_none_for_the_same_space(self):
        a, b = _multiscale(y=2, x=2), _multiscale(y=2, x=2)
        scene = Scene.from_multiscale_pair(a=a, b=b, a_to_b=None)

        assert scene.relation_from(a, to=b) is None
        assert scene.relation_from(a, to=a) is None

    def test_dropped_axes_can_only_be_traced_forwards(self):
        volume, flat = _multiscale(z=2, y=4, x=4), _multiscale(y=4, x=4)
        scene = Scene.from_multiscale_pair(a=volume, b=flat, a_to_b=ProjectionTo("yx"))

        assert scene.relation_from(volume, to=flat) == ProjectionTo("yx")
        with pytest.raises(ValueError, match="no \\(invertible\\) path"):
            scene.relation_from(flat, to=volume)

    def test_rejects_unrelated_multiscales(self):
        a, b = _multiscale(y=2, x=2), _multiscale(y=2, x=2)
        scene = Scene.from_shared_coordinate_system(coordinate_system="stage", connections=[(a, None)])

        with pytest.raises(ValueError, match="no \\(invertible\\) path"):
            scene.relation_from(a, to=b)

    def test_rejects_unknown_spaces(self):
        a = _multiscale(y=2, x=2)
        scene = Scene.from_shared_coordinate_system(coordinate_system="stage", connections=[(a, None)])

        with pytest.raises(ValueError, match="no coordinate system matching 'nowhere'"):
            scene.relation_from(a, to="nowhere")

    def test_requires_resolved_spaces_and_works_after_resolving(self):
        tiles = [_multiscale(y=2, x=2), _multiscale(y=2, x=2)]
        scene = Scene.from_shared_coordinate_system(
            coordinate_system="stage", connections=[(tiles[0], None), (tiles[1], Translation(y=0.0, x=-4.0))]
        )
        paths = {"a.zarr": tiles[0], "b.zarr": tiles[1]}
        scene = scene.with_multiscale_paths(paths)
        unresolved = Scene.from_ome_zarr(scene.to_ome_zarr(version="0.6"))
        name_of_unresolved_multiscale = tiles[1]._intrinsic_ref.name

        with pytest.raises(ValueError, match="not resolved yet"):
            unresolved.relation_from(name_of_unresolved_multiscale, to="stage")
        assert unresolved.with_resolved(paths).relation_from(tiles[1], to="stage") == Translation(y=0.0, x=-4.0)

    def test_reports_connections_that_are_no_spatial_relation(self):
        a, b = _multiscale(y=2, x=2), _multiscale(y=2, x=2)
        scene = Scene._from_graph_edges([(a, CoordinatesTransform(path="coordinates"), b)])

        with pytest.raises(CannotConvertToSpatialRelationError):
            scene.relation_from(a, to=b)
