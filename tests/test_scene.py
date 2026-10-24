import pytest

from clearscale import (
    Affine,
    ChainStep,
    Connection,
    Factor,
    Multiscale,
    PermutationTo,
    ProjectionTo,
    Scale,
    Scene,
    Shape,
    Translation,
    Unit,
    OmeZarrGroup,
)
from clearscale._errors import CannotConvertToSpatialRelationError
from clearscale._spatial_relations import SpatialRelationSequence
from clearscale.ome_zarr import Axis as OmeZarrAxis
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
)


def _multiscale(**shape: int) -> Multiscale:
    return Multiscale({"s0": Scale(Shape(**shape))})


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


def _two_tile_scene():
    tile_0 = _multiscale(y=2, x=3)
    tile_1 = _multiscale(y=2, x=3)
    scene = Scene._from_graph_edges([(tile_0, TranslationTransform((1, 1)), tile_1)])
    return scene, tile_0, tile_1


def test_scene_accepts_paths_for_graph_multiscales():
    scene, tile_0, tile_1 = _two_tile_scene()

    with_paths = scene.with_multiscale_paths({"tile_0": tile_0, "tile_1": tile_1})

    assert with_paths.multiscale_paths == {"tile_0": tile_0, "tile_1": tile_1}


def test_scene_accepts_paths_for_only_some_graph_multiscales():
    scene, tile_0, _tile_1 = _two_tile_scene()

    with_paths = scene.with_multiscale_paths({"tile_0": tile_0})

    assert with_paths.multiscale_paths == {"tile_0": tile_0}


def test_scene_rejects_path_for_multiscale_outside_graph():
    scene, _tile_0, _tile_1 = _two_tile_scene()
    stray = _multiscale(y=5, x=7)

    with pytest.raises(ValueError, match=r"'stray' is not part of this Scene's graph") as excinfo:
        scene.with_multiscale_paths({"stray": stray})

    assert "'s0'" in str(excinfo.value), "message describes the Multiscale by its scale keys"
    assert "'y': 5" in str(excinfo.value), "message describes the Multiscale by its shape"


def test_scene_rejects_path_for_equal_but_not_identical_multiscale():
    scene, tile_0, _tile_1 = _two_tile_scene()
    assert tile_0.copy() == tile_0

    with pytest.raises(ValueError, match="equal to a Multiscale in this Scene's graph, but is not the same instance"):
        scene.with_multiscale_paths({"tile_0": tile_0.copy()})


def test_scene_rejects_same_instance_at_two_paths():
    scene, tile_0, _tile_1 = _two_tile_scene()

    with pytest.raises(ValueError, match=r"more than one path: 'a' and 'b'"):
        scene.with_multiscale_paths({"a": tile_0, "b": tile_0})


@pytest.mark.parametrize("path", ["", None, 3])
def test_scene_rejects_non_string_or_empty_paths(path):
    scene, tile_0, _tile_1 = _two_tile_scene()

    with pytest.raises(ValueError, match="non-empty strings"):
        scene.with_multiscale_paths({path: tile_0})


def test_scene_rejects_non_multiscale_path_values():
    scene, _tile_0, _tile_1 = _two_tile_scene()

    with pytest.raises(TypeError, match="must map to a Multiscale"):
        scene.with_multiscale_paths({"tile_0": "not a multiscale"})  # type: ignore[arg-type]


def test_scene_stray_path_error_suggests_with_resolved_for_unresolved_path():
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

    with pytest.raises(ValueError, match=r"with_resolved\(\{'tile_0': multiscale\}\)"):
        scene.with_multiscale_paths({"tile_0": _multiscale(y=2, x=3)})


def test_with_resolved_rejects_same_multiscale_for_two_paths():
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

    with pytest.raises(ValueError, match=r"more than one path: 'tile_0' and 'tile_1'"):
        scene.with_resolved({"tile_0": ms, "tile_1": ms})


def test_transforms_between_with_children_through_axis_changing_child_system():
    from clearscale import ProjectionTo

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
    # The child graph's own (non-invertible) projection edge is ported without tripping axis validation
    full_edges = scene._full_graph.transforms
    assert any(
        t.source == multiscale._as_ref_to_intrinsic() and t.target == multiscale._as_ref("preview") for t in full_edges
    )


@pytest.mark.parametrize("key", [(1, "name"), ("not a multiscale", "name")])
def test_transforms_between_rejects_malformed_multiscale_system_key(key):
    scene = Scene._from_graph_edges([])

    with pytest.raises(TypeError, match=r"must be tuple\(multiscale, system_name\)"):
        scene._transforms_between(key, "world")  # type: ignore[arg-type]


def test_transforms_between_rejects_non_string_system_name_in_key():
    multiscale = _multiscale(y=2, x=3)
    scene = Scene._from_graph_edges([])

    with pytest.raises(TypeError, match=r"must be tuple\(multiscale, system_name\)"):
        scene._transforms_between((multiscale, 5), "world")  # type: ignore[arg-type]


def _tile(unit=None, **shape: int) -> Multiscale:
    return Multiscale({"s0": Scale(Shape(**shape), unit=unit)})


def _shared_system_axes(scene: Scene, name: str):
    system = next(ref for ref in scene._internal_graph.all_system_refs if ref.name == name)
    return system.owner


class TestFromSharedCoordinateSystem:
    def test_connects_each_multiscale_to_the_shared_system(self):
        tile1 = _tile(y=2, x=3)
        tile2 = _tile(y=2, x=3)

        scene = Scene.from_shared_coordinate_system(
            coordinate_system="stage",
            connections=[
                Connection(tile1, to_shared=Translation(y=1.0, x=2.0)),
                (tile2, Affine.from_factor(Factor(y=2, x=2))),
            ],
        )

        path1 = scene._transforms_between(tile1, "stage")
        path2 = scene._transforms_between(tile2, "stage")
        assert path1 is not None and path2 is not None
        assert path1[0].unbound() == TranslationTransform((-1.0, -2.0)), "relations are inverted into transforms"
        assert path2[0].unbound() == ScaleTransform((0.5, 0.5))
        assert path1[0].source == tile1._as_ref_to_intrinsic()
        assert path1[0].target == path2[0].target
        assert len(scene._transforms_between(tile1, tile2) or []) == 2
        assert scene.is_fully_resolved

    def test_none_relation_is_identity(self):
        tile = _tile(y=2, x=3)

        scene = Scene.from_shared_coordinate_system(coordinate_system="stage", connections=[(tile, None)])

        path = scene._transforms_between(tile, "stage")
        assert path is not None and path[0].unbound() == IdentityTransform()

    def test_connection_and_plain_tuples_are_equivalent(self):
        tile1 = _tile(y=2, x=3)
        tile2 = _tile(y=2, x=3)
        relations = [Translation(y=1.0, x=2.0), Translation(y=-1.0, x=0.5)]

        from_named = Scene.from_shared_coordinate_system(
            coordinate_system="stage", connections=[Connection(tile1, relations[0]), Connection(tile2, relations[1])]
        )
        from_plain = Scene.from_shared_coordinate_system(
            coordinate_system="stage", connections=[(tile1, relations[0]), (tile2, relations[1])]
        )

        assert from_named == from_plain

    def test_accepts_one_shot_iterables(self):
        tile1 = _tile(y=2, x=3)
        tile2 = _tile(y=2, x=3)
        connections = [(tile1, None), (tile2, Translation(y=1.0, x=2.0))]

        from_generator = Scene.from_shared_coordinate_system(
            coordinate_system="stage", connections=(connection for connection in connections)
        )

        assert from_generator == Scene.from_shared_coordinate_system(coordinate_system="stage", connections=connections)

    def test_axis_properties_are_inherited_from_the_connected_multiscales(self):
        tile1 = _tile(unit=Unit(y="micrometer", x="micrometer"), y=2, x=3)
        tile2 = _tile(y=2, x=3)  # blank unit, compatible

        scene = Scene.from_shared_coordinate_system(
            coordinate_system="stage", connections=[(tile1, None), (tile2, Translation(y=1.0, x=0.0))]
        )

        assert _shared_system_axes(scene, "stage") == {
            "y": OmeZarrAxis(name="y", type="space", unit="micrometer"),
            "x": OmeZarrAxis(name="x", type="space", unit="micrometer"),
        }

    def test_explicit_unit_overrides_inherited_one(self):
        tile = _tile(unit=Unit(y="micrometer", x="micrometer"), y=2, x=3)

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

        assert _shared_system_axes(scene, "stage").get_unit() == Unit(y="millimeter", x="millimeter")

    def test_axes_follow_the_relations(self):
        tile = _tile(y=2, x=3)

        scene = Scene.from_shared_coordinate_system(
            coordinate_system="stage", connections=[(tile, [ProjectionTo("zyx"), Translation(z=0.0, y=2.0, x=2.0)])]
        )

        axes = _shared_system_axes(scene, "stage")
        assert axes.axes == ("z", "y", "x")
        assert axes["z"].type == "space", "inserted axes get their standard type inferred"
        path = scene._transforms_between(tile, "stage")
        assert path is not None and len(path) == 1, "the 2D multiscale can reach the 3D system"

    def test_can_connect_other_coordinate_systems_of_a_multiscale(self):
        volume = _tile(z=2, y=3, x=4).with_coordinate_system("preview", reached_by=ProjectionTo("yx"))
        flat = _tile(y=3, x=4)

        scene = Scene.from_shared_coordinate_system(
            coordinate_system="stage",
            connections=[((volume, "preview"), Translation(y=1.0, x=1.0)), (flat, None)],
        )

        path = scene._transforms_between((volume, "preview"), "stage")
        assert path is not None and path[0].source == volume._as_ref("preview")
        assert _shared_system_axes(scene, "stage").axes == ("y", "x")

    def test_same_multiscale_can_connect_via_different_systems(self):
        volume = _tile(z=2, y=3, x=4).with_coordinate_system("preview", reached_by=ProjectionTo("yx"))

        scene = Scene.from_shared_coordinate_system(
            coordinate_system="stage",
            connections=[(volume, ProjectionTo("yx")), ((volume, "preview"), None)],
        )

        assert len(scene._internal_graph.transforms) == 2

    def test_rejects_same_coordinate_system_twice(self):
        tile = _tile(y=2, x=3)

        with pytest.raises(
            ValueError, match=r"connections\[1\] selects the same coordinate system as connections\[0\]"
        ):
            Scene.from_shared_coordinate_system(coordinate_system="stage", connections=[(tile, None), (tile, None)])

    def test_rejects_connections_reaching_different_axes(self):
        volume = _tile(z=2, y=3, x=4)
        flat = _tile(y=3, x=4)

        with pytest.raises(ValueError, match=r"connections\[0\] reaches .*z.*connections\[1\] reaches .*y.*x"):
            Scene.from_shared_coordinate_system(coordinate_system="stage", connections=[(volume, None), (flat, None)])

    def test_rejects_conflicting_axis_properties(self):
        um = _tile(unit=Unit(y="micrometer", x="micrometer"), y=2, x=3)
        mm = _tile(unit=Unit(y="millimeter", x="millimeter"), y=2, x=3)

        with pytest.raises(ValueError, match="Conflicting 'unit'"):
            Scene.from_shared_coordinate_system(coordinate_system="stage", connections=[(um, None), (mm, None)])

    def test_reports_relations_that_cannot_be_applied(self):
        tile = _tile(y=2, x=3)

        with pytest.raises(ValueError, match=r"connections\[0\] cannot be applied to axes \('y', 'x'\)"):
            Scene.from_shared_coordinate_system(coordinate_system="stage", connections=[(tile, PermutationTo("zyx"))])

    def test_rejects_unknown_system_name(self):
        tile = _tile(y=2, x=3)

        with pytest.raises(ValueError, match="does not specify a coordinate system named: 'nope'"):
            Scene.from_shared_coordinate_system(coordinate_system="stage", connections=[((tile, "nope"), None)])

    @pytest.mark.parametrize("name", ["", None, 5])
    def test_rejects_invalid_shared_system_name(self, name):
        with pytest.raises(ValueError, match="needs a name"):
            Scene.from_shared_coordinate_system(
                coordinate_system=name, connections=[(_tile(y=2, x=3), None)]  # type: ignore[arg-type]
            )

    def test_rejects_no_connections(self):
        with pytest.raises(ValueError, match="at least one connection"):
            Scene.from_shared_coordinate_system(coordinate_system="stage", connections=[])

    @pytest.mark.parametrize("entry", [_tile(y=2, x=3), (_tile(y=2, x=3),), (_tile(y=2, x=3), None, None)])
    def test_rejects_malformed_connections(self, entry):
        with pytest.raises(TypeError, match="must be a Connection or a"):
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
        tiles = [_tile(y=2, x=2), _tile(y=2, x=2)]

        assert self._stitched(*tiles) == self._stitched(*tiles)
        assert hash(self._stitched(*tiles)) == hash(self._stitched(*tiles))

    def test_equal_multiscales_are_interchangeable_despite_different_intrinsic_names(self):
        first = [_tile(y=2, x=2), _tile(y=2, x=2)]
        second = [_tile(y=2, x=2), _tile(y=2, x=2)]
        assert first[0]._intrinsic_ref.name != second[0]._intrinsic_ref.name

        assert self._stitched(*first) == self._stitched(*second)
        assert hash(self._stitched(*first)) == hash(self._stitched(*second))

    def test_differs_by_transform_system_name_and_axis_properties(self):
        tiles = [_tile(y=2, x=2), _tile(y=2, x=2)]
        scene = self._stitched(*tiles)

        shifted = Scene.from_shared_coordinate_system(
            coordinate_system="stage",
            connections=[(tiles[0], Translation(y=0.0, x=0.0)), (tiles[1], Translation(y=0.0, x=5.0))],
        )
        assert scene != shifted
        assert scene != self._stitched(*tiles, name="world")
        assert scene != self._stitched(*tiles, unit=Unit(y="micrometer", x="micrometer"))

    def test_differs_by_multiscale_content(self):
        assert self._stitched(_tile(y=2, x=2)) != self._stitched(_tile(y=3, x=2))

    def test_differs_by_number_of_equal_multiscales(self):
        connect = lambda n: Scene.from_shared_coordinate_system(
            coordinate_system="stage", connections=[(_tile(y=2, x=2), None) for _ in range(n)]
        )

        assert connect(1) != connect(2)
        assert connect(2) == connect(2)

    def test_interchanging_equal_multiscales_gives_equal_scene(self):
        a, b = _tile(y=2, x=2), _tile(y=2, x=2)

        assert self._stitched(a, b) == self._stitched(b, a)

    def test_ignores_connection_order(self):
        a, b = _tile(y=2, x=2), _tile(y=3, x=2)
        connect = lambda *connections: Scene.from_shared_coordinate_system(
            coordinate_system="stage", connections=connections
        )

        assert connect((a, None), (b, Translation(y=1.0, x=0.0))) == connect((b, Translation(y=1.0, x=0.0)), (a, None))

    def test_not_equal_to_other_types(self):
        assert self._stitched(_tile(y=2, x=2)) != "stage"

    def test_survives_ome_zarr_round_trip(self):
        tiles = [_tile(unit=Unit(y="micrometer", x="micrometer"), y=2, x=2), _tile(y=2, x=2)]
        paths = {"a.zarr": tiles[0], "b.zarr": tiles[1]}
        scene = self._stitched(*tiles).with_multiscale_paths(paths)

        read_back = Scene.from_ome_zarr(scene.to_ome_zarr(version="0.6"))

        assert read_back != scene, "unresolved: refs are placeholders for the Multiscales"
        assert read_back.with_resolved(paths) == scene


class TestFromMultiscalePair:
    def test_connects_a_to_b(self):
        raw = _tile(y=4, x=4)
        atlas = _tile(y=4, x=4)

        scene = Scene.from_multiscale_pair(a=raw, b=atlas, a_to_b=Translation(y=1.0, x=2.0))

        path = scene._transforms_between(raw, atlas)
        assert path is not None and len(path) == 1
        assert path[0].unbound() == TranslationTransform((-1.0, -2.0)), "relations are inverted into transforms"
        assert path[0].source == raw._as_ref_to_intrinsic()
        assert path[0].target == atlas._as_ref_to_intrinsic()
        inverse = scene._transforms_between(atlas, raw)
        assert inverse is not None and inverse[0].unbound() == TranslationTransform((1.0, 2.0))
        assert scene.is_fully_resolved

    def test_none_relation_is_identity(self):
        raw = _tile(y=4, x=4)
        atlas = _tile(y=4, x=4)

        scene = Scene.from_multiscale_pair(a=raw, b=atlas, a_to_b=None)

        path = scene._transforms_between(raw, atlas)
        assert path is not None and path[0].unbound() == IdentityTransform()

    def test_relation_chains_are_applied_in_order(self):
        volume = _tile(z=2, y=4, x=4)
        flat = _tile(y=2, x=2)

        scene = Scene.from_multiscale_pair(
            a=volume,
            b=flat,
            a_to_b=[ProjectionTo("yx"), Affine.from_factor(Factor(y=2, x=2)), Translation(y=1.0, x=1.0)],
        )

        path = scene._transforms_between(volume, flat)
        assert path is not None and len(path) == 1

    def test_dropping_axes_cannot_be_traversed_backwards(self):
        volume = _tile(z=2, y=4, x=4)
        flat = _tile(y=4, x=4)

        scene = Scene.from_multiscale_pair(a=volume, b=flat, a_to_b=ProjectionTo("yx"))

        assert scene._transforms_between(volume, flat) is not None
        assert scene._transforms_between(flat, volume) is None

    def test_inserting_axes_can_be_traversed_backwards(self):
        flat = _tile(y=4, x=4)
        volume = _tile(z=2, y=4, x=4)

        scene = Scene.from_multiscale_pair(a=flat, b=volume, a_to_b=ProjectionTo("zyx"))

        assert scene._transforms_between(flat, volume) is not None
        assert scene._transforms_between(volume, flat) is not None

    def test_can_relate_other_coordinate_systems_of_the_multiscales(self):
        volume = _tile(z=2, y=3, x=4).with_coordinate_system("preview", reached_by=ProjectionTo("yx"))
        flat = _tile(y=3, x=4).with_coordinate_system("shifted", reached_by=Translation(y=1.0, x=1.0))

        scene = Scene.from_multiscale_pair(a=(volume, "preview"), b=(flat, "shifted"), a_to_b=None)

        path = scene._transforms_between((volume, "preview"), (flat, "shifted"))
        assert path is not None and len(path) == 1
        assert path[0].source == volume._as_ref("preview") and path[0].target == flat._as_ref("shifted")
        assert scene._transforms_between(volume, flat) is None, "intrinsic systems are not connected"
        children = scene._transforms_between(volume, flat, include_children=True)
        assert children is not None and len(children) == 3, "connected through the multiscales' own graphs"

    def test_rejects_relation_not_producing_axes_of_b(self):
        volume = _tile(z=2, y=4, x=4)
        flat = _tile(y=4, x=4)

        with pytest.raises(ValueError, match=r"produces \('z', 'y', 'x'\), but `b` has axes \('y', 'x'\)"):
            Scene.from_multiscale_pair(a=volume, b=flat, a_to_b=None)

    def test_reports_relations_that_cannot_be_applied(self):
        with pytest.raises(ValueError, match=r"`a_to_b` cannot be applied to the axes of `a` \('y', 'x'\)"):
            Scene.from_multiscale_pair(a=_tile(y=2, x=2), b=_tile(z=2, y=2, x=2), a_to_b=PermutationTo("zyx"))

    def test_rejects_relating_a_multiscale_to_itself(self):
        multiscale = _tile(z=2, y=3, x=4).with_coordinate_system("preview", reached_by=ProjectionTo("yx"))

        with pytest.raises(ValueError, match="same Multiscale"):
            Scene.from_multiscale_pair(a=multiscale, b=(multiscale, "preview"), a_to_b=ProjectionTo("yx"))

    def test_accepts_equal_but_distinct_multiscales(self):
        scene = Scene.from_multiscale_pair(a=_tile(y=2, x=2), b=_tile(y=2, x=2), a_to_b=None)

        assert len(scene._internal_graph.transforms) == 1

    def test_rejects_unknown_system_name_and_malformed_keys(self):
        tile = _tile(y=2, x=2)

        with pytest.raises(ValueError, match="does not specify a coordinate system named: 'nope'"):
            Scene.from_multiscale_pair(a=(tile, "nope"), b=_tile(y=2, x=2), a_to_b=None)
        with pytest.raises(TypeError, match="Expected a Multiscale"):
            Scene.from_multiscale_pair(a="stage", b=tile, a_to_b=None)  # type: ignore[arg-type]

    def test_equality_depends_on_direction_and_relation(self):
        a, b = _tile(y=2, x=2), _tile(y=3, x=3)
        scene = Scene.from_multiscale_pair(a=a, b=b, a_to_b=None)

        assert scene == Scene.from_multiscale_pair(a=a, b=b, a_to_b=None)
        assert scene != Scene.from_multiscale_pair(a=b, b=a, a_to_b=None)
        assert scene != Scene.from_multiscale_pair(a=a, b=b, a_to_b=Translation(y=1.0, x=0.0))


class TestFromMultiscaleChain:
    @staticmethod
    def _workflow():
        raw = _tile(unit=Unit(y="micrometer", x="micrometer"), y=8, x=8)
        downscaled = _tile(y=4, x=4)
        cropped = _tile(y=2, x=2)
        return raw, downscaled, cropped

    def test_each_step_relates_the_previous_multiscale_to_its_result(self):
        raw, downscaled, cropped = self._workflow()

        scene = Scene.from_multiscale_chain(
            start=raw,
            steps=[(Affine.from_factor(Factor(y=2, x=2)), downscaled), ChainStep(Translation(y=1.0, x=1.0), cropped)],
        )

        first = scene._transforms_between(raw, downscaled)
        second = scene._transforms_between(downscaled, cropped)
        assert first is not None and second is not None
        assert first[0].unbound() == ScaleTransform((0.5, 0.5)), "relations are inverted into transforms"
        assert second[0].unbound() == TranslationTransform((-1.0, -1.0))
        assert first[0].source == raw._as_ref_to_intrinsic() and first[0].target == downscaled._as_ref_to_intrinsic()
        assert second[0].source == downscaled._as_ref_to_intrinsic()
        assert len(scene._transforms_between(raw, cropped) or []) == 2
        assert len(scene._internal_graph.transforms) == 3, "start identity, and one edge per step"

    def test_start_coordinate_system_is_an_identity_entry_point_with_the_axes_of_start(self):
        raw, downscaled, cropped = self._workflow()

        scene = Scene.from_multiscale_chain(start=raw, steps=[(Translation(y=1.0, x=1.0), downscaled), (None, cropped)])

        path = scene._transforms_between(raw, "start")
        assert path is not None and path[0].unbound() == IdentityTransform()
        assert _shared_system_axes(scene, "start") == {
            "y": OmeZarrAxis(name="y", type="space", unit="micrometer"),
            "x": OmeZarrAxis(name="x", type="space", unit="micrometer"),
        }
        from_entry_point = scene._transforms_between("start", cropped)
        assert from_entry_point is not None and len(from_entry_point) == 3

    def test_start_coordinate_system_can_be_renamed_or_omitted(self):
        raw, downscaled, _ = self._workflow()
        steps = [(Translation(y=1.0, x=1.0), downscaled)]

        renamed = Scene.from_multiscale_chain(start=raw, steps=steps, start_coordinate_system="original")
        omitted = Scene.from_multiscale_chain(start=raw, steps=steps, start_coordinate_system=None)

        assert renamed._transforms_between(raw, "original") is not None
        assert omitted._internal_graph.all_system_refs == ()
        assert len(omitted._internal_graph.transforms) == 1

    def test_steps_can_offer_their_own_coordinate_systems(self):
        raw, downscaled, cropped = self._workflow()

        scene = Scene.from_multiscale_chain(
            start=raw,
            steps=[
                (Translation(y=1.0, x=1.0), downscaled, "overview"),
                ChainStep(None, cropped, coordinate_system="detail"),
            ],
        )

        assert [ref.name for ref in scene._internal_graph.all_system_refs] == ["start", "overview", "detail"]
        overview = scene._transforms_between(downscaled, "overview")
        assert overview is not None and overview[0].unbound() == IdentityTransform()
        assert scene._transforms_between("detail", "start") is not None

    def test_entry_points_after_dropping_axes_are_only_reachable_in_one_direction(self):
        volume = _tile(z=2, y=4, x=4)
        projection = _tile(y=4, x=4)

        scene = Scene.from_multiscale_chain(
            start=volume, steps=[ChainStep(ProjectionTo("yx"), projection, coordinate_system="projection")]
        )

        assert scene._transforms_between("start", "projection") is not None, "volume data can be shown in projection"
        assert scene._transforms_between("projection", "start") is None, "but not the other way round"

    def test_can_chain_other_coordinate_systems_of_multiscales(self):
        volume = _tile(z=2, y=4, x=4).with_coordinate_system("preview", reached_by=ProjectionTo("yx"))
        flat = _tile(y=4, x=4)
        result = _tile(y=4, x=4).with_coordinate_system("shifted", reached_by=Translation(y=1.0, x=1.0))

        scene = Scene.from_multiscale_chain(
            start=(volume, "preview"), steps=[(None, flat), (None, (result, "shifted"))], start_coordinate_system=None
        )

        assert scene._transforms_between((volume, "preview"), flat) is not None
        assert scene._transforms_between(flat, (result, "shifted")) is not None
        assert scene._transforms_between(volume, flat) is None, "start's intrinsic space isn't connected"

    def test_accepts_one_shot_iterables(self):
        raw, downscaled, cropped = self._workflow()
        steps = [(Translation(y=1.0, x=1.0), downscaled), (Translation(y=2.0, x=2.0), cropped)]

        from_generator = Scene.from_multiscale_chain(start=raw, steps=(step for step in steps))

        assert from_generator == Scene.from_multiscale_chain(start=raw, steps=steps)

    @pytest.mark.parametrize("steps_index", [0, 1])
    def test_rejects_axes_mismatch_between_relation_and_result(self, steps_index):
        raw = _tile(y=4, x=4)
        steps = [(None, _tile(y=4, x=4)), (None, _tile(y=4, x=4))]
        steps[steps_index] = (None, _tile(z=2, y=4, x=4))

        with pytest.raises(
            ValueError, match=rf"steps\[{steps_index}\].by to the axes .* produces .*, but its result has"
        ):
            Scene.from_multiscale_chain(start=raw, steps=steps)

    def test_reports_relations_that_cannot_be_applied(self):
        with pytest.raises(ValueError, match=r"steps\[0\] cannot be applied to axes \('y', 'x'\)"):
            Scene.from_multiscale_chain(start=_tile(y=2, x=2), steps=[(PermutationTo("zyx"), _tile(z=1, y=2, x=2))])

    def test_rejects_repeating_a_multiscale(self):
        raw, downscaled, cropped = self._workflow()

        with pytest.raises(ValueError, match=r"steps\[0\] has the same Multiscale as start"):
            Scene.from_multiscale_chain(start=raw, steps=[(None, raw)])
        with pytest.raises(ValueError, match=r"steps\[2\] has the same Multiscale as steps\[0\]"):
            Scene.from_multiscale_chain(start=raw, steps=[(None, downscaled), (None, cropped), (None, downscaled)])

    def test_rejects_repeating_a_multiscale_via_another_of_its_systems(self):
        volume = _tile(z=2, y=4, x=4).with_coordinate_system("preview", reached_by=ProjectionTo("yx"))

        with pytest.raises(ValueError, match="same Multiscale as start"):
            Scene.from_multiscale_chain(start=volume, steps=[(ProjectionTo("yx"), (volume, "preview"))])

    def test_accepts_equal_but_distinct_multiscales(self):
        scene = Scene.from_multiscale_chain(start=_tile(y=2, x=2), steps=[(None, _tile(y=2, x=2))])

        assert len(scene._internal_graph.transforms) == 2

    def test_rejects_invalid_or_repeated_coordinate_system_names(self):
        raw, downscaled, cropped = self._workflow()

        with pytest.raises(ValueError, match="needs a name"):
            Scene.from_multiscale_chain(start=raw, steps=[(None, downscaled)], start_coordinate_system="")
        with pytest.raises(ValueError, match="needs a name"):
            Scene.from_multiscale_chain(start=raw, steps=[ChainStep(None, downscaled, "")])
        with pytest.raises(ValueError, match=r"steps\[0\] repeats 'start'"):
            Scene.from_multiscale_chain(start=raw, steps=[ChainStep(None, downscaled, "start")])
        with pytest.raises(ValueError, match=r"steps\[1\] repeats 'view'"):
            Scene.from_multiscale_chain(
                start=raw, steps=[ChainStep(None, downscaled, "view"), ChainStep(None, cropped, "view")]
            )

    def test_rejects_no_steps_and_malformed_steps(self):
        raw, downscaled, _ = self._workflow()

        with pytest.raises(ValueError, match="at least one step"):
            Scene.from_multiscale_chain(start=raw, steps=[])
        for entry in (downscaled, (downscaled,), (None, downscaled, "view", "extra")):
            with pytest.raises(TypeError, match="must be a ChainStep or a"):
                Scene.from_multiscale_chain(start=raw, steps=[entry])  # type: ignore[list-item]

    def test_equality_depends_on_order_of_the_chain(self):
        raw, downscaled, cropped = self._workflow()
        scene = Scene.from_multiscale_chain(start=raw, steps=[(None, downscaled), (None, cropped)])

        assert scene == Scene.from_multiscale_chain(start=raw, steps=[(None, downscaled), (None, cropped)])
        assert scene != Scene.from_multiscale_chain(start=raw, steps=[(None, cropped), (None, downscaled)])


class TestRelationBetween:
    @pytest.mark.parametrize(
        ("relation", "a_axes", "b_axes"),
        [
            pytest.param(Affine.from_factor(Factor(y=2.0, x=4.0)), dict(y=8, x=8), dict(y=4, x=2), id="scale"),
            pytest.param(Translation(y=1.5, x=-2.0), dict(y=4, x=4), dict(y=4, x=4), id="translation"),
            pytest.param(PermutationTo("xy"), dict(y=4, x=4), dict(x=4, y=4), id="permutation"),
            pytest.param(ProjectionTo("yx"), dict(z=2, y=4, x=4), dict(y=4, x=4), id="drop axis"),
            pytest.param(ProjectionTo("zyx"), dict(y=4, x=4), dict(z=1, y=4, x=4), id="insert axis"),
        ],
    )
    def test_returns_the_relation_given_to_from_multiscale_pair(self, relation, a_axes, b_axes):
        a, b = _tile(**a_axes), _tile(**b_axes)
        scene = Scene.from_multiscale_pair(a=a, b=b, a_to_b=relation)

        assert scene.relation_from(a, to=b) == relation

    def test_reverse_direction_is_the_inverse_relation(self):
        a, b = _tile(y=8, x=8), _tile(y=4, x=4)
        scene = Scene.from_multiscale_pair(
            a=a, b=b, a_to_b=[Affine.from_factor(Factor(y=2, x=2)), Translation(y=1.0, x=3.0)]
        )

        assert scene.relation_from(a, to=b) == SpatialRelationSequence(
            (Affine.from_factor(Factor(y=2, x=2)), Translation(y=1.0, x=3.0))
        )
        # Forward coordinates are x' = x / 2 - (1, 3). Backwards that is x = 2 * x' + (2, 6): Upscale, then shift by
        # (-2, -6) in the upscaled space. Canonical form always puts the scaling first.
        assert scene.relation_from(b, to=a) == SpatialRelationSequence(
            (Affine.from_factor(Factor(y=0.5, x=0.5)), Translation(y=-2.0, x=-6.0))
        )

    def test_follows_several_hops_and_returns_canonical_relation(self):
        raw, downscaled, cropped = (_tile(y=8, x=8), _tile(y=4, x=4), _tile(y=2, x=2))
        scene = Scene.from_multiscale_chain(
            start=raw,
            steps=[
                (Affine.from_factor(Factor(y=2, x=2)), downscaled),
                (Translation(y=1.0, x=1.0), cropped),
                (Translation(y=1.0, x=0.0), _tile(y=2, x=2)),
            ],
        )

        assert scene.relation_from(downscaled, to=cropped) == Translation(y=1.0, x=1.0)
        assert scene.relation_from(raw, to=cropped) == SpatialRelationSequence(
            (Affine.from_factor(Factor(y=2, x=2)), Translation(y=1.0, x=1.0))
        )

    def test_returns_relation_from_shared_coordinate_system(self):
        tile1, tile2 = _tile(y=2, x=2), _tile(y=2, x=2)
        scene = Scene.from_shared_coordinate_system(
            coordinate_system="stage", connections=[(tile1, None), (tile2, Translation(y=0.0, x=-4.0))]
        )

        assert scene.relation_from(tile2, to="stage") == Translation(y=0.0, x=-4.0)
        assert scene.relation_from("stage", to=tile2) == Translation(y=0.0, x=4.0)
        assert scene.relation_from(tile1, to=tile2) == Translation(y=0.0, x=4.0)

    def test_returns_none_for_the_same_space(self):
        a, b = _tile(y=2, x=2), _tile(y=2, x=2)
        scene = Scene.from_multiscale_chain(start=a, steps=[(None, b)])

        assert scene.relation_from(a, to="start") is None
        assert scene.relation_from(a, to=b) is None
        assert scene.relation_from(a, to=a) is None

    def test_dropped_axes_can_only_be_traced_forwards(self):
        volume, flat = _tile(z=2, y=4, x=4), _tile(y=4, x=4)
        scene = Scene.from_multiscale_pair(a=volume, b=flat, a_to_b=ProjectionTo("yx"))

        assert scene.relation_from(volume, to=flat) == ProjectionTo("yx")
        with pytest.raises(ValueError, match="no \\(invertible\\) path"):
            scene.relation_from(flat, to=volume)

    def test_selects_other_coordinate_systems_of_multiscales(self):
        volume = _tile(z=2, y=4, x=4).with_coordinate_system("preview", reached_by=ProjectionTo("yx"))
        flat = _tile(y=4, x=4)
        scene = Scene.from_multiscale_pair(a=(volume, "preview"), b=flat, a_to_b=Translation(y=1.0, x=1.0))

        assert scene.relation_from((volume, "preview"), to=flat) == Translation(y=1.0, x=1.0)

    def test_follows_connections_inside_multiscales_unless_told_not_to(self):
        volume = _tile(z=2, y=4, x=4).with_coordinate_system("preview", reached_by=ProjectionTo("yx"))
        flat = _tile(y=4, x=4)
        scene = Scene.from_multiscale_pair(a=(volume, "preview"), b=flat, a_to_b=Translation(y=1.0, x=1.0))

        assert scene.relation_from(volume, to=flat) == SpatialRelationSequence(
            (ProjectionTo("yx"), Translation(y=1.0, x=1.0))
        )
        with pytest.raises(ValueError, match="no \\(invertible\\) path"):
            scene.relation_from(volume, to=flat, include_children=False)

    def test_rejects_unrelated_multiscales(self):
        a, b = _tile(y=2, x=2), _tile(y=2, x=2)
        scene = Scene.from_shared_coordinate_system(coordinate_system="stage", connections=[(a, None)])

        with pytest.raises(ValueError, match="no \\(invertible\\) path"):
            scene.relation_from(a, to=b)

    def test_rejects_unknown_spaces(self):
        a = _tile(y=2, x=2)
        scene = Scene.from_shared_coordinate_system(coordinate_system="stage", connections=[(a, None)])

        with pytest.raises(ValueError, match="no coordinate system matching 'nowhere'"):
            scene.relation_from(a, to="nowhere")

    def test_requires_resolved_spaces_and_works_after_resolving(self):
        tiles = [_tile(y=2, x=2), _tile(y=2, x=2)]
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
        a, b = _tile(y=2, x=2), _tile(y=2, x=2)
        scene = Scene._from_graph_edges([(a, CoordinatesTransform(path="coordinates"), b)])

        with pytest.raises(CannotConvertToSpatialRelationError):
            scene.relation_from(a, to=b)


class TestMultiscales:
    def test_lists_multiscales_in_order_of_appearance(self):
        tiles = [_tile(y=2, x=2), _tile(y=3, x=3), _tile(y=4, x=4)]
        scene = Scene.from_shared_coordinate_system(
            coordinate_system="stage", connections=[(tile, None) for tile in tiles]
        )

        assert len(scene.multiscales) == 3
        assert all(found is tile for found, tile in zip(scene.multiscales, tiles))

    def test_lists_multiscales_that_look_alike_separately(self):
        tiles = [_tile(y=2, x=2), _tile(y=2, x=2)]
        assert tiles[0] == tiles[1]
        scene = Scene.from_shared_coordinate_system(
            coordinate_system="stage", connections=[(tile, None) for tile in tiles]
        )

        assert len(scene.multiscales) == 2

    def test_lists_a_multiscale_once_however_many_of_its_systems_are_connected(self):
        volume = _tile(z=2, y=3, x=4).with_coordinate_system("preview", reached_by=ProjectionTo("yx"))
        scene = Scene.from_shared_coordinate_system(
            coordinate_system="stage", connections=[(volume, ProjectionTo("yx")), ((volume, "preview"), None)]
        )

        assert len(scene.multiscales) == 1
        assert scene.multiscales[0] is volume

    def test_excludes_multiscales_that_are_not_resolved_yet(self):
        tile = _tile(y=2, x=2)
        written = Scene.from_shared_coordinate_system(
            coordinate_system="stage", connections=[(tile, None)]
        ).with_multiscale_paths({"tile": tile})
        unresolved = Scene.from_ome_zarr(written.to_ome_zarr(version="0.6"))

        assert unresolved.multiscales == ()
        assert unresolved.with_resolved({"tile": tile}).multiscales == (tile,)

    def test_full_graph_includes_each_childs_connections_once(self):
        volumes = [
            _tile(z=2, y=3, x=4).with_coordinate_system("preview", reached_by=ProjectionTo("yx")) for _ in range(2)
        ]
        scene = Scene.from_shared_coordinate_system(
            coordinate_system="stage",
            connections=[
                (volumes[0], ProjectionTo("yx")),
                ((volumes[0], "preview"), None),
                (volumes[1], ProjectionTo("yx")),
            ],
        )

        child_edges = len(scene._full_graph.transforms) - len(scene._internal_graph.transforms)

        assert child_edges == 2, "one edge inside each of the two (equal looking) Multiscales"


def test_multiscale_paths_is_empty_for_scene_without_paths():
    scene, _tile_0, _tile_1 = _two_tile_scene()

    assert scene.multiscale_paths == {}


def test_with_multiscale_paths_does_not_modify_original_scene():
    scene, tile_0, _tile_1 = _two_tile_scene()

    with_paths = scene.with_multiscale_paths({"tile_0": tile_0})

    assert with_paths is not scene
    assert scene.multiscale_paths == {}
    assert with_paths.multiscale_paths == {"tile_0": tile_0}


def test_multiscale_paths_returns_copy():
    scene, tile_0, _tile_1 = _two_tile_scene()
    with_paths = scene.with_multiscale_paths({"tile_0": tile_0})

    with_paths.multiscale_paths["other"] = tile_0

    assert with_paths.multiscale_paths == {"tile_0": tile_0}


def test_with_multiscale_paths_empty_mapping_returns_same_scene():
    scene, _tile_0, _tile_1 = _two_tile_scene()

    assert scene.with_multiscale_paths({}) is scene


def test_with_multiscale_paths_requires_mapping():
    scene, tile_0, _tile_1 = _two_tile_scene()

    with pytest.raises(TypeError, match="must be a mapping"):
        scene.with_multiscale_paths([("tile_0", tile_0)])  # type: ignore[arg-type]


def test_with_multiscale_paths_keeps_paths_not_mentioned():
    scene, tile_0, tile_1 = _two_tile_scene()
    scene = scene.with_multiscale_paths({"tile_0": tile_0})

    scene = scene.with_multiscale_paths({"tile_1": tile_1})

    assert scene.multiscale_paths == {"tile_0": tile_0, "tile_1": tile_1}


def test_with_multiscale_paths_overwrites_existing_path_of_same_instance():
    scene, tile_0, tile_1 = _two_tile_scene()
    scene = scene.with_multiscale_paths({"old": tile_0, "tile_1": tile_1})

    scene = scene.with_multiscale_paths({"new": tile_0})

    assert scene.multiscale_paths == {"tile_1": tile_1, "new": tile_0}, "the old key must not linger"


def test_with_multiscale_paths_rejects_path_of_different_instance():
    scene, tile_0, tile_1 = _two_tile_scene()
    scene = scene.with_multiscale_paths({"tile_0": tile_0})

    with pytest.raises(ValueError, match=r"Path 'tile_0' is already used by the Multiscale"):
        scene.with_multiscale_paths({"tile_0": tile_1})

    assert scene.multiscale_paths == {"tile_0": tile_0}


def test_with_multiscale_paths_allows_swapping_paths():
    scene, tile_0, tile_1 = _two_tile_scene()
    scene = scene.with_multiscale_paths({"a": tile_0, "b": tile_1})

    swapped = scene.with_multiscale_paths({"a": tile_1, "b": tile_0})

    assert swapped.multiscale_paths == {"a": tile_1, "b": tile_0}


def test_with_multiscale_paths_allows_moving_into_path_vacated_in_same_call():
    scene, tile_0, tile_1 = _two_tile_scene()
    scene = scene.with_multiscale_paths({"a": tile_0, "b": tile_1})

    moved = scene.with_multiscale_paths({"c": tile_0, "a": tile_1})

    assert moved.multiscale_paths == {"c": tile_0, "a": tile_1}


def test_with_multiscale_paths_reports_copy_as_copy_even_if_instance_has_this_path():
    scene, tile_0, _tile_1 = _two_tile_scene()
    scene = scene.with_multiscale_paths({"tile_0": tile_0})

    with pytest.raises(ValueError, match="not the same instance"):
        scene.with_multiscale_paths({"tile_0": tile_0.copy()})


def test_with_multiscale_paths_overrides_paths_recorded_by_with_resolved():
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


def test_manually_built_scene_can_be_written_after_with_multiscale_paths():
    scene, tile_0, tile_1 = _two_tile_scene()
    scene = scene.with_multiscale_paths({"tile_0": tile_0, "tile_1": tile_1})

    attrs = OmeZarrGroup.from_single(scene).to_attrs(version="0.6")

    (transform,) = attrs["ome"]["scene"]["coordinateTransformations"]
    assert transform["input"]["path"] == "tile_0"
    assert transform["output"]["path"] == "tile_1"


def test_scene_with_multiscale_paths_roundtrips_through_ome_zarr():
    scene, tile_0, tile_1 = _two_tile_scene()
    scene = scene.with_multiscale_paths({"tile_0": tile_0, "tile_1": tile_1})
    attrs = OmeZarrGroup.from_single(scene).to_attrs(version="0.6")

    read = OmeZarrGroup.from_attrs(attrs, shape_source="singletons").scenes[0]

    assert read.unresolved_paths == ["tile_0", "tile_1"]
    assert read.with_resolved({"tile_0": tile_0, "tile_1": tile_1}).multiscale_paths == {
        "tile_0": tile_0,
        "tile_1": tile_1,
    }


def test_to_ome_zarr_requires_paths_for_all_graph_multiscales():
    scene, _tile_0, _tile_1 = _two_tile_scene()

    with pytest.raises(ValueError, match=r"no path is known for 2 of its Multiscales") as excinfo:
        scene.to_ome_zarr(version="0.6")

    assert "with_multiscale_paths" in str(excinfo.value)


def test_to_ome_zarr_reports_only_multiscales_without_path():
    scene, tile_0, _tile_1 = _two_tile_scene()
    scene = scene.with_multiscale_paths({"tile_0": tile_0})

    with pytest.raises(ValueError, match=r"no path is known for 1 of its Multiscales"):
        scene.to_ome_zarr(version="0.6")


def test_ome_zarr_group_from_single_reports_missing_multiscale_paths():
    scene, _tile_0, _tile_1 = _two_tile_scene()

    with pytest.raises(ValueError, match=r"with_multiscale_paths"):
        OmeZarrGroup.from_single(scene)


def test_to_ome_zarr_does_not_require_paths_for_coordinate_systems():
    world = CoordinateSystem.fromkeys("yx")._as_ref("world")
    other = CoordinateSystem.fromkeys("yx")._as_ref("other")
    scene = Scene._from_graph_edges([(world, TranslationTransform((1, 1)), other)])

    assert scene.to_ome_zarr(version="0.6")["coordinateTransformations"]
