import pytest

from clearscale import (
    Affine,
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
)
from clearscale.ome_zarr import Axis as OmeZarrAxis
from clearscale._transforms import (
    CoordinateSystem,
    TranslationTransform,
    TransformGraph,
    _UnresolvedRef,
    IdentityTransform,
    ScaleTransform,
    AffineTransform,
    NodeRef,
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


def test_scene_from_tiles_translations_empty():
    scene = Scene.from_tiles_translations([])

    assert scene._transforms_between("world", "world") is None
    assert scene._internal_graph.transforms == ()


def test_scene_from_tiles_translations_multiple_multiscales():
    ms1 = _multiscale(z=2, y=3, x=4)
    ms2 = _multiscale(z=2, y=3, x=4)
    ms3 = _multiscale(z=2, y=3, x=4)

    t1 = Translation(z=0, y=0, x=0)
    t2 = Translation(z=0, y=512, x=0)
    t3 = Translation(z=0, y=0, x=512)

    scene = Scene.from_tiles_translations([(ms1, t1), (ms2, t2), (ms3, t3)])

    p1 = scene._transforms_between(ms1, "world")
    p2 = scene._transforms_between(ms2, "world")
    p3 = scene._transforms_between(ms3, "world")

    assert p1
    assert p2
    assert p3
    assert len(p1) == len(p2) == len(p3) == 1

    world_ref = p1[0].target
    assert world_ref != ms1._intrinsic_ref, "central reference system should be copy, not the exact object"
    assert isinstance(world_ref, NodeRef), "for pyright"
    assert world_ref.owner == ms1._intrinsic_ref.owner, "central reference system should be copy"
    assert world_ref.owner is not ms1._intrinsic_ref.owner, "central reference system should be copy, not same instance"
    assert p2[0].target == world_ref
    assert p3[0].target == world_ref

    assert p1[0] == TranslationTransform.from_translation(t1).bound(source=ms1._as_ref_to_intrinsic(), target=world_ref)
    assert p2[0] == TranslationTransform.from_translation(t2).bound(source=ms2._as_ref_to_intrinsic(), target=world_ref)
    assert p3[0] == TranslationTransform.from_translation(t3).bound(source=ms3._as_ref_to_intrinsic(), target=world_ref)

    p13 = scene._transforms_between(ms1, ms3)
    assert p13
    assert len(p13) == 2


def test_scene_from_tiles_translations_rejects_mismatching_axes_across_multiscales():
    ms1 = _multiscale(z=2, y=3, x=4)
    ms2 = _multiscale(c=2, y=3, x=4)

    with pytest.raises(ValueError, match="identical axis keys"):
        Scene.from_tiles_translations(
            [
                (ms1, Translation.identity("zyx")),
                (ms2, Translation.identity("cyx")),
            ]
        )


def test_scene_from_tiles_translations_rejects_mismatching_axes_in_translations():
    ms1 = _multiscale(z=2, y=3, x=4)
    ms2 = _multiscale(z=2, y=3, x=4)

    with pytest.raises(ValueError, match="identical axis keys"):
        Scene.from_tiles_translations(
            [
                (ms1, Translation.identity("zyx")),
                (ms2, Translation.identity("cyx")),
            ]
        )


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
        paths = {"a": tile1, "b": tile2}
        relations = [Translation(y=1.0, x=2.0), Translation(y=-1.0, x=0.5)]

        from_named = Scene.from_shared_coordinate_system(
            coordinate_system="stage", connections=[Connection(tile1, relations[0]), Connection(tile2, relations[1])]
        )
        from_plain = Scene.from_shared_coordinate_system(
            coordinate_system="stage", connections=iter([(tile1, relations[0]), (tile2, relations[1])])
        )

        assert from_named.to_ome_zarr(version="0.6", multiscales_by_path=paths) == from_plain.to_ome_zarr(
            version="0.6", multiscales_by_path=paths
        )

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
