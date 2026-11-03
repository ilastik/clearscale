import pytest

from clearscale import Multiscale, Scale, Scene, Shape, Translation, ProjectionTo
from clearscale._transforms import (
    CoordinateSystem,
    TranslationTransform,
    TransformGraph,
    _UnresolvedRef,
    IdentityTransform,
    ScaleTransform,
    AffineTransform,
    FileRef,
    ProjectAxisTransform,
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
        original = Scene._from_graph_edges([(tile, IdentityTransform(), stage)])
        read_back = Scene.from_ome_zarr(original.to_ome_zarr(version="0.6", multiscales_by_path={"tile": tile}))

        assert read_back.multiscales == ()
        assert read_back.with_resolved({"tile": tile}).multiscales == (tile,)
