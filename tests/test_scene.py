from unittest import mock

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
    ProjectAxisTransform,
)


def _multiscale(unit=None, **shape: int) -> Multiscale:
    return Multiscale({"s0": Scale(Shape(**shape), unit=unit)})


def test_from_graph_edges_binds_between_multiscales():
    source = _multiscale(z=2, y=3, x=4)
    target = _multiscale(z=2, y=3, x=4)

    scene = Scene._from_graph_edges(
        [(source, ScaleTransform((2, 2, 2)), target)], multiscale_paths={"source": source, "target": target}
    )

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
        ],
        multiscale_paths={"image": image},
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
        ],
        multiscale_paths={"moving": moving, "fixed": fixed},
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
        Scene._from_graph_edges([("some_string", ScaleTransform((2, 2, 2)), world)], multiscale_paths={})  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Use CoordinateSystem._as_ref"):
        Scene._from_graph_edges([(world, ScaleTransform((2, 2, 2)), image)], multiscale_paths={"image": image})  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Use CoordinateSystem._as_ref"):
        Scene._from_graph_edges([(image, ScaleTransform((2, 2, 2)), world)], multiscale_paths={"image": image})  # type: ignore[arg-type]


def test_transforms_between_in_empty_scene():
    scene = Scene._from_graph_edges([], multiscale_paths={})

    assert scene._transforms_between("world", "world") is None
    assert scene._internal_graph.transforms == ()


def test_transforms_between_accepts_path_addressed_unresolved_refs():
    world = CoordinateSystem.fromkeys("yx")._as_ref("world")
    transform = TranslationTransform(
        translation=(1, 2),
        source=_UnresolvedRef(name="physical", path="tile_0"),
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
    scene = Scene(TransformGraph([scene_transform], system_refs=(world,)), _multiscale_paths={"multiscale": multiscale})

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
    scene = Scene(TransformGraph([scene_transform], system_refs=(world,)), _multiscale_paths={"multiscale": multiscale})

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
    scene = Scene._from_graph_edges([], multiscale_paths={})

    with pytest.raises(TypeError, match=r"must be tuple\(multiscale, system_name\)"):
        scene._transforms_between(key, "world")  # type: ignore[arg-type]


def test_multiscale_paths_is_read_only():
    a, b = _multiscale(y=2, x=3), _multiscale(y=2, x=3)
    scene = Scene.from_multiscale_pair(a=a, b=b, a_to_b=None, a_path="tiles/a", b_path="tiles/b")
    with pytest.raises(TypeError):
        scene.multiscale_paths["other"] = a  # type: ignore[index]
    assert scene.multiscale_paths == {"tiles/a": a, "tiles/b": b}
    assert scene.multiscale_paths["tiles/a"] is a
    assert scene.multiscale_paths["tiles/b"] is b


@pytest.mark.parametrize("resolve_separately", [False, True])
def test_with_resolved_rejects_same_multiscale_for_two_paths(resolve_separately):
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

    if resolve_separately:
        scene = scene.with_resolved({"tile_0": ms})  # Just one should work
    with pytest.raises(ValueError, match="Received two paths for one instance: 'tile_0' and 'tile_1'"):
        scene.with_resolved({"tile_0": ms, "tile_1": ms})


class TestMultiscalesProperty:
    def test_lists_multiscales_in_order_of_appearance(self):
        tiles = [_multiscale(y=2, x=2), _multiscale(y=3, x=3), _multiscale(y=4, x=4)]
        stage = CoordinateSystem.fromkeys("yx")._as_ref("stage")

        scene = Scene._from_graph_edges(
            [(tile, IdentityTransform(), stage) for tile in tiles],
            multiscale_paths={f"tile_{i}": tile for i, tile in enumerate(tiles)},
        )

        assert len(scene.multiscales) == 3
        assert all(found is tile for found, tile in zip(scene.multiscales, tiles))

    def test_lists_multiscales_that_look_alike_separately(self):
        tiles = [_multiscale(y=2, x=2), _multiscale(y=2, x=2)]
        assert tiles[0] == tiles[1]
        stage = CoordinateSystem.fromkeys("yx")._as_ref("stage")

        scene = Scene._from_graph_edges(
            [(tile, IdentityTransform(), stage) for tile in tiles],
            multiscale_paths={f"tile_{i}": tile for i, tile in enumerate(tiles)},
        )

        assert len(scene.multiscales) == 2

    def test_does_not_duplicate_multiply_referenced_instances(self):
        volume = _multiscale(z=2, y=3, x=4).with_coordinate_system("preview", reached_by=ProjectionTo("yx"))
        stage = CoordinateSystem.fromkeys("yx")._as_ref("stage")

        scene = Scene._from_graph_edges(
            [
                (volume._as_ref_to_intrinsic(), ProjectAxisTransform(drops=(0,)), stage),
                (volume._as_ref("preview"), IdentityTransform(), stage),
            ],
            multiscale_paths={"volume": volume},
        )

        assert len(scene.multiscales) == 1
        assert scene.multiscales[0] is volume

    def test_excludes_unresolved_multiscales(self):
        tile = _multiscale(y=2, x=2)
        stage = CoordinateSystem.fromkeys("yx")._as_ref("stage")
        original = Scene._from_graph_edges([(tile, IdentityTransform(), stage)], multiscale_paths={"tile": tile})
        read_back = Scene.from_ome_zarr(original.to_ome_zarr(version="0.6"))

        assert read_back.multiscales == ()
        assert read_back.with_resolved({"tile": tile}).multiscales == (tile,)


@pytest.fixture()
def unit_validation_mock(monkeypatch):
    import clearscale._scene as module

    validator = mock.Mock(wraps=module.require_matching_units_or_affine_relation)
    monkeypatch.setattr(module, "require_matching_units_or_affine_relation", validator)
    return validator


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
                Scene.Connection(tile1, to_shared=Translation(y=1.0, x=2.0), path="tile1"),
                (tile2, Affine.from_factor(Factor(y=0.5, x=0.5)), "tile2"),
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

        scene = Scene.from_shared_coordinate_system(coordinate_system="stage", connections=[(tile, None, "tile")])

        path = scene._transforms_between(tile, "stage")
        assert path is not None and path[0].unbound() == IdentityTransform()

    def test_connection_and_plain_tuples_are_equivalent(self):
        tile1 = _multiscale(y=2, x=3)
        tile2 = _multiscale(y=2, x=3)
        relations = [Translation(y=1.0, x=2.0), Translation(y=-1.0, x=0.5)]

        from_named = Scene.from_shared_coordinate_system(
            coordinate_system="stage",
            connections=[
                Scene.Connection(tile1, relations[0], path="tile1"),
                Scene.Connection(tile2, relations[1], path="tile2"),
            ],
        )
        from_plain = Scene.from_shared_coordinate_system(
            coordinate_system="stage", connections=[(tile1, relations[0], "tile1"), (tile2, relations[1], "tile2")]
        )

        assert from_named == from_plain

    def test_accepts_one_shot_iterables(self):
        tile1 = _multiscale(y=2, x=3)
        tile2 = _multiscale(y=2, x=3)
        connections = [(tile1, None, "tile1"), (tile2, Translation(y=1.0, x=2.0), "tile2")]

        from_generator = Scene.from_shared_coordinate_system(
            coordinate_system="stage", connections=(connection for connection in connections)
        )

        assert from_generator == Scene.from_shared_coordinate_system(coordinate_system="stage", connections=connections)

    def test_axis_properties_are_inherited_from_the_connected_multiscales(self):
        tile1 = _multiscale(unit=Unit(y="micrometer", x="micrometer"), y=2, x=3)
        tile2 = _multiscale(y=2, x=3)  # blank unit, compatible

        scene = Scene.from_shared_coordinate_system(
            coordinate_system="stage", connections=[(tile1, None, "tile1"), (tile2, Translation(y=1.0, x=0.0), "tile2")]
        )

        assert self._system_ome_axes(scene, "stage") == {
            "y": ome_zarr.Axis(name="y", type="space", unit="micrometer"),
            "x": ome_zarr.Axis(name="x", type="space", unit="micrometer"),
        }

    def test_accepts_mismatching_units_with_explicit_target_unit_and_converting_affine_for_mismatches(self):
        first_unit = Unit(y="micrometer", x="um")
        converted_unit = Unit(y="millimeter", x="mm")
        tile1 = _multiscale(unit=first_unit, y=2, x=3)
        tile2 = _multiscale(unit=converted_unit, y=2, x=3)

        scene = Scene.from_shared_coordinate_system(
            coordinate_system="stage",
            connections=[
                (tile1, None, "tile1"),
                (tile2, Affine.from_unit_conversion(converted_unit, first_unit), "tile2"),
            ],
            unit=first_unit,
        )

        assert self._system_ome_axes(scene, "stage") == {
            "y": ome_zarr.Axis(name="y", type="space", unit="micrometer"),
            "x": ome_zarr.Axis(name="x", type="space", unit="um"),
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
                    "tile",
                )
            ],
            unit=Unit(y="millimeter", x="millimeter"),
        )

        assert self._system_ome_axes(scene, "stage").get_unit() == Unit(y="millimeter", x="millimeter")

    def test_axes_follow_the_relations(self):
        tile = _multiscale(y=2, x=3)

        scene = Scene.from_shared_coordinate_system(
            coordinate_system="stage",
            connections=[(tile, [ProjectionTo("zyx"), Translation(z=0.0, y=2.0, x=2.0)], "tile")],
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
            connections=[((volume, "preview"), Translation(y=1.0, x=1.0), "volume"), (flat, None, "flat")],
        )

        path = scene._transforms_between((volume, "preview"), "stage")
        assert path is not None and path[0].source == volume._as_ref("preview")
        assert self._system_ome_axes(scene, "stage").axes == ("y", "x")
        assert scene.multiscale_paths["volume"] is volume
        assert scene.multiscale_paths["flat"] is flat

    def test_same_multiscale_can_connect_via_different_systems(self):
        volume = _multiscale(z=2, y=3, x=4).with_coordinate_system("preview", reached_by=ProjectionTo("yx"))

        scene = Scene.from_shared_coordinate_system(
            coordinate_system="stage",
            connections=[(volume, ProjectionTo("yx"), "volume"), ((volume, "preview"), None, "volume")],
        )

        assert len(scene._internal_graph.transforms) == 2

    def test_rejects_same_coordinate_system_twice(self):
        tile = _multiscale(y=2, x=3)

        with pytest.raises(
            ValueError, match=r"connections\[1\] selects the same coordinate system as connections\[0\]"
        ):
            Scene.from_shared_coordinate_system(
                coordinate_system="stage", connections=[(tile, None, "tile"), (tile, None, "tile")]
            )

    def test_rejects_connections_reaching_different_axes(self):
        volume = _multiscale(z=2, y=3, x=4)
        flat = _multiscale(y=3, x=4)

        with pytest.raises(ValueError, match=r"connections\[0\] reaches .*z.*connections\[1\] reaches .*y.*x"):
            Scene.from_shared_coordinate_system(
                coordinate_system="stage", connections=[(volume, None, "volume"), (flat, None, "flat")]
            )

    def test_rejects_conflicting_axis_properties(self):
        um = _multiscale(unit=Unit(y="micrometer", x="micrometer"), y=2, x=3)
        mm = _multiscale(unit=Unit(y="millimeter", x="millimeter"), y=2, x=3)

        with pytest.raises(ValueError, match="Conflicting 'unit'"):
            Scene.from_shared_coordinate_system(
                coordinate_system="stage", connections=[(um, None, "um"), (mm, None, "mm")]
            )

    def test_rejects_conflicting_units_even_with_converting_affine_but_no_provided_unit(self):
        """
        Without an explicit target unit for the shared system, we would need to check whether each mismatching source
        has an Affine that might convert its units to the central consensus.
        Finding the central consensus in itself could be impossible: if *all* multiscales relate with a converting Affine,
        we would have no way to know what the conversion target unit was supposed to be.
        The conversion target is only known to the Affine.from_unit_conversion call (not the resulting Affine).
        """
        first_unit = Unit(y="micrometer", x="um")
        converted_unit = Unit(y="millimeter", x="mm")
        um = _multiscale(unit=first_unit, y=2, x=3)
        mm = _multiscale(unit=converted_unit, y=2, x=3)

        with pytest.raises(
            ValueError,
            match=r"Conflicting 'unit'.*Specify the shared coordinate system's conflicting axis properties explicitly using `ome_zarr_axes=` \(or `unit=` for units\)",
        ):
            Scene.from_shared_coordinate_system(
                coordinate_system="stage",
                connections=[(um, None, "um"), (mm, Affine.from_unit_conversion(converted_unit, first_unit), "mm")],
            )

    @pytest.mark.parametrize("properties", [{"unit": {"x": "nm"}}, {"ome_zarr_axes": {"x": ome_zarr.Axis(unit="nm")}}])
    def test_explicit_target_unit_does_not_hide_missing_conversion(self, properties):
        converted = _multiscale(unit=Unit(x="um"), y=4, x=4)
        unconverted = _multiscale(unit=Unit(x="um"), y=4, x=4)
        with pytest.raises(ValueError, match=r"connections\[1\] and coordinate system 'stage'.*no relation converts"):
            Scene.from_shared_coordinate_system(
                coordinate_system="stage",
                connections=[
                    (converted, Affine.from_factor({"y": 1, "x": 1000}), "converted"),
                    (unconverted, None, "unconverted"),
                ],
                **properties,
            )

    def test_validates_each_selected_source_against_final_merged_properties(self, unit_validation_mock):
        blank = _multiscale(y=4, x=4)
        specified = _multiscale(unit=Unit(x="um"), y=4, x=4).with_coordinate_system(
            "selected", reached_by=Affine.from_factor({"y": 1, "x": 1000}), unit={"x": "nm"}
        )
        steps = [Translation(y=0, x=1), Translation(y=0, x=2)]
        scene = Scene.from_shared_coordinate_system(
            coordinate_system="stage",
            connections=[(blank, None, "blank"), ((specified, "selected"), steps, "specified")],
        )
        shared = ome_zarr.Axes(self._system_ome_axes(scene, "stage"))
        assert shared["x"].unit == "nm"
        assert unit_validation_mock.call_args_list == [
            mock.call(
                blank.ome_zarr_axes,
                shared,
                None,
                source_label="connections[0]",
                target_label="coordinate system 'stage'",
            ),
            mock.call(
                specified.coordinate_system_ome_zarr_axes("selected"),
                shared,
                SpatialRelationSequence(tuple(steps)),
                source_label="connections[1]",
                target_label="coordinate system 'stage'",
            ),
        ]

    def test_reports_relations_that_cannot_be_applied(self):
        tile = _multiscale(y=2, x=3)

        with pytest.raises(ValueError, match=r"connections\[0\] cannot be applied to axes \('y', 'x'\)"):
            Scene.from_shared_coordinate_system(
                coordinate_system="stage", connections=[(tile, PermutationTo("zyx"), "tile")]
            )

    def test_rejects_unknown_system_name(self):
        tile = _multiscale(y=2, x=3)

        with pytest.raises(ValueError, match="does not specify a coordinate system named: 'nope'"):
            Scene.from_shared_coordinate_system(coordinate_system="stage", connections=[((tile, "nope"), None, "tile")])

    @pytest.mark.parametrize("name", ["", None, 5])
    def test_rejects_invalid_shared_system_name(self, name):
        with pytest.raises(ValueError, match="needs a name"):
            Scene.from_shared_coordinate_system(
                coordinate_system=name, connections=[(_multiscale(y=2, x=3), None, "tile")]  # type: ignore[arg-type]
            )

    def test_rejects_no_connections(self):
        with pytest.raises(ValueError, match="at least one connection"):
            Scene.from_shared_coordinate_system(coordinate_system="stage", connections=[])

    @pytest.mark.parametrize(
        "entry",
        [
            _multiscale(y=2, x=3),
            (_multiscale(y=2, x=3),),
            (_multiscale(y=2, x=3), None),
            (_multiscale(y=2, x=3), None, "tile", None),
        ],
    )
    def test_rejects_malformed_connections(self, entry):
        with pytest.raises(TypeError, match="must be a Scene.Connection or a"):
            Scene.from_shared_coordinate_system(coordinate_system="stage", connections=[entry])  # type: ignore[list-item]

    @pytest.mark.parametrize("named_source", [False, True])
    def test_accepts_different_systems_of_one_instance_at_same_path(self, named_source):
        ms = _multiscale(y=2, x=3).with_coordinate_system("first").with_coordinate_system("second")
        source = (ms, "first") if named_source else ms
        scene = Scene.from_shared_coordinate_system(
            coordinate_system="stage", connections=[(source, None, "tile"), ((ms, "second"), None, "tile")]
        )
        assert scene.multiscale_paths == {"tile": ms}
        assert len(scene._internal_graph.transforms) == 2

    @pytest.mark.parametrize("named_source", [False, True])
    def test_rejects_same_instance_at_two_paths(self, named_source):
        ms = _multiscale(y=2, x=3).with_coordinate_system("first").with_coordinate_system("second")
        source = (ms, "first") if named_source else ms
        with pytest.raises(ValueError, match="Received two paths for one instance: 'a' and 'b'"):
            Scene.from_shared_coordinate_system(
                coordinate_system="stage", connections=[(source, None, "a"), ((ms, "second"), None, "b")]
            )

    @pytest.mark.parametrize("equal_multiscales", [False, True])
    def test_rejects_same_path_for_different_instances(self, equal_multiscales):
        a = _multiscale(y=2, x=3)
        b = a.copy() if equal_multiscales else _multiscale(y=4, x=5)
        with pytest.raises(ValueError, match="Path 'tile' is already used by another Multiscale"):
            Scene.from_shared_coordinate_system(
                coordinate_system="stage", connections=[(a, None, "tile"), (b, None, "tile")]
            )

    @pytest.mark.parametrize("path", ["", None, 3, []])
    def test_rejects_non_string_or_empty_paths(self, path):
        with pytest.raises(ValueError, match="non-empty strings"):
            Scene.from_shared_coordinate_system(
                coordinate_system="stage", connections=[(_multiscale(y=2, x=3), None, path)]
            )


class TestSceneEquality:
    def test_paths_do_not_affect_equality(self):
        a, b = _multiscale(y=2, x=2), _multiscale(y=3, x=3)
        first = Scene.from_multiscale_pair(a=a, b=b, a_to_b=None, a_path="a", b_path="b")
        second = Scene.from_multiscale_pair(a=a, b=b, a_to_b=None, a_path="tiles/0", b_path="tiles/1")
        assert first == second
        assert hash(first) == hash(second)

    @staticmethod
    def _stitched(*tiles: Multiscale, name: str = "stage", **kwargs) -> Scene:
        return Scene.from_shared_coordinate_system(
            coordinate_system=name,
            connections=[(tile, Translation(y=0.0, x=4.0 * i), f"tile_{i}") for i, tile in enumerate(tiles)],
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
            connections=[
                (tiles[0], Translation(y=0.0, x=0.0), "tile_0"),
                (tiles[1], Translation(y=0.0, x=5.0), "tile_1"),
            ],
        )
        assert scene != shifted
        assert scene != self._stitched(*tiles, name="world")
        assert scene != self._stitched(*tiles, unit=Unit(y="micrometer", x="micrometer"))

    def test_differs_by_multiscale_content(self):
        assert self._stitched(_multiscale(y=2, x=2)) != self._stitched(_multiscale(y=3, x=2))

    def test_differs_by_number_of_equal_multiscales(self):
        connect = lambda n: Scene.from_shared_coordinate_system(
            coordinate_system="stage", connections=[(_multiscale(y=2, x=2), None, f"tile_{i}") for i in range(n)]
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

        assert connect((a, None, "a"), (b, Translation(y=1.0, x=0.0), "b")) == connect(
            (b, Translation(y=1.0, x=0.0), "b"), (a, None, "a")
        )

    def test_not_equal_to_other_types(self):
        assert self._stitched(_multiscale(y=2, x=2)) != "stage"

    def test_survives_ome_zarr_round_trip(self):
        tiles = [_multiscale(unit=Unit(y="micrometer", x="micrometer"), y=2, x=2), _multiscale(y=2, x=2)]
        paths = {"tile_0": tiles[0], "tile_1": tiles[1]}
        scene = self._stitched(*tiles)

        read_back = Scene.from_ome_zarr(scene.to_ome_zarr(version="0.6"))

        assert read_back != scene, "unresolved: refs are placeholders for the Multiscales"
        assert read_back.with_resolved(paths) == scene


class TestFromMultiscalePair:
    def test_connects_a_to_b(self):
        raw = _multiscale(y=4, x=4)
        atlas = _multiscale(y=4, x=4)

        scene = Scene.from_multiscale_pair(a=raw, b=atlas, a_to_b=Translation(y=1.0, x=2.0), a_path="a", b_path="b")

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

        scene = Scene.from_multiscale_pair(a=raw, b=atlas, a_to_b=None, a_path="a", b_path="b")

        path = scene._transforms_between(raw, atlas)
        assert path is not None and path[0].unbound() == IdentityTransform()

    def test_relation_chains_are_applied_in_order(self):
        volume = _multiscale(z=2, y=4, x=4)
        flat = _multiscale(y=2, x=2)

        scene = Scene.from_multiscale_pair(
            a=volume,
            b=flat,
            a_to_b=[ProjectionTo("yx"), Affine.from_factor(Factor(y=0.5, x=0.5)), Translation(y=1.0, x=1.0)],
            a_path="a",
            b_path="b",
        )

        path = scene._transforms_between(volume, flat)
        assert path is not None and len(path) == 1

    def test_dropping_axes_cannot_be_traversed_backwards(self):
        volume = _multiscale(z=2, y=4, x=4)
        flat = _multiscale(y=4, x=4)

        scene = Scene.from_multiscale_pair(a=volume, b=flat, a_to_b=ProjectionTo("yx"), a_path="a", b_path="b")

        assert scene._transforms_between(volume, flat) is not None
        assert scene._transforms_between(flat, volume) is None

    def test_inserting_axes_can_be_traversed_backwards(self):
        flat = _multiscale(y=4, x=4)
        volume = _multiscale(z=2, y=4, x=4)

        scene = Scene.from_multiscale_pair(a=flat, b=volume, a_to_b=ProjectionTo("zyx"), a_path="a", b_path="b")

        assert scene._transforms_between(flat, volume) is not None
        assert scene._transforms_between(volume, flat) is not None

    def test_can_relate_other_coordinate_systems_of_the_multiscales(self):
        volume = _multiscale(z=2, y=3, x=4).with_coordinate_system("preview", reached_by=ProjectionTo("yx"))
        flat = _multiscale(y=3, x=4).with_coordinate_system("shifted", reached_by=Translation(y=1.0, x=1.0))

        scene = Scene.from_multiscale_pair(
            a=(volume, "preview"), b=(flat, "shifted"), a_to_b=None, a_path="a", b_path="b"
        )

        path = scene._transforms_between((volume, "preview"), (flat, "shifted"))
        assert path is not None and len(path) == 1
        assert path[0].source == volume._as_ref("preview") and path[0].target == flat._as_ref("shifted")
        assert scene.multiscale_paths["a"] is volume
        assert scene.multiscale_paths["b"] is flat
        assert scene._transforms_between(volume, flat) is None, "intrinsic systems are not connected"
        children = scene._transforms_between(volume, flat, include_children=True)
        assert children is not None and len(children) == 3, "connected through the multiscales' own graphs"

    def test_rejects_relation_not_producing_axes_of_b(self):
        volume = _multiscale(z=2, y=4, x=4)
        flat = _multiscale(y=4, x=4)

        with pytest.raises(ValueError, match=r"produces \('z', 'y', 'x'\), but `b` has axes \('y', 'x'\)"):
            Scene.from_multiscale_pair(a=volume, b=flat, a_to_b=None, a_path="a", b_path="b")

    def test_rejects_unit_mismatch_without_conversion(self):
        a = _multiscale(unit=Unit(x="um"), y=4, x=4)
        b = _multiscale(unit=Unit(x="nm"), y=4, x=4)
        with pytest.raises(ValueError, match=r"units of `a` and `b` differ.*no relation converts"):
            Scene.from_multiscale_pair(a=a, b=b, a_to_b=None, a_path="a", b_path="b")

    def test_forwards_selected_systems_and_normalized_relation_to_unit_validator(self, unit_validation_mock):
        a = _multiscale(unit=Unit(x="um"), y=4, x=4).with_coordinate_system(
            "selected", reached_by=Affine.from_factor({"y": 1, "x": 1000}), unit={"x": "nm"}
        )
        b = _multiscale(unit=Unit(x="um"), y=4, x=4).with_coordinate_system(
            "selected", reached_by=Affine.from_factor({"y": 1, "x": 0.001}), unit={"x": "mm"}
        )
        steps = [Translation(y=0, x=1), Affine.from_factor({"y": 1, "x": 0.000001})]
        Scene.from_multiscale_pair(a=(a, "selected"), b=(b, "selected"), a_to_b=steps, a_path="a", b_path="b")
        unit_validation_mock.assert_called_once_with(
            a.coordinate_system_ome_zarr_axes("selected"),
            b.coordinate_system_ome_zarr_axes("selected"),
            SpatialRelationSequence(tuple(steps)),
            source_label="`a`",
            target_label="`b`",
        )

    def test_reports_relations_that_cannot_be_applied(self):
        with pytest.raises(ValueError, match=r"`a_to_b` cannot be applied to the axes of `a` \('y', 'x'\)"):
            Scene.from_multiscale_pair(
                a=_multiscale(y=2, x=2),
                b=_multiscale(z=2, y=2, x=2),
                a_to_b=PermutationTo("zyx"),
                a_path="a",
                b_path="b",
            )

    def test_rejects_relating_a_multiscale_to_itself(self):
        multiscale = _multiscale(z=2, y=3, x=4).with_coordinate_system("preview", reached_by=ProjectionTo("yx"))

        with pytest.raises(ValueError, match="same Multiscale"):
            Scene.from_multiscale_pair(
                a=multiscale, b=(multiscale, "preview"), a_to_b=ProjectionTo("yx"), a_path="a", b_path="b"
            )

    def test_accepts_equal_but_distinct_multiscales(self):
        scene = Scene.from_multiscale_pair(
            a=_multiscale(y=2, x=2), b=_multiscale(y=2, x=2), a_to_b=None, a_path="a", b_path="b"
        )

        assert len(scene._internal_graph.transforms) == 1

    def test_rejects_unknown_system_name_and_malformed_keys(self):
        tile = _multiscale(y=2, x=2)

        with pytest.raises(ValueError, match="does not specify a coordinate system named: 'nope'"):
            Scene.from_multiscale_pair(a=(tile, "nope"), b=_multiscale(y=2, x=2), a_to_b=None, a_path="a", b_path="b")
        with pytest.raises(TypeError, match="Expected a Multiscale"):
            Scene.from_multiscale_pair(a="stage", b=tile, a_to_b=None, a_path="a", b_path="b")  # type: ignore[arg-type]

    def test_equality_depends_on_direction_and_relation(self):
        a, b = _multiscale(y=2, x=2), _multiscale(y=3, x=3)
        scene = Scene.from_multiscale_pair(a=a, b=b, a_to_b=None, a_path="a", b_path="b")

        assert scene == Scene.from_multiscale_pair(a=a, b=b, a_to_b=None, a_path="a", b_path="b")
        assert scene != Scene.from_multiscale_pair(a=b, b=a, a_to_b=None, a_path="a", b_path="b")
        assert scene != Scene.from_multiscale_pair(a=a, b=b, a_to_b=Translation(y=1.0, x=0.0), a_path="a", b_path="b")

    @pytest.mark.parametrize("equal_multiscales", [False, True])
    def test_rejects_same_path_for_different_instances(self, equal_multiscales):
        a = _multiscale(y=2, x=3)
        b = a.copy() if equal_multiscales else _multiscale(y=4, x=5)
        with pytest.raises(ValueError, match="Path 'tile' is already used by another Multiscale"):
            Scene.from_multiscale_pair(a=a, b=b, a_to_b=None, a_path="tile", b_path="tile")

    @pytest.mark.parametrize("path", ["", None, 3, []])
    @pytest.mark.parametrize("parameter", ["a_path", "b_path"])
    def test_rejects_non_string_or_empty_paths(self, path, parameter):
        paths = {"a_path": "a", "b_path": "b", parameter: path}
        with pytest.raises(ValueError, match="non-empty strings"):
            Scene.from_multiscale_pair(a=_multiscale(y=2, x=3), b=_multiscale(y=2, x=3), a_to_b=None, **paths)


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
        scene = Scene.from_multiscale_pair(a=a, b=b, a_to_b=relation, a_path="a", b_path="b")

        assert scene.relation_from(a, to=b) == relation

    def test_reverse_direction_is_the_inverse_relation(self):
        a, b = _multiscale(y=8, x=8), _multiscale(y=4, x=4)
        scene = Scene.from_multiscale_pair(
            a=a,
            b=b,
            a_to_b=[Affine.from_factor(Factor(y=0.5, x=0.5)), Translation(y=1.0, x=3.0)],
            a_path="a",
            b_path="b",
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
            coordinate_system="stage",
            connections=[(tile1, None, "tile1"), (tile2, Translation(y=0.0, x=-4.0), "tile2")],
        )

        assert scene.relation_from(tile2, to="stage") == Translation(y=0.0, x=-4.0)
        assert scene.relation_from("stage", to=tile2) == Translation(y=0.0, x=4.0)
        assert scene.relation_from(tile1, to=tile2) == Translation(y=0.0, x=4.0)

    def test_can_select_other_coordinate_systems_of_multiscales(self):
        volume = _multiscale(z=2, y=4, x=4).with_coordinate_system("preview", reached_by=ProjectionTo("yx"))
        flat = _multiscale(y=4, x=4)
        scene = Scene.from_multiscale_pair(
            a=(volume, "preview"), b=flat, a_to_b=Translation(y=1.0, x=1.0), a_path="a", b_path="b"
        )

        assert scene.relation_from((volume, "preview"), to=flat) == Translation(y=1.0, x=1.0)
        assert scene.relation_from(volume, to=flat) == SpatialRelationSequence(
            (ProjectionTo("yx"), Translation(y=1.0, x=1.0))
        ), "should follow connections within child Multiscales"

    def test_returns_none_for_the_same_space(self):
        a, b = _multiscale(y=2, x=2), _multiscale(y=2, x=2)
        scene = Scene.from_multiscale_pair(a=a, b=b, a_to_b=None, a_path="a", b_path="b")

        assert scene.relation_from(a, to=b) is None
        assert scene.relation_from(a, to=a) is None

    def test_dropped_axes_can_only_be_traced_forwards(self):
        volume, flat = _multiscale(z=2, y=4, x=4), _multiscale(y=4, x=4)
        scene = Scene.from_multiscale_pair(a=volume, b=flat, a_to_b=ProjectionTo("yx"), a_path="a", b_path="b")

        assert scene.relation_from(volume, to=flat) == ProjectionTo("yx")
        with pytest.raises(ValueError, match="no \\(invertible\\) path"):
            scene.relation_from(flat, to=volume)

    def test_rejects_unrelated_multiscales(self):
        a, b = _multiscale(y=2, x=2), _multiscale(y=2, x=2)
        scene = Scene.from_shared_coordinate_system(coordinate_system="stage", connections=[(a, None, "a")])

        with pytest.raises(ValueError, match="no \\(invertible\\) path"):
            scene.relation_from(a, to=b)

    def test_rejects_unknown_spaces(self):
        a = _multiscale(y=2, x=2)
        scene = Scene.from_shared_coordinate_system(coordinate_system="stage", connections=[(a, None, "a")])

        with pytest.raises(ValueError, match="no coordinate system matching 'nowhere'"):
            scene.relation_from(a, to="nowhere")

    def test_requires_resolved_spaces_and_works_after_resolving(self):
        tiles = [_multiscale(y=2, x=2), _multiscale(y=2, x=2)]
        scene = Scene.from_shared_coordinate_system(
            coordinate_system="stage",
            connections=[(tiles[0], None, "tile_0"), (tiles[1], Translation(y=0.0, x=-4.0), "tile_1")],
        )
        paths = {"tile_0": tiles[0], "tile_1": tiles[1]}
        unresolved = Scene.from_ome_zarr(scene.to_ome_zarr(version="0.6"))
        name_of_unresolved_multiscale = tiles[1]._intrinsic_ref.name

        with pytest.raises(ValueError, match="not resolved yet"):
            unresolved.relation_from(name_of_unresolved_multiscale, to="stage")
        assert unresolved.with_resolved(paths).relation_from(tiles[1], to="stage") == Translation(y=0.0, x=-4.0)

    def test_reports_connections_that_are_no_spatial_relation(self):
        a, b = _multiscale(y=2, x=2), _multiscale(y=2, x=2)
        scene = Scene._from_graph_edges(
            [(a, CoordinatesTransform(path="coordinates"), b)], multiscale_paths={"a": a, "b": b}
        )

        with pytest.raises(CannotConvertToSpatialRelationError):
            scene.relation_from(a, to=b)
