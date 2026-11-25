from typing import Any, Dict
import pytest

from clearscale import Affine, PermutationTo, Scene, Multiscale, Scale, Shape
from clearscale._spatial_relations import SpatialRelationSequence
from clearscale._transforms import CoordinateSystem, TranslationTransform, _UnresolvedRef, TransformGraph

from tests.ome_zarr.scene_examples import (
    all_invalid_scene_examples,
    all_supported_scene_examples,
    all_external_payload_scene_examples,
    scene_registration,
    scene_stitching,
)


def _multiscale():
    ms = Multiscale({"s0": Scale(Shape(y=2, x=3))})
    object.__setattr__(ms._intrinsic_ref, "name", "physical")
    return ms


def test_stitching_example():
    scene = Scene.from_ome_zarr(scene_stitching())
    assert scene.unresolved_multiscale_paths == ("tile_0", "tile_1", "tile_2", "tile_3")


def test_registration_example():
    with pytest.raises(ValueError, match="Transforms with external data are currently unsupported"):
        Scene.from_ome_zarr(scene_registration())


@pytest.mark.parametrize("meta", all_supported_scene_examples())
def test_scene_public_examples_roundtrip(meta):
    # https://ngff.openmicroscopy.org/specifications/dev/examples/transformations/transformations.html
    roundtrip: Dict[str, Any] = Scene.from_ome_zarr(meta).to_ome_zarr(version="0.6")
    assert "coordinateTransformations" in roundtrip
    assert "coordinateTransformations" in meta
    assert len(roundtrip["coordinateTransformations"]) == len(meta["coordinateTransformations"])
    for actual, expected in zip(roundtrip["coordinateTransformations"], meta["coordinateTransformations"]):
        # compare individually for easier debugging
        assert actual == expected
    # Full check because zip can miss one containing more items than the other
    assert roundtrip == meta


def test_relation_from_a_bijection_combines_physical_transforms_across_a_permutation():
    axes = [{"name": name, "type": "space"} for name in "yx"]
    scene = Scene.from_ome_zarr(
        {
            "coordinateSystems": [{"name": "source", "axes": axes}, {"name": "target", "axes": axes[::-1]}],
            "coordinateTransformations": [
                {
                    "type": "bijection",
                    "input": {"name": "source"},
                    "output": {"name": "target"},
                    "forward": {
                        "type": "sequence",
                        "transformations": [
                            {"type": "translation", "translation": [1, 2]},
                            {"type": "mapAxis", "mapAxis": [1, 0]},
                            {"type": "scale", "scale": [3, 4]},
                        ],
                    },
                    "inverse": {
                        "type": "sequence",
                        "transformations": [
                            {"type": "scale", "scale": [1 / 3, 1 / 4]},
                            {"type": "mapAxis", "mapAxis": [1, 0]},
                            {"type": "translation", "translation": [-1, -2]},
                        ],
                    },
                }
            ],
        }
    )
    assert scene.relation_from("source", to="target") == SpatialRelationSequence(
        (PermutationTo("xy"), Affine.from_array(((3, 0, 6), (0, 4, 4)), "xy"))
    )
    assert scene.relation_from("target", to="source") == SpatialRelationSequence(
        (PermutationTo("yx"), Affine.from_array(((1 / 4, 0, -1), (0, 1 / 3, -2)), "yx"))
    )


@pytest.mark.parametrize("meta", all_external_payload_scene_examples())
def test_scene_external_payload_public_examples_rejected(meta):
    with pytest.raises(ValueError, match="Transforms with external data are currently unsupported"):
        Scene.from_ome_zarr(meta)


@pytest.mark.parametrize("meta", all_invalid_scene_examples())
def test_scene_invalid_public_examples(meta):
    with pytest.raises(ValueError, match="ByDimensionTransform target axes must be globally unique"):
        _ = Scene.from_ome_zarr(meta)


def test_load_then_resolve():
    paths = {path: _multiscale() for path in ["tile_0", "tile_1", "tile_2", "tile_3"]}
    scene = Scene.from_ome_zarr(scene_stitching())
    assert not scene.is_fully_resolved

    resolved_scene = scene.with_resolved(paths)
    assert resolved_scene.is_fully_resolved


def test_with_resolved_remembers_only_paths_that_resolved_transform_endpoints():
    """
    This ensures Scene._full_graph cannot produce a disjunct graph.
    Could probably be relaxed since it's private anyway.
    """
    used = _multiscale()
    unused = _multiscale()
    scene = Scene.from_ome_zarr(
        {
            "coordinateSystems": [
                {
                    "name": "world",
                    "axes": [{"name": "y"}, {"name": "x"}],
                }
            ],
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

    resolved = scene.with_resolved({"tile_0": used, "unused": unused})

    assert resolved.multiscale_paths == {"tile_0": used}


def test_with_resolved_does_not_resolve_by_name():
    multiscale = _multiscale()
    world = CoordinateSystem.fromkeys("yx")._as_ref("world")
    transform = TranslationTransform(
        translation=(0, 0),
        source=_UnresolvedRef(name="world"),
        target=_UnresolvedRef(name="physical", path="tile_0"),
    )
    scene = Scene(TransformGraph([transform], system_refs=(world,)), _multiscale_paths={})
    resolved = scene.with_resolved({"tile_0": multiscale})

    assert isinstance(resolved._internal_graph.transforms[0].source, _UnresolvedRef)
    assert resolved._internal_graph.transforms[0].target == multiscale._as_ref("physical")


def test_scene_from_ome_zarr_rejects_axis_orientation_instead_of_dropping_it():
    example = scene_stitching()
    example["coordinateSystems"][0]["axes"][0]["orientation"] = {"type": "anatomical", "value": "left-to-right"}

    with pytest.raises(ValueError, match="anatomical orientation is not supported"):
        Scene.from_ome_zarr(example)
