import pytest

from clearscale import Multiscale, Scale, Scene, Shape, Translation
from clearscale._transforms import CoordinateSystem, ScaleTransform, TransformGraph, TranslationTransform

from tests.ome_zarr.scene_examples import scene_stitching


def _multiscale(**shape: int) -> Multiscale:
    return Multiscale({"s0": Scale(Shape(**shape))})


def test_stitching_example_roundtrip():
    example = scene_stitching()
    scene = Scene.from_ome_zarr(example)
    output_json = scene.to_ome_zarr(version="0.6")

    assert output_json == example


class TestMultipleCoordinateSystemsToOmeZarr06:
    def test_writes_all_reachable_systems_from_graph_edges(self):
        world_ref = CoordinateSystem.fromkeys("yx")._as_ref("world")
        mm_ref = CoordinateSystem.fromkeys("yx")._as_ref("mm")
        a = _multiscale(y=2, x=2)
        b = _multiscale(y=2, x=2)
        scene = Scene.from_graph_edges(
            [
                (a, TranslationTransform((1.0, 2.0)), world_ref),
                (b, ScaleTransform((2.0, 2.0)), world_ref),
                (world_ref, ScaleTransform((0.001, 0.001)), mm_ref),
            ]
        )
        ms_by_path = {"a": a, "b": b}

        result = scene.to_ome_zarr(version="0.6", multiscales_by_path=ms_by_path)

        names = {s["name"] for s in result["coordinateSystems"]}
        assert names == {"world", "mm"}
        edges = {(t["input"]["name"], t["output"]["name"]) for t in result["coordinateTransformations"]}
        assert (a._intrinsic_ref.name, "world") in edges
        assert (b._intrinsic_ref.name, "world") in edges
        assert ("world", "mm") in edges

        read_back = Scene.from_ome_zarr(result).with_resolved(multiscales_by_path=ms_by_path)

        chain = read_back.transforms_between(a, "mm")
        assert chain is not None and len(chain) == 2
        assert read_back.to_ome_zarr(version="0.6") == result

    def test_writes_all_reachable_systems_from_tiles_translations(self):
        tile_a = _multiscale(y=2, x=2)
        tile_b = _multiscale(y=2, x=2)
        scene = Scene.from_tiles_translations([(tile_a, Translation(y=0, x=0)), (tile_b, Translation(y=0, x=4))])
        paths = {"a.zarr": tile_a, "b.zarr": tile_b}

        result = scene.to_ome_zarr(version="0.6", multiscales_by_path=paths)

        names = [s["name"] for s in result["coordinateSystems"]]
        # "world" is the hard-coded name for the central system. Subject to change.
        assert names == ["world"]

        read_back = Scene.from_ome_zarr(result).with_resolved(paths)
        assert read_back.to_ome_zarr(version="0.6") == result
        chain = read_back.transforms_between(tile_a, tile_b)
        assert chain is not None and len(chain) == 2

    def test_writes_explicit_system_refs_in_declared_order(self):
        atlas = CoordinateSystem.fromkeys("yx")._as_ref("atlas")
        world = CoordinateSystem.fromkeys("yx")._as_ref("world")
        edge = TranslationTransform((1.0, 2.0)).bound(source=atlas, target=world)
        scene = Scene(TransformGraph([edge], system_refs=(world, atlas)), _multiscale_paths={})

        result = scene.to_ome_zarr(version="0.6")

        assert [s["name"] for s in result["coordinateSystems"]] == ["world", "atlas"]

    def test_with_resolved_does_not_interfere_with_paths(self):
        tile = _multiscale(y=2, x=2)
        example = {
            "coordinateSystems": [
                {"name": "world", "axes": [{"name": "y"}, {"name": "x"}]},
                {"name": "mm", "axes": [{"name": "y", "unit": "mm"}, {"name": "x", "unit": "mm"}]},
            ],
            "coordinateTransformations": [
                {
                    "type": "translation",
                    "translation": [1.0, 2.0],
                    "input": {"path": "tile0.zarr", "name": tile._intrinsic_ref.name},
                    "output": {"name": "world"},
                },
                {
                    "type": "scale",
                    "scale": [0.001, 0.001],
                    "input": {"name": "world"},
                    "output": {"name": "mm"},
                },
            ],
        }

        scene = Scene.from_ome_zarr(example)
        assert scene.to_ome_zarr(version="0.6") == example

        resolved = scene.with_resolved({"tile0.zarr": tile})
        assert resolved.is_fully_resolved
        assert resolved.to_ome_zarr(version="0.6") == example

    def test_empty_scene_serializes_to_empty_dict(self):
        scene = Scene.from_tiles_translations([])

        assert scene.to_ome_zarr(version="0.6") == {}

    def test_rejects_versions_other_than_0_6(self):
        a = _multiscale(y=2, x=2)
        b = _multiscale(y=2, x=2)
        scene = Scene.from_graph_edges([(a, TranslationTransform((1.0, 2.0)), b)])

        for version in ("0.4", "0.5"):
            with pytest.raises(ValueError, match="Scenes can only be written in OME-Zarr version 0.6"):
                scene.to_ome_zarr(version=version)
