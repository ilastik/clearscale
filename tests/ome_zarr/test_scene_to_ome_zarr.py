from typing import Any, Dict
import pytest

from clearscale import Affine, Factor, Multiscale, SameSpace, Scale, Scene, Shape, Translation, OmeZarrGroup
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
        scene = Scene._from_graph_edges(
            [
                (a, TranslationTransform((1.0, 2.0)), world_ref),
                (b, ScaleTransform((2.0, 2.0)), world_ref),
                (world_ref, ScaleTransform((0.001, 0.001)), mm_ref),
            ],
            multiscale_paths={"a": a, "b": b},
        )
        ms_by_path = {"a": a, "b": b}

        result: Dict[str, Any] = scene.to_ome_zarr(version="0.6")

        names = {s["name"] for s in result["coordinateSystems"]}
        assert names == {"world", "mm"}
        edges = {(t["input"]["name"], t["output"]["name"]) for t in result["coordinateTransformations"]}
        assert (a._intrinsic_ref.name, "world") in edges
        assert (b._intrinsic_ref.name, "world") in edges
        assert ("world", "mm") in edges

        read_back = Scene.from_ome_zarr(result).with_resolved(multiscales_by_path=ms_by_path)

        chain = read_back._transforms_between(a, "mm")
        assert chain is not None and len(chain) == 2
        assert read_back.to_ome_zarr(version="0.6") == result

    def test_writes_and_reads_back_shared_coordinate_system(self):
        tile_a = _multiscale(y=2, x=2)
        tile_b = _multiscale(y=2, x=2)
        scene = Scene.from_shared_coordinate_system(
            coordinate_system="stage",
            connections=[(tile_a, SameSpace(), "a.zarr"), (tile_b, Translation(y=0, x=4), "b.zarr")],
        )
        paths = {"a.zarr": tile_a, "b.zarr": tile_b}

        result: Dict[str, Any] = scene.to_ome_zarr(version="0.6")

        assert [s["name"] for s in result["coordinateSystems"]] == ["stage"]
        assert [(t["type"], t["input"]["path"]) for t in result["coordinateTransformations"]] == [
            ("identity", "a.zarr"),
            ("translation", "b.zarr"),
        ]
        read_back = Scene.from_ome_zarr(result).with_resolved(paths)
        assert read_back.to_ome_zarr(version="0.6") == result
        chain = read_back._transforms_between(tile_a, tile_b)
        assert chain is not None and len(chain) == 2

    def test_writes_and_reads_back_multiscale_pair(self):
        raw = _multiscale(y=4, x=4)
        atlas = _multiscale(y=2, x=2)
        scene = Scene.from_multiscale_pair(
            a=raw, b=atlas, a_to_b=Affine.from_factor(Factor(y=0.5, x=0.5)), a_path="raw.zarr", b_path="atlas.zarr"
        )
        paths = {"raw.zarr": raw, "atlas.zarr": atlas}

        result = scene.to_ome_zarr(version="0.6")

        assert "coordinateSystems" not in result, "no coordinate systems of its own"
        assert result["coordinateTransformations"] == [
            {
                "type": "scale",
                "scale": [0.5, 0.5],
                "input": {"name": raw._intrinsic_ref.name, "path": "raw.zarr"},
                "output": {"name": atlas._intrinsic_ref.name, "path": "atlas.zarr"},
            }
        ]
        read_back = Scene.from_ome_zarr(result).with_resolved(paths)
        assert read_back == scene
        assert read_back.to_ome_zarr(version="0.6") == result

    def test_writes_and_reads_back_through_ome_zarr_group(self):
        tile_0 = _multiscale(y=2, x=3)
        tile_1 = _multiscale(y=2, x=3)
        scene = Scene._from_graph_edges(
            [(tile_0, TranslationTransform((1, 1)), tile_1)], multiscale_paths={"tile_0": tile_0, "tile_1": tile_1}
        )

        attrs: Dict[str, Any] = OmeZarrGroup.from_single(scene).to_attrs(version="0.6")

        (transform,) = attrs["ome"]["scene"]["coordinateTransformations"]
        assert transform["input"]["path"] == "tile_0"
        assert transform["output"]["path"] == "tile_1"

        read = OmeZarrGroup.from_attrs(attrs, shape_source="singletons").scenes[0]

        assert read.unresolved_paths == ["tile_0", "tile_1"]
        assert read.with_resolved({"tile_0": tile_0, "tile_1": tile_1}).multiscale_paths == {
            "tile_0": tile_0,
            "tile_1": tile_1,
        }

    def test_writes_explicit_system_refs_in_declared_order(self):
        atlas = CoordinateSystem.fromkeys("yx")._as_ref("atlas")
        world = CoordinateSystem.fromkeys("yx")._as_ref("world")
        edge = TranslationTransform((1.0, 2.0)).bound(source=atlas, target=world)
        scene = Scene(TransformGraph([edge], system_refs=(world, atlas)), _multiscale_paths={})

        result: Dict[str, Any] = scene.to_ome_zarr(version="0.6")

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
        scene = Scene._from_graph_edges([], multiscale_paths={})

        assert scene.to_ome_zarr(version="0.6") == {}

    def test_rejects_versions_other_than_0_6(self):
        a = _multiscale(y=2, x=2)
        b = _multiscale(y=2, x=2)
        scene = Scene._from_graph_edges([(a, TranslationTransform((1.0, 2.0)), b)], multiscale_paths={"a": a, "b": b})

        for version in ("0.4", "0.5"):
            with pytest.raises(ValueError, match="Scenes can only be written in OME-Zarr version 0.6"):
                scene.to_ome_zarr(version=version)
