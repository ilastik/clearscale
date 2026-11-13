"""Dedicated tests for precautionary rejection until we implement a public API for handling path-backed transforms.

Like require_inline_transform_payloads, this can be reduced and entirely removed when we implement support."""

import copy
from typing import Any, Dict

import pytest

from clearscale import Multiscale, OmeZarrGroup, Scene
from clearscale._collections import GroupKind
from clearscale._transforms import TransformGraph
from tests.ome_zarr.scene_examples import all_valid_scene_examples, scene_registration


def _metadata(transform: Dict[str, Any], *, multiscale: bool = False) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "coordinateSystems": [
            {"name": name, "axes": [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]}
            for name in ("physical", "world")
        ],
        "coordinateTransformations": [{**transform, "input": {"name": "physical"}, "output": {"name": "world"}}],
    }
    if multiscale:
        metadata.update(
            version="0.6",
            axes=[{"name": "y", "type": "space"}, {"name": "x", "type": "space"}],
            datasets=[
                {
                    "path": "s0",
                    "coordinateTransformations": [{"type": "scale", "scale": [1, 1], "output": {"name": "physical"}}],
                }
            ],
        )
    return metadata


def _nested(transform: Dict[str, Any], container: str) -> Dict[str, Any]:
    if container == "sequence":
        return {"type": "sequence", "transformations": [{"type": "identity"}, transform]}
    if container == "forward":
        return {"type": "bijection", "forward": transform, "inverse": {"type": "identity"}}
    if container == "inverse":
        return {"type": "bijection", "forward": {"type": "identity"}, "inverse": transform}
    if container == "byDimension":
        return {
            "type": "byDimension",
            "transformations": [{"inputAxes": [0, 1], "outputAxes": [0, 1], "transformation": transform}],
        }
    return transform


@pytest.mark.parametrize(
    "transform_type", ["coordinates", "displacements", "scale", "translation", "rotation", "affine"]
)
@pytest.mark.parametrize("container", ["direct", "sequence", "forward", "inverse", "byDimension"])
@pytest.mark.parametrize("path", ["transforms/data", "/transforms/data"])
@pytest.mark.parametrize("multiscale", [False, True], ids=["scene", "multiscale"])
def test_readers_reject_external_payloads(transform_type, container, path, multiscale):
    metadata = _metadata(_nested({"type": transform_type, "path": path}, container), multiscale=multiscale)
    original = copy.deepcopy(metadata)
    with pytest.raises(ValueError, match="Transforms with external data are currently unsupported") as error:
        if multiscale:
            Multiscale.from_ome_zarr(metadata, shape_source=lambda _: pytest.fail("must reject before reading shapes"))
        else:
            Scene.from_ome_zarr(metadata)
    assert transform_type in str(error.value)
    assert path in str(error.value)
    assert "coordinateTransformations[0]" in str(error.value)
    if container in ("forward", "inverse"):
        assert f".{container}" in str(error.value)
    elif container == "byDimension":
        assert ".transformations[0].transformation" in str(error.value)
    elif container == "sequence":
        assert ".transformations[1]" in str(error.value)
    assert metadata == original


@pytest.mark.parametrize("mode", ["broken_graph", "legacy_list", "legacy_mapping"])
def test_multiscale_fallbacks_cannot_discard_external_payloads(mode):
    metadata = _metadata({"type": "scale", "path": "coefficients"}, multiscale=True)
    if mode == "broken_graph":
        metadata["coordinateSystems"] = [{"broken": True}]
    else:
        metadata.pop("coordinateSystems")
        metadata["version"] = "0.4"
        metadata["datasets"][0]["coordinateTransformations"][0].pop("output")
        if mode == "legacy_mapping":
            metadata["coordinateTransformations"] = metadata["coordinateTransformations"][0]
    with pytest.raises(ValueError, match="coefficients"):
        Multiscale.from_ome_zarr(metadata, shape_source="singletons")


@pytest.mark.parametrize(
    "transform_type,payload",
    [("scale", [2, 3]), ("translation", [1, 2]), ("rotation", [[0, -1], [1, 0]]), ("affine", [[1, 0, 2], [0, 1, 3]])],
)
@pytest.mark.parametrize("container", ["direct", "sequence", "forward", "inverse", "byDimension"])
def test_inline_payload_precedes_extra_path(transform_type, payload, container):
    transform = _nested({"type": transform_type, transform_type: payload, "path": "unused"}, container)
    metadata = _metadata(transform)
    scene = Scene.from_ome_zarr(metadata)
    written = scene.to_ome_zarr(version="0.6")
    assert "unused" not in repr(written)
    assert Scene.from_ome_zarr(written) == scene
    multiscale = Multiscale.from_ome_zarr(_metadata(transform, multiscale=True), shape_source="singletons")
    assert Multiscale.from_ome_zarr(multiscale.to_ome_zarr(version="0.6"), shape_source="singletons") == multiscale


def test_accepted_multiscale_derivation_still_serializes():
    original = Multiscale.from_ome_zarr(
        _metadata({"type": "scale", "scale": [2, 3], "path": "unused"}, multiscale=True), shape_source="singletons"
    )
    derived = Multiscale.from_ome_zarr(
        _metadata({"type": "identity"}, multiscale=True), shape_source="singletons"
    ).as_derived_from(original)
    written = derived.to_ome_zarr(version="0.6")
    assert "unused" not in repr(written)
    assert Multiscale.from_ome_zarr(written, shape_source="singletons") == derived


def test_external_payload_in_deeply_nested_composite_is_rejected():
    transform = _nested(
        _nested(_nested({"type": "displacements", "path": "field"}, "inverse"), "byDimension"), "sequence"
    )
    with pytest.raises(
        ValueError,
        match=r"coordinateTransformations\[0\].transformations\[1\].transformations\[0\].transformation.inverse",
    ):
        Scene.from_ome_zarr(_metadata(transform))


@pytest.mark.parametrize(
    "transform_type", ["scale", "translation", "rotation", "affine", "coordinates", "displacements"]
)
def test_dataset_external_payloads_keep_permissive_parsing(transform_type):
    metadata = _metadata({"type": "identity"}, multiscale=True)
    metadata["datasets"][0]["coordinateTransformations"] = [
        {"type": transform_type, "path": "ignored", "output": {"name": "physical"}}
    ]
    multiscale = Multiscale.from_ome_zarr(metadata, shape_source="singletons")
    assert tuple(multiscale) == ("s0",)
    assert "ignored" not in repr(multiscale.to_ome_zarr(version="0.6"))


@pytest.mark.parametrize("meta", all_valid_scene_examples())
def test_internal_graphs_still_roundtrip_all_valid_scene_examples(meta):
    graph = TransformGraph.from_ome_zarr(meta["coordinateTransformations"], meta.get("coordinateSystems"))
    assert graph.to_ome_zarr(version="0.6") == meta


@pytest.mark.parametrize("version", ["0.4", "0.6"])
def test_discovery_retains_valid_siblings_and_unsupported_metadata(version):
    valid = _metadata({"type": "identity"}, multiscale=True)
    valid["version"] = version
    invalid = copy.deepcopy(valid)
    invalid["coordinateTransformations"] = [{"type": "scale", "path": "coefficients"}]
    scene = scene_registration()
    attrs = {"multiscales": [invalid, valid], "scene": scene}
    original = copy.deepcopy(attrs)
    group = OmeZarrGroup.from_attrs(attrs, shape_source="singletons")
    assert group.kind is GroupKind.MULTISCALE
    assert len(group.multiscales) == 1
    assert not group.scenes
    assert not group.children
    assert [(obj.kind, obj.metadata) for obj in group.invalid_objects] == [("multiscale", invalid), ("scene", scene)]
    assert all("Transforms with external data are currently unsupported" in obj.error for obj in group.invalid_objects)
    assert group.version == version
    written = group.to_attrs(version="0.6")
    ome = written["ome"]
    assert isinstance(ome, dict)
    multiscales = ome["multiscales"]
    assert isinstance(multiscales, list)
    assert len(multiscales) == 1
    assert "scene" not in ome
    assert "coefficients" not in repr(written)
    assert attrs == original


def test_rejected_multiscale_preserves_version():
    metadata = _metadata({"type": "coordinates", "path": "field"}, multiscale=True)
    group = OmeZarrGroup.from_attrs({"multiscales": [metadata]}, shape_source="singletons")
    assert group.kind is GroupKind.INVALID
    assert group.version == "0.6"
    assert group.invalid_objects[0].metadata == metadata


def test_rejected_scene_preserves_independently_declared_children():
    attrs = {"scene": scene_registration(), "labels": ["segmentation"]}
    group = OmeZarrGroup.from_attrs(attrs, shape_source="singletons")
    assert not group.scenes
    assert [(child.child_type, child.path) for child in group.children] == [("label_multiscale", "segmentation")]
    assert group.invalid_objects[0].metadata == attrs["scene"]
