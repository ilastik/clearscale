from typing import cast

import pytest
from clearscale._scene import Scene
from clearscale._collections import OmeZarrGroup, GroupKind
from clearscale.ome_zarr import make_all_singleton_shapes, SUPPORTED_VERSIONS_WRITE

from tests.ome_zarr.multiscale_examples import (
    group_metadata_examples_params,
    minimal_multiscale_examples_params,
    maximal_multiscale_examples_params,
    MultiscaleMetadataExample,
)
from tests.ome_zarr.scene_examples import (
    all_invalid_scene_examples,
    all_valid_scene_examples,
    scene_registration,
    scene_stitching,
    scene_to_group_attrs,
)


class MockZarrGroup:
    def __init__(self, attrs, shape_source=None):
        self.attrs = attrs
        self.shape_source = shape_source

    def __getitem__(self, path: str):
        if self.shape_source is None:
            raise ValueError("provide shape_source for tests that need group[] indexing")
        return self.shape_source(path)


@pytest.mark.parametrize("example", minimal_multiscale_examples_params())
def test_ome_zarr_group_parses_minimal_multiscale_examples(example: MultiscaleMetadataExample):
    zarr_group = MockZarrGroup(example.to_group_attrs(), make_all_singleton_shapes(example.ndim))
    ome_group = OmeZarrGroup.from_group(zarr_group)

    assert ome_group.kind is GroupKind.MULTISCALE
    assert len(ome_group.multiscales) == 1
    assert tuple(ome_group.multiscales[0].keys()) == example.expected_paths
    if "version" in example.metadata:
        assert ome_group.version == example.id


@pytest.mark.parametrize("example", maximal_multiscale_examples_params())
def test_ome_zarr_group_parses_maximal_multiscale_examples(example: MultiscaleMetadataExample):
    zarr_group = MockZarrGroup(example.to_group_attrs(), make_all_singleton_shapes(example.ndim))
    ome_group = OmeZarrGroup.from_group(zarr_group)

    assert ome_group.kind is GroupKind.MULTISCALE
    assert len(ome_group.multiscales) == 1
    assert tuple(ome_group.multiscales[0].keys()) == example.expected_paths
    assert ome_group.version == example.id


class TestOmeroAndImageLabel:
    """In "multiscales" OME-Zarr groups, the "omero" and "image-label" json keys sit *next to* "multiscales".
    They describe the multiscale objects, but live outside of them. Hence, tested here."""

    @pytest.mark.parametrize("example", group_metadata_examples_params("omero"))
    def test_ome_zarr_group_parses_omero_and_attaches_it_to_the_multiscale(self, example: MultiscaleMetadataExample):
        ome_group = OmeZarrGroup.from_group(
            MockZarrGroup(example.to_group_attrs(), make_all_singleton_shapes(example.ndim))
        )
        omero = ome_group.multiscales[0].ome.omero

        assert omero is not None
        assert omero.to_ome_zarr() == example.group_metadata["omero"]
        channel = omero.channels[0]
        assert (channel.color, channel.window.start, channel.window.end) == ("0000FF", 0, 1500)
        assert (channel.window.min, channel.window.max) == (0, 65535)
        # Keys clearscale doesn't model are kept, not lost
        assert channel.extra == {
            "active": True,
            "coefficient": 1,
            "family": "linear",
            "inverted": False,
            "label": "LaminB1",
        }
        assert omero.extra["name"] == "example.tif"
        assert omero.extra["rdefs"] == {"defaultT": 0, "defaultZ": 118, "model": "color"}

    @pytest.mark.parametrize("example", group_metadata_examples_params("image-label"))
    def test_ome_zarr_group_parses_image_label_and_attaches_it_to_the_multiscale(
        self, example: MultiscaleMetadataExample
    ):
        ome_group = OmeZarrGroup.from_group(
            MockZarrGroup(example.to_group_attrs(), make_all_singleton_shapes(example.ndim))
        )
        image_label = ome_group.multiscales[0].ome.image_label

        assert image_label is not None
        assert image_label.source is not None and image_label.source.path == "../../"
        assert list(image_label.labels) == [1, 2, 3]
        assert image_label.labels[1].color == (255, 0, 0, 128)
        assert image_label.labels[1].properties == {"class": "cell", "area (pixels)": 1200}
        assert image_label.labels[2].properties == {
            "@color:hexColor": "#00FF00",  # An "additional key under colors"
            "class": "nucleus",
            "confidence": 0.5,
        }
        assert image_label.labels[3].color is None
        assert image_label.labels[3].properties == {"class": "background"}

    @pytest.mark.parametrize("example", maximal_multiscale_examples_params())
    def test_ome_zarr_group_only_attaches_omero_and_image_label_if_present(self, example: MultiscaleMetadataExample):
        ome_group = OmeZarrGroup.from_group(
            MockZarrGroup(example.to_group_attrs(), make_all_singleton_shapes(example.ndim))
        )
        ms = ome_group.multiscales[0]

        assert (ms.ome.omero is not None) == ("omero" in example.group_metadata)
        assert (ms.ome.image_label is not None) == ("image-label" in example.group_metadata)

    @pytest.mark.parametrize("example", group_metadata_examples_params("omero"))
    def test_ome_zarr_group_roundtrips_omero(self, example: MultiscaleMetadataExample):
        if example.id not in SUPPORTED_VERSIONS_WRITE:
            pytest.skip(f"Writing version {example.id} not supported")
        ome_group = OmeZarrGroup.from_group(
            MockZarrGroup(example.to_group_attrs(), make_all_singleton_shapes(example.ndim))
        )

        output = ome_group.to_attrs(version=example.id)
        ome_attrs = output["ome"] if "ome" in output else output

        assert ome_attrs["omero"] == example.group_metadata["omero"]
        assert "omero" not in ome_attrs["multiscales"][0], "omero is a group key, not a multiscale key"

    @pytest.mark.parametrize("example", group_metadata_examples_params("image-label"))
    def test_ome_zarr_group_roundtrips_image_label(self, example: MultiscaleMetadataExample):
        if example.id not in SUPPORTED_VERSIONS_WRITE:
            pytest.skip(f"Writing version {example.id} not supported")
        ome_group = OmeZarrGroup.from_group(
            MockZarrGroup(example.to_group_attrs(), make_all_singleton_shapes(example.ndim))
        )

        output = ome_group.to_attrs(version=example.id)
        ome_attrs = output["ome"] if "ome" in output else output

        assert ome_attrs["image-label"] == example.group_metadata["image-label"]
        assert "image-label" not in ome_attrs["multiscales"][0], "image-label is a group key, not a multiscale key"

    @pytest.mark.parametrize("example", maximal_multiscale_examples_params("0.5"))
    def test_ome_zarr_group_writes_omero_and_image_label_in_other_writable_versions(
        self, example: MultiscaleMetadataExample
    ):
        """Read as 0.5, write as 0.4 and 0.6: omero is version-independent, image-label only carries its `version` in
        the versions that have one."""
        ome_group = OmeZarrGroup.from_group(
            MockZarrGroup(example.to_group_attrs(), make_all_singleton_shapes(example.ndim))
        )

        as_0_4 = ome_group.to_attrs(version="0.4")
        output_0_6 = ome_group.to_attrs(version="0.6")
        as_0_6 = output_0_6["ome"] if "ome" in output_0_6 else output_0_6

        expected_image_label = example.group_metadata["image-label"]
        assert as_0_4["omero"] == as_0_6["omero"] == example.group_metadata["omero"]
        assert as_0_4["image-label"] == {**expected_image_label, "version": "0.4"}
        assert as_0_6["image-label"] == {k: v for k, v in expected_image_label.items() if k != "version"}

    def test_ome_zarr_group_ignores_invalid_omero(self):
        """A single invalid channel makes the whole omero unusable
        (dropping it would make omero.channels != image channels)."""
        example = cast(MultiscaleMetadataExample, maximal_multiscale_examples_params("0.4")[0].values[0])
        attrs = example.to_group_attrs()
        attrs["omero"]["channels"].append({"color": "FF0000"})

        with pytest.warns(UserWarning, match="Invalid entry in 'omero.channels'"):
            ome_group = OmeZarrGroup.from_group(MockZarrGroup(attrs, make_all_singleton_shapes(example.ndim)))

        assert ome_group.kind is GroupKind.MULTISCALE
        assert ome_group.multiscales[0].ome.omero is None
        assert ome_group.multiscales[0].ome.image_label is not None, "invalid omero must not affect image-label"

    def test_ome_zarr_group_ignores_omero_and_image_label_without_multiscales(self):
        attrs = {"omero": {"channels": []}, "image-label": {"source": {"image": "../../"}}}

        ome_group = OmeZarrGroup.from_group(MockZarrGroup(attrs))

        assert ome_group.kind is None
        assert not ome_group.multiscales


def test_ome_zarr_group_ignores_invalid_multiscale():
    invalid_meta = {"multiscales": [{"datasets": [{"malformed": "idk"}]}]}
    zarr_group = MockZarrGroup(invalid_meta, make_all_singleton_shapes(1))
    ome_group = OmeZarrGroup.from_group(zarr_group)

    assert ome_group.kind is GroupKind.INVALID
    assert len(ome_group.multiscales) == 0
    assert ome_group.version is None
    assert [(obj.kind, obj.metadata) for obj in ome_group.invalid_objects] == [
        ("multiscale", invalid_meta["multiscales"][0])
    ]
    assert ome_group.invalid_objects[0].error


def test_group_records_invalid_multiscale_reason():
    invalid_meta = {
        "multiscales": [
            {
                "version": "0.4",
                "axes": [{"name": "y"}, {"name": "x"}],  # invalid: MUST have 2 or 3 type="space" axes
                "datasets": [{"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0]}]}],
            }
        ]
    }
    group = OmeZarrGroup.from_attrs(invalid_meta, shape_source="singletons")
    assert group.kind is GroupKind.INVALID and group.multiscales == ()
    assert group.version == "0.4"
    assert "require 2 or 3 axes of type 'space', but this Multiscale has 0" in group.invalid_objects[0].error


def test_ome_zarr_group_parses_scene_stitching_example():
    zarr_group = MockZarrGroup(scene_to_group_attrs(scene_stitching()))
    ome_group = OmeZarrGroup.from_group(zarr_group)
    expected_paths = ["tile_0", "tile_1", "tile_2", "tile_3"]

    assert ome_group.kind is GroupKind.SCENE
    assert len(ome_group.scenes) == 1
    assert ome_group.scenes[0].unresolved_paths == expected_paths
    assert [child.file.path for child in ome_group.children] == expected_paths
    assert ome_group.version == "0.6"


def test_ome_zarr_group_parses_scene_registration_example():
    zarr_group = MockZarrGroup(scene_to_group_attrs(scene_registration()))
    ome_group = OmeZarrGroup.from_group(zarr_group)

    assert ome_group.kind is GroupKind.SCENE
    assert len(ome_group.scenes) == 1
    assert ome_group.scenes[0].unresolved_paths == ["JRC2018F", "FCWB"]
    assert [child.file.path for child in ome_group.children] == ["JRC2018F", "FCWB"]
    assert ome_group.version == "0.6"


@pytest.mark.parametrize("meta", all_valid_scene_examples())
def test_ome_zarr_group_parses_scene_public_examples(meta):
    zarr_group = MockZarrGroup(scene_to_group_attrs(meta))
    ome_group = OmeZarrGroup.from_group(zarr_group)

    assert ome_group.kind is GroupKind.SCENE
    assert len(ome_group.scenes) == 1
    assert isinstance(ome_group.scenes[0], Scene)
    assert ome_group.version == "0.6"


@pytest.mark.parametrize("meta", all_invalid_scene_examples())
def test_ome_zarr_group_ignores_scene_invalid_examples(meta):
    zarr_group = MockZarrGroup(scene_to_group_attrs(meta))
    ome_group = OmeZarrGroup.from_group(zarr_group)

    assert ome_group.kind is GroupKind.INVALID
    assert not ome_group.scenes
    assert ome_group.version == "0.6"
    assert [obj.kind for obj in ome_group.invalid_objects] == ["scene"]
