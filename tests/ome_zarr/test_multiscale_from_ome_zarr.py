import pytest

from clearscale import Multiscale, PixelSize, Shape
from clearscale.ome_zarr import make_all_singleton_shapes, make_proportional_shapes, SUPPORTED_OME_ZARR_VERSIONS_READ

from tests.ome_zarr.multiscale_examples import (
    MultiscaleMetadataExample,
    minimal_multiscale_examples_params,
    maximal_multiscale_examples_params,
)


def test_all_versions_covered():
    example_params = minimal_multiscale_examples_params()
    versions = [params.id for params in example_params]
    assert set(versions) == set(
        SUPPORTED_OME_ZARR_VERSIONS_READ
    ), "Add at least a minimal test example when adding support for new OME-Zarr versions"


@pytest.mark.parametrize("example", minimal_multiscale_examples_params())
def test_from_ome_zarr_parses_minimal_multiscale_examples(example: MultiscaleMetadataExample):
    multiscale = Multiscale.from_ome_zarr(example.metadata, shape_source=make_all_singleton_shapes(example.ndim))

    assert tuple(multiscale.keys()) == example.expected_paths


@pytest.mark.parametrize("example", maximal_multiscale_examples_params())
def test_from_ome_zarr_parses_maximal_multiscale_examples(example: MultiscaleMetadataExample):
    multiscale = Multiscale.from_ome_zarr(example.metadata, shape_source=make_all_singleton_shapes(example.ndim))

    assert tuple(multiscale.keys()) == example.expected_paths


@pytest.mark.parametrize(
    "broken_example, expected_error",
    [
        ({"bioformats2raw.layout": 3}, "no datasets"),
        ({"datasets": 3}, "no datasets"),
        ({"datasets": [{"noop": 0}]}, "dataset missing path"),
        ({"datasets": [{"path": 0}]}, "dataset missing path"),
        ({"version": "0.4", "datasets": [{"path": "0", "coordinateTransformations": 0}]}, "invalid transformations"),
    ],
)
def test_from_ome_zarr_raises_invalid(broken_example, expected_error):
    with pytest.raises(ValueError, match=expected_error):
        _ = Multiscale.from_ome_zarr(broken_example, shape_source=make_proportional_shapes(broken_example))


def test_from_ome_zarr_accepts_shape_mapping():
    metadata = {
        "axes": [{"name": "y"}, {"name": "x"}],
        "datasets": [
            {"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0]}]},
            {"path": "s1", "coordinateTransformations": [{"type": "scale", "scale": [2.0, 2.0]}]},
        ],
    }

    multiscale = Multiscale.from_ome_zarr(metadata, shape_source={"s0": (100, 200), "s1": (50, 100)})

    assert multiscale["s0"].shape == Shape(y=100, x=200)
    assert multiscale["s1"].shape == Shape(y=50, x=100)


def test_from_ome_zarr_normalizes_zero_scale_values():
    metadata = {
        "axes": [{"name": "c"}, {"name": "y"}, {"name": "x"}],
        "datasets": [
            {"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [0.0, 1.0, 1.0]}]},
        ],
    }

    multiscale = Multiscale.from_ome_zarr(metadata, shape_source={"s0": (3, 100, 200)})

    assert multiscale["s0"].pixel_size == PixelSize(c=1.0, y=1.0, x=1.0)


def test_from_ome_zarr_normalizes_zero_scale_values_with_proportional_shape_source():
    metadata = {
        "axes": [{"name": "c"}, {"name": "y"}, {"name": "x"}],
        "datasets": [
            {"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [0.0, 1.0, 1.0]}]},
            {"path": "s1", "coordinateTransformations": [{"type": "scale", "scale": [0.0, 2.0, 2.0]}]},
        ],
    }

    multiscale = Multiscale.from_ome_zarr(metadata, shape_source=make_proportional_shapes(metadata))

    assert tuple(multiscale.keys()) == ("s0", "s1")
    assert multiscale["s0"].pixel_size == PixelSize(c=1.0, y=1.0, x=1.0)
    assert multiscale["s1"].pixel_size == PixelSize(c=1.0, y=2.0, x=2.0)


def test_from_ome_zarr_accepts_array_mapping():
    class Array:
        def __init__(self, shape):
            self.shape = shape

    metadata = {
        "axes": [{"name": "y"}, {"name": "x"}],
        "datasets": [{"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0]}]}],
    }

    multiscale = Multiscale.from_ome_zarr(metadata, shape_source={"s0": Array((100, 200))})

    assert multiscale["s0"].shape == Shape(y=100, x=200)


def test_from_ome_zarr_accepts_shape_values():
    metadata = {
        "axes": [{"name": "y"}, {"name": "x"}],
        "datasets": [{"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0]}]}],
    }

    multiscale = Multiscale.from_ome_zarr(metadata, shape_source={"s0": Shape(y=100, x=200)})

    assert multiscale["s0"].shape == Shape(y=100, x=200)


def test_from_ome_zarr_rejects_plain_string_shape_source():
    metadata = {
        "axes": [{"name": "y"}, {"name": "x"}],
        "datasets": [{"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0]}]}],
    }

    with pytest.raises(TypeError, match="Cannot obtain array shape from plain path"):
        Multiscale.from_ome_zarr(metadata, shape_source="image.ome.zarr")
