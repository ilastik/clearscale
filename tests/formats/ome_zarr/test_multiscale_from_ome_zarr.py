import pytest

from clearscale import Multiscale
from clearscale.ome_zarr import make_all_singleton_shapes

from tests.formats.ome_zarr.multiscale_examples import (
    MultiscaleMetadataExample,
    minimal_multiscale_examples_params,
    maximal_multiscale_examples_params,
)


@pytest.mark.parametrize("example", minimal_multiscale_examples_params())
def test_from_ome_zarr_parses_minimal_multiscale_examples(example: MultiscaleMetadataExample):
    multiscale = Multiscale.from_ome_zarr(example.metadata, get_shape=make_all_singleton_shapes(example.ndim))

    assert tuple(multiscale.keys()) == example.expected_paths


@pytest.mark.parametrize("example", maximal_multiscale_examples_params())
def test_from_ome_zarr_parses_maximal_multiscale_examples(example: MultiscaleMetadataExample):
    multiscale = Multiscale.from_ome_zarr(example.metadata, get_shape=make_all_singleton_shapes(example.ndim))

    assert tuple(multiscale.keys()) == example.expected_paths
