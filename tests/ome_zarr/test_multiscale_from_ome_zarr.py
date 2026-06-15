import pytest

from clearscale import Multiscale
from clearscale.ome_zarr import make_all_singleton_shapes, SUPPORTED_OME_ZARR_VERSIONS_READ

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
    multiscale = Multiscale.from_ome_zarr(example.metadata, get_shape=make_all_singleton_shapes(example.ndim))

    assert tuple(multiscale.keys()) == example.expected_paths


@pytest.mark.parametrize("example", maximal_multiscale_examples_params())
def test_from_ome_zarr_parses_maximal_multiscale_examples(example: MultiscaleMetadataExample):
    multiscale = Multiscale.from_ome_zarr(example.metadata, get_shape=make_all_singleton_shapes(example.ndim))

    assert tuple(multiscale.keys()) == example.expected_paths
