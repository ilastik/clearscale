from typing import Any

import pytest
from clearscale import Multiscale
from clearscale.ome_zarr import make_all_singleton_shapes, SUPPORTED_OME_ZARR_VERSIONS_WRITE

from tests.ome_zarr.multiscale_examples import (
    minimal_multiscale_examples_params,
    maximal_multiscale_examples_params,
    MultiscaleMetadataExample,
)


def with_written_version(metadata: dict[str, Any], version: str) -> dict[str, Any]:
    if "version" in metadata:
        assert metadata["version"] == version
    return metadata | {"version": version}


@pytest.mark.parametrize("example", minimal_multiscale_examples_params())
def test_multiscale_roundtrips_minimal_ome_zarr(example: MultiscaleMetadataExample):
    if example.id not in SUPPORTED_OME_ZARR_VERSIONS_WRITE:
        pytest.skip(f"Writing version {example.id} not supported")
    multiscale = Multiscale.from_ome_zarr(example.metadata, get_shape=make_all_singleton_shapes(example.ndim))
    output_json = multiscale.to_ome_zarr(version=example.id)

    assert output_json == with_written_version(example.metadata, example.id)


@pytest.mark.skip("Still failing: Version carryover needs to be ignored + optional metadata carryover not implemented")
@pytest.mark.parametrize("example", maximal_multiscale_examples_params())
def test_multiscale_roundtrips_maximal_ome_zarr(example: MultiscaleMetadataExample):
    if example.id not in SUPPORTED_OME_ZARR_VERSIONS_WRITE:
        pytest.skip(f"Writing version {example.id} not supported")
    multiscale = Multiscale.from_ome_zarr(example.metadata, get_shape=make_all_singleton_shapes(example.ndim))
    output_json = multiscale.to_ome_zarr(version=example.id)

    assert example.metadata == output_json
