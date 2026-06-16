import copy
from typing import Any

import pytest
from clearscale import Multiscale
from clearscale.ome_zarr import make_all_singleton_shapes, SUPPORTED_OME_ZARR_VERSIONS_WRITE

from tests.ome_zarr.multiscale_examples import (
    minimal_multiscale_examples_params,
    maximal_multiscale_examples_params,
    MultiscaleMetadataExample,
)

known_keys_that_should_roundtrip_but_todo = ("type", "labels", "omero", "metadata")


def with_written_version(metadata: dict[str, Any], version: str) -> dict[str, Any]:
    if "version" in metadata:
        assert metadata["version"] == version
    return metadata | {"version": version}


def without_known_feature_gaps(metadata: dict[str, Any]) -> dict[str, Any]:
    round_trippable_metadata = copy.deepcopy(metadata)
    for key in known_keys_that_should_roundtrip_but_todo:
        del round_trippable_metadata[key]
    return round_trippable_metadata


@pytest.mark.parametrize("example", minimal_multiscale_examples_params())
def test_multiscale_roundtrips_minimal_ome_zarr(example: MultiscaleMetadataExample):
    if example.id not in SUPPORTED_OME_ZARR_VERSIONS_WRITE:
        pytest.skip(f"Writing version {example.id} not supported")
    multiscale = Multiscale.from_ome_zarr(example.metadata, get_shape=make_all_singleton_shapes(example.ndim))
    output_json = multiscale.to_ome_zarr(version=example.id)

    expected_output = with_written_version(example.metadata, example.id)
    assert output_json == expected_output


@pytest.mark.parametrize("example", maximal_multiscale_examples_params())
def test_multiscale_roundtrips_maximal_ome_zarr(example: MultiscaleMetadataExample):
    if example.id not in SUPPORTED_OME_ZARR_VERSIONS_WRITE:
        pytest.skip(f"Writing version {example.id} not supported")
    multiscale = Multiscale.from_ome_zarr(example.metadata, get_shape=make_all_singleton_shapes(example.ndim))
    output_json = multiscale.to_ome_zarr(version=example.id)

    for key in known_keys_that_should_roundtrip_but_todo:
        assert key not in output_json, "Update test when implementing round-trip for previously unsupported optionals"
    expected_output = with_written_version(without_known_feature_gaps(example.metadata), example.id)
    assert output_json == expected_output
