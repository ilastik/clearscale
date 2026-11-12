from typing import Any, Dict
from unittest.mock import Mock, MagicMock

import clearscale as cs


def test_minimal_ome_zarr_example():
    # 1. Data: Make a group and write the array into it
    # (no clearscale involved)
    array_path = "s0"

    # 2. Metadata: Scale -> Multiscale -> Group -> attrs
    written: Dict[str, Any] = cs.OmeZarrGroup.from_single(  # Group = one zarr group's OME metadata
        cs.Multiscale.from_single(  # Multiscale = an image
            scale=cs.Scale.from_lists("zyx"), scale_key=array_path  # Scale = an array
        )
    ).to_attrs(version="0.5")

    assert "ome" in written
    assert "multiscales" in written["ome"]
    assert "version" in written["ome"] and written["ome"]["version"] == "0.5"
    assert len(written["ome"]["multiscales"]) == 1

    expected = {
        "ome": {
            "multiscales": [
                {
                    "axes": [
                        {"name": "z", "type": "space"},
                        {"name": "y", "type": "space"},
                        {"name": "x", "type": "space"},
                    ],
                    "datasets": [
                        {
                            "coordinateTransformations": [{"scale": [1.0, 1.0, 1.0], "type": "scale"}],
                            "path": "s0",
                        }
                    ],
                    "version": "0.5",
                }
            ],
            "version": "0.5",
        }
    }

    assert written == expected


def test_metadata_first_example():
    my_data = Mock(shape=(3, 54, 1024, 1024))

    # 1. Define operation plan (the scaling "blueprint")
    shape = cs.Shape(zip("tzyx", my_data.shape))
    scaling_blueprint = cs.BlueprintShapes.downscale_powers_of_2_xyz(
        base_shape=shape,
        rounding="ceil",
        limit_all=cs.Shape(z=8, y=128, x=128),
    )

    # 2. Scale data according to the metadata

    # 3. Write metadata: Scale -> Multiscale -> Group -> attrs
    pixel_size = cs.PixelSize(t=5.0, z=260.0, y=0.53, x=0.53)
    unit = cs.Unit(t="s", z="micrometer", y="micrometer", x="micrometer")
    base = cs.Scale(shape, pixel_size, unit)

    ms = cs.Multiscale.from_single(base, blueprint=scaling_blueprint)
    written: Dict[str, Any] = cs.OmeZarrGroup.from_single(ms).to_attrs(version="0.6")

    assert "ome" in written
    assert "multiscales" in written["ome"]
    assert "version" in written["ome"] and written["ome"]["version"] == "0.6"
    assert len(written["ome"]["multiscales"]) == 1
    written_ms_dict = written["ome"]["multiscales"][0]
    assert len(written_ms_dict["coordinateSystems"]) == 1
    written_system_name = written_ms_dict["coordinateSystems"][0]["name"]  # Capture the randomly generated name

    # Downscaled levels are shifted by the default half-pixel translation (the README's `skimage.transform.resize`)
    expected = {
        "coordinateSystems": [
            {
                "axes": [
                    {"name": "t", "type": "time", "unit": "s"},
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "name": written_system_name,
            }
        ],
        "datasets": [
            {
                "coordinateTransformations": [
                    {
                        "input": {"path": "s0"},
                        "output": {"name": written_system_name},
                        "scale": [5.0, 260.0, 0.53, 0.53],
                        "type": "scale",
                    }
                ],
                "path": "s0",
            },
            {
                "coordinateTransformations": [
                    {
                        "input": {"path": "s1"},
                        "output": {"name": written_system_name},
                        "transformations": [
                            {"scale": [5.0, 520.0, 1.06, 1.06], "type": "scale"},
                            {"translation": [0.0, 130.0, 0.265, 0.265], "type": "translation"},
                        ],
                        "type": "sequence",
                    }
                ],
                "path": "s1",
            },
            {
                "coordinateTransformations": [
                    {
                        "input": {"path": "s2"},
                        "output": {"name": written_system_name},
                        "transformations": [
                            {"scale": [5.0, 1002.8571428571429, 2.12, 2.12], "type": "scale"},
                            {"translation": [0.0, 371.42857142857144, 0.795, 0.795], "type": "translation"},
                        ],
                        "type": "sequence",
                    }
                ],
                "path": "s2",
            },
            {
                "coordinateTransformations": [
                    {
                        "input": {"path": "s3"},
                        "output": {"name": written_system_name},
                        "transformations": [
                            {"scale": [5.0, 2005.7142857142858, 4.24, 4.24], "type": "scale"},
                            {"translation": [0.0, 872.8571428571429, 1.855, 1.855], "type": "translation"},
                        ],
                        "type": "sequence",
                    }
                ],
                "path": "s3",
            },
        ],
        "version": "0.6",
    }

    assert written_ms_dict == expected


def test_extract_single_scale_example():
    remote_attrs = {
        "multiscales": [
            {
                "axes": [
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": [
                    {"coordinateTransformations": [{"scale": [0.025, 0.01, 0.01], "type": "scale"}], "path": "s0"},
                    {"coordinateTransformations": [{"scale": [0.025, 0.02, 0.02], "type": "scale"}], "path": "s1"},
                    {"coordinateTransformations": [{"scale": [0.05, 0.04, 0.04], "type": "scale"}], "path": "s2"},
                    {"coordinateTransformations": [{"scale": [0.1, 0.08, 0.08], "type": "scale"}], "path": "s3"},
                    {"coordinateTransformations": [{"scale": [0.2, 0.16, 0.16], "type": "scale"}], "path": "s4"},
                    {"coordinateTransformations": [{"scale": [0.4, 0.32, 0.32], "type": "scale"}], "path": "s5"},
                    {"coordinateTransformations": [{"scale": [0.8, 0.64, 0.64], "type": "scale"}], "path": "s6"},
                    {"coordinateTransformations": [{"scale": [1.6, 1.28, 1.28], "type": "scale"}], "path": "s7"},
                    {"coordinateTransformations": [{"scale": [3.2, 2.56, 2.56], "type": "scale"}], "path": "s8"},
                    {"coordinateTransformations": [{"scale": [6.4, 5.12, 5.12], "type": "scale"}], "path": "s9"},
                ],
                "name": "platy-em",
                "version": "0.4",
            }
        ]
    }
    expected_written_meta = {
        "multiscales": [
            {
                "axes": [
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": [
                    {"coordinateTransformations": [{"scale": [0.8, 0.64, 0.64], "type": "scale"}], "path": "s6"}
                ],
                "version": "0.4",
            }
        ]
    }

    array_mock = Mock()
    array_mock.shape = (357, 405, 430)
    remote_group_mock = MagicMock()
    remote_group_mock.attrs = remote_attrs
    remote_group_mock.return_value = array_mock
    local_group_mock = Mock()
    zarr = Mock()
    zarr.open_group = Mock(side_effect=[remote_group_mock, local_group_mock])

    URL = "https://s3.embl.de/i2k-2020/platy-raw.ome.zarr"
    SCALE_KEY = "s6"
    LOCAL_PATH = f"demo-output/platy-raw-{SCALE_KEY}.ome.zarr"

    # 1. Discover what's on the server
    remote_group = zarr.open_group(URL)
    remote_meta = cs.OmeZarrGroup.from_group(remote_group)
    assert remote_meta.kind is cs.GroupKind.MULTISCALE, "This tells us there is exactly one Multiscale in this group"
    assert remote_meta.version == "0.4"
    source_multiscale = remote_meta.multiscales[0]
    assert SCALE_KEY in source_multiscale, "Multiscale is dict-like, with the sub-paths to arrays as keys"
    assert source_multiscale[SCALE_KEY].shape == {"z": 357, "y": 405, "x": 430}

    # 2. Create the local target and download the data
    source_array = remote_group[SCALE_KEY]
    local_group = zarr.open_group(str(LOCAL_PATH), mode="w", zarr_format=2)
    local_array = local_group.create_array(SCALE_KEY, data=source_array, overwrite=True)

    # 3. Extract the correct scale metadata and upgrade it to valid independent metadata
    target_multiscale = source_multiscale.derive(SCALE_KEY)
    local_group_meta = cs.OmeZarrGroup.from_single(target_multiscale)

    # 4. Write the new metadata to the downloaded store
    local_group.attrs.update(local_group_meta.to_attrs(version="0.4"))

    assert local_group_meta.to_attrs(version="0.4") == expected_written_meta
    local_group.attrs.update.assert_called_once_with(expected_written_meta)
