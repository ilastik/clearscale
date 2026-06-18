import pytest
from clearscale import Multiscale

WEILER14 = {
    "num_channels": 1,
    "type": "image",
    "data_type": "uint16",
    "scales": [
        {
            "encoding": "raw",
            "chunk_sizes": [[512, 512, 16]],
            "key": "100_100_70",
            "resolution": [100, 100, 70],
            "voxel_offset": [0, 0, 0],
            "size": [3409, 3337, 70],
        },
        {
            "encoding": "raw",
            "chunk_sizes": [[512, 512, 16]],
            "key": "200_200_70",
            "resolution": [200, 200, 70],
            "voxel_offset": [0, 0, 0],
            "size": [1705, 1669, 70],
        },
        {
            "encoding": "raw",
            "chunk_sizes": [[512, 512, 16]],
            "key": "400_400_70",
            "resolution": [400, 400, 70],
            "voxel_offset": [0, 0, 0],
            "size": [853, 835, 70],
        },
        {
            "encoding": "raw",
            "chunk_sizes": [[512, 512, 16]],
            "key": "800_800_70",
            "resolution": [800, 800, 70],
            "voxel_offset": [0, 0, 0],
            "size": [427, 418, 70],
        },
        {
            "encoding": "raw",
            "chunk_sizes": [[512, 512, 16]],
            "key": "1600_1600_70",
            "resolution": [1600, 1600, 70],
            "voxel_offset": [0, 0, 0],
            "size": [214, 209, 70],
        },
    ],
}
"""
precomputed://https://open-neurodata.s3.amazonaws.com/weiler14/Ex10R55/DAPI_1
https://open-neurodata.s3.amazonaws.com/weiler14/Ex10R55/DAPI_1/info
"""

COLLMAN = {
    "num_channels": 1,
    "type": "image",
    "data_type": "uint8",
    "scales": [
        {
            "encoding": "raw",
            "chunk_sizes": [[512, 512, 16]],
            "key": "2.24_2.24_70",
            "resolution": [2.24, 2.24, 70],
            "voxel_offset": [0, 0, 0],
            "size": [6306, 4518, 27],
        },
        {
            "encoding": "raw",
            "chunk_sizes": [[512, 512, 16]],
            "key": "4.48_4.48_70",
            "resolution": [4.48, 4.48, 70],
            "voxel_offset": [0, 0, 0],
            "size": [3153, 2259, 27],
        },
        {
            "encoding": "raw",
            "chunk_sizes": [[512, 512, 16]],
            "key": "8.96_8.96_70",
            "resolution": [8.96, 8.96, 70],
            "voxel_offset": [0, 0, 0],
            "size": [1577, 1130, 27],
        },
        {
            "encoding": "raw",
            "chunk_sizes": [[512, 512, 16]],
            "key": "17.92_17.92_70",
            "resolution": [17.92, 17.92, 70],
            "voxel_offset": [0, 0, 0],
            "size": [789, 565, 27],
        },
        {
            "encoding": "raw",
            "chunk_sizes": [[512, 512, 16]],
            "key": "35.84_35.84_70",
            "resolution": [35.84, 35.84, 70],
            "voxel_offset": [0, 0, 0],
            "size": [395, 283, 27],
        },
        {
            "encoding": "raw",
            "chunk_sizes": [[512, 512, 16]],
            "key": "71.68_71.68_70",
            "resolution": [71.68, 71.68, 70],
            "voxel_offset": [0, 0, 0],
            "size": [198, 142, 27],
        },
    ],
}
"""
precomputed://https://open-neurodata.s3.amazonaws.com/collman/collman15v2/EM25K
https://open-neurodata.s3.amazonaws.com/collman/collman15v2/EM25K/info
"""


def test_weiler():
    ms = Multiscale.from_precomputed(WEILER14)
    expected_paths = ("100_100_70", "200_200_70", "400_400_70", "800_800_70", "1600_1600_70")

    assert tuple(ms.keys()) == expected_paths


def test_collman():
    ms = Multiscale.from_precomputed(COLLMAN)
    expected_paths = (
        "2.24_2.24_70",
        "4.48_4.48_70",
        "8.96_8.96_70",
        "17.92_17.92_70",
        "35.84_35.84_70",
        "71.68_71.68_70",
    )

    assert tuple(ms.keys()) == expected_paths
