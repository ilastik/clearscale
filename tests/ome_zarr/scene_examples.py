import copy

import pytest

SCENE_STITCHING = {
    "coordinateTransformations": [
        {
            "type": "translation",
            "output": {"name": "world"},
            "input": {"path": "tile_0", "name": "physical"},
            "translation": [0, 0],
            "name": "tile_0_mm to world",
        },
        {
            "type": "translation",
            "output": {"name": "world"},
            "input": {"path": "tile_1", "name": "physical"},
            "translation": [0, 348],
            "name": "tile_1_mm to world",
        },
        {
            "type": "translation",
            "output": {"name": "world"},
            "input": {"path": "tile_2", "name": "physical"},
            "translation": [276, 0],
            "name": "tile_2_mm to world",
        },
        {
            "type": "translation",
            "output": {"name": "world"},
            "input": {"path": "tile_3", "name": "physical"},
            "translation": [276, 348],
            "name": "tile_3_mm to world",
        },
    ],
    "coordinateSystems": [
        {
            "name": "world",
            "axes": [
                {"type": "space", "name": "x", "unit": "micrometer", "discrete": False},
                {"type": "space", "name": "y", "unit": "micrometer", "discrete": False},
            ],
        }
    ],
}


@pytest.fixture
def scene_stitching():
    return copy.deepcopy(SCENE_STITCHING)
