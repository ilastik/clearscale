from clearscale import Scene

from tests.ome_zarr.scene_examples import scene_stitching


def test_stitching_example(scene_stitching):
    scene = Scene.from_ome_zarr(scene_stitching)
    assert scene.unresolved_paths == ["tile_0", "tile_1", "tile_2", "tile_3"]
