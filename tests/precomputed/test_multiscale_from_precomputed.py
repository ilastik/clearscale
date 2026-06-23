from clearscale import Multiscale, PixelSize


def test_accepts_degenerate_meta():
    degenerate_meta = {
        "type": "image",
        "data_type": "uint16",
        "num_channels": 1,
        "scales": [{"key": "foo", "chunk_sizes": [[1, 1, 1]], "resolution": [0, 0, 0], "size": [1, 1, 1]}],
    }

    ms = Multiscale.from_precomputed(degenerate_meta)
    assert ms["foo"].pixel_size == PixelSize.identity("czyx")
