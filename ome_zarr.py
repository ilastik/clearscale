"""OME-Zarr helpers to ease interaction with Zarr stores"""

from typing import Any, Dict, Union, Tuple

from lazyflow.utility.io_util.clearscale._multiscale import GetShapeFunction


def make_all_singleton_shapes(ndim: int) -> GetShapeFunction:
    """
    Construct OME-Zarr Multiscale without accessing actual array shapes,
    when all datasets you expect have the same number of axes:
    `Multiscale.from_ome_zarr(ome_ms_dict, get_shape=make_all_singleton_shapes(ndim=3))`
    """
    return lambda _: (1,) * ndim


def make_fake_shapes(multiscale_spec: Dict[str, Any]) -> GetShapeFunction:
    """
    Construct OME-Zarr Multiscale without accessing actual array shapes:
    `Multiscale.from_ome_zarr(ome_ms_dict, get_shape=make_fake_shapes(ome_ms_dict)`

    Returns a fake get_shape(path) callable that makes an all-singletons shape for the smallest scale,
    and proportionally larger shapes for the others.
    """
    # Try to read metadata extremely permissively just to get *anything* that could work as a shape
    datasets = {ds["path"]: ds for ds in multiscale_spec["datasets"]}
    ndim = len(multiscale_spec.get("axes", [])) or 5  # OME-Zarr 0.1 and 0.2 without axes

    def scale_vector(ds: Dict[str, Any]) -> Union[None, Tuple[float, ...]]:
        cts = ds.get("coordinateTransformations", [])
        for t in cts:
            if t.get("type") == "scale":
                scale = tuple(float(x) for x in t.get("scale", ()))
                return scale
        return None

    smallest_ds = next(reversed(datasets.values()))  # OME-Zarr datasets must be largest-to-smallest
    smallest_scale = scale_vector(smallest_ds)

    def get_fake_shape(path: str) -> Tuple[int, ...]:
        ds = datasets[path]
        cur_scale = scale_vector(ds)
        try:
            return tuple(max(1, int(base / c)) for base, c in zip(cur_scale, smallest_scale))
        except TypeError:
            return (1,) * ndim

    return get_fake_shape


__all__ = ["make_fake_shapes"]
