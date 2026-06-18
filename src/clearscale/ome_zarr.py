"""OME-Zarr helpers to ease interaction with Zarr stores"""

from typing import Any, Dict, Union, Tuple

from clearscale._services.ome_zarr import (
    GetShapeFunction,
    ShapeSource,
    SUPPORTED_OME_ZARR_VERSIONS_READ,
    SUPPORTED_OME_ZARR_VERSIONS_WRITE,
)


def make_all_singleton_shapes(ndim: int) -> GetShapeFunction:
    """
    Construct OME-Zarr Multiscale without accessing actual array shapes,
    when all datasets you expect have the same number of axes:
    `Multiscale.from_ome_zarr(ome_ms_dict, shape_source=make_all_singleton_shapes(ndim=3))`
    """
    return lambda _: (1,) * ndim


def make_proportional_shapes(multiscale_spec: Dict[str, Any]) -> GetShapeFunction:
    """
    Construct OME-Zarr Multiscale without accessing actual array shapes:
    `Multiscale.from_ome_zarr(ome_ms_dict, shape_source=make_proportional_shapes(ome_ms_dict))`

    Returns a fake shape_source callable that makes an all-singletons shape for the smallest scale,
    and proportionally larger shapes for the others.
    """
    # Try to read metadata extremely permissively just to get *anything* that could work as a shape
    try:
        ds_by_path = {ds["path"]: ds for ds in multiscale_spec["datasets"]}
    except (TypeError, KeyError):
        # This spec is invalid and will bounce on ome_zarr validation anyway
        return lambda path: (1,) * 5
    ndim = len(multiscale_spec.get("axes", [])) or 5  # OME-Zarr 0.1 and 0.2 without axes

    def scale_vector(ds: Dict[str, Any]) -> Union[None, Tuple[float, ...]]:
        cts = ds.get("coordinateTransformations", [])
        if not isinstance(cts, list):
            return None
        for t in cts:
            if t.get("type") == "scale":
                scale = tuple(float(x) for x in t.get("scale", ()))
                return scale
        return None

    smallest_ds = next(reversed(ds_by_path.values()))  # OME-Zarr datasets must be largest-to-smallest
    smallest_scale: Tuple[float, ...] = scale_vector(smallest_ds) or (1.0,) * ndim

    def get_fake_shape(path: str) -> Tuple[int, ...]:
        cur_scale: Tuple[float, ...] = scale_vector(ds_by_path[path]) or (1.0,) * ndim
        try:
            return tuple(max(1, int(base / c)) for base, c in zip(cur_scale, smallest_scale))
        except TypeError:
            return (1,) * ndim

    return get_fake_shape


__all__ = [
    "GetShapeFunction",
    "ShapeSource",
    "make_all_singleton_shapes",
    "make_proportional_shapes",
    "SUPPORTED_OME_ZARR_VERSIONS_READ",
    "SUPPORTED_OME_ZARR_VERSIONS_WRITE",
]
