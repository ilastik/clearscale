"""OME-Zarr helpers to ease interaction with Zarr stores"""

from clearscale._services.ome_zarr import (
    GetShapeFunction,
    make_all_singleton_shapes,
    SUPPORTED_OME_ZARR_VERSIONS_READ,
    SUPPORTED_OME_ZARR_VERSIONS_WRITE,
    MultiscaleProperties,
    Omero,
    OmeroChannel,
    OmeroWindow,
    ImageLabel,
    LabelEntry,
    MissingAxisTypes,
)
from clearscale._transforms import OmeZarrAxis as Axis, OmeZarrAxes as Axes

__all__ = [
    "GetShapeFunction",
    "make_all_singleton_shapes",
    "SUPPORTED_OME_ZARR_VERSIONS_READ",
    "SUPPORTED_OME_ZARR_VERSIONS_WRITE",
    "MultiscaleProperties",
    "Omero",
    "OmeroChannel",
    "OmeroWindow",
    "ImageLabel",
    "LabelEntry",
    "MissingAxisTypes",
    "Axis",
    "Axes",
]
