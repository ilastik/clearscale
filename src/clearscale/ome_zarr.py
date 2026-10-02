"""Parts of the public API that are specific to OME-Zarr"""

from clearscale._services.ome_zarr import (
    GetShapeFunction,
    make_all_singleton_shapes,
    SUPPORTED_VERSIONS_READ,
    SUPPORTED_VERSIONS_WRITE,
    MultiscaleProperties,
    Omero,
    OmeroChannel,
    OmeroWindow,
    ImageLabel,
    LabelEntry,
)
from clearscale._transforms import OmeZarrAxis as Axis, OmeZarrAxes as Axes

__all__ = [
    "GetShapeFunction",
    "make_all_singleton_shapes",
    "SUPPORTED_VERSIONS_READ",
    "SUPPORTED_VERSIONS_WRITE",
    "MultiscaleProperties",
    "Omero",
    "OmeroChannel",
    "OmeroWindow",
    "ImageLabel",
    "LabelEntry",
    "Axis",
    "Axes",
]
