"""Transforms are not part of the public API"""

from clearscale._transforms._base import (
    RelativePath,
    CoordinateSystemName,
    FileRef,
    NodeRef,
    _UnresolvedRef,
    AnyRef,
    NodeSignature,
    TransformSignature,
    Transform,
    TransformGraph,
    IdentityTransform,
    TransformSequence,
    CoordinateSystem,
    TransformGraphNode,
    PRE_TRANSFORMS_VERSIONS,
    PRE_COLLECTIONS_VERSIONS,
    OmeZarrAxis,
    OmeZarrAxes,
)
from clearscale._transforms._transform_types import (
    AffineTransform,
    BijectionTransform,
    ByDimensionTransform,
    _ByDimensionChild,
    CoordinatesTransform,
    DisplacementsTransform,
    MapAxisTransform,
    ProjectAxisTransform,
    RotationTransform,
    ScaleTransform,
    TranslationTransform,
    IDENTITY_TOLERANCE,
)
from clearscale._transforms._to_from_spatial_relation import relation_to_transform_canonic

__all__ = ["FileRef", "OmeZarrAxes", "OmeZarrAxis"]
"""Transforms are generally not part of the public API.

FileRef is public via OmeZarrGroup.
OmeZarrAxes/OmeZarrAxis are for specifying axis properties specific to OME-Zarr for users familiar with the spec.
They are importable from `clearscale.ome_zarr.Axes`/`clearscale.ome_zarr.Axis`"""
