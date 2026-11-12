"""Drop-in alternative to ``import clearscale`` for code that only uses ``str`` axis keys.

The axis value classes (``Shape``, ``PixelSize``, ``Factor``, ``Unit``, ``PixelOffset``, ``Translation``) are generic in
their axis key type. This module re-exports everything from `clearscale`, but with those classes already specialized to
``str`` keys, so you don't have to annotate ``Shape[str]`` everywhere::

    import clearscale.str_axis_keys as cs

    def crop(shape: cs.Shape, ...) -> cs.Shape:  # a Shape[str], also a Mapping[str, int]
        ...

Note that these names are specialized generic aliases and therefore cannot be used in ``isinstance`` checks. Use the
unspecialized classes from `clearscale` for that: ``isinstance(x, clearscale.Shape)``.
"""

from typing import TypeAlias

from clearscale import _axis_values as _av
from clearscale import (
    __version__,
    SpatialRelation,
    PermutationTo,
    ProjectionTo,
    AxisRearrangementTo,
    AxisRenaming,
    SameSpace,
    SpatialRelationSequence,
    BlueprintFactors,
    BlueprintShapes,
    DuplicatePolicy,
    Multiscale,
    ome_zarr,
    OmeZarrGroup,
    ChildRef,
    GroupKind,
    Scale,
    Scene,
    Affine,
    Coefficient,
    Linear,
)

Factor: TypeAlias = _av.Factor[str]
PixelOffset: TypeAlias = _av.PixelOffset[str]
PixelSize: TypeAlias = _av.PixelSize[str]
Shape: TypeAlias = _av.Shape[str]
Translation: TypeAlias = _av.Translation[str]
Unit: TypeAlias = _av.Unit[str]

__all__ = [
    "__version__",
    "SpatialRelation",
    "PermutationTo",
    "ProjectionTo",
    "AxisRearrangementTo",
    "AxisRenaming",
    "SameSpace",
    "SpatialRelationSequence",
    "BlueprintFactors",
    "BlueprintShapes",
    "DuplicatePolicy",
    "Factor",
    "Multiscale",
    "ome_zarr",
    "OmeZarrGroup",
    "ChildRef",
    "GroupKind",
    "PixelOffset",
    "PixelSize",
    "Scale",
    "Scene",
    "Shape",
    "Translation",
    "Unit",
    "Affine",
    "Coefficient",
    "Linear",
]
