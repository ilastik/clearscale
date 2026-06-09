# clearscale

`clearscale` is a small pure-Python package for keeping multi-scale image metadata explicit. It models ordered,
immutable axis metadata, scale levels, multiscale blueprints, OME-Zarr metadata, Precomputed metadata, and
coordinate transforms.

## Install

```bash
python -m pip install -e .
```

## Quick Examples

```python
from clearscale import (
    BlueprintFactors,
    BlueprintShapes,
    DuplicatePolicy,
    Factor,
    Multiscale,
    PixelOffset,
    PixelSize,
    Scale,
    Shape,
    Translation,
    Unit,
)

# Shape: integer axis sizes (order matters!)
shape = Shape(y=1024, x=2048)

# Factor: relative scale divisors.
factor = Factor.uniform(shape.keys(), 2)
half_shape = shape.scaled_by(factor, rounding="ceil")

# PixelSize and Unit: physical spacing metadata.
pixel_size = PixelSize(y=0.5, x=0.5)
unit = Unit(y="um", x="um")

# PixelOffset and Translation: crops and their physical representation.
offset = PixelOffset(y=10, x=20)
translation = offset.to_physical(pixel_size)
manual_translation = Translation(y=5.0, x=10.0)

# Scale: one image level with shape and physical metadata.
base = Scale(shape=shape, pixel_size=pixel_size, unit=unit, translation=translation)

# BlueprintShapes: name target shapes for a pyramid.
shape_blueprint = BlueprintShapes.uniform_steps(
    base_shape=shape,
    step=2,
    rounding="ceil",
    only="yx",
    max_levels=3,
    on_duplicate=DuplicatePolicy.KEEP_FIRST,
)

# BlueprintFactors: derive or apply relative scaling factors.
factor_blueprint = BlueprintFactors.from_shapes(shape_blueprint, reference=shape)

# Multiscale: expand a base scale from a blueprint.
multiscale = Multiscale.from_shapes(shape_blueprint, base=base)
```

`Multiscale.from_ome_zarr(...)` accepts OME-Zarr multiscale metadata and a `get_shape(path)` callable.
`Multiscale.from_precomputed(...)` accepts a Neuroglancer Precomputed `info` dictionary.
