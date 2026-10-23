# clearscale examples

These examples are meant to show where clearscale fits in existing image code: You keep using your preferred array and zarr libraries for data, and clearscale keeps the multiscale metadata aligned with the shapes you produce.

## Add metadata to existing arrays

```python
from clearscale import BlueprintShapes, PixelSize, Scale, Shape, Unit, Multiscale, OmeZarrGroup

# zarr_group can be zarr-python, TensorStore-backed code, or your own wrapper.
# The only thing clearscale needs here is each array's shape.
scale_keys = ("s0", "s1", "s2")
recorded_shapes = [
    (scale_key, Shape(zip("zyx", zarr_group[scale_key].shape)))
    for scale_key in scale_keys
]

base_scale = Scale(
    shape=recorded_shapes[0][1],
    pixel_size=PixelSize(z=0.5, y=0.25, x=0.25),
    unit=Unit(z="micrometer", y="micrometer", x="micrometer"),
)

blueprint = BlueprintShapes(recorded_shapes)
multiscale = Multiscale.from_single(base_scale, blueprint=blueprint)

zarr_group.attrs.update(
    OmeZarrGroup.from_single(multiscale).to_attrs(version="0.5")
)
```

## Reuse a multiscale's scaling pattern for a new image

```python
from clearscale import BlueprintShapes, Multiscale, PixelSize, Scale, Shape, Unit

# Let's assume you have some existing Multiscale (maybe loaded by Multiscale.from_ome_zarr)
template = Multiscale(
    {
        "s0": Scale(
            shape=Shape(c=1, z=64, y=1024, x=1024),
            pixel_size=PixelSize(c=1, z=0.5, y=0.25, x=0.25),
            unit=Unit(c="", z="micrometer", y="micrometer", x="micrometer"),
        ),
        "s1": Scale(
            shape=Shape(c=1, z=32, y=512, x=512),
            pixel_size=PixelSize(c=1, z=1.0, y=0.5, x=0.5),
            unit=Unit(c="", z="micrometer", y="micrometer", x="micrometer"),
        ),
        "s2": Scale(
            shape=Shape(c=1, z=16, y=256, x=256),
            pixel_size=PixelSize(c=1, z=2.0, y=1.0, x=1.0),
            unit=Unit(c="", z="micrometer", y="micrometer", x="micrometer"),
        ),
    }
)

target_base = Scale(
    shape=Shape(t=65, c=3, z=40, y=2048, x=2048),
    pixel_size=PixelSize(t=0.5, c=1.0, z=0.7, y=0.1, x=0.1),
    unit=Unit(t="seconds", c="", z="micrometer", y="micrometer", x="micrometer"),
)

blueprint = BlueprintShapes.from_multiscale_rescaled(
    template,
    target_shape=target_base.shape,
    rounding="ceil",
).with_axes("tczyx")

target_multiscale = blueprint.apply_to_scale(target_base)

assert target_multiscale["s0"].shape == Shape(t=65, c=3, z=40, y=2048, x=2048)
assert target_multiscale["s1"].shape == Shape(t=65, c=3, z=20, y=1024, x=1024)
assert target_multiscale["s2"].shape == Shape(t=65, c=3, z=10, y=512, x=512)
```

You can also restrict the inherited scaling pattern to selected axes:

```python
xy_only = BlueprintShapes.from_multiscale_rescaled(
    template,
    target_shape=target_base.shape,
    rounding="ceil",
    scaled_axes="yx",
)

assert xy_only["s2"] == Shape(t=65, c=3, z=40, y=512, x=512)  # z not scaled
```

Or rebase from a non-root scale when your input data is already downsampled:

```python
rebased = BlueprintShapes.from_multiscale_rescaled(
    template,
    target_shape=Shape(c=3, z=20, y=1024, x=1024),
    source_key="s1",
    rounding="ceil",
)

assert rebased["s0"] == Shape(c=3, z=40, y=2048, x=2048)
assert rebased["s1"] == Shape(c=3, z=20, y=1024, x=1024)
assert rebased["s2"] == Shape(c=3, z=10, y=512, x=512)
```

## Make `scipy.ndimage.affine_transform` match the metadata

`affine_transform` knows nothing about pixel sizes: Its `matrix` and `offset` alone decide where the output samples lie (`index_in = matrix * index_out + offset`).
With `matrix` equal to the scaling factor and `offset=0`, the output samples start at the first input sample and are spaced exactly `factor` apart (see [characterization](characterization.md)).
Resampling that preserves the extent of the data (like `skimage.transform.resize` does) needs the shape ratio as the matrix, and a half-pixel offset.

Instead of deriving these numbers by hand, clearscale can compute the metadata of the behaviour you want.
You can then take the arguments from the Scales, although a little additional arithmetic is still needed.

Source and output describe the same physical space, so for every axis, the source index of an output pixel is
`matrix * index_out + offset` with
* `matrix = target.pixel_size / base.pixel_size` and 
* `offset = (target.translation - base.translation) / base.pixel_size`

```python
import numpy as np
from clearscale import BlueprintShapes, Multiscale, PixelSize, Scale, Shape, Unit
from clearscale.characterization import half_pixel_space_preservation

base = Scale(
    shape=Shape(y=1001, x=1536),
    pixel_size=PixelSize(y=0.3, x=0.3),
    unit=Unit(y="micrometer", x="micrometer"),
)
blueprint = BlueprintShapes.uniform_steps(
    step=2,
    base_shape=base.shape,
    rounding="round",
    limit_all=Shape(y=250, x=384),
)

# The metadata that the extent-preserving resampling should produce:
multiscale = Multiscale.from_single(
    base,
    blueprint=blueprint,
    pixel_sizing="shape_ratio",
    translating=half_pixel_space_preservation,
)

print(multiscale)
# Multiscale(
#     "s0" (y: 1001, x: 1536) at pixel size: y: 0.3 µm, x: 0.3 µm
#     "s1" (y: 500, x: 768) at pixel size: y: 0.6006 µm, x: 0.6 µm
#     "s2" (y: 250, x: 384) at pixel size: y: 1.2012 µm, x: 1.2 µm
# )

for scale_key, target in multiscale.items():
    shift = target.translation - base.translation
    matrix = (target.pixel_size / base.pixel_size).values_tuple()
    offset = tuple(shift[axis] / base.pixel_size[axis] for axis in shift)

    # Data scaling:
    # scaled = ndimage.affine_transform(
    #     raw_yx, matrix, offset=offset, output_shape=target.shape.values_tuple(), order=1
    # )

    if scale_key == "s1":
        assert np.allclose(matrix, (2.002, 2.0))
        assert np.allclose(offset, (0.501, 0.5))
```

Note the `2.002`: The base shape of 1001 does not halve evenly, so the output has 500 rows and the matrix is the exact ratio `1001 / 500`, not `2`.
Matrix and offset follow from the Scales, so data and metadata cannot drift apart.
Choose another `translating` function (or `pixel_sizing`) and the arguments change accordingly.

`affine_transform` interpolates between neighboring input samples, but it does not low-pass filter first, so strong downscaling will alias.
You would most likely want to smooth the data before you scale it, e.g. using `ndimage.gaussian_filter`.

For the other direction, i.e. turning a matrix and offset that you already have into an `Affine` between two Scales, see the section on [Matrices in pixel index space](space.md#matrices-in-pixel-index-space).
