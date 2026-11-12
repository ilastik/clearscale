# clearscale examples

These examples are meant to show where clearscale fits in existing image code: You keep using your preferred array and zarr libraries for data, and clearscale keeps the multiscale metadata aligned with the shapes you produce.

Most examples assume you have `numpy`, `scikit-image` and `zarr-python 3.*` (and of course `clearscale`) installed.

### Downsample a numpy array and save as OME-Zarr multiscale

```python
import os, numpy, skimage, zarr

# Placeholders - plug in any zarr backend and image scaling method
my_data = numpy.random.rand(3, 12, 512, 512)
do_my_scaling = skimage.transform.resize
zarr_group = zarr.open_group(os.path.expanduser("demo-output/plan_first.ome.zarr"), mode="w")

import clearscale as cs

# 1. Define operation plan (the scaling "blueprint")
shape = cs.Shape(zip("tzyx", my_data.shape))
scaling_blueprint = cs.BlueprintShapes.downscale_powers_of_2_xyz(
    base_shape=shape,
    rounding="ceil",
    limit_all=cs.Shape(z=8, y=128, x=128),
)

# 2. Scale data according to the metadata
for scale_key, target_shape in scaling_blueprint.items():
    # Target shape from blueprint
    scaled_data = do_my_scaling(my_data, target_shape.values_tuple())
    # Target array path from blueprint
    zarr_group.create_array(scale_key, data=scaled_data)

# 3. Write metadata: Scale -> Multiscale -> Group -> attrs
pixel_size = cs.PixelSize(t=5.0, z=260.0, y=0.53, x=0.53)
unit       = cs.Unit(t="s", z="micrometer", y="micrometer", x="micrometer")
base       = cs.Scale(shape, pixel_size, unit)

ms = cs.Multiscale.from_single(
    base,
    blueprint=scaling_blueprint,
    pixel_sizing="shape_ratio",  # skimage.transform.resize scales pixel size by the shape ratio...
    translating="half_pixel_space_preservation",  # ...and preserves the image's extent
)
group_meta = cs.OmeZarrGroup.from_single(ms).to_attrs(version="0.6")
zarr_group.attrs.update(group_meta)
```

### Download a single scale of a public OME-Zarr as a valid local OME-Zarr

```python
from pathlib import Path
import zarr  # This example assumes zarr-python 3.* is installed (plus fsspec, requests and aiohttp for https reading)
import clearscale as cs

URL = "https://s3.embl.de/i2k-2020/platy-raw.ome.zarr"
SCALE_KEY = "s6"
LOCAL_PATH = Path(f"demo-output/platy-raw-{SCALE_KEY}.ome.zarr")

# 1. Discover what's on the server
remote_group = zarr.open_group(URL)

remote_meta = cs.OmeZarrGroup.from_group(remote_group)
assert remote_meta.kind is cs.GroupKind.MULTISCALE, "This tells us there is exactly one Multiscale in this group"
assert remote_meta.version == "0.4", "OME-Zarr version is a property of the group, not of individual Multiscales"

remote_multiscale = remote_meta.multiscales[0]
assert SCALE_KEY in remote_multiscale, "Multiscale is dict-like, with the sub-paths to arrays as keys"
assert remote_multiscale[SCALE_KEY].shape == {"z": 357, "y": 405, "x": 430}

# 2. Create the local target and download the data
source_array = remote_group[SCALE_KEY]
local_group = zarr.open_group(str(LOCAL_PATH), mode="w", zarr_format=2)  # OME-Zarr v0.4 must use zarr-format v2
print(f"Downloading {SCALE_KEY} data...")
local_array = local_group.create_array(SCALE_KEY, data=source_array)

# 3. Derive a new Multiscale from the chosen Scale entry in the existing Multiscale
local_multiscale = remote_multiscale.derive(SCALE_KEY)

# 4. Write the new metadata to the downloaded store
local_group_meta = cs.OmeZarrGroup.from_single(local_multiscale)
local_group.attrs.update(local_group_meta.to_attrs(version="0.4"))
```

## Add metadata to existing arrays

```python
import clearscale as cs

# zarr_group can be zarr-python, TensorStore-backed code, or your own wrapper.
# The only thing clearscale needs here is each array's shape.
scale_keys = ("s0", "s1", "s2")
recorded_shapes = [
    (scale_key, cs.Shape(zip("zyx", zarr_group[scale_key].shape)))
    for scale_key in scale_keys
]

base_scale = cs.Scale(
    shape=recorded_shapes[0][1],
    pixel_size=cs.PixelSize(z=0.5, y=0.25, x=0.25),
    unit=cs.Unit(z="micrometer", y="micrometer", x="micrometer"),
)

blueprint = cs.BlueprintShapes(recorded_shapes)
multiscale = cs.Multiscale.from_single(
    base_scale,
    blueprint=blueprint,
    # How were these arrays scaled? These values describe interpolating resizing, like skimage.transform.resize.
    pixel_sizing="shape_ratio",
    translating="half_pixel_space_preservation",
)

zarr_group.attrs.update(
    cs.OmeZarrGroup.from_single(multiscale).to_attrs(version="0.5")
)
```

Array shapes cannot reveal how the arrays were scaled, so clearscale cannot guess `pixel_sizing` and `translating` for you.
If you did not run the scaling yourself, or are unsure how it behaves, [characterize it](characterization.md).
For methods that scale by an exact factor (binning, striding), use a `BlueprintFactors` instead.

## Reuse a multiscale's scaling pattern for a new image

```python
import clearscale as cs

# Let's assume you have some existing Multiscale (maybe loaded by Multiscale.from_ome_zarr)
template = cs.Multiscale(
    {
        "s0": cs.Scale(
            shape=cs.Shape(c=1, z=64, y=1024, x=1024),
            pixel_size=cs.PixelSize(c=1, z=0.5, y=0.25, x=0.25),
            unit=cs.Unit(c="", z="micrometer", y="micrometer", x="micrometer"),
        ),
        "s1": cs.Scale(
            shape=cs.Shape(c=1, z=32, y=512, x=512),
            pixel_size=cs.PixelSize(c=1, z=1.0, y=0.5, x=0.5),
            unit=cs.Unit(c="", z="micrometer", y="micrometer", x="micrometer"),
        ),
        "s2": cs.Scale(
            shape=cs.Shape(c=1, z=16, y=256, x=256),
            pixel_size=cs.PixelSize(c=1, z=2.0, y=1.0, x=1.0),
            unit=cs.Unit(c="", z="micrometer", y="micrometer", x="micrometer"),
        ),
    }
)

target_base = cs.Scale(
    shape=cs.Shape(t=65, c=3, z=40, y=2048, x=2048),
    pixel_size=cs.PixelSize(t=0.5, c=1.0, z=0.7, y=0.1, x=0.1),
    unit=cs.Unit(t="seconds", c="", z="micrometer", y="micrometer", x="micrometer"),
)

blueprint = cs.BlueprintShapes.from_multiscale_rescaled(
    template,
    target_shape=target_base.shape,
    rounding="ceil",
).with_axes("tczyx")

# Describe how the new image's data will be scaled. These values are for interpolating resizing.
target_multiscale = blueprint.apply_to_scale(
    target_base,
    pixel_sizing="shape_ratio",
    translating="half_pixel_space_preservation",
)

assert target_multiscale["s0"].shape == cs.Shape(t=65, c=3, z=40, y=2048, x=2048)
assert target_multiscale["s1"].shape == cs.Shape(t=65, c=3, z=20, y=1024, x=1024)
assert target_multiscale["s2"].shape == cs.Shape(t=65, c=3, z=10, y=512, x=512)
```

You can also restrict the inherited scaling pattern to selected axes:

```python
xy_only = cs.BlueprintShapes.from_multiscale_rescaled(
    template,
    target_shape=target_base.shape,
    rounding="ceil",
    scaled_axes="yx",
)

assert xy_only["s2"] == cs.Shape(t=65, c=3, z=40, y=512, x=512)  # z not scaled
```

Or rebase from a non-root scale when your input data is already downsampled:

```python
rebased = cs.BlueprintShapes.from_multiscale_rescaled(
    template,
    target_shape=cs.Shape(c=3, z=20, y=1024, x=1024),
    source_key="s1",
    rounding="ceil",
)

assert rebased["s0"] == cs.Shape(c=3, z=40, y=2048, x=2048)
assert rebased["s1"] == cs.Shape(c=3, z=20, y=1024, x=1024)
assert rebased["s2"] == cs.Shape(c=3, z=10, y=512, x=512)
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
import clearscale as cs
from clearscale.characterization import half_pixel_space_preservation

base = cs.Scale(
    shape=cs.Shape(y=1001, x=1536),
    pixel_size=cs.PixelSize(y=0.3, x=0.3),
    unit=cs.Unit(y="micrometer", x="micrometer"),
)
blueprint = cs.BlueprintShapes.uniform_steps(
    step=2,
    base_shape=base.shape,
    rounding="round",
    limit_all=cs.Shape(y=250, x=384),
)

# The metadata that the extent-preserving resampling should produce:
multiscale = cs.Multiscale.from_single(
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

To turn a matrix and offset that you already have into an `Affine` between two Scales, and vice versa, see the section on [Matrices in pixel index space](space.md#matrices-in-pixel-index-space).
