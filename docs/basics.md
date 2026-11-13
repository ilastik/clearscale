# clearscale basics

clearscale is designed to sit next to the code that already reads, writes, scales, crops, or reorders your image arrays.

The core idea is simple: always keep axis keys with the values they describe, in the order that applies to your data.

## At a glance

If you've been coding for a bit already, you most likely just need to know:
* Shape, Factor, PixelSize, Unit etc. are immutable, ordered `Mapping[AxisKey, <int/float/str respectively>]`
* The mappings are generic over axis key type:
  * You can narrow e.g. `Shape[str]` to match `Mapping[str, int]`.
  * You can use any hashable type as key.
  * For interoperability, your keys should be, or should str-convert to, single lowercase letters ("t", "c", "z", "y", "x"), like in the examples throughout these docs. Other keys and axis orders are generally badly supported in the OME-Zarr ecosystem.
* `.with_axes(new_axis_keys)` reorders, inserts and drops as needed. Each class has its respective default value for inserted axes (`1` for Shape, `""` for Unit etc.)
* Mathematical operations are axis-wise (PixelSize * Factor, etc.)
* Blueprints and Multiscale are immutable, ordered `Mapping[ScaleKey, <Shape/Factor/Scale respectively>]`
* Multiscale is the central (smart) metadata class; Scale is the dumb bag-of-metadata you use to describe an individual data array

Since OME-Zarr is the only serialization target, its minimum restrictions apply to Multiscale construction (so you don't end up with a Multiscale that you then can't write as valid OME-Zarr):

* 2 to 5 axes, not more or less
* Exactly 2 or 3 axes of type "space"
* Axes that have types must be ordered "time-channel-space" by type
* Scale keys must be valid relative path strings
* Scale shape must not increase across the Multiscale (highest-resolution must come first)

Potential gotchas:

* When describing relative scaling, `Factor` is a *multiplier* for `PixelSize` and a *divisor* for `Shape`. `Factor(x=2)` means "downscale by factor 2" - doubling pixel size and halving image shape. Dividing `Shape / Factor` needs to know how your scaling method rounds uneven divisions, so can only be called as `shape.scaled_by(factor, rounding="ceil")`. Note that many scaling methods like `scipy.ndimage.zoom` expect *shape multiplier* factors, so you'd need to pass `factor.inverted().values_tuple()`.
* Scaled pixel size cannot be trivially determined. 
    `shape_ratio` is common (`output_spacing = input_spacing * input_shape / output_shape`), but some methods use `"corner_ratio"` or `"exact_factor"`.
    There is no default, because there is no way to guess which one applies: Shapes do not reveal how an array was scaled. Only `BlueprintFactors` can express `"exact_factor"`, because shapes do not contain the factor.
* Scaling usually introduces a shift, or `Translation`, to the scaled image's origin in physical space. Methods that handle `Scale` objects (and hence `Scale.translation`) compute this using a `TranslationShiftFunction`

All three characteristics (`rounding`, `pixel_sizing` and `translating`) can be determined for your particular method using [`clearscale.characterization`](characterization.md).

## Axis values: Dicts are better than tuples

Most clearscale types are immutable ordered mappings:

```python
import clearscale as cs

# Use whichever dict construction pattern you're familiar with
shape1 = cs.Shape(c=3, x=1024, y=1024)
shape3 = cs.Shape(zip("xcy", (1024, 3, 1024)))
shape2 = cs.Shape({"y": 1024, "x": 1024, "c": 3})

assert shape1 != shape2, "The difference between plain dicts and Shape: Axis order matters"

# Iterate like a regular dict (Shape implements Mapping)
for axis, size in shape2.items():
    print(f"{size} pixels along {axis}")

# But immutable
# shape1["x"] = 5  # would raise TypeError: 'Shape' object does not support item assignment
```

Adapting existing code with clearscale is simple if you are already keeping track of axes and values using dicts:

```
scaling_limit = {"x": 1, "y": 1, "z": 120}
limit_shape = Shape(scaling_limit)
```

Same for the other primitives. clearscale calls these "axis values":

```python
import clearscale as cs

pixel_size  = cs.PixelSize(zip("cyx", (1.0, 25.0, 25.0)))      # axis -> float
crop_offset = cs.PixelOffset(t=20)                             # axis -> int
translation = cs.Translation(y=0.5, x=0.5)                     # axis -> float
factor      = cs.Factor(x=2.0, y=2.0, z=12.0)                  # axis -> float
unit        = cs.Unit({"t": "seconds", "x": "nm", "y": "nm"})  # axis -> str
```

Axis keys don't have to be plain strings.
You can use any hashable object, such as frozen dataclass instances.
Or maybe you just want syntactic sugar like `x = "x"` to enable `my_shape[x]` instead of `my_shape["x"]`.

Axis keys do need to support conversion to string with `str(axis_key)` for OME-Zarr export.

### Tracking metadata across axis rearrangements

All axis values implement `.with_axes(target_axes)`.

Permutation / transposition:

```python
import clearscale as cs
import numpy as np

image = np.zeros((3, 40, 512, 512))
source_axes = "czyx"
target_axes = "xyzc"

# Data permutation
permuted_indices = [list(source_axes).index(axis) for axis in target_axes]
transposed = image.transpose(permuted_indices)

# Metadata permutation
source_shape = cs.Shape(zip(source_axes, image.shape))
target_shape = source_shape.with_axes(target_axes)

assert transposed.shape == target_shape.values_tuple()
```

Dropping and inserting an axis:

```python
import clearscale as cs
import numpy as np

time_series = np.zeros((25, 128, 128))
source_axes = "tyx"
target_axes = "cyx"

# Data
with_channel_data = time_series[None, 0, :, :]

# Metadata
plane_shape = cs.Shape(zip(source_axes, time_series.shape))
# with_axes removes t and detects c as a new axis key. It inserts the default Shape value for c (1)
with_channel_shape = plane_shape.with_axes(target_axes)

assert with_channel_data.shape == with_channel_shape.values_tuple()
```

Likewise for the other axis values:

```python
import clearscale as cs

pixel_size  = cs.PixelSize(x=25.0, y=25.0).with_axes("xyc")  # pixel_size["c"]  == 1.0
crop_offset = cs.PixelOffset(x=5, y=3).with_axes("xyc")      # crop_offset["c"] == 0
translation = cs.Translation(x=4.0, y=3.0).with_axes("xyc")  # translation["c"] == 0.0
factor      = cs.Factor(x=2.0, y=2.0).with_axes("xyc")       # factor["c"]      == 1.0
unit        = cs.Unit(x="cm", y="cm").with_axes("xyc")       # unit["c"]        == ""
```

Most common manipulations should be supported with expressively named methods on clearscale objects.
Worst-case scenario, construction by dict comprehension is the escape hatch:

```python
import clearscale as cs

shape1 = cs.Shape(x=1024, y=1024, z=4)
shape2 = cs.Shape(x=905, y=744)
shape3 = cs.Shape(z=24)

custom_derived_shape = cs.Shape({
    axis: (
        shape1[axis] + (
            shape2[axis] if axis in shape2 else shape3[axis]  # wild
        )
    )
    for axis in shape1.keys()
})
```

Though clearscale aims to support all manipulations.
If you find yourself needing to manually construct dicts, please consider opening an issue / feature request :)

### Axis-wise operations

```python
import clearscale as cs

pixel_size = cs.PixelSize(z=0.5, y=0.25, x=0.25)
downsample_by_2_yx = cs.Factor(y=2, x=2)

assert pixel_size * downsample_by_2_yx == cs.PixelSize(z=0.5, y=0.5, x=0.5), "Factors are multipliers for pixel size"

offset = cs.PixelOffset(y=32, x=48)

assert offset * pixel_size == cs.Translation(y=8.0, x=12.0), "Pixel size converts pixel offsets to physical translations"

shape = cs.Shape(z=40, y=125, x=125)

# scaled_shape = shape * downsample_xy  # Would raise TypeError. Factors are divisors for Shape

# scaled_shape = shape / downsample_xy  # Would still raise TypeError, because...

# you need to specify how your scaling implementation handles rounding when the Factor does not evenly divide the Shape:
assert shape.scaled_by(downsample_by_2_yx, rounding="ceil") == cs.Shape(z=40, y=63, x=63)
```

Missing axes in a `Factor` default to identity (`1.0`), so `z` is unchanged in this `shape.scaled_by` example.
If the factor had extra axes, it would be rejected because they would describe metadata for data that is not there.

Same-kind arithmetic, such as `Translation(...) + Translation(...)`, requires identical axes in identical order:
```python
import clearscale as cs

# Shapes with identical axes can be compared
shape = cs.Shape(z=40, y=125, x=125)
downsampled_shape = cs.Shape(z=20, y=25, x=25)

assert shape / downsampled_shape           == cs.Factor(z=2.0, y=5.0, x=5.0)  # Can be hard to remember which way to divide :)
assert shape.scaling_to(downsampled_shape) == cs.Factor(z=2.0, y=5.0, x=5.0)  # Explicit phrasing might be more intuitive

# Translations can be added or subtracted, but only with identical axes
crop1_transl = cs.Translation(y=8.0, x=12.0)
crop2_transl = cs.Translation(y=4.0)

# total_crop = crop1_transl + crop2_transl  # would raise ValueError

# If metadata are sourced from separate places with different axes, bringing them together needs to be explicit:
total_crop = crop1_transl.with_axes("cyx") + crop2_transl.with_axes("cyx")

assert total_crop == cs.Translation(c=0.0, y=12.0, x=12.0)
```

## Immutability

The original objects used in any of these operations are never modified; a new object is returned:

```python
import clearscale as cs

shape = cs.Shape(x=5, y=5)
reordered = shape.with_axes("czyx")
assert shape == cs.Shape(x=5, y=5), "original Shape is not modified"
assert reordered is not shape, "with_axes returns a new instance"
```

## Going multiscale: Dicts of dicts

`BlueprintShapes`, `BlueprintFactors` and `Multiscale` also mostly behave like immutable ordered mappings.
clearscale calls these "scale mappings".

```python
import clearscale as cs

bps = cs.BlueprintShapes({"s0": cs.Shape(x=1, y=1)})       # scale_key -> Shape

bpf = cs.BlueprintFactors({"s0": cs.Factor(x=2.0, y=2.0)}) # scale_key -> Factor

ms  = cs.Multiscale({"s0": cs.Scale(shape=...)})           # scale_key -> Scale
```

Scale keys are relative paths that point to data arrays, so they are naturally plain strings.

Blueprints are simple but flexible containers that specify what scaling has been done or will be done.
The idea is to use their values as parameters for scaling, to ensure metadata and data operations stay synchronised.

### Scaling by target shape

```python
import clearscale as cs

base = cs.Scale(
    shape=cs.Shape(z=40, y=512, x=512),
    pixel_size=cs.PixelSize(z=0.5, y=0.25, x=0.25),
    unit=cs.Unit(z="micrometer", y="micrometer", x="micrometer"),
)

blueprint = cs.BlueprintShapes.uniform_steps(
    step=2,
    scaled_axes="yx",
    base_shape=base.shape,
    rounding="ceil",
    limit_all=cs.Shape(y=128, x=128),
)

# Scale your data using this shape blueprint:
#for scale_key, target_shape in blueprint.items():
#    scaled_data = do_my_scaling(raw_data, target_shape.values_tuple())

multiscale = cs.Multiscale.from_single(
    base,
    blueprint=blueprint,
    pixel_sizing="shape_ratio",
    translating="half_pixel_space_preservation",
)

assert list(multiscale.keys()) == ["s0", "s1", "s2"]
assert multiscale["s2"].shape == cs.Shape(z=40, y=128, x=128)
assert multiscale["s2"].pixel_size == cs.PixelSize(z=0.5, y=1.0, x=1.0)
```

The shapes alone do not say how your data was scaled, so you have to describe your scaling method with `pixel_sizing` and `translating`.
The values above multiply the pixel size by the `input_shape / output_shape` ratio (`shape_ratio`), and shift the translation so that the scaled image covers the same physical extent as the base (`half_pixel_space_preservation`).
This is how interpolating resize methods like `skimage.transform.resize` behave.

Wrong values can produce valid-looking but inaccurate metadata.
You can automatically determine the appropriate parameters for your specific scaling method using [scaling method characterization](characterization.md).

### Scaling by factors

Some scaling methods take a factor instead of a target shape.
There are two conventions, as described in [Scaling methods](characterization.md):

| Input to your scaling method | Meaning                    | Example: downscale by 2                                         |
|------------------------------|----------------------------|-----------------------------------------------------------------|
| **Shape factor**             | Multiplies the input shape | `0.5`, as in `scipy.ndimage.zoom`                               |
| **Step factor**              | Divides the input shape    | `2`, as in striding or `skimage.transform.downscale_local_mean` |

A clearscale `Factor` always uses the **step factor** convention: `Factor(y=2, x=2)` means halve the shape along `y` and `x`.
For a method that expects shape factors, pass `factor.inverted().values_tuple()`.
The factor convention alone does not tell you the output pixel size or translation; those still depend on the scaling method.

Use `BlueprintFactors` to plan scaling in either convention.
Its constructors generate exact factors directly, while using `base_shape` and `rounding` to decide how many levels fit your limits:

* `uniform_steps`: powers of a chosen step along selected axes.
* `downscale_powers_of_2_xyz`: the common case of factors 1, 2, 4, ... along spatial axes named `x`, `y`, and `z`.
* `adaptive_steps` and `adaptive_powers_of_2_xyz`: use the base pixel sizes to bring anisotropic data toward isotropy before continuing to scale all selected axes.

Each factor describes scaling **from the base image**, not from the preceding level.
With `step=2`, the factors are 1, 2, 4, ...; use each one on the original data.

### Step factors: striding

For example, plain striding (`data[::2]`) keeps the first, and then every other input sample.
It rounds the output shape up, multiplies the pixel spacing by exactly 2, and leaves the first sample in place.
These properties correspond to `rounding="ceil"`, `pixel_sizing="exact_factor"`, and `translating="first_value_decimation"`.

```python
import clearscale as cs

base = cs.Scale(
    shape=cs.Shape(z=40, y=513, x=513),
    pixel_size=cs.PixelSize(z=0.5, y=0.25, x=0.25),
    unit=cs.Unit(z="micrometer", y="micrometer", x="micrometer")
)
factors = cs.BlueprintFactors.uniform_steps(
    step=2,
    scaled_axes="yx",
    base_shape=base.shape,
    rounding="ceil",
    limit_all=cs.Shape(y=129, x=129)
)

assert factors["s0"] == cs.Factor(z=1, y=1, x=1)
assert factors["s2"] == cs.Factor(z=1, y=4, x=4)

# Apply each factor to the original munpy array in zyx order:
# for scale_key, factor in factors.items():
#     slices = tuple(slice(None, None, int(step)) for step in factor.values())
#     scaled_data = raw_data[slices]
#     ... save scaled_data at scale_key ...

multiscale = cs.Multiscale.from_single(
    base, 
    blueprint=factors, 
    rounding="ceil", 
    pixel_sizing="exact_factor", 
    translating="first_value_decimation"
)

assert tuple(multiscale) == ("s0", "s1", "s2")
assert multiscale["s2"].shape == cs.Shape(z=40, y=129, x=129)
assert multiscale["s2"].pixel_size == cs.PixelSize(z=0.5, y=1.0, x=1.0)
assert multiscale["s2"].translation == base.translation
```

The integer conversion in the slicing example is appropriate because this blueprint generates integer powers of 2.
Other factor-based methods can accept fractional factors.

Notice the uneven division: `513 / 4` rounds up to 129 pixels, but the output spacing is still exactly 4 times the input spacing.
The shape ratio `513 / 129` is not 4.
Converting a shape blueprint with `.to_factors()` would recover that ratio, not the original stride.
Use `BlueprintFactors` when the factors themselves are part of the operation you need to describe.

### Shape factors: interpolating zoom

For a method that takes shape factors, keep step factors in the blueprint and invert them when calling the method.
Using the same `base`, this example describes `scipy.ndimage.zoom(..., order=1, grid_mode=False)`:

```python
factors = cs.BlueprintFactors.uniform_steps(
    step=2,
    scaled_axes="yx",
    base_shape=base.shape, 
    rounding="round",
    limit_all=cs.Shape(y=129, x=129)
)

assert factors["s2"].inverted().values_tuple() == (1.0, 0.25, 0.25)

# from scipy.ndimage import zoom
# for scale_key, factor in factors.items():
#     scaled_data = zoom(raw_data, factor.inverted().values_tuple(), order=1, grid_mode=False)
#     ... save scaled_data at scale_key ...

multiscale = cs.Multiscale.from_single(
    base,
    blueprint=factors,
    rounding="round",
    pixel_sizing="corner_ratio",
    translating="first_value_decimation"
)

assert multiscale["s2"].shape == cs.Shape(z=40, y=128, x=128)
```

With `zoom(..., grid_mode=False)`, the pixel spacing follows the first-to-last sample distance (`corner_ratio`), rather than the exact requested factor.
Passing factors to a scaling function does not by itself imply `pixel_sizing="exact_factor"`.

### Choosing the scaling parameters

Applying a `BlueprintFactors` to a `Scale` requires all three characteristics of your scaling method:

* `rounding`: how fractional output lengths become integer shapes. Use the same rounding when generating the blueprint and applying it to the base scale.
* `pixel_sizing`: how the spacing between samples changes.
* `translating`: how the first output sample moves relative to the base image's origin.

The blueprint constructor only uses `rounding` to project shapes for stopping limits and duplicate detection; it does not store the scaling method's behavior.
That is why you supply it again when creating the `Multiscale`.

If your method accepts an output shape, use `BlueprintShapes` as in the earlier example; applying it does not need `rounding` because the shapes are already known.
Use [scaling method characterization](characterization.md) to determine the parameters for your actual function and options.

### Axis types

clearscale enforces OME-Zarr's axis requirements whenever a `Multiscale` is created.

This is to avoid situations where you created a Multiscale and then cannot write the metadata.
Since OME-Zarr is currently the only way to write metadata supported, its requirements apply to all Multiscales.

* There must be 2 to 5 axes.
* 2 or 3 of the axes must have `Axis.type="space"`.
* If any axis has type `"time"`, it must be first, and if any has type `"channel"`, it must come right after `"time"` (or be first).

Likewise, scale keys must be valid relative paths (letters, digits, `.`, `_`, `-`, separated by `/`), and scales must be ordered from largest to smallest.

Axis types are part of the `Scale`.
If you are using the standard axis keys (`t`, `c`, `z`, `y`, `x`), they are automatically inferred.
For any other axis keys, or to set other axis properties, pass the `ome_zarr_axes` explicitly:

```python
import clearscale as cs

scale = cs.Scale(
    shape=cs.Shape(w=3, u=100, v=100),
    ome_zarr_axes={
        "w": cs.ome_zarr.Axis(type="channel"),
        "u": cs.ome_zarr.Axis(type="space"),
        "v": cs.ome_zarr.Axis(type="space"),
    },
)
```

> [!WARNING]
> The upcoming version of OME-Zarr will remove the requirement for axis types to be specified.
> If you write logic that relies on axis types, it will be brittle when this version becomes official.
> In other words, for now you must specify types (or use standard axis keys), but you should not expect types to exist in the future.
> See https://ngff.openmicroscopy.org/specifications/dev/index.html#multiscales-metadata
