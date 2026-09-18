# Scaling methods

There are many ways to scale images, and there is no simple way to predict the results of any scaling method without knowing some specifics about its implementation.

There are three aspects of the scaling method you are using which `clearscale` needs to know to calculate the resulting image shapes, pixel size and translation accurately:

* rounding behavior
* correspondence between effective scaling factor and pixel size change
* shift of the first value's position in space relative to the raw image's origin

The `clearscale` methods that interact with these properties accept parameters called `rounding`, `pixel_sizing`, and `translating`.

This documentation section covers some details of what these properties mean, and guidance for choosing the right ones.

## How to characterize a scaling method

To determine which parameters are the correct ones for your particular scaling method, you can use the functions from the `characterization` module.
Which of these functions you should use depends on whether your scaling methods accepts:

* a target *shape* (example: `skimage.transform.resize`)
* a shape factor that *multiplies* the input shape (i.e. `factor=0.5` means "downscale to half the shape", example: `scipy.ndimage.zoom`)
* a step factor that *divides* the input shape (i.e. `factor=2` means "downscale by 2", example: `skimage.transform.downscale_local_mean`)

The functions are named accordingly (`from clearscale.characterization import ...`):
* `characterize_shape_scaling_method` 
* `characterize_shape_factor_scaling_method`
* `characterize_step_factor_scaling_method`

The characterization will then give you the correct `rounding`, `pixel_sizing` and `translating` parameters that you need to plug into methods like `Multiscale.from_single`.

To use them, wrap the scaling method you actually use into a function that takes a 1D sequence of floats, and the respective scaling target parameter, and returns the scaled sequence.

If your scaling method requires a NumPy array, convert the received sequence inside your wrapper (as in the examples below).

```python
from clearscale.characterization import (
    characterize_shape_scaling_method,
    characterize_shape_factor_scaling_method,
    characterize_step_factor_scaling_method,
    half_pixel_space_preservation,
    discrete_bin_center,
    first_value_decimation,
)
import numpy as np

from skimage.transform import resize

characterization = characterize_shape_scaling_method(
    lambda x, shape: resize(
        np.asarray(x),
        (shape,),
        order=1,              # Put the exact parameters your actual scaling code uses
        preserve_range=True,
        anti_aliasing=True,
    )
)

# Use characterization.to_kwargs() when constructing your multiscale.
assert characterization.rounding is None, "shape-based scaling methods don't round"
assert characterization.pixel_sizing == "shape_ratio"
assert characterization.pixel_sizing_error < 1e-13
assert characterization.translating is half_pixel_space_preservation
assert characterization.translating_error < 1e-13


from scipy.ndimage import zoom

characterization = characterize_shape_factor_scaling_method(
    lambda x, factor: zoom(
        np.asarray(x),
        factor,
        order=1,
        grid_mode=False,
    )
)

assert characterization.rounding == "round"
assert characterization.pixel_sizing == "corner_ratio"
assert characterization.pixel_sizing_error < 1e-13
assert characterization.translating is first_value_decimation
assert characterization.translating_error < 1e-13


from skimage.transform import downscale_local_mean

characterization = characterize_step_factor_scaling_method(
    lambda x, factor: downscale_local_mean(
        np.asarray(x),
        (factor,),
    )
)

assert characterization.rounding == "ceil"
assert characterization.pixel_sizing == "exact_factor"
assert characterization.pixel_sizing_error < 1e-13
assert characterization.translating is discrete_bin_center
assert characterization.translating_error < 1e-13
```

## Rounding

Factor-based scaling methods have a problem:

```python
import skimage, numpy

arr = numpy.array([0, 1, 2, 3, 4])

print(len(arr))  # 5

# Scale to half: 5 * 0.5 = 2 or 3?
rescaled_arr = skimage.transform.rescale(arr, 0.5)

print(len(rescaled_arr))  # 2

# Downscale by 2: 5 / 2 = 2 or 3?
local_mean_arr = skimage.transform.downscale_local_mean(arr, 2)

print(len(local_mean_arr))  # 3
```

The `rounding` parameter that various `clearscale` functions ask for should accurately reflect how your scaling method rounds output shapes when `input_shape * shape_factor` (or `input_shape / step_factor`) does not produce an exact integer number.

You can pass `rounding=` either:

* one of the keywords for common rounding methods: `"floor", "ceil", "round"` or `"round_half_up"` (`round_half_up` means `.5` rounds to the next *larger* integer; `round` means to the next *even* integer, i.e. Python's own `round()`)
* a function (`Callable[[float], int]`) that accepts the theoretical fractional output array length along one axis (`float`) and rounds it as your own scaling method would, returning the rounded length (`int`)

## Pixel size

Different scaling methods implement various forms of how exactly they modify the spacing of pixel values in the output.
The `pixel_sizing` parameter in `clearscale` reflects this characteristic.

You can pass `pixel_sizing=` either:

* `"shape_ratio"` if the output spacing is `input_length / output_length`
* `"corner_ratio"` if the output spacing is `(input_length - 1) / (output_length - 1)` (the first and last pixel, i.e. the image corners, are preserved, and the rest are evenly spaced between them)

For factor-based scaling methods, there is one additional option:

* `"exact_factor"` if the output spacing is `input_spacing / shape_factor`, respectively `input_spacing * step_factor`

## Translation shift

`Scale.translation` is the physical coordinate of the first pixel center.

Pass a `translating` function to `blueprint.apply_to_scale()` when your scaling operation changes where the first scaled pixel belongs in physical space:

```python
multiscale = blueprint.apply_to_scale(base, translating=half_pixel_space_preservation)
```

Most common scaling methods apply one of the provided shift functions explained below (`from clearscale.characterization import ...`):

* `half_pixel_space_preservation`
* `discrete_bin_center`
* `first_value_decimation`

If your scaling method applies a different shift, you need to implement your own function as a `Callable[[Scale, Scale], Translation]`.
The `translating` function receives both the base `Scale` including its existing `.translation`, and the target `Scale` with a zero-translation.
It should use the two Scale's properties (primarily `.shape` and `.pixel_size`) to determine and return the predicted `Scale`'s `.translation`.

### `half_pixel_space_preservation`: extent-preserving resize

Use `half_pixel_space_preservation` for interpolation or resampling methods that preserve the full image extent under a pixel-center convention.
This is the convention where the data samples are considered to be positioned at the center of a box ranging from `-0.5 * pixel_size` to `0.5 * pixel_size`.

Common examples include output-shape based interpolation such as `skimage.transform.resize(..., anti_aliasing=True)`, or in deep learning contexts, `nn.Upsample(..., align_corners=False)`.

```python
from clearscale import (
    BlueprintShapes,
    PixelSize,
    Scale,
    Shape,
    Translation,
    Unit,
    half_pixel_space_preservation,
)

base = Scale(
    shape=Shape(y=1024, x=1536),
    pixel_size=PixelSize(y=0.3, x=0.3),
    unit=Unit(y="micrometer", x="micrometer"),
    translation=Translation(y=0.0, x=0.0),
)

blueprint = BlueprintShapes.uniform_steps(
    step=2,
    base_shape=base.shape,
    rounding="round",
    limit_all=Shape(y=256, x=384),
)

# Data scaling, for example with scikit-image:
# from skimage.transform import resize
#
# scaled_arrays = {}
# for scale_key, target_shape in blueprint.items():
#     scaled_arrays[scale_key] = resize(
#         raw_yx,
#         output_shape=target_shape.to_tuple(),
#         anti_aliasing=True,
#         preserve_range=True,
#     )

multiscale = blueprint.apply_to_scale(
    base,
    translating=half_pixel_space_preservation,
)

assert multiscale["s1"].shape == Shape(y=512, x=768)
assert multiscale["s1"].pixel_size == PixelSize(y=0.6, x=0.6)
assert multiscale["s1"].translation == Translation(y=0.15, x=0.15)
```

The first output pixel center comes out `0.15 micrometer` translated.

Under the pixel-center convention, the first *raw* pixel at coordinate `0.0`, with pixel size `0.3 micrometer`, represents the space from `-0.15 micrometer` to `0.15 micrometer` (half a pixel before and after its center coordinate).
Therefore, the data space represented by the image begins at `-0.15`.

For `s1`, the output pixel size is `0.6 micrometer`.
The first `s1` pixel's coordinate (i.e. its center) is therefore exactly half of that (`0.3 micrometer`) shifted from the beginning of the space it represents.
If the scaling method preserves the full data space, then the space still begins at `-0.15`.
The first `s1` pixel is therefore at `-0.15 + 0.3 = 0.15`.

### `discrete_bin_center`: bin-averaging

Use `discrete_bin_center` for block or bin downsampling where each output value represents the center of the source pixels that contributed to that bin.
This cleanly fits a local mean pooling method, but it is also the closest simple approximation for non-linear pooling methods like max, min or median.

For exact integer scaling factors, the numeric shift can be the same as `half_pixel_space_preservation`.
The distinction still matters: choosing the appropriate function documents the pooling convention.

```python
from clearscale import (
    BlueprintFactors,
    Factor,
    PixelSize,
    Scale,
    Shape,
    Translation,
    Unit,
    discrete_bin_center,
)

base = Scale(
    shape=Shape(y=2048, x=2048),
    pixel_size=PixelSize(y=0.25, x=0.25),
    unit=Unit(y="micrometer", x="micrometer"),
    translation=Translation(y=0.0, x=0.0),
)

block_factors = BlueprintFactors(
    {
        "s0": Factor(y=1, x=1),
        "s1": Factor(y=2, x=2),
        "s2": Factor(y=4, x=4),
    }
)

# Data scaling, for example with scikit-image:
# import numpy as np
# from skimage.measure import block_reduce
#
# binned_arrays = {}
# for scale_key, factor in block_factors.items():
#     block_size = tuple(int(factor[axis]) for axis in base.shape)
#     binned_arrays[scale_key] = block_reduce(
#         raw_yx,
#         block_size=block_size,
#         func=np.mean,
#     )

multiscale = block_factors.apply_to_scale(
    base,
    rounding="ceil",
    translating=discrete_bin_center,
)

assert multiscale["s2"].shape == Shape(y=512, x=512)
assert multiscale["s2"].pixel_size == PixelSize(y=1.0, x=1.0)
assert multiscale["s2"].translation == Translation(y=0.375, x=0.375)
```

For `s2`, each output pixel summarizes a `4 x 4` block.
The first output pixel is pooled from input pixel centers at `0.0`, `0.25`, `0.5`, and `0.75 micrometer` along each axis.
These represent the space from `-0.125` to `0.875 micrometer` placing the pooled pixel's center at `0.375 micrometer`.

### `first_value_decimation`: stride from the first sample

Use `first_value_decimation` when the scaled array keeps the first source value exactly, and evenly spaces the rest.
For example, `data[::4, ::4]`.

`translating=first_value_decimation` is equivalent to `translating=None`, but you can still pass it to make the choice explicit in your code.

```python
from clearscale import (
    BlueprintFactors,
    Factor,
    PixelSize,
    Scale,
    Shape,
    Translation,
    Unit,
    first_value_decimation,
)

base = Scale(
    shape=Shape(y=2048, x=2048),
    pixel_size=PixelSize(y=0.25, x=0.25),
    unit=Unit(y="micrometer", x="micrometer"),
    translation=Translation(y=12.0, x=-3.0),
)

stride_factors = BlueprintFactors(
    {
        "s0": Factor(y=1, x=1),
        "s1": Factor(y=2, x=2),
        "s2": Factor(y=4, x=4),
    }
)

# Data scaling with first-value stride slicing:
# decimated_arrays = {}
# for scale_key, factor in stride_factors.items():
#     stride_y = int(factor["y"])
#     stride_x = int(factor["x"])
#     decimated_arrays[scale_key] = labels_yx[::stride_y, ::stride_x]

multiscale = stride_factors.apply_to_scale(
    base,
    rounding="ceil",
    translating=first_value_decimation,
)

assert multiscale["s2"].shape == Shape(y=512, x=512)
assert multiscale["s2"].pixel_size == PixelSize(y=1.0, x=1.0)
assert multiscale["s2"].translation == Translation(y=12.0, x=-3.0)
```

The first output value is the first input value, so the translation does not move.
Only the pixel size changes to describe the larger spacing between retained samples.
