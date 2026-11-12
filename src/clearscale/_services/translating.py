import math
from typing import Dict, Tuple, TYPE_CHECKING

from clearscale._axis_values import Translation
from clearscale.types import TranslatingMethod, TranslationShiftFunction

if TYPE_CHECKING:
    from clearscale._multiscale import Scale


def half_pixel_space_preservation(base: "Scale", target: "Scale") -> "Translation":
    """The translation shift is the first scaled pixel's coordinate within the original (unscaled) space.

    Space-preserving half-pixel shift is the appropriate shift for downsampling methods that
    preserve the full extent of the data space under the pixel-center (a.k.a. cell-center) convention,
    with uniformly spaced sampling or interpolation of the scaled points.

    1D Example:

    - Example intensities = [12, 11, 9, 95, 95]
    - Pixel size = 0.6
    - Data coordinates = [0.0, 0.6, 1.2, 1.8, 2.4]
    - Full data space under pixel-center convention (+- half-pixel): -0.3 to 2.7 (total extent 5 * 0.6 = 3.0)

    When resampling 2 uniformly spaced points:

    - New pixel size (preserving the full data space of 3.0): 3.0 / 2 = 1.5
    - First scaled pixel data space range: -0.3 to (-0.3 + 1.5) = 1.2
    - First scaled pixel coordinate (center of its space range): -0.3 + (1.5 / 2) = 0.45

    The general, simple formula is `(new_pixel_size - base_pixel_size) / 2`.
    In this example: (1.5 - 0.6) / 2 = 0.9 / 2 = 0.45
    """
    if list(base.pixel_size.keys()) != list(target.pixel_size.keys()):
        raise ValueError("Axis mismatch. Cannot compute half-pixel shift between unrelated Scales.")
    shift_items = []
    for axis, target_pixel_size in target.pixel_size.items():
        base_pixel_size = base.pixel_size[axis]
        shift_items.append((axis, 0.5 * (target_pixel_size - base_pixel_size)))
    return Translation(shift_items)


def discrete_bin_center(base: "Scale", target: "Scale") -> "Translation":
    """The translation shift is the first scaled pixel's coordinate within the original (unscaled) space.

    Discrete bin center is the appropriate shift for downsampling methods that pool an integer number of
    raw pixels into a bin and compute their values into a new pixel that represents the center of the bin.
    Most commonly, averaging the bin, or for uneven bin sizes, keeping only the center value.

    1D Example:

    - Example intensities = [12, 11, 9, 95, 95]
    - Pixel size = 0.6
    - Data coordinates = [0.0, 0.6, 1.2, 1.8, 2.4]
    - Full data space under pixel-center convention (+- half-pixel): -0.3 to 2.7 (total extent 5 * 0.6 = 3.0)

    When binning with bin size 3:

    - New pixel size: 3 * 0.6 = 1.8
    - First bin: [12, 11, 9] at coordinates [0.0, 0.6, 1.2] representing space from -0.3 to 1.5
    - Averaging the bin means that the new value represents the data in the center, so first pixel coordinate: 0.6

    The general formula for the first bin center from the data space origin would be:
    `data_space_origin + raw_pixel_size * bin_pixels / 2`
    In this example: -0.3 + 0.6 * 3 / 2 = -0.3 + 0.9 = 0.6
    The origin itself is `-0.5 * raw_pixel_size`, so the formula simplifies to `raw_pixel_size * (bin_pixels - 1) / 2`.
    In this example: 0.6 * (3-1) / 2 = 0.6
    """
    if list(base.pixel_size.keys()) != list(target.pixel_size.keys()):
        raise ValueError("Axis mismatch. Cannot compute bin-center shift between unrelated Scales.")
    shift_items = []
    for axis, target_pixel_size in target.pixel_size.items():
        base_pixel_size = base.pixel_size[axis]
        implicit_bin_size = target_pixel_size / base_pixel_size
        source_pixels_in_first_bin = max(math.ceil(implicit_bin_size), 1)
        shift_items.append((axis, 0.5 * (source_pixels_in_first_bin - 1) * base_pixel_size))
    return Translation(shift_items)


def first_value_decimation(base: "Scale", target: "Scale") -> "Translation":
    """The translation shift is the first scaled pixel's coordinate within the original (unscaled) space.

    First-value decimation is the appropriate shift for downsampling methods that decimate the raw pixels
    by only keeping every n-th value. A simple example is `decimated = 1d_raw_data[::2]`.
    """
    if list(base.pixel_size.keys()) != list(target.pixel_size.keys()):
        # Technically we can return 0 regardless, but this is still a good sanity guard
        raise ValueError("Axis mismatch. Trying to compute first-pixel shift between unrelated Scales.")
    return Translation.identity(base.shape.keys())


known_shift_functions: Tuple[TranslationShiftFunction, ...] = (
    half_pixel_space_preservation,
    discrete_bin_center,
    first_value_decimation,
)


SHIFT_FUNCTIONS_BY_NAME: Dict[str, TranslationShiftFunction] = {f.__name__: f for f in known_shift_functions}
"""The string aliases accepted wherever a `translating` function is expected: the functions' own names."""


def resolve_translating(translating: TranslatingMethod) -> TranslationShiftFunction:
    """Turn a `translating` argument into a callable: names resolve to the built-in function of that name,
    callables pass through unchanged."""
    if callable(translating):
        return translating
    if isinstance(translating, str):
        try:
            return SHIFT_FUNCTIONS_BY_NAME[translating]
        except KeyError:
            raise ValueError(
                f"Unknown translating method {translating!r}. "
                f"Use one of {list(SHIFT_FUNCTIONS_BY_NAME)} or pass your own function."
            ) from None
    hint = " Use translating='first_value_decimation' for no shift." if translating is None else ""
    raise TypeError(
        f"translating must be a function or one of {list(SHIFT_FUNCTIONS_BY_NAME)}, got {type(translating).__name__}."
        + hint
    )
