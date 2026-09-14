import math
import warnings
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from clearscale._axis_values import Translation, PixelSize, RoundingMethod, Shape
from clearscale._multiscale import Scale, TranslationShiftFunction, PixelSizingMethod


def half_pixel_space_preservation(base: "Scale", target: "Scale") -> "Translation":
    """
    The translation shift is the first scaled pixel's coordinate within the original (unscaled) space.

    Space-preserving half-pixel shift is the appropriate shift for downsampling methods that
    preserve the full extent of the data space under the pixel-center (a.k.a. cell-center) convention,
    with uniformly spaced sampling or interpolation of the scaled points.

    1D Example:
    Example intensities = [12, 11, 9, 95, 95]
    Pixel size = 0.6
    Data coordinates = [0.0, 0.6, 1.2, 1.8, 2.4]
    Full data space under pixel-center convention (+- half-pixel): -0.3 to 2.7 (total extent 5 * 0.6 = 3.0)

    When resampling 2 uniformly spaced points:
    New pixel size (preserving the full data space of 3.0): 3.0 / 2 = 1.5
    First scaled pixel data space range: -0.3 to (-0.3 + 1.5) = 1.2
    First scaled pixel coordinate (center of its space range): -0.3 + (1.5 / 2) = 0.45

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
    """
    The translation shift is the first scaled pixel's coordinate within the original (unscaled) space.

    Discrete bin center is the appropriate shift for downsampling methods that pool an integer number of
    raw pixels into a bin and compute their values into a new pixel that represents the center of the bin.
    Most commonly, averaging the bin, or for uneven bin sizes, keeping only the center value.

    1D Example:
    Example intensities = [12, 11, 9, 95, 95]
    Pixel size = 0.6
    Data coordinates = [0.0, 0.6, 1.2, 1.8, 2.4]
    Full data space under pixel-center convention (+- half-pixel): -0.3 to 2.7 (total extent 5 * 0.6 = 3.0)

    When binning with bin size 3:
    New pixel size: 3 * 0.6 = 1.8
    First bin: [12, 11, 9] at coordinates [0.0, 0.6, 1.2] representing space from -0.3 to 1.5
    Averaging the bin means that the new value represents the data in the center, so first pixel coordinate: 0.6

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
    """
    The translation shift is the first scaled pixel's coordinate within the original (unscaled) space.

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


@dataclass(frozen=True, slots=True)
class ScalingMethodCharacterization:
    translating: TranslationShiftFunction
    translating_error: float
    pixel_sizing: PixelSizingMethod
    pixel_sizing_error: float
    rounding: Optional[RoundingMethod]
    """Rounding method that correctly predicted output shape for *all* probes.
    None only for `characterize_shape_scaling_method` (shape-parametrized calls = no shape-rounding)."""
    rounding_error: Optional[float]
    """Number of probes for which the runner-up rounding method (not `rounding` itself) was wrong."""

    def to_kwargs(self) -> Dict[str, Any]:
        """kwargs to `**`-splat into the matching `apply_to_scale` call.
        `rounding is None` -> `BlueprintShapes.apply_to_scale(**kwargs)`.
        `rounding is not None` -> `BlueprintFactors.apply_to_scale(**kwargs)`."""
        kwargs: Dict[str, Any] = {"translating": self.translating, "pixel_sizing": self.pixel_sizing}
        if self.rounding is not None:
            kwargs["rounding"] = self.rounding
        return kwargs


RoundingProbe = Tuple[int, float]
"""Tuple of input vector length and scaling factor, 
for probing the rounding behavior of factor-based scaling methods.
The scaling method is provided with an input sequence of the given length, 
and asked to scale by the given factor."""
RoundingBehavior = Callable[[int, float], int]
"""Provides output length for a RoundingProbe (input length, factor)
expected if the scaling function matches this behavior."""

_SHAPE_FACTOR_ROUNDING_RULES: Tuple[Tuple[RoundingMethod, RoundingBehavior], ...] = (
    ("floor", lambda n, s: math.floor(n * s)),
    ("ceil", lambda n, s: math.ceil(n * s)),
    ("round", lambda n, s: round(n * s)),
    ("round_half_up", lambda n, s: math.floor(n * s + 0.5)),
)
_SHAPE_FACTOR_EXACT_LENGTH_RULE: RoundingBehavior = lambda n, s: int(n * s)
_STEP_FACTOR_ROUNDING_RULES: Tuple[Tuple[RoundingMethod, RoundingBehavior], ...] = (
    ("floor", lambda n, s: math.floor(n / s)),
    ("ceil", lambda n, s: math.ceil(n / s)),
    ("round", lambda n, s: round(n / s)),
    ("round_half_up", lambda n, s: math.floor(n / s + 0.5)),
)
_STEP_FACTOR_EXACT_LENGTH_RULE: RoundingBehavior = lambda n, s: int(n / s)

_SHAPE_FACTOR_ROUNDING_PROBES: Tuple[RoundingProbe, ...] = (
    (1003, 0.25),  # Distinguish ceil
    (1003, 0.75),  # vs floor for downscaling
    (1003, 1.25),  # and for upscaling.
    (1003, 1.75),
    (1001, 0.5),  # Distinguish round (500) from round_half_up (501);
    (1001, 1.5),  # confirm true round, which in this case is == round_half_up (both 1502); + cover upscaling
    (1025, 0.37),  # Check special handling of powers of 2 for good measure
    (1025, 1.37),  # also for upscaling
)
# Two probes used to confirm a method only accepts factors that scale the input exactly (no rounding allowed)
_SHAPE_FACTOR_EXACT_PROBES: Tuple[RoundingProbe, ...] = ((1024, 0.5), (1200, 0.25))
# Scaling functions that accept "step factors" (2 = downscale by 2) usually don't work with fractions < 1
_STEP_FACTOR_ROUNDING_PROBES: Tuple[RoundingProbe, ...] = (
    (1025, 4),  # Distinguish ceil
    (999, 4),  # vs floor
    (1001, 2),  # vs round / round_half_up
)
_STEP_FACTOR_EXACT_PROBES: Tuple[RoundingProbe, ...] = ((1000, 4), (1200, 3))


def characterize_shape_scaling_method(
    scaling_function: Callable[[Sequence[float], int], Iterable[float]],
) -> ScalingMethodCharacterization:
    """
    Characterize a scaling implementation that accepts target shape as its scaling parameter
    (e.g. `skimage.transform.resize(x, output_shape=...)`).
    Feed the result to `BlueprintShapes.apply_to_scale(scale, **characterization.to_kwargs())`.
    """
    return _characterize(lambda coords: scaling_function(coords, 257), source_length=1025, exact_factor_spacing=None)


def characterize_shape_factor_scaling_method(
    scaling_function: Callable[[Sequence[float], float], Iterable[float]],
) -> ScalingMethodCharacterization:
    """
    Characterize a scaling implementation that accepts a shape *multiplier*
    (scipy.ndimage.zoom / skimage.transform.rescale convention: factor < 1 for downscaling,
    factor > 1 for upscaling).
    Feed the result to `BlueprintFactors.apply_to_scale(scale, **characterization.to_kwargs())`.

    Note that clearscale.Factor is a shape *divisor*, so when calling such a scaling function
    to execute the blueprint, pass `factor.inverted().to_tuple()`.
    """
    rounding, rounding_error = _probe_rounding(
        scaling_function,
        _SHAPE_FACTOR_ROUNDING_PROBES,
        _SHAPE_FACTOR_ROUNDING_RULES,
        exact_probes=_SHAPE_FACTOR_EXACT_PROBES,
        exact_length_rule=_SHAPE_FACTOR_EXACT_LENGTH_RULE,
    )

    try:
        characterization = _characterize(
            lambda coords: scaling_function(coords, 0.37), source_length=1025, exact_factor_spacing=1.0 / 0.37
        )
    except Exception:
        # Try without rounding
        characterization = _characterize(
            lambda coords: scaling_function(coords, 0.375), source_length=1000, exact_factor_spacing=1.0 / 0.375
        )

    return replace(characterization, rounding=rounding, rounding_error=rounding_error)


def characterize_step_factor_scaling_method(
    scaling_function: Callable[[Sequence[float], float], Iterable[float]],
) -> ScalingMethodCharacterization:
    """
    Characterize a scaling implementation that accepts a shape *divisor*
    (skimage.measure.block_reduce convention, manual striding, and similar:
    factor > 1 for downscaling, factor < 1 for upscaling).
    Feed the result to `BlueprintFactors.apply_to_scale(scale, **characterization.to_kwargs())`.

    Since clearscale.Factor is itself a shape divisor, you can directly pass the blueprint's
    factors into such a scaling function, usually like `factor.to_tuple()`.
    Most such methods only accept integer scaling factors.
    """
    rounding, rounding_error = _probe_rounding(
        scaling_function,
        _STEP_FACTOR_ROUNDING_PROBES,
        _STEP_FACTOR_ROUNDING_RULES,
        exact_probes=_STEP_FACTOR_EXACT_PROBES,
        exact_length_rule=_STEP_FACTOR_EXACT_LENGTH_RULE,
    )

    try:
        characterization = _characterize(
            lambda coords: scaling_function(coords, 4.0), source_length=1025, exact_factor_spacing=4.0
        )
    except Exception:
        # Try without rounding
        characterization = _characterize(
            lambda coords: scaling_function(coords, 4.0), source_length=1000, exact_factor_spacing=4.0
        )

    return replace(characterization, rounding=rounding, rounding_error=rounding_error)


def _accepts_exact_scaling_only(
    scaling_function: Callable[[Sequence[float], float], Iterable[float]],
    exact_probes: Sequence[RoundingProbe],
    exact_length_rule: RoundingBehavior,
) -> bool:
    """
    Final check with two probes that require no rounding.
    When all other rounding probes, this should confirm whether the method refuses to do
    any rounding at all, or if it's plain broken and always errors.
    """
    assert len(exact_probes) >= 2, "hard-code at least 2 exact probes for confidence"
    for source_length, probe_value in exact_probes:
        coords = [float(i) for i in range(source_length)]
        try:
            observed = len(_as_1d_float_list(scaling_function(coords, probe_value)))
        except Exception:
            return False
        if observed != exact_length_rule(source_length, probe_value):
            return False
    return True


def _probe_rounding(
    scaling_function: Callable[[Sequence[float], float], Iterable[float]],
    probes: Sequence[RoundingProbe],
    rounding_rules: Sequence[Tuple[RoundingMethod, RoundingBehavior]],
    *,
    exact_probes: Sequence[RoundingProbe],
    exact_length_rule: RoundingBehavior,
) -> Tuple[RoundingMethod, float]:
    """
    Probe `scaling_function` with each (source_length, probe_value) pair and check its result
    against `rounding_rules`.
    A rule is only considered matching if it *exactly* predicts the output length.
    If a function errors on all probes, it is checked against `exact_probes` to confirm
    it at least works for input combinations that do *not* require rounding.

    Returns (rounding, margin), where `margin` is the total-error gap between the winning
    rule and the closest runner-up (always >= 1 among successful probes).
    """
    total_error: Dict[RoundingMethod, float] = {name: 0.0 for name, _ in rounding_rules}
    successful_probes = 0

    for source_length, probe_value in probes:
        coords = [float(i) for i in range(source_length)]
        try:
            observed = len(_as_1d_float_list(scaling_function(coords, probe_value)))
        except Exception:
            continue
        successful_probes += 1
        for name, rule in rounding_rules:
            total_error[name] += abs(observed - rule(source_length, probe_value))

    if successful_probes < 2:
        if _accepts_exact_scaling_only(scaling_function, exact_probes, exact_length_rule):
            return "error_on_round", math.inf
        if successful_probes == 0:
            raise ValueError("Scaling function rejected every rounding probe (doesn't seem to work at all).")
        raise ValueError(
            "Scaling function accepted only one rounding probe; rounding behaviour cannot be characterized."
        )

    ranked = sorted(total_error.items(), key=lambda item: item[1])
    winner, winner_error = ranked[0]
    _, runner_up_error = ranked[1]

    if winner_error != 0.0:
        raise ValueError(
            f"Scaling function's output length does not exactly match any known rounding rule "
            f"(closest match {winner!r} still mismatched by {winner_error:.3g} total across "
            f"{successful_probes} successful probes). Rounding behaviour might depend on input size, "
            "or the method uses a non-standard convention."
        )
    if runner_up_error == 0.0:
        raise ValueError("Rounding characterization is ambiguous: multiple rounding rules match equally well.")

    return winner, runner_up_error


def _characterize(
    parametrized_scaling_function: Callable[[Sequence[float]], Iterable[float]],
    *,
    source_length: int,
    exact_factor_spacing: Optional[float],
) -> ScalingMethodCharacterization:
    """
    Characterize a fully parametrized scaling implementation.
    `exact_factor_spacing` is the expected pixel spacing multiplier given the scaling factor
    the method has been parametrized with, if any
    (i.e. expected if the method multiplies input spacing exactly by the factor).
    """
    coords = [float(i) for i in range(source_length)]
    out = _as_1d_float_list(parametrized_scaling_function(coords))
    target_length = len(out)
    min_target_length = 150
    """Below this, the middle-half fit window has too few points for a reliable estimate."""

    if target_length == source_length:
        raise ValueError("Scaling function did not change the array length.")
    if target_length < min_target_length:
        raise ValueError(
            f"Scaling function returned {target_length} samples for {source_length}; "
            f"need at least {min_target_length} to fit a reliable estimate."
        )

    # Fit over the middle half: margin in *source*-space is ~source_length/4 regardless of
    # the scaling factor (the factor cancels between target-space window size and per-sample
    # source-space stride), so this stays a safe margin against antialiasing/boundary
    # artifacts across the whole range of factors these probes use.
    first = target_length // 4
    last = target_length - first
    xs = [float(i) for i in range(first, last)]
    ys = out[first:last]

    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    spacing = sxy / sxx
    offset = mean_y - spacing * mean_x
    affine_error = math.sqrt(sum((y - (spacing * x + offset)) ** 2 for x, y in zip(xs, ys)) / n)

    pixel_size_candidates: List[Tuple[PixelSizingMethod, float]] = [
        ("shape_ratio", source_length / target_length),
        ("corner_ratio", (source_length - 1) / (target_length - 1)),
    ]
    if exact_factor_spacing is not None:
        pixel_size_candidates.append(("exact_factor", exact_factor_spacing))

    relative_error_warning_threshold = 0.01
    """If the winning candidate's error exceeds 1% of the measured spacing, warn: this would
    mean boundary artifacts leaked into the fit window (unusually large antialiasing kernel),
    or the scaling method's response is genuinely non-affine (some custom convention)."""

    pixel_sizing, expected_spacing = min(pixel_size_candidates, key=lambda item: abs(item[1] - spacing))
    pixel_sizing_error = max(abs(expected_spacing - spacing), affine_error)
    if pixel_sizing_error > relative_error_warning_threshold * abs(spacing):
        warnings.warn(
            f"Ambiguous or noisy pixel-size characterization (matched {pixel_sizing!r} with "
            f"error {pixel_sizing_error:.3g}, spacing {spacing:.3g}). Check for boundary "
            "artifacts (e.g. an unusually large antialiasing kernel) or a non-affine response."
        )

    base = Scale(shape=Shape(x=source_length), pixel_size=PixelSize(x=1.0))
    target = Scale(shape=Shape(x=target_length), pixel_size=PixelSize(x=expected_spacing))

    translation_scores: List[Tuple[TranslationShiftFunction, float]] = []
    for rule in known_shift_functions:
        predicted = rule(base, target)["x"]
        rms = math.sqrt(sum((y - (spacing * x + predicted)) ** 2 for x, y in zip(xs, ys)) / n)
        translation_scores.append((rule, rms))
    translating, translating_error = min(translation_scores, key=lambda item: item[1])
    if translating_error > relative_error_warning_threshold * abs(spacing):
        warnings.warn(
            f"Ambiguous or noisy translation characterization (best match error {translating_error:.3g}, "
            f"spacing {spacing:.3g}). Check for boundary artifacts or a non-affine response."
        )

    return ScalingMethodCharacterization(
        translating=translating,
        translating_error=translating_error,
        pixel_sizing=pixel_sizing,
        pixel_sizing_error=pixel_sizing_error,
        rounding=None,
        rounding_error=None,
    )


def _as_1d_float_list(values: Iterable[float]) -> list[float]:
    if getattr(values, "ndim", 1) != 1:
        raise ValueError(f"Scaling function must return a one-dimensional array. Received: {values!r}")

    try:
        return [float(value) for value in values]
    except (TypeError, ValueError) as e:
        raise ValueError(f"Scaling function must return a one-dimensional array. Received: {values!r}") from e
