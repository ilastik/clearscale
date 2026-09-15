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
    """None iff produced by `characterize_shape_scaling_method`: shape-parametrized calls
    have no Factor->Shape rounding step to characterize (the caller already supplies the
    exact target shape). Always set for the two factor-parametrized variants."""
    rounding_error: Optional[float]
    """None iff `rounding` is None. Otherwise the total-error margin by which the returned
    `rounding` beat the closest alternative rounding rule across successful probes (always
    >= 1; a returned `rounding` always matches every successful probe exactly -- an
    imperfect or ambiguous match raises instead of being returned). Larger margins indicate
    a more decisively distinguished winner."""
    warnings: Tuple[str, ...] = ()
    """Non-fatal concerns about the characterization's reliability (noisy/ambiguous fits etc).
    Non-empty means `translating`, `pixel_sizing` and/or `rounding` might be plain wrong."""

    def to_kwargs(self) -> Dict[str, Any]:
        """kwargs to `**`-splat into the matching `apply_to_scale` call.
        `rounding is None` -> `BlueprintShapes.apply_to_scale(**kwargs)`.
        `rounding is not None` -> `BlueprintFactors.apply_to_scale(**kwargs)`."""
        kwargs: Dict[str, Any] = {"translating": self.translating, "pixel_sizing": self.pixel_sizing}
        if self.rounding is not None:
            kwargs["rounding"] = self.rounding
        return kwargs


Probe = Tuple[int, float]
"""Tuple of input vector length and scaling factor.
The scaling method is provided with an input sequence of the given length, 
and asked to scale by the given factor."""
RoundingImplementation = Callable[[int, float], int]
"""Provides output length for a Probe (input length, factor)
that would be expected if the scaling function matches this rounding behavior."""


def characterize_shape_scaling_method(
    scaling_function: Callable[[Sequence[float], int], Iterable[float]],
) -> ScalingMethodCharacterization:
    """
    Characterize a scaling implementation that accepts target shape as its scaling parameter
    (e.g. `skimage.transform.resize(x, output_shape=...)`).
    Feed the result to `BlueprintShapes.apply_to_scale(scale, **characterization.to_kwargs())`.
    """
    source_length = 1025
    out = _as_1d_float_list(scaling_function([float(i) for i in range(source_length)], 257))
    characterization, is_tied = _characterize(out, source_length, exact_factor_spacing=None)
    assert not is_tied, "unreachable: no exact_factor candidate and source != target"
    return characterization


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
    _SHAPE_FACTOR_ROUNDING_RULES: Tuple[Tuple[RoundingMethod, RoundingImplementation], ...] = (
        ("floor", lambda n, s: math.floor(n * s)),
        ("ceil", lambda n, s: math.ceil(n * s)),
        ("round", lambda n, s: round(n * s)),
        ("round_half_up", lambda n, s: math.floor(n * s + 0.5)),
    )
    _SHAPE_FACTOR_ROUNDING_PROBES: Tuple[Probe, ...] = (
        (1003, 0.25),  # Distinguish ceil
        (1003, 0.75),  # vs floor for downscaling
        (1003, 1.25),  # and for upscaling.
        (1003, 1.75),
        (1001, 0.5),  # Distinguish round (500) from round_half_up (501);
        (1001, 1.5),  # confirm true round, which in this case is == round_half_up (both 1502); + cover upscaling
        (1025, 0.37),  # Check special handling of powers of 2 for good measure,
        (1025, 1.37),  # also for upscaling.
        (1002, 0.25),  # And an even input.
    )
    # Two probes used to confirm a method only accepts factors that scale the input exactly (no rounding allowed)
    _SHAPE_FACTOR_EXACT_PROBES: Tuple[Probe, ...] = ((1024, 0.5), (1200, 0.25))
    return _characterize_factor_scaling_method(
        scaling_function,
        discriminating_probes=_SHAPE_FACTOR_ROUNDING_PROBES,
        exact_probes=_SHAPE_FACTOR_EXACT_PROBES,
        length_rule_by_rounding=_SHAPE_FACTOR_ROUNDING_RULES,
        factor_to_spacing=lambda f: 1.0 / f,
    )


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
    _STEP_FACTOR_ROUNDING_RULES: Tuple[Tuple[RoundingMethod, RoundingImplementation], ...] = (
        ("floor", lambda n, s: math.floor(n / s)),
        ("ceil", lambda n, s: math.ceil(n / s)),
        ("round", lambda n, s: round(n / s)),
        ("round_half_up", lambda n, s: math.floor(n / s + 0.5)),
    )
    # Scaling functions that accept "step factors" (2 = downscale by 2) usually don't work with fractions < 1
    _STEP_FACTOR_ROUNDING_PROBES: Tuple[Probe, ...] = (
        (1025, 4),  # Distinguish ceil
        (999, 4),  # vs floor
        (1001, 2),  # vs round / round_half_up.
        (1002, 4),  # Even input
    )
    _STEP_FACTOR_EXACT_PROBES: Tuple[Probe, ...] = ((1000, 4), (1200, 3))
    return _characterize_factor_scaling_method(
        scaling_function,
        discriminating_probes=_STEP_FACTOR_ROUNDING_PROBES,
        exact_probes=_STEP_FACTOR_EXACT_PROBES,
        length_rule_by_rounding=_STEP_FACTOR_ROUNDING_RULES,
        factor_to_spacing=lambda f: float(f),
    )


def _characterize_factor_scaling_method(
    scaling_function: Callable[[Sequence[float], float], Iterable[float]],
    *,
    discriminating_probes: Sequence[Probe],
    exact_probes: Sequence[Probe],
    length_rule_by_rounding: Sequence[Tuple[RoundingMethod, RoundingImplementation]],
    factor_to_spacing: Callable[[float], float],
) -> ScalingMethodCharacterization:
    """
    Shared implementation for the two factor-parametrized characterize_* functions.
    `spacing_for_factor` resolves a probe's factor value to its predicted "exact_factor"
    pixel spacing under the caller's convention (1/f for multiplier, f for divisor).

    Reuses each successfully-probed (n, f) call's actual output for both rounding
    detection and pixel-sizing/shift fitting -- no probe is ever discarded after use,
    and no separate "fallback factor" mechanism is needed: shape_ratio and exact_factor
    are numerically identical under exact division, so any fallback restricted to
    exact-dividing pairs would be structurally unable to distinguish them.
    """
    successes: List[Tuple[Probe, List[float], bool]] = []
    exact_set = set(exact_probes)
    for source_length, probe_value in (*discriminating_probes, *exact_probes):
        coords = [float(i) for i in range(source_length)]
        try:
            out = _as_1d_float_list(scaling_function(coords, probe_value))
        except Exception:
            continue
        successes.append(((source_length, probe_value), out, (source_length, probe_value) in exact_set))

    discriminating_successes = [s for s in successes if not s[2]]

    # --- rounding ---
    if len(discriminating_successes) >= 2:
        total_error = {name: 0.0 for name, _ in length_rule_by_rounding}
        for (source_length, probe_value), out, _ in discriminating_successes:
            for name, rule in length_rule_by_rounding:
                total_error[name] += abs(len(out) - rule(source_length, probe_value))
        ranked = sorted(total_error.items(), key=lambda item: item[1])
        if ranked[0][1] != 0.0:
            raise ValueError(
                f"Scaling function's output length does not exactly match any known rounding rule "
                f"(closest: {ranked[0][0]!r}, total mismatch {ranked[0][1]:.3g} across "
                f"{len(discriminating_successes)} probes). Rounding may be input-size-dependent "
                "(e.g. a pooling method that discards a partial final window), or non-standard."
            )
        if ranked[1][1] == 0.0:
            raise ValueError("Rounding characterization is ambiguous: multiple rounding rules match equally well.")
        rounding, rounding_error = ranked[0][0], ranked[1][1]
    elif _accepts_exact_scaling_only(successes, exact_probes, next(iter(length_rule_by_rounding))[1]):
        # Not necessarily "rejects non-exact input" -- some methods (e.g. padding-style
        # block-reduce) silently accept non-exact input instead of raising, in which case
        # this branch is never reached and they're characterized normally below.
        rounding: RoundingMethod
        rounding, rounding_error = "error_on_round", math.inf
    elif len(discriminating_successes) == 0:
        raise ValueError("Scaling function rejected every rounding probe.")
    else:
        raise ValueError("Scaling function accepted only one rounding probe; rounding cannot be characterized.")

    # --- pixel-sizing + shift: walk probes (discriminating first) until one is unambiguous ---
    accumulated_warnings: List[str] = []
    for (source_length, probe_value), out, is_exact in (*discriminating_successes, *(s for s in successes if s[2])):
        try:
            characterization, is_tied = _characterize(
                out, source_length, exact_factor_spacing=factor_to_spacing(probe_value)
            )
        except ValueError:
            continue

        if not is_tied:
            return replace(
                characterization,
                rounding=rounding,
                rounding_error=rounding_error,
                warnings=(*accumulated_warnings, *characterization.warnings),
            )

        if rounding == "error_on_round" and characterization.pixel_sizing in ("shape_ratio", "exact_factor"):
            # Provably identical for a method whose only valid calls are exact-dividing
            # (see docstring proof in ScalingMethodCharacterization) -- not a real ambiguity.
            return replace(
                characterization,
                pixel_sizing="exact_factor",
                rounding=rounding,
                rounding_error=rounding_error,
                warnings=(
                    *accumulated_warnings,
                    *characterization.warnings,
                    "shape_ratio and exact_factor are provably identical for this exact-division-only "
                    "method; exact_factor reported by convention, not by discriminating measurement.",
                ),
            )

        accumulated_warnings.append(
            f"Probe (n={source_length}, factor={probe_value}) could not distinguish pixel-sizing "
            "candidates; tried next probe."
        )

    raise ValueError("No probe produced an unambiguous pixel-sizing/shift characterization.")


def _accepts_exact_scaling_only(
    successes: Sequence[Tuple[Probe, List[float], bool]],
    exact_probes: Sequence[Probe],
    exact_length_rule: RoundingImplementation,
) -> bool:
    """
    Final check with two probes that require no rounding.
    When all other rounding probes, this should confirm whether the method refuses to do
    any rounding at all, or if it's plain broken and always errors.
    """
    exact_results = {(n, f): out for (n, f), out, is_exact in successes if is_exact}
    if len(exact_results) < len(exact_probes):
        return False
    return all(len(exact_results[(n, f)]) == exact_length_rule(n, f) for n, f in exact_probes)


def _characterize(
    out: Sequence[float],
    source_length: int,
    *,
    exact_factor_spacing: Optional[float],
) -> Tuple[ScalingMethodCharacterization, bool]:
    """
    Characterize the output produced after running a particular probe.
    `exact_factor_spacing` is the expected pixel spacing multiplier given the scaling factor
    the method was parametrized with, if any
    (i.e. expected if the method multiplies input spacing exactly by the factor).
    """
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
    tie_threshold = 1e-12
    """If two pixel-size rules predict the same pixel size within tie_threshold, they are considered equal."""

    errors = sorted(
        ((name, abs(predicted - spacing)) for name, predicted in pixel_size_candidates), key=lambda item: item[1]
    )
    pixel_sizing, winner_error = errors[0]
    pixel_sizing_error = max(winner_error, affine_error)
    is_tied = len(errors) > 1 and errors[1][1] < tie_threshold

    base = Scale(shape=Shape(x=source_length), pixel_size=PixelSize(x=1.0))
    target = Scale(shape=Shape(x=target_length), pixel_size=PixelSize(x=spacing))
    translation_scores: List[Tuple[TranslationShiftFunction, float]] = []
    for rule in known_shift_functions:
        predicted = rule(base, target)["x"]
        rms = math.sqrt(sum((y - (spacing * x + predicted)) ** 2 for x, y in zip(xs, ys)) / n)
        translation_scores.append((rule, rms))
    translating, translating_error = min(translation_scores, key=lambda item: item[1])

    warning_msgs: List[str] = []
    if pixel_sizing_error > relative_error_warning_threshold * abs(spacing):
        warning_msgs.append(
            f"Ambiguous or noisy pixel-size characterization (matched {pixel_sizing!r}, "
            f"error {pixel_sizing_error:.3g}, spacing {spacing:.3g})."
        )
    if translating_error > relative_error_warning_threshold * abs(spacing):
        warning_msgs.append(
            f"Ambiguous or noisy translation characterization (matched {translating.__name__!r}, "
            f"error {translating_error:.3g}, spacing {spacing:.3g})."
        )

    characterization = ScalingMethodCharacterization(
        translating=translating,
        translating_error=translating_error,
        pixel_sizing=pixel_sizing,
        pixel_sizing_error=pixel_sizing_error,
        rounding=None,
        rounding_error=None,
        warnings=tuple(warning_msgs),
    )
    return characterization, is_tied


def _as_1d_float_list(values: Iterable[float]) -> list[float]:
    if getattr(values, "ndim", 1) != 1:
        raise ValueError(f"Scaling function must return a one-dimensional array. Received: {values!r}")

    try:
        return [float(value) for value in values]
    except (TypeError, ValueError) as e:
        raise ValueError(f"Scaling function must return a one-dimensional array. Received: {values!r}") from e
