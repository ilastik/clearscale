import math
from dataclasses import dataclass, replace
from typing import (
    Callable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Mapping,
    NamedTuple,
    TypedDict,
    Union,
    Literal,
    cast,
)

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


class ScalingMethodKwargs(TypedDict):
    """kwargs for `Blueprint*.apply_to_scale`: everything needed to ensure shape, pixel size and
    translation are computed accurately for output Scales."""

    # Unpack unavailable in py3.10 -- make sure these kwargs stay synchronised with the method signatures
    pixel_sizing: PixelSizingMethod
    translating: TranslationShiftFunction
    rounding: Optional[RoundingMethod]


@dataclass(frozen=True, slots=True)
class ScalingMethodCharacterization:
    """Characteristics of a scaling method, determined by one of the `characterize_*` functions."""

    translating: TranslationShiftFunction
    translating_error: float
    pixel_sizing: PixelSizingMethod
    pixel_sizing_error: float
    rounding: Union[RoundingMethod, Literal["indeterminate"], None]
    """One of:
    - Rounding method that correctly predicted output shape for *all* probes.
    - None for `characterize_shape_scaling_method` (shape-parametrized calls = no shape-rounding).
    - "indeterminate" for methods that scale by factor, but showed inconsistent shape rounding behavior."""
    rounding_error: Optional[float]
    """Number of probes for which the runner-up rounding method (not `rounding` itself) was wrong."""
    warnings: Tuple[str, ...] = ()
    """Non-fatal concerns about the characterization's reliability (noisy/ambiguous fits etc).
    Non-empty means `translating`, `pixel_sizing` and/or `rounding` might be plain wrong."""

    def to_kwargs(self) -> ScalingMethodKwargs:
        """kwargs to `**`-splat into the matching `apply_to_scale` call.
        `rounding is None` -> `BlueprintShapes.apply_to_scale(scale, **kwargs)`.
        `rounding is not None` -> `BlueprintFactors.apply_to_scale(scale, **kwargs)`."""
        if self.rounding == "indeterminate":
            raise ValueError(
                "The characterization could not determine rounding behavior of the supplied scaling method. "
                "This means `BlueprintFactors.apply_to_scale` will predict incorrect output shapes. "
                "You should instead first run the actual scaling, and record the shapes it produced along the way. "
                "Construct `BlueprintShapes(zip(scale_keys, recorded_shapes))`, and use "
                "`blueprint.apply_to_scale(scale, **characterization.to_shapes_kwargs())`."
            )
        rounding = cast(Optional[RoundingMethod], self.rounding)
        return ScalingMethodKwargs(
            pixel_sizing=self.pixel_sizing,
            translating=self.translating,
            rounding=rounding,
        )

    def to_shapes_kwargs(self) -> ScalingMethodKwargs:
        if self.rounding != "indeterminate":
            raise AssertionError(
                "This is convenience for scaling methods whose rounding behavior cannot be determined."
            )
        return ScalingMethodKwargs(
            pixel_sizing=self.pixel_sizing,
            translating=self.translating,
            rounding=None,
        )


class Probe(NamedTuple):
    source_length: int
    factor: float


class FailedUnevenScalingError(ValueError):
    pass


class IndeterminateRoundingError(ValueError):
    pass


RoundingRule = Callable[[Probe], int]
"""Predicts the output length a specific rounding convention would produce for `probe`."""
RoundingRuleTable = Mapping[RoundingMethod, RoundingRule]


def characterize_shape_scaling_method(
    scaling_function: Callable[[Sequence[float], int], Iterable[float]],
) -> ScalingMethodCharacterization:
    """
    Characterize a scaling implementation that accepts target shape as its scaling parameter
    (e.g. `skimage.transform.resize(x, output_shape=...)`).
    Feed the result to `BlueprintShapes.apply_to_scale(scale, **characterization.to_kwargs())`.
    """
    source_length = 1025
    # 263 is prime to avoid special behaviours, and leads to distinguishable spacings and translations under all known conventions:
    # pixel_sizing:
    #    shape_ratio: 1025 / 263 = 3.89733...
    #    corner_ratio: (1025-1) / (263-1) = 3.90839...
    # translating:
    #    half_pixel_space_preservation: 0.5 * (<3.89733 or 3.90839> - 1) = <1.44866 or 1.45419>
    #    discrete_bin_center: 0.5 * (ceil(<3.89733 or 3.90839>) - 1) = 1.5
    #    first_value_decimation: 0.0
    out = _as_1d_float_list(scaling_function([float(i) for i in range(source_length)], 263))
    characterization, is_tied = _characterize_affine(out, source_length, exact_factor_spacing=None)
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
    rounding_rules: RoundingRuleTable = {
        "floor": lambda p: math.floor(p.source_length * p.factor),
        "ceil": lambda p: math.ceil(p.source_length * p.factor),
        "round": lambda p: round(p.source_length * p.factor),
        "round_half_up": lambda p: math.floor(p.source_length * p.factor + 0.5),
    }
    # Order of probes doesn't matter for rounding detection, but affine characterization uses the first successful one
    discriminating_probes = (
        Probe(1025, 0.37),  # Preferably use not-a-power-of-2
        Probe(1025, 1.37),  # and check upscaling.
        Probe(1003, 0.25),  # Distinguish ceil
        Probe(1003, 0.75),  # vs floor for downscaling
        Probe(1003, 1.25),  # and for upscaling.
        Probe(1003, 1.75),
        Probe(1001, 0.5),  # Distinguish round (500) from round_half_up (501);
        Probe(1001, 1.5),  # confirm true round, which here == round_half_up (both 1502); covers upscaling too.
        Probe(1002, 0.25),  # And an even input length.
    )
    # Two probes used to confirm a method at least accepts factors that scale the input exactly (no rounding allowed)
    exact_probes = (Probe(1024, 0.5), Probe(1200, 0.25))

    return _characterize_factor_scaling_method(
        scaling_function,
        discriminating_probes=discriminating_probes,
        exact_probes=exact_probes,
        rounding_to_implementation=rounding_rules,
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
    rounding_rules: RoundingRuleTable = {
        "floor": lambda p: math.floor(p.source_length / p.factor),
        "ceil": lambda p: math.ceil(p.source_length / p.factor),
        "round": lambda p: round(p.source_length / p.factor),
        "round_half_up": lambda p: math.floor(p.source_length / p.factor + 0.5),
    }
    # Order of probes doesn't matter for rounding detection, but affine characterization uses the first successful one.
    # Scaling functions that accept "step factors" (2 = downscale by 2) usually don't work with fractions
    discriminating_probes = (
        Probe(1025, 3.7),  # Worth a try anyway,
        Probe(1025, 0.67),  # also upscaling.
        Probe(1025, 4),  # Distinguish ceil
        Probe(999, 4),  # vs floor
        Probe(1001, 2),  # vs round / round_half_up.
        Probe(1002, 4),  # Even input
    )
    exact_probes = (Probe(1000, 4), Probe(1200, 3))

    return _characterize_factor_scaling_method(
        scaling_function,
        discriminating_probes=discriminating_probes,
        exact_probes=exact_probes,
        rounding_to_implementation=rounding_rules,
        factor_to_spacing=lambda f: float(f),
    )


def _characterize_factor_scaling_method(
    scaling_function: Callable[[Sequence[float], float], Iterable[float]],
    *,
    discriminating_probes: Sequence[Probe],
    exact_probes: Sequence[Probe],
    rounding_to_implementation: RoundingRuleTable,
    factor_to_spacing: Callable[[float], float],
) -> ScalingMethodCharacterization:
    """
    Shared implementation for the two factor-parametrized characterize_* functions.

    `factor_to_spacing` resolves a probe's factor value to its predicted "exact_factor"
    pixel spacing under the caller's convention (1/f for multiplier, f for divisor).
    """
    discriminating_results = dict(zip(discriminating_probes, _run_probes(scaling_function, discriminating_probes)))
    exact_results = dict(zip(exact_probes, _run_probes(scaling_function, exact_probes)))

    rounding: Union[RoundingMethod, Literal["indeterminate"]]
    try:
        rounding, rounding_error = _detect_rounding(discriminating_results, rounding_to_implementation)
    except FailedUnevenScalingError:
        some_rounding = next(iter(rounding_to_implementation.values()))
        if all(output is not None and len(output) == some_rounding(probe) for probe, output in exact_results.items()):
            # All the regular probes failed, but at least the scaling function accepted probes that need no rounding
            rounding = "error_on_round"
            rounding_error = math.inf
        else:
            raise ValueError(
                "Scaling method failed to execute more than one attempted parameter combination. Attempts with rounding: "
                f"{discriminating_results!r}. Attempts without rounding: {exact_results!r}"
            )
    except IndeterminateRoundingError:
        rounding = "indeterminate"
        rounding_error = math.inf

    # Try discriminating first, then exact as fallback
    all_successes = [(p, o) for p, o in (*discriminating_results.items(), *exact_results.items()) if o is not None]
    # exact_results can't distinguish pixel_sizing "shape_ratio" from "exact_factor" (without shape rounding, both produce the same number)
    pixel_sizing_tie_expected = rounding == "error_on_round"
    characterization = _get_first_unambiguous_affine_characterization(
        all_successes, factor_to_spacing, pixel_sizing_tie_expected
    )

    if (
        all(isinstance(probe.factor, int) or probe.factor.is_integer() for probe, _o in all_successes)
        and characterization.translating is half_pixel_space_preservation
    ):
        # No probe with a fractional factor succeeded; the method only accepts integer scaling factors
        assert (
            characterization.translating_error == 0.0
        ), "translating should be exact for scaling methods that only accept int step factors"
        # half_pixel_space_preservation and discrete_bin_center are numerically *identical* in this case: 0.5 * (spacing - 1) == 0.5 * (ceil(spacing) - 1)
        # But the more descriptive name is discrete_bin_center, because that's what the method is more likely to actually be doing if it only accepts int.
        characterization = replace(characterization, translating=discrete_bin_center)

    return replace(
        characterization,
        rounding=rounding,
        rounding_error=float(rounding_error),
    )


def _run_probes(
    scaling_function: Callable[[Sequence[float], float], Iterable[float]], probes: Sequence[Probe]
) -> List[Optional[List[float]]]:
    results: List[Optional[List[float]]] = []
    for probe in probes:
        coords = [float(i) for i in range(probe.source_length)]
        try:
            scaled = scaling_function(coords, probe.factor)
        except Exception:
            scaled = None
        results.append(None if scaled is None else _as_1d_float_list(scaled))
    return results


def _detect_rounding(
    discriminating_results: Mapping[Probe, Optional[List[float]]],
    rounding_to_implementation: RoundingRuleTable,
) -> Tuple[RoundingMethod, int]:
    successful_discriminating_results = {p: o for p, o in discriminating_results.items() if o is not None}
    if len(successful_discriminating_results) == 0:
        raise FailedUnevenScalingError(
            "Scaling function failed to execute all attempts that require rounding output shape."
        )
    if len(successful_discriminating_results) == 1:
        raise FailedUnevenScalingError(
            "Scaling function executed only one attempt with shape rounding successfully; not enough information to determine rounding behavior."
        )

    total_error = {name: 0 for name in rounding_to_implementation}
    for probe, output in successful_discriminating_results.items():
        for name, predict_length in rounding_to_implementation.items():
            total_error[name] += abs(len(output) - predict_length(probe))

    ranked = sorted(total_error.items(), key=lambda item: item[1])
    best_rounding, best_error = ranked[0]
    if best_error != 0:
        # Common e.g. for convolutions, where output shape depends on multiple params (kernel size, stride, ...)
        raise IndeterminateRoundingError(
            "Scaling function's rounding behavior does not exactly match any known rounding rule (closest: "
            f"{best_rounding!r}, but {best_error}/{len(discriminating_results)} probes did not match this rule)."
        )
    _, runner_up_error = ranked[1]
    if runner_up_error == 0:
        raise IndeterminateRoundingError(
            "Rounding characterization is ambiguous: multiple rounding rules match equally well."
        )
    assert best_rounding in ("ceil", "floor", "round", "round_half_up")
    return best_rounding, runner_up_error


def _get_first_unambiguous_affine_characterization(
    successes: Sequence[Tuple[Probe, List[float]]],
    factor_to_spacing: Callable[[float], float],
    pixel_sizing_tie_expected: bool,
) -> ScalingMethodCharacterization:
    for probe, output in successes:
        characterization, is_pixel_sizing_tied = _characterize_affine(
            output,
            probe.source_length,
            exact_factor_spacing=factor_to_spacing(probe.factor),
        )

        if not is_pixel_sizing_tied or pixel_sizing_tie_expected:
            return characterization

    raise ValueError("No probe produced an unambiguous pixel-sizing characterization.")


def _characterize_affine(
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
    if spacing <= 0:
        # Sanity check, but also guards against trying to construct PixelSize(x=( <=0 )) further down
        raise ValueError("Scaling function produced zero or negative spacing.")
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
    is_pixel_sizing_tie = len(errors) > 1 and errors[1][1] < tie_threshold

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

    assert pixel_sizing in ("shape_ratio", "corner_ratio", "exact_factor")
    characterization = ScalingMethodCharacterization(
        translating=translating,
        translating_error=translating_error,
        pixel_sizing=pixel_sizing,
        pixel_sizing_error=pixel_sizing_error,
        rounding=None,
        rounding_error=None,
        warnings=tuple(warning_msgs),
    )
    return characterization, is_pixel_sizing_tie


def _as_1d_float_list(values: Iterable[float]) -> list[float]:
    if getattr(values, "ndim", 1) != 1:
        raise ValueError(f"Scaling function must return a one-dimensional array. Received: {values!r}")

    try:
        return [float(value) for value in values]
    except (TypeError, ValueError) as e:
        raise ValueError(f"Scaling function must return a one-dimensional array. Received: {values!r}") from e


__all__ = [
    discrete_bin_center,
    half_pixel_space_preservation,
    first_value_decimation,
    characterize_shape_scaling_method,
    characterize_shape_factor_scaling_method,
    characterize_step_factor_scaling_method,
]
