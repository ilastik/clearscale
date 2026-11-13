import math
from dataclasses import dataclass, field, replace
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
)

from clearscale._axis_values import PixelSize, Shape
from clearscale._multiscale import Scale
from clearscale._services.translating import (
    discrete_bin_center,
    first_value_decimation,
    half_pixel_space_preservation,
    known_shift_functions,
)
from clearscale.types import PixelSizingMethod, TranslationShiftFunction, RoundingMethod


class ShapeScalingKwargs(TypedDict):
    """kwargs for `BlueprintShapes.apply_to_scale`: everything needed to ensure shape, pixel size and
    translation are computed accurately for output Scales."""

    # Unpack unavailable in py3.10 -- make sure these kwargs stay synchronised with the method signatures
    pixel_sizing: PixelSizingMethod
    translating: TranslationShiftFunction


# Splitting doesn't buy us a huge amount right now (could just Optional `rounding` on BlueprintShapes.apply_to_scale and use one kwargs interface),
# but if we want to cover convolutional scaling at some point, more params would end up involved, and we'd need separate Kwargs then.
class FactorScalingKwargs(ShapeScalingKwargs):
    """kwargs for `BlueprintFactors.apply_to_scale`: everything needed to ensure shape, pixel size and
    translation are computed accurately for output Scales."""

    rounding: RoundingMethod


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
    - "indeterminate" for methods that scale by factor, but showed inconsistent shape rounding behavior.
    """
    warnings: Tuple[str, ...] = ()
    """Non-fatal concerns about the characterization's reliability (noisy/ambiguous fits etc).
    Non-empty means `translating`, `pixel_sizing` and/or `rounding` might be plain wrong."""
    probe_results: Tuple["ProbeResult", ...] = field(default=(), compare=False, repr=False)
    """What the scaling function returned (or the error it raised) for every probe the characterization tried.
    For debugging how the method behaves."""

    def to_shape_kwargs(self) -> ShapeScalingKwargs:
        """Return kwargs to `**`-splat into `BlueprintShapes.apply_to_scale(scale, **shape_kwargs)`.

        Raises:
            ValueError: If `pixel_sizing` is "exact_factor". Only `BlueprintFactors` can compute that, because shapes
                do not contain the factor. Use `to_factor_kwargs` instead.
        """
        if self.pixel_sizing == "exact_factor":
            if self.rounding == "indeterminate":
                raise self._exact_factor_with_indeterminate_rounding_error()
            raise ValueError(
                'The characterization found pixel_sizing="exact_factor", which `BlueprintShapes` cannot compute. '
                "Use `BlueprintFactors.apply_to_scale(scale, **characterization.to_factor_kwargs())` instead."
            )
        return ShapeScalingKwargs(pixel_sizing=self.pixel_sizing, translating=self.translating)

    def to_factor_kwargs(self) -> FactorScalingKwargs:
        """Return kwargs to `**`-splat into `BlueprintFactors.apply_to_scale(scale, **factor_kwargs)`.

        Raises:
            ValueError: If the characterization describes a shape-scaling method (use `to_shape_kwargs`), or if it
                could not determine the `rounding` of a factor-scaling method. For that case, the error says what to do.
        """
        if self.rounding == "indeterminate":
            if self.pixel_sizing == "exact_factor":
                raise self._exact_factor_with_indeterminate_rounding_error()
            raise ValueError(
                "The characterization could not determine rounding behavior of the supplied scaling method. "
                "This means `BlueprintFactors.apply_to_scale` will predict incorrect output shapes. "
                "You could instead first run the actual scaling, and record the shapes it produced along the way. "
                "Construct `BlueprintShapes(zip(scale_keys, recorded_shapes))`, and use "
                "`blueprint.apply_to_scale(scale, **characterization.to_shape_kwargs())`."
            )
        if self.rounding is None:
            raise ValueError("This characterization describes a shape-scaling method. Use `.to_shape_kwargs`.")
        return FactorScalingKwargs(
            pixel_sizing=self.pixel_sizing,
            translating=self.translating,
            rounding=self.rounding,
        )

    @staticmethod
    def _exact_factor_with_indeterminate_rounding_error() -> ValueError:
        """Neither blueprint can describe such a method from its characterization alone.
        `BlueprintShapes` cannot compute exact_factor pixel sizes, and `BlueprintFactors` needs the rounding."""
        return ValueError(
            "The characterization could not determine rounding behavior of the supplied scaling method, "
            'and its pixel_sizing is "exact_factor". '
            "Only `BlueprintFactors` can compute pixel sizes in this case, "
            "but without accurate rounding it will predict incorrect output shapes. "
            "See the 'Rounding' section of the characterization docs. "
            "Please open an issue on GitHub if you run into this."
        )


class Probe(NamedTuple):
    source_length: int
    factor: float


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """The outcome of calling the scaling function with one `Probe`: Either its (1D) output, or the error it raised."""

    probe: Probe
    """NamedTuple[source_length: int, factor: float].
    For `characterize_shape_scaling_method`, `probe.factor` is the target length instead."""
    output: Optional[Tuple[float, ...]]
    """The scaled sequence. None if the scaling function raised."""
    error: Optional[str]
    """`ExceptionType: message` if the scaling function raised, else None."""

    def describe(self) -> str:
        if self.output is None:
            return f"{self.probe!r}: failed with {self.error}"
        return f"{self.probe!r}: returned {len(self.output)} samples"


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
    """Characterize a scaling implementation that accepts target shape as its scaling parameter
    (e.g. `skimage.transform.resize(x, output_shape=...)`).
    Feed the result to `BlueprintShapes.apply_to_scale(scale, **characterization.to_shape_kwargs())`.
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
    target_length = 263
    out = _as_1d_float_tuple(scaling_function([float(i) for i in range(source_length)], target_length))
    characterization, is_tied = _characterize_affine(out, source_length, exact_factor_spacing=None)
    assert not is_tied, "unreachable: no exact_factor candidate and source != target"
    return replace(
        characterization,
        probe_results=(ProbeResult(Probe(source_length, target_length), out, None),),
    )


def characterize_shape_factor_scaling_method(
    scaling_function: Callable[[Sequence[float], float], Iterable[float]],
) -> ScalingMethodCharacterization:
    """Characterize a scaling implementation that accepts a shape *multiplier*
    (scipy.ndimage.zoom / skimage.transform.rescale convention: factor < 1 for downscaling,
    factor > 1 for upscaling).
    Feed the result to `BlueprintFactors.apply_to_scale(scale, **characterization.to_factor_kwargs())`.

    Note:
        clearscale.Factor is a shape *divisor*, so when calling such a scaling function
        to execute the blueprint, pass `factor.inverted().values_tuple()`.
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
    """Characterize a scaling implementation that accepts a shape *divisor*
    (skimage.measure.block_reduce convention, manual striding, and similar:
    factor > 1 for downscaling, factor < 1 for upscaling).
    Feed the result to `BlueprintFactors.apply_to_scale(scale, **characterization.to_factor_kwargs())`.

    Since clearscale.Factor is itself a shape divisor, you can directly pass the blueprint's
    factors into such a scaling function, usually like `factor.values_tuple()`.
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
    """Shared implementation for the two factor-parametrized characterize_* functions.

    `factor_to_spacing` resolves a probe's factor value to its predicted "exact_factor"
    pixel spacing under the caller's convention (1/f for multiplier, f for divisor).
    """
    discriminating_runs = _run_probes(scaling_function, discriminating_probes)
    exact_runs = _run_probes(scaling_function, exact_probes)
    discriminating_results = {run.probe: run.output for run in discriminating_runs}
    exact_results = {run.probe: run.output for run in exact_runs}

    rounding: Union[RoundingMethod, Literal["indeterminate"]]
    try:
        rounding = _detect_rounding(discriminating_results, rounding_to_implementation)
    except FailedUnevenScalingError:
        some_rounding = next(iter(rounding_to_implementation.values()))
        if all(output is not None and len(output) == some_rounding(probe) for probe, output in exact_results.items()):
            # All the regular probes failed, but at least the scaling function accepted probes that need no rounding
            rounding = "error_on_round"
        else:
            raise ValueError(
                "Scaling method failed to execute more than one attempted parameter combination. Attempts with rounding: "
                f"{'; '.join(run.describe() for run in discriminating_runs)}. "
                f"Attempts without rounding: {'; '.join(run.describe() for run in exact_runs)}"
            )
    except IndeterminateRoundingError:
        rounding = "indeterminate"

    # Try discriminating first, then exact as fallback
    all_successes = [(p, o) for p, o in (*discriminating_results.items(), *exact_results.items()) if o is not None]
    # exact_results can't distinguish pixel_sizing "shape_ratio" from "exact_factor" (without shape rounding, both produce the same number)
    pixel_sizing_tie_expected = rounding == "error_on_round"
    # If no probe with a fractional factor succeeded, the method only accepts integer scaling factors
    uses_integer_factors_only = all(
        isinstance(probe.factor, int) or probe.factor.is_integer() for probe, _o in all_successes
    )
    characterization = _get_first_unambiguous_affine_characterization(
        all_successes, factor_to_spacing, pixel_sizing_tie_expected, uses_integer_factors_only
    )

    return replace(
        characterization,
        rounding=rounding,
        probe_results=(*discriminating_runs, *exact_runs),
    )


def _run_probes(
    scaling_function: Callable[[Sequence[float], float], Iterable[float]], probes: Sequence[Probe]
) -> List[ProbeResult]:
    results: List[ProbeResult] = []
    for probe in probes:
        coords = [float(i) for i in range(probe.source_length)]
        try:
            scaled = scaling_function(coords, probe.factor)
        except Exception as e:
            results.append(ProbeResult(probe, None, f"{type(e).__name__}: {e}"))
            continue
        results.append(ProbeResult(probe, _as_1d_float_tuple(scaled), None))
    return results


def _detect_rounding(
    discriminating_results: Mapping[Probe, Optional[Sequence[float]]],
    rounding_to_implementation: RoundingRuleTable,
) -> RoundingMethod:
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
    assert (
        ranked[1][1] != 0
    ), f"Dev error: Probes should be chosen such that only one rounding behavior can ever match perfectly ({best_rounding!r} and {ranked[1][0]!r} both matched)"
    assert best_rounding in ("ceil", "floor", "round", "round_half_up")
    return best_rounding


def _get_first_unambiguous_affine_characterization(
    successes: Sequence[Tuple[Probe, Sequence[float]]],
    factor_to_spacing: Callable[[float], float],
    pixel_sizing_tie_expected: bool,
    uses_integer_factors_only: bool,
) -> ScalingMethodCharacterization:
    for probe, output in successes:
        try:
            characterization, is_pixel_sizing_tied = _characterize_affine(
                output,
                probe.source_length,
                exact_factor_spacing=factor_to_spacing(probe.factor),
                uses_integer_factors_only=uses_integer_factors_only,
            )
        except ValueError as e:
            raise ValueError(f"Probe {probe!r} produced unusable output. {e}") from e

        if not is_pixel_sizing_tied or pixel_sizing_tie_expected:
            return characterization

    raise ValueError("No probe produced an unambiguous pixel-sizing characterization.")


def _characterize_affine(
    out: Sequence[float],
    source_length: int,
    *,
    exact_factor_spacing: Optional[float],
    uses_integer_factors_only: bool = False,
) -> Tuple[ScalingMethodCharacterization, bool]:
    """Characterize the output produced after running a particular probe.

    `exact_factor_spacing` is the expected pixel spacing multiplier given the scaling factor
    the method was parametrized with, if any
    (i.e. expected if the method multiplies input spacing exactly by the factor).

    `uses_integer_factors_only` is whether the method only accepted integer scaling factors (probably works in discrete bins).
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
    if uses_integer_factors_only and translating is half_pixel_space_preservation:
        # A method that only accepts integer factors is more likely using discrete bins.
        # half_pixel_space_preservation and discrete_bin_center are numerically identical when spacing is integer:
        # 0.5 * (spacing - 1) == 0.5 * (ceil(spacing) - 1).
        # So prefer the more descriptive discrete_bin_center when this is the case.
        translating_tie_threshold = 1e-9 * abs(spacing)
        """Shift rules whose errors differ by less than this are considered equal."""
        bin_center_error = dict(translation_scores)[discrete_bin_center]
        if bin_center_error - translating_error <= translating_tie_threshold:
            translating, translating_error = discrete_bin_center, bin_center_error

    warning_msgs: List[str] = []
    if pixel_sizing_error > relative_error_warning_threshold * abs(spacing):
        warning_msgs.append(
            f"Ambiguous or noisy pixel-size characterization (matched {pixel_sizing!r}, "
            f"error {pixel_sizing_error:.3g}, spacing {spacing:.3g})."
        )
    if translating_error > relative_error_warning_threshold * abs(spacing):
        warning_msgs.append(
            f"Ambiguous or noisy translation characterization (matched {translating.__name__!r}, "
            f"error {translating_error:.3g}, spacing {spacing:.3g}). "
            "If no built-in shift function describes your method, you can pass your own as `translating`."
        )

    assert pixel_sizing in ("shape_ratio", "corner_ratio", "exact_factor")
    characterization = ScalingMethodCharacterization(
        translating=translating,
        translating_error=translating_error,
        pixel_sizing=pixel_sizing,
        pixel_sizing_error=pixel_sizing_error,
        rounding=None,
        warnings=tuple(warning_msgs),
    )
    return characterization, is_pixel_sizing_tie


def _as_1d_float_tuple(values: Iterable[float]) -> Tuple[float, ...]:
    if getattr(values, "ndim", 1) != 1:
        raise ValueError(f"Scaling function must return a one-dimensional array. Received: {values!r}")

    try:
        return tuple(float(value) for value in values)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Scaling function must return a one-dimensional array. Received: {values!r}") from e


__all__ = [
    "Probe",
    "ProbeResult",
    "discrete_bin_center",
    "half_pixel_space_preservation",
    "first_value_decimation",
    "characterize_shape_scaling_method",
    "characterize_shape_factor_scaling_method",
    "characterize_step_factor_scaling_method",
]
