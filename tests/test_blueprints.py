from typing import cast, Literal, Dict, TypedDict

import pytest

from clearscale import BlueprintFactors, BlueprintShapes, DuplicatePolicy, Factor, PixelSize, Shape
from clearscale._axis_values import RoundingMethod


class TestFactorsToShapes:
    colliding_factors = BlueprintFactors(
        {
            "s0": Factor(y=3.0),  # 10 // 3.0 = 3
            "s1": Factor(y=3.1),  # 10 // 3.1 = 3  (dup of s0)
            "s2": Factor(y=3.3),  # 10 // 3.3 = 3  (dup of s0)
            "s3": Factor(y=3.5),  # 10 // 3.5 = 2
            "s4": Factor(y=3.9),  # 10 // 3.9 = 2  (dup of s2)
        }
    )
    reference = Shape(y=10)

    @pytest.mark.parametrize(
        "on_duplicate", [DuplicatePolicy.KEEP_ALL, DuplicatePolicy.KEEP_FIRST, DuplicatePolicy.KEEP_LAST]
    )
    def test_no_collisions_unaffected_by_policy(self, on_duplicate):
        factors = BlueprintFactors({"s0": Factor(y=1.0), "s1": Factor(y=2.0), "s2": Factor(y=4.0)})
        shapes = factors.to_shapes(Shape(y=16), rounding="ceil", on_duplicate=on_duplicate)
        assert list(shapes.items()) == [("s0", Shape(y=16)), ("s1", Shape(y=8)), ("s2", Shape(y=4))]

    def test_default_keeps_first_of_each_colliding_group(self):
        shapes = self.colliding_factors.to_shapes(self.reference, rounding="floor")  # default: KEEP_FIRST
        assert list(shapes.items()) == [("s0", Shape(y=3)), ("s3", Shape(y=2))]

    def test_keep_last_of_each_colliding_group(self):
        shapes = self.colliding_factors.to_shapes(
            self.reference, rounding="floor", on_duplicate=DuplicatePolicy.KEEP_LAST
        )
        assert list(shapes.items()) == [("s2", Shape(y=3)), ("s4", Shape(y=2))]

    def test_keep_all_preserves_every_factor_including_collisions(self):
        shapes = self.colliding_factors.to_shapes(
            self.reference, rounding="floor", on_duplicate=DuplicatePolicy.KEEP_ALL
        )
        assert list(shapes.items()) == [
            ("s0", Shape(y=3)),
            ("s1", Shape(y=3)),
            ("s2", Shape(y=3)),
            ("s3", Shape(y=2)),
            ("s4", Shape(y=2)),
        ]

    def test_on_duplicate_error_raises(self):
        with pytest.raises(ValueError, match="Duplicate values not allowed"):
            self.colliding_factors.to_shapes(self.reference, rounding="floor", on_duplicate=DuplicatePolicy.ERROR)


class TestUniformSteps:
    uniform_steps_cases = [
        dict(
            id="downscale-basic-ceil",
            kwargs=dict(step=2, base_shape=Shape(y=64, x=64), rounding="ceil"),
            expected=[
                Shape(y=64, x=64),
                Shape(y=32, x=32),
                Shape(y=16, x=16),
                Shape(y=8, x=8),
                Shape(y=4, x=4),
                Shape(y=2, x=2),
                Shape(y=1, x=1),
            ],
        ),
        dict(
            id="downscale-custom-limit-all-as-plain-dict",
            kwargs=dict(step=2, base_shape=Shape(y=64, x=64), rounding="ceil", limit_all={"y": 8, "x": 8}),
            expected=[Shape(y=64, x=64), Shape(y=32, x=32), Shape(y=16, x=16), Shape(y=8, x=8)],
        ),
        dict(
            id="downscale-limit-any-stops-before-limit-all",
            # limit_any(z<=2) triggers at s2; default limit_all (all-ones) wouldn't trigger until s3.
            kwargs=dict(step=2, base_shape=Shape(z=8, y=64, x=64), rounding="ceil", limit_any=Shape(z=2)),
            expected=[Shape(z=8, y=64, x=64), Shape(z=4, y=32, x=32), Shape(z=2, y=16, x=16)],
        ),
        dict(
            id="downscale-scaled-axes-subset-leaves-channel-untouched",
            kwargs=dict(step=2, base_shape=Shape(c=3, y=64, x=64), rounding="ceil", scaled_axes="yx"),
            expected=[
                Shape(c=3, y=64, x=64),
                Shape(c=3, y=32, x=32),
                Shape(c=3, y=16, x=16),
                Shape(c=3, y=8, x=8),
                Shape(c=3, y=4, x=4),
                Shape(c=3, y=2, x=2),
                Shape(c=3, y=1, x=1),
            ],
        ),
        dict(
            id="downscale-non-power-of-two-step",
            kwargs=dict(step=3, base_shape=Shape(y=100), rounding="ceil"),
            expected=[Shape(y=100), Shape(y=34), Shape(y=12), Shape(y=4), Shape(y=2), Shape(y=1)],
        ),
        dict(
            id="upscale-orders-largest-first",
            # Regression case for the ordering fix: generation is smallest-first internally
            # (4,16 -> 8,32 -> 16,64 -> 32,128) but must come out largest-first, base_shape last.
            kwargs=dict(
                step=0.5,
                base_shape=Shape(y=4, x=16),
                rounding="round",
                limit_all=Shape(y=32, x=64),
                max_levels=10,
            ),
            expected=[Shape(y=32, x=128), Shape(y=16, x=64), Shape(y=8, x=32), Shape(y=4, x=16)],
        ),
        dict(
            id="duplicate-shapes-floor-rounding-default-dedup",
            # 5/1.1^i via floor: 5,4,4,3,3,3,2,2,2,2,1 -- default on_duplicate=KEEP_FIRST collapses
            # consecutive repeats. See dedicated tests below for KEEP_ALL / ERROR on this same case.
            kwargs=dict(step=1.1, base_shape=Shape(y=5), rounding="floor"),
            expected=[Shape(y=5), Shape(y=4), Shape(y=3), Shape(y=2), Shape(y=1)],
        ),
        dict(
            id="max-levels-cutoff-before-reaching-limit",
            kwargs=dict(step=2, base_shape=Shape(y=1024), rounding="ceil", max_levels=3),
            expected=[Shape(y=1024), Shape(y=512), Shape(y=256)],
        ),
        dict(
            id="custom-name-pattern",
            kwargs=dict(step=2, base_shape=Shape(y=16), rounding="ceil", name_pattern="lvl{}"),
            expected=[Shape(y=16), Shape(y=8), Shape(y=4), Shape(y=2), Shape(y=1)],
        ),
    ]

    @pytest.mark.parametrize("case", uniform_steps_cases, ids=lambda c: c["id"])
    def test_shapes_progression(self, case):
        name_pattern = case["kwargs"].get("name_pattern", "s{}")
        blueprint = BlueprintShapes.uniform_steps(**case["kwargs"])
        expected_keys = [name_pattern.format(i) for i in range(len(case["expected"]))]
        assert list(blueprint.keys()) == expected_keys
        assert list(blueprint.values()) == case["expected"]

    @pytest.mark.parametrize("case", uniform_steps_cases, ids=lambda c: c["id"])
    def test_factors_round_trip_to_shapes(self, case):
        shapes = BlueprintShapes.uniform_steps(**case["kwargs"])
        factors = BlueprintFactors.uniform_steps(**case["kwargs"])
        assert list(factors.keys()) == list(shapes.keys())
        assert factors.to_shapes(case["kwargs"]["base_shape"], rounding=case["kwargs"]["rounding"]) == shapes

    def test_factors_are_exact_powers_of_step(self):
        factors = BlueprintFactors.uniform_steps(step=2, base_shape=Shape(y=64, x=64), rounding="ceil")
        assert [factors[k] for k in factors.keys()] == [Factor(y=2**i, x=2**i) for i in range(7)]

    class DuplicateKwargs(TypedDict):
        step: int | float
        base_shape: Shape
        rounding: RoundingMethod

    duplicate_kwargs: DuplicateKwargs = {"step": 1.1, "base_shape": Shape(y=5), "rounding": "floor"}

    def test_duplicate_policy_keep_all_preserves_shapes(self):
        blueprint = BlueprintShapes.uniform_steps(on_duplicate=DuplicatePolicy.KEEP_ALL, **self.duplicate_kwargs)
        assert list(blueprint.values()) == [
            Shape(y=5),
            Shape(y=4),
            Shape(y=4),
            Shape(y=3),
            Shape(y=3),
            Shape(y=3),
            Shape(y=2),
            Shape(y=2),
            Shape(y=2),
            Shape(y=2),
            Shape(y=1),
        ]

    def test_duplicate_policy_keep_all_preserves_factors(self):
        factors = BlueprintFactors.uniform_steps(on_duplicate=DuplicatePolicy.KEEP_ALL, **self.duplicate_kwargs)
        assert [factors[k] for k in factors.keys()] == [Factor(y=1.1**i) for i in range(11)]

    def test_duplicate_policy_error_raises(self):
        with pytest.raises(ValueError, match="Duplicate values not allowed"):
            BlueprintShapes.uniform_steps(on_duplicate=DuplicatePolicy.ERROR, **self.duplicate_kwargs)

    def test_duplicate_policy_keep_first_factor(self):
        # KEEP_FIRST is the default policy.
        # Unlike BlueprintShapes (where duplicate *values* collapse to an indistinguishable result
        # regardless of KEEP_FIRST/KEEP_LAST), BlueprintFactors keeps the exact factor from whichever
        # original index survives dedup -- so this is where KEEP_FIRST is actually observable.
        factors = BlueprintFactors.uniform_steps(**self.duplicate_kwargs)
        assert [factors[k] for k in factors.keys()] == [Factor(y=1.1**i) for i in (0, 1, 3, 6, 10)]

    @pytest.mark.parametrize("blueprint_cls", [BlueprintShapes, BlueprintFactors], ids=["Shapes", "Factors"])
    def test_step_one_is_noop(self, blueprint_cls):
        base_shape = Shape(y=64, x=64)
        blueprint = blueprint_cls.uniform_steps(step=1, base_shape=base_shape, rounding="ceil")
        assert list(blueprint.keys()) == ["s0"]
        expected = base_shape if blueprint_cls is BlueprintShapes else Factor.identity(base_shape)
        assert blueprint["s0"] == expected

    @pytest.mark.parametrize("blueprint_cls", [BlueprintShapes, BlueprintFactors], ids=["Shapes", "Factors"])
    def test_no_matching_scaled_axes_is_noop(self, blueprint_cls):
        base_shape = Shape(y=64, x=64)
        blueprint = blueprint_cls.uniform_steps(step=2, base_shape=base_shape, rounding="ceil", scaled_axes="t")
        assert list(blueprint.keys()) == ["s0"]
        expected = base_shape if blueprint_cls is BlueprintShapes else Factor.identity(base_shape)
        assert blueprint["s0"] == expected

    @pytest.mark.parametrize("blueprint_cls", [BlueprintShapes, BlueprintFactors], ids=["Shapes", "Factors"])
    def test_downscale_powers_of_2_xyz_matches_uniform_steps(self, blueprint_cls):
        base_shape = Shape(c=3, t=5, z=8, y=64, x=64)
        common_kwargs = dict(base_shape=base_shape, rounding="ceil", limit_all=Shape(z=1, y=4, x=4), max_levels=6)
        via_wrapper = blueprint_cls.downscale_powers_of_2_xyz(**common_kwargs)
        via_direct = blueprint_cls.uniform_steps(step=2, scaled_axes="xyz", **common_kwargs)
        assert via_wrapper == via_direct

    @pytest.mark.parametrize("blueprint_cls", [BlueprintShapes, BlueprintFactors], ids=["Shapes", "Factors"])
    def test_downscale_powers_of_2_xyz_leaves_c_and_t_untouched(self, blueprint_cls):
        base_shape = Shape(c=3, t=5, z=8, y=64, x=64)
        blueprint = blueprint_cls.downscale_powers_of_2_xyz(base_shape=base_shape, rounding="ceil", max_levels=2)
        expected_c_t = (3, 5) if blueprint_cls is BlueprintShapes else (1, 1)
        for key in blueprint.keys():
            assert (blueprint[key]["c"], blueprint[key]["t"]) == expected_c_t

    @pytest.mark.parametrize("blueprint_cls", [BlueprintShapes, BlueprintFactors], ids=["Shapes", "Factors"])
    def test_downscale_powers_of_2_xyz_forwards_limit_validation_errors(self, blueprint_cls):
        with pytest.raises(ValueError, match="already exceeded"):
            blueprint_cls.downscale_powers_of_2_xyz(
                base_shape=Shape(z=8, y=8, x=8), rounding="ceil", limit_all=Shape(z=999)
            )

    error_cases = [
        dict(
            id="non-positive-step-zero",
            kwargs=dict(step=0, base_shape=Shape(y=8, x=8), rounding="ceil"),
            match="negative step size",
        ),
        dict(
            id="non-positive-step-negative",
            kwargs=dict(step=-2, base_shape=Shape(y=8, x=8), rounding="ceil"),
            match="negative step size",
        ),
        dict(
            id="limit-all-irrelevant-axis",
            kwargs=dict(step=2, base_shape=Shape(y=8, x=8), rounding="ceil", limit_all=Shape(c=1)),
            match="none of its axes",
        ),
        dict(
            id="limit-any-irrelevant-axis",
            kwargs=dict(step=2, base_shape=Shape(y=8, x=8), rounding="ceil", limit_any=Shape(c=1)),
            match="none of its axes",
        ),
        dict(
            id="limit-all-already-exceeded-downscaling",
            kwargs=dict(step=2, base_shape=Shape(y=8, x=8), rounding="ceil", limit_all=Shape(y=999, x=8)),
            match="already exceeded",
        ),
        dict(
            id="limit-any-already-exceeded-downscaling",
            kwargs=dict(step=2, base_shape=Shape(y=8, x=8), rounding="ceil", limit_any=Shape(y=999)),
            match="already exceeded",
        ),
        dict(
            id="limit-all-already-exceeded-upscaling",
            kwargs=dict(step=0.5, base_shape=Shape(y=8, x=8), rounding="round", limit_all=Shape(y=1, x=8)),
            match="already exceeded",
        ),
        dict(
            id="upscaling-without-full-coverage-or-max-levels",
            kwargs=dict(step=0.5, base_shape=Shape(y=8, x=8), rounding="round", limit_all=Shape(y=64), max_levels=0),
            match="When upscaling, limit_all must cover all scaled axes",
        ),
    ]

    @pytest.mark.parametrize("blueprint_cls", [BlueprintShapes, BlueprintFactors], ids=["Shapes", "Factors"])
    @pytest.mark.parametrize("case", error_cases, ids=lambda c: c["id"])
    def test_error_cases(self, blueprint_cls, case):
        with pytest.raises(ValueError, match=case["match"]):
            blueprint_cls.uniform_steps(**case["kwargs"])


class TestAdaptiveSteps:
    adaptive_steps_cases = [
        dict(
            id="z-lags-then-joins",
            base_shape=Shape(z=16, y=128, x=128),
            pixel_size=PixelSize(z=4.0, y=1.0, x=1.0),  # z is 4x coarser than xy
            step=2,
            rounding="ceil",
            scaled_axes="zyx",
            max_levels=8,
            limit_all=None,
            limit_any=None,
            # xy halves alone (z holds) until pixel sizes match at level 2, then all three
            # step together.
            expected=[
                ("s0", Shape(z=16, y=128, x=128)),
                ("s1", Shape(z=16, y=64, x=64)),
                ("s2", Shape(z=16, y=32, x=32)),
                ("s3", Shape(z=8, y=16, x=16)),  # z and xy now both at effective pixel size 4.0
                ("s4", Shape(z=4, y=8, x=8)),
                ("s5", Shape(z=2, y=4, x=4)),
                ("s6", Shape(z=1, y=2, x=2)),
                ("s7", Shape(z=1, y=1, x=1)),
            ],
        ),
        dict(
            id="z-lags-stopped-by-limit-any",
            base_shape=Shape(z=8, y=64, x=64),
            pixel_size=PixelSize(z=4.0, y=1.0, x=1.0),
            step=2,
            rounding="ceil",
            scaled_axes="zyx",
            max_levels=42,
            limit_all=None,
            limit_any=Shape(z=4),  # stop the moment z alone reaches 4, regardless of xy
            expected=[
                ("s0", Shape(z=8, y=64, x=64)),
                ("s1", Shape(z=8, y=32, x=32)),
                ("s2", Shape(z=8, y=16, x=16)),
                ("s3", Shape(z=4, y=8, x=8)),
            ],
        ),
        dict(
            id="inexact-isotropy-does-not-oscillate",
            base_shape=Shape(z=8, y=64, x=64),
            pixel_size=PixelSize(z=3.5, y=1.0, x=1.0),
            step=2,  # step size 2 will never let 1.0 converge to 3.5
            rounding="ceil",
            scaled_axes="zyx",
            max_levels=42,
            limit_all=None,
            limit_any=None,
            expected=[
                ("s0", Shape(z=8, y=64, x=64)),  # PixelSize(z=3.5, y=1.0, x=1.0)
                ("s1", Shape(z=8, y=32, x=32)),  # PixelSize(z=3.5, y=2.0, x=2.0)
                ("s2", Shape(z=8, y=16, x=16)),  # PixelSize(z=3.5, y=4.0, x=4.0) -- closest possible to isotropy
                # Simple "always scale the coarsest axis" rule would intersperse Shape(z=4, y=16, x=16)) here,
                # and then continue to oscillate between scaling xy one step, z the other step.
                ("s3", Shape(z=4, y=8, x=8)),  # PixelSize(z=7.0, y=8.0, x=8.0)
                ("s4", Shape(z=2, y=4, x=4)),  # PixelSize(z=14.0, y=16.0, x=16.0)
                ("s5", Shape(z=1, y=2, x=2)),  # PixelSize(z=28.0, y=32.0, x=32.0)
                ("s6", Shape(z=1, y=1, x=1)),
            ],
        ),
        dict(
            id="isotropic-odd-size-ceil",
            base_shape=Shape(y=17, x=17),
            pixel_size=PixelSize(y=1.0, x=1.0),  # already isotropic -> same as uniform_steps
            step=2,
            rounding="ceil",
            scaled_axes="yx",
            max_levels=10,
            limit_all=None,
            limit_any=None,
            expected=[
                ("s0", Shape(y=17, x=17)),
                ("s1", Shape(y=9, x=9)),
                ("s2", Shape(y=5, x=5)),
                ("s3", Shape(y=3, x=3)),
                ("s4", Shape(y=2, x=2)),
                ("s5", Shape(y=1, x=1)),
            ],
        ),
        dict(
            id="isotropic-odd-size-floor",
            base_shape=Shape(y=17, x=17),
            pixel_size=PixelSize(y=1.0, x=1.0),
            step=2,
            rounding="floor",
            scaled_axes="yx",
            max_levels=10,
            limit_all=None,
            limit_any=None,
            expected=[
                # floor rounding leads to one less level than ceil rounding in the previous case
                ("s0", Shape(y=17, x=17)),
                ("s1", Shape(y=8, x=8)),
                ("s2", Shape(y=4, x=4)),
                ("s3", Shape(y=2, x=2)),
                ("s4", Shape(y=1, x=1)),
            ],
        ),
        dict(
            id="channel-axis-untouched",
            base_shape=Shape(c=3, z=8, y=64, x=64),
            pixel_size=PixelSize(z=4.0, y=1.0, x=1.0),  # no value for c should be ok (not scaled)
            step=2,
            rounding="ceil",
            scaled_axes="zyx",
            max_levels=4,
            limit_all=None,
            limit_any=None,
            expected=[
                ("s0", Shape(c=3, z=8, y=64, x=64)),
                ("s1", Shape(c=3, z=8, y=32, x=32)),
                ("s2", Shape(c=3, z=8, y=16, x=16)),
                ("s3", Shape(c=3, z=4, y=8, x=8)),
            ],
        ),
        dict(
            id="upscaling-coarse-axis-catches-up",
            base_shape=Shape(y=4, x=16),
            pixel_size=PixelSize(y=4.0, x=1.0),
            step=0.5,
            rounding="round",
            scaled_axes="yx",
            max_levels=4,
            limit_all=Shape(y=64, x=64),
            limit_any=None,
            expected=[
                # Inverted order (base at bottom); even for upscaling, scales must still be largest-to-smallest.
                # 64x64 excluded due to max_levels=4.
                ("s0", Shape(y=32, x=32)),
                ("s1", Shape(y=16, x=16)),  # y has caught up to x pixel size here
                ("s2", Shape(y=8, x=16)),
                ("s3", Shape(y=4, x=16)),  # original base_shape
            ],
        ),
    ]

    @pytest.mark.parametrize("case", adaptive_steps_cases, ids=lambda c: c["id"])
    def test_shapes_progression(self, case):
        blueprint = BlueprintShapes.adaptive_steps(
            step=case["step"],
            base_shape=case["base_shape"],
            pixel_size=case["pixel_size"],
            rounding=cast(Literal["ceil", "floor", "round"], case["rounding"]),
            scaled_axes=case["scaled_axes"],
            max_levels=case["max_levels"],
            limit_all=case["limit_all"],
            limit_any=case["limit_any"],
        )
        assert list(blueprint.items()) == case["expected"]
        assert list(blueprint.keys()) == [key for key, _ in case["expected"]]
        for key, expected_shape in case["expected"]:
            assert blueprint[key] == expected_shape

    @pytest.mark.parametrize("scaled_axes", ["yx", "y"])
    def test_matches_uniform_steps_when_isotropic(self, scaled_axes):
        base_shape = Shape(y=17, x=17)
        pixel_size = PixelSize.identity(base_shape)
        shared_kwargs = dict(step=2, base_shape=base_shape, rounding="ceil", scaled_axes=scaled_axes, max_levels=8)

        adaptive = BlueprintShapes.adaptive_steps(pixel_size=pixel_size, **shared_kwargs)  # pyright: ignore
        uniform = BlueprintShapes.uniform_steps(**shared_kwargs)  # pyright: ignore

        assert adaptive == uniform

    def test_step_one_is_noop(self):
        base_shape = Shape(z=8, y=64, x=64)
        blueprint = BlueprintShapes.adaptive_steps(
            step=1, base_shape=base_shape, pixel_size=PixelSize(z=4.0, y=1.0, x=1.0), rounding="ceil"
        )
        assert list(blueprint.keys()) == ["s0"]
        assert blueprint["s0"] == base_shape

    def test_no_matching_scaled_axes_is_noop(self):
        base_shape = Shape(y=64, x=64)
        blueprint = BlueprintShapes.adaptive_steps(
            step=2,
            base_shape=base_shape,
            pixel_size=PixelSize(t=1.0),  # only covers an axis that isn't in base_shape
            rounding="ceil",
            scaled_axes="t",
        )
        assert list(blueprint.keys()) == ["s0"]
        assert blueprint["s0"] == base_shape

    def test_factors_are_exact_powers_of_step(self):
        factors = BlueprintFactors.adaptive_steps(
            step=2,
            base_shape=Shape(z=8, y=64, x=64),
            pixel_size=PixelSize(z=4.0, y=1.0, x=1.0),
            rounding="ceil",
            scaled_axes="zyx",
            max_levels=7,
        )
        assert list(factors.keys()) == ["s0", "s1", "s2", "s3", "s4", "s5", "s6"]
        assert factors["s0"] == Factor(z=1, y=1, x=1)
        assert factors["s1"] == Factor(z=1, y=2, x=2)
        assert factors["s2"] == Factor(z=1, y=4, x=4)
        assert factors["s3"] == Factor(z=2, y=8, x=8)
        assert factors["s4"] == Factor(z=4, y=16, x=16)
        assert factors["s5"] == Factor(z=8, y=32, x=32)
        assert factors["s6"] == Factor(z=16, y=64, x=64)

    @pytest.mark.parametrize("case", adaptive_steps_cases, ids=lambda c: c["id"])
    def test_factors_round_trip_to_shapes(self, case):
        rounding = cast(Literal["ceil", "floor", "round"], case["rounding"])
        shared_kwargs = dict(
            step=case["step"],
            base_shape=case["base_shape"],
            pixel_size=case["pixel_size"],
            rounding=rounding,
            scaled_axes=case["scaled_axes"],
            max_levels=case["max_levels"],
            limit_all=case["limit_all"],
            limit_any=case["limit_any"],
        )
        shapes = BlueprintShapes.adaptive_steps(**shared_kwargs)  # pyright: ignore
        factors = BlueprintFactors.adaptive_steps(**shared_kwargs)  # pyright: ignore

        assert list(factors.keys()) == list(shapes.keys())
        assert factors.to_shapes(case["base_shape"], rounding=rounding) == shapes

    @pytest.mark.parametrize("blueprint_cls", [BlueprintShapes, BlueprintFactors], ids=["Shapes", "Factors"])
    def test_adaptive_powers_of_2_xyz_matches_uniform_steps(self, blueprint_cls):
        base_shape = Shape(c=3, t=5, z=8, y=64, x=64)
        base_pixel_size = PixelSize(z=6.0, y=1.5, x=1.5)
        common_kwargs = dict(
            base_shape=base_shape,
            pixel_size=base_pixel_size,
            rounding="ceil",
            limit_all=Shape(z=1, y=4, x=4),
            max_levels=6,
        )
        via_wrapper = blueprint_cls.adaptive_powers_of_2_xyz(**common_kwargs)
        via_direct = blueprint_cls.adaptive_steps(step=2, scaled_axes="xyz", **common_kwargs)
        assert list(via_wrapper.items()) == list(via_direct.items())

    error_cases = [
        pytest.param(
            dict(step=0, base_shape=Shape(y=8, x=8), pixel_size=PixelSize(y=1.0, x=1.0), rounding="ceil"),
            "negative step size",
            id="non-positive-step",
        ),
        pytest.param(
            dict(
                step=2,
                base_shape=Shape(z=8, y=8, x=8),
                pixel_size=PixelSize(y=1.0, x=1.0),
                rounding="ceil",
            ),
            "pixel_size must provide values for all scaled_axes. Missing",
            id="pixel_size-missing-scaled-axis",
        ),
        pytest.param(
            dict(
                step=2,
                base_shape=Shape(y=8, x=8),
                pixel_size=PixelSize(y=1.0, x=1.0),
                rounding="ceil",
                limit_all=Shape(c=1),
            ),
            "Cannot apply a limit if none of its axes",
            id="limit_all-irrelevant-axis",
        ),
        pytest.param(
            dict(
                step=2,
                base_shape=Shape(y=8, x=8),
                pixel_size=PixelSize(y=1.0, x=1.0),
                rounding="ceil",
                limit_all=Shape(y=999, x=999),
            ),
            "limit already exceeded",
            id="limit_all-already-exceeded",
        ),
        pytest.param(
            dict(
                step=2,
                base_shape=Shape(y=8, x=8),
                pixel_size=PixelSize(y=1.0, x=1.0),
                rounding="ceil",
                limit_any=Shape(y=999),
            ),
            "limit already exceeded",
            id="limit_any-already-exceeded",
        ),
        pytest.param(
            dict(
                step=0.5,
                base_shape=Shape(y=8, x=8),
                pixel_size=PixelSize(y=1.0, x=1.0),
                rounding="round",
                limit_all=Shape(y=64),  # doesn't cover x
                max_levels=0,
            ),
            "When upscaling, limit_all must cover all scaled axes",
            id="upscaling-without-full-coverage-or-max-levels",
        ),
    ]

    @pytest.mark.parametrize("blueprint_cls", [BlueprintShapes, BlueprintFactors], ids=["Shapes", "Factors"])
    @pytest.mark.parametrize("kwargs,expected_message", error_cases)
    def test_error_cases(self, blueprint_cls, kwargs: Dict, expected_message):
        with pytest.raises(ValueError, match=str(expected_message)):
            blueprint_cls.adaptive_steps(**kwargs)  # pyright: ignore


class TestOutputOrder:
    """Ensure that all blueprint-generating methods produce largest-to-smallest ordering,
    as required by every multiscale metadata format."""

    @staticmethod
    def _assert_non_increasing(shapes, *, scaled_axes):
        for prev, cur in zip(shapes, shapes[1:]):
            for axis in scaled_axes:
                assert cur[axis] <= prev[axis], f"{axis}: {prev} -> {cur} increased"

    directions = [2, 0.5]  # downscaling, upscaling
    roundings = ["ceil", "floor", "round"]

    @pytest.mark.parametrize("step", directions)
    @pytest.mark.parametrize("rounding", roundings)
    def test_uniform_steps_shapes_ordered(self, step, rounding):
        rounding = cast(Literal["ceil", "floor", "round"], rounding)
        base_shape = Shape(y=17, x=17)
        bp = BlueprintShapes.uniform_steps(
            step=step,
            base_shape=base_shape,
            rounding=rounding,
            limit_all=Shape(y=64, x=64) if step < 1 else None,
            max_levels=6,
        )
        self._assert_non_increasing(list(bp.values()), scaled_axes=("y", "x"))

    @pytest.mark.parametrize("step", directions)
    @pytest.mark.parametrize("rounding", roundings)
    def test_uniform_steps_factors_ordered(self, step, rounding):
        rounding = cast(Literal["ceil", "floor", "round"], rounding)
        base_shape = Shape(y=17, x=17)
        factors = BlueprintFactors.uniform_steps(
            step=step,
            base_shape=base_shape,
            rounding=rounding,
            limit_all=Shape(y=64, x=64) if step < 1 else None,
            max_levels=6,
        )
        shapes = factors.to_shapes(base_shape, rounding=rounding)
        self._assert_non_increasing(list(shapes.values()), scaled_axes=("y", "x"))

    @pytest.mark.parametrize("step", directions)
    @pytest.mark.parametrize("rounding", roundings)
    def test_adaptive_steps_shapes_ordered(self, step, rounding):
        rounding = cast(Literal["ceil", "floor", "round"], rounding)
        base_shape = Shape(z=8, y=64, x=64)
        pixel_size = PixelSize(z=4.0, y=1.0, x=1.0)
        bp = BlueprintShapes.adaptive_steps(
            step=step,
            base_shape=base_shape,
            pixel_size=pixel_size,
            rounding=rounding,
            limit_all=Shape(z=64, y=256, x=256) if step < 1 else None,
            max_levels=8,
        )
        self._assert_non_increasing(list(bp.values()), scaled_axes=("z", "y", "x"))

    @pytest.mark.parametrize("step", directions)
    @pytest.mark.parametrize("rounding", roundings)
    def test_adaptive_steps_factors_ordered(self, step, rounding):
        rounding = cast(Literal["ceil", "floor", "round"], rounding)
        base_shape = Shape(z=8, y=64, x=64)
        pixel_size = PixelSize(z=4.0, y=1.0, x=1.0)
        factors = BlueprintFactors.adaptive_steps(
            step=step,
            base_shape=base_shape,
            pixel_size=pixel_size,
            rounding=rounding,
            limit_all=Shape(z=64, y=256, x=256) if step < 1 else None,
            max_levels=8,
        )
        shapes = factors.to_shapes(base_shape, rounding=rounding)
        self._assert_non_increasing(list(shapes.values()), scaled_axes=("z", "y", "x"))

    def test_uniform_steps_upscaling_puts_largest_first(self):
        base_shape = Shape(y=4, x=4)
        bp = BlueprintShapes.uniform_steps(
            step=0.5, base_shape=base_shape, rounding="round", limit_all=Shape(y=32, x=32), max_levels=4
        )
        keys = list(bp.keys())
        shapes = list(bp.values())
        assert shapes == sorted(shapes, key=lambda s: s["y"], reverse=True)
        assert shapes[0]["y"] > shapes[-1]["y"]
        assert shapes[-1] == base_shape  # smallest (original) shape ends up last, not first
        assert keys[0] == "s0" and keys[-1] == f"s{len(keys) - 1}"

    def test_adaptive_steps_upscaling_puts_largest_first(self):
        base_shape = Shape(y=4, x=16)
        bp = BlueprintShapes.adaptive_steps(
            step=0.5,
            base_shape=base_shape,
            pixel_size=PixelSize(y=4.0, x=1.0),
            rounding="round",
            limit_all=Shape(y=64, x=64),
            max_levels=4,
        )
        shapes = list(bp.values())
        assert shapes[0]["y"] >= shapes[-1]["y"] and shapes[0]["x"] >= shapes[-1]["x"]
        assert shapes[-1] == base_shape

    def test_powers_of_2_xyz_wrappers_ordered(self):
        base_shape = Shape(z=8, y=64, x=64)
        kwargs = dict(base_shape=base_shape, rounding="ceil", max_levels=6)

        bp1 = BlueprintShapes.downscale_powers_of_2_xyz(**kwargs)  # pyright: ignore
        self._assert_non_increasing(list(bp1.values()), scaled_axes=("z", "y", "x"))

        kwargs["pixel_size"] = PixelSize(z=4.0, y=1.0, x=1.0)  # pyright: ignore
        bp2 = BlueprintShapes.adaptive_powers_of_2_xyz(**kwargs)  # pyright: ignore
        self._assert_non_increasing(list(bp2.values()), scaled_axes=("z", "y", "x"))
