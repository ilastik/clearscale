import pytest
from clearscale import BlueprintShapes, Multiscale, PixelSize, Scale, Shape, Translation, Unit, half_pixel_shift


def test_blueprint_hash_matches_value_equality():
    left = BlueprintShapes({"s0": Shape(y=2, x=3)})
    right = BlueprintShapes({"s0": Shape(y=2, x=3)})

    assert left == right
    assert hash(left) == hash(right)


def test_multiscale_equality_and_hash_are_value_based():
    left = Multiscale({"s0": Scale(Shape(y=2, x=3))})
    right = Multiscale({"s0": Scale(Shape(y=2, x=3))})

    assert left == right
    assert {left, right} == {left}, "Value hash should lead to collapse in sets"


def test_multiscale_refs_are_hashable():
    left = Multiscale({"s0": Scale(Shape(y=2, x=3))})
    right = Multiscale({"s0": Scale(Shape(y=2, x=3))})

    assert len({left.as_ref("physical"), right.as_ref("physical")}) == 2


def test_blueprint_shapes_apply_to_scale_derives_scale_metadata_from_base():
    blueprint = BlueprintShapes({"s0": Shape(c=3, y=8, x=12), "s1": Shape(c=3, y=4, x=3)})
    base = Scale(
        shape=Shape(c=3, y=8, x=12),
        pixel_size=PixelSize(c=1.0, y=0.5, x=2.0),
        unit=Unit(c="", y="um", x="um"),
        translation=Translation(c=0.0, y=1.0, x=2.0),
    )

    multiscale = blueprint.apply_to_scale(base)

    assert multiscale == Multiscale(
        {
            "s0": base,
            "s1": Scale(
                shape=Shape(c=3, y=4, x=3),
                pixel_size=PixelSize(c=1.0, y=1.0, x=8.0),
                unit=base.unit,
                translation=base.translation,
            ),
        }
    )


def test_blueprint_shapes_apply_to_scale_can_apply_half_pixel_shift():
    blueprint = BlueprintShapes({"s0": Shape(y=8, x=8), "s1": Shape(y=4, x=2)})
    base = Scale(
        shape=Shape(y=8, x=8),
        pixel_size=PixelSize(y=2.0, x=3.0),
        translation=Translation(y=10.0, x=-5.0),
    )

    multiscale = blueprint.apply_to_scale(base, translation_shift_func=half_pixel_shift)

    assert multiscale["s1"].pixel_size == PixelSize(y=4.0, x=12.0)


@pytest.mark.parametrize("shift_func", [(lambda param1: True), (lambda scale1, scale2: True)])
def test_blueprint_shapes_apply_to_scale_rejects_malformed_shift_functions(shift_func):
    bp = BlueprintShapes({"s0": Shape(x=2)})
    base = Scale(shape=Shape(x=1))

    with pytest.raises(TypeError, match="See clearscale.half_pixel_shift for an example implementation"):
        _ = bp.apply_to_scale(base, translation_shift_func=shift_func)  # noqa


def test_proportional_blueprint_from_multiscale_template():
    ms = Multiscale(
        {
            "s0": Scale(
                shape=Shape(c=3, y=8, x=12),
                pixel_size=PixelSize(c=1.0, y=1.0, x=2.0),
                unit=Unit(c="", y="nm", x="nm"),
                translation=Translation.identity("cyx"),
            ),
            "s1": Scale(
                shape=Shape(c=3, y=4, x=3),
                pixel_size=PixelSize(c=1.0, y=2.0, x=8.0),
                unit=Unit(c="", y="nm", x="nm"),
                translation=Translation.identity("cyx"),
            ),
        }
    )

    target_shape = Shape(c=3, y=2, x=6)
    bp = BlueprintShapes.from_multiscale_rescaled(ms, target_shape=target_shape, rounding="floor")

    assert bp == BlueprintShapes({"s0": target_shape, "s1": Shape(c=3, y=1, x=1)})


def test_proportional_blueprint_rebased_on_downscale():
    ms = Multiscale(
        {
            "s0": Scale(
                shape=Shape(c=3, y=8, x=12),
                pixel_size=PixelSize(c=1.0, y=1.0, x=2.0),
                unit=Unit(c="", y="nm", x="nm"),
                translation=Translation.identity("cyx"),
            ),
            "s1": Scale(
                shape=Shape(c=3, y=4, x=3),
                pixel_size=PixelSize(c=1.0, y=2.0, x=8.0),
                unit=Unit(c="", y="nm", x="nm"),
                translation=Translation.identity("cyx"),
            ),
        }
    )

    target_shape = Shape(c=3, y=2, x=6)
    bp = BlueprintShapes.from_multiscale_rescaled(ms, target_shape=target_shape, rounding="floor", source_key="s1")

    assert bp == BlueprintShapes({"s0": Shape(c=3, y=4, x=24), "s1": target_shape})


def test_proportional_blueprint_restricted_scaling_axes():
    ms = Multiscale(
        {
            "s0": Scale(
                shape=Shape(c=3, y=8, x=12),
                pixel_size=PixelSize(c=1.0, y=1.0, x=2.0),
                unit=Unit(c="", y="nm", x="nm"),
                translation=Translation.identity("cyx"),
            ),
            "s1": Scale(
                shape=Shape(c=3, y=4, x=3),
                pixel_size=PixelSize(c=1.0, y=2.0, x=8.0),
                unit=Unit(c="", y="nm", x="nm"),
                translation=Translation.identity("cyx"),
            ),
        }
    )

    target_shape = Shape(c=3, y=2, x=6)
    bp = BlueprintShapes.from_multiscale_rescaled(ms, target_shape=target_shape, rounding="floor", scaled_axes="y")

    assert bp == BlueprintShapes({"s0": target_shape, "s1": Shape(c=3, y=1, x=6)})
