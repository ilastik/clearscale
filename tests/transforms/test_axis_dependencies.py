import math

import pytest

from clearscale._transforms import (
    AffineTransform,
    BijectionTransform,
    ByDimensionTransform,
    CoordinatesTransform,
    DisplacementsTransform,
    IdentityTransform,
    MapAxisTransform,
    ProjectAxisTransform,
    RotationTransform,
    ScaleTransform,
    TranslationTransform,
    _ByDimensionChild,
)

COS, SIN = math.cos(0.3), math.sin(0.3)


@pytest.mark.parametrize(
    "transform, source_ndim, expected",
    [
        (IdentityTransform(), 3, ((0,), (1,), (2,))),
        (ScaleTransform((2.0, 2.0)), 2, ((0,), (1,))),
        (TranslationTransform((1.0, 1.0, 1.0)), 3, ((0,), (1,), (2,))),
        (MapAxisTransform((2, 0, 1)), 3, ((2,), (0,), (1,))),
        (ProjectAxisTransform(drops=(0,)), 3, ((1,), (2,))),
        (ProjectAxisTransform(inserts=(0,)), 2, ((), (0,), (1,))),
        (ProjectAxisTransform(drops=(1,), inserts=(0, 3)), 3, ((), (0,), (2,), ())),
        (ProjectAxisTransform(drops=(0,), inserts=(0,)), 3, ((), (1,), (2,))),
        (RotationTransform(rotation=((1.0, 0.0), (0.0, 1.0))), 2, ((0,), (1,))),  # identity
        # Mixing: rotation in (1, 2) leaves 0 alone; e.g. rotate yx of zyx
        (RotationTransform(rotation=((1.0, 0.0, 0.0), (0.0, COS, -SIN), (0.0, SIN, COS))), 3, ((0,), (1, 2), (1, 2))),
        (RotationTransform(rotation=((0.0, -1.0), (1.0, 0.0))), 2, ((1,), (0,))),  # one-to-one, though it flips
        (AffineTransform(affine=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))), 2, ((0,), (1,))),  # identity
        (AffineTransform(affine=((1.0, 0.5, 3.0), (0.0, 1.0, 4.0))), 2, ((0, 1), (1,))),  # shear
        (AffineTransform(affine=((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 5.0))), 3, ((0,), ())),  # constant output axis
        (AffineTransform(affine=((2.0, 0.0, 0.0, 1.0),)), 3, ((0,),)),  # dropping affine
        (AffineTransform(affine=((1.0, 1e-17, 0.0), (0.0, 1.0, 0.0))), 2, ((0,), (1,))),  # tolerance
    ],
)
def test_axis_dependencies(transform, source_ndim, expected):
    # Expected: tuple of, per target axis, one tuple of source axes it depends on
    assert transform._axis_dependencies(source_ndim) == expected


def test_unknown_when_payload_is_not_loaded_or_not_determinable():
    assert RotationTransform(_ome_zarr_path="r")._axis_dependencies(2) is None
    assert AffineTransform(_ome_zarr_path="a")._axis_dependencies(2) is None
    assert CoordinatesTransform(path="c")._axis_dependencies(2) is None
    assert DisplacementsTransform(path="d")._axis_dependencies(2) is None


def test_bijection_prefers_the_forward_transform():
    forward = ScaleTransform((2.0, 2.0))
    inverse = RotationTransform(rotation=((COS, -SIN), (SIN, COS)))
    assert forward._axis_dependencies(2) != inverse._axis_dependencies(
        2
    ), "different forward vs inverse dependencies is the point"

    bijection = BijectionTransform(forward=forward, inverse=inverse)
    assert bijection._axis_dependencies(2) == forward._axis_dependencies(2)


@pytest.mark.parametrize(
    "inverse, expected",
    [
        (ScaleTransform((0.5, 0.5, 0.5)), ((0,), (1,), (2,))),  # not a real test (same as-is and inverse)
        (MapAxisTransform((2, 0, 1)), ((1,), (2,), (0,))),  # non-inverse would be ((2,), (0,), (1,))
        # Same as the MapAxis above, but expressed as a rotation matrix
        (RotationTransform(rotation=((0, 0, 1), (1, 0, 0), (0, 1, 0))), ((1,), (2,), (0,))),
        (
            AffineTransform(affine=((1.0, 1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0))),
            ((1,), (0, 1), (2,)),  # non-inverse would be ((0, 1), (0,), (2,))
        ),
    ],
)
def test_bijection_with_undeterminable_forward_uses_the_inverted_inverse(inverse, expected):
    bijection = BijectionTransform(forward=CoordinatesTransform(path="c"), inverse=inverse)
    assert bijection.forward._axis_dependencies(3) is None
    assert bijection._axis_dependencies(3) == expected


@pytest.mark.parametrize("inverse", [DisplacementsTransform(path="i"), AffineTransform(_ome_zarr_path="a")])
def test_bijection_is_unknown_if_neither_direction_can_tell(inverse):
    bijection = BijectionTransform(forward=CoordinatesTransform(path="f"), inverse=inverse)
    assert bijection._axis_dependencies(2) is None


def test_by_dimension_remaps_child_dependencies_into_global_indices():
    rotation = RotationTransform(rotation=((COS, -SIN), (SIN, COS)))
    transform = ByDimensionTransform(
        (
            _ByDimensionChild(source_indices=(1, 2), target_indices=(2, 0), transform=rotation),
            _ByDimensionChild(source_indices=(0,), target_indices=(1,), transform=ScaleTransform((2.0,))),
        )
    )
    # target 0 and 2 come from source 1 / source 2 mixing (rotation)
    # target 1 comes from source 0 (Scale)
    assert transform._axis_dependencies(3) == ((1, 2), (0,), (1, 2))


def test_by_dimension_with_unknown_child_is_unknown():
    transform = ByDimensionTransform(
        (_ByDimensionChild(source_indices=(0,), target_indices=(0,), transform=CoordinatesTransform(path="c")),)
    )
    assert transform._axis_dependencies(1) is None


@pytest.mark.parametrize(
    "transform, source_ndim",
    [
        (ScaleTransform((2.0, 2.0)), 3),
        (TranslationTransform((1.0,)), 2),
        (MapAxisTransform((1, 0)), 3),
        (ProjectAxisTransform(drops=(2,)), 2),
        (ProjectAxisTransform(inserts=(3,)), 2),
        (RotationTransform(rotation=((0.0, -1.0), (1.0, 0.0))), 3),
        (AffineTransform(affine=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))), 3),
    ],
)
def test_ndim_not_fitting_payload_raises(transform, source_ndim):
    # Slightly awkward error message. Public methods that call this should catch and provide something nicer.
    with pytest.raises(
        ValueError, match=r"(expects|requires).*(source|target) axes, but its (source|target) coordinate system has"
    ):
        transform._axis_dependencies(source_ndim)
