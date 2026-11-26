import pytest

from clearscale._transforms import (
    AffineTransform,
    IdentityTransform,
    MapAxisTransform,
    ProjectAxisTransform,
    RotationTransform,
    ScaleTransform,
    TranslationTransform,
)

SHEAR = ((1.0, 0.5, 0.0), (0.0, 1.0, 0.0))
"""2x2 affine (with offset column) that mixes the axes"""


@pytest.mark.parametrize(
    "transform, source_ndim, expected",
    [
        (IdentityTransform(), 3, (0, 1, 2)),
        (ScaleTransform((2.0, 3.0)), 2, (0, 1)),
        (TranslationTransform((1.0, 2.0)), 2, (0, 1)),
        (MapAxisTransform((2, 0, 1)), 3, (2, 0, 1)),
        (ProjectAxisTransform(drops=(0,)), 3, (1, 2)),
        (ProjectAxisTransform(inserts=(0,)), 2, (None, 0, 1)),
        (ProjectAxisTransform(drops=(1,), inserts=(1,)), 3, (0, None, 2)),
    ],
)
def test_axes_that_are_mapped_one_to_one_continue_their_source_axis(transform, source_ndim, expected):
    assert transform._axis_lineage(source_ndim) == expected


def test_mixing_square_transforms_keep_their_axes_in_place():
    assert AffineTransform(affine=SHEAR)._axis_lineage(2) == (0, 1)
    assert RotationTransform(rotation=((0.0, -1.0), (1.0, 0.0)))._axis_lineage(2) == (0, 1)


def test_affines_that_add_or_drop_axes_do_not_say_which_axes_they_continue():
    adding = AffineTransform(affine=((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)))  # 3 outputs from 1 input
    dropping = AffineTransform(affine=((1.0, 1.0, 0.0),))  # 1 output from 2 inputs
    assert adding._axis_lineage(1) is None
    assert dropping._axis_lineage(2) is None


def test_path_backed_scale_still_knows_its_axes_but_rotation_and_affine_do_not():
    assert ScaleTransform(scale=(), _ome_zarr_path="s")._axis_lineage(2) == (0, 1)  # per axis, whatever its values
    assert RotationTransform(_ome_zarr_path="r")._axis_lineage(2) is None
    assert AffineTransform(_ome_zarr_path="a")._axis_lineage(2) is None
