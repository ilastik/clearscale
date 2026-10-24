import random

import pytest

from clearscale._transforms import AffineTransform, ProjectAxisTransform, TransformSequence
from clearscale._services.matrices import matrix_determinant, pivot_columns


def flatten_matrix(matrix):
    return [v for row in matrix for v in row]


def recomposed(simplified) -> AffineTransform:
    """The single AffineTransform that the simplification is equivalent to"""
    coll = TransformSequence((simplified,)).collapsed()
    assert isinstance(coll, AffineTransform), "expected only affine-representable components"
    return coll


def test_dropping_affine_simplified_residual_affine_is_invertible():
    # y and x are mixed into the single output.
    # Naively padding unit rows to make the residual affine square produces
    # ((0, 1, 1), (0, 1, 0), (0, 0, 1)) -- the all-0 first col makes this still non-invertible
    affine = ((0.0, 1.0, 1.0, 5.0),)
    transform = AffineTransform(affine=affine)
    simplified = transform.simplified()

    assert isinstance(simplified, TransformSequence)
    assert len(simplified) == 3 and isinstance(simplified[0], AffineTransform)
    residual_affine, project, translate = simplified
    assert residual_affine.is_invertible

    assert isinstance(project, ProjectAxisTransform) and project.drops == (1, 2)
    assert flatten_matrix(recomposed(simplified).affine) == pytest.approx(flatten_matrix(affine))


@pytest.mark.parametrize(
    "affine",
    [
        # Outputs (x, x, y) from inputs (y, x): Naively padding with a 0/0/1 unit column produces a square
        # residual affine with two identical rows -> non-invertible
        ((1.0, 0.0, 2.0), (1.0, 0.0, 3.0), (0.0, 1.0, 4.0)),
        ((1.0, 2.0, 3.0), (1.0, 2.0, -1.0), (0.0, 1.0, 5.0)),
    ],
)
def test_inserting_affine_simplified_residual_affine_is_invertible(affine):
    # 3 outputs from 2 inputs: Rows are [linear (3x2) | translation]
    transform = AffineTransform(affine=affine)
    simplified = transform.simplified()

    assert isinstance(simplified, TransformSequence)
    assert len(simplified) == 3 and isinstance(simplified[1], AffineTransform)
    project, residual_affine, translate = simplified
    assert residual_affine.is_invertible

    assert isinstance(project, ProjectAxisTransform) and project.inserts == (2,)
    assert flatten_matrix(recomposed(simplified).affine) == pytest.approx(flatten_matrix(affine))


def test_padding_is_unchanged_where_the_default_already_is_invertible():
    # Dropping: Rows (1, 0, 1), (0, 1, 0) get the default unit row (0, 0, 1)
    dropping = AffineTransform(affine=((1.0, 0.0, 1.0, 0.0), (0.0, 1.0, 0.0, 0.0))).simplified()
    assert isinstance(dropping, TransformSequence)
    assert len(dropping) == 2 and isinstance(dropping[0], AffineTransform)
    assert dropping[0].is_invertible
    assert dropping[0].affine == ((1, 0, 1, 0), (0, 1, 0, 0), (0, 0, 1, 0))
    # Inserting: Columns of the 3x2 matrix get the default unit column (0, 0, 1)
    inserting = AffineTransform(affine=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0))).simplified()
    assert isinstance(inserting, TransformSequence)
    assert len(inserting) == 2 and isinstance(inserting[1], AffineTransform)
    assert inserting[1].is_invertible
    assert inserting[1].affine == ((1, 0, 0, 0), (0, 1, 0, 0), (1, 1, 1, 0))


def test_matrices_that_cannot_be_padded_stay_as_they_are():
    # Existing rows are linearly dependent (row1 = row0 * 2) and hence already non-invertible (rank 1 matrix)
    simplified = AffineTransform(affine=((1.0, 2.0, 0.0, 0.0), (2.0, 4.0, 0.0, 0.0))).simplified()
    assert isinstance(simplified, TransformSequence)
    assert len(simplified) == 2 and isinstance(simplified[1], AffineTransform)
    assert not simplified[1].is_invertible
    assert simplified[1].affine == ((1.0, 2.0, 0.0), (2.0, 4.0, 0.0))


@pytest.mark.parametrize(
    "matrix, expected",
    [
        (((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), (0, 1, 2)),
        (((0.0, 1.0, 1.0), (0.0, 0.0, 1.0)), (1, 2)),  # first column is all zero: Nothing to pivot on
        (((1.0, 1.0, 0.0), (1.0, 1.0, 0.0)), (0,)),  # second row repeats the first: rank 1
        (((0.0, 0.0), (0.0, 0.0)), ()),
        (((1.0, 2.0, 3.0),), (0,)),
        (((0.0, 2.0, 3.0),), (1,)),
    ],
)
def test_pivot_columns(matrix, expected):
    assert pivot_columns(matrix, tolerance=1e-13) == expected


@pytest.mark.parametrize(
    "matrix, expected",
    [
        # Fewer rows than columns: Extra unit rows. The identity padding wherever that is invertible
        (((2.0, 0.0, 0.0), (1.0, 3.0, 0.0)), ((2.0, 0.0, 0.0), (1.0, 3.0, 0.0), (0.0, 0.0, 1.0))),
        # ... and the unit row of an unpivoted column otherwise (the identity padding (0, 0, 1) repeats row 2)
        (((0.0, 1.0, 1.0), (0.0, 0.0, 1.0)), ((0.0, 1.0, 1.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))),
        # More rows than columns: The same for columns
        (((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 1.0))),
        (((1.0, 0.0), (1.0, 0.0), (0.0, 1.0)), ((1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 0.0))),
        # Square matrices are unchanged
        (((1.0, 2.0), (3.0, 4.0)), ((1.0, 2.0), (3.0, 4.0))),
    ],
)
def test_pad_to_square_matrix(matrix, expected):
    assert AffineTransform._pad_to_square_matrix(matrix) == expected


@pytest.mark.parametrize("seed", range(200))
def test_padding_full_rank_matrices_is_always_invertible(seed):
    rng = random.Random(seed)
    rows, cols = rng.randint(1, 5), rng.randint(1, 5)
    matrix = tuple(tuple(rng.choice([0.0, 0.0, 1.0, -1.0, 2.0, 0.5]) for _ in range(cols)) for _ in range(rows))
    full_rank = len(pivot_columns(matrix, tolerance=1e-13)) == min(rows, cols)
    padded = AffineTransform._pad_to_square_matrix(matrix)
    assert len(padded) == len(padded[0]) == max(rows, cols)
    assert tuple(tuple(row[:cols]) for row in padded[:rows]) == matrix, "the original entries are kept"
    if full_rank:
        assert matrix_determinant(padded, tolerance=1e-13) != 0.0
