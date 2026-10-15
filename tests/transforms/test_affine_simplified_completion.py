import math
import random

import pytest

from clearscale._transforms import AffineTransform, ProjectAxisTransform, TransformSequence
from clearscale._services.matrices import matrix_determinant, pivot_columns
from clearscale._transforms._to_from_spatial_relation import _flatten_leaves


def flat(matrix):
    return [v for row in matrix for v in row]


def recomposed(simplified):
    """The single AffineTransform that the (possibly multi-leaf) simplification is equivalent to"""
    return TransformSequence((simplified,)).collapsed()


def affine_leaves(simplified):
    return [leaf for leaf in _flatten_leaves(TransformSequence((simplified,))) if isinstance(leaf, AffineTransform)]


@pytest.mark.parametrize(
    "affine",
    [
        # Rows y and x of a matrix. The default completion for the dropped axis is the unit row (0, 0, 1) which equals
        # the second row: Not invertible. The matrix still drops an axis and is meaningful.
        ((0.0, 1.0, 1.0, 5.0), (0.0, 0.0, 1.0, 6.0)),
        # A rotation in the z/y plane, then dropping z (rows y and x of it): x is untouched
        ((0.0, math.cos(0.3), -math.sin(0.3), 1.0), (0.0, 0.0, 1.0, 2.0)),
    ],
)
def test_dropping_affine_is_completed_with_an_invertible_matrix(affine):
    transform = AffineTransform(affine=affine)
    simplified = transform.simplified()
    leaves = affine_leaves(simplified)
    assert leaves and all(leaf.is_invertible for leaf in leaves)
    assert any(isinstance(leaf, ProjectAxisTransform) and leaf.drops for leaf in _flatten_leaves(simplified))
    assert flat(recomposed(simplified).affine) == pytest.approx(flat(affine))


@pytest.mark.parametrize(
    "affine",
    [
        # Outputs (x, x, y) from inputs (y, x): The default completion column for the inserted axis is the unit column
        # (0, 0, 1), which makes the first two rows equal: Not invertible.
        ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ((1.0, 2.0, 3.0), (1.0, 2.0, -1.0), (0.0, 1.0, 5.0)),
    ],
)
def test_inserting_affine_is_completed_with_an_invertible_matrix(affine):
    # 3 outputs from 2 inputs: Rows are [linear (3x2) | translation]
    transform = AffineTransform(affine=affine)
    simplified = transform.simplified()
    leaves = affine_leaves(simplified)
    assert leaves and all(leaf.is_invertible for leaf in leaves)
    assert any(isinstance(leaf, ProjectAxisTransform) and leaf.inserts for leaf in _flatten_leaves(simplified))
    assert flat(recomposed(simplified).affine) == pytest.approx(flat(transform.affine))


def test_padding_is_unchanged_where_the_default_already_is_invertible():
    # Dropping: Rows (1, 0, 1), (0, 1, 0) get the default unit row (0, 0, 1)
    dropping = AffineTransform(affine=((1.0, 0.0, 1.0, 0.0), (0.0, 1.0, 0.0, 0.0))).simplified()
    assert flat(affine_leaves(dropping)[0].affine) == [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    # Inserting: Columns of the 3x2 matrix get the default unit column (0, 0, 1)
    inserting = AffineTransform(affine=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0))).simplified()
    assert flat(affine_leaves(inserting)[0].affine) == [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]


def test_matrices_that_cannot_be_completed_stay_as_they_are():
    # Rank 1: Whichever unit row is appended, the matrix stays singular
    simplified = AffineTransform(affine=((1.0, 2.0, 0.0, 0.0), (2.0, 4.0, 0.0, 0.0))).simplified()
    assert not all(leaf.is_invertible for leaf in affine_leaves(simplified))


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


def test_pad_to_square_matrix_of_rank_deficient_matrix_stays_singular():
    padded = AffineTransform._pad_to_square_matrix(((1.0, 2.0, 0.0), (2.0, 4.0, 0.0)))
    assert len(padded) == len(padded[0]) == 3
    assert matrix_determinant(padded, tolerance=1e-13) == 0.0


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
