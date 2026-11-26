import pytest

from clearscale._services.matrices import has_full_rank, unit_vector_completion


class TestUnitVectorCompletion:
    def test_inserted_column_gets_a_unit_vector_at_the_row_that_the_earlier_rows_do_not_span(self):
        # Rows over the real columns: (1, 0), (0, 1), (1, 1): the third is spanned by the first two
        rows = ((1.0, 0.0, 9.0), (0.0, 1.0, 9.0), (1.0, 1.0, 9.0))
        assert unit_vector_completion(rows, (2,)) == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 1.0))

    def test_zero_rows_are_never_spanned(self):
        rows = ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        assert unit_vector_completion(rows, (1,)) == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))

    def test_values_in_the_columns_to_fill_are_ignored(self):
        assert unit_vector_completion(((1.0, 5.0), (2.0, -7.0)), (1,)) == ((1.0, 0.0), (2.0, 1.0))

    def test_several_columns_take_the_unspanned_rows_in_order(self):
        rows = ((1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0))
        assert unit_vector_completion(rows, (1, 2)) == ((1.0, 0.0, 0.0), (2.0, 1.0, 0.0), (3.0, 0.0, 1.0))

    def test_extra_columns_stay_zero_when_all_rows_are_independent(self):
        rows = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        assert unit_vector_completion(rows, (2,)) == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))

    def test_too_few_columns_complete_the_first_unspanned_rows(self):
        rows = ((1.0, 0.0), (1.0, 0.0), (1.0, 0.0))
        assert unit_vector_completion(rows, (1,)) == ((1.0, 0.0), (1.0, 1.0), (1.0, 0.0))

    @pytest.mark.parametrize("scale", [1.0, 1e-9, 1e-15, 1e6])
    def test_decisions_do_not_depend_on_the_scale_of_the_coefficients(self, scale):
        dependent = tuple(tuple(scale * v for v in row) for row in ((1.0, 2.0, 0.0), (2.0, 4.0, 0.0)))
        assert unit_vector_completion(dependent, (2,)) == (
            (1.0 * scale, 2.0 * scale, 0.0),
            (2.0 * scale, 4.0 * scale, 1.0),
        )
        independent = tuple(tuple(scale * v for v in row) for row in ((1.0, 2.0, 0.0), (0.0, 1.0, 0.0)))
        assert unit_vector_completion(independent, (2,)) == (
            (1.0 * scale, 2.0 * scale, 0.0),
            (0.0, 1.0 * scale, 0.0),
        )

    def test_nearly_dependent_rows_are_still_independent_at_small_scale(self):
        # Relative to the largest entry these differ at the 1e-3 level: A tolerance of 1e-13 would call them dependent.
        rows = ((1e-9, 1e-9, 0.0), (1e-9, 1.001e-9, 0.0))
        assert unit_vector_completion(rows, (2,)) == rows


class TestHasFullRank:
    @pytest.mark.parametrize("scale", [1.0, 1e-9, 1e-15])
    def test_is_independent_of_the_scale(self, scale):
        assert has_full_rank(tuple(tuple(scale * v for v in row) for row in ((1.0, 2.0), (3.0, 5.0))))
        assert not has_full_rank(tuple(tuple(scale * v for v in row) for row in ((1.0, 2.0), (2.0, 4.0))))

    def test_zero_matrix_has_no_rank(self):
        assert not has_full_rank(((0.0, 0.0), (0.0, 0.0)))
