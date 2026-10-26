import pytest

from clearscale import Affine, Factor, ProjectionTo, AxisRearrangementTo, Translation
from clearscale._affines import Linear
from clearscale._spatial_relations import SpatialRelationSequence, PermutationTo


@pytest.mark.parametrize(
    ("source", "target", "dropped", "inserted"),
    [
        # Identity
        ("zyx", "zyx", (), ()),
        # Drop
        ("tczyx", "zyx", ("t", "c"), ()),
        ("tczyx", "tyx", ("c", "z"), ()),
        # Insert
        ("zyx", "tczyx", (), ("t", "c")),
        ("tyx", "tcyx", (), ("c",)),
        # Drop and insert
        ("tzyx", "zcyx", ("t",), ("c",)),
        # No retained axes
        ("zyx", "tc", ("z", "y", "x"), ("t", "c")),
    ],
)
def test_projection_to(source, target, dropped, inserted):
    projection = ProjectionTo(target)

    assert projection.target_axes(source) == tuple(target)
    assert projection.dropped_axes(source) == dropped
    assert projection.inserted_axes(source) == inserted


@pytest.mark.parametrize(
    "target, expected_error",
    [
        ("", "requires at least one axis"),
        ("zz", "target_axes must be unique"),
        ("zyxz", "target_axes must be unique"),
    ],
)
def test_projection_to_rejects_invalid_target_axes(target, expected_error):
    with pytest.raises(ValueError, match=expected_error):
        ProjectionTo(target)


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("zyx", "xyz"),  # pure reorder
        ("tczyx", "zcyx"),  # reorder among retained axes
        ("zyx", "txzy"),  # insert + reorder
        ("tczyx", "xy"),  # drop + reorder
    ],
)
@pytest.mark.parametrize(
    "method",
    [
        ProjectionTo.target_axes,
        ProjectionTo.dropped_axes,
        ProjectionTo.inserted_axes,
    ],
)
def test_projection_to_public_methods_reject_reordering(source, target, method):
    projection = ProjectionTo(target)

    with pytest.raises(ValueError, match="cannot reorder retained axes"):
        method(projection, source)


def test_relation_sequence_target_axes_across_hops():
    relations = SpatialRelationSequence((ProjectionTo(("t", "z", "y", "x")), PermutationTo(("t", "y", "x", "z"))))
    assert relations.target_axes("zyx") == ("t", "y", "x", "z")


@pytest.mark.parametrize("source", ["yx", ("y", "x"), ["y", "x"]])
def test_affine_target_axes_are_own_axes(source):
    affine = Affine(linear=Linear.from_array(((2, 0), (0, 2)), "yx"))

    assert affine.target_axes(source) == ("y", "x")


@pytest.mark.parametrize("source", ["xy", "zyx", "y"])
def test_affine_target_axes_requires_identical_source_axes(source):
    with pytest.raises(ValueError, match="Incompatible axes/order"):
        Affine.identity("yx").target_axes(source)


def test_affine_in_relation_sequence_target_axes():
    relations = SpatialRelationSequence((Affine.identity("zyx"), PermutationTo(("x", "y", "z"))))

    assert relations.target_axes("zyx") == ("x", "y", "z")


class TestAxisDependencies:
    @pytest.mark.parametrize(
        "relation, source, expected",
        [
            (PermutationTo("xyz"), "zyx", (("x",), ("y",), ("z",))),
            (ProjectionTo("cyx"), "zyx", ((), ("y",), ("x",))),
            (AxisRearrangementTo("xcy"), "zyx", (("x",), (), ("y",))),
            (Affine.from_factor(Factor(y=2, x=2)), "yx", (("y",), ("x",))),
            (Translation(y=1.0, x=1.0), "yx", (("y",), ("x",))),
        ],
    )
    def test_single_relations(self, relation, source, expected):
        assert relation._axis_dependencies(source) == expected

    @pytest.mark.parametrize(
        "relation, source",
        [
            (PermutationTo("xy"), "zyx"),
            (Affine.from_factor(Factor(y=2)), "yx"),
        ],
    )
    def test_reject_incompatible_source_axes(self, relation, source):
        with pytest.raises(ValueError):
            relation._axis_dependencies(source)

    @pytest.mark.parametrize(
        "rows, expected",
        [
            (((2, 0, 0), (0, 3, 0)), (("y",), ("x",))),  # diagonal
            (((0, 1, 0), (1, 0, 0)), (("x",), ("y",))),  # axes swapped by values: new y <- old x, new x <- old y
            (((1, 0, 0), (-1, 1, 0)), (("y",), ("y", "x"))),
            (((1, 2, 5), (0, 1, 7)), (("y", "x"), ("x",))),  # translation does not matter
        ],
    )
    def test_affine_follow_the_matrix_pattern(self, rows, expected):
        assert Affine.from_array(rows, "yx")._axis_dependencies("yx") == expected

    def test_affine_are_those_of_the_inverse_not_of_the_transposed_matrix(self):
        # x_old = L x_new with L upper bidiagonal. L's inverse is a full upper triangle, so the new z depends on all.
        affine = Affine.from_array(((1, 1, 0, 0), (0, 1, 1, 0), (0, 0, 1, 0)), "zyx")
        assert affine._axis_dependencies("zyx") == (("z", "y", "x"), ("y", "x"), ("x",))

    def test_affine_require_an_invertible_matrix_and_matching_axes(self):
        with pytest.raises(ValueError, match="must be invertible"):
            Affine.from_array(((1, 2, 0), (2, 4, 0)), "yx")._axis_dependencies("yx")
        with pytest.raises(ValueError, match="Incompatible axes"):
            Affine.identity("yx")._axis_dependencies("xy")
