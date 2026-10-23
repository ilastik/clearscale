import pytest

from clearscale import Affine, Factor, ProjectionTo, AxisRearrangementTo, AxisRenaming, Translation
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


def test_axis_renaming_target_axes():
    assert AxisRenaming(z="Z").target_axes("zyx") == ("Z", "y", "x")
    assert AxisRenaming({"z": "Z", "x": "X"}).target_axes("zyx") == ("Z", "y", "X")
    assert AxisRenaming({"z": 0, "y": 1}).target_axes("zyx") == (0, 1, "x"), "any hashable key"


def test_axis_renaming_applies_simultaneously():
    assert AxisRenaming(y="x", x="y").target_axes("zyx") == ("z", "x", "y")
    assert AxisRenaming(z="y", y="z").target_axes("zyx") == ("y", "z", "x")
    assert AxisRenaming(z="y").target_axes("zx") == ("y", "x")


def test_axis_renaming_equality_ignores_order_and_is_hashable():
    assert AxisRenaming(z="Z", y="Y") == AxisRenaming({"y": "Y", "z": "Z"})
    assert hash(AxisRenaming(z="Z", y="Y")) == hash(AxisRenaming(y="Y", z="Z"))
    assert AxisRenaming(z="Z") != AxisRenaming(z="Y")
    assert AxisRenaming(z="Z") != PermutationTo("Z")
    assert AxisRenaming({"z": "Z"}).mapping == {"z": "Z"}
    assert repr(AxisRenaming(z="Z")) == "AxisRenaming({'z': 'Z'})"


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda: AxisRenaming(), "at least one"),
        (lambda: AxisRenaming({"z": "a", "y": "a"}), "same name"),
    ],
)
def test_axis_renaming_rejects_invalid_construction(build, message):
    with pytest.raises(ValueError, match=message):
        build()


def test_axis_renaming_rejects_incompatible_source_axes():
    with pytest.raises(ValueError, match="not present"):
        AxisRenaming(c="C").target_axes("zyx")
    with pytest.raises(ValueError, match="duplicate"):
        AxisRenaming(z="y").target_axes("zyx")


@pytest.mark.parametrize(
    "relation, source, expected",
    [
        (PermutationTo("xyz"), "zyx", (("x",), ("y",), ("z",))),
        (ProjectionTo("cyx"), "zyx", ((), ("y",), ("x",))),
        (AxisRearrangementTo("xcy"), "zyx", (("x",), (), ("y",))),
        (AxisRenaming(z="Z"), "zyx", (("z",), ("y",), ("x",))),
        (AxisRenaming(y="x", x="y"), "yx", (("y",), ("x",))),
        (Affine.from_factor(Factor(y=2, x=2)), "yx", (("y",), ("x",))),
        (Translation(y=1.0, x=1.0), "yx", (("y",), ("x",))),
    ],
)
def test_axis_dependencies_of_single_relations(relation, source, expected):
    assert relation._axis_dependencies(source) == expected


def test_axis_dependencies_reject_incompatible_source_axes():
    for relation, source in (
        (PermutationTo("xy"), "zyx"),
        (Affine.from_factor(Factor(y=2)), "yx"),
        (AxisRenaming(c="C"), "zyx"),
    ):
        with pytest.raises(ValueError):
            relation._axis_dependencies(source)


def test_axis_dependencies_of_sequence_follow_axes_through_renamings():
    sequence = SpatialRelationSequence((AxisRenaming(z="Z"), PermutationTo("yxZ"), AxisRenaming(Z="depth")))
    assert sequence._axis_dependencies("zyx") == (("y",), ("x",), ("z",))


def test_axis_dependencies_of_sequence_do_not_carry_across_drop_and_reinsert():
    # Z is dropped, and a *new* z is inserted. Net rename z->Z followed by Z->z would claim the new z descends from z.
    sequence = SpatialRelationSequence((AxisRenaming(z="Z"), ProjectionTo("zyx")))
    assert sequence._axis_dependencies("zyx") == ((), ("y",), ("x",))


@pytest.mark.parametrize(
    "rows, expected",
    [
        (((2, 0, 0), (0, 3, 0)), (("y",), ("x",))),  # diagonal
        (((0, 1, 0), (1, 0, 0)), (("x",), ("y",))),  # axes swapped by values: new y <- old x, new x <- old y
        (((1, 0, 0), (-1, 1, 0)), (("y",), ("y", "x"))),
        (((1, 2, 5), (0, 1, 7)), (("y", "x"), ("x",))),  # translation does not matter
    ],
)
def test_affine_axis_dependencies_follow_the_matrix_pattern(rows, expected):
    assert Affine.from_array(rows, "yx")._axis_dependencies("yx") == expected


def test_affine_axis_dependencies_are_those_of_the_inverse_not_of_the_transposed_matrix():
    # x_old = L x_new with L upper bidiagonal. L's inverse is a full upper triangle, so the new z depends on all.
    affine = Affine.from_array(((1, 1, 0, 0), (0, 1, 1, 0), (0, 0, 1, 0)), "zyx")
    assert affine._axis_dependencies("zyx") == (("z", "y", "x"), ("y", "x"), ("x",))


def test_affine_axis_dependencies_require_an_invertible_matrix_and_matching_axes():
    with pytest.raises(ValueError, match="must be invertible"):
        Affine.from_array(((1, 2, 0), (2, 4, 0)), "yx")._axis_dependencies("yx")
    with pytest.raises(ValueError, match="Incompatible axes"):
        Affine.identity("yx")._axis_dependencies("xy")
