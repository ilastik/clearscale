import pytest

from clearscale import ProjectionTo
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
