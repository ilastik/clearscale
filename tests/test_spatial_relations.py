import pytest

from clearscale import Affine, Factor, ProjectionTo, AxisRenaming, AxisRearrangementTo, SameSpace, Translation
from clearscale._affines import Linear
from clearscale._services.axis_links import AxisLinks
from clearscale._spatial_relations import SpatialRelationSequence, PermutationTo, normalize_relations_param


class SomeAxisKey:  # Axis keys can be any object that can convert to str
    def __repr__(self):
        return "x"


@pytest.mark.parametrize("axes", ["x", "yx", "tczyx", ("a", "b"), (SomeAxisKey(),)])
def test_same_space_keeps_any_axes(axes):
    assert SameSpace().target_axes(axes) == tuple(axes)
    assert SameSpace()._axis_links(axes) == AxisLinks.identity(axes)


def test_same_space_instances_are_equal_and_hashable():
    assert SameSpace() == SameSpace()
    assert len({SameSpace(), SameSpace()}) == 1
    assert SameSpace() != Translation(x=0.0)


def test_normalize_relations_param_rejects_none():
    with pytest.raises(TypeError, match=r"Expected a SpatialRelation.*received None.*Use `SameSpace\(\)`"):
        normalize_relations_param(None)  # type: ignore[arg-type]


@pytest.mark.parametrize("empty", [[], ()])
def test_normalize_relations_param_turns_an_empty_sequence_into_same_space(empty):
    assert normalize_relations_param(empty) == SameSpace()


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


def test_axis_renaming_rejects_invalid_construction():
    with pytest.raises(ValueError, match="at least one"):
        AxisRenaming()
    with pytest.raises(ValueError, match="same name"):
        AxisRenaming({"z": "a", "y": "a"})


def test_axis_renaming_rejects_incompatible_source_axes():
    with pytest.raises(ValueError, match="not present"):
        AxisRenaming(c="C").target_axes("zyx")
    with pytest.raises(ValueError, match="duplicate"):
        AxisRenaming(z="y").target_axes("zyx")


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


def _exact(links):
    return [link.exact for link in links.links]


class TestAxisLinks:
    @pytest.mark.parametrize(
        "relation, source, expected_sources, expected_exact",
        [
            (PermutationTo("xyz"), "zyx", (("x",), ("y",), ("z",)), [("x", 1), ("y", 1), ("z", 1)]),
            (ProjectionTo("cyx"), "zyx", ((), ("y",), ("x",)), [None, ("y", 1), ("x", 1)]),
            (AxisRenaming(z="Z"), "zyx", (("z",), ("y",), ("x",)), [("z", 1), ("y", 1), ("x", 1)]),
            (AxisRenaming(y="x", x="y"), "yx", (("y",), ("x",)), [("y", 1), ("x", 1)]),
            (AxisRearrangementTo("xcy"), "zyx", (("x",), (), ("y",)), [("x", 1), None, ("y", 1)]),
            (Affine.from_factor(Factor(y=0.5, x=0.5)), "yx", (("y",), ("x",)), [("y", 0.5), ("x", 0.5)]),
            (Translation(y=1.0, x=1.0), "yx", (("y",), ("x",)), [("y", 1), ("x", 1)]),
        ],
    )
    def test_single_relations(self, relation, source, expected_sources, expected_exact):
        links = relation._axis_links(source)

        assert links.source_axes == tuple(source)
        assert links.target_axes == relation.target_axes(source)
        assert links.sources == expected_sources
        assert _exact(links) == expected_exact

    @pytest.mark.parametrize(
        "relation, source",
        [
            (PermutationTo("xy"), "zyx"),
            (Affine.from_factor(Factor(y=0.5)), "yx"),
            (AxisRenaming(c="C"), "zyx"),
        ],
    )
    def test_reject_incompatible_source_axes(self, relation, source):
        with pytest.raises(ValueError):
            relation._axis_links(source)

    @pytest.mark.parametrize(
        "rows, sources, exact",
        [
            (((2, 0, 0), (0, 3, 0)), (("y",), ("x",)), [("y", 2), ("x", 3)]),  # diagonal
            (
                ((0, 1, 0), (1, 0, 0)),
                (("x",), ("y",)),
                [("x", 1), ("y", 1)],
            ),  # axes swapped: new y <- old x, new x <- old y
            (((-1, 0, 0), (0, 1, 0)), (("y",), ("x",)), [("y", -1), ("x", 1)]),  # flip
            (((1, 1, 0), (0, 1, 0)), (("y", "x"), ("x",)), [None, ("x", 1)]),  # x is mixed into y, but x stays exact
            (((1, 0, 0), (-1, 1, 0)), (("y",), ("y", "x")), [("y", 1), None]),  # inverse shear
            (((1, 1, 0), (-1, 1, 0)), (("y", "x"), ("y", "x")), [None, None]),  # rotation
            (((0, 0, 0), (0, 1, 0)), ((), ("x",)), [None, ("x", 1)]),  # y collapsed
            (((1, 1e-14, 0), (0, 4, 0)), (("y",), ("x",)), [("y", 1), ("x", 4)]),  # negligible coefficient
            (((1, 1e-9, 0), (0, 1, 0)), (("y", "x"), ("x",)), [None, ("x", 1)]),  # small but non-negligible
            (((1, 2, 5), (0, 1, 7)), (("y", "x"), ("x",)), [None, ("x", 1)]),  # translation does not matter
        ],
    )
    def test_affine_rows(self, rows, sources, exact):
        links = Affine.from_array(rows, "yx")._axis_links("yx")

        assert links.source_axes == links.target_axes == ("y", "x")
        assert links.sources == sources
        assert _exact(links) == exact

    def test_affine_links_use_the_forward_matrix(self):
        # The forward upper bidiagonal pattern differs from the full upper triangle of its inverse.
        affine = Affine.from_array(((1, 1, 0, 0), (0, 1, 1, 0), (0, 0, 1, 0)), "zyx")
        links = affine._axis_links("zyx")
        assert links.sources == (("z", "y"), ("y", "x"), ("x",))
        assert _exact(links) == [None, None, ("x", 1)]

    def test_sequence_follows_exact_scales_through_renamings_and_permutations(self):
        sequence = SpatialRelationSequence(
            (
                AxisRenaming(z="Z"),
                PermutationTo("yxZ"),
                AxisRenaming(Z="depth"),
                Affine.from_factor(Factor(y=2.0, x=3.0, depth=4.0)),
            )
        )

        links = sequence._axis_links("zyx")

        assert links.target_axes == ("y", "x", "depth")
        assert links.sources == (("y",), ("x",), ("z",))
        assert _exact(links) == [("y", 2.0), ("x", 3.0), ("z", 4.0)]

    def test_sequence_multiplies_scales_and_cancels_inverse_ones(self):
        sequence = SpatialRelationSequence(
            (Affine.from_factor(Factor(y=1000.0, x=4.0)), Affine.from_array(((0.001, 0, 0), (0, -0.5, 0)), "yx"))
        )

        assert _exact(sequence._axis_links("yx")) == [("y", pytest.approx(1.0)), ("x", -2.0)]

    def test_sequence_is_not_exact_across_a_mixing_step(self):
        shear = Affine.from_array(((1, 1, 0), (0, 1, 0)), "yx")
        sequence = SpatialRelationSequence((shear, Affine.from_factor(Factor(y=2.0, x=2.0))))

        links = sequence._axis_links("yx")

        assert links.sources == (("y", "x"), ("x",))
        assert _exact(links) == [None, ("x", 2.0)]

    def test_sequence_is_not_exact_across_drop_and_reinsert(self):
        # Z is dropped, and a *new* z is inserted. Net rename z->Z followed by Z->z would claim the new z descends from z.
        sequence = SpatialRelationSequence((AxisRenaming(z="Z"), ProjectionTo("zyx")))

        links = sequence._axis_links("zyx")
        assert links.sources == ((), ("y",), ("x",))
        assert _exact(links) == [None, ("y", 1.0), ("x", 1.0)]

    def test_axis_mixed_with_an_inserted_axis_has_untraced_coupling(self):
        mixing = Affine.from_array(((1, 0, 0, 0), (1, 1, 0, 0), (0, 0, 1, 0)), "zyx")  # y' = z + y
        sequence = SpatialRelationSequence((ProjectionTo("zyx"), mixing))

        links = sequence._axis_links("yx")

        assert links.sources == ((), ("y",), ("x",))
        assert [link.has_untraced_coupling for link in links.links] == [False, True, False]


class TestBidirectionalAxisLinks:
    """Main coverage in test_axis_links (TestReversed, TestComposedBefore)"""

    @pytest.mark.parametrize(
        "relation, source_axes",
        [
            (PermutationTo("xyz"), "zyx"),
            (ProjectionTo("cyx"), "zyx"),
            (AxisRenaming(z="Z"), "zyx"),
            (AxisRearrangementTo("xcy"), "zyx"),
            (AxisRenaming(y="x", x="y"), "yx"),
            (Translation(y=1.0, x=1.0), "yx"),
            (Affine.from_factor(Factor(y=0.5, x=0.5)), "yx"),
            (Affine.from_array(((1, 0, 0), (1, 1, 0)), "yx"), "yx"),
        ],
    )
    def test_non_sequence_bidirectional_is_reversed_links(self, relation, source_axes):
        assert relation._bidirectional_axis_links(source_axes) == relation._axis_links(source_axes).reversed()

    def test_reject_incompatible_source_axes(self):
        with pytest.raises(ValueError):
            PermutationTo("xy")._bidirectional_axis_links("zyx")

    def test_affine_couples_mixed_axes_but_keeps_the_exact_partner(self):
        # y' = y, x' = y + x: y is exactly y', but x is mixed into x' with y, which also feeds y'
        links = Affine.from_array(((1, 0, 0), (1, 1, 0)), "yx")._bidirectional_axis_links("yx")

        assert links.sources == (("y", "x"), ("y", "x"))
        assert _exact(links) == [("y", 1), None]

    def test_singular_affine_is_supported(self):
        links = Affine.from_array(((0, 0, 0), (0, 2, 0)), "yx")._bidirectional_axis_links("yx")

        assert links.sources == ((), ("x",))
        assert _exact(links) == [None, ("x", 0.5)]

    def test_sequence_chains_the_steps_against_the_relation(self):
        sequence = SpatialRelationSequence((AxisRenaming(y="Y"), Affine.from_factor(Factor(Y=2.0, x=3.0))))

        links = sequence._bidirectional_axis_links("yx")

        assert links.source_axes == ("Y", "x")
        assert links.target_axes == ("y", "x")
        assert links.sources == (("Y",), ("x",))
        assert _exact(links) == [("Y", 0.5), ("x", pytest.approx(1 / 3))]

    def test_sequence_traces_coupling_even_after_drop(self):
        # b' = a + b couples a and b. Dropping b' afterwards does not undo that a and b could not be told apart.
        mixing = Affine.from_array(((1, 0, 0, 0), (1, 1, 0, 0), (0, 0, 1, 0)), "abc")
        sequence = SpatialRelationSequence((mixing, ProjectionTo("ac")))

        links = sequence._bidirectional_axis_links("abc")

        assert sequence._axis_links("abc").reversed().sources == (("a",), (), ("c",))
        assert links.sources == (("a",), ("a",), ("c",))
        assert [link.has_untraced_coupling for link in links.links] == [
            True,
            True,
            False,
        ]  # b' is lost: agreement means nothing
        assert [link.exact for link in links.links] == [("a", 1.0), None, ("c", 1.0)]  # but a is still exactly a
