import pytest

from clearscale._services.axis_links import AxisLink, AxisLinks


def _links_for_linear(rows, source_axes="abc", target_axes=None):
    """Forward links for a relation given as `{target: {source: coefficient}}`, like the rows of a Linear."""
    target_axes = tuple(rows) if target_axes is None else tuple(target_axes)
    dependencies = tuple(tuple(source for source in source_axes if source in rows[t]) for t in target_axes)
    scales = [next(iter(rows[t].values())) if len(rows[t]) == 1 else None for t in target_axes]
    return AxisLinks.from_dependencies(source_axes, target_axes, dependencies, scales)


def _exact(links: AxisLinks):
    return [link.exact for link in links.links]


class TestAxisLink:
    def test_exact_source_must_be_one_of_the_sources(self):
        with pytest.raises(AssertionError, match="not in sources"):
            AxisLink(("a",), ("b", 1.0))

    @pytest.mark.parametrize("scale", [0.0, float("inf"), float("nan")])
    def test_exact_scale_must_be_finite_and_nonzero(self, scale):
        with pytest.raises(AssertionError, match="Invalid scale"):
            AxisLink(("a",), ("a", scale))


class TestAxisLinksConstruction:
    def test_requires_one_link_per_target_axis(self):
        with pytest.raises(AssertionError, match="one link per target axis"):
            AxisLinks(("a",), ("a", "b"), (AxisLink(("a",)),))

    def test_links_must_refer_to_source_axes(self):
        with pytest.raises(AssertionError, match="not in source axes"):
            AxisLinks(("a",), ("b",), (AxisLink(("c",)),))

    def test_identity(self):
        links = AxisLinks.identity("zyx")

        assert links.source_axes == links.target_axes == ("z", "y", "x")
        assert links.sources == (("z",), ("y",), ("x",))
        assert _exact(links) == [("z", 1.0), ("y", 1.0), ("x", 1.0)]

    def test_from_dependencies_without_scales_is_exact_for_single_sources(self):
        links = AxisLinks.from_dependencies("abc", "xyz", (("b",), (), ("a", "c")))

        assert links.sources == (("b",), (), ("a", "c"))
        assert _exact(links) == [("b", 1.0), None, None]

    def test_from_dependencies_uses_scales_for_single_sources_only(self):
        links = AxisLinks.from_dependencies(
            "abc", "wxyz", (("b",), (), ("a", "c"), ("c",)), scales=[4.0, 1.0, 2.0, None]
        )

        assert _exact(links) == [("b", 4.0), None, None, None]

    def test_link_to_looks_up_by_target_axis(self):
        links = AxisLinks.from_dependencies("ab", "xy", (("b",), ("a",)))

        assert links.link_to("x").sources == ("b",)
        with pytest.raises(KeyError):
            links.link_to("a")


class TestForwardLinks:
    """Forward links of relations that are practically relevant. Exact means a unit can follow."""

    def test_unit_conversion_is_exact_with_its_scale(self):
        # um -> nm
        links = _links_for_linear({"y": {"y": 1.0}, "x": {"x": 1000.0}}, source_axes="yx")

        assert _exact(links) == [("y", 1.0), ("x", 1000.0)]

    def test_flip_and_rotation_by_90_degrees_are_exact_with_negative_scales(self):
        # y' = -x, x' = y
        links = _links_for_linear({"y": {"x": -1.0}, "x": {"y": 1.0}}, source_axes="yx")

        assert links.sources == (("x",), ("y",))
        assert _exact(links) == [("x", -1.0), ("y", 1.0)]

    def test_shear_is_exact_only_for_the_row_that_is_not_mixed(self):
        # x' = x + y, y' = y: y' is exactly y even though y also feeds x'
        links = _links_for_linear({"x": {"x": 1, "y": 1}, "y": {"y": 1}}, source_axes="xy")

        assert links.sources == (("x", "y"), ("y",))
        assert _exact(links) == [None, ("y", 1)]

    def test_rotation_is_not_exact(self):
        links = _links_for_linear({"x": {"x": 1, "y": 1}, "y": {"x": -1, "y": 1}}, source_axes="xy")

        assert links.sources == (("x", "y"), ("x", "y"))
        assert _exact(links) == [None, None]

    def test_collapsed_axis_has_no_sources(self):
        links = _links_for_linear({"z": {}, "y": {"y": 1}}, source_axes="zy")

        assert links.sources == ((), ("y",))
        assert _exact(links) == [None, ("y", 1)]


class TestReversed:
    def test_one_to_one_links_are_transposed(self):
        # a -> y, b -> x (permutation), c dropped, z inserted
        links = AxisLinks.from_dependencies("abc", "yxz", (("a",), ("b",), ()))

        reverse = links.reversed()

        assert reverse.source_axes == ("y", "x", "z")
        assert reverse.target_axes == ("a", "b", "c")
        assert reverse.sources == (("y",), ("x",), ())
        assert _exact(reverse) == [("y", 1.0), ("x", 1.0), None]

    def test_exact_scales_are_inverted(self):
        links = _links_for_linear({"x": {"a": 4.0}, "y": {"b": -0.5}}, source_axes="ab")

        assert _exact(links.reversed()) == [("x", 0.25), ("y", -2.0)]

    def test_mixed_block_depends_on_the_whole_block(self):
        links = _links_for_linear({"x": {"a": 1, "b": 1}, "y": {"b": 1, "c": 1}})

        reverse = links.reversed()

        assert reverse.sources == (("x", "y"), ("x", "y"), ("x", "y"))
        assert _exact(reverse) == [None, None, None]

    def test_blocks_stay_separate(self):
        links = _links_for_linear({"x": {"a": 1, "b": 1}, "y": {"c": 3.0}})

        reverse = links.reversed()

        assert reverse.sources == (("x",), ("x",), ("y",))
        assert _exact(reverse) == [None, None, ("y", pytest.approx(1 / 3))]

    def test_shear_keeps_the_exact_partner_but_still_couples_the_block(self):
        # x' = x + y, y' = y: y is exactly y'. x is not exactly anything, and may be inherited from both.
        links = _links_for_linear({"x": {"x": 1, "y": 1}, "y": {"y": 1}}, source_axes="xy")

        reverse = links.reversed()

        assert reverse.sources == (("x", "y"), ("x", "y"))
        assert _exact(reverse) == [None, ("y", 1.0)]

    def test_transposed_shear_does_the_same_for_the_other_axis(self):
        # x' = x, y' = x + y
        links = _links_for_linear({"x": {"x": 1}, "y": {"x": 1, "y": 1}}, source_axes="xy")

        reverse = links.reversed()

        assert reverse.sources == (("x", "y"), ("x", "y"))
        assert _exact(reverse) == [("x", 1.0), None]

    def test_axis_nothing_depends_on_has_no_sources(self):
        # Singular: a is collapsed. Reversing does not require an invertible relation.
        links = _links_for_linear({"x": {}, "y": {"b": 1}}, source_axes="ab")

        reverse = links.reversed()

        assert reverse.sources == ((), ("y",))
        assert _exact(reverse) == [None, ("y", 1)]

    def test_several_exact_targets_of_the_same_source_are_no_partner(self):
        links = _links_for_linear({"x": {"a": 1}, "y": {"a": 1}}, source_axes="a")

        reverse = links.reversed()

        assert reverse.sources == (("x", "y"),)
        assert _exact(reverse) == [None]

    def test_reversing_one_to_one_links_twice_restores_them(self):
        links = _links_for_linear({"x": {"b": 2.0}, "y": {"a": -5.0}}, source_axes="ab")

        assert links.reversed().reversed() == links

    def test_reversing_mixed_links_twice_keeps_conservative_dependencies(self):
        links = AxisLinks.from_dependencies(
            "yx",
            "ba",
            (("y", "x"), ("x",)),  # b = y + x; a = x
        )

        reverse = links.reversed()
        roundtrip = reverse.reversed()

        assert links.sources == (("y", "x"), ("x",))
        assert _exact(links) == [None, ("x", 1.0)]

        assert reverse.sources == (("b", "a"), ("b", "a"))
        assert _exact(reverse) == [None, ("a", 1.0)]

        assert roundtrip.sources == (("y", "x"), ("y", "x"))
        assert _exact(roundtrip) == [None, ("x", 1.0)]
        assert roundtrip != links


class TestComposedBefore:
    def test_requires_matching_axes(self):
        with pytest.raises(ValueError, match="Cannot chain"):
            AxisLinks.identity("ab").composed_before(AxisLinks.identity("ba"))

    def test_chains_sources_and_sorts_them_by_original_order(self):
        first = AxisLinks.from_dependencies("abc", "xyz", (("c",), ("a",), ("b",)))
        second = _links_for_linear({"p": {"z": 1, "x": 1}, "q": {"y": 1}}, source_axes="xyz")

        chained = first.composed_before(second)

        assert chained.source_axes == ("a", "b", "c")
        assert chained.target_axes == ("p", "q")
        assert chained.sources == (("b", "c"), ("a",))
        assert _exact(chained) == [None, ("a", 1)]

    def test_scales_multiply(self):
        first = _links_for_linear({"x": {"a": 3.0}}, source_axes="a")
        second = _links_for_linear({"y": {"x": 4.0}}, source_axes="x")

        assert _exact(first.composed_before(second)) == [("a", 12.0)]

    def test_scale_and_its_inverse_cancel_to_identity(self):
        first = _links_for_linear({"x": {"a": 1000.0}}, source_axes="a")
        second = _links_for_linear({"b": {"x": 0.001}}, source_axes="x")

        (exact,) = _exact(first.composed_before(second))

        assert exact == ("a", pytest.approx(1.0))

    def test_flipping_twice_is_exactly_identity(self):
        flip = _links_for_linear({"x": {"x": -1.0}}, source_axes="x")

        assert _exact(flip.composed_before(flip)) == [("x", 1.0)]

    def test_exact_survives_composition_only_on_unmixed_paths(self):
        mixing = _links_for_linear({"x": {"a": 1, "b": 1}, "y": {"b": 1}}, source_axes="ab")
        exact = _links_for_linear({"p": {"x": 2.0}, "q": {"y": 2.0}}, source_axes="xy")

        mixing_first = mixing.composed_before(exact)

        assert _exact(mixing) == [None, ("b", 1.0)]
        assert _exact(exact) == [("x", 2.0), ("y", 2.0)]
        assert mixing_first.sources == (("a", "b"), ("b",))
        assert _exact(mixing_first) == [None, ("b", 2.0)], "(b, 1.0) * (y, 2.0) = (b, 2.0)"

        merging = _links_for_linear({"s": {"p": 1, "q": 1}}, source_axes="pq")

        mixed_second = exact.composed_before(merging)

        assert _exact(exact) == [("x", 2.0), ("y", 2.0)], "(see above)"
        assert _exact(merging) == [None]
        assert mixed_second.sources == (("x", "y"),)
        assert _exact(mixed_second) == [None], "merging invalidates both exact links"

    def test_inserted_and_dropped_axes(self):
        first = AxisLinks.from_dependencies("ab", "x", (("a",),))  # drops b
        second = AxisLinks.from_dependencies("x", "xn", (("x",), ()))  # inserts n

        chained = first.composed_before(second)

        assert chained.sources == (("a",), ())
        assert _exact(chained) == [("a", 1.0), None]

    def test_reverse_of_a_chain_is_the_chain_of_reverses_for_one_to_one_links(self):
        first = _links_for_linear({"x": {"b": 2.0}, "y": {"a": 5.0}}, source_axes="ab")
        second = _links_for_linear({"p": {"y": 3.0}, "q": {"x": 7.0}}, source_axes="xy")

        chained = first.composed_before(second)

        assert chained.reversed() == second.reversed().composed_before(first.reversed())


class TestUntracedCoupling:
    """Mixing with axes that cannot be traced makes agreement among the traceable sources meaningless."""

    @staticmethod
    def _untraced(links: AxisLinks):
        return [link.has_untraced_coupling for link in links.links]

    def test_links_have_no_untraced_coupling_by_default(self):
        assert not AxisLink(("a",)).has_untraced_coupling
        assert self._untraced(AxisLinks.identity("ab")) == [False, False]
        assert self._untraced(_links_for_linear({"x": {"a": 1, "b": 1}, "y": {"c": 1}})) == [False, False]

    def test_mixing_with_an_inserted_axis_is_untraced(self):
        # z is inserted, then y' = y + z
        insert = AxisLinks.from_dependencies("yx", "zyx", ((), ("y",), ("x",)))
        mix = _links_for_linear({"z": {"z": 1}, "y": {"z": 1, "y": 1}, "x": {"x": 1}}, source_axes="zyx")

        chained = insert.composed_before(mix)

        assert chained.sources == ((), ("y",), ("x",)), "dependencies are in original source axes: z does not exist"
        assert self._untraced(chained) == [
            False,
            True,
            False,
        ], "flag indicates y mixes axes the sequence as a whole doesn't know"
        assert _exact(chained) == [None, None, ("x", 1.0)]

    def test_an_inserted_axis_on_its_own_is_not_untraced(self):
        insert = AxisLinks.from_dependencies("yx", "zyx", ((), ("y",), ("x",)))
        rename = AxisLinks.from_dependencies("zyx", "zyx", (("z",), ("y",), ("x",)))

        assert self._untraced(insert.composed_before(rename)) == [False, False, False]

    def test_mixing_two_traceable_axes_is_not_untraced(self):
        first = AxisLinks.from_dependencies("ab", "ab", (("a",), ("b",)))
        mix = _links_for_linear({"x": {"a": 1, "b": 1}, "y": {"b": 1}}, source_axes="ab")

        assert self._untraced(first.composed_before(mix)) == [False, False]

    def test_untraced_coupling_carries_through_later_steps(self):
        insert = AxisLinks.from_dependencies("yx", "zyx", ((), ("y",), ("x",)))
        mix = _links_for_linear({"z": {"z": 1}, "y": {"z": 1, "y": 1}, "x": {"x": 1}}, source_axes="zyx")
        rename = AxisLinks.from_dependencies("zyx", "zYx", (("z",), ("y",), ("x",)))

        assert self._untraced(insert.composed_before(mix).composed_before(rename)) == [False, True, False]

    def test_reversed_links_have_untraced_coupling_if_their_component_does(self):
        # y' = y + z with z inserted before: y feeds y', which has untraced coupling
        links = AxisLinks(
            ("y", "x"),
            ("z", "y", "x"),
            (AxisLink(()), AxisLink(("y",), has_untraced_coupling=True), AxisLink(("x",), ("x", 1.0))),
        )

        assert self._untraced(links.reversed()) == [True, False]

    def test_dropping_a_mixed_target_afterwards_makes_its_sources_untraced(self):
        # y' = z + y, then y' is dropped. Reversed, y is only linked to z'': What y' would have said is lost.
        mix = _links_for_linear({"z": {"z": 1}, "y": {"z": 1, "y": 1}, "x": {"x": 1}}, source_axes="zyx")
        drop = AxisLinks.from_dependencies("zyx", "zx", (("z",), ("x",)))

        reverse = drop.reversed().composed_before(mix.reversed())

        assert reverse.sources == (("z",), ("z",), ("x",))
        assert _exact(reverse) == [("z", 1.0), None, ("x", 1.0)]
        assert self._untraced(reverse) == [True, True, False]
