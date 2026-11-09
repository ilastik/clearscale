import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from clearscale.types import AxisKey, OrderedAxes

AxisKeyDependencies = Tuple[Tuple[AxisKey, ...], ...]
"""AxisKey counterpart to `Transform._axis_dependencies`, which uses axis indices.
One tuple of source axes per target axis, in target axis order.
() = target depends on nothing (is new insert)
(s,) = target is s
(a, b, ...) = target mixes source axes a, b, ...
"""


@dataclass(frozen=True, slots=True)
class AxisLink:
    """How one target axis relates to the source axes it depends on. Basically a tuple `(sources, exact)`."""

    sources: Tuple[AxisKey, ...]
    """Source axes the target relates to.
    * When computed directly from a relation (forward links): the axes that the target directly depends on.
    * After .reversed() (bidirectional links): all forward target axes of the mixing component the axis participates in.
    """
    exact: Optional[Tuple[AxisKey, float]] = None
    """
    `(source, scale)` if the target is exactly `scale` times the coordinate of that source (translation
    aside) and is not mixed with anything else: The relation has a single coefficient for it. The source may still
    contribute to other targets. Properties that scaling or mixing invalidate (e.g. unit) can only follow exact links.
    After .reversed(), it is the one axis that is exactly this one, with the scale inverted.
    """

    def __post_init__(self):
        if self.exact is not None:
            source, scale = self.exact
            assert source in self.sources, f"Exact source {source!r} not in sources {self.sources!r}"
            # Dependencies are only traced for coefficients that are not ~0, guard against overflow
            assert math.isfinite(scale) and scale != 0, f"Invalid scale {scale!r}"


@dataclass(frozen=True, slots=True)
class AxisLinks:
    """For each of `target_axes`, how it relates to `source_axes`: One AxisLink per target axis, in target order.

    Allows axis properties (type, unit, ...) to follow the axes they belong to through relations.
    Similar in spirit to `_AxisLineageTrace`, which traces axis identity through Transforms by index, but only through
    one-to-one steps and refuses mixing.
    This class works on SpatialRelations by axis key, and following properties through mixing is the reason it exists.
    Build from a relation's dependencies-tuple with `from_dependencies`, and use `reversed` for properties
    that flow against the relation.
    """

    source_axes: Tuple[AxisKey, ...]
    target_axes: Tuple[AxisKey, ...]
    links: Tuple[AxisLink, ...]
    """One AxisLink per target_axes"""

    def __post_init__(self):
        assert len(self.links) == len(self.target_axes), f"Expected one link per target axis {self.target_axes!r}"
        known = set(self.source_axes)
        for link in self.links:
            assert known.issuperset(link.sources), f"Sources {link.sources!r} not in source axes {self.source_axes!r}"

    @classmethod
    def identity(cls, axes: OrderedAxes) -> "AxisLinks":
        """Every axis is exactly itself."""
        axes = tuple(axes)
        return cls(axes, axes, tuple(AxisLink((axis,), (axis, 1.0)) for axis in axes))

    @classmethod
    def from_dependencies(
        cls,
        source_axes: OrderedAxes,
        target_axes: OrderedAxes,
        dependencies: AxisKeyDependencies,
        scales: Optional[Sequence[Optional[float]]] = None,
    ) -> "AxisLinks":
        """Forward links from the direct dependencies of a relation (one tuple of source axes per target axis).

        A target with a single source is exact. `scales[i]` is the coefficient of target i, only used if it has a
        single source. `scales=None` is for relations that never scale, so exact targets have scale 1.0."""
        links: List[AxisLink] = []
        for index, sources in enumerate(dependencies):
            scale = 1.0 if scales is None else scales[index]
            exact = (sources[0], scale) if len(sources) == 1 and scale is not None else None
            links.append(AxisLink(sources, exact))
        return cls(tuple(source_axes), tuple(target_axes), tuple(links))

    @property
    def sources(self) -> AxisKeyDependencies:
        """The `sources` of each link, in target order. For forward links, the direct dependencies of a relation."""
        return tuple(link.sources for link in self.links)

    def link_to(self, target_axis: AxisKey) -> AxisLink:
        try:
            return self.links[self.target_axes.index(target_axis)]
        except ValueError:
            raise KeyError(f"{target_axis!r} is not one of the target axes {self.target_axes!r}.") from None

    def composed_before(self, later: "AxisLinks") -> "AxisLinks":
        """Links of `self` followed by `later`, i.e. from `self.source_axes` to `later.target_axes`.

        Traced step by step, so the sources are an upper bound: Mixing that cancels out across steps is not detected.
        A composed link is exact if it is in both steps, with the scales multiplied."""
        if self.target_axes != later.source_axes:
            raise ValueError(
                f"Cannot chain links: Target axes {self.target_axes!r} are not the source axes {later.source_axes!r}."
            )
        first_by_middle: Dict[AxisKey, AxisLink] = dict(zip(self.target_axes, self.links))
        links: List[AxisLink] = []
        for link in later.links:
            reached = {source for middle in link.sources for source in first_by_middle[middle].sources}
            sources = tuple(sorted(reached, key=self.source_axes.index))
            exact = None
            if link.exact is not None:
                middle, second_scale = link.exact
                first_exact = first_by_middle[middle].exact
                if first_exact is not None:
                    source, first_scale = first_exact
                    exact = (source, first_scale * second_scale)
            links.append(AxisLink(sources, exact))
        return AxisLinks(self.source_axes, later.target_axes, tuple(links))

    def reversed(self) -> "AxisLinks":
        """Links for properties flowing against the relation, from forward links: For each of the source axes, which
        target axes it relates to and may inherit properties from.

        `AxisLinks.from_dependencies("yx", "ba",
          (("y", "x"), ("x"))  # b = y + x ; a = x
        ).reversed() == AxisLinks(
            source_axes=('b', 'a'),
            target_axes=('y', 'x'),
            links=(
                # y = b - x = b - a
                AxisLink(sources=('b', 'a'), exact=None),
                # x has been mixed into both b and a, but it can be obtained exactly from a
                AxisLink(sources=('b', 'a'), exact=('a', 1.0))
            )
        )`
        For axis property carryover and correspondence validation, this is more conservative than transposing forward
        dependencies or taking dependencies of the inverse (and it can be determined for non-invertible relations).
        The distinction is only relevant for mixing Affines. For one-to-one relations, reversed sources are the
        transpose of forward sources (this is always accurate).

        A reversed `exact` is the target that is exactly this source, with the scale inverted, even if the same source
        also contributes to other targets. (But only if there is only one such target.)"""
        components: List[Set[AxisKey]] = []
        for link in self.links:
            if not link.sources:
                continue
            overlapping = [component for component in components if component & set(link.sources)]
            merged = set(link.sources)
            for component in overlapping:
                merged |= component
                components.remove(component)
            components.append(merged)

        links: List[AxisLink] = []
        for source in self.source_axes:
            component = next((component for component in components if source in component), set())
            sources = tuple(
                target
                for target, link in zip(self.target_axes, self.links)
                if link.sources and link.sources[0] in component
            )
            partners = [
                (target, link.exact[1])
                for target, link in zip(self.target_axes, self.links)
                if link.exact is not None and link.exact[0] == source
            ]
            exact = (partners[0][0], 1 / partners[0][1]) if len(partners) == 1 else None
            links.append(AxisLink(sources, exact))
        return AxisLinks(self.target_axes, self.source_axes, tuple(links))
