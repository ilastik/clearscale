from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Tuple, Iterator, Mapping, Sequence, overload, Union, Optional, Dict

from clearscale._services.axis_links import AxisLinks
from clearscale.types import AxisKey, OrderedAxes


class SpatialRelation(ABC):
    """A mapping of coordinates from a source space to a target space."""

    @abstractmethod
    def target_axes(self, source_axes: OrderedAxes) -> Tuple[AxisKey, ...]:
        """Raise if self is not compatible with `source_axes`, and return the axes this relation results in when applied to source_axes."""

    @abstractmethod
    def _axis_links(self, source_axes: OrderedAxes) -> AxisLinks:
        """Raise if self is not compatible with `source_axes`, and return for each `self.target_axes` the links to
        the `source_axes` it depends on / descends from.

        Allows axis properties (type, unit, ...) to follow the axes they belong to through renamings, rearrangements
        and scalings.
        """

    def _bidirectional_axis_links(self, source_axes: OrderedAxes) -> AxisLinks:
        """Raise if self is not compatible with `source_axes`, and return for each of them the links to the
        `self.target_axes` it may inherit from, i.e. for axis properties flowing against the relation.
        Properties should only be transferred if all linked target axes agree on the property's value.

        The links are bidirectional: Each target may mix several sources, and each source may be mixed into several
        targets, so a source is linked to every target in its connected component of the relation's links.
        See `AxisLinks.reversed`.
        """
        return self._axis_links(source_axes).reversed()


@dataclass(frozen=True, slots=True)
class SameSpace(SpatialRelation):
    """The identity relation: The target is the same space as the source.

    Every coordinate maps to itself, and the axes stay as they are.
    Use it where a relation is expected and the two spaces only differ by name, e.g.
    `ms.with_coordinate_system("same", reached_by=SameSpace())`.
    It works with any source axes. All `SameSpace` instances are equal.
    """

    def target_axes(self, source_axes: OrderedAxes) -> Tuple[AxisKey, ...]:
        return tuple(source_axes)

    def _axis_links(self, source_axes: OrderedAxes) -> AxisLinks:
        return AxisLinks.identity(source_axes)


@dataclass(frozen=True, slots=True)
class PermutationTo(SpatialRelation):
    _order: Tuple[AxisKey, ...]

    def __init__(self, target_axes: OrderedAxes):
        target_axes = tuple(target_axes)
        if not target_axes:
            raise ValueError("PermutationTo requires at least one axis.")
        if len(set(target_axes)) != len(target_axes):
            raise ValueError(f"target_axes must be unique. Received: {target_axes!r}")
        object.__setattr__(self, "_order", target_axes)

    def target_axes(self, source_axes: OrderedAxes) -> Tuple[AxisKey, ...]:
        source_axes = tuple(source_axes)
        if set(source_axes) != set(self._order):
            raise ValueError(
                f"PermutationTo cannot insert or drop axes, only reorder. "
                f"Source axes {source_axes!r} are not the same set as target axes {self._order!r}."
            )
        return self._order

    def _axis_links(self, source_axes: OrderedAxes) -> AxisLinks:
        target_axes = self.target_axes(source_axes)
        return AxisLinks.from_dependencies(source_axes, target_axes, tuple((axis,) for axis in target_axes))


@dataclass(frozen=True, slots=True)
class ProjectionTo(SpatialRelation):
    _targets: Tuple[AxisKey, ...]

    def __init__(self, target_axes: OrderedAxes):
        target_axes = tuple(target_axes)
        if not target_axes:
            raise ValueError("ProjectionTo requires at least one axis.")
        if len(set(target_axes)) != len(target_axes):
            raise ValueError(f"target_axes must be unique. Received: {target_axes!r}")
        object.__setattr__(self, "_targets", target_axes)

    def target_axes(self, source_axes: OrderedAxes) -> Tuple[AxisKey, ...]:
        self._require_retained_axes_not_reordered(source_axes)
        return self._targets

    def dropped_axes(self, source_axes: OrderedAxes) -> Tuple[AxisKey, ...]:
        source_axes = tuple(source_axes)
        self._require_retained_axes_not_reordered(source_axes)
        return tuple(a for a in source_axes if a not in self._targets)

    def inserted_axes(self, source_axes: OrderedAxes) -> Tuple[AxisKey, ...]:
        source_axes = tuple(source_axes)
        self._require_retained_axes_not_reordered(source_axes)
        return tuple(a for a in self._targets if a not in source_axes)

    def _axis_links(self, source_axes: OrderedAxes) -> AxisLinks:
        source_axes = tuple(source_axes)
        target_axes = self.target_axes(source_axes)
        dependencies = tuple((axis,) if axis in source_axes else () for axis in target_axes)
        return AxisLinks.from_dependencies(source_axes, target_axes, dependencies)

    def _require_retained_axes_not_reordered(self, source_axes: OrderedAxes) -> None:
        shared_source = tuple(a for a in source_axes if a in self._targets)
        shared_target = tuple(a for a in self._targets if a in source_axes)
        if shared_source != shared_target:
            raise ValueError(
                f"Projection cannot reorder retained axes ({shared_source} -> {shared_target}). "
                f"Source axes: {source_axes!r}; target axes: {self._targets!r}."
            )


@dataclass(frozen=True, slots=True)
class AxisRearrangementTo(SpatialRelation):
    """Rearrange axes to a target ordering, inserting or dropping axes as needed."""

    _targets: Tuple[AxisKey, ...]

    def __init__(self, target_axes: OrderedAxes):
        target_axes = tuple(target_axes)
        if not target_axes:
            raise ValueError("RearrangeAxesTo requires at least one axis.")
        if len(set(target_axes)) != len(target_axes):
            raise ValueError(f"target_axes must be unique. Received: {target_axes!r}")
        object.__setattr__(self, "_targets", target_axes)

    def target_axes(self, source_axes: OrderedAxes) -> Tuple[AxisKey, ...]:
        self._require_unique_source_axes(source_axes)
        return self._targets

    def dropped_axes(self, source_axes: OrderedAxes) -> Tuple[AxisKey, ...]:
        """Return dropped axes in source order."""
        source_axes = tuple(source_axes)
        self._require_unique_source_axes(source_axes)
        return tuple(axis for axis in source_axes if axis not in self._targets)

    def inserted_axes(self, source_axes: OrderedAxes) -> Tuple[AxisKey, ...]:
        """Return inserted axes in target order."""
        source_axes = tuple(source_axes)
        self._require_unique_source_axes(source_axes)
        return tuple(axis for axis in self._targets if axis not in source_axes)

    def retained_axes(self, source_axes: OrderedAxes) -> Tuple[AxisKey, ...]:
        """Return retained axes in source order."""
        source_axes = tuple(source_axes)
        self._require_unique_source_axes(source_axes)
        return tuple(axis for axis in source_axes if axis in self._targets)

    def _axis_links(self, source_axes: OrderedAxes) -> AxisLinks:
        source_axes = tuple(source_axes)
        target_axes = self.target_axes(source_axes)
        dependencies = tuple((axis,) if axis in source_axes else () for axis in target_axes)
        return AxisLinks.from_dependencies(source_axes, target_axes, dependencies)

    @staticmethod
    def _require_unique_source_axes(source_axes: OrderedAxes) -> None:
        source_axes = tuple(source_axes)
        if len(set(source_axes)) != len(source_axes):
            raise ValueError(f"source_axes must be unique. Received: {source_axes!r}")


@dataclass(frozen=True, slots=True, eq=False)
class AxisRenaming(SpatialRelation):
    """Rename axes, leaving the image unchanged: `AxisRenaming({"z": "Z"})` or `AxisRenaming(z="Z")`.

    Axes that aren't mentioned keep their names. All renames apply simultaneously, so
    `AxisRenaming(y="x", x="y")` swaps the names (it does not move any data, unlike `PermutationTo`).
    Applying it to axes that don't exist, or producing duplicate axis names, raises.
    """

    _renames: Tuple[Tuple[AxisKey, AxisKey], ...]

    def __init__(self, renames: Optional[Mapping[AxisKey, AxisKey]] = None, /, **kwargs: AxisKey):
        pairs = tuple({**(renames or {}), **kwargs}.items())
        if not pairs:
            raise ValueError("AxisRenaming requires at least one renamed axis.")
        new_names = [new for _old, new in pairs]
        if len(set(new_names)) != len(new_names):
            raise ValueError(f"Cannot rename different axes to the same name. Received: {dict(pairs)!r}")
        object.__setattr__(self, "_renames", pairs)

    @property
    def mapping(self) -> Dict[AxisKey, AxisKey]:
        """`{old_name: new_name}` (a copy)."""
        return dict(self._renames)

    def target_axes(self, source_axes: OrderedAxes) -> Tuple[AxisKey, ...]:
        source_axes = tuple(source_axes)
        mapping = self.mapping
        missing = [old for old in mapping if old not in source_axes]
        if missing:
            raise ValueError(f"Cannot rename axes {missing!r}: not present in source axes {source_axes!r}.")
        target_axes = tuple(mapping.get(axis, axis) for axis in source_axes)
        if len(set(target_axes)) != len(target_axes):
            raise ValueError(f"Renaming {mapping!r} would produce duplicate axes {target_axes!r} from {source_axes!r}.")
        return target_axes

    def _axis_links(self, source_axes: OrderedAxes) -> AxisLinks:
        source_axes = tuple(source_axes)
        target_axes = self.target_axes(source_axes)
        return AxisLinks.from_dependencies(source_axes, target_axes, tuple((axis,) for axis in source_axes))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AxisRenaming) and frozenset(self._renames) == frozenset(other._renames)

    def __hash__(self) -> int:
        return hash(frozenset(self._renames))

    def __repr__(self) -> str:
        return f"AxisRenaming({self.mapping!r})"


@dataclass(frozen=True, slots=True)
class SpatialRelationSequence(SpatialRelation, Sequence[SpatialRelation]):
    """A left-to-right chain of SpatialRelations, itself usable as a single SpatialRelation."""

    relations: Tuple[SpatialRelation, ...]

    def __post_init__(self):
        if not self.relations:
            raise ValueError("Cannot make empty SpatialRelationSequence.")
        if any(not isinstance(r, SpatialRelation) for r in self.relations):
            raise TypeError(f"All children must be SpatialRelation instances. Received: {self.relations!r}")

    def target_axes(self, source_axes: OrderedAxes) -> Tuple[AxisKey, ...]:
        axes = tuple(source_axes)
        for relation in self.relations:
            axes = relation.target_axes(axes)
        return axes

    def _axis_links(self, source_axes: OrderedAxes) -> AxisLinks:
        """Trace links step-by-step.

        As a consequence, it cannot detect when inverse dependencies across steps cancel out.
        This is a (probably mostly theoretical) chains-of-Affine edge case, see
        test_affine_chain_dependencies_are_an_upper_bound_when_the_steps_cancel_out.
        """
        links = AxisLinks.identity(source_axes)
        for relation in self.relations:
            links = links.composed_before(relation._axis_links(links.target_axes))
        return links

    def _bidirectional_axis_links(self, source_axes: OrderedAxes) -> AxisLinks:
        """Fold the bidirectional links of the steps from last to first, rather than reversing the links of the whole
        chain, which are traced forward step by step and hence cannot see coupling that a later step discards.
        """
        axes = [tuple(source_axes)]
        for relation in self.relations:
            axes.append(relation.target_axes(axes[-1]))
        links = AxisLinks.identity(axes[-1])
        for index in reversed(range(len(self.relations))):
            links = links.composed_before(self.relations[index]._bidirectional_axis_links(axes[index]))
        return links

    def __iter__(self) -> Iterator[SpatialRelation]:
        return iter(self.relations)

    def __len__(self) -> int:
        return len(self.relations)

    @overload
    def __getitem__(self, index: int) -> SpatialRelation: ...

    @overload
    def __getitem__(self, index: slice) -> Tuple[SpatialRelation, ...]: ...

    def __getitem__(self, index: int | slice) -> Union[SpatialRelation, Tuple[SpatialRelation, ...]]:
        return self.relations[index]


RelationsParam = Union[SpatialRelation, Sequence[SpatialRelation]]
"""What public APIs accept to describe a relation: One SpatialRelation, or a chain of them (applied first to last).

An empty chain is the same as `SameSpace()`.
"""


def normalize_relations_param(relations: RelationsParam) -> SpatialRelation:
    """Normalize the ergonomic `Union[SpatialRelation, Sequence[SpatialRelation]]` accepted by
    public APIs (`Multiscale.with_coordinate_system`, `Multiscale.as_derived_from`, ...) into a single
    SpatialRelation (a `SpatialRelationSequence` for more than one hop).

    An empty sequence results in `SameSpace()`.
    """
    if isinstance(relations, SpatialRelation):
        return relations
    if relations is None:
        raise TypeError(
            "Expected a SpatialRelation or a sequence of them, but received None. "
            "Use `SameSpace()` if the target space is the same as the source space."
        )
    relations = tuple(relations)
    if not relations:
        return SameSpace()
    if not all(isinstance(r, SpatialRelation) for r in relations):
        raise ValueError(
            f"Relations must be expressed using SpatialRelations like `Affine`, `Translation` or `AxisRearrangementTo`. Received {relations!r}"
        )
    if len(relations) == 1:
        return relations[0]
    return SpatialRelationSequence(relations)
