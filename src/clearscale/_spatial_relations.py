from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Tuple, Iterator, Sequence, overload, Union, Optional

from clearscale.types import AxisKey, OrderedAxes


class SpatialRelation(ABC):
    """Any object that can describe how a new image was derived from an existing one is a spatial relation."""

    @abstractmethod
    def target_axes(self, source_axes: OrderedAxes) -> Tuple[AxisKey, ...]:
        """Raise if self is not compatible with `source_axes`, and return the axes this relation results in when applied to source_axes."""


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

    @staticmethod
    def _require_unique_source_axes(source_axes: OrderedAxes) -> None:
        source_axes = tuple(source_axes)
        if len(set(source_axes)) != len(source_axes):
            raise ValueError(f"source_axes must be unique. Received: {source_axes!r}")


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


def normalize_relations_param(
    relations: Union[SpatialRelation, Sequence[SpatialRelation], None],
) -> Optional[SpatialRelation]:
    """
    Normalize the ergonomic `Union[SpatialRelation, Sequence[SpatialRelation], None]` accepted by
    public APIs (`Multiscale.derive`, `Multiscale.with_coordinate_system`, ...) into a single
    SpatialRelation (a `SpatialRelationSequence` for more than one hop), or None.
    """
    if relations is None:
        return None
    if isinstance(relations, SpatialRelation):
        return relations
    relations = tuple(relations)
    if not relations:
        return None
    if not all(isinstance(r, SpatialRelation) for r in relations):
        raise ValueError(
            f"Relations must be expressed using SpatialRelations like `Factor`, `Translation` or `AxisRearrangementTo`. Received {relations!r}"
        )
    if len(relations) == 1:
        return relations[0]
    return SpatialRelationSequence(relations)
