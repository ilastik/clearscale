import itertools
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from clearscale._affines import Affine
from clearscale._errors import CannotConvertToSpatialRelationError
from clearscale._axis_values import Translation
from clearscale._spatial_relations import (
    PermutationTo,
    ProjectionTo,
    SpatialRelation,
    AxisRearrangementTo,
    SpatialRelationSequence,
)
from clearscale._transforms._base import IdentityTransform, Transform, TransformSequence
from clearscale._transforms._transform_types import (
    AffineTransform,
    MapAxisTransform,
    ProjectAxisTransform,
    TranslationTransform,
)
from clearscale.types import AxisKey, OrderedAxes


def relation_to_transform(relation: SpatialRelation, source_axes: OrderedAxes) -> Transform:
    """Build one Transform for a single SpatialRelation hop. Validate that `relation`
    can operate on `source_axes` and if so, reshape it to produce a Transform payload that fits `source_axes`."""
    source_axes = tuple(source_axes)

    if isinstance(relation, SpatialRelationSequence):
        steps: List[Transform] = []
        current_axes = source_axes
        for child in relation.relations:
            steps.append(relation_to_transform(child, current_axes))
            current_axes = child.target_axes(current_axes)
        return TransformSequence(tuple(steps))

    if isinstance(relation, Translation):
        _ensure_compatible = relation.target_axes(source_axes)
        # Inversion: When acting as a SpatialRelation, the Translation is exactly the inverse of what the
        # coordinate transformation needs to be.
        # `derived_ms = B.as_derived_from(A, by=Translation(x=1.0))`
        # means "B was derived by shifting A by 1".
        # Then A[0] == B[-1] (A's origin is outside B, because B is shifted), and B[0] == A[1]
        # i.e. to obtain B-coordinates, A-coordinates must have 1 subtracted (-1 added).
        # hence, A--Translation(1.0)-->B is the coordinate transform A--TranslationTransform(-1.0)-->B.
        # Likewise:
        # `embedded_ms = A.with_coordinate_system("B", reached_by=Translation(x=1.0))`
        # means "B is reached by shifting A by 1".
        # Again, A[0] == B[-1] - if shifting A produces B, then A's origin is outside B.
        return TranslationTransform(translation=tuple(relation.inverted().values()))

    if isinstance(relation, Affine):
        _ensure_compatible = relation.target_axes(source_axes)
        # Affine uses a convention such that Factor values become the matrix diagonal, and Translation
        # becomes the offset column as-is, without any changes to the numbers.
        # This means it needs to be inverted the same way.
        try:
            inverse = relation.inverted()
        except ValueError as e:
            raise ValueError(f"Affine relation must be invertible to connect two coordinate systems: {e}") from e
        return AffineTransform(affine=inverse.to_tuples())

    if isinstance(relation, ProjectionTo):
        target_axes = relation.target_axes(source_axes)
        dropped = set(source_axes) - set(target_axes)
        inserted = set(target_axes) - set(source_axes)
        return ProjectAxisTransform(
            drops=tuple(i for i, axis in enumerate(source_axes) if axis in dropped),
            inserts=tuple(i for i, axis in enumerate(target_axes) if axis in inserted),
        )

    if isinstance(relation, PermutationTo):
        target_axes = relation.target_axes(source_axes)
        return MapAxisTransform(map_axis=tuple(source_axes.index(axis) for axis in target_axes))

    if isinstance(relation, AxisRearrangementTo):
        target_axes = relation.target_axes(source_axes)
        if not relation.dropped_axes(source_axes) and not relation.inserted_axes(source_axes):
            return relation_to_transform(PermutationTo(target_axes), source_axes)
        try:
            return relation_to_transform(ProjectionTo(target_axes), source_axes)
        except ValueError:
            pass
        intermediate_axes = relation.retained_axes(source_axes) + relation.inserted_axes(source_axes)
        project_axes = relation_to_transform(ProjectionTo(intermediate_axes), source_axes)
        map_axis = relation_to_transform(PermutationTo(target_axes), intermediate_axes)
        return TransformSequence((project_axes, map_axis))

    raise NotImplementedError(f"Conversion to Transform not yet implemented for {relation.__class__.__name__}")


def relation_to_transform_canonic(relation: SpatialRelation, source_axes: OrderedAxes) -> Transform:
    raw_relation = relation_to_transform(relation, source_axes)
    return TransformSequence((raw_relation,)).canonicalized()


@dataclass(frozen=True, slots=True)
class _AxisLineageTrace:
    """
    Positional trace of axis identity through a flat chain of transform leaves.
    Every axis gets an integer lineage id when it is born (source axes, or axes inserted by a ProjectAxisTransform).
    Transforms only know positions, so a lineage is the only thing that survives across MapAxis/ProjectAxis hops.
    Names exist only at the two ends of the chain (source_axes/target_axes) and are propagated along lineages.
    """

    leaves: Tuple[Transform, ...]
    stages: Tuple[Tuple[int, ...], ...]
    """stages[i] are the lineage ids on the input of leaves[i]; stages[-1] are the ids of the final output."""
    names: Dict[int, AxisKey]
    """Lineage ids that are alive at either end, with their name. Lineages born and dropped in between are unnamed."""

    def __post_init__(self):
        assert not any(isinstance(t, TransformSequence) for t in self.leaves), "must be flat"
        assert len(self.stages) == len(self.leaves) + 1, "dev error: one stage before each leaf, plus the final output"
        assert all(
            len(set(stage)) == len(stage) for stage in self.stages
        ), f"dev error: duplicate lineage in {self.stages}"
        assert set(self.stages[0]) | set(self.stages[-1]) <= set(self.names), "dev error: end lineages must be named"

    def named_stage(self, index: int) -> Tuple[AxisKey, ...]:
        """The axis names at stage `index`. Raises if any axis there is unnameable or the names are ambiguous."""
        stage = self.stages[index]
        unnamed = [lineage for lineage in stage if lineage not in self.names]
        if unnamed:
            raise CannotConvertToSpatialRelationError(
                f"intermediate axes between transforms cannot be named (stage {index} has {len(unnamed)} unnamed)"
            )
        axes = tuple(self.names[lineage] for lineage in stage)
        if len(set(axes)) != len(axes):
            raise CannotConvertToSpatialRelationError(
                f"ambiguous intermediate axes: the same name is used by multiple axes at stage {index}: {axes!r}"
            )
        return axes


def _flatten_leaves(transform: Transform) -> Tuple[Transform, ...]:
    """Recursively flatten sequences (including legacy MultiscaleTransforms) and drop identities."""
    if isinstance(transform, TransformSequence):
        return tuple(leaf for child in transform.transforms for leaf in _flatten_leaves(child))
    return () if isinstance(transform, IdentityTransform) else (transform,)


def _advance_lineages(leaf: Transform, stage: Tuple[int, ...], new_lineage: Callable[[], int]) -> Tuple[int, ...]:
    """Map the lineage ids on the input of `leaf` to those on its output.
    Raises ValueError (not CannotConvert...) if `leaf` doesn't fit `len(stage)` axes: the caller supplied wrong axes."""
    dependencies = leaf._axis_dependencies(len(stage))
    if dependencies is None:
        raise CannotConvertToSpatialRelationError(
            f"the dependencies of {type(leaf).__name__}'s output axes on its input axes are unknown, "
            "so it has no SpatialRelation equivalent"
        )
    assert all(
        0 <= source < len(stage) for target_dependencies in dependencies for source in target_dependencies
    ), f"dev error: {type(leaf).__name__}._axis_dependencies({len(stage)}) returned {dependencies!r}"
    if any(len(target_dependencies) > 1 for target_dependencies in dependencies):
        raise CannotConvertToSpatialRelationError(
            f"{type(leaf).__name__} mixes axes (output axes depend on input axes {dependencies!r}), "
            "which no SpatialRelation can express"
        )
    return tuple(stage[d[0]] if d else new_lineage() for d in dependencies)


def _trace_axis_lineages(
    leaves: Tuple[Transform, ...], source_axes: OrderedAxes, target_axes: OrderedAxes
) -> _AxisLineageTrace:
    """
    Trace axis lineages through `leaves` and name them from `source_axes` and `target_axes`.
    Raises ValueError if the axes are not unique, or the chain does not fit them (ndim of `source_axes` or `target_axes`).
    Raises CannotConvertToSpatialRelationError if the chain would have to relabel an axis (a lineage present at both
    ends with differing names), or contains leaves that don't map axes one-to-one. Name collisions between dropped and inserted lineages are *not* an error
    here; they are resolved on emission (a drop step followed by an insert step).
    """
    source = tuple(source_axes)
    target = tuple(target_axes)
    for label, axes in (("source", source), ("target", target)):
        if len(set(axes)) != len(axes):
            raise ValueError(f"{label} axes must be unique. Received: {axes!r}")
    next_id = itertools.count(len(source))
    stage = tuple(range(len(source)))
    stages = [stage]
    for leaf in leaves:
        stage = _advance_lineages(leaf, stage, lambda: next(next_id))
        stages.append(stage)
    if len(stage) != len(target):
        raise ValueError(f"transform produces {len(stage)} axes from {source!r}, but target axes are {target!r}")
    names: Dict[int, AxisKey] = dict(enumerate(source))
    for lineage, axis in zip(stage, target):
        if names.setdefault(lineage, axis) != axis:
            raise CannotConvertToSpatialRelationError(
                f"axis {names[lineage]!r} would have to be relabeled {axis!r} "
                f"(source {source!r}, target {target!r}); relabeling is not expressible"
            )
    return _AxisLineageTrace(leaves=leaves, stages=tuple(stages), names=names)
