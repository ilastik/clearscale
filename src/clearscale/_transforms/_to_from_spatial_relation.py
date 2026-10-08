import itertools
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from clearscale._affines import Affine
from clearscale._errors import CannotConvertToAffineError, CannotConvertToSpatialRelationError
from clearscale._axis_values import Translation
from clearscale._spatial_relations import (
    AxisRenaming,
    normalize_relations_param,
    PermutationTo,
    ProjectionTo,
    SpatialRelation,
    AxisRearrangementTo,
    SpatialRelationSequence,
)
from clearscale._transforms._base import IdentityTransform, Transform, TransformSequence
from clearscale._transforms._transform_types import (
    ScaleTransform,
    AffineTransform,
    MapAxisTransform,
    ProjectAxisTransform,
    RotationTransform,
    TranslationTransform,
)
from clearscale.types import AxisKey, OrderedAxes

_PHYSICAL_LEAVES = (ScaleTransform, TranslationTransform, RotationTransform, AffineTransform)
"""Leaves that change values, not axes: Expressed as Translation and Affine relations, which keep axis names."""
_STRUCTURAL_LEAVES = (MapAxisTransform, ProjectAxisTransform)
"""Leaves that change axes, not values: Expressed as PermutationTo, ProjectionTo, AxisRearrangementTo."""


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

    if isinstance(relation, AxisRenaming):
        _ensure_compatible = relation.target_axes(source_axes)
        # The new names live on the target coordinate system, not on the transform: Axes are the same, positionally.
        return IdentityTransform()

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
                f"ambiguous axes: the same name is used by multiple axes at stage {index}: {axes!r}. "
                "(A renamed axis is called by its original name until the final renaming, so a name can't be reused for a "
                "newly created axis while the original axis lives on under another name.)"
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
    if isinstance(leaf, _PHYSICAL_LEAVES):
        # Translation and Affine relations keep the names of their axes, even if the values of the axes mix:
        # Output axis i of such a leaf is still axis i, whichever other axes its values depend on.
        assert len(dependencies) == len(stage), (
            f"dev error: Canonicalization splits axis changes off AffineTransforms (as ProjectAxisTransform), but "
            f"{type(leaf).__name__} maps {len(stage)} axes to {len(dependencies)}"
        )
        return stage
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
    Raises CannotConvertToSpatialRelationError if the chain contains leaves that don't map axes one-to-one.
    Axes present at both ends may have different names at the two ends: `names` calls them by their source name,
    which makes `named_stage(-1)` the final stage's names *before* the final renaming. Name collisions between dropped and inserted lineages are *not* an error
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
    # Working names: Axes that exist in the source are called by their source name until the final renaming,
    # axes created along the way take their name from the target.
    names: Dict[int, AxisKey] = dict(enumerate(source))
    for lineage, axis in zip(stage, target):
        names.setdefault(lineage, axis)
    return _AxisLineageTrace(leaves=leaves, stages=tuple(stages), names=names)


def _rearrangement(source: Tuple[AxisKey, ...], target: Tuple[AxisKey, ...]) -> Optional[SpatialRelation]:
    """The most specific name-based relation from `source` to `target`; None if they are the same."""
    if source == target:
        return None
    if set(source) == set(target):
        return PermutationTo(target)
    retained_in_source = [a for a in source if a in target]
    retained_in_target = [a for a in target if a in source]
    if retained_in_source == retained_in_target:
        return ProjectionTo(target)
    return AxisRearrangementTo(target)


def _structural_relations(trace: _AxisLineageTrace, start: int, end: int) -> List[SpatialRelation]:
    """
    Relations for the run of structural leaves trace.leaves[start:end], judged only by its in and out stages.
    If an axis is dropped and a *different* axis of the same name is inserted, a single name-based relation would read
    it as "keep", so it is expressed as two steps: drop first, then insert (and reorder).
    """
    in_lineages, out_lineages = trace.stages[start], trace.stages[end]
    if in_lineages == out_lineages:
        return []
    in_axes, out_axes = trace.named_stage(start), trace.named_stage(end)
    retained = tuple(axis for lineage, axis in zip(in_lineages, in_axes) if lineage in out_lineages)
    dropped, inserted = set(in_axes) - set(retained), set(out_axes) - set(retained)
    if dropped & inserted:
        if not retained:
            raise CannotConvertToSpatialRelationError(
                f"all axes {in_axes!r} are dropped and re-inserted under the same names; "
                "there is no intermediate state to express this as a sequence of relations"
            )
        steps = [ProjectionTo(retained), _rearrangement(retained, out_axes)]
        return [step for step in steps if step is not None]
    relation = _rearrangement(in_axes, out_axes)
    return [] if relation is None else [relation]


def _physical_relation(leaf: Transform, axes: Tuple[AxisKey, ...]) -> SpatialRelation:
    """Inverse of the Translation and Affine branches of relation_to_transform (which invert, see there)."""
    if isinstance(leaf, TranslationTransform):
        if not leaf.translation:
            raise CannotConvertToSpatialRelationError(
                f"{type(leaf).__name__} values are not loaded (path-backed); resolve and inline them first"
            )
        return Translation(zip(axes, leaf.translation)).inverted()
    # Scale, rotation and affine: Whatever their payload, their matrix is the inverse of the Affine relation's
    try:
        as_affine = leaf._to_affine_transform()
    except CannotConvertToAffineError:
        raise CannotConvertToSpatialRelationError(
            f"{type(leaf).__name__} values are not loaded (path-backed); resolve and inline them first"
        ) from None
    assert as_affine.affine is not None, "_to_affine_transform returns transforms with values"
    try:
        return Affine.from_array(as_affine.affine, axes).inverted()
    except ValueError as e:
        raise CannotConvertToSpatialRelationError(
            f"{type(leaf).__name__} cannot be inverted, but an Affine relation is the inverse of its transform: {e}"
        ) from None


def transform_to_relation(
    transform: Transform, source_axes: OrderedAxes, target_axes: OrderedAxes
) -> Optional[SpatialRelation]:
    """
    Express `transform` as the SpatialRelation(s) that describe how an image with `source_axes` is turned into one
    with `target_axes`. Inverse of `relation_to_transform_canonic`:
    `relation_to_transform_canonic(transform_to_relation(t, S, T), S) == t` for canonical `t` (up to float rounding).
    The reverse does not hold: canonicalization merges and reorders, so the result is always in canonical form.

    Axes are passed in explicitly: transforms only know positions, and may be unbound (or bound to systems whose
    axes need not be consulted). Axis correspondence is traced by position through the whole chain, and names are
    taken only from the two ends. Axes that have different names at the two ends are renamed in a final
    AxisRenaming step. Returns None if the transform is an identity (and `source_axes == target_axes`).

    Raises CannotConvertToSpatialRelationError if there is no faithful relation, which callers may expect and handle:
    transform types that don't map axes one-to-one (after canonicalization, and apart from the values of
    scales, rotations and affines, which are expressed as `Affine`), path-backed values, matrices that are not
    invertible, or intermediate axes that cannot be named (including: a name that is reused for a new axis while the
    axis that used to have that name lives on under another one).
    Known limitation: An Affine that mixes axes, together with relations that drop, insert or permute axes, is
    collapsed by canonicalization into one matrix, which cannot always be separated into these parts again.
    Chains that don't mix axes always convert back (apart from the naming ambiguity described above).
    Raises plain ValueError if `transform` doesn't fit the given axes (caller error).
    Anything else (AssertionError) is a bug in clearscale.
    """
    target_axes = tuple(target_axes)
    canonical = TransformSequence((transform.unbound(),)).canonicalized()
    trace = _trace_axis_lineages(_flatten_leaves(canonical), source_axes, target_axes)
    for leaf in trace.leaves:
        if not isinstance(leaf, _PHYSICAL_LEAVES + _STRUCTURAL_LEAVES):
            raise CannotConvertToSpatialRelationError(f"{type(leaf).__name__} has no SpatialRelation equivalent")
    relations: List[SpatialRelation] = []
    index = 0
    while index < len(trace.leaves):
        if isinstance(trace.leaves[index], _PHYSICAL_LEAVES):
            relations.append(_physical_relation(trace.leaves[index], trace.named_stage(index)))
            index += 1
            continue
        run_end = index
        while run_end < len(trace.leaves) and not isinstance(trace.leaves[run_end], _PHYSICAL_LEAVES):
            run_end += 1
        relations.extend(_structural_relations(trace, index, run_end))
        index = run_end
    # All relations above spoke the source's vocabulary. Whatever has a different name at the end is renamed last.
    final_axes = trace.named_stage(len(trace.leaves))
    renames = {axis: target for axis, target in zip(final_axes, target_axes) if axis != target}
    if renames:
        relations.append(AxisRenaming(renames))
    return normalize_relations_param(relations)
