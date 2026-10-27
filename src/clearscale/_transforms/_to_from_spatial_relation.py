import itertools
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Type

from clearscale._affines import Affine
from clearscale._errors import CannotConvertToAffineError, CannotConvertToSpatialRelationError
from clearscale._axis_values import Translation
from clearscale._services.matrices import AxisIndices
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
    BijectionTransform,
)
from clearscale.types import AxisKey, OrderedAxes

_PHYSICAL_TYPES = (ScaleTransform, TranslationTransform, RotationTransform, AffineTransform)
"""Types that change values, not axes: Expressed as Translation and Affine."""
_STRUCTURAL_TYPES: Tuple[Type[Transform], ...] = (MapAxisTransform, ProjectAxisTransform)
"""Types that change axes, not values: Expressed as PermutationTo, ProjectionTo, AxisRearrangementTo."""


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
    Positional trace of axis identity through a chain of atomic transforms (sequences flattened, bijections unwrapped).
    Every axis gets an integer lineage id when it first appears (source axes / inserted by ProjectAxisTransform).
    Names exist only at the two ends of the chain (source_axes/target_axes) and are propagated along lineages.
    """

    transforms: Tuple[Transform, ...]
    stages: Tuple[AxisIndices, ...]
    """stages[i] = lineage ids on the input of transforms[i]; stages[-1] = ids of the final output."""
    axis_keys: Dict[int, AxisKey]
    """All lineage ids that have a key from source or target. Axes that are inserted and later dropped again 
    never have a name. axis_keys_at raises when requesting a stage where such a temporary axis exists."""

    def __post_init__(self):
        assert not any(isinstance(t, TransformSequence) for t in self.transforms), "must be flat"
        assert (
            len(self.stages) == len(self.transforms) + 1
        ), "dev error: one stage before each leaf, plus the final output"
        assert all(
            len(set(stage)) == len(stage) for stage in self.stages
        ), f"dev error: duplicate lineage in {self.stages}"
        assert set(self.stages[0]) | set(self.stages[-1]) <= set(
            self.axis_keys
        ), "dev error: end lineages must be named"

    def axis_keys_at(self, index: int) -> Tuple[AxisKey, ...]:
        """The axis names at stage `index`. Raises if any axis there is unnameable or the names are ambiguous."""
        stage = self.stages[index]
        unnamed = [lineage for lineage in stage if lineage not in self.axis_keys]
        if unnamed:
            raise CannotConvertToSpatialRelationError(
                f"intermediate axes between transforms cannot be named (stage {index} has {len(unnamed)} unnamed)"
            )
        axes = tuple(self.axis_keys[lineage] for lineage in stage)
        if len(set(axes)) != len(axes):
            raise CannotConvertToSpatialRelationError(
                f"ambiguous axes: the same name is used by multiple axes at stage {index}: {axes!r}. "
                "(A renamed axis is called by its original name until the final renaming, so a name can't be reused for a "
                "newly created axis while the original axis lives on under another name.)"
            )
        return axes

    @classmethod
    def trace(
        cls, atomic_transforms: Tuple[Transform, ...], source_axes: OrderedAxes, target_axes: OrderedAxes
    ) -> "_AxisLineageTrace":
        """
        Trace axis lineages through `leaves` and name them from `source_axes` and `target_axes`.
        Raises ValueError if the axes are not unique, or the chain does not fit them (ndim of `source_axes` or `target_axes`).
        Raises CannotConvertToSpatialRelationError if the chain contains leaves that don't map axes one-to-one.
        Axes present at both ends may have different names at the two ends: `names` calls them by their source name,
        which makes `named_stage(-1)` the final stage's names *before* the final renaming. Name collisions between dropped and inserted lineages are *not* an error
        here; they are resolved on emission (a drop step followed by an insert step).
        """

        def _target_ids_for(t: Transform, source_ids: AxisIndices, new_lineage: Callable[[], int]) -> AxisIndices:
            """Map source_ids to target_ids based on how targets correspond to sources for `t`."""
            sources_by_target = t._axis_dependencies(len(source_ids))
            if sources_by_target is None:
                raise CannotConvertToSpatialRelationError(
                    f"{type(t).__name__} values are not loaded (path-backed), so it cannot determine how input axes correspond to outputs"
                )
            assert all(
                0 <= s < len(source_ids) for sources in sources_by_target for s in sources
            ), f"dev error: {type(t).__name__}._axis_dependencies({len(source_ids)}) returned a target depending on a non-existent source: {sources_by_target!r}"
            if isinstance(t, _PHYSICAL_TYPES):
                # Translation and Affine relations use source_axes==target_axes:
                # Even for a mixing affine, output axis i is still axis i.
                assert len(sources_by_target) == len(
                    source_ids
                ), f"dev error: should canonicalize the chain first, got non-canonical {t!r} (canonical affines don't change ndim)"
                return source_ids
            if any(len(sources) > 1 for sources in sources_by_target):
                # Basically only possible with ByDimension now
                raise CannotConvertToSpatialRelationError(
                    f"{type(t).__name__} mixes axes (target axes depend on source axes {sources_by_target!r}), "
                    "which no SpatialRelation can express"
                )
            assert all(len(sources) <= 1 for sources in sources_by_target), "dev error: no more mixing by now"
            return tuple(source_ids[sources[0]] if sources else new_lineage() for sources in sources_by_target)

        source = tuple(source_axes)
        target = tuple(target_axes)
        for label, axes in (("source", source), ("target", target)):
            if len(set(axes)) != len(axes):
                raise ValueError(f"{label} axes must be unique. Received: {axes!r}")
        next_id = itertools.count(len(source))
        stage = tuple(range(len(source)))
        stages = [stage]
        for t in atomic_transforms:
            stage = _target_ids_for(t, stage, lambda: next(next_id))
            stages.append(stage)
        if len(stage) != len(target):
            raise ValueError(f"transform produces {len(stage)} axes from {source!r}, but target axes are {target!r}")
        # These are working IDs filled from source and target, before the final AxisRenaming
        id_to_key: Dict[int, AxisKey] = dict(enumerate(source))
        for lineage, axis in zip(stage, target):
            id_to_key.setdefault(lineage, axis)
        return cls(transforms=atomic_transforms, stages=tuple(stages), axis_keys=id_to_key)


def _rearrangement(source: Tuple[AxisKey, ...], target: Tuple[AxisKey, ...]) -> Optional[SpatialRelation]:
    """The most specific structural relation from `source` to `target`; None if they are the same."""
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
    in_axes, out_axes = trace.axis_keys_at(start), trace.axis_keys_at(end)
    retained_axes = tuple(axis for lineage, axis in zip(in_lineages, in_axes) if lineage in out_lineages)
    dropped_axes, inserted_axes = set(in_axes) - set(retained_axes), set(out_axes) - set(retained_axes)
    if dropped_axes & inserted_axes:
        if not retained_axes:
            raise CannotConvertToSpatialRelationError(
                f"all axes {in_axes!r} are dropped and re-inserted under the same names; "
                "there is no intermediate state to express this as a sequence of relations"
            )
        steps = [ProjectionTo(retained_axes), _rearrangement(retained_axes, out_axes)]
        return [step for step in steps if step is not None]
    relation = _rearrangement(in_axes, out_axes)
    return [] if relation is None else [relation]


def _physical_relation(transform: Transform, axes: Tuple[AxisKey, ...]) -> SpatialRelation:
    """Inverse of the Translation and Affine branches of relation_to_transform (which invert, see there)."""
    assert isinstance(transform, _PHYSICAL_TYPES)
    if isinstance(transform, TranslationTransform):
        if not transform.translation:
            raise CannotConvertToSpatialRelationError(
                f"{type(transform).__name__} values are not loaded (path-backed); please request if you need this"
            )
        return Translation(zip(axes, transform.translation)).inverted()
    # Scale, rotation and affine: Whatever their payload, their matrix is the inverse of the Affine relation's
    try:
        as_affine = transform._to_affine_transform()
    except CannotConvertToAffineError:
        raise CannotConvertToSpatialRelationError(
            f"{type(transform).__name__} values are not loaded (path-backed); please request if you need this"
        ) from None
    assert as_affine.affine is not None, "_to_affine_transform returns transforms with values"
    try:
        return Affine.from_array(as_affine.affine, axes).inverted()
    except ValueError as e:
        raise CannotConvertToSpatialRelationError(
            f"{type(transform).__name__} cannot be inverted, but an Affine relation is the inverse of its transform: {e}"
        ) from None


def _flatten_atomic(transform: Transform) -> Tuple[Transform, ...]:
    """Recursively flatten sequences down to atomic transforms for per-transform conversion to relation.
    (i.e. also ome_zarr.MultiscaleTransforms, and unwrap BijectionTransform)."""
    if isinstance(transform, TransformSequence):
        return tuple(leaf for child in transform.transforms for leaf in _flatten_atomic(child))
    if isinstance(transform, BijectionTransform):
        return _flatten_atomic(transform.forward)
    return () if isinstance(transform, IdentityTransform) else (transform,)


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
    atomic_transforms = _flatten_atomic(canonical)
    for t in atomic_transforms:
        if not isinstance(t, _PHYSICAL_TYPES + _STRUCTURAL_TYPES):
            raise CannotConvertToSpatialRelationError(f"{type(t).__name__} has no SpatialRelation equivalent")
    trace = _AxisLineageTrace.trace(atomic_transforms, source_axes, target_axes)
    relations: List[SpatialRelation] = []
    index = 0
    while index < len(trace.transforms):
        if isinstance(trace.transforms[index], _PHYSICAL_TYPES):
            relations.append(_physical_relation(trace.transforms[index], trace.axis_keys_at(index)))
            index += 1
            continue
        run_end = index
        while run_end < len(trace.transforms) and not isinstance(trace.transforms[run_end], _PHYSICAL_TYPES):
            run_end += 1
        relations.extend(_structural_relations(trace, index, run_end))
        index = run_end
    # All relations above used the source axis keys. Add a final renaming if target uses aliases
    final_axes = trace.axis_keys_at(len(trace.transforms))
    renames = {axis: target for axis, target in zip(final_axes, target_axes) if axis != target}
    if renames:
        relations.append(AxisRenaming(renames))
    return normalize_relations_param(relations)
