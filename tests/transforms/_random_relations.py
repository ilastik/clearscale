"""Seeded generator of random SpatialRelation chains for property-style tests."""

import random
from typing import List, Tuple

from clearscale import (
    Affine,
    AxisRearrangementTo,
    AxisRenaming,
    Factor,
    PermutationTo,
    ProjectionTo,
    SpatialRelation,
    Translation,
)
from clearscale._spatial_relations import SpatialRelationSequence

SOURCE_POOL = "tczyx"
FRESH_POOL = "tczyxabdefghij"


def random_relation_chain(seed: int, *, max_steps: int = 5) -> Tuple[Tuple[str, ...], SpatialRelation]:
    """Returns (source_axes, relation): A chain of 1-5 relations that are valid for each other.
    Axes that were renamed away are never reused as names for inserted axes, and only axes that descend from the
    source axes are renamed: A transform_to_relation limitation (see its docs) is that an axis can't be given the
    name that an axis of the source still carries at the same time, until the final renaming.
    Axes that were dropped may be re-inserted."""
    rng = random.Random(seed)
    source = tuple(a for a in SOURCE_POOL if rng.random() < 0.7)
    if len(source) < 2:
        source = ("y", "x")
    axes = source
    retired: set = set()
    descends_from_source = {a: True for a in source}
    relations: List[SpatialRelation] = []

    def fresh_names(count: int) -> List[str]:
        candidates = [a for a in FRESH_POOL if a not in axes and a not in retired]
        rng.shuffle(candidates)
        return candidates[:count]

    def with_inserts(retained: List[str]) -> List[str]:
        inserts = fresh_names(rng.randint(0, 2)) if len(retained) < 4 else []
        result = list(retained)
        for name in inserts:
            result.insert(rng.randint(0, len(result)), name)
        return result

    for _ in range(rng.randint(1, max_steps)):
        kind = rng.choice(
            ["scaling", "affine", "translation", "permutation", "projection", "rearrangement", "renaming"]
        )
        if kind == "scaling":
            relation = Affine.from_factor(Factor({a: rng.choice([0.5, 1.0, 2.0, 3.5, 4.0]) for a in axes}))
        elif kind == "affine":
            # Triangular with a non-zero diagonal: Always invertible, and mixes axes
            n = len(axes)
            lower = rng.random() < 0.5
            rows = [
                [
                    (
                        rng.choice([0.5, 1.0, 2.0, 3.0])
                        if i == j
                        else (rng.choice([0.0, 0.0, 0.5, 1.0, 2.0]) if (j < i) == lower else 0.0)
                    )
                    for j in range(n)
                ]
                + [round(rng.uniform(-5, 5), 3)]
                for i in range(n)
            ]
            relation = Affine.from_array(rows, axes)
        elif kind == "translation":
            relation = Translation({a: round(rng.uniform(-5, 5), 3) for a in axes})
        elif kind == "permutation":
            order = list(axes)
            rng.shuffle(order)
            relation = PermutationTo(order)
        elif kind == "projection":
            retained = [a for a in axes if rng.random() < 0.7] or [axes[0]]
            relation = ProjectionTo(with_inserts(retained))
        elif kind == "rearrangement":
            retained = [a for a in axes if rng.random() < 0.7] or [axes[0]]
            rng.shuffle(retained)
            relation = AxisRearrangementTo(with_inserts(retained))
        else:
            renamable = [a for a in axes if descends_from_source[a]]
            if not renamable:
                continue
            chosen = [a for a in renamable if rng.random() < 0.5] or [renamable[0]]
            if len(chosen) > 1 and rng.random() < 0.5:
                new_names = chosen[1:] + chosen[:1]  # Rotate the names among the axes: a simultaneous swap
            else:
                new_names = fresh_names(len(chosen))
                if len(new_names) < len(chosen):
                    continue
            relation = AxisRenaming(dict(zip(chosen, new_names)))
            retired.update(chosen)
        dependencies = relation._axis_dependencies(axes)
        descends_from_source = {
            target: len(deps) == 1 and descends_from_source[deps[0]]
            for target, deps in zip(relation.target_axes(axes), dependencies)
        }
        axes = relation.target_axes(axes)
        relations.append(relation)
    if len(relations) == 1:
        return source, relations[0]
    return source, SpatialRelationSequence(tuple(relations))


def chain_mixes_axes(relation: SpatialRelation, source_axes) -> bool:
    """Whether any relation of the chain has output axes that depend on several input axes (an Affine, in practice)"""
    relations = relation.relations if isinstance(relation, SpatialRelationSequence) else (relation,)
    axes = tuple(source_axes)
    for step in relations:
        if any(len(deps) > 1 for deps in step._axis_dependencies(axes)):
            return True
        axes = step.target_axes(axes)
    return False
