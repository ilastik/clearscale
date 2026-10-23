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
    Axes that were renamed away are never reused as names for inserted axes: This would make the same name stand for
    two axes that are alive at the same time, which a chain of relations can express but is deliberately avoided here.
    Axes that were dropped may be re-inserted."""
    rng = random.Random(seed)
    source = tuple(a for a in SOURCE_POOL if rng.random() < 0.7)
    if len(source) < 2:
        source = ("y", "x")
    axes = source
    retired: set = set()
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
            chosen = [a for a in axes if rng.random() < 0.5] or [axes[0]]
            if len(chosen) > 1 and rng.random() < 0.5:
                new_names = chosen[1:] + chosen[:1]  # Rotate the names among the axes: a simultaneous swap
            else:
                new_names = fresh_names(len(chosen))
                if len(new_names) < len(chosen):
                    continue
            relation = AxisRenaming(dict(zip(chosen, new_names)))
            retired.update(chosen)
        axes = relation.target_axes(axes)
        relations.append(relation)
    if len(relations) == 1:
        return source, relations[0]
    return source, SpatialRelationSequence(tuple(relations))
