import pytest

from clearscale import Affine, Translation
from clearscale._affines import Linear
from clearscale._spatial_relations import ProjectionTo, SpatialRelationSequence
from clearscale._transforms import (
    AffineTransform,
    ProjectAxisTransform,
    ScaleTransform,
    TransformSequence,
    TranslationTransform,
    relation_to_transform_canonic,
)
from clearscale._transforms._to_from_spatial_relation import relation_to_transform


def test_relation_to_transform_maintains_sequence():
    relations = SpatialRelationSequence((ProjectionTo(("t", "z", "y", "x")),))
    result = relation_to_transform(relations, source_axes=("z", "y", "x"))
    assert isinstance(result, TransformSequence)
    assert len(result.transforms) == 1


def test_relation_to_transform_canonic_unwraps_sequence():
    relations = SpatialRelationSequence((ProjectionTo(("t", "z", "y", "x")),))
    result = relation_to_transform_canonic(relations, source_axes=("z", "y", "x"))
    assert isinstance(result, ProjectAxisTransform)
    assert result.inserts == (0,)


def test_affine_relation_to_transform_inverts_matrix_and_translation():
    affine = Affine(linear=Linear.from_array(((1, 2), (3, 4)), "yx"), translation=Translation(y=5, x=6))

    result = relation_to_transform(affine, source_axes=("y", "x"))

    assert isinstance(result, AffineTransform)
    assert result.affine is not None
    inverse = affine.inverted()
    assert [v for row in result.affine for v in row] == pytest.approx([v for row in inverse.to_tuples() for v in row])


def test_affine_relation_canonic_diagonal_is_inverted_scale():
    axes = ("z", "y", "x")

    result = relation_to_transform_canonic(Affine.from_factor({"z": 1.0, "y": 2.0, "x": 4.0}), axes)

    assert result == ScaleTransform(scale=(1.0, 0.5, 0.25))


def test_affine_relation_canonic_pure_translation_matches_translation():
    axes = ("y", "x")
    affine = Affine(translation=Translation(y=1.0, x=2.0))

    result = relation_to_transform_canonic(affine, axes)

    assert result == TranslationTransform(translation=(-1.0, -2.0))
    assert result == relation_to_transform_canonic(Translation(y=1.0, x=2.0), axes)


def test_affine_relation_canonic_scale_and_translation_is_sequence():
    affine = Affine(linear=Linear.from_array(((2, 0), (0, 2)), "yx"), translation=Translation(y=3.0, x=3.0))

    result = relation_to_transform_canonic(affine, ("y", "x"))

    assert isinstance(result, TransformSequence)
    assert result.transforms == (ScaleTransform(scale=(0.5, 0.5)), TranslationTransform(translation=(-1.5, -1.5)))


def test_affine_relation_rejects_singular_matrix():
    # Row 0 and 1 are linearly dependent (row1 == 2*row0), determinant 1*4-2*2=4-4=0
    affine = Affine(linear=Linear.from_array(((1, 2), (2, 4)), "yx"))

    with pytest.raises(ValueError, match="must be invertible"):
        relation_to_transform(affine, source_axes=("y", "x"))


def test_affine_relation_rejects_mismatching_source_axes():
    with pytest.raises(ValueError, match="Incompatible axes/order"):
        relation_to_transform(Affine.identity("yx"), source_axes=("z", "y", "x"))
