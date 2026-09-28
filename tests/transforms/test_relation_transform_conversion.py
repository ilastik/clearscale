from clearscale._spatial_relations import ProjectionTo, SpatialRelationSequence
from clearscale._transforms import TransformSequence, relation_to_transform_canonic, ProjectAxisTransform
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
