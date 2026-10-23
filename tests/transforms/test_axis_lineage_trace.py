import pytest

from clearscale._errors import CannotConvertToSpatialRelationError
from clearscale._transforms import (
    MapAxisTransform,
    ProjectAxisTransform,
    RotationTransform,
    ScaleTransform,
    TranslationTransform,
)
from clearscale._transforms._to_from_spatial_relation import (
    _AxisLineageTrace,
    _flatten_leaves,
    _trace_axis_lineages,
)


def test_position_preserving_leaves_keep_lineages():
    leaves = (ScaleTransform((1.0, 2.0)), TranslationTransform((0.0, 1.0)))
    trace = _trace_axis_lineages(leaves, "yx", "yx")
    assert trace.stages == ((0, 1), (0, 1), (0, 1))
    assert trace.named_stage(1) == ("y", "x")


def test_map_axis_follows_lineages():
    trace = _trace_axis_lineages((MapAxisTransform((1, 0)),), "yx", "xy")
    assert trace.stages == ((0, 1), (1, 0))
    assert trace.named_stage(0) == ("y", "x")
    assert trace.named_stage(1) == ("x", "y")


def test_project_drops_and_inserts_lineages():
    trace = _trace_axis_lineages((ProjectAxisTransform(drops=(0,), inserts=(0,)),), "zyx", "cyx")
    assert trace.stages == ((0, 1, 2), (3, 1, 2))
    assert trace.names == {0: "z", 1: "y", 2: "x", 3: "c"}


def test_inserted_lineage_is_named_by_target_even_across_hops():
    leaves = (ProjectAxisTransform(inserts=(0,)), ScaleTransform((1.0, 2.0, 2.0)), MapAxisTransform((0, 2, 1)))
    trace = _trace_axis_lineages(leaves, "yx", "cxy")
    assert trace.stages[1] == (2, 0, 1)
    assert trace.named_stage(1) == ("c", "y", "x")
    assert trace.named_stage(3) == ("c", "x", "y")


def test_temporary_axis_inside_run_is_unnamed_but_harmless():
    leaves = (ProjectAxisTransform(inserts=(0,)), ProjectAxisTransform(drops=(0,)))
    trace = _trace_axis_lineages(leaves, "yx", "yx")
    assert trace.named_stage(0) == ("y", "x")
    assert trace.named_stage(2) == ("y", "x")
    with pytest.raises(CannotConvertToSpatialRelationError, match="cannot be named"):
        trace.named_stage(1)


def test_name_collision_between_dropped_and_inserted_is_not_an_error_in_trace():
    trace = _trace_axis_lineages((ProjectAxisTransform(drops=(0,), inserts=(0,)),), "cyx", "cyx")
    assert trace.stages == ((0, 1, 2), (3, 1, 2))
    assert trace.names[0] == trace.names[3] == "c"


def test_ambiguous_names_at_a_stage_raise():
    # Source "c" is still alive while the lineage inserted in the first hop is named "c" by the target
    leaves = (ProjectAxisTransform(inserts=(0,)), ScaleTransform((1.0, 1.0, 1.0)), ProjectAxisTransform(drops=(1,)))
    trace = _trace_axis_lineages(leaves, "cx", "cx")
    assert trace.names[0] == "c"
    with pytest.raises(CannotConvertToSpatialRelationError, match="ambiguous"):
        trace.named_stage(1)


def test_relabeling_raises():
    with pytest.raises(CannotConvertToSpatialRelationError, match="relabeled"):
        _trace_axis_lineages((ScaleTransform((1.0, 1.0)),), "yx", "YX")
    with pytest.raises(CannotConvertToSpatialRelationError, match="relabeled"):
        _trace_axis_lineages((), "yx", "xy")


def test_transform_not_fitting_the_axes_is_a_caller_error_not_a_conversion_error():
    for leaves, source, target in (
        ((ScaleTransform((1.0, 1.0)),), "zyx", "zyx"),
        ((ProjectAxisTransform(inserts=(0,)),), "yx", "yx"),
        ((ProjectAxisTransform(drops=(5,)),), "yx", "x"),
        ((ProjectAxisTransform(inserts=(0,)), ScaleTransform((1.0, 1.0))), "yx", "cyx"),
    ):
        with pytest.raises(ValueError) as exc_info:
            _trace_axis_lineages(leaves, source, target)
        assert not isinstance(exc_info.value, CannotConvertToSpatialRelationError)


def test_axis_mixing_leaf_raises_conversion_error():
    rotation = RotationTransform(rotation=((0.8, -0.6), (0.6, 0.8)))
    with pytest.raises(CannotConvertToSpatialRelationError, match="RotationTransform mixes axes"):
        _trace_axis_lineages((rotation,), "yx", "yx")


def test_one_to_one_rotation_is_traced_like_a_permutation():
    # Whether it is *convertible* is up to relation emission
    rotation = RotationTransform(rotation=((0.0, -1.0), (1.0, 0.0)))
    assert _trace_axis_lineages((rotation,), "yx", "xy").stages == ((0, 1), (1, 0))


def test_inconsistent_trace_is_an_internal_error():
    leaf = ScaleTransform((1.0, 1.0))
    with pytest.raises(AssertionError):
        _AxisLineageTrace(leaves=(leaf,), stages=((0, 1),), names={0: "y", 1: "x"})
    with pytest.raises(AssertionError):
        _AxisLineageTrace(leaves=(leaf,), stages=((0, 0), (0, 0)), names={0: "y"})
    with pytest.raises(AssertionError):
        _AxisLineageTrace(leaves=(leaf,), stages=((0, 1), (0, 1)), names={0: "y"})


def test_duplicate_axes_raise():
    with pytest.raises(ValueError, match="unique"):
        _trace_axis_lineages((), "yy", "yy")


def test_flatten_leaves_flattens_nested_sequences_and_drops_identities():
    from clearscale._transforms import IdentityTransform, TransformSequence

    scale, translation = ScaleTransform((2.0, 2.0)), TranslationTransform((1.0, 1.0))
    nested = TransformSequence((IdentityTransform(), TransformSequence((scale, translation))))
    assert _flatten_leaves(nested) == (scale, translation)
