import pytest

from clearscale._errors import CannotConvertToSpatialRelationError
from clearscale._transforms import (
    ByDimensionTransform,
    MapAxisTransform,
    ProjectAxisTransform,
    RotationTransform,
    ScaleTransform,
    TranslationTransform,
    TransformSequence,
    _ByDimensionChild,
)
from clearscale._transforms._to_from_spatial_relation import _AxisLineageTrace


def test_position_preserving_leaves_keep_lineages():
    leaves = (ScaleTransform((1.0, 2.0)), TranslationTransform((0.0, 1.0)))
    trace = _AxisLineageTrace.trace(leaves, "yx", "yx")
    assert trace.stages == ((0, 1), (0, 1), (0, 1))
    assert trace.axis_keys_at(1) == ("y", "x")


def test_map_axis_follows_lineages():
    trace = _AxisLineageTrace.trace((MapAxisTransform((1, 0)),), "yx", "xy")
    assert trace.stages == ((0, 1), (1, 0))
    assert trace.axis_keys_at(0) == ("y", "x")
    assert trace.axis_keys_at(1) == ("x", "y")


def test_project_drops_and_inserts_lineages():
    trace = _AxisLineageTrace.trace((ProjectAxisTransform(drops=(0,), inserts=(0,)),), "zyx", "cyx")
    assert trace.stages == ((0, 1, 2), (3, 1, 2))
    assert trace.axis_keys == {0: "z", 1: "y", 2: "x", 3: "c"}


def test_inserted_lineage_is_named_by_target_across_hops():
    leaves = (ProjectAxisTransform(inserts=(0,)), ScaleTransform((1.0, 2.0, 2.0)), MapAxisTransform((0, 2, 1)))
    trace = _AxisLineageTrace.trace(leaves, "yx", "cxy")
    assert trace.stages == ((0, 1), (2, 0, 1), (2, 0, 1), (2, 1, 0))
    assert trace.axis_keys_at(0) == ("y", "x")
    assert trace.axis_keys_at(1) == ("c", "y", "x")
    assert trace.axis_keys_at(2) == ("c", "y", "x")
    assert trace.axis_keys_at(3) == ("c", "x", "y")


def test_temporary_axis_inside_run_is_unnamed_but_harmless():
    leaves = (ProjectAxisTransform(inserts=(0,)), ProjectAxisTransform(drops=(0,)))
    trace = _AxisLineageTrace.trace(leaves, "yx", "yx")
    assert trace.axis_keys_at(0) == ("y", "x")
    assert trace.axis_keys_at(2) == ("y", "x")
    with pytest.raises(CannotConvertToSpatialRelationError, match="cannot be named"):
        trace.axis_keys_at(1)


def test_name_collision_between_dropped_and_inserted_is_not_an_error_in_trace():
    trace = _AxisLineageTrace.trace((ProjectAxisTransform(drops=(0,), inserts=(0,)),), "cyx", "cyx")
    assert trace.stages == ((0, 1, 2), (3, 1, 2))
    assert trace.axis_keys[0] == trace.axis_keys[3] == "c"


def test_ambiguous_names_at_a_stage_raise():
    # Source "c" is still alive while the lineage inserted in the first hop is named "c" by the target
    leaves = (ProjectAxisTransform(inserts=(0,)), ScaleTransform((1.0, 1.0, 1.0)), ProjectAxisTransform(drops=(1,)))
    trace = _AxisLineageTrace.trace(leaves, "cx", "cx")
    assert trace.stages == ((0, 1), (2, 0, 1), (2, 0, 1), (2, 1))
    assert trace.axis_keys == {0: "c", 1: "x", 2: "c"}
    assert trace.axis_keys_at(0) == ("c", "x")
    assert trace.axis_keys_at(3) == ("c", "x")
    with pytest.raises(CannotConvertToSpatialRelationError, match="ambiguous"):
        trace.axis_keys_at(1)
    with pytest.raises(CannotConvertToSpatialRelationError, match="ambiguous"):
        trace.axis_keys_at(2)


def test_relabeling_raises():
    with pytest.raises(CannotConvertToSpatialRelationError, match="relabeled"):
        _AxisLineageTrace.trace((ScaleTransform((1.0, 1.0)),), "yx", "YX")
    with pytest.raises(CannotConvertToSpatialRelationError, match="relabeled"):
        _AxisLineageTrace.trace((), "yx", "xy")


@pytest.mark.parametrize(
    "transforms, source, target",
    [
        ((ScaleTransform((1.0, 1.0)),), "zyx", "zyx"),
        ((ProjectAxisTransform(inserts=(0,)),), "yx", "yx"),
        ((ProjectAxisTransform(drops=(5,)),), "yx", "x"),
        ((ProjectAxisTransform(inserts=(0,)), ScaleTransform((1.0, 1.0))), "yx", "cyx"),
    ],
)
def test_transform_not_fitting_axes_is_caller_error_not_conversion_error(transforms, source, target):
    with pytest.raises(ValueError) as exc_info:
        _AxisLineageTrace.trace(transforms, source, target)
    assert not isinstance(exc_info.value, CannotConvertToSpatialRelationError)


def test_rejects_duplicate_axes():
    with pytest.raises(ValueError, match="unique"):
        _AxisLineageTrace.trace((), "yy", "yy")


def test_non_physical_transform_that_mixes_axes_raises_conversion_error():
    mixing_child = _ByDimensionChild(
        source_indices=(0, 1), target_indices=(0, 1), transform=RotationTransform(rotation=((0.8, -0.6), (0.6, 0.8)))
    )
    with pytest.raises(CannotConvertToSpatialRelationError, match="ByDimensionTransform mixes axes"):
        _AxisLineageTrace.trace((ByDimensionTransform((mixing_child,)),), "yx", "yx")


def test_physical_transforms_keep_axes_in_place_even_if_their_values_mix():
    # Translation and Affine relations keep the axis names: Output axis i is still axis i
    rotation = RotationTransform(rotation=((0.8, -0.6), (0.6, 0.8)))
    assert _AxisLineageTrace.trace((rotation,), "yx", "yx").stages == ((0, 1), (0, 1))
    swap = RotationTransform(rotation=((0.0, -1.0), (1.0, 0.0)))
    assert _AxisLineageTrace.trace((swap,), "yx", "yx").stages == ((0, 1), (0, 1))


def test_inconsistent_trace_is_internal_error():
    transform = ScaleTransform((1.0, 1.0))
    assert _AxisLineageTrace(
        transforms=(transform,), stages=((0, 1), (0, 1)), axis_keys={0: "y", 1: "x"}
    ), "correct form"
    with pytest.raises(AssertionError):  # not enough stages
        _AxisLineageTrace(transforms=(transform,), stages=((0, 1),), axis_keys={0: "y", 1: "x"})
    with pytest.raises(AssertionError):  # duplicate lineage IDs
        _AxisLineageTrace(transforms=(transform,), stages=((0, 0), (0, 0)), axis_keys={0: "y"})
    with pytest.raises(AssertionError):  # not enough axis keys
        _AxisLineageTrace(transforms=(transform,), stages=((0, 1), (0, 1)), axis_keys={0: "y"})
    with pytest.raises(AssertionError):  # not flattened
        _AxisLineageTrace(
            transforms=(TransformSequence((transform,)),), stages=((0, 1), (0, 1)), axis_keys={0: "y", 1: "x"}
        )
