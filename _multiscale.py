import uuid
import warnings
from abc import ABC
from collections import OrderedDict, defaultdict
from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from typing import (
    Optional,
    TypeVar,
    Mapping,
    Generic,
    Union,
    Sequence,
    Callable,
    Iterable,
    List,
    Tuple,
    Literal,
    Dict,
    Any,
    TYPE_CHECKING,
    Hashable,
)

from lazyflow.utility.io_util.clearscale import (
    Shape,
    Factor,
    Spacing,
    Unit,
    Translation,
    PixelOffset,
    _ome_zarr,
    _precomputed,
)
from lazyflow.utility.io_util.clearscale._axis_values import (
    ShapeLike,
    Axes,
    RoundingMethod,
    OrderedAxes,
    _AxisValues,
    AxisKey,
)
from lazyflow.utility.io_util.clearscale._transforms import (
    CoordinateSystemName,
    CoordinateSystem,
    _TransformGraph,
    CoordinateSystemRef,
    IdentityTransform,
    TransformGraphNode,
    PRE_TRANSFORMS_VERSIONS,
    TransformSequence,
)

ScaleKey = TypeVar("ScaleKey", bound=str)
ValueType = TypeVar("ValueType", Shape, Factor, "Scale")
AxisValuesType = TypeVar("AxisValuesType", Shape, Factor)
DEFAULT_NAME_PATTERN = "s{}"

TranslationShiftFunction = Callable[["Scale", "Scale"], "Translation"]
"""
base_scale: the reference scale being transformed from
target_scale: the new scale being created (with 0 translation)
Returns: target_scale's translation
"""

if TYPE_CHECKING:
    try:
        from typing import Self  # py 3.11+
    except ImportError:
        try:
            from typing_extensions import Self  # py 3.10 + optional dep
        except ImportError:
            _Self = TypeVar("_Self")
            Self = _Self


class DuplicatePolicy(StrEnum):
    ERROR = "error"
    KEEP_ALL = "keep_all"
    KEEP_FIRST = "keep_first"
    KEEP_LAST = "keep_last"


def half_pixel_shift(base: "Scale", target: "Scale") -> "Translation":
    """Apply half-pixel shift in each downsampled axis."""
    if list(base.spacing.keys()) != list(target.spacing.keys()):
        raise ValueError("Axis mismatch. Cannot compute half-pixel shift between unrelated Scales.")
    shift_items = []
    for axis, target_spacing in target.spacing.items():
        base_spacing = base.spacing[axis]
        if target_spacing > base_spacing:
            # Downsampled - apply half pixel shift
            shift_items.append((axis, 0.5 * (target_spacing - base_spacing)))
        else:
            shift_items.append((axis, 0.0))
    return Translation(shift_items)


_hps: TranslationShiftFunction = half_pixel_shift  # pseudo-registry for grepping


@dataclass(frozen=True, slots=True)
class Scale:
    shape: Shape
    spacing: Optional[Spacing] = None
    unit: Optional[Unit] = None
    translation: Optional[Translation] = None

    def __post_init__(self):
        object.__setattr__(self, "shape", Shape(self.shape))
        if self.spacing is None:
            object.__setattr__(self, "spacing", Spacing.fromkeys(self.shape))
        else:
            object.__setattr__(self, "spacing", Spacing(self.spacing))
        if self.unit is None:
            object.__setattr__(self, "unit", Unit.fromkeys(self.shape))
        else:
            object.__setattr__(self, "unit", Unit(self.unit))
        if self.translation is None:
            object.__setattr__(self, "translation", Translation.fromkeys(self.shape))
        else:
            object.__setattr__(self, "translation", Translation(self.translation))
        if (
            self.shape.keys() != self.spacing.keys()
            or self.shape.keys() != self.unit.keys()
            or self.shape.keys() != self.translation.keys()
        ):
            raise ValueError(
                f"Tried to set up invalid scale: Axiskeys differ "
                f"(shape={list(self.shape.keys())}, "
                f"spacing={list(self.spacing.keys())}, "
                f"translation={list(self.translation.keys())}, "
                f"unit={list(self.unit.keys())})"
            )

    def with_axes(self, axes: OrderedAxes) -> "Scale":
        """Build a Scale with all properties produced by their respective `.with_axes`."""
        if not axes:
            raise ValueError(f"Cannot create empty {self.__class__.__name__}. Attempted reorder to: '{axes}'")
        return Scale(
            shape=self.shape.with_axes(axes),
            spacing=self.spacing.with_axes(axes),
            unit=self.unit.with_axes(axes),
            translation=self.translation.with_axes(axes),
        )

    def has_pixel_size(self):
        return not self.unit.is_default() or not self.spacing.is_default()

    def to_display_string(self, name=""):
        shape = ", ".join(f"{axis}: {size}" for axis, size in self.shape.items())
        name_and_shape = f'"{name}" ({shape})' if name else f"{shape}"
        pixel_size = ""
        if self.has_pixel_size():
            axis_strings = []
            for axis in self.shape.keys():
                if axis == "c":
                    continue
                spacing = self.spacing[axis]
                unit = ""
                if self.unit[axis]:
                    unit = f" {self.unit[axis]}"
                elif axis != "t":
                    unit = " px"
                axis_strings.append(f"{axis}: {spacing:g}{unit}")
            pixel_size = " at pixel size: " + ", ".join(axis_strings)
        return f"{name_and_shape}{pixel_size}"


class _ScaleMapping(ABC, ABCMapping[ScaleKey, ValueType], Generic[ScaleKey, ValueType]):
    """Common base class for Multiscale, BlueprintShapes and BlueprintFactors"""

    def __init__(self, *args, **kwargs):
        self._mapping = OrderedDict(*args, **kwargs)
        if not self._mapping:
            raise ValueError(f"Cannot instantiate empty {self.__class__.__name__}")
        if any(v is None for v in self._mapping.values()):
            raise ValueError(f"None values not allowed. Received: {list(self._mapping.values())}")

    def __repr__(self):
        map_substr = self._mapping.__repr__()[len(type(self._mapping).__name__) :]
        return str(self.__class__.__name__) + map_substr

    def __getitem__(self, key: ScaleKey) -> ValueType:
        if key not in self:
            raise KeyError(f"No such scale: '{key}' (available: {list(self.keys())})")
        return self._mapping[key]

    def __contains__(self, key: ScaleKey):
        return key in self._mapping

    def __iter__(self):
        return iter(self._mapping)

    def __len__(self):
        return len(self._mapping)

    def keys(self):
        return self._mapping.keys()

    def values(self):
        return self._mapping.values()

    def first_value(self) -> ValueType:
        return next(iter(self.values()))

    def items(self):
        return self._mapping.items()

    def __eq__(self, other):
        if isinstance(other, _ScaleMapping):
            return self._mapping == other._mapping
        if isinstance(other, OrderedDict) or isinstance(other, dict):
            return self._mapping == other
        return False

    def copy(self):
        return self.__class__(self._mapping)

    def filter_items(self, keep_func: Callable[[ScaleKey, ValueType], bool]) -> "Self":
        items = [(k, v) for k, v in self.items() if keep_func(k, v)]
        return self.__class__(items)

    def with_keys(
        self,
        keys_pattern_or_func: Union[Sequence[ScaleKey], str, Callable[[int, ScaleKey, "Scale"], ScaleKey]],
    ) -> "Self":
        """
        Assign new scale keys using one of:
        - a sequence of new scale keys (one per current scale, unique)
        - a string format pattern with placeholder for the scale's int index
        - a function that takes the int index, the old scale key, and the Scale object, and returns a new key
        """
        if isinstance(keys_pattern_or_func, str):
            pattern = keys_pattern_or_func
            if pattern.format(0) == pattern:
                raise ValueError(
                    f"Name pattern must contain exactly one placeholder for scale index (received: '{pattern}')"
                )
            items = [(keys_pattern_or_func.format(i), v) for i, v in enumerate(self.values())]
            return self.__class__(items)
        elif callable(keys_pattern_or_func):
            new_keys = self._generate_and_validate_new_keys(keys_pattern_or_func)
            return self.__class__(zip(new_keys, self.values()))
        else:
            new_keys = keys_pattern_or_func
            if len(new_keys) != len(self):
                raise ValueError(
                    f"Must provide a new key for every current key: {list(self.keys())}. Received: {new_keys}"
                )
            if not self._all_unique(new_keys):
                raise ValueError(f"All new scale keys must be unique. Received: {new_keys}")
            return self.__class__(zip(new_keys, self.values()))

    def drop_before(self, key: ScaleKey, inclusive=False) -> "Self":
        keys = list(self.keys())
        if key not in keys:
            raise KeyError(f"No such scale: '{key}' (available: {keys})")

        start_idx = keys.index(key)
        if inclusive:
            start_idx += 1

        items = [(k, v) for k, v in self.items() if k in keys[start_idx:]]
        return self.__class__(items)

    def _generate_and_validate_new_keys(self, keys_pattern_or_func: Callable):
        new_keys = []
        for i, (key, value) in enumerate(self.items()):
            try:
                new_key = keys_pattern_or_func(i, key, value)
            except TypeError as e:
                if "positional argument" in str(e):
                    raise TypeError(
                        "Key-generating function must accept scale's integer index, "
                        "the old scale key, and the corresponding value object, e.g.: "
                        "lambda i, old_key, factor: f\"scale{i}-{factor['x']}\""
                    ) from e
                raise e
            new_keys.append(new_key)
        if not self._all_unique(new_keys):
            raise ValueError(f"All new scale keys must be unique. Generated: {new_keys}")
        return new_keys

    @staticmethod
    def _all_unique(things: Sequence[Hashable]) -> bool:
        seen = set()
        for item in things:
            if item in seen:
                return False
            seen.add(item)
        return True


class _ScaledAxisValues(_ScaleMapping[str, AxisValuesType], Generic[AxisValuesType]):
    """Base class for BlueprintShapes and BlueprintFactors"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._mapping:
            return

        if len(set(self._mapping.keys())) != len(self._mapping.keys()):
            raise ValueError(f"Scale keys must be unique. Received: {list(self._mapping.keys())}")

        axes = next(iter(self._mapping.values())).keys()
        for k, v in self._mapping.items():
            if v.keys() != axes:
                raise ValueError(
                    f"All values must have the same axes. (Expected {axes}, received {v.keys()} for key '{k}')"
                )

    def to_dict(self) -> OrderedDict[ScaleKey, OrderedDict[AxisKey, Union[int, float]]]:
        return OrderedDict([(scale_key, OrderedDict(axis_values)) for scale_key, axis_values in self.items()])

    def _with_values(self, values: Sequence[_AxisValues]) -> "Self":
        return self.__class__(zip(self.keys(), values))

    def with_axes(self, axes: OrderedAxes) -> "Self":
        return self._with_values([value.with_axes(axes) for value in self.values()])

    @staticmethod
    def _resolve_duplicates(
        raw_items: Iterable[Tuple[ScaleKey, _AxisValues]],
        on_duplicate: DuplicatePolicy,
        on_duplicate_prefer: Optional[ScaleKey],
    ) -> List[Tuple[ScaleKey, _AxisValues]]:
        """
        Ensure raw_items contains no duplicate values. Resolve duplicates according to on_duplicate:
        "error": Raise error if there are duplicates.
        "keep_all": Skip (return raw_items as list)
        "keep_first": Keep the first key seen with any particular value.
        "keep_last": Keep the last key seen with any particular value.
        The two "keep" policies can be combined with `on_duplicate_prefer`.
        In this case, if the `on_duplicate_prefer` key is involved in a duplication, it has priority over first/last.
        """
        raw_items = list(raw_items)
        if on_duplicate == DuplicatePolicy.KEEP_ALL:
            return raw_items

        by_value = defaultdict(list)
        for k, v in raw_items:
            by_value[tuple(v.items())].append(k)
        duplicates = {tuple(ks): v for v, ks in by_value.items() if len(ks) > 1}

        if duplicates and on_duplicate == DuplicatePolicy.ERROR:
            raise ValueError(f"Duplicate values not allowed. Collisions: {duplicates}")

        pop_keys = []
        for dup_keys in duplicates:
            if on_duplicate_prefer is not None and on_duplicate_prefer in dup_keys:
                keep = on_duplicate_prefer
            elif on_duplicate == DuplicatePolicy.KEEP_FIRST:
                keep = dup_keys[0]
            elif on_duplicate == DuplicatePolicy.KEEP_LAST:
                keep = dup_keys[-1]
            else:
                raise AssertionError(f"Invalid duplicate scale policy: '{on_duplicate}'")

            pop_keys.extend(k for k in dup_keys if k != keep)

        return [(k, v) for k, v in raw_items if k not in pop_keys]


class BlueprintShapes(_ScaledAxisValues[Shape]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self._mapping.items():
            self._mapping[k] = Shape(v)

    @classmethod
    def from_multiscale(cls, multiscale: "Multiscale") -> "BlueprintShapes":
        return cls([(key, scale.shape) for key, scale in multiscale.items()])

    @classmethod
    def from_multiscale_rescaled(
        cls,
        multiscale: "Multiscale",
        *,
        target_shape: ShapeLike,
        rounding: RoundingMethod,
        source_key: Optional[ScaleKey] = None,
        scaled_axes: Optional[Axes] = None,
    ) -> "BlueprintShapes":
        """
        Build a blueprint rescaling shapes from `multiscale`
        such that the shape at `source_key` matches `target_shape`.
        If no `source_key`, `target_shape` becomes the blueprint's base shape.
        All other shapes are rescaled from `target_shape` according to their relative factor to `source_key`
        """
        if source_key is None:
            source_key = next(iter(multiscale.keys()))
        source_shape = multiscale[source_key].shape

        factors = BlueprintFactors.from_multiscale(multiscale, reference=source_shape)
        if scaled_axes:
            factors = factors.with_identity_except(scaled_axes)

        return factors.to_shapes(reference=target_shape, rounding=rounding)

    @classmethod
    def uniform_steps(
        cls,
        *,
        step: Union[int, float],
        base_shape: Shape,
        rounding: RoundingMethod,
        shape_limit: Optional[ShapeLike] = None,
        only: Optional[Axes] = None,
        max_levels=42,
        name_pattern=DEFAULT_NAME_PATTERN,
        on_duplicate=DuplicatePolicy.KEEP_FIRST,
        on_duplicate_prefer: Optional[ScaleKey] = None,
    ) -> "BlueprintShapes":
        """Generate Blueprint where each scale is a `step` downsampling of the previous scale.
        Applies scaling uniformly to all axes until they become singleton."""
        cls._validate_resampling_step(step)
        if step == 1:
            return cls({name_pattern.format(0): base_shape})

        if only is None:
            only = base_shape.keys()
        only = [a for a in only if a in base_shape]
        if not only:
            return cls({name_pattern.format(0): base_shape})

        if not shape_limit:
            shape_limit = base_shape.with_ones(only)

        cls._validate_shape_limit(base_shape, only, shape_limit, max_levels, step)
        shape_limit = Shape(shape_limit).without_axes_except(base_shape)

        scales_items = []
        for i in range(0, max_levels):
            scale_key = name_pattern.format(i)
            scale_factor = step**i
            scaling = Factor.uniform(base_shape, scale_factor).with_identity_except(only)
            scaled_shape = base_shape.scaled_by(scaling, rounding=rounding)
            scales_items.append((scale_key, scaled_shape))
            if (step > 1 and all(scaled_shape[axis] <= shape_limit[axis] for axis in only)) or (
                step < 1 and all(scaled_shape[axis] >= shape_limit[axis] for axis in only)
            ):
                break
        scales_items = cls._resolve_duplicates(scales_items, on_duplicate, on_duplicate_prefer)
        bp = cls(scales_items)
        return bp.with_keys(name_pattern)

    @classmethod
    def downscale_powers_of_2_xyz(
        cls,
        *,
        base_shape: Shape,
        rounding: RoundingMethod,
        shape_limit: Optional[ShapeLike] = None,
        max_levels: int = 42,
        name_pattern=DEFAULT_NAME_PATTERN,
        on_duplicate=DuplicatePolicy.KEEP_FIRST,
        on_duplicate_prefer: Optional[ScaleKey] = None,
    ):
        return cls.uniform_steps(
            step=2,
            only="xyz",
            base_shape=base_shape,
            rounding=rounding,
            shape_limit=shape_limit,
            max_levels=max_levels,
            name_pattern=name_pattern,
            on_duplicate=on_duplicate,
            on_duplicate_prefer=on_duplicate_prefer,
        )

    def axes(self):
        return self.first_value().keys()

    def scaled_axes(self) -> Tuple[AxisKey, ...]:
        """Axes where shapes differ across scales."""
        if len(self) < 2:
            return ()

        shapes = list(self.values())
        first_shape = shapes[0]
        scaled = []

        for axis in first_shape.keys():
            first_value = first_shape[axis]
            if any(shape[axis] != first_value for shape in shapes[1:]):
                scaled.append(axis)

        return tuple(scaled)

    def to_factors(self, reference: Shape) -> "BlueprintFactors":
        factors = [Shape(reference).scaling_to(scale_shape) for scale_shape in self.values()]
        return BlueprintFactors(zip(self.keys(), factors))

    def apply_to_scale(
        self, base: Scale, *, translation_shift_func: Optional[TranslationShiftFunction] = None
    ) -> "Multiscale":
        if list(self.first_value().keys()) != list(base.shape.keys()):
            raise ValueError(
                f"Cannot apply blueprint with axes {list(self.first_value().keys())} "
                f"to base scale with axes {list(base.shape.keys())}. "
                "Axes must match exactly. Maybe blueprint.with_axes(base.shape) first?"
            )

        scales = []
        for scale_key, target_shape in self.items():
            factor = base.shape.scaling_to(target_shape)
            new_spacing = base.spacing.scaled_by(factor)

            if translation_shift_func is not None:
                target_scale_pre_shift = Scale(
                    shape=target_shape, spacing=new_spacing, unit=base.unit, translation=base.translation
                )
                shift = self._compute_and_validate_shift(translation_shift_func, base, target_scale_pre_shift)
                new_translation = base.translation + shift
            else:
                new_translation = base.translation

            scales.append(
                (scale_key, Scale(shape=target_shape, spacing=new_spacing, unit=base.unit, translation=new_translation))
            )

        return Multiscale(scales)

    def with_sizes(self, other: ShapeLike, axes: Axes):
        return self._with_values([shape.with_values(other, axes) for shape in self.values()])

    @staticmethod
    def _validate_resampling_step(step: Union[int, float]):
        if step <= 0:
            raise ValueError(f"Cannot downsample by a negative step size (received: {step})")

    @staticmethod
    def _validate_shape_limit(
        base_shape: Shape, only: Axes, shape_limit: ShapeLike, max_levels, step: Union[int, float]
    ):
        applicable_limit_axes = [a for a in shape_limit if a in base_shape]
        if not applicable_limit_axes:
            raise ValueError(
                f"Cannot scale to limit if none of the axes in shape_limit "
                f"({list(shape_limit.keys())}) are in base_shape ({list(base_shape.keys())})."
            )
        if step < 1 and set(only) != set(applicable_limit_axes) and not max_levels:
            raise ValueError(
                f"When upscaling, either max_levels must be set, or shape_limit must limit all axes in `only`. "
                f"Received: {only=}, {max_levels=}, {shape_limit=}"
            )
        for axis in only:
            if axis not in applicable_limit_axes:
                continue
            if step > 1 and shape_limit[axis] > base_shape[axis]:
                raise ValueError(f"Cannot limit downsampling to a shape larger than the base (along {axis}).")
            if step < 1 and shape_limit[axis] < base_shape[axis]:
                raise ValueError(f"Cannot limit upsampling to a shape smaller than the base (along {axis}).")

    @staticmethod
    def _compute_and_validate_shift(translation_shift_func, base, target_scale_pre_shift):
        try:
            shift = translation_shift_func(base, target_scale_pre_shift)
        except TypeError as e:
            if "argument" in str(e):
                raise TypeError(
                    "translation_shift_func must accept two positional arguments (base and target scale). "
                    "See clearscale.half_pixel_shift for an example implementation."
                ) from e
            raise e
        if not isinstance(shift, Translation):
            raise TypeError(
                f"translation_shift_func must return a Translation, got {type(shift).__name__}. "
                "See clearscale.half_pixel_shift for an example implementation."
            )
        if list(shift.keys()) != list(target_scale_pre_shift.shape.keys()):
            raise ValueError(
                f"translation_shift_func returned Translation with axes {list(shift.keys())}, "
                f"but target scale has axes {list(target_scale_pre_shift.shape.keys())}."
            )
        return shift


class BlueprintFactors(_ScaledAxisValues[Factor]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self._mapping.items():
            self._mapping[k] = Factor(v)

    @classmethod
    def from_shapes(cls, shapes: Mapping[ScaleKey, ShapeLike], reference: Shape) -> "BlueprintFactors":
        return BlueprintShapes(shapes).to_factors(reference)

    @classmethod
    def from_multiscale(cls, multiscale: "Multiscale", reference: Shape) -> "BlueprintFactors":
        return BlueprintShapes.from_multiscale(multiscale).to_factors(reference)

    def axes(self):
        return self.first_value().keys()

    @property
    def scaled_axes(self) -> Tuple[AxisKey, ...]:
        """Axes where any factor is not 1.0."""
        if len(self) < 2:
            return ()

        scaled = set()
        for factor in self.values():
            scaled.update(axis for axis, value in factor.items() if value != 1.0)

        all_axes = next(iter(self.values())).keys()
        return tuple(axis for axis in all_axes if axis in scaled)

    def to_shapes(self, reference: ShapeLike, *, rounding: RoundingMethod) -> BlueprintShapes:
        ref = Shape(reference)
        shapes = [ref.scaled_by(scale_factor, rounding=rounding) for scale_factor in self.values()]
        return BlueprintShapes(zip(self.keys(), shapes))

    def apply_to_scale(self):
        return NotImplementedError()

    def with_identity(self, axes: Axes) -> "Self":
        return self._with_values([factor.with_identity(axes) for factor in self.values()])

    def with_identity_except(self, axes: Axes) -> "Self":
        return self._with_values([factor.with_identity_except(axes) for factor in self.values()])


def _random_multiscale_name() -> str:
    return f"ms-{uuid.uuid4()}"


class Multiscale(_ScaleMapping[str, Scale], TransformGraphNode):
    _transform_graph: _TransformGraph
    """Transform graph that by default consists only of one isolated node: _intrinsic_ref."""
    _intrinsic_ref: CoordinateSystemRef[CoordinateSystem]
    """The system in which the Scales' shape, spacing, translation etc. are correct."""

    def __init__(
        self,
        *args,
        _transform_graph: Optional[_TransformGraph] = None,
        _intrinsic_ref: Optional[CoordinateSystemRef[CoordinateSystem]] = None,
        **kwargs,
    ):
        """
        Multiscales can be constructed from a `scale_key : Scale` mapping, but this should be avoided.
        Multiscale objects should reflect either metadata read from a file (`.from_ome_zarr`, `.from_precomputed`),
        or expand a single Scale according to a scaling blueprint (`.from_shapes`, `.from_factors`).
        """
        super().__init__(*args, **kwargs)
        for key, scale in self._mapping.items():
            if scale.shape.keys() != self.axes():
                raise ValueError(
                    f"All Scales must have identical axes. Scale at '{key}' has {list(scale.shape.keys())}"
                )

        if _intrinsic_ref is None:
            if _transform_graph:
                raise AssertionError("Must specify _intrinsic_ref when _transform_graph is given.")
            self._transform_graph = self._make_single_system_graph()
            self._intrinsic_ref = next(iter(self._transform_graph.isolated_system_refs))
        else:
            if _intrinsic_ref not in _transform_graph.all_system_refs:
                raise AssertionError("_intrinsic_ref must be inside _transform_graph")
            self._transform_graph = _transform_graph or self._make_single_system_graph(_intrinsic_ref)
            self._intrinsic_ref = _intrinsic_ref

    @staticmethod
    def from_shapes(
        blueprint: BlueprintShapes,
        *,
        base: Optional[Scale] = None,
        translation_shift_func: Optional[TranslationShiftFunction] = None,
    ):
        bp = BlueprintShapes(blueprint)
        base = base or Scale(shape=bp.first_value())
        return bp.apply_to_scale(base, translation_shift_func=translation_shift_func)

    @staticmethod
    def from_factors(blueprint: BlueprintFactors, base: Scale, *args, **kwargs):
        return blueprint.apply_to_scale(base, *args, **kwargs)

    @classmethod
    def from_ome_zarr(
        cls,
        multiscale_dict: _ome_zarr.OME_ZARR_MULTISCALE,
        get_shape: Callable[[str], Tuple[int, ...]],
    ):
        # TODO: for really perfect round-tripping, get_shape needs to become get_array, and to_ome_zarr needs write_array
        #  Otherwise, arrayCoordinateSystem metadata in the array zarr.json can be lost.
        _ome_zarr.validate_multiscales_dict(multiscale_dict)
        intrinsic_system_name = _ome_zarr.intrinsic_system_name_from_multiscale(multiscale_dict)
        if intrinsic_system_name:
            graph, intrinsic_system_ref = _ome_zarr.multiscale_graph_from_transforms(
                multiscale_dict, name=intrinsic_system_name
            )
            global_transforms = None
        else:
            intrinsic_system_name = _random_multiscale_name()
            multiscale_tf_list = multiscale_dict.get("coordinateTransformations")
            global_transforms = _ome_zarr.MultiscaleTransforms.from_list(multiscale_tf_list)
            if multiscale_tf_list and global_transforms is None:
                warnings.warn("Pixel size metadata at multiscale-level was invalid.")
            graph, intrinsic_system_ref, global_transforms = _ome_zarr.multiscale_graph_from_legacy(
                multiscale_dict, name=intrinsic_system_name, global_transforms=global_transforms
            )
        axis_keys = list(intrinsic_system_ref.owner.axes())
        unit = intrinsic_system_ref.owner.get_unit()
        datasets = multiscale_dict["datasets"]
        base_shape = None
        scales_items = []
        for dataset in datasets:
            scale_key = dataset["path"]
            scale_shape = Shape(zip(axis_keys, get_shape(scale_key)))
            if base_shape is None:
                base_shape = scale_shape
            transformations = dataset.get("coordinateTransformations")
            dataset_transforms = _ome_zarr.MultiscaleTransforms.from_list(transformations)
            if dataset_transforms is None:
                # OME-Zarr up to v0.3 didn't have coordinateTransformations
                scale_factor = base_shape.scaling_to(scale_shape)
                scale_spacing = Spacing.identity(axis_keys).scaled_by(scale_factor)
                scale_translation = None
            else:
                if global_transforms is not None:
                    dataset_transforms = TransformSequence((dataset_transforms, global_transforms)).collapsed(
                        raise_uncollapsed=True
                    )
                scale_spacing = dataset_transforms.scale.to_spacing(axis_keys)
                scale_translation = (
                    dataset_transforms.translation.to_translation(axis_keys) if dataset_transforms.translation else None
                )
            scales_items.append(
                (
                    scale_key,
                    Scale(shape=scale_shape, spacing=scale_spacing, translation=scale_translation, unit=unit),
                )
            )
        return cls(scales_items, _transform_graph=graph, _intrinsic_ref=intrinsic_system_ref)

    @classmethod
    def from_precomputed(cls, info_dict: _precomputed.INFO_DICT):
        _precomputed.validate_info_dict(info_dict)
        scales_list = info_dict["scales"]
        num_channels = info_dict.get("num_channels", 1)
        axis_keys = ["c", "z", "y", "x"]  # Precomputed is always czyx (x varies fastest)

        scales_items = []
        for scale_dict in scales_list:
            scale_key = scale_dict["key"]

            size = scale_dict["size"]
            if len(size) != 3:
                raise ValueError(f"Scale {scale_key!r} must have 'size' as [x, y, z]")
            shape = Shape(zip(axis_keys, [num_channels] + list(reversed(size))))

            resolution = scale_dict["resolution"]
            if len(resolution) != 3:
                raise ValueError(f"Scale {scale_key!r} must have 'resolution' as [x, y, z]")
            spacing = Spacing(zip(axis_keys, [1.0] + list(reversed(resolution))))

            voxel_offset = scale_dict.get("voxel_offset", [0, 0, 0])
            if len(voxel_offset) != 3:
                warnings.warn(f"Scale {scale_key!r} has invalid voxel_offset. Using [0, 0, 0].")
                voxel_offset = [0, 0, 0]
            offset = PixelOffset(zip(axis_keys, [0] + list(reversed(voxel_offset))))
            translation = offset.to_physical(spacing)

            unit = Unit(zip(axis_keys, ["", "nm", "nm", "nm"]))

            scale = Scale(shape, spacing, unit, translation)
            scales_items.append((scale_key, scale))

        return cls(scales_items)

    def axes(self) -> Iterable[AxisKey]:
        return self.first_value().shape.keys()

    def scaled_axes(self) -> Tuple[AxisKey, ...]:
        """Axes where spacings differ across scales."""
        if len(self) < 2:
            return ()

        spacings = list(scale.spacing for scale in self.values())
        first_spacing = spacings[0]
        scaled = []

        for axis in first_spacing.keys():
            first_value = first_spacing[axis]
            if any(spacing[axis] != first_value for spacing in spacings[1:]):
                scaled.append(axis)

        return tuple(scaled)

    @cached_property
    def keys_by_shape(self) -> Mapping[Shape, Tuple[ScaleKey, ...]]:
        grouped = defaultdict(list)
        for key, scale in self.items():
            grouped[scale.shape].append(key)
        return {shape: tuple(keys) for shape, keys in grouped.items()}

    def to_ome_zarr(
        self,
        *,
        version: Literal["0.4", "0.5", "rfc-5"],
        name: Optional[str] = None,
        axis_types: Union[None, Literal["infer"], Mapping[str, Literal["space", "time", "channel"]]] = None,
    ) -> Dict[str, Any]:
        if version not in ("0.4", "0.5", "rfc-5"):
            raise ValueError("Cannot write OME-Zarr versions other than 0.4, 0.5 and rfc-5.")
        _ome_zarr.validate_multiscale(self)
        result = {"version": version, "datasets": []}

        if name:
            result["name"] = name

        if version not in PRE_TRANSFORMS_VERSIONS:
            result.update(self._transform_graph.to_ome_zarr(version=version))
            for key, scale in self.items():
                dataset = _ome_zarr.build_dataset_dict(
                    version, key, scale.spacing, scale.translation, self._intrinsic_ref
                )
                result["datasets"].append(dataset)
            return result

        intrinsic_system_dict = self._intrinsic_ref.owner.to_ome_zarr(
            name="", version=version, axis_types=axis_types, unit=self.first_value().unit
        )
        result["axes"] = intrinsic_system_dict["axes"]

        legacy_tfs = []
        if self._transform_graph:
            legacy_tfs = [t for t in self._transform_graph.transforms if isinstance(t, _ome_zarr.MultiscaleTransforms)]
        if not legacy_tfs:
            # "Clean" legacy multiscale without global/multiscale-level transforms
            for key, scale in self.items():
                dataset = _ome_zarr.build_dataset_dict(version, key, scale.spacing, scale.translation)
                result["datasets"].append(dataset)
            return result

        # Legacy compatibility with convention where constant parts of the dataset scale transform were
        # placed on multiscale-level. (Or other arbitrary custom multiscale-level transforms)
        assert (
            len(legacy_tfs) <= 1
        ), f"Dev error: More than one multiscale-level transform in {self._transform_graph.transforms}"
        result["coordinateTransformations"] = legacy_tfs[0].to_ome_zarr(version, for_scene=False)
        global_scale = legacy_tfs[0].scale.to_spacing()
        global_translation = Translation.identity(list(global_scale.keys()))
        if legacy_tfs[0].translation:
            global_translation = legacy_tfs[0].translation.to_translation()
        # Multiscale.from_ome_zarr collapses the global transforms into each Scale so that
        # Scale.spacing/.translation are correct independent of their containing Multiscale.
        # That means we have to decompose them back out for perfect metadata round-trip.
        for key, scale in self.items():
            dataset_scale = scale.spacing.scaled_by(Factor(global_scale).inverted())
            dataset_translation = scale.translation - global_translation
            dataset = _ome_zarr.build_dataset_dict(version, key, dataset_scale, dataset_translation)
            result["datasets"].append(dataset)
        return result

    def _make_single_system_graph(
        self, sys_ref: Optional[CoordinateSystemRef[CoordinateSystem]] = None
    ) -> _TransformGraph:
        if sys_ref is None:
            intrinsic_sys = CoordinateSystem.without_semantics(list(self.axes()))
            intrinsic_name = _random_multiscale_name()
            sys_ref = intrinsic_sys.as_ref(intrinsic_name)
        return _TransformGraph.single_isolated_system(sys_ref)

    def as_ref(self, name: CoordinateSystemName) -> CoordinateSystemRef["Multiscale"]:
        return CoordinateSystemRef(name=name, owner=self)

    def _get_interface_transform(self):
        """Allows a scene to traverse into this subgraph"""
        return IdentityTransform(source=self._intrinsic_ref, target=self.as_ref(self._intrinsic_ref.name))
