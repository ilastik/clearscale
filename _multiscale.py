import uuid
import warnings
from abc import ABC
from collections import OrderedDict, defaultdict
from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass
from enum import StrEnum
from functools import wraps
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
    FrozenSet,
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
    _UnresolvedRef,
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


class _DuplicatePolicy(StrEnum):
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
        raw_items: Iterable[Tuple[ScaleKey, _AxisValues]], on_duplicate: _DuplicatePolicy, on_duplicate_prefer
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
        if on_duplicate == _DuplicatePolicy.KEEP_ALL:
            return raw_items

        by_value = defaultdict(list)
        for k, v in raw_items:
            by_value[tuple(v.items())].append(k)
        duplicates = {tuple(ks): v for v, ks in by_value.items() if len(ks) > 1}

        if duplicates and on_duplicate == _DuplicatePolicy.ERROR:
            raise ValueError(f"Duplicate values not allowed. Collisions: {duplicates}")

        pop_keys = []
        for dup_keys in duplicates:
            if on_duplicate_prefer is not None and on_duplicate_prefer in dup_keys:
                keep = on_duplicate_prefer
            elif on_duplicate == _DuplicatePolicy.KEEP_FIRST:
                keep = dup_keys[0]
            elif on_duplicate == _DuplicatePolicy.KEEP_LAST:
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
    def uniform_steps(
        cls,
        *,
        step: Union[int, float],
        base_shape: Shape,
        rounding: RoundingMethod,
        shape_limit: Optional[ShapeLike] = None,
        only: Optional[Axes] = None,
        max_levels: Optional[int] = 42,
        name_pattern=DEFAULT_NAME_PATTERN,
        on_duplicate=_DuplicatePolicy.KEEP_FIRST,
        on_duplicate_prefer: ScaleKey = None,
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
        shape_limit = Shape(shape_limit).with_axes_preserving_order(base_shape)

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
        on_duplicate=_DuplicatePolicy.KEEP_FIRST,
        on_duplicate_prefer: ScaleKey = None,
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

    def scaled_axes(self) -> tuple[AxisKey, ...]:
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
        self, base: Scale, *, translation_shift_func: Union[None, TranslationShiftFunction]
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
    def scaled_axes(self) -> tuple[AxisKey, ...]:
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


class Multiscale(_ScaleMapping[str, Scale], TransformGraphNode):
    transform_graph: _TransformGraph
    intrinsic_ref: CoordinateSystemRef
    """The system in which the Scales' shape, spacing, translation etc. are correct."""

    def __init__(
        self,
        *args,
        transform_graph: Optional[_TransformGraph] = None,
        intrinsic_ref: Optional[CoordinateSystemRef] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        for key, scale in self._mapping.items():
            if scale.shape.keys() != self.axes():
                raise ValueError(
                    f"All Scales must have identical axes. Scale at '{key}' has {list(scale.shape.keys())}"
                )

        self.transform_graph = transform_graph or self._get_default_graph()
        self.intrinsic_ref = intrinsic_ref or next(iter(self.transform_graph.isolated_system_refs))

    @staticmethod
    @wraps(BlueprintShapes.apply_to_scale)
    def from_shapes(blueprint: BlueprintShapes, base: Scale, *args, **kwargs):
        return blueprint.apply_to_scale(base, *args, **kwargs)

    @staticmethod
    @wraps(BlueprintFactors.apply_to_scale)
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
        intrinsic_system_name = _ome_zarr.get_intrinsic_from_multiscale(multiscale_dict)
        is_v06_or_newer = bool(intrinsic_system_name)
        if is_v06_or_newer:
            graph = _TransformGraph.from_ome_zarr(
                multiscale_dict.get("coordinateTransformations"), multiscale_dict.get("coordinateSystems")
            )
            potential_intrinsics = [ref for ref in graph.all_system_refs if ref.name == intrinsic_system_name]
            if len(potential_intrinsics) != 1:
                raise ValueError(
                    "Invalid OME-Zarr multiscale metadata: Expected exactly one coordinate system named "
                    f"{intrinsic_system_name!r}. Received: {multiscale_dict}"
                )
            intrinsic_system_ref = potential_intrinsics[0]
            axis_keys = list(intrinsic_system_ref.owner.keys())
            unit = intrinsic_system_ref.owner.get_unit()
            datasets = multiscale_dict["datasets"]
            scales_items = []
            for scale in datasets:
                # TODO: Now with proper Transforms, we should be able to do this a bit more neatly...
                scale_key = scale["path"]
                dataset_transforms = _ome_zarr.validate_transforms(scale.get("coordinateTransformations"))
                scales_items.append(
                    (
                        scale_key,
                        Scale(
                            shape=Shape(zip(axis_keys, get_shape(scale_key))),
                            spacing=_ome_zarr.compute_spacing(axis_keys, scale_key, None, dataset_transforms),
                            translation=_ome_zarr.compute_translation(axis_keys, None, dataset_transforms),
                            unit=unit,
                        ),
                    )
                )
        else:
            intrinsic_system = CoordinateSystem.from_ome_zarr(multiscale_dict)
            intrinsic_system_name = f"multiscale-{uuid.uuid4()}"
            intrinsic_system_ref = intrinsic_system.as_ref(intrinsic_system_name)
            axis_keys = list(intrinsic_system.keys())
            unit = intrinsic_system.get_unit()
            graph = None
            multiscale_transforms_raw = multiscale_dict.get("coordinateTransformations")
            multiscale_transforms = _ome_zarr.validate_transforms(multiscale_transforms_raw)
            if multiscale_transforms is not None:
                if not isinstance(multiscale_transforms, tuple):
                    warnings.warn("Pixel resolution metadata at pyramid level was invalid.")
                else:
                    mock_ref = _UnresolvedRef(name=f"{intrinsic_system_name}-intermediate")
                    transform = _ome_zarr.LegacyMultiscaleTransforms.from_ome_zarr(multiscale_transforms_raw)
                    t_bound = transform.bound(source=mock_ref, target=intrinsic_system_ref)
                    graph = _TransformGraph([t_bound])
            datasets = multiscale_dict["datasets"]
            scales_items = []
            for scale in datasets:
                scale_key = scale["path"]
                scale_transforms_raw = scale.get("coordinateTransformations")
                dataset_transforms = _ome_zarr.validate_transforms(scale_transforms_raw)
                scales_items.append(
                    (
                        scale_key,
                        Scale(
                            shape=Shape(zip(axis_keys, get_shape(scale_key))),
                            spacing=_ome_zarr.compute_spacing(
                                axis_keys, scale_key, multiscale_transforms, dataset_transforms
                            ),
                            translation=_ome_zarr.compute_translation(
                                axis_keys, multiscale_transforms, dataset_transforms
                            ),
                            unit=unit,
                        ),
                    )
                )
        return cls(scales_items, transform_graph=graph, intrinsic_ref=intrinsic_system_ref)

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

    def axes(self):
        return self.first_value().shape.keys()

    def scaled_axes(self) -> tuple[AxisKey, ...]:
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

    def to_ome_zarr(
        self,
        *,
        version: Literal["0.4", "0.5", "rfc-5"],
        name: Optional[str] = None,
        axis_types: Union[None, Literal["infer"], Mapping[str, Literal["space", "time", "channel"]]] = None,
    ) -> Dict[str, Any]:
        _ome_zarr.validate_multiscale(self)
        result = {"version": version, "datasets": []}

        if name:
            result["name"] = name

        if version == "rfc-5":
            result.update(self.transform_graph.to_ome_zarr(version=version))
        elif self.intrinsic_ref:
            ome_zarr_0_6_sys = self.intrinsic_ref.owner.to_ome_zarr(
                name="", version=version, axis_types=axis_types, unit=self.first_value().unit
            )
            result["axes"] = ome_zarr_0_6_sys["axes"]
            if self.transform_graph:
                legacy_tfs = [
                    t for t in self.transform_graph.transforms if isinstance(t, _ome_zarr.LegacyMultiscaleTransforms)
                ]
                assert len(legacy_tfs) <= 1, (
                    "Dev error: More than one multiscale-level transform tuple " f"in {self.transform_graph.transforms}"
                )
                if legacy_tfs:
                    result["coordinateTransformations"] = legacy_tfs[0].to_ome_zarr(version, for_scene=False)
                    global_scale = legacy_tfs[0].scale.spacing
                    global_translation = Translation.identity(list(global_scale.keys()))
                    if legacy_tfs[0].translation:
                        global_translation = legacy_tfs[0].translation.translation
                    for key, scale in self.items():
                        dataset_scale = scale.spacing.scaled_by(Factor(global_scale))
                        dataset_translation = scale.translation - global_translation
                        dataset = _ome_zarr.build_dataset_dict(key, dataset_scale, dataset_translation)
                        result["datasets"].append(dataset)
                else:
                    for key, scale in self.items():
                        dataset = _ome_zarr.build_dataset_dict(key, scale.spacing, scale.translation)
                        result["datasets"].append(dataset)
            else:
                for key, scale in self.items():
                    dataset = _ome_zarr.build_dataset_dict(key, scale.spacing, scale.translation)
                    result["datasets"].append(dataset)
        else:
            first_scale = self.first_value()
            axes = list(first_scale.shape.keys())
            ome_axes = _ome_zarr.build_axis_dicts(axes, first_scale.unit, axis_types)
            result["axes"] = ome_axes
            # Either have no multiscale-level transforms, or impossible to reconstruct now.
            # Export scale meta to datasets as-is.
            for key, scale in self.items():
                dataset = _ome_zarr.build_dataset_dict(key, scale.spacing, scale.translation)
                result["datasets"].append(dataset)
        return result

    @property
    def coordinate_system(self) -> CoordinateSystem:
        assert isinstance(self.intrinsic_ref.owner, CoordinateSystem), "should always have a coord system"
        return self.intrinsic_ref.owner

    def _get_default_graph(self):
        intrinsic_sys = CoordinateSystem.without_semantics(list(self.axes()))
        intrinsic_name = f"ms-{uuid.uuid4()}"
        return _TransformGraph(
            transforms=frozenset(), isolated_system_refs=frozenset((intrinsic_sys.as_ref(intrinsic_name),))
        )

    def as_ref(self, name: CoordinateSystemName):
        return CoordinateSystemRef(name, self)

    def get_interface_transform(self):
        """Allows a scene to traverse into this subgraph"""
        return IdentityTransform(source=self.intrinsic_ref, target=self.as_ref(self.intrinsic_ref.name))
