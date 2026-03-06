import re
import warnings
from dataclasses import dataclass, field
from typing import Union, Literal, Mapping, Dict, List, Any, Optional, Tuple

from lazyflow.utility.io_util.clearscale import Translation, Unit, Spacing, Factor
from lazyflow.utility.io_util.clearscale._axis_values import OrderedAxes, AxisKey
from lazyflow.utility.io_util.clearscale._transforms import (
    TransformSequence,
    ScaleTransform,
    TranslationTransform,
    _TransformGraph,
    CoordinateSystemRef,
    CoordinateSystem,
    _UnresolvedRef,
)

####
# Reading
####


OME_ZARR_DATASET = Dict[Literal["path", "coordinateTransformations"], Any]  # single dataset (= scale)
OME_ZARR_MULTISCALE = Dict[  # single multiscales entry of a json-validated OME-Zarr zattrs (any version)
    # The spec allows for multiple multiscales, but in practice we only ever see one.
    Literal["axes", "datasets", "version", "coordinateTransformations", "name", "coordinateSystems"],
    Union[List[Dict], List[OME_ZARR_DATASET], str],
]


@dataclass(frozen=True, slots=True)
class LegacyMultiscaleTransforms(TransformSequence):
    """Special class for legacy interpretation of 'coordinateTransformations'
    in OME-Zarr 0.4 and 0.5 multiscale metadata."""

    scale_transform: ScaleTransform = field(default=None)
    translation_transform: Optional[TranslationTransform] = field(default=None)

    def __post_init__(self):
        if self.scale_transform is None:
            raise ValueError("LegacyMultiscaleTransforms requires a scale transform.")
        transforms = (
            (self.scale_transform, self.translation_transform)
            if self.translation_transform
            else (self.scale_transform,)
        )
        object.__setattr__(self, "transforms", transforms)
        TransformSequence.__post_init__(self)

    @property
    def scale(self) -> ScaleTransform:
        return self.transforms[0]  # noqa

    @property
    def translation(self) -> Optional[TranslationTransform]:
        return self.transforms[1] if len(self.transforms) == 2 else None

    @classmethod
    def from_ome_zarr(cls, ome_transforms: List[Dict]) -> "LegacyMultiscaleTransforms":
        ome_dict = {"type": "sequence", "transformations": ome_transforms}
        seq = TransformSequence.from_ome_zarr(ome_dict)
        if not seq.is_valid_for_ome_zarr_multiscale():
            raise ValueError(
                "Invalid coordinateTransformations metadata: Expected exactly one 'scale' and "
                f"optionally one 'translation' transform. Received: {ome_transforms}"
            )
        return cls(
            scale_transform=seq.transforms[0],
            translation_transform=seq.transforms[1] if len(seq.transforms) == 2 else None,
        )


class InvalidTransformationError(ValueError):
    pass


def validate_multiscales_dict(raw: Dict):
    """Light top-level checks. coordinateTransformations are validated later."""
    version = raw.get("version")
    if version not in ("0.1", "0.2", "0.3", "0.4", "0.5", "rfc-5"):
        v = raw.get("version")
        warnings.warn(f"Attempting to parse unknown OME-Zarr version '{v}'. This might break...")

    if "datasets" not in raw or not raw["datasets"]:
        raise ValueError(f"Invalid OME-Zarr datasets metadata: no datasets. Received:\n{raw}")
    if (
        version is not None
        and version not in ("0.1", "0.2", "0.3")
        and any(not d["coordinateTransformations"] for d in raw["datasets"])
    ):
        raise ValueError(f"Invalid OME-Zarr datasets metadata: datasets without transformations. Received:\n{raw}")


def axes_from_multiscale(multiscale: OME_ZARR_MULTISCALE) -> List[str]:
    if "axes" in multiscale:
        ome_axes = multiscale["axes"]
        if "name" in ome_axes[0]:
            # v0.4: spec["axes"] requires name, recommends type and unit; like:
            # [
            #   {'name': 'c', 'type': 'channel'},
            #   {'name': 'y', 'type': 'space', 'unit': 'nanometer'},
            #   {'name': 'x', 'type': 'space', 'unit': 'nanometer'}
            # ]
            axis_keys = [d["name"] for d in ome_axes]
        else:
            # v0.3: ['t', 'c', 'y', 'x']
            axis_keys = ome_axes
    else:
        # v0.1 and v0.2 did not allow variable axes
        axis_keys = ["t", "c", "z", "y", "x"]
    return axis_keys


def units_from_multiscale(multiscale: OME_ZARR_MULTISCALE) -> Unit:
    axis_keys = axes_from_multiscale(multiscale)
    if "axes" in multiscale and "name" in multiscale["axes"][0]:
        # v0.4: Each axis entry may contain a unit key
        units = [a["unit"] if "unit" in a else "" for a in multiscale["axes"]]
    else:
        # v0.1 to v0.3 did not provide a standard for keeping unit metadata
        units = ["" for _ in axis_keys]
    return Unit(zip(axis_keys, units))


def intrinsic_system_name_from_multiscale(multiscale: OME_ZARR_MULTISCALE) -> Optional[str]:
    transforms: Optional[List[Dict]] = multiscale["datasets"][0].get("coordinateTransformations")
    if not transforms:
        return None
    return transforms[0].get("output")


def multiscale_graph_from_transforms(
    multiscale: OME_ZARR_MULTISCALE, *, name: str
) -> Tuple[_TransformGraph, CoordinateSystemRef[CoordinateSystem]]:
    try:
        graph = _TransformGraph.from_ome_zarr(
            multiscale.get("coordinateTransformations"), multiscale.get("coordinateSystems")
        )
        potential_intrinsics = [ref for ref in graph.all_system_refs if ref.name == name]
        if len(potential_intrinsics) != 1:
            raise ValueError(
                "Invalid OME-Zarr multiscale metadata: Expected exactly one coordinate system named "
                f"{name!r}. Received: {multiscale}"
            )
        intrinsic_system_ref = potential_intrinsics[0]
        return graph, intrinsic_system_ref
    except ValueError as e:
        try:
            # Best effort: Is there any coordinate system we can use at all?
            name_matches = [sys_d for sys_d in multiscale["coordinateSystems"] if sys_d["name"] == name]
            if name_matches:
                intrinsic_sys = CoordinateSystem.from_ome_zarr(name_matches[0])
            elif multiscale["coordinateSystems"]:
                intrinsic_sys = CoordinateSystem.from_ome_zarr(multiscale["coordinateSystems"][0])
            else:
                raise ValueError()
        except (KeyError, ValueError):
            raise e
        warnings.warn(
            "Invalid coordinateTransformations and/or coordinateSystems metadata. Proceeding without. "
            f"Error: {str(e)}"
            f"Received: {multiscale}"
        )
        intrinsic_system_ref = intrinsic_sys.as_ref(name)
        graph = _TransformGraph.single_isolated_system(intrinsic_system_ref)
        return graph, intrinsic_system_ref


def multiscale_graph_from_legacy(
    multiscale: OME_ZARR_MULTISCALE,
    *,
    name: str,
    validated_multiscale_transforms: Optional["ValidTransformations"] = None,
) -> Tuple[_TransformGraph, CoordinateSystemRef[CoordinateSystem]]:
    intrinsic_system = CoordinateSystem.from_ome_zarr(multiscale)
    intrinsic_system_ref = intrinsic_system.as_ref(name)
    graph = _TransformGraph.single_isolated_system(intrinsic_system_ref)
    if validated_multiscale_transforms is not None:
        # Store the multiscale-level transforms as a transform to a non-existent mock system.
        # This allows Multiscale.to_ome_zarr to divide/subtract them back out of Scale.spacing/.translation
        # for perfect metadata round-trip.
        mock_ref = _UnresolvedRef(name=f"{name}-intermediate")
        transform = LegacyMultiscaleTransforms.from_ome_zarr(multiscale.get("coordinateTransformations"))
        bound_transform = transform.bound(source=mock_ref, target=intrinsic_system_ref)
        graph = _TransformGraph([bound_transform])
    return graph, intrinsic_system_ref


@dataclass(frozen=True)
class Transformation:
    """Used by OME-Zarr export to adjust export metadata according to input."""

    type: Literal["scale", "translation"]
    values: Optional[List[float]]

    @classmethod
    def from_json(cls, json_data: Dict) -> "Transformation":
        """Expected dicts look like
        {
          "type": Literal["scale", "translation"]
          and EITHER "scale": List[number] OR "translation": List[number]
        }
        Unfortunately, the spec is internally inconsistent, so there is a chance that we may encounter
        a coordinateTransformation with a "path" key instead of "scale" or "translation"; and possibly
        coordinateTransformations with "type": "identity".
        Afaik, none of the more popular converters/writers do this.
        """
        if (
            json_data["type"] not in ("scale", "translation")
            or ("scale" not in json_data and "translation" not in json_data)
            or "path" in json_data
        ):
            raise InvalidTransformationError()
        # Could raise KeyError for real nonsense like {"type": "scale", "translation": [0, 0]}
        return cls(type=json_data["type"], values=json_data[json_data["type"]])


ValidTransformations = Tuple[Transformation, Optional[Transformation]]
"""tuple(scale_transform, Optional[translation_transform])"""

TransformationsOrError = Union[ValidTransformations, InvalidTransformationError]


def validate_transforms(
    coordinate_transformations: Optional[List[Dict[str, Union[str, List[float]]]]],
) -> Union[None, ValidTransformations, InvalidTransformationError]:
    """
    Resolves the OME-Zarr spec's inconsistency in the coordinateTransformations field.
    Avoids raising errors because valid metadata are not required to load and work with the data.
    Distinguishes between None and invalid transformations so that caller can warn on the latter.
    Returns:
    - None if input was None (allowed for multiscale_transformations)
    - Tuple of scale transform and optionally translation transform if valid
    - InvalidTransformationError if invalid (e.g. not None but also no scale transform present)
    Inattentive writers might produce invalid transforms, depending on what part of the spec they read.
    The Transformations spec [1] allows for "identity" transforms and arbitrary numbers of transforms,
    but the Multiscales spec [2] only allows exactly one "scale", optionally followed by one "translation"
    transform.
    The "official" validator's schema [3] implements neither of these rules exactly :) It instead allows
    for exactly one "scale" transform, plus an arbitrary number of "translation" transforms, in any order.
    But this, plus the example at the start of the OME-Zarr spec, make a clear enough indicator that
    "one scale + one optional translation" is the convention, and all public datasets conform to this.
    To be graceful, we'll accept the first scale and translation.
    [1] https://ngff.openmicroscopy.org/latest/index.html#trafo-md
    [2] https://ngff.openmicroscopy.org/latest/index.html#multiscale-md
    [3] https://github.com/ome/ngff/blob/1383ce6218539baf9fe4350c46d992f2dbfe7af1/0.4/schemas/image.schema#L167
    """
    if coordinate_transformations is None:
        return None
    if not isinstance(coordinate_transformations, list) or not coordinate_transformations:
        return InvalidTransformationError()
    scale_transform = translation_transform = None
    for t in coordinate_transformations:
        try:
            transform = Transformation.from_json(t)
        except (InvalidTransformationError, KeyError):
            continue
        if scale_transform is None and transform.type == "scale":
            scale_transform = transform
        if translation_transform is None and transform.type == "translation":
            translation_transform = transform
    return (scale_transform, translation_transform) if scale_transform else InvalidTransformationError()


def combine_spacings(
    axis_keys: List[str],
    dataset_path: str,
    multiscale_transforms: Union[None, ValidTransformations, InvalidTransformationError],
    dataset_transforms: Union[None, ValidTransformations, InvalidTransformationError],
) -> Spacing:
    def has_valid_resolution(transforms):
        return isinstance(transforms, tuple) and transforms[0].values and len(transforms[0].values) == len(axis_keys)

    if not has_valid_resolution(dataset_transforms):
        warnings.warn(f"Missing or invalid pixel resolution metadata for dataset {dataset_path}.")
        return Spacing.fromkeys(axis_keys)

    dataset_resolution = dataset_transforms[0].values

    if not has_valid_resolution(multiscale_transforms):
        return Spacing(zip(axis_keys, dataset_resolution))

    spacing = Spacing(zip(axis_keys, multiscale_transforms[0].values))
    scale = Factor([(k, v) for k, v in zip(axis_keys, dataset_resolution) if v != 0])
    return spacing.scaled_by(scale)


def combine_translations(
    axis_keys: List[str],
    multiscale_transforms: Union[None, ValidTransformations, InvalidTransformationError],
    dataset_transforms: Union[None, ValidTransformations, InvalidTransformationError],
) -> Translation:
    def has_valid_translation(transforms):
        return (
            isinstance(transforms, tuple)
            and transforms[1] is not None
            and transforms[1].values is not None
            and len(transforms[1].values) == len(axis_keys)
        )

    dataset_translation = Translation.identity(axis_keys)
    if has_valid_translation(dataset_transforms):
        dataset_translation = Translation(zip(axis_keys, dataset_transforms[1].values))

    multiscale_translation = Translation.identity(axis_keys)
    if has_valid_translation(multiscale_transforms):
        multiscale_translation = Translation(zip(axis_keys, multiscale_transforms[1].values))

    return multiscale_translation + dataset_translation


####
# Writing
####


OME_ZARR_PATH_RE = re.compile(
    r"""
    ^                       # start of string
    [A-Za-z0-9._-]+         # first path segment: no empty, no special chars
    (?:                     # additional segments: (non-capturing)
        /                   #   forward slash as separator
        [A-Za-z0-9._-]+     #   another valid segment
    )*                      # zero or more additional segments
    $                       # end of string
    """,
    re.VERBOSE,
)


def validate_multiscale(multiscale: "Multiscale"):
    for scale_key in multiscale.keys():
        if not _is_valid_relative_path(str(scale_key)):
            raise ValueError(f"Scale key '{scale_key}' is not a valid relative filesystem path")

    axes = multiscale.axes()
    standard_axes_set = set("tczyx")

    if all(ax in standard_axes_set for ax in axes):
        expected_order = [ax for ax in "tczyx" if ax in axes]
        if axes != expected_order:
            warnings.warn(
                f"Axes {axes} are all standard (t,c,z,y,x) but not in OME-Zarr "
                f"canonical order. Expected: {expected_order}. "
                f"This may cause issues with some OME-Zarr readers."
            )


def _is_valid_relative_path(path: str) -> bool:
    if not OME_ZARR_PATH_RE.fullmatch(path):
        return False
    return all(seg not in {".", ".."} for seg in path.split("/"))


def build_axis_dicts(
    axes: OrderedAxes,
    unit: Unit,
    axis_types: Union[None, Literal["infer"], Mapping[AxisKey, Literal["space", "time", "channel"]]] = None,
) -> List[Dict[str, Any]]:
    if axis_types and axis_types != "infer" and not any(ax in axes for ax in axis_types):
        warnings.warn(f"Provided axis_types {set(axis_types.keys())} don't match any axes in this Multiscale: {axes}")
    elif axis_types == "infer":
        axis_types = {
            "t": "time",
            "time": "time",
            "timestep": "time",
            "timepoint": "time",
            "c": "channel",
            "ch": "channel",
            "channel": "channel",
            "channels": "channel",
            "z": "space",
            "y": "space",
            "x": "space",
        }

    ome_axes = []
    for axis in axes:
        axis_dict = {"name": str(axis)}
        if axis_types and axis in axis_types:
            axis_dict["type"] = axis_types[axis]
        if unit[axis]:
            axis_dict["unit"] = unit[axis]
        ome_axes.append(axis_dict)
    return ome_axes


def build_multiscale_transforms(global_scale: Spacing, global_translation: Translation) -> List[Dict[str, Any]]:
    global_transforms = []
    if not global_scale.is_identity():
        global_transforms.append({"type": "scale", "scale": global_scale.to_list()})
    if not global_translation.is_identity():
        if not global_transforms:  # Must have scale before translation
            global_transforms.append({"type": "scale", "scale": global_scale.to_list()})
        global_transforms.append({"type": "translation", "translation": global_translation.to_list()})
    return global_transforms


def build_dataset_dict(key, dataset_scale: Spacing, dataset_translation: Translation) -> Dict[str, Any]:
    dataset_transforms = [{"type": "scale", "scale": dataset_scale.to_list()}]
    if not dataset_translation.is_identity():
        dataset_transforms.append({"type": "translation", "translation": dataset_translation.to_list()})
    dataset_dict = {"path": str(key), "coordinateTransformations": dataset_transforms}
    return dataset_dict
