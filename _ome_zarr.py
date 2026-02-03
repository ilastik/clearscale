import re
import warnings
from dataclasses import dataclass
from typing import Union, Literal, Mapping, Dict, List, Any, Optional, Tuple

from lazyflow.utility.io_util.clearscale import Translation, Unit, Spacing, Factor
from lazyflow.utility.io_util.clearscale._axis_values import OrderedAxes

####
# Reading
####


OME_ZARR_DATASET = Dict[Literal["path", "coordinateTransformations"], Any]  # single dataset (= scale)
OME_ZARR_MULTISCALE = Dict[  # single multiscales entry of a json-validated OME-Zarr zattrs (any version)
    # The spec allows for multiple multiscales, but in practice we only ever see one.
    Literal["axes", "datasets", "version", "coordinateTransformations", "name"],
    Union[List[Dict], List[OME_ZARR_DATASET], str],
]


class InvalidTransformationError(ValueError):
    pass


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


def _validate_transforms(
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


def spacing_from_multiscale(multiscale: OME_ZARR_MULTISCALE, dataset: str) -> Spacing:
    def has_valid_resolution(transforms: Union[None, ValidTransformations, InvalidTransformationError]):
        return isinstance(transforms, tuple) and transforms[0].values and len(transforms[0].values) == len(axis_keys)

    axis_keys = axes_from_multiscale(multiscale)
    try:
        dataset_spec = next(d for d in multiscale["datasets"] if d["path"] == dataset)
    except StopIteration:
        raise ValueError(f'Dataset "{dataset}" not defined in OME-Zarr "datasets" metadata:\n{multiscale["datasets"]}')
    dataset_transforms = _validate_transforms(dataset_spec.get("coordinateTransformations"))
    if not has_valid_resolution(dataset_transforms):
        warnings.warn(f"Missing or invalid pixel resolution metadata for dataset={dataset_spec['path']}.")
        return Spacing.fromkeys(axis_keys)

    dataset_resolution = dataset_transforms[0].values

    multiscale_transforms = _validate_transforms(multiscale.get("coordinateTransformations"))
    if not has_valid_resolution(multiscale_transforms):
        if multiscale_transforms is not None:
            warnings.warn("Pixel resolution metadata at pyramid level was invalid.")
        return Spacing(zip(axis_keys, dataset_resolution))
    else:
        spacing = Spacing(zip(axis_keys, multiscale_transforms[0].values))
        scale = Factor([(k, v) for k, v in zip(axis_keys, dataset_resolution) if v != 0])
        return spacing.scaled_by(scale)


def translation_from_multiscale(multiscale: OME_ZARR_MULTISCALE, dataset: str):
    # todo
    return Translation.fromkeys(axes_from_multiscale(multiscale))


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
    axis_types: Union[None, Literal["infer"], Mapping[str, Literal["space", "time", "channel"]]] = None,
) -> List[Dict[str, Any]]:
    if axis_types and axis_types != "infer" and not any(ax in axes for ax in axis_types):
        warnings.warn(f"Provided axis_types {set(axis_types.keys())} don't match any axes in this Multiscale: {axes}")
    elif axis_types == "infer":
        axis_types = {"t": "time", "c": "channel", "z": "space", "y": "space", "x": "space"}

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
