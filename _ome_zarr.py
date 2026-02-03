import re
import warnings
from typing import Union, Literal, Mapping, Dict, List, Any

from lazyflow.utility.io_util.clearscale import Translation, Unit, Spacing
from lazyflow.utility.io_util.clearscale._axis_values import OrderedAxes


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
