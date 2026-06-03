import re
import warnings
from dataclasses import dataclass, replace
from typing import Union, Literal, Dict, List, Any, Optional, Tuple

from lazyflow.utility.io_util.clearscale import Translation, Spacing, Factor
from lazyflow.utility.io_util.clearscale._transforms import (
    TransformSequence,
    ScaleTransform,
    TranslationTransform,
    _TransformGraph,
    CoordinateSystemRef,
    CoordinateSystem,
    _UnresolvedRef,
    PRE_TRANSFORMS_VERSIONS,
    Transform,
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
class MultiscaleTransforms(TransformSequence):
    def __post_init__(self):
        if len(self.transforms) not in (1, 2):
            raise ValueError("MultiscaleTransforms requires one or two transforms.")
        if not isinstance(self.transforms[0], ScaleTransform):
            raise TypeError("First transform must be a ScaleTransform.")
        if len(self.transforms) == 2 and not isinstance(self.transforms[1], TranslationTransform):
            raise TypeError("Second transform must be a TranslationTransform.")

        TransformSequence.__post_init__(self)

    @property
    def scale_transform(self) -> ScaleTransform:
        return self.transforms[0]  # noqa

    @property
    def translation_transform(self) -> Optional[TranslationTransform]:
        return self.transforms[1] if len(self.transforms) == 2 else None

    @classmethod
    def from_list(cls, ome_transformations: Optional[List[Dict]]) -> Optional["MultiscaleTransforms"]:
        """
        Possibilities for ome_transformations:
        RFC-5 multiscale[datasets][n][coordinateTransformations]:
        - List of one ScaleTransform
        - List of one IdentityTransform
        - List of one TransformSequence containing one ScaleTransform and one TranslationTransform
        OME-Zarr v0.4 and 0.5:
        - multiscale[coordinateTransformations]:
          - absent or empty
          - List of one ScaleTransform
          - List of one ScaleTransform and one TranslationTransform
        - multiscale[datasets][][coordinateTransformations]:
          - List of one ScaleTransform
          - List of one ScaleTransform and one TranslationTransform
        """
        if not ome_transformations or not hasattr(ome_transformations, "__len__"):
            return None
        scale = None
        translation = None
        for t_dict in ome_transformations:
            # Best effort: Find first valid combination,
            # and accept even a valid translation without valid scale
            try:
                t = Transform.from_ome_zarr(t_dict)
            except ValueError:
                continue
            if isinstance(t, TransformSequence):
                if (len(t) == 1 and isinstance(t[0], ScaleTransform)) or (
                    len(t) == 2 and isinstance(t[0], ScaleTransform) and isinstance(t[1], TranslationTransform)
                ):
                    scale, translation = t.transforms
                    break
            if isinstance(t, ScaleTransform) and scale is None:
                scale = t
                continue
            if isinstance(t, TranslationTransform) and translation is None:
                translation = t
                if scale:
                    break
                continue
        if scale is None and translation is None:
            return None
        elif scale is None:
            scale = ScaleTransform(scale=tuple(1.0 for _ in range(len(translation.translation))))
        return cls(transforms=(scale,) if translation is None else (scale, translation))

    def composed_with(self, earlier: "Transform") -> Optional["Transform"]:
        if not isinstance(earlier, MultiscaleTransforms):
            return None
        if earlier.target is not None and self.source is not None and earlier.target != self.source:
            return None
        scale_product = self.scale_transform.composed_with(earlier.scale_transform)
        if earlier.translation_transform is not None and self.translation_transform is not None:
            translation_sum = self.translation_transform.composed_with(earlier.translation_transform)
            transforms = (scale_product, translation_sum)
        elif earlier.translation_transform is not None:
            transforms = (scale_product, earlier.translation_transform)
        elif self.translation_transform is not None:
            transforms = (scale_product, self.translation_transform)
        else:
            transforms = (scale_product,)
        return replace(self, source=earlier.source, transforms=transforms)


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
        and any("coordinateTransformations" not in d or not d["coordinateTransformations"] for d in raw["datasets"])
    ):
        raise ValueError(f"Invalid OME-Zarr datasets metadata: datasets without transformations. Received:\n{raw}")


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
    global_transforms: Optional[MultiscaleTransforms] = None,
) -> Tuple[_TransformGraph, CoordinateSystemRef[CoordinateSystem], Optional[MultiscaleTransforms]]:
    intrinsic_system = CoordinateSystem.from_ome_zarr(multiscale)
    intrinsic_system_ref = intrinsic_system.as_ref(name)
    graph = _TransformGraph.single_isolated_system(intrinsic_system_ref)
    bound_transform = None
    if global_transforms is not None:
        # Store the multiscale-level transforms as a transform to a non-existent mock system.
        # This allows Multiscale.to_ome_zarr to divide/subtract them back out of Scale.spacing/.translation
        # for perfect metadata round-trip.
        mock_ref = _UnresolvedRef(name=f"{name}-intermediate")
        bound_transform = global_transforms.bound(source=intrinsic_system_ref, target=mock_ref)
        graph = _TransformGraph([bound_transform])
    return graph, intrinsic_system_ref, bound_transform


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


def build_dataset_dict(
    version,
    key,
    dataset_scale: Spacing,
    dataset_translation: Translation,
    intrinsic_ref: Optional[CoordinateSystemRef[CoordinateSystem]] = None,
) -> Dict[str, Any]:
    scale = ScaleTransform.from_spacing(dataset_scale)
    if not dataset_translation.is_identity():
        translation = TranslationTransform.from_translation(dataset_translation)
        final = TransformSequence((scale, translation)).bound(source=_UnresolvedRef(name=key), target=intrinsic_ref)
    elif version in PRE_TRANSFORMS_VERSIONS:
        final = TransformSequence((scale,))
    else:
        final = scale.bound(source=_UnresolvedRef(name=key), target=intrinsic_ref)
    dataset_transforms = final.to_ome_zarr(version, for_scene=False)
    dataset_dict = {"path": str(key), "coordinateTransformations": dataset_transforms}
    return dataset_dict
