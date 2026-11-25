# Changelog

Notable changes to clearscale. Versions before 0.9.0 are not listed.

While clearscale is at 0.9.x, breaking changes may happen in any release. They are marked **Breaking**.

## Unreleased

### Features

* Partial support for experimental OME-Zarr version 0.9.dev2. Its removal of axis number, type and order restrictions (RFC-3), as well as loosening required "omero" properties, is supported; anatomical orientation (RFC-4) is not.

### Changed

* **Breaking:** A `Multiscale` can have any number, types and order of axes.
  Only distinct axis names are required when constructing one.
  Until now, constructing a `Multiscale` raised unless it had 2 to 5 axes, of which 2 or 3 were `"space"`, with `"time"` and `"channel"` axes first.
  Those are restrictions of OME-Zarr 0.4, 0.5 and 0.6, so they are checked in `Multiscale.to_ome_zarr` (and `OmeZarrGroup.to_attrs`) for those versions instead, which raise for axes they cannot hold.
  Code that relied on the constructor raising now sees the error when writing.
  This also means that data with such axes can be read, which was requested by users of data that does not follow the restrictions.
* **Breaking:** All `OmeroChannel` properties are optional (`OmeroChannel()` is valid), and `OmeroWindow.min` and `OmeroWindow.max` are optional.
  OME-Zarr 0.4, 0.5 and 0.6 require `color`, and `window` with all four of its numbers; this is now checked when writing those versions, naming the channel that lacks something.
  This also means that `omero` metadata that lacks them is no longer discarded when reading.
* **Breaking:** `OmeroChannel.label`, `.active` and `.inverted` are properties, and no longer in `OmeroChannel.extra`.
  If `extra` still has such a key, it is ignored when writing.
* **Breaking:** `Omero.to_ome_zarr`, `OmeroChannel.to_ome_zarr` and `OmeroWindow.to_ome_zarr` require the OME-Zarr `version` to write.
* **Breaking:** The relations returned by `Multiscale.relation_to_coordinate_system`, `Multiscale.relation_from_coordinate_system` and `Scene.relation_from` combine all scaling, rotation, shearing and translation into one `Affine`.
  Previously, the translation was split off again as a separate `Translation` (so a scale followed by a shift came back as two relations, and a chain could contain several `Affine`s).
  Now, `Translation` is never returned: A pure translation is an `Affine` with identity `linear`, and the shift is `affine.translation`.
  The result contains at most one `Affine`, next to `ProjectionTo`, `PermutationTo`, `AxisRearrangementTo` or `AxisRenaming` if the transformation changes axes; if it does not, the result is that single `Affine` (or `SameSpace()`).
  Code that matched on `Translation` in retrieved relations must check for `Affine` instead (`affine.linear.is_identity()` tells if it is only a translation).
* A transform that needs a general `affine` is written with its translation inside the `affine`, no longer as an additional `translation` (this applies to affines that also add or drop axes; those that don't already worked this way). The transformation is the same.
* `Multiscale.lowest_lossless_ome_zarr_version` considers `.ome.omero`, and skips versions that cannot write the axes or omero metadata of the Multiscale.
  It may return `"0.9.dev2"` if no other version can.

### Fixed

* Retrieving the relation of a coordinate system whose transformation inserts or drops axes and also mixes axes returned `None` or a spurious `AxisRenaming`, for example a tilted plane in 3D (insert `z`, then make it depend on `x`), or a dropped axis that is not the last one but feeds another.
  This also covers replacing one axis with another while keeping the same number of dimensions.
  Where axes are inserted and dropped is now kept when transformations are combined, and such an edge is written with a `projectAxis` followed by an `affine` (and a trailing `projectAxis` for dropped axes that are still needed).
  What cannot be retrieved is a new axis that has the same name as an axis which is dropped after the affine.
* Reading raises a `ValueError` if the number of dimensions of a dataset does not match the number of axes.
  Previously, the extra axes or dimensions were silently dropped.

### Not yet supported

* Anatomical axis orientation (introduced in the in-development OME-Zarr version 0.9.dev2): Reading raises if an axis has an `orientation`, instead of dropping it.
  clearscale can write and read the other contents of 0.9.dev2, but does not support the version otherwise (so it is not documented as supported).
