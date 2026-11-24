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
* `Multiscale.lowest_lossless_ome_zarr_version` considers `.ome.omero`, and skips versions that cannot write the axes or omero metadata of the Multiscale.
  It may return `"0.9.dev2"` if no other version can.

### Fixed

* Reading raises a `ValueError` if the number of dimensions of a dataset does not match the number of axes.
  Previously, the extra axes or dimensions were silently dropped.

### Not yet supported

* Anatomical axis orientation (introduced in the in-development OME-Zarr version 0.9.dev2): Reading raises if an axis has an `orientation`, instead of dropping it.
  clearscale can write and read the other contents of 0.9.dev2, but does not support the version otherwise (so it is not documented as supported).
