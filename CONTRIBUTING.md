# Contributing
Feedback, suggestions and contributions in any form are welcome!

[AGENTS.md](AGENTS.md) is also for humans.
Use it as a guideline.

# Developing clearscale
It's dependency-free :)
Clone the repo, `pip install -e .[dev]`, and start coding.

`pre-commit install` to get black autoformat, run tests with `pytest`, typecheck with `pyright`.

# Feature scope
Planned future development ("deferred" features):
* Transforms
    * Remaining SpatialRelation subclasses for currently unrepresented Transforms (Coordinates, Displacements, Bijection, ByDimension)
    * (Related:) The API for Scene, Multiscale and SpatialRelation to expose and receive paths to transform data
    * Composition, decomposition, matrix algebra with Linear/Affine (the algebra already exists in `matrices.py`, just needs to be extracted and used for Affine)
* Building metadata for entire zarr group hierarchies (-> HCS plates, bioformats2raw, OME-Zarr RFC-8 collections).
  Something like an OmeZarrLayout.
* Optional deep-resolving OmeZarrGroup (to discover nested child Multiscales, and resolve Scene children, in one call)
* More documentation :)

Unplanned but within scope:

* Refinements to `__str__` and `__repr__`
* Other currently missing operator methods to complete axis-wise algebra (with semantically unambiguous operations)
* Maybe more AxisValues subtypes (e.g. index-space fractional offsets?)
* Other publicly specified multiscale formats (improving the basic Neuroglancer Precomputed support, OME-TIFF, DICOM,...)
* Adapters to other Python packages for better interoperability (to the extent this can be done strictly dependency-free)
* API simplifications and other refinements for more intuitiveness

Maybe, but probably out of scope / not worth it:

* Dependency-free zarr attrs reading (supporting the multitude of read protocols needed could be tough - from disk, via https, with authentication... - and the consumer must already be using some zarr backend to read data, so not expecting a lot of value from bringing our own)

Out of scope:

* Reading or processing image data in any form (would require dependencies)
* Features that would require optional dependencies