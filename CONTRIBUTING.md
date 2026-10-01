# Contributing
Feedback and contributions in any form are welcome!

[AGENTS.md](AGENTS.md) is also for humans.
Use it as a guideline.

# Developing clearscale
It's dependency-free :)
Clone the repo, `pip install -e .[dev]`, and start coding.

`pre-commit install` to get black autoformat, run tests with `pytest`, typecheck with `pyright`. 

# Feature scope
Planned future development ("deferred" features):
* Transforms
    * Remaining SpatialRelation subclasses for currently unrepresented Transforms (Rotation, Affine, Coordinates, Displacements, Bijection, ByDimension)
    * Transform -> SpatialRelation conversion
    * Multiscale and Scene methods for retrieving SpatialRelations out of their TransformGraph
    * Composition, decomposition and matrix algebra with Linear/Affine (the algebra already exists in `matrices.py`, just needs to be extracted and used for Affine)
* Building metadata for entire zarr group hierarchies (-> HCS plates, bioformats2raw, OME-Zarr RFC-8 collections)
* Nicer `__str__` and `__repr__` for public classes at least...
* More documentation :)

Unplanned but within scope:

* Other publicly specified multiscale formats (Neuroglancer Precomputed, OME-TIFF, DICOM,...)
* Adapters to other Python packages for better interoperability (to the extent this can be done strictly dependency-free)
* API simplifications and other refinements for more intuitiveness

Maybe, but probably out of scope / not worth it:
* Dependency-free zarr attrs reading (supporting the multitude of read protocols needed could be tough - from disk, via https, with authentication... - and the consumer must already be using some zarr backend to read data, so not expecting a lot of value from bringing our own)

Out of scope:

* Reading or processing image data in any form (would require dependencies)
* Features that would require optional dependencies