class NoSuchCoordinateSystemError(ValueError):
    """
    Raised by Multiscale._as_ref when called with a name that does not match any of the Multiscale's coordinate systems.
    """

    def __init__(self, name):
        super().__init__(f"This Multiscale does not specify a coordinate system named: {name!r}")


class MismatchingMultiscaleError(ValueError):
    """
    Raised by Transform.with_resolved when a Multiscale provided for `path`
    does not have a coordinate system named `name` as expected.
    """

    path: str
    name: str

    def __init__(self, path: str, name: str):
        super().__init__(f"No coordinate system named {name!r} in Multiscale provided for path {path!r}")
        self.path = path
        self.name = name


class CannotConvertToAffineError(ValueError):
    """
    Raised by AffineRepresentableTransform.to_affine_transform when the transform's payload is missing
    (path given instead).
    In case of ProjectAxisTransform.to_affine_transform, raised when the provided ndim params are nonsense.
    """

    def __init__(self, transform: object):
        msg = f"{transform.__class__.__name__} cannot be converted to AffineTransform with the available values."
        super().__init__(msg)


class CannotConvertToSpatialRelationError(ValueError):
    """
    Raised by transform_to_relation when a Transform cannot be expressed as SpatialRelation(s), for reasons that are
    properties of the *metadata* and that callers may expect and handle: unsupported transform type, path-backed or
    non-positive scale values, axis relabeling, intermediate axes that cannot be named.
    Transforms that don't fit the axes they are converted for raise plain ValueError instead (caller error), and
    violated internal invariants are assertions.
    """

    def __init__(self, reason: str):
        super().__init__(f"Cannot express transform as SpatialRelation: {reason}")
