from clearscale import BlueprintShapes, Multiscale, Scale, Shape


def test_blueprint_hash_matches_value_equality():
    left = BlueprintShapes({"s0": Shape(y=2, x=3)})
    right = BlueprintShapes({"s0": Shape(y=2, x=3)})

    assert left == right
    assert hash(left) == hash(right)


def test_multiscale_equality_and_hash_are_identity_based():
    left = Multiscale({"s0": Scale(Shape(y=2, x=3))})
    right = Multiscale({"s0": Scale(Shape(y=2, x=3))})

    assert left == left
    assert left != right
    assert {left, right} == {left, right}


def test_multiscale_refs_are_hashable():
    left = Multiscale({"s0": Scale(Shape(y=2, x=3))})
    right = Multiscale({"s0": Scale(Shape(y=2, x=3))})

    assert len({left.as_ref("physical"), right.as_ref("physical")}) == 2
