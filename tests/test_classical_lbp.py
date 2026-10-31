import numpy as np
import pytest


def test_lbp_bits_have_documented_clockwise_order_and_exclude_border():
    from face_surveillance.classical.lbp import lbp_codes

    # NW,N,NE,E,SE,S,SW,W; only N,E,S,W >= center -> 10101010.
    image = np.array([[0, 9, 0], [9, 5, 9], [0, 9, 0]], dtype=np.uint8)
    assert lbp_codes(image).tolist() == [[170]]
    assert lbp_codes(np.full((3, 3), 8, dtype=np.uint8)).tolist() == [[255]]


def test_descriptor_has_normalized_spatial_cells_and_brightness_invariance():
    from face_surveillance.classical.lbp import describe, distance

    image = np.arange(100, dtype=np.uint8).reshape(10, 10)
    descriptor = describe(image, grid=2)
    assert descriptor.shape == (1024,)
    np.testing.assert_allclose(descriptor.reshape(4, 256).sum(axis=1), 1)
    assert distance(descriptor, describe(image + 20, grid=2)) == pytest.approx(0)
    assert distance(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1)


def test_match_rejects_unknown_and_distinct_identity_ties():
    from face_surveillance.classical.lbp import match

    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert match(a, {}, 0.2, 0.02)["state"] == "insufficient"
    assert match(b, {"a": [a]}, 0.2, 0.02)["state"] == "unknown"
    assert match(a, {"a": [a, a]}, 0.2, 0.02)["person_id"] == "a"
    assert match(a, {"a": [a], "b": [a]}, 0.2, 0.02)["state"] == "insufficient"


@pytest.mark.parametrize(
    "bad", [np.zeros((2, 2), dtype=np.uint8), np.zeros((8, 8, 3)), np.full((8, 8), np.nan)]
)
def test_lbp_rejects_invalid_images(bad):
    from face_surveillance.classical.lbp import lbp_codes

    with pytest.raises(ValueError):
        lbp_codes(bad)
