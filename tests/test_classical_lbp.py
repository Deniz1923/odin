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


OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def naive_codes(gray):
    """Deliberately slow per-pixel reference for the documented bit order and >= rule."""
    height, width = gray.shape
    codes = np.zeros((height - 2, width - 2), dtype=np.uint8)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            for bit, (dy, dx) in enumerate(OFFSETS):
                if gray[y + dy, x + dx] >= gray[y, x]:
                    codes[y - 1, x - 1] |= 1 << bit
    return codes


@pytest.mark.parametrize("levels", [2, 4, 256])
def test_lbp_codes_match_naive_reference_including_ties(levels):
    from face_surveillance.classical.lbp import lbp_codes

    rng = np.random.default_rng(levels)
    for shape in [(3, 3), (5, 7), (31, 17)]:
        gray = rng.integers(0, levels, shape).astype(np.uint8)
        assert np.array_equal(lbp_codes(gray), naive_codes(gray))


def test_full_size_descriptor_layout_and_distance_properties():
    from face_surveillance.classical.lbp import describe, distance, lbp_codes

    rng = np.random.default_rng(0)
    gray = rng.integers(0, 256, (128, 128)).astype(np.uint8)
    descriptor = describe(gray)
    assert descriptor.shape == (16384,) and descriptor.dtype == np.float32
    cells = descriptor.reshape(64, 256)
    np.testing.assert_allclose(cells.sum(axis=1), 1, atol=1e-6)
    # 126 code rows split 16/15: the first cell covers the top-left 16x16 code block.
    first = np.bincount(lbp_codes(gray)[:16, :16].ravel(), minlength=256) / 256
    np.testing.assert_allclose(cells[0], first, atol=1e-7)
    others = [describe(rng.integers(0, 256, (128, 128)).astype(np.uint8)) for _ in range(4)]
    for a in others:
        assert distance(a, a) == 0
        for b in others:
            assert distance(a, b) == pytest.approx(distance(b, a), abs=1e-6)
            assert 0 <= distance(a, b) <= 1
