import array
import brle
import numpy as np


def test_encode():
    a = np.array(
        [[0, 0, 1, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=np.uint8
    )
    expected = array.array("I", [4, 4, 0, 2, 1, 2, 2, 3, 1, 5])
    b = brle.encode(a)
    assert b == expected, f"Expected {expected}, got {b}"


def test_decode():
    a = array.array("I", [4, 4, 0, 2, 1, 2, 2, 3, 1, 5])
    expected = np.array(
        [[0, 0, 1, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=np.uint8
    )
    b = brle.decode(a)
    assert np.array_equal(b, expected), f"Expected {expected}, got {b}"


def test_brle():
    ys, xs = np.meshgrid(np.arange(1000), np.arange(1000))
    mask = ((ys - 500) ** 2 + (xs - 500) ** 2) < 250**2
    img = np.zeros((1000, 1000), dtype=np.uint8)
    img[mask] = 1
    encoded = brle.encode(img)
    decoded = brle.decode(encoded)
    assert np.array_equal(img, decoded), "Decoded image does not match original"
