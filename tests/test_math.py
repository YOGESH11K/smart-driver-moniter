import math
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.math_helpers import compute_ear, compute_mar


class SimpleLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def make_landmarks(eye_points):
    return [SimpleLandmark(x, y) for x, y in eye_points]


def test_ear_open_eye():
    w, h = 640, 480
    lm = make_landmarks([
        (200, 200), (202, 200), (204, 200), (210, 200),
        (202, 204), (204, 204),
        (400, 200), (402, 200), (404, 200), (410, 200),
        (402, 204), (404, 204),
        (100, 400), (105, 400), (110, 400), (115, 400),
        (102, 410), (113, 410), (100, 410),
    ])
    ear = compute_ear(lm, [0, 1, 2, 3, 4, 5], w, h)
    assert ear > 0.2, f"EAR should be high for open eye, got {ear}"


def test_ear_closed_eye():
    w, h = 640, 480
    lm = make_landmarks([
        (200, 100), (202, 100), (204, 100), (210, 100),
        (204, 101), (202, 101),
        (400, 100), (402, 100), (404, 100), (410, 100),
        (404, 101), (402, 101),
        (100, 400), (105, 400), (110, 400), (115, 400),
        (102, 410), (113, 410), (100, 410),
    ])
    ear = compute_ear(lm, [0, 1, 2, 3, 4, 5], w, h)
    assert ear < 0.15, f"EAR should be low for closed eye, got {ear}"


def test_ear_symmetric_next_eye():
    w, h = 640, 480
    lm = make_landmarks([
        (100, 100), (102, 100), (104, 100), (110, 100),
        (102, 103), (104, 103),
        (400, 300), (402, 300), (404, 300), (410, 300),
        (402, 306), (404, 306),
        (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
    ])
    ear_left = compute_ear(lm, [0, 1, 2, 3, 4, 5], w, h)
    ear_right = compute_ear(lm, [6, 7, 8, 9, 10, 11], w, h)
    assert ear_left != ear_right


def _make_full_landmarks(mouth_points):
    lms = [SimpleLandmark(0, 0) for _ in range(292)]
    for idx, pt in mouth_points.items():
        lms[idx] = SimpleLandmark(pt[0], pt[1])
    return lms


def test_mar_closed_mouth():
    w, h = 640, 480
    lm = _make_full_landmarks({
        13: (100, 200),
        14: (100, 205),
        61: (80, 202),
        291: (140, 202),
    })
    mar = compute_mar(lm, w, h)
    assert mar < 0.5, f"MAR should be low for closed mouth, got {mar}"


def test_mar_open_mouth():
    w, h = 640, 480
    lm = _make_full_landmarks({
        13: (100, 200),
        14: (100, 280),
        61: (80, 240),
        291: (140, 240),
    })
    mar = compute_mar(lm, w, h)
    assert mar > 0.5, f"MAR should be high for open mouth, got {mar}"


if __name__ == "__main__":
    test_ear_open_eye()
    test_ear_closed_eye()
    test_mar_closed_mouth()
    test_mar_open_mouth()
    print("All math helper tests passed.")