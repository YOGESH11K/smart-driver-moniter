import math


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_ear(landmarks, eye_indices, w, h):
    p1 = (landmarks[eye_indices[0]].x * w, landmarks[eye_indices[0]].y * h)
    p2 = (landmarks[eye_indices[1]].x * w, landmarks[eye_indices[1]].y * h)
    p3 = (landmarks[eye_indices[2]].x * w, landmarks[eye_indices[2]].y * h)
    p4 = (landmarks[eye_indices[3]].x * w, landmarks[eye_indices[3]].y * h)
    p5 = (landmarks[eye_indices[4]].x * w, landmarks[eye_indices[4]].y * h)
    p6 = (landmarks[eye_indices[5]].x * w, landmarks[eye_indices[5]].y * h)

    vertical1 = euclidean_distance(p2, p6)
    vertical2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)

    if horizontal == 0:
        return 0.0
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear


def compute_mar(landmarks, w, h):
    upper = (landmarks[13].x * w, landmarks[13].y * h)
    lower = (landmarks[14].x * w, landmarks[14].y * h)
    left = (landmarks[61].x * w, landmarks[61].y * h)
    right = (landmarks[291].x * w, landmarks[291].y * h)

    vertical = euclidean_distance(upper, lower)
    horizontal = euclidean_distance(left, right)

    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def smooth_value(current, new_value, alpha=0.3):
    return alpha * new_value + (1 - alpha) * current


class ConsecutiveCounter:
    def __init__(self):
        self._count = 0
        self._active = False

    def update(self, condition):
        if condition:
            self._count += 1
            self._active = True
        else:
            self._count = 0
            self._active = False

    @property
    def count(self):
        return self._count

    @property
    def is_active(self):
        return self._active

    def reset(self):
        self._count = 0
        self._active = False
