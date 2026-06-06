from __future__ import annotations

import math

import cv2
import numpy as np
import pandas as pd

from insideout2.config import EMOTIONS


def seconds_to_time(sec: float) -> str:
    sec = int(sec)
    m, s = divmod(sec, 60)
    return f"{m:02d}:{s:02d}"


def label_entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = pd.Series(values).value_counts(normalize=True)
    ent = -sum(p * math.log(p + 1e-9) for p in counts)
    max_ent = math.log(len(EMOTIONS))
    return float(ent / max_ent) if max_ent > 0 else 0.0


def is_blurry(image: np.ndarray, threshold: float = 20) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


def is_poor_lighting(image: np.ndarray, low: float = 30, high: float = 225) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    return avg_brightness < low or avg_brightness > high


def is_duplicate(
    img1: np.ndarray | None,
    img2: np.ndarray | None,
    threshold: float = 0.85,
) -> bool:
    if img1 is None or img2 is None:
        return False
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray1, gray2, cv2.TM_CCORR_NORMED)
    return res[0][0] > threshold
