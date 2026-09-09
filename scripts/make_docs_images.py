#!/usr/bin/env python3
"""Render the figures used in the README from the synthetic scene generator.

    python3 scripts/make_docs_images.py

Everything it produces is generated from the same code the tests run, so the
pictures in the README cannot drift away from the measured behaviour.
"""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src" / "dice_vision"), str(ROOT / "tests")]

from dice_vision.detector import DiceDetector  # noqa: E402
from synthetic import DIE_PALETTE, SceneConfig, render  # noqa: E402

OUT = ROOT / "docs" / "images"


def _label(image: np.ndarray, text: str) -> np.ndarray:
    banner = np.full((26, image.shape[1], 3), 245, np.uint8)
    cv2.putText(banner, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack((banner, image))


def _crop(scene, margin: int = 70) -> np.ndarray:
    cx, cy = (int(v) for v in scene.die_center_px)
    h, w = scene.image.shape[:2]
    x0, x1 = max(0, cx - margin), min(w, cx + margin)
    y0, y1 = max(0, cy - margin), min(h, cy + margin)
    return scene.image[y0:y1, x0:x1], (x0, y0)


def colour_sweep(detector: DiceDetector) -> None:
    """One crop per die colour, annotated with the detection."""
    tiles = []
    for i, colour in enumerate(sorted(DIE_PALETTE)):
        face = 1 + (i % 6)
        scene = render(SceneConfig(die_colour=colour, face_value=face, seed=11))
        detection = detector.detect(scene.image)
        annotated = detector.annotate(scene.image, detection)

        crop, (x0, y0) = _crop(scene, margin=52)
        crop = annotated[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]]
        crop = cv2.resize(crop, (210, 210), interpolation=cv2.INTER_LANCZOS4)
        read = "none" if detection is None else detection.face_value
        tiles.append(_label(crop, f"{colour}: true {face}, read {read}"))

    grid = np.vstack([np.hstack(tiles[i : i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imwrite(str(OUT / "colour_sweep.png"), grid)
    print("wrote colour_sweep.png")


def pipeline_stages(detector: DiceDetector) -> None:
    """The three masks the detector builds, side by side with the input."""
    scene = render(SceneConfig(die_colour="yellow", face_value=5,
                               die_xy_m=(0.20, 0.12), shadow_strength=0.5, seed=3))
    board = detector.board_mask(scene.image)
    foreground, _ = detector.foreground_mask(scene.image)
    detection = detector.detect(scene.image)
    annotated = detector.annotate(scene.image, detection)

    def small(img):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.resize(img, (420, 236), interpolation=cv2.INTER_AREA)

    row = np.hstack(
        [
            _label(small(scene.image), "1. input"),
            _label(small(board), "2. board region (shadow included)"),
            _label(small(foreground), "3. foreground = on board, not board-coloured"),
            _label(small(annotated), "4. top face, axes, pips, reading"),
        ]
    )
    cv2.imwrite(str(OUT / "pipeline.png"), row)
    print("wrote pipeline.png")


def top_face_recovery(detector: DiceDetector) -> None:
    """Why the silhouette is not the top face away from the image centre."""
    scene = render(
        SceneConfig(die_colour="yellow", face_value=6, die_xy_m=(-0.30, -0.20),
                    die_size_m=0.045, camera_height_m=0.85, seed=5)
    )
    detection = detector.detect(scene.image)
    crop, (x0, y0) = _crop(scene, margin=90)
    canvas = crop.copy()

    import cv2 as _cv2

    foreground, _ = detector.foreground_mask(scene.image)
    contours, _ = _cv2.findContours(
        foreground[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]],
        _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE,
    )
    if contours:
        biggest = max(contours, key=_cv2.contourArea)
        _cv2.polylines(canvas, [_cv2.convexHull(biggest)], True, (255, 200, 0), 2)
    if detection is not None:
        shifted = detection.quad - np.array([x0, y0], dtype=np.float32)
        _cv2.polylines(canvas, [shifted.astype(int)], True, (0, 0, 255), 2)

    canvas = cv2.resize(canvas, (420, 420), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(OUT / "top_face.png"),
                _label(canvas, "blue: silhouette   red: recovered top face"))
    print("wrote top_face.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    detector = DiceDetector()
    pipeline_stages(detector)
    colour_sweep(detector)
    top_face_recovery(detector)


if __name__ == "__main__":
    main()
