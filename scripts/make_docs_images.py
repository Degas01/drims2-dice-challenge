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
sys.path[:0] = [str(ROOT / "src" / "dice_vision")]

from dice_vision.detector import DiceDetector  # noqa: E402
from dice_vision.scene_simulator import (  # noqa: E402
    DIE_PALETTE,
    SceneConfig,
    render,
)

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


def published_topics(detector: DiceDetector) -> None:
    """The four images the node actually publishes, as the node tiles them."""
    scene = render(SceneConfig(die_colour="yellow", face_value=5,
                               die_xy_m=(0.06, -0.04), die_yaw_rad=0.5, seed=3))
    detection = detector.detect(scene.image)
    stages = detector.stages(scene.image, detection, (0.062, -0.041))
    cv2.imwrite(str(OUT / "topics.png"), stages["mosaic"])
    print("wrote topics.png")

    # The readout card is 3% of the frame at full size, so crop and enlarge it.
    x, y = (int(round(v)) for v in detection.center_px)
    crop = stages["overlay"][max(0, y - 150):y + 90, max(0, x - 230):x + 230]
    cv2.imwrite(
        str(OUT / "readout.png"),
        cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST),
    )
    print("wrote readout.png")


def _readout_tiles(detector, scenes, captions):
    tiles = []
    for scene, caption in zip(scenes, captions):
        detection = detector.detect(scene.image)
        overlay = detector.annotate(scene.image, detection)
        x, y = (int(round(v)) for v in detection.center_px)
        crop = overlay[max(0, y - 120):y + 80, max(0, x - 170):x + 170]
        tiles.append(_label(cv2.resize(crop, (340, 200)), caption))
    return tiles


def named_colours(detector: DiceDetector) -> None:
    """Every palette colour, named, with a swatch of the sampled pixels."""
    names = sorted(DIE_PALETTE)
    scenes = [
        render(SceneConfig(die_colour=name, face_value=1 + i % 6, die_size_m=0.03,
                           die_xy_m=(0.06, -0.04), die_yaw_rad=0.5, seed=i))
        for i, name in enumerate(names)
    ]
    tiles = _readout_tiles(detector, scenes, names)
    cv2.imwrite(str(OUT / "colours.png"),
                np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])]))
    print("wrote colours.png")


def every_face(detector: DiceDetector) -> None:
    """All six faces on the hardest die to read: black pips on a black body."""
    scenes = [
        render(SceneConfig(die_colour="black", face_value=face, die_size_m=0.03,
                           die_xy_m=(0.06, -0.04), die_yaw_rad=0.5, seed=face))
        for face in range(1, 7)
    ]
    tiles = _readout_tiles(detector, scenes, [f"face {f}" for f in range(1, 7)])
    cv2.imwrite(str(OUT / "faces.png"),
                np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
    print("wrote faces.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    detector = DiceDetector()
    pipeline_stages(detector)
    published_topics(detector)
    colour_sweep(detector)
    named_colours(detector)
    every_face(detector)
    top_face_recovery(detector)


if __name__ == "__main__":
    main()
