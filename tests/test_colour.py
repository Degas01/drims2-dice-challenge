"""Tests for naming the die's colour, and for the staged debug images.

No ROS and no camera: the scenes come from the renderer in
``dice_vision.scene_simulator``, the same one the fake camera node publishes.
"""

import cv2
import numpy as np
import pytest

from dice_vision.colour import (
    HUE_BANDS,
    ColourReading,
    body_colour,
    classify_bgr,
    read_die_colour,
)
from dice_vision.detector import DiceDetector, mosaic
from dice_vision.scene_simulator import DIE_PALETTE, SceneConfig, render


# --------------------------------------------------------------------------- #
# The hue bands themselves
# --------------------------------------------------------------------------- #


def test_hue_bands_tile_the_whole_circle_without_gaps_or_overlaps():
    """Every hue gets exactly one name, red's wrap-around included."""
    for hue in range(180):
        matches = [
            name for name, low, high in HUE_BANDS if (hue - low) % 180.0 < (high - low)
        ]
        assert len(matches) == 1, f"hue {hue} matched {matches}"


def test_red_is_a_single_arc_through_zero():
    """Split into two half-bands, a perfect red would report as barely-red."""
    for hue in (170, 175, 179, 0, 4, 7):
        pixel = cv2.cvtColor(
            np.array([[[hue, 220, 200]]], np.uint8), cv2.COLOR_HSV2BGR
        )[0, 0]
        reading = classify_bgr(pixel)
        assert reading.name == "red", f"hue {hue} -> {reading.name}"
    # ...and hue 179 sits mid-arc, so it is fully confident.
    peak = cv2.cvtColor(np.array([[[179, 230, 210]]], np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    assert classify_bgr(peak).confidence == pytest.approx(1.0)


@pytest.mark.parametrize(
    "hue,expected",
    [(14, "orange"), (27, "yellow"), (60, "green"), (92, "cyan"),
     (115, "blue"), (141, "purple"), (161, "pink")],
)
def test_each_band_names_its_own_centre(hue, expected):
    pixel = cv2.cvtColor(np.array([[[hue, 220, 200]]], np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    reading = classify_bgr(pixel)
    assert reading.name == expected
    assert reading.confidence == pytest.approx(1.0)


def test_a_hue_on_a_boundary_reports_low_confidence_rather_than_a_confident_guess():
    boundary = cv2.cvtColor(np.array([[[20, 220, 200]]], np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    reading = classify_bgr(boundary)
    assert reading.name in ("orange", "yellow")
    assert reading.confidence < 0.2


# --------------------------------------------------------------------------- #
# Achromatic dice
# --------------------------------------------------------------------------- #


def test_greyscale_is_named_from_value_not_from_hue():
    assert classify_bgr((12, 12, 12)).name == "black"
    assert classify_bgr((245, 245, 245)).name == "white"
    assert classify_bgr((120, 120, 120)).name == "grey"


def test_a_mid_grey_is_confidently_grey_and_a_borderline_one_is_not():
    middle = classify_bgr((120, 120, 120))
    assert middle.name == "grey" and not middle.is_chromatic
    assert middle.confidence == pytest.approx(1.0)

    # Just the light side of the black threshold: the name is a coin toss and
    # the number should say so rather than claiming certainty.
    borderline = classify_bgr((74, 74, 74))
    assert borderline.name == "grey"
    assert borderline.confidence < 0.2


def test_a_desaturated_colour_is_reported_less_confidently_than_a_vivid_one():
    vivid = cv2.cvtColor(np.array([[[115, 240, 200]]], np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    washed = cv2.cvtColor(np.array([[[115, 70, 200]]], np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    assert classify_bgr(vivid).name == classify_bgr(washed).name == "blue"
    assert classify_bgr(washed).confidence < classify_bgr(vivid).confidence


# --------------------------------------------------------------------------- #
# Sampling the body out of a two-tone face
# --------------------------------------------------------------------------- #


def _painted_face(body, pip, pip_cells):
    """A 120x120 face with 18 px pips on a 3x3 grid, as a die really looks."""
    image = np.zeros((160, 160, 3), np.uint8)
    image[:] = (90, 90, 90)
    cv2.rectangle(image, (20, 20), (140, 140), body, -1)
    for cx, cy in pip_cells:
        cv2.circle(image, (80 + 40 * cx, 80 + 40 * cy), 15, pip, -1)
    quad = np.array([[20, 20], [140, 20], [140, 140], [20, 140]], np.float32)
    return image, quad


def test_the_body_colour_wins_even_when_the_pips_cover_more_of_the_sample():
    """The six is the face that breaks a majority vote.

    Inside the inset sampling window a six's pips cover more area than the body
    between them, so counting pixels names a blue die "white".  Comparing the
    largest connected region instead is immune: one pip is never larger than the
    face around it.
    """
    six = [(-1, -1), (-1, 0), (-1, 1), (1, -1), (1, 0), (1, 1)]
    image, quad = _painted_face((190, 90, 30), (245, 245, 245), six)
    assert read_die_colour(image, quad).name == "blue"


@pytest.mark.parametrize(
    "cells",
    [[], [(0, 0)], [(-1, -1), (1, 1)],
     [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)],
     [(-1, -1), (-1, 0), (-1, 1), (1, -1), (1, 0), (1, 1)]],
)
def test_every_pip_count_reads_the_same_body_colour(cells):
    image, quad = _painted_face((40, 200, 235), (25, 25, 25), cells)
    assert read_die_colour(image, quad).name == "yellow"


def test_a_blank_patch_falls_back_to_the_median():
    image, quad = _painted_face((45, 45, 205), (45, 45, 205), [])
    sample = body_colour(image, quad)
    assert sample is not None
    assert np.allclose(sample, (45, 45, 205), atol=2)


def test_body_colour_returns_none_for_a_degenerate_quad():
    image = np.zeros((40, 40, 3), np.uint8)
    assert body_colour(image, np.array([[5.0, 5.0], [6.0, 6.0]])) is None


# --------------------------------------------------------------------------- #
# End to end, against the renderer
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("colour", sorted(DIE_PALETTE))
def test_every_palette_colour_is_named_correctly_on_every_face(colour):
    detector = DiceDetector()
    expected = colour.split("_")[0]
    for face in range(1, 7):
        scene = render(
            SceneConfig(
                face_value=face,
                die_colour=colour,
                die_size_m=0.03,
                die_xy_m=(0.05, -0.03),
                die_yaw_rad=0.4,
                seed=face,
            )
        )
        detection = detector.detect(scene.image)
        assert detection is not None, f"{colour} face {face} not detected"
        assert detection.colour_name == expected, (
            f"{colour} face {face} read as {detection.colour_name} "
            f"(hsv {detection.colour.hsv})"
        )
        assert detection.colour.confidence > 0.5


def test_the_swatch_matches_the_pixels_the_name_came_from():
    """The overlay draws the sampled colour, so a wrong name is visible."""
    detector = DiceDetector()
    scene = render(SceneConfig(face_value=3, die_colour="red", die_size_m=0.03, seed=1))
    detection = detector.detect(scene.image)
    assert detection is not None
    assert isinstance(detection.colour, ColourReading)
    b, g, r = detection.colour.bgr
    assert r > 120 and g < 90 and b < 90


def test_colour_is_absent_rather_than_wrong_when_there_is_no_detection():
    detector = DiceDetector()
    blank = np.full((240, 320, 3), (70, 175, 95), np.uint8)
    assert detector.detect(blank) is None


# --------------------------------------------------------------------------- #
# The staged debug images
# --------------------------------------------------------------------------- #


def _scene():
    return render(
        SceneConfig(
            face_value=5,
            die_colour="yellow",
            die_size_m=0.03,
            die_xy_m=(0.06, -0.04),
            die_yaw_rad=0.5,
            seed=3,
        )
    )


def test_stages_returns_every_topic_the_hands_on_asks_for():
    detector = DiceDetector()
    scene = _scene()
    stages = detector.stages(scene.image)
    assert set(stages) == {
        "board_mask", "object_mask", "bounding_box", "overlay", "mosaic"
    }
    for name, image in stages.items():
        assert image.dtype == np.uint8, name
        if name != "mosaic":  # the mosaic is deliberately a different size
            assert image.shape == scene.image.shape, name


def test_the_board_mask_covers_most_of_the_board_and_little_else():
    detector = DiceDetector()
    scene = _scene()
    mask = detector.stages(scene.image)["board_mask"]
    covered = np.count_nonzero(mask[:, :, 0]) / mask[:, :, 0].size
    assert 0.15 < covered < 0.75


def test_the_object_mask_keeps_the_die_and_drops_the_board():
    detector = DiceDetector()
    scene = _scene()
    detection = detector.detect(scene.image)
    assert detection is not None
    masked = detector.stages(scene.image, detection)["object_mask"]

    cx, cy = (int(round(v)) for v in detection.center_px)
    assert masked[cy, cx].any(), "the die itself was masked away"
    # A patch of open board well clear of the die must be black.
    assert not masked[cy, max(0, cx - 200)].any()


def test_the_bounding_box_is_drawn_on_the_die():
    detector = DiceDetector()
    scene = _scene()
    detection = detector.detect(scene.image)
    boxed = detector.stages(scene.image, detection)["bounding_box"]
    difference = np.any(boxed != scene.image, axis=2)
    assert difference.any()
    ys, xs = np.nonzero(difference)
    # Everything drawn sits within a die's width of the die.
    assert abs(xs.mean() - detection.center_px[0]) < detection.size_px
    assert abs(ys.mean() - detection.center_px[1]) < detection.size_px


def test_the_overlay_stays_inside_the_frame_wherever_the_die_is():
    """The readout card is pinned above the die, so a die near an edge is the
    case that would otherwise crop it away or crash the drawing."""
    detector = DiceDetector()
    for x, y in ((-0.32, -0.22), (0.32, -0.22), (-0.32, 0.22), (0.32, 0.22), (0.0, 0.0)):
        scene = render(
            SceneConfig(
                face_value=6, die_colour="black", die_size_m=0.03,
                die_xy_m=(x, y), die_yaw_rad=0.2, seed=2,
            )
        )
        detection = detector.detect(scene.image)
        overlay = detector.annotate(scene.image, detection, (x, y))
        assert overlay.shape == scene.image.shape
        assert np.any(overlay != scene.image)


def test_the_overlay_handles_an_empty_board():
    detector = DiceDetector()
    blank = np.full((240, 320, 3), (70, 175, 95), np.uint8)
    overlay = detector.annotate(blank, None)
    assert overlay.shape == blank.shape
    assert np.any(overlay != blank), "an empty board should still say so"


def test_the_mosaic_tiles_every_stage_into_one_panel():
    detector = DiceDetector()
    scene = _scene()
    stages = detector.stages(scene.image)
    tiled = stages["mosaic"]
    assert tiled.ndim == 3 and tiled.shape[2] == 3
    # Two columns, two rows, so twice as wide as one tile and twice as tall.
    assert tiled.shape[1] == 1280
    assert tiled.shape[0] == pytest.approx(
        2 * int(round(640 * scene.image.shape[0] / scene.image.shape[1])), abs=2
    )


def test_the_mosaic_is_not_four_copies_of_the_same_picture():
    detector = DiceDetector()
    tiled = detector.stages(_scene().image)["mosaic"]
    h, w = tiled.shape[0] // 2, tiled.shape[1] // 2
    quadrants = [tiled[:h, :w], tiled[:h, w:], tiled[h:, :w], tiled[h:, w:]]
    for i in range(4):
        for j in range(i + 1, 4):
            assert np.any(quadrants[i] != quadrants[j])


def test_mosaic_rejects_an_empty_set_of_stages():
    with pytest.raises(ValueError):
        mosaic({})
