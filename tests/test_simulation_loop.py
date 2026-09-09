"""The integration that runs in simulation: fake camera -> detector -> base frame.

The fake-camera node renders the die at a board position derived from TF, and
the vision node converts what it detects back into ``base_link``. Those two
conversions are inverses of each other, and if they disagree -- a sign, a yaw, a
board origin -- every grasp is silently offset. This checks the round trip with
the same numbers the launch files use, without needing ROS.
"""

import math

import numpy as np
import pytest

from dice_vision.board_geometry import DEFAULT_BOARD_SIZE_M, HomographyMapper, board_to_base
from dice_vision.detector import DiceDetector
from dice_vision.scene_simulator import DIE_PALETTE, SceneConfig, render

# The defaults the launch files ship, i.e. cell 4 of the dice simulator.
BOARD_ORIGIN = (-0.05, 0.675, -0.02)
CAMERA_HEIGHT = 1.0
DIE_SIZE = 0.03
IMAGE_SIZE = (1280, 720)


def _base_to_board(point_base, origin=BOARD_ORIGIN, board_yaw=0.0):
    """Inverse of board_to_base -- what the fake camera node does."""
    dx, dy = point_base[0] - origin[0], point_base[1] - origin[1]
    cos, sin = math.cos(-board_yaw), math.sin(-board_yaw)
    return cos * dx - sin * dy, sin * dx + cos * dy


def _detect_in_base(scene, cfg):
    """What the vision node does: detect, map to the board, then to base_link."""
    detector = DiceDetector()
    detection = detector.detect(scene.image)
    if detection is None:
        return None
    quad = detector.board_quad(scene.image)
    if quad is None:
        return None
    mapper = HomographyMapper.from_board_quad(
        quad,
        DEFAULT_BOARD_SIZE_M,
        camera_height_m=cfg.camera_height_m,
        nadir_px=(cfg.image_size[0] / 2.0, cfg.image_size[1] / 2.0),
    )
    x, y, _ = mapper.pixel_to_board(detection.center_px, height=cfg.die_size_m)
    return detection, board_to_base((x, y, 0.0), BOARD_ORIGIN, 0.0)


def _render_die_at(base_xy, face=5, colour="yellow", **overrides):
    board_x, board_y = _base_to_board((base_xy[0], base_xy[1]))
    cfg = SceneConfig(
        face_value=face,
        die_colour=colour,
        die_size_m=DIE_SIZE,
        die_xy_m=(board_x, board_y),
        image_size=IMAGE_SIZE,
        camera_height_m=CAMERA_HEIGHT,
        seed=17,
        **overrides,
    )
    return cfg, render(cfg)


def test_board_conversions_are_exact_inverses():
    for base in [(-0.05, 0.675), (0.10, 0.55), (-0.25, 0.80), (0.15, 0.72)]:
        board = _base_to_board(base)
        back = board_to_base((board[0], board[1], 0.0), BOARD_ORIGIN, 0.0)
        assert back[0] == pytest.approx(base[0], abs=1e-12)
        assert back[1] == pytest.approx(base[1], abs=1e-12)


@pytest.mark.parametrize(
    "base_xy",
    [(-0.05, 0.675), (0.10, 0.60), (-0.20, 0.75), (0.12, 0.80), (-0.22, 0.56)],
)
def test_round_trip_from_base_frame_through_the_camera_and_back(base_xy):
    """Put the die at a base_link position; the pipeline must recover it."""
    cfg, scene = _render_die_at(base_xy)
    result = _detect_in_base(scene, cfg)
    assert result is not None, f"nothing detected with the die at {base_xy}"
    _, recovered = result
    error = math.hypot(recovered[0] - base_xy[0], recovered[1] - base_xy[1])
    assert error < 0.006, f"die at {base_xy}: {1000 * error:.1f} mm off"


@pytest.mark.parametrize("colour", sorted(DIE_PALETTE))
def test_every_palette_colour_survives_the_simulated_camera(colour):
    """The parameter the fake camera exposes, exercised the way a user would."""
    cfg, scene = _render_die_at((0.05, 0.70), face=3, colour=colour)
    result = _detect_in_base(scene, cfg)
    assert result is not None, f"{colour} die not detected"
    detection, recovered = result
    assert detection.face_value == 3, f"{colour}: read {detection.face_value}"
    assert math.hypot(recovered[0] - 0.05, recovered[1] - 0.70) < 0.008


@pytest.mark.parametrize("face", [1, 2, 3, 4, 5, 6])
def test_every_face_is_read_through_the_simulated_camera(face):
    cfg, scene = _render_die_at((-0.05, 0.675), face=face)
    result = _detect_in_base(scene, cfg)
    assert result is not None
    assert result[0].face_value == face


def test_the_simulated_lens_can_be_given_distortion():
    """The fake camera's dist_coeffs parameter reaches the rendered image."""
    _, sharp = _render_die_at((0.10, 0.60))
    _, bent = _render_die_at((0.10, 0.60), dist_coeffs=(-0.28, 0.09, 0.0, 0.0, 0.0))
    assert not np.array_equal(sharp.image, bent.image)
    # Distortion moves the board's corners; that is the whole point.
    detector = DiceDetector()
    sharp_quad = detector.board_quad(sharp.image)
    bent_quad = detector.board_quad(bent.image)
    assert sharp_quad is not None and bent_quad is not None
    assert np.max(np.abs(sharp_quad - bent_quad)) > 5.0


def test_a_wrong_board_origin_shows_up_as_a_constant_offset():
    """Guards the most likely misconfiguration when running in simulation.

    If the fake camera and the vision node disagree about where the board is,
    nothing errors -- every die is just reported in the wrong place, by exactly
    the difference. Worth having a test that names the symptom.
    """
    base_xy = (0.05, 0.70)
    cfg, scene = _render_die_at(base_xy)
    detector = DiceDetector()
    detection = detector.detect(scene.image)
    quad = detector.board_quad(scene.image)
    assert detection is not None and quad is not None

    mapper = HomographyMapper.from_board_quad(
        quad, DEFAULT_BOARD_SIZE_M,
        camera_height_m=cfg.camera_height_m,
        nadir_px=(cfg.image_size[0] / 2.0, cfg.image_size[1] / 2.0),
    )
    x, y, _ = mapper.pixel_to_board(detection.center_px, height=cfg.die_size_m)

    wrong_origin = (BOARD_ORIGIN[0] + 0.07, BOARD_ORIGIN[1] - 0.04, BOARD_ORIGIN[2])
    wrong = board_to_base((x, y, 0.0), wrong_origin, 0.0)
    right = board_to_base((x, y, 0.0), BOARD_ORIGIN, 0.0)

    assert wrong[0] - right[0] == pytest.approx(0.07, abs=1e-9)
    assert wrong[1] - right[1] == pytest.approx(-0.04, abs=1e-9)
