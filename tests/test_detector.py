"""Regression tests for the perception pipeline, run against synthetic scenes.

The point of these tests is the requirement from the DRIMS hands-on: the die is
"yellow (**maybe**)", the boards have different lighting, and the final setup
photo shows dice in five different colours at once.  So every test here sweeps
over colours and nuisance parameters rather than checking one happy path.
"""

import numpy as np
import pytest

from dice_vision.board_geometry import HomographyMapper, PinholeMapper, board_to_base
from dice_vision.detector import DiceDetector, DetectorConfig

from synthetic import BOARD_SIZE_M, DIE_PALETTE, SceneConfig, random_scene, render


@pytest.fixture(scope="module")
def detector():
    return DiceDetector()


# --------------------------------------------------------------------------- #
# Detection: does it find the die at all?
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("colour", sorted(DIE_PALETTE))
def test_die_is_found_for_every_colour(detector, colour):
    scene = render(SceneConfig(die_colour=colour, face_value=4, seed=7))
    detection = detector.detect(scene.image)
    assert detection is not None, f"missed the {colour} die entirely"
    error = np.linalg.norm(np.array(detection.center_px) - np.array(scene.die_center_px))
    assert error < 4.0, f"{colour}: centre off by {error:.1f} px"


@pytest.mark.parametrize("colour", sorted(DIE_PALETTE))
@pytest.mark.parametrize("face", [1, 2, 3, 4, 5, 6])
def test_face_value_is_read_for_every_colour_and_face(detector, colour, face):
    scene = render(SceneConfig(die_colour=colour, face_value=face, seed=11))
    detection = detector.detect(scene.image)
    assert detection is not None
    assert detection.face_value == face, (
        f"{colour} die showing {face} was read as {detection.face_value}"
    )


def test_shadow_is_not_mistaken_for_a_die(detector):
    """The cast shadow keeps the board hue, so it must stay out of the mask."""
    scene = render(SceneConfig(shadow_strength=0.6, shadow_offset_m=(0.05, 0.04), seed=3))
    detection = detector.detect(scene.image)
    assert detection is not None
    error = np.linalg.norm(np.array(detection.center_px) - np.array(scene.die_center_px))
    assert error < 5.0, "detector locked onto the shadow instead of the die"


def test_clutter_is_rejected(detector):
    """Cables and clamps are on the board but are not square."""
    scene = render(SceneConfig(clutter=True, seed=5))
    detection = detector.detect(scene.image)
    assert detection is not None
    error = np.linalg.norm(np.array(detection.center_px) - np.array(scene.die_center_px))
    assert error < 5.0


def test_no_detection_on_an_empty_board(detector):
    """Clutter alone must not be reported, even after the clutter-splitting pass."""
    scene = render(SceneConfig(seed=2))
    empty = render(SceneConfig(seed=2, die_size_m=0.0005, shadow_strength=0.0))
    assert detector.detect(empty.image) is None
    assert detector.detect(scene.image) is not None


def test_detect_handles_degenerate_input(detector):
    assert detector.detect(None) is None
    assert detector.detect(np.zeros((0, 0, 3), np.uint8)) is None
    assert detector.detect(np.zeros((64, 64, 3), np.uint8)) is None


# --------------------------------------------------------------------------- #
# Robustness sweeps
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("gradient", [0.0, 0.15, 0.3, 0.45])
def test_survives_uneven_lighting(detector, gradient):
    for direction in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2):
        scene = render(
            SceneConfig(
                face_value=6,
                light_gradient=gradient,
                light_direction_rad=direction,
                seed=13,
            )
        )
        detection = detector.detect(scene.image)
        assert detection is not None
        assert detection.face_value == 6


@pytest.mark.parametrize("size", [0.020, 0.027, 0.035, 0.045])
def test_survives_die_size_changes(detector, size):
    scene = render(SceneConfig(die_size_m=size, face_value=3, seed=17))
    detection = detector.detect(scene.image)
    assert detection is not None
    assert detection.face_value == 3


@pytest.mark.parametrize("yaw_deg", list(range(0, 90, 10)))
def test_face_reading_is_rotation_invariant(detector, yaw_deg):
    """Layouts 2, 3 and 6 are not 90-degree symmetric, so this matters."""
    for face in (2, 3, 6):
        scene = render(
            SceneConfig(face_value=face, die_yaw_rad=np.deg2rad(yaw_deg), seed=19)
        )
        detection = detector.detect(scene.image)
        assert detection is not None
        assert detection.face_value == face, f"{face} at {yaw_deg} deg"


@pytest.mark.parametrize("board", [(55, 150, 75), (70, 175, 95), (88, 198, 112)])
def test_survives_board_shade_changes(detector, board):
    scene = render(SceneConfig(board_bgr=board, face_value=2, seed=23))
    detection = detector.detect(scene.image)
    assert detection is not None
    assert detection.face_value == 2


def test_yaw_is_recovered_modulo_ninety_degrees(detector):
    """A square die's orientation is only observable modulo 90 degrees."""
    for yaw_deg in (0, 12, 31, 47, 63, 78):
        scene = render(SceneConfig(die_yaw_rad=np.deg2rad(yaw_deg), face_value=1, seed=29))
        detection = detector.detect(scene.image)
        assert detection is not None
        # The camera flips Y, so image yaw is the negative of board yaw.
        expected = -np.deg2rad(yaw_deg)
        error = (detection.yaw_rad - expected + np.pi / 4) % (np.pi / 2) - np.pi / 4
        assert abs(np.degrees(error)) < 5.0, f"yaw {yaw_deg}: off by {np.degrees(error):.1f}"


# --------------------------------------------------------------------------- #
# Board geometry
# --------------------------------------------------------------------------- #


def test_homography_recovers_board_position(detector):
    """Auto-calibrate from the board corners, then localise the die in metres."""
    for xy in [(0.0, 0.0), (0.22, 0.14), (-0.25, -0.15), (0.10, -0.18)]:
        cfg = SceneConfig(die_xy_m=xy, camera_height_m=1.05, seed=31)
        scene = render(cfg)
        detection = detector.detect(scene.image)
        assert detection is not None

        quad = detector.board_quad(scene.image)
        assert quad is not None
        mapper = HomographyMapper.from_board_quad(
            quad,
            BOARD_SIZE_M,
            camera_height_m=cfg.camera_height_m,
            nadir_px=(cfg.image_size[0] / 2.0, cfg.image_size[1] / 2.0),
        )
        x, y, _ = mapper.pixel_to_board(detection.center_px, height=cfg.die_size_m)
        error = np.hypot(x - xy[0], y - xy[1])
        assert error < 0.006, f"die at {xy}: {1000 * error:.1f} mm error"


def test_parallax_correction_actually_helps():
    """Skipping the height correction should measurably hurt."""
    cfg = SceneConfig(die_xy_m=(0.28, 0.17), die_size_m=0.032, camera_height_m=0.9, seed=37)
    scene = render(cfg)
    detector = DiceDetector()
    detection = detector.detect(scene.image)
    quad = detector.board_quad(scene.image)
    nadir = (cfg.image_size[0] / 2.0, cfg.image_size[1] / 2.0)

    corrected = HomographyMapper.from_board_quad(
        quad, BOARD_SIZE_M, camera_height_m=cfg.camera_height_m, nadir_px=nadir
    ).pixel_to_board(detection.center_px, height=cfg.die_size_m)
    naive = HomographyMapper.from_board_quad(quad, BOARD_SIZE_M).pixel_to_board(
        detection.center_px
    )

    err_corrected = np.hypot(corrected[0] - cfg.die_xy_m[0], corrected[1] - cfg.die_xy_m[1])
    err_naive = np.hypot(naive[0] - cfg.die_xy_m[0], naive[1] - cfg.die_xy_m[1])
    assert err_corrected < err_naive
    assert err_naive - err_corrected > 0.004, "parallax correction should be worth >4 mm here"


def test_pinhole_mapper_matches_the_renderer():
    """Round-trip a known 3D point through the true camera model."""
    cfg = SceneConfig(die_xy_m=(0.18, -0.11), seed=41)
    scene = render(cfg)
    k = scene.extra["camera_matrix"]

    # The renderer's camera: no tilt here, so board-from-camera is a flip plus
    # a translation along Z.
    rotation = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    translation = np.array([0.0, 0.0, cfg.camera_height_m])
    mapper = PinholeMapper.from_camera_info(k.flatten(), np.zeros(5), rotation, translation)

    x, y, z = mapper.pixel_to_board(scene.die_center_px, height=cfg.die_size_m)
    assert np.hypot(x - cfg.die_xy_m[0], y - cfg.die_xy_m[1]) < 0.002
    assert z == pytest.approx(cfg.die_size_m)


def test_pinhole_mapper_rejects_impossible_rays():
    k = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1.0]])

    # Camera lying in the board plane and looking along it: the principal ray
    # is parallel to the plane and never meets it.
    grazing = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
    parallel = PinholeMapper.from_camera_info(
        k.flatten(), np.zeros(5), grazing, [0.0, 0.0, 0.0]
    )
    with pytest.raises(ValueError, match="parallel"):
        parallel.pixel_to_board((640.0, 360.0), height=0.0)

    # Camera below the board looking down: the plane is behind it.
    looking_away = PinholeMapper.from_camera_info(
        k.flatten(), np.zeros(5), np.diag([1.0, -1.0, -1.0]), [0.0, 0.0, -1.0]
    )
    with pytest.raises(ValueError, match="behind"):
        looking_away.pixel_to_board((640.0, 360.0), height=0.0)


def test_board_to_base_transform():
    assert board_to_base((0.1, 0.0, 0.0), (0.6, 0.0, -0.04)) == pytest.approx(
        (0.7, 0.0, -0.04)
    )
    rotated = board_to_base((0.1, 0.0, 0.0), (0.6, 0.0, -0.04), np.pi / 2)
    assert rotated == pytest.approx((0.6, 0.1, -0.04), abs=1e-9)


# --------------------------------------------------------------------------- #
# End-to-end benchmark
# --------------------------------------------------------------------------- #


def test_randomised_benchmark(detector):
    """200 random scenes: detection rate, face accuracy and position error.

    The thresholds here are the numbers quoted in the README.
    """
    rng = np.random.default_rng(20260909)
    found = correct = 0
    errors = []
    failures = []

    trials = 200
    for _ in range(trials):
        cfg = random_scene(rng)
        scene = render(cfg)
        detection = detector.detect(scene.image)
        if detection is None:
            failures.append((cfg.die_colour, cfg.face_value, "not found"))
            continue
        found += 1

        if detection.face_value == cfg.face_value:
            correct += 1
        else:
            failures.append((cfg.die_colour, cfg.face_value, detection.face_value))

        quad = detector.board_quad(scene.image)
        if quad is not None:
            mapper = HomographyMapper.from_board_quad(
                quad,
                BOARD_SIZE_M,
                camera_height_m=cfg.camera_height_m,
                nadir_px=(cfg.image_size[0] / 2.0, cfg.image_size[1] / 2.0),
            )
            x, y, _ = mapper.pixel_to_board(detection.center_px, height=cfg.die_size_m)
            errors.append(np.hypot(x - cfg.die_xy_m[0], y - cfg.die_xy_m[1]))

    detection_rate = found / trials
    face_accuracy = correct / trials
    median_error_mm = 1000 * float(np.median(errors))
    p95_error_mm = 1000 * float(np.percentile(errors, 95))

    print(
        f"\ndetection rate {detection_rate:.3f}  face accuracy {face_accuracy:.3f}  "
        f"median {median_error_mm:.1f} mm  p95 {p95_error_mm:.1f} mm"
    )
    if failures:
        print(f"failures ({len(failures)}): {failures[:10]}")

    assert detection_rate >= 0.99
    assert face_accuracy >= 0.97
    assert found >= correct
    assert median_error_mm < 3.0
    assert p95_error_mm < 6.0
