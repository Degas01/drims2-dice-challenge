"""Tests for the pinhole camera model and PnP pose estimation.

These check the implementation against three independent references:

* the closed-form equations from the vision lecture,
* OpenCV's own ``projectPoints`` / ``undistortPoints``,
* the synthetic renderer's ground truth, which knows the true camera and die
  poses exactly.
"""

import cv2
import numpy as np
import pytest

from dice_vision.camera_model import CameraModel
from dice_vision.detector import DiceDetector
from dice_vision.pose_estimation import (
    die_centre_from_top_face,
    estimate_top_face_pose,
    pose_in_frame,
    refine_pose_onto_plane,
)

from synthetic import SceneConfig, render

REAL_DISTORTION = (-0.28, 0.09, 0.001, -0.0005, 0.0)


@pytest.fixture(scope="module")
def camera():
    return CameraModel(fx=900.0, fy=900.0, cx=640.0, cy=360.0, image_size=(1280, 720))


@pytest.fixture(scope="module")
def wide_camera():
    return CameraModel(
        fx=700.0, fy=705.0, cx=632.0, cy=366.0,
        distortion=REAL_DISTORTION, image_size=(1280, 720),
    )


# --------------------------------------------------------------------------- #
# Intrinsics
# --------------------------------------------------------------------------- #


def test_projection_matches_the_lecture_equations(camera):
    """x = fx X/Z + cx, y = fy Y/Z + cy."""
    points = np.array([[0.0, 0.0, 1.0], [0.1, -0.05, 2.0], [-0.3, 0.2, 0.8]])
    expected = np.column_stack(
        (
            camera.fx * points[:, 0] / points[:, 2] + camera.cx,
            camera.fy * points[:, 1] / points[:, 2] + camera.cy,
        )
    )
    assert np.allclose(camera.project(points), expected)


def test_optical_axis_lands_on_the_principal_point(camera):
    assert np.allclose(camera.project([[0.0, 0.0, 3.0]])[0], camera.principal_point)


def test_projection_is_scale_invariant_along_a_ray(camera):
    """Distances are not perceivable: doubling Z and the point gives one pixel."""
    near = camera.project([[0.1, 0.05, 1.0]])[0]
    far = camera.project([[0.2, 0.10, 2.0]])[0]
    assert np.allclose(near, far)


def test_intrinsic_matrix_round_trips_through_camera_info(wide_camera):
    rebuilt = CameraModel.from_camera_info(
        wide_camera.matrix.flatten(), wide_camera.distortion, wide_camera.image_size
    )
    assert np.allclose(rebuilt.matrix, wide_camera.matrix)
    assert np.allclose(rebuilt.distortion, wide_camera.distortion)


def test_projection_rejects_points_behind_the_camera(camera):
    with pytest.raises(ValueError):
        camera.project([[0.0, 0.0, -1.0]])
    with pytest.raises(ValueError):
        camera.project([[0.0, 0.0, 0.0]])


def test_focal_length_must_be_positive():
    with pytest.raises(ValueError):
        CameraModel(fx=0.0, fy=900.0, cx=640.0, cy=360.0)


def test_field_of_view_and_scale(camera):
    horizontal, vertical = camera.field_of_view()
    assert np.degrees(horizontal) == pytest.approx(2 * np.degrees(np.arctan(640 / 900)))
    assert horizontal > vertical
    # At 1 m a pixel covers roughly 1/900 m.
    assert camera.metres_per_pixel(1.0) == pytest.approx(1 / 900.0, rel=1e-9)


# --------------------------------------------------------------------------- #
# Distortion
# --------------------------------------------------------------------------- #


def test_projection_with_distortion_matches_opencv(wide_camera):
    rng = np.random.default_rng(5)
    points = np.column_stack(
        (
            rng.uniform(-0.5, 0.5, 200),
            rng.uniform(-0.4, 0.4, 200),
            rng.uniform(0.6, 2.0, 200),
        )
    )
    expected, _ = cv2.projectPoints(
        points, np.zeros(3), np.zeros(3),
        wide_camera.matrix, wide_camera.distortion.reshape(1, -1),
    )
    assert np.allclose(wide_camera.project(points), expected.reshape(-1, 2), atol=1e-9)


def test_undistortion_inverts_distortion(wide_camera):
    rng = np.random.default_rng(9)
    normalised = np.column_stack(
        (rng.uniform(-0.55, 0.55, 500), rng.uniform(-0.4, 0.4, 500))
    )
    round_trip = wide_camera.undistort_normalised(
        wide_camera.distort_normalised(normalised)
    )
    assert np.max(np.abs(round_trip - normalised)) < 1e-9


def test_undistortion_matches_opencv_run_to_convergence(wide_camera):
    """Cross-check against OpenCV, and note that its default stops early.

    ``cv2.undistortPoints`` defaults to five fixed-point iterations, which on a
    wide-angle lens leaves about a sixth of a pixel of error at the frame edge.
    Iterating to convergence agrees with this implementation to 1e-9, so the
    difference is OpenCV's stopping rule rather than a disagreement about the
    model. A sixth of a pixel is around 0.2 mm on the board -- small, but it is
    free to not have it.
    """
    rng = np.random.default_rng(11)
    pixels = np.column_stack((rng.uniform(0, 1280, 300), rng.uniform(0, 720, 300)))
    mine = wide_camera.undistort_pixels(pixels)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-12)
    converged = cv2.undistortPointsIter(
        pixels.reshape(-1, 1, 2).astype(np.float64),
        wide_camera.matrix, wide_camera.distortion.reshape(1, -1),
        None, wide_camera.matrix, criteria,
    ).reshape(-1, 2)
    assert np.max(np.abs(mine - converged)) < 1e-8

    five_iterations = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2).astype(np.float64),
        wide_camera.matrix, wide_camera.distortion.reshape(1, -1),
        P=wide_camera.matrix,
    ).reshape(-1, 2)
    assert np.max(np.abs(five_iterations - converged)) > 1e-3


def test_undistortion_is_a_no_op_without_distortion(camera):
    pixels = np.array([[100.0, 200.0], [640.0, 360.0], [1200.0, 700.0]])
    assert np.allclose(camera.undistort_pixels(pixels), pixels)
    assert not camera.has_distortion


def test_distortion_grows_towards_the_image_edge(wide_camera):
    """Barrel distortion is negligible at the centre and large at the corner."""
    centre_shift = np.linalg.norm(
        wide_camera.undistort_pixels([[640.0, 360.0]])[0] - [640.0, 360.0]
    )
    corner_shift = np.linalg.norm(
        wide_camera.undistort_pixels([[20.0, 20.0]])[0] - [20.0, 20.0]
    )
    assert centre_shift < 1.0
    assert corner_shift > 20.0


def test_backprojection_inverts_projection(wide_camera):
    rng = np.random.default_rng(13)
    for _ in range(200):
        point = np.array(
            [rng.uniform(-0.6, 0.6), rng.uniform(-0.4, 0.4), rng.uniform(0.7, 2.0)]
        )
        pixel = wide_camera.project([point])[0]
        recovered = wide_camera.backproject(pixel, point[2])
        assert np.allclose(recovered, point, atol=1e-7)


def test_rays_are_unit_length(wide_camera):
    for pixel in ([0.0, 0.0], [640.0, 360.0], [1279.0, 719.0]):
        assert np.linalg.norm(wide_camera.ray(pixel)) == pytest.approx(1.0)


def test_undistortion_makes_straight_lines_straight_again(wide_camera):
    """The defining property of the model: collinear in space, collinear on film.

    A lens bends a straight edge -- the board's frame is visibly bowed in a
    wide-angle shot -- and undistorting must put it back. Measured directly on
    projected points, so this tests the model rather than the image processing.
    """
    # Twenty collinear points spanning the board, at a metre's distance.
    line = np.column_stack(
        (np.linspace(-0.35, 0.35, 20), np.full(20, 0.24), np.full(20, 1.0))
    )
    distorted = wide_camera.project(line)
    undistorted = wide_camera.undistort_pixels(distorted)

    def max_deviation_from_a_straight_line(points):
        start, end = points[0], points[-1]
        direction = end - start
        direction = direction / np.linalg.norm(direction)
        offsets = points - start
        along = offsets @ direction
        perpendicular = offsets - along[:, None] * direction
        return float(np.max(np.linalg.norm(perpendicular, axis=1)))

    bowed = max_deviation_from_a_straight_line(distorted)
    straight = max_deviation_from_a_straight_line(undistorted)
    assert bowed > 5.0, "the test lens should visibly bow a straight edge"
    assert straight < 0.01, f"still bowed by {straight:.3f} px after undistortion"


def test_undistort_image_runs_on_a_real_frame():
    """End to end on an actual rendered image, and the board survives it."""
    cfg = SceneConfig(dist_coeffs=REAL_DISTORTION, shadow_strength=0.0, seed=2)
    scene = render(cfg)
    camera = CameraModel.from_camera_info(
        scene.extra["camera_matrix"].flatten(), cfg.dist_coeffs, cfg.image_size
    )
    rectified = camera.undistort_image(scene.image)
    assert rectified.shape == scene.image.shape

    detector = DiceDetector()
    assert detector.board_quad(rectified) is not None
    assert detector.detect(rectified) is not None


# --------------------------------------------------------------------------- #
# PnP pose of the die's top face
# --------------------------------------------------------------------------- #


def _camera_model_for(scene, cfg):
    return CameraModel.from_camera_info(
        scene.extra["camera_matrix"].flatten(), cfg.dist_coeffs, cfg.image_size
    )


def _world_from_camera(scene):
    """The renderer's true extrinsics, inverted.

    Using an assumed nadir camera here instead would quietly charge PnP for the
    scene's camera tilt and offset -- which is exactly the mistake that made an
    earlier version of this comparison look four times worse than it was.
    """
    rotation_camera_from_world, _ = cv2.Rodrigues(scene.extra["rvec"])
    translation_camera_from_world = np.asarray(scene.extra["tvec"], float).reshape(3)
    rotation = rotation_camera_from_world.T
    translation = -rotation_camera_from_world.T @ translation_camera_from_world
    return rotation, translation


def _board_plane_in_camera(scene, height):
    """The plane z = height (world) expressed in the camera frame."""
    rotation_camera_from_world, _ = cv2.Rodrigues(scene.extra["rvec"])
    translation_camera_from_world = np.asarray(scene.extra["tvec"], float).reshape(3)
    point = rotation_camera_from_world @ np.array([0.0, 0.0, height]) + translation_camera_from_world
    normal = rotation_camera_from_world @ np.array([0.0, 0.0, 1.0])
    return point, normal


def test_pnp_recovers_the_true_die_pose():
    """Ground truth is known exactly, so the error can be checked outright."""
    for xy in [(0.0, 0.0), (0.22, 0.14), (-0.25, -0.15)]:
        cfg = SceneConfig(die_xy_m=xy, die_size_m=0.03, camera_height_m=1.0, seed=31)
        scene = render(cfg)
        camera = _camera_model_for(scene, cfg)

        pose = estimate_top_face_pose(
            scene.extra["top_face_px"], cfg.die_size_m, camera
        )
        assert pose is not None
        assert pose.reprojection_error < 0.5

        # The renderer's camera looks straight down from camera_height_m, so the
        # top face sits (height - die_size) below it.
        expected_depth = cfg.camera_height_m - cfg.die_size_m
        assert pose.translation[2] == pytest.approx(expected_depth, rel=0.02)

        centre = die_centre_from_top_face(pose, cfg.die_size_m)
        assert centre[2] == pytest.approx(
            expected_depth + 0.5 * cfg.die_size_m, rel=0.02
        )


def test_pnp_recovers_the_position_on_the_board():
    """Convert the camera-frame pose into board coordinates and compare."""
    for xy in [(0.0, 0.0), (0.20, 0.12), (-0.24, -0.16), (0.28, -0.10)]:
        cfg = SceneConfig(die_xy_m=xy, die_size_m=0.03, camera_height_m=1.0, seed=37)
        scene = render(cfg)
        camera = _camera_model_for(scene, cfg)

        pose = estimate_top_face_pose(
            scene.extra["top_face_px"], cfg.die_size_m, camera
        )
        assert pose is not None
        centre_camera = die_centre_from_top_face(pose, cfg.die_size_m)

        rotation, translation = _world_from_camera(scene)
        centre_board = pose_in_frame(centre_camera, rotation, translation)

        error = np.hypot(centre_board[0] - xy[0], centre_board[1] - xy[1])
        assert error < 0.003, f"die at {xy}: {1000 * error:.1f} mm"
        assert centre_board[2] == pytest.approx(0.5 * cfg.die_size_m, abs=0.003)


@pytest.mark.parametrize("yaw_deg", [0, 11, 23, 37, 44])
def test_pnp_recovers_the_yaw_modulo_ninety(yaw_deg):
    cfg = SceneConfig(
        die_yaw_rad=np.deg2rad(yaw_deg), die_size_m=0.03, camera_height_m=1.0, seed=41
    )
    scene = render(cfg)
    camera = _camera_model_for(scene, cfg)
    pose = estimate_top_face_pose(scene.extra["top_face_px"], cfg.die_size_m, camera)
    assert pose is not None

    # The renderer's camera flips Y, so a board yaw appears negated.
    expected = -np.deg2rad(yaw_deg)
    error = (pose.yaw_about_normal() - expected + np.pi / 4) % (np.pi / 2) - np.pi / 4
    assert abs(np.degrees(error)) < 3.0


def test_pnp_handles_lens_distortion():
    """Ignoring distortion should measurably hurt; passing it should not."""
    cfg = SceneConfig(
        die_xy_m=(0.28, 0.17), die_size_m=0.03, camera_height_m=0.95,
        dist_coeffs=REAL_DISTORTION, seed=43,
    )
    scene = render(cfg)
    corners = scene.extra["top_face_px"]

    honest = _camera_model_for(scene, cfg)
    ignoring = honest.without_distortion()

    good = estimate_top_face_pose(corners, cfg.die_size_m, honest)
    bad = estimate_top_face_pose(corners, cfg.die_size_m, ignoring)
    assert good is not None and bad is not None

    rotation, translation = _world_from_camera(scene)

    def board_error(pose):
        centre = pose_in_frame(
            die_centre_from_top_face(pose, cfg.die_size_m), rotation, translation
        )
        return float(np.hypot(centre[0] - cfg.die_xy_m[0], centre[1] - cfg.die_xy_m[1]))

    assert board_error(good) < board_error(bad)
    assert board_error(good) < 0.004


def test_pnp_reports_the_planar_flip_ambiguity():
    """A square seen near head-on has two poses; both must be offered."""
    cfg = SceneConfig(die_size_m=0.03, camera_height_m=1.0, seed=47)
    scene = render(cfg)
    camera = _camera_model_for(scene, cfg)
    pose = estimate_top_face_pose(scene.extra["top_face_px"], cfg.die_size_m, camera)
    assert pose is not None
    assert pose.alternative is not None
    assert pose.reprojection_error <= pose.alternative.reprojection_error


def test_pnp_rejects_degenerate_input(camera):
    assert estimate_top_face_pose([[0, 0], [1, 0], [1, 1]], 0.03, camera) is None
    assert estimate_top_face_pose(
        [[0, 0], [1, 0], [1, 1], [0, 1]], 0.03, camera
    ) is None  # a 1-pixel square is not a pose
    assert estimate_top_face_pose(
        [[0, 0], [100, 0], [100, 100], [0, 100]], 0.0, camera
    ) is None


def test_pnp_rejects_a_bad_corner_fit(camera):
    """A quad that is not a projected square must fail the residual check."""
    skewed = [[500.0, 300.0], [700.0, 305.0], [690.0, 500.0], [400.0, 380.0]]
    pose = estimate_top_face_pose(skewed, 0.03, camera, max_reprojection_error=0.5)
    assert pose is None


def test_pnp_matches_the_detector_end_to_end():
    """Detector corners -> PnP -> board position, with nothing hand-fed."""
    cfg = SceneConfig(die_xy_m=(0.18, -0.11), die_size_m=0.03, camera_height_m=1.0, seed=53)
    scene = render(cfg)
    camera = _camera_model_for(scene, cfg)

    detection = DiceDetector().detect(scene.image)
    assert detection is not None

    pose = estimate_top_face_pose(detection.quad, cfg.die_size_m, camera)
    assert pose is not None

    rotation, translation = _world_from_camera(scene)
    on_plane = refine_pose_onto_plane(
        pose, *_board_plane_in_camera(scene, cfg.die_size_m)
    )
    centre = pose_in_frame(
        die_centre_from_top_face(on_plane, cfg.die_size_m), rotation, translation
    )
    error = float(np.hypot(centre[0] - cfg.die_xy_m[0], centre[1] - cfg.die_xy_m[1]))
    assert error < 0.008, f"{1000 * error:.1f} mm end to end"


def test_the_plane_constraint_beats_raw_pnp_on_detector_corners():
    """PnP's range comes from apparent size, and the size is biased; fix it.

    The detector's quad is systematically a little larger than the true top
    face, so PnP places the die too close. Sliding the pose along its own
    viewing ray onto the known board plane discards that bad range estimate and
    keeps the well-conditioned bearing.
    """
    detector = DiceDetector()
    raw_errors, refined_errors, depth_bias = [], [], []

    for xy in [(0.0, 0.0), (0.22, 0.14), (-0.25, -0.15), (0.28, -0.10), (-0.18, 0.16)]:
        cfg = SceneConfig(die_xy_m=xy, die_size_m=0.03, camera_height_m=1.0, seed=59)
        scene = render(cfg)
        camera = _camera_model_for(scene, cfg)
        detection = detector.detect(scene.image)
        assert detection is not None

        pose = estimate_top_face_pose(detection.quad, cfg.die_size_m, camera)
        assert pose is not None

        rotation, translation = _world_from_camera(scene)

        def error(candidate):
            centre = pose_in_frame(
                die_centre_from_top_face(candidate, cfg.die_size_m), rotation, translation
            )
            return float(np.hypot(centre[0] - xy[0], centre[1] - xy[1]))

        on_plane = refine_pose_onto_plane(
            pose, *_board_plane_in_camera(scene, cfg.die_size_m)
        )
        raw_errors.append(error(pose))
        refined_errors.append(error(on_plane))
        depth_bias.append(pose.translation[2] / (cfg.camera_height_m - cfg.die_size_m) - 1.0)

    assert np.median(depth_bias) < -0.02, "expected PnP to place the die too close"
    assert np.median(refined_errors) < np.median(raw_errors)
    assert np.median(refined_errors) < 0.006


def test_refining_onto_a_parallel_plane_is_a_no_op(camera):
    from dice_vision.pose_estimation import SquarePose

    pose = SquarePose(np.eye(3), np.array([0.1, 0.0, 1.0]), 0.2)
    unchanged = refine_pose_onto_plane(pose, [0, 0, 1.0], [1.0, 0.0, 0.0])
    assert np.allclose(unchanged.translation, pose.translation)
