"""Colour-agnostic detection of a die on the DRIMS green board.

Pure OpenCV/NumPy: no ROS import anywhere in this module, so the whole
perception pipeline can be developed and regression-tested offline against
recorded bags or synthetic imagery (``tests/test_detector.py``).

Pipeline
--------
1. **Find the board, not the die.**  The DRIMS cells all use the same green
   board, and the board is by far the largest saturated region in the frame.
   Segmenting the *board* and then treating "everything on the board that is not
   board-coloured" as foreground makes the detector independent of the die's
   colour -- the hands-on slides warn the die is "yellow (maybe)", and the final
   setup photo shows black, blue, red, orange and green dice on the same board.
2. **Model the board's colour, don't hard-code it.**  A fixed green hue window
   fails on the one case that matters most: a *green* die on a green board.  So
   a coarse green window is only used to find the board region, after which the
   board's own median chromaticity is measured and used as the classifier.  A
   dark green die differs from the board in chromaticity even though it shares
   the hue bin, so it separates cleanly.
3. **Reject shadows.**  Chromaticity (r, g normalised by intensity) is
   approximately illumination-invariant, so a shadow on the board keeps the
   board's chromaticity and stays classified as board.  That is what stops the
   strong cast shadow visible in the course imagery from being detected as a
   second die, and it is why the classifier is built on chromaticity rather than
   on HSV value.
4. **Pick the die.**  Among the foreground blobs, score by area plausibility,
   squareness of the minimum-area rectangle, and fill ratio.  A die viewed from
   above is a convex quadrilateral that nearly fills its own bounding box; the
   robot arm, cables and board clamps are not.
5. **Recover the top face from the silhouette.**  Away from the image centre a
   cube shows its side faces, so the blob is a hexagon, not the top square: its
   centroid is biased towards the board and its rectified patch is contaminated
   by shaded side faces, which wrecks pip counting.  :func:`top_face_from_hull`
   undoes that exactly -- see its docstring.
6. **Read the face.**  Rectify the *top face* to a canonical square, then count
   pips with a polarity-agnostic threshold so that dark pips on a light die and
   light pips on a dark die are both handled, and validate the count against the
   six canonical pip layouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

__all__ = [
    "DetectorConfig",
    "Detection",
    "DiceDetector",
    "PIP_LAYOUTS",
    "top_face_from_hull",
]


# Canonical pip layouts on a 3x3 grid, in units of "cells from centre".
PIP_LAYOUTS = {
    1: [(0, 0)],
    2: [(-1, -1), (1, 1)],
    3: [(-1, -1), (0, 0), (1, 1)],
    4: [(-1, -1), (-1, 1), (1, -1), (1, 1)],
    5: [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)],
    6: [(-1, -1), (-1, 0), (-1, 1), (1, -1), (1, 0), (1, 1)],
}


@dataclass
class DetectorConfig:
    """Tunables.  The defaults work on the DRIMS boards and on synthetic data."""

    # --- board segmentation (OpenCV HSV: H in 0..179, S and V in 0..255) ---
    board_hue_range: Tuple[int, int] = (30, 95)
    board_sat_min: int = 45
    board_val_min: int = 25
    board_erode_px: int = 12

    # --- board colour model ---
    #: Maximum chromaticity distance from the board's median for a pixel to
    #: still count as board.  Chromaticity is (R, G) / (R+G+B), so this is in
    #: units of the 2-simplex; 0.05 is roughly "same colour, any brightness".
    chroma_tolerance: float = 0.055
    #: Pixels darker than this fraction of the board's median intensity are
    #: forced to foreground because chromaticity becomes meaningless as the
    #: pixel approaches black.  Keep it *low*: chromaticity already separates a
    #: black die (neutral) from a deep shadow (still board-green), and a high
    #: value pulls the umbra of the cast shadow into the die blob, which shifts
    #: the fitted quad and misaligns the pip probes.
    dark_pixel_fraction: float = 0.10
    #: Absolute floor on R+G+B (out of 765) below which chromaticity is noise.
    dark_pixel_floor: float = 42.0

    # --- top-face recovery ---
    #: Principal point in pixels.  Defaults to the image centre when None.
    nadir_px: Optional[Tuple[float, float]] = None
    #: Below this sweep length (in pixels) the silhouette is treated as the top
    #: face directly; the correction would be noise.
    min_sweep_px: float = 1.5

    # --- die candidate filtering ---
    min_die_area_frac: float = 2.0e-5  # of the board area
    max_die_area_frac: float = 8.0e-2
    min_squareness: float = 0.62  # short edge / long edge of the min-area rect
    min_fill: float = 0.68  # contour area / min-area-rect area
    open_px: int = 3
    close_px: int = 3
    #: Opening sizes tried in order when separating the die from clutter it is
    #: touching.  The first size that produces a valid candidate is used.
    open_px_ladder: Tuple[int, ...] = (3, 7, 11)

    # --- pip counting ---
    patch_px: int = 96
    pip_inset: float = 0.14  # fraction of the patch trimmed off each edge
    min_pip_area_frac: float = 0.004  # of the (trimmed) patch area
    max_pip_area_frac: float = 0.13
    #: Geometry of the DRIMS die model, as fractions of the die edge; these
    #: are the defaults of drims_dice_simulator's dice_spawner_parameters.yaml.
    pip_spacing: float = 0.26
    pip_diameter: float = 0.21
    #: Minimum grey-level spread across the face before a reading is attempted.
    min_pip_contrast: float = 25.0
    #: Fraction of the nominal pip radius used as the probe disk; a smaller
    #: probe is less sensitive to residual scale and centring error.
    pip_probe_shrink: float = 0.72
    #: Grid centre offsets searched, as a fraction of the patch side, to absorb
    #: a small centring error in the fitted quad.
    pip_shift_search: float = 0.05
    #: Scale factors searched to absorb error in the estimated die size.
    pip_scale_search: Tuple[float, ...] = (0.78, 0.86, 0.94, 1.02, 1.10)
    #: Minimum layout score (pip positions vs. the other grid positions).
    min_layout_score: float = 0.18
    #: Score gap that counts as full confidence in the winning hypothesis.
    confidence_margin: float = 0.10

    def board_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lo = np.array([self.board_hue_range[0], self.board_sat_min, self.board_val_min])
        hi = np.array([self.board_hue_range[1], 255, 255])
        return lo.astype(np.uint8), hi.astype(np.uint8)


@dataclass
class Detection:
    """One detected die."""

    face_value: int
    center_px: Tuple[float, float]
    yaw_rad: float
    size_px: float
    quad: np.ndarray  # (4, 2) float32, oriented bounding box corners
    pips_px: List[Tuple[float, float]] = field(default_factory=list)
    face_confidence: float = 0.0
    blob_score: float = 0.0

    @property
    def found_face(self) -> bool:
        return 1 <= self.face_value <= 6


def _odd_kernel(size: int) -> np.ndarray:
    size = max(1, int(size) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _circularity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    perim = cv2.arcLength(contour, True)
    if perim <= 1e-6:
        return 0.0
    return float(4.0 * np.pi * area / (perim * perim))


class DiceDetector:
    """Stateless detector; call :meth:`detect` per frame."""

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.cfg = config or DetectorConfig()
        self._mask_cache: dict = {}
        self._layout_cache: dict = {}

    # ------------------------------------------------------------------ #
    # Board
    # ------------------------------------------------------------------ #

    @staticmethod
    def _chromaticity(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(chroma, intensity)`` where chroma is (R, G) / (R+G+B).

        Dividing out intensity removes most of the effect of the uneven lighting
        and cast shadows that the DRIMS boards suffer from, which is precisely
        the invariance the board classifier needs.
        """
        img = bgr.astype(np.float32)
        intensity = img.sum(axis=2)
        safe = np.maximum(intensity, 1.0)
        chroma = np.dstack((img[:, :, 2] / safe, img[:, :, 1] / safe))
        return chroma, intensity

    def _board_colour_model(
        self, bgr: np.ndarray
    ) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
        """Locate the board coarsely, then measure its own colour statistics.

        Returns ``(median_chroma, median_intensity, coarse_board_mask)`` or
        ``None`` when no board-like region is present.
        """
        cfg = self.cfg
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lo, hi = cfg.board_bounds()
        coarse = cv2.inRange(hsv, lo, hi)
        coarse = cv2.morphologyEx(coarse, cv2.MORPH_CLOSE, _odd_kernel(9))
        coarse = cv2.morphologyEx(coarse, cv2.MORPH_OPEN, _odd_kernel(5))

        count, labels, stats, _ = cv2.connectedComponentsWithStats(coarse, 8)
        if count <= 1:
            return None
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        region = labels == largest
        if np.count_nonzero(region) < 1000:
            return None

        chroma, intensity = self._chromaticity(bgr)
        # Erode before sampling so the board's own edge pixels, which blend into
        # the aluminium frame, do not drag the median around.
        core = cv2.erode(region.astype(np.uint8) * 255, _odd_kernel(9)) > 0
        if np.count_nonzero(core) < 500:
            core = region

        median_chroma = np.median(chroma[core], axis=0)
        median_intensity = float(np.median(intensity[core]))
        return median_chroma, median_intensity, region.astype(np.uint8) * 255

    def board_colour_mask(self, bgr: np.ndarray) -> np.ndarray:
        """Every pixel whose colour matches the board, shadows included."""
        cfg = self.cfg
        model = self._board_colour_model(bgr)
        if model is None:
            return np.zeros(bgr.shape[:2], np.uint8)
        median_chroma, median_intensity, _ = model

        chroma, intensity = self._chromaticity(bgr)
        distance = np.linalg.norm(chroma - median_chroma[None, None, :], axis=2)
        matches = distance < cfg.chroma_tolerance
        # Very dark pixels have unreliable chromaticity: call them foreground.
        matches &= intensity > max(
            cfg.dark_pixel_floor, cfg.dark_pixel_fraction * median_intensity
        )
        return (matches.astype(np.uint8)) * 255

    def board_mask(self, bgr: np.ndarray) -> np.ndarray:
        """Region of interest: the board outline with all holes filled.

        The convex hull of the board's largest component fills in the die-shaped
        hole (and any other object on the board), which is what turns the colour
        mask into a region of interest.
        """
        colour = self.board_colour_mask(bgr)
        if not colour.any():
            return np.zeros(bgr.shape[:2], np.uint8)

        cleaned = cv2.morphologyEx(colour, cv2.MORPH_OPEN, _odd_kernel(5))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
        if count <= 1:
            return np.zeros(bgr.shape[:2], np.uint8)
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        board = np.where(labels == largest, 255, 0).astype(np.uint8)

        contours, _ = cv2.findContours(board, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return board
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        filled = np.zeros_like(board)
        cv2.fillConvexPoly(filled, hull, 255)
        return filled

    def board_quad(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        """Four corners of the board, ordered TL, TR, BR, BL.

        Handy for building an image-to-board homography without hand-clicking
        points (see :mod:`dice_vision.board_geometry`).
        """
        mask = self.board_mask(bgr)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4:
            approx = cv2.boxPoints(cv2.minAreaRect(contour)).reshape(-1, 1, 2)
        return _order_quad(approx.reshape(-1, 2).astype(np.float32))

    # ------------------------------------------------------------------ #
    # Die
    # ------------------------------------------------------------------ #

    def foreground_mask(
        self, bgr: np.ndarray, open_px: Optional[int] = None
    ) -> Tuple[np.ndarray, float]:
        """Objects sitting on the board.  Returns ``(mask, board_area_px)``."""
        cfg = self.cfg
        board = self.board_mask(bgr)
        board_area = float(np.count_nonzero(board))
        if board_area < 1.0:
            # No board in view: fall back to the whole frame so the detector
            # still works on cropped or synthetic close-ups.
            board = np.full(bgr.shape[:2], 255, np.uint8)
            board_area = float(board.size)
            roi = board
        else:
            roi = cv2.erode(board, _odd_kernel(cfg.board_erode_px))

        is_board_colour = self.board_colour_mask(bgr)

        fg = cv2.bitwise_and(roi, cv2.bitwise_not(is_board_colour))
        fg = cv2.morphologyEx(
            fg, cv2.MORPH_OPEN, _odd_kernel(cfg.open_px if open_px is None else open_px)
        )
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _odd_kernel(cfg.close_px))
        return fg, board_area

    def _score_candidate(
        self, contour: np.ndarray, board_area: float
    ) -> Optional[Tuple[float, tuple]]:
        cfg = self.cfg
        area = cv2.contourArea(contour)
        if area <= 0:
            return None
        frac = area / board_area
        if not (cfg.min_die_area_frac <= frac <= cfg.max_die_area_frac):
            return None

        rect = cv2.minAreaRect(contour)
        (w, h) = rect[1]
        if w < 4 or h < 4:
            return None
        squareness = min(w, h) / max(w, h)
        if squareness < cfg.min_squareness:
            return None
        fill = area / (w * h)
        if fill < cfg.min_fill:
            return None

        # Prefer square, well-filled, reasonably sized blobs.
        score = squareness * fill * float(np.clip(np.log10(frac / cfg.min_die_area_frac), 0.1, 3.0))
        return score, rect

    def detect(self, bgr: np.ndarray) -> Optional[Detection]:
        """Detect the most die-like object on the board, or ``None``."""
        if bgr is None or bgr.size == 0:
            return None

        # A die resting against a cable or a clamp merges with it into one long,
        # unsquare blob and gets rejected.  Re-running with a larger opening
        # severs the thin connection while leaving the die, which is solid,
        # essentially intact.  Candidates rescued that way must actually read as
        # a die before they are accepted, otherwise a chopped-up cable fragment
        # would be reported on an empty board.
        for rung, open_px in enumerate(self.cfg.open_px_ladder):
            fg, board_area = self.foreground_mask(bgr, open_px=open_px)
            contours, _ = cv2.findContours(
                fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            best = None
            for contour in contours:
                scored = self._score_candidate(contour, board_area)
                if scored is None:
                    continue
                if best is None or scored[0] > best[0]:
                    best = (scored[0], scored[1], contour)

            if best is None:
                continue

            detection = self._build_detection(bgr, *best)
            if detection is None:
                continue
            if rung == 0 or detection.found_face:
                return detection

        return None

    def _build_detection(
        self, bgr: np.ndarray, score: float, silhouette_rect, contour: np.ndarray
    ) -> Optional[Detection]:
        nadir = self.cfg.nadir_px or (bgr.shape[1] / 2.0, bgr.shape[0] / 2.0)
        hull = cv2.convexHull(contour).reshape(-1, 2).astype(np.float32)
        top_face = top_face_from_hull(hull, nadir, self.cfg.min_sweep_px)

        # Fit the actual quadrilateral rather than a bounding rectangle.  The
        # top face of a die is a quadrilateral under perspective, and -- more
        # importantly -- a bounding rectangle is inflated by any small
        # protrusion on the blob (a sliver of cast shadow, a morphology
        # artefact), which scales the rectified patch and slides the pip probes
        # off the pips.  Polygon simplification absorbs such bumps.
        quad = _fit_quad(top_face)
        if quad is None:
            quad = _order_quad(cv2.boxPoints(cv2.minAreaRect(top_face)).astype(np.float32))

        edges = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
        if float(edges.min()) < 4.0:
            quad = _order_quad(cv2.boxPoints(silhouette_rect).astype(np.float32))
            edges = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
            if float(edges.min()) < 4.0:
                return None

        cx, cy = (float(v) for v in quad.mean(axis=0))

        # A square has 90-degree symmetry, so the yaw is only defined modulo 90.
        edge = quad[1] - quad[0]
        yaw = float(np.arctan2(edge[1], edge[0]))
        yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4

        face, confidence, pips = self.read_face(bgr, quad)

        return Detection(
            face_value=face,
            center_px=(cx, cy),
            yaw_rad=yaw,
            size_px=float(edges.mean()),
            quad=quad,
            pips_px=pips,
            face_confidence=confidence,
            blob_score=float(score),
        )

    # ------------------------------------------------------------------ #
    # Pips
    # ------------------------------------------------------------------ #

    def read_face(
        self, bgr: np.ndarray, quad: np.ndarray
    ) -> Tuple[int, float, List[Tuple[float, float]]]:
        """Read the number of pips on the rectified top face.

        Returns ``(face_value, confidence, pip_centres_in_image_px)``.  A value
        of ``0`` means the face could not be read reliably.

        Counting blobs is the obvious approach, and it is what the hands-on
        slides suggest (``findContours`` / ``contourArea``), but it is fragile
        here: the DRIMS die model puts the pips 0.26 die-widths apart with a
        diameter of 0.21, so neighbouring pips are separated by only 0.05 of the
        die -- about one pixel on a 25 px die.  They merge, and the count
        collapses.  Face 6, whose two columns of three are the tightest pattern,
        fails first.

        Instead the face is read by sampling the nine cells of the pip grid and
        matching that nine-vector against the six canonical layouts over all
        four 90-degree rotations.  Merged pips stop mattering, because each cell
        is integrated rather than segmented.

        Two details matter for correctness:

        * **The grid scale is estimated once, before any layout is scored.**
          Letting each hypothesis pick its own scale is subtly wrong: layout 1
          can shrink the grid until its eight "empty" cells slide off the real
          pips, inflating its score until it beats the truth.  That is exactly
          how faces 3 and 5 came to be read as 1 in testing.  The scale is
          chosen instead by maximising the variance across the nine cells --
          a hypothesis-free criterion for "the grid is lined up with whatever
          is printed on this face".
        * **Both polarities are tried**, so dark pips on a light die and light
          pips on a dark die are handled by the same code.
        """
        cfg = self.cfg
        n = cfg.patch_px
        dst = np.array([[0, 0], [n - 1, 0], [n - 1, n - 1], [0, n - 1]], np.float32)
        homography = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
        patch = cv2.warpPerspective(bgr, homography, (n, n), flags=cv2.INTER_CUBIC)

        inset = int(round(cfg.pip_inset * n))
        core = patch[inset : n - inset, inset : n - inset]
        if core.size < 64:
            return 0, 0.0, []

        gray = cv2.cvtColor(core, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        # Percentiles, not min/max: a single bright sliver of board left inside
        # the quad would otherwise set the top of the range and crush the real
        # pip contrast, which is how dark dice were being misread.
        lo, hi = (float(v) for v in np.percentile(gray, (4.0, 96.0)))
        if hi - lo < cfg.min_pip_contrast:
            return 0, 0.0, []  # a blank face, or the die is out of focus
        normalised = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)

        side = core.shape[0]
        cells, scale = self._best_grid(normalised, side)

        best_value, best_score, runner_up = 0, -np.inf, -np.inf
        for response in (1.0 - cells, cells):
            for value in range(1, 7):
                score = max(
                    self._layout_score(response, value, rotation)
                    for rotation in range(4)
                )
                if score > best_score:
                    runner_up, best_score, best_value = best_score, score, value
                elif score > runner_up:
                    runner_up = score

        if best_score < cfg.min_layout_score:
            return 0, 0.0, []

        margin = float(np.clip((best_score - runner_up) / cfg.confidence_margin, 0.0, 1.0))
        confidence = float(np.clip(0.5 * best_score + 0.5 * margin, 0.0, 1.0))

        centres_core = self._pip_blobs(normalised, side)
        inverse = np.linalg.inv(homography)
        if centres_core:
            pts = np.array(
                [[[c[0] + inset, c[1] + inset]] for c in centres_core], dtype=np.float32
            )
            mapped = cv2.perspectiveTransform(pts, inverse).reshape(-1, 2)
            pips = [(float(p[0]), float(p[1])) for p in mapped]
        else:
            pips = []

        return best_value, confidence, pips

    #: The nine pip-grid cells, in a fixed order, as (column, row) offsets.
    GRID_CELLS: Tuple[Tuple[int, int], ...] = (
        (-1, -1), (0, -1), (1, -1),
        (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
    )

    def _best_grid(self, normalised: np.ndarray, side: int) -> Tuple[np.ndarray, float]:
        """Sample the nine grid cells at the best-fitting scale.

        The quad comes from a morphologically processed blob, so it is typically
        a little larger than the true top face -- on a 25 px die a couple of
        pixels is already a ~10% scale error, enough to slide the probes off the
        pips.  The scale that maximises the spread of the nine cell means is the
        one whose grid is best aligned with whatever is printed on the face.
        """
        shift = self.cfg.pip_shift_search * side
        offsets = ((0.0, 0.0), (shift, 0.0), (-shift, 0.0), (0.0, shift), (0.0, -shift))

        best_cells, best_key, best_spread = None, (1.0, (0.0, 0.0)), -np.inf
        for scale in self.cfg.pip_scale_search:
            for offset in offsets:
                cells = np.array(
                    [
                        # Median, not mean: a probe that only mostly covers its
                        # pip still reports the pip's value, which keeps the
                        # corner cells (nearest the rectified edge, where blur
                        # and JPEG bleed the board in) from being dragged
                        # towards blank.
                        float(
                            np.median(
                                normalised[self._cell_mask(cell, side, scale, offset)]
                            )
                        )
                        for cell in self.GRID_CELLS
                    ]
                )
                spread = float(cells.var())
                if spread > best_spread:
                    best_cells, best_key, best_spread = cells, (scale, offset), spread
        return best_cells, best_key[0]

    def _layout_score(self, cells: np.ndarray, value: int, rotation: int) -> float:
        """Mean response on the layout's cells minus the mean on the others.

        Comparing pip positions against the *other grid positions* rather than
        against the blank face is what separates a layout from its subsets:
        1 is a subset of 3 and 5, 2 of 4 and 6, and scoring against blank face
        would rank them almost equally.
        """
        indices = self._layout_indices(value, rotation)
        mask = np.zeros(9, dtype=bool)
        mask[list(indices)] = True
        return float(cells[mask].mean() - cells[~mask].mean())

    def _layout_indices(self, value: int, rotation: int) -> Tuple[int, ...]:
        key = (value, rotation)
        cached = self._layout_cache.get(key)
        if cached is not None:
            return cached
        indices = []
        for gx, gy in PIP_LAYOUTS[value]:
            x, y = gx, gy
            for _ in range(rotation):
                x, y = -y, x
            indices.append(self.GRID_CELLS.index((x, y)))
        result = tuple(sorted(indices))
        self._layout_cache[key] = result
        return result

    def _cell_mask(
        self,
        cell: Tuple[int, int],
        side: int,
        scale: float,
        offset: Tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        """Probe disk for one grid cell, cached per (cell, size, scale, offset)."""
        key = (cell, side, round(scale, 3), (round(offset[0], 2), round(offset[1], 2)))
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached

        cfg = self.cfg
        core_fraction = 1.0 - 2.0 * cfg.pip_inset
        spacing = scale * cfg.pip_spacing / core_fraction * side
        radius = max(
            1,
            int(round(scale * cfg.pip_probe_shrink * 0.5 * cfg.pip_diameter
                      / core_fraction * side)),
        )
        mask = np.zeros((side, side), np.uint8)
        cv2.circle(
            mask,
            (int(round(side / 2.0 + cell[0] * spacing + offset[0])),
             int(round(side / 2.0 + cell[1] * spacing + offset[1]))),
            radius, 255, -1,
        )
        boolean = mask > 0
        self._mask_cache[key] = boolean
        return boolean

    def _pip_blobs(self, normalised: np.ndarray, side: int) -> List[Tuple[float, float]]:
        """Best-effort pip centres, for the debug overlay only."""
        cfg = self.cfg
        eight_bit = (normalised * 255).astype(np.uint8)
        centres: List[Tuple[float, float]] = []
        for flag in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
            _, binary = cv2.threshold(eight_bit, 0, 255, flag + cv2.THRESH_OTSU)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, _odd_kernel(3))
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            found: List[Tuple[float, float]] = []
            for contour in contours:
                frac = cv2.contourArea(contour) / float(side * side)
                if not (cfg.min_pip_area_frac <= frac <= cfg.max_pip_area_frac):
                    continue
                moments = cv2.moments(contour)
                if moments["m00"] <= 0:
                    continue
                found.append(
                    (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
                )
            if 1 <= len(found) <= 6 and len(found) > len(centres):
                centres = found
        return centres

    # ------------------------------------------------------------------ #
    # Debug overlay
    # ------------------------------------------------------------------ #

    def annotate(self, bgr: np.ndarray, detection: Optional[Detection]) -> np.ndarray:
        """Draw the detection for the ``~/debug_image`` topic.

        The overlay scales with the die, so it stays readable whether the die
        fills the frame or is 20 px across on a 1280 px board.
        """
        out = bgr.copy()
        quad = self.board_quad(bgr)
        if quad is not None:
            cv2.polylines(out, [quad.astype(int)], True, (120, 120, 120), 1, cv2.LINE_AA)

        if detection is None:
            cv2.putText(out, "no die", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2, cv2.LINE_AA)
            return out

        half = 0.5 * detection.size_px
        thickness = max(1, int(round(detection.size_px / 22.0)))
        font_scale = float(np.clip(detection.size_px / 90.0, 0.35, 1.0))
        cx, cy = detection.center_px

        cv2.polylines(out, [detection.quad.astype(int)], True, (0, 0, 255),
                      thickness, cv2.LINE_AA)

        # Die axes: red along the face edge, green perpendicular to it.
        for angle, colour in (
            (detection.yaw_rad, (40, 40, 255)),
            (detection.yaw_rad + np.pi / 2, (60, 200, 60)),
        ):
            tip = (int(cx + half * np.cos(angle)), int(cy + half * np.sin(angle)))
            cv2.arrowedLine(out, (int(cx), int(cy)), tip, colour, thickness,
                            cv2.LINE_AA, tipLength=0.3)

        for px, py in detection.pips_px:
            cv2.circle(out, (int(px), int(py)), max(1, thickness), (0, 220, 255),
                       -1, cv2.LINE_AA)

        label = f"{detection.face_value}" if detection.found_face else "?"
        text = f"face {label}  {detection.face_confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        origin = (int(cx - tw / 2), int(cy - half - 8))
        cv2.rectangle(out, (origin[0] - 3, origin[1] - th - 4),
                      (origin[0] + tw + 3, origin[1] + 4), (255, 255, 255), -1)
        cv2.putText(out, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (10, 10, 10), 1, cv2.LINE_AA)
        return out


def top_face_from_hull(
    hull: np.ndarray, nadir: Sequence[float], min_sweep_px: float = 1.5
) -> np.ndarray:
    """Recover a cube's top face from its silhouette.

    Under a pinhole camera, a cube standing on a plane projects to the convex
    hull of two squares: the top face, and the bottom face translated *towards*
    the principal point (the top face is nearer the camera, so it is magnified
    outwards).  Writing ``u`` for the unit vector from the nadir to the blob and
    ``k`` for the length of that translation, the silhouette is exactly the
    Minkowski sum

        ``hull = top (+) segment[-k*u, 0]``

    Intersecting the hull with a copy of itself shifted by ``+k*u`` gives

        ``hull ∩ (hull + k*u) = top (+) (segment[-k*u,0] ∩ segment[0,k*u]) = top``

    so the top face comes back exactly, with no camera calibration needed.  The
    only unknown is ``k``, and that follows from the square's own symmetry: a
    square has the same extent along any two perpendicular directions, and the
    sweep only stretches the silhouette along ``u``, so

        ``k = extent_along(u) - extent_along(u_perp)``.

    This matters because using the raw silhouette instead would bias the die's
    centre towards the board centre by up to half the sweep -- several
    millimetres at the edge of the board -- and would feed shaded side faces
    into the pip-counting patch, which is what actually breaks the face reading.
    """
    pts = np.asarray(hull, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return pts

    centroid = pts.mean(axis=0)
    radial = centroid - np.asarray(nadir, dtype=np.float32)
    norm = float(np.linalg.norm(radial))
    if norm < 1e-6:
        return pts  # directly under the camera: no side faces visible

    u = radial / norm
    u_perp = np.array([-u[1], u[0]], dtype=np.float32)

    along = pts @ u
    across = pts @ u_perp
    sweep = float((along.max() - along.min()) - (across.max() - across.min()))
    if sweep < min_sweep_px:
        return pts

    shifted = (pts + sweep * u).astype(np.float32)
    area, intersection = cv2.intersectConvexConvex(pts, shifted)
    if intersection is None or len(intersection) < 3 or area <= 0:
        return pts
    return intersection.reshape(-1, 2).astype(np.float32)


def _fit_quad(polygon: np.ndarray) -> Optional[np.ndarray]:
    """Simplify a convex polygon to four corners, or return ``None``."""
    pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    if len(pts) < 4:
        return None
    perimeter = cv2.arcLength(pts, True)
    for epsilon in (0.02, 0.035, 0.05, 0.07):
        approx = cv2.approxPolyDP(pts, epsilon * perimeter, True)
        if len(approx) != 4:
            continue
        quad = _order_quad(approx.reshape(-1, 2).astype(np.float32))
        edges = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
        if edges.min() < 4.0:
            continue
        # Only trust the simplification if it still looks like a square seen
        # under mild perspective.  A sloppy fit rectifies the patch with a
        # skew, which is worse than an honest bounding rectangle.
        if edges.max() / edges.min() > 1.30:
            continue
        rect_area = cv2.contourArea(cv2.boxPoints(cv2.minAreaRect(pts)))
        if rect_area > 0 and cv2.contourArea(quad) / rect_area < 0.80:
            continue
        return quad
    return None


def _order_quad(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    centre = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    ordered = pts[np.argsort(angles)]
    # Rotate so the first corner is the one closest to the top-left.
    start = int(np.argmin(ordered.sum(axis=1)))
    return np.roll(ordered, -start, axis=0)
