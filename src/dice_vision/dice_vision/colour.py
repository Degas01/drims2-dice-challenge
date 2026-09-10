"""Naming the die's colour.

The hands-on slides describe the yellow die and then immediately hedge --
"Yellow dice with black dots (**maybe**)" -- and the final photo of the real
setup shows black, red, blue, orange and green dice piled together.  So the
detector deliberately never *searches* for a colour: it finds the die by
subtracting the board, which works whatever the die is made of.

That leaves the colour as something to *report* rather than something to rely
on, which is the useful way round.  This module reads the die's body colour out
of the image and gives it a name, so the overlay and the logs can say "yellow
die, face 5" and a mistake is obvious at a glance.

Method
------
A die face is made of exactly two colours: the body and the pips.  So the body
colour is found by splitting the pixels inside the top face into **two clusters**
and keeping the larger one.  That is worth the extra dozen lines over a plain
median: on a six-pip face the pips cover enough of the sampled area to drag a
median a long way towards them, and the answer that came back for a blue die
with white pips was "white".  Two-means has no such failure mode -- the pips
form their own cluster and are discarded whole, however many of them there are.

Convert that one pixel value to HSV, which separates the two questions the
lecture asks you to separate:

* **Is there a colour at all?**  Saturation answers that.  Black, grey and white
  dice have no hue worth reading -- their hue channel is pure noise -- so they
  are named from value instead.
* **Which colour?**  Hue answers that, and it is largely invariant to the
  lighting gradient and cast shadow that vary across the DRIMS boards, which is
  exactly why the classification survives conditions that a raw BGR nearest-
  neighbour would not.

Confidence is the distance from the nearest band edge, so a hue sitting between
"orange" and "yellow" reports a low number rather than a confident wrong answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

__all__ = ["ColourReading", "HUE_BANDS", "classify_bgr", "body_colour", "read_die_colour"]


#: Hue is an angle, and the bands are arcs of it: ``(name, low, high)`` on
#: OpenCV's 0..179 scale, read modulo 180 so red's arc can run 170 -> 8 through
#: zero as a single band instead of being split in two.  Splitting it is the
#: tempting shortcut and it quietly ruins the confidence number: a perfect red
#: sits at hue 179, which is dead centre of the real arc but hard against the
#: edge of a 170..180 half-band, so it would report as barely-red.
#:
#: The arcs are contiguous and cover the whole circle, so every hue gets a name.
HUE_BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("red", 170.0, 188.0),
    ("orange", 8.0, 20.0),
    ("yellow", 20.0, 34.0),
    ("green", 34.0, 85.0),
    ("cyan", 85.0, 100.0),
    ("blue", 100.0, 130.0),
    ("purple", 130.0, 152.0),
    ("pink", 152.0, 170.0),
)

#: Hue margin, in degrees of OpenCV's half-circle, at which a name counts as
#: fully confident.  The measured hue of a flat painted face is stable to about
#: a degree, so six is comfortably several standard deviations.
HUE_MARGIN_FOR_CONFIDENCE = 6.0


@dataclass(frozen=True)
class ColourReading:
    """The die's body colour, as a name and as the numbers behind it."""

    name: str
    bgr: Tuple[int, int, int]
    hsv: Tuple[int, int, int]
    confidence: float

    @property
    def is_chromatic(self) -> bool:
        return self.name not in ("black", "grey", "white")

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.name} ({self.confidence:.2f})"


def _find_band(hue: float) -> Tuple[str, float]:
    """Name a hue, and say how many degrees it is from being named otherwise.

    The margin -- distance to the nearer end of the arc -- is a better confidence
    than distance from the arc's centre, because the arcs are not the same width.
    Green spans fifty degrees and blue thirty, so a hue ten degrees off centre
    means something quite different in each; ten degrees clear of the boundary
    means the same thing in both.
    """
    for name, low, high in HUE_BANDS:
        if (hue - low) % 180.0 < (high - low):
            margin = min((hue - low) % 180.0, (high - hue) % 180.0)
            return name, float(margin)
    return "unknown", 0.0  # pragma: no cover - the arcs cover the circle


def classify_bgr(
    bgr: Sequence[float],
    *,
    saturation_min: float = 60.0,
    black_value_max: float = 70.0,
    white_value_min: float = 170.0,
) -> ColourReading:
    """Name a single BGR colour.

    ``saturation_min`` is the line between "this has a hue" and "this is a shade
    of grey".  It has to sit low enough to catch a dark die under a shadow and
    high enough to reject the near-neutral speckle of a white one; 60 of 255
    does both on the recorded boards and on the renderer.
    """
    pixel = np.array([[list(int(round(float(v))) for v in bgr)]], dtype=np.uint8)
    h, s, v = (float(c) for c in cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0])

    if s < saturation_min:
        # No usable hue.  Value alone separates the three achromatic dice, and
        # the confidence reports how far the reading is from the two thresholds
        # so a mid-grey die does not claim to be white.
        if v <= black_value_max:
            name = "black"
            confidence = float(np.clip((black_value_max - v) / 40.0, 0.0, 1.0))
        elif v >= white_value_min:
            name = "white"
            confidence = float(np.clip((v - white_value_min) / 40.0, 0.0, 1.0))
        else:
            span = max(white_value_min - black_value_max, 1.0)
            middle = 0.5 * (black_value_max + white_value_min)
            name = "grey"
            confidence = float(np.clip(1.0 - 2.0 * abs(v - middle) / span, 0.0, 1.0))
        return ColourReading(name, _as_bgr(pixel), (int(h), int(s), int(v)), confidence)

    name, margin = _find_band(h)
    confidence = float(np.clip(margin / HUE_MARGIN_FOR_CONFIDENCE, 0.0, 1.0))

    # A weak saturation makes even a well-centred hue uncertain, because hue
    # gets noisier as a colour approaches grey.  Fold that in, so the single
    # number means "how much would I bet on this name".
    saturation_weight = float(np.clip((s - saturation_min) / 60.0, 0.0, 1.0))
    confidence *= 0.5 + 0.5 * saturation_weight
    return ColourReading(name, _as_bgr(pixel), (int(h), int(s), int(v)), confidence)


def _as_bgr(pixel: np.ndarray) -> Tuple[int, int, int]:
    b, g, r = (int(c) for c in pixel[0, 0])
    return (b, g, r)


def _dominant(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The body colour of a two-tone patch.

    Split the pixels into two colour clusters, then decide which one is the body
    by **shape**, not by pixel count: the body is a single connected region with
    holes punched in it, while the pips are several separate blobs.  So the body
    is the cluster with the largest *connected component*.

    Counting pixels instead is the obvious approach and it is wrong on exactly
    one face.  A six shows six pips, and inside the inset sampling window they
    together cover more area than the body does, so a blue die with white pips
    reported "white" -- but only ever when the six was up, which is a wonderful
    way to lose an afternoon.  Comparing the largest blob is immune: one pip is
    never bigger than the whole face around it.

    Falls back to the median when the patch is too small to cluster, or when the
    two clusters sit so close together that the face is effectively one colour
    (a blank face, or a die with engraved rather than painted pips).
    """
    pixels = bgr[mask > 0]
    samples = pixels.reshape(-1, 3).astype(np.float32)
    if samples.shape[0] < 32:
        return np.median(samples, axis=0)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
    _, labels, centres = cv2.kmeans(
        samples, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    if float(np.linalg.norm(centres[0] - centres[1])) < 30.0:
        return np.median(samples, axis=0)

    # Paint the cluster labels back into image space so they have a shape again.
    painted = np.zeros(mask.shape[:2], np.uint8)
    painted[mask > 0] = labels.ravel().astype(np.uint8) + 1

    best_label, best_area = 1, -1
    for value in (1, 2):
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            (painted == value).astype(np.uint8), 8
        )
        area = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
        if area > best_area:
            best_label, best_area = value, area

    # Report the median of the winning cluster rather than its k-means centre:
    # the centre is pulled outward by the anti-aliased pixels along every pip
    # edge, which belong to neither colour.
    return np.median(bgr[painted == best_label].reshape(-1, 3).astype(np.float32), axis=0)


def body_colour(
    bgr: np.ndarray, quad: Sequence[Sequence[float]], inset: float = 0.30
) -> Optional[np.ndarray]:
    """BGR of the die's body, sampled inside ``quad``.

    ``inset`` shrinks the quadrilateral towards its centre before sampling.  The
    edge pixels of the top face are the worst ones available -- JPEG ringing,
    the dark seam where the face meets the side, and a pixel or two of board
    bleeding in -- so they are dropped rather than clustered.
    """
    corners = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    if corners.shape[0] < 3:
        return None
    centre = corners.mean(axis=0)
    shrunk = centre + (1.0 - float(np.clip(inset, 0.0, 0.9))) * (corners - centre)

    mask = np.zeros(bgr.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, shrunk.astype(np.int32), 255)
    if not np.any(mask):
        return None
    return _dominant(bgr, mask)


def read_die_colour(
    bgr: np.ndarray, quad: Sequence[Sequence[float]], inset: float = 0.30, **kwargs
) -> Optional[ColourReading]:
    """Sample the die body inside ``quad`` and name its colour."""
    sample = body_colour(bgr, quad, inset)
    if sample is None:
        return None
    return classify_bgr(sample, **kwargs)
