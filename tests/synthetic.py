"""Test-suite alias for the scene renderer.

The renderer lives in the package (``dice_vision.scene_simulator``) because the
fake-camera node uses it to drive the vision pipeline in simulation. The tests
imported it as ``synthetic`` first, so this keeps that name working rather than
touching every test file.
"""

from dice_vision.scene_simulator import *  # noqa: F401,F403
from dice_vision.scene_simulator import (  # noqa: F401
    BOARD_SIZE_M,
    DIE_PALETTE,
    PIP_LAYOUTS,
    Scene,
    SceneConfig,
    random_scene,
    render,
)
