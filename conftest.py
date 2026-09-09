"""Make the two ROS packages importable when running pytest from the repo root.

The offline test-suite deliberately does not need a ROS installation: it imports
``dice_vision.detector``, ``dice_vision.board_geometry``, ``dice_task.die_model``
and ``dice_task.grasping``, none of which import rclpy.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
for path in (ROOT / "src" / "dice_vision", ROOT / "src" / "dice_task", ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
