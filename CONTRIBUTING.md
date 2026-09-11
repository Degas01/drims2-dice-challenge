# Working on this repository

## Run the tests without ROS

The perception and planning modules deliberately import no ROS, so the whole
test suite runs on any machine:

```bash
pip install opencv-python-headless numpy pytest
pytest -q                          # ~4 minutes, 355 tests
pytest tests/test_die_model.py -q  # the planners alone, under a second
```

CI runs exactly this on every push.

## Regenerate the README figures

Every image in the README is produced from the code, so it cannot drift away
from what the detector actually does:

```bash
python3 scripts/make_docs_images.py
```

## Where to put a change

| If it has no ROS import today, keep it that way | |
| --- | --- |
| `src/dice_vision/detector.py`, `colour.py`, `camera_model.py`, `board_geometry.py`, `pose_estimation.py`, `scene_simulator.py` | perception |
| `src/dice_task/die_model.py`, `grasping.py`, `trajectory.py`, `cartesian_executor.py` | decision and motion generation |

The two ROS nodes (`dice_vision_node.py`, `dice_task_node.py`) are wrappers:
parameters, topics, services, TF. Logic that lands in them cannot be tested
offline, so it belongs one level down wherever possible.

## Before opening a pull request

* `pytest -q` green;
* new behaviour has a test that fails without it;
* if the change came from a real failure, add it to
  [`docs/ENGINEERING_LOG.md`](docs/ENGINEERING_LOG.md) — symptom, cause,
  evidence. That file is the most useful thing in the repository for anyone
  picking this up.
