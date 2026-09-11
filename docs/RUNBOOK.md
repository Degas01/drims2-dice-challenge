# Run book

Every command, in order, with what each one should print. Written to be followed
top to bottom while recording — nothing is assumed and nothing is skipped.

Two conventions used throughout:

* **host** = your WSL2 shell, prompt `giacomo@...`
* **container** = inside the DRIMS image, prompt `drims@...`

Run commands **one line at a time**. A trailing `\` that gets pasted wrong turns
the next line into an argument of the previous one, which fails silently and
wastes a rebuild.

---

## 0. One-time setup

### Start the container (host)

```bash
cd ~/DRIMS2-2026
./start.sh 34
```

`34` is the ROS domain ID; any number 0–101 works as long as it is the same for
every terminal. If this drops you into the container the prompt changes to
`drims@`; otherwise run `./connect.sh`.

### Make the workspace source itself (container, once)

```bash
echo 'source ~/drims_ws/install/setup.bash' >> ~/.bashrc
```

The DRIMS image writes its own source line into **root's** `.bashrc`, not yours,
so without this every new shell reports "package not found" even though the build
succeeded. It takes effect in *new* shells — in the one you typed it in, run
`source ~/drims_ws/install/setup.bash` by hand.

### Build (container)

```bash
cd ~/drims_ws
colcon build --packages-select dice_task dice_vision --symlink-install
source install/setup.bash
ros2 pkg executables dice_vision
```

Expected:

```
dice_vision dice_vision_node
dice_vision fake_camera_node
```

Both. If only one appears, the build is stale — `rm -rf build/dice_vision
install/dice_vision` and build again.

---

## 1. The offline test suite — no robot needed

Worth recording first: it runs anywhere, takes four minutes, and proves the parts
that do not need hardware.

```bash
cd ~/drims_ws/src/dice_task/../..     # or wherever the repo is checked out
pytest -q
```

Expected: `355 passed`.

The fast subset, if you want a short clip:

```bash
pytest tests/test_die_model.py -q     # the planners, under a second
pytest tests/test_colour.py -q        # colour naming, all 8 dice
```

---

## 2. The simulation — five terminals

Each terminal: `cd ~/DRIMS2-2026 && ./connect.sh`, then the command. Wait for the
expected output of each before starting the next.

### Terminal 1 — the cell

```bash
ros2 launch drims_description ur5e_1_start.launch.py fake:=true
```

Wait for `You can start planning now!` and for RViz to show the arm.

### Terminal 2 — the die

```bash
ros2 launch drims_dice_simulator spawn_dice.launch.py face_up:=5 position:="[-0.1, 0.58, -0.04]"
```

Expected:

```
Spawned dice with:
 - face 5 up
 - position [0.7099, 0.4106, 0.8385]
 - size 0.03
AddObject response: ... success=True
```

**Use `y` around 0.58.** The far half of the board is inside the arm's reach but
not its *dexterous* workspace — near full extension the wrist approaches a
singularity and grasp orientations lose their IK solutions while the positions
stay perfectly reachable. At `y = 0.65` the arm refuses grasp after grasp; at
`y = 0.58` the cycle runs. The node now falls back to alternative routes rather
than giving up, but it still costs time it does not need to spend.

`selected_cell:=N` does nothing — the DRIMS launch file declares it as
`selected_cell` and forwards it as `cell_id`, so it is silently dropped. The
active cell is whatever the YAML says.

### Terminal 3 — camera and vision

```bash
ros2 launch dice_vision vision_in_simulation.launch.py die_colour:=yellow
```

Expected:

```
fake camera publishing a yellow die at 10.0 Hz on ~/image_raw/compressed
camera model: f=(900.0, 900.0) px, principal point (640.0, 360.0), no distortion
```

### Terminal 4 — watch the vision

```bash
rviz2
```

**Add → By topic → `/dice_vision/mosaic` → Image → OK.**

One panel, four stages: board mask, die pixels alone, oriented bounding box, and
the readout card with the face value, the colour name, a swatch of the pixels
that name came from, the board position in metres and the yaw. This is the panel
to record.

`rqt_image_view` is *not* in the DRIMS image; rviz2 is.

Numbers without a GUI, if you prefer:

```bash
ros2 service call /dice_vision/dice_identification easy_motion_msgs/srv/DiceIdentification "{}"
```

Terminal 3 logs each call:

```
yellow die, face 5 at (-0.100, 0.580) confidence 0.97
```

### Terminal 5 — the challenge

```bash
ros2 launch dice_task dice_challenge.launch.py target_face:=3 strategy:=deduce
```

Expected, with the reasoning printed as it goes:

```
moving to home configuration
face 5 is up (target 3, bottom 2)
reasoning: target 3 is on a side; turning once about X to read a side face and pin the configuration down
plan: rotate +90deg about X [deduce], or 2 equally good alternative(s)
grasping at (-0.100, 0.580, -0.005), yaw 28.6 deg, lean -45 deg, rotate +90deg about X
face 3 is up (target 3, bottom 4)
SUCCESS: face 3 up (target 3) after 1 re-grasp(s) in 149.9 s
```

The grasp height should read about **−0.005**, the die's centre. `+0.011` means
an old build — `dice_tf` sits on the die's *top face* and the current code
subtracts half a die.

---

## 3. What to record

Five short demonstrations, in increasing order of how interesting they are.

### 3.1 The deduction, narrated

```bash
ros2 launch dice_task dice_challenge.launch.py target_face:=3 strategy:=deduce
```

The line worth pausing on is the second `reasoning:`:

```
reasoning: top was 5, X-turn showed 1: configuration is top=1, bottom=6, +X=4, -X=3, +Y=2, -Y=5
```

One 90° turn, and the robot now knows where all six faces are — because a die is
chiral, so top-plus-one-side determines the whole configuration.

### 3.2 Ground truth versus camera only

```bash
# uses the simulator's face1..6_tf frames: mean 1.0 re-grasps, never more than 2
ros2 launch dice_task dice_challenge.launch.py target_face:=6 strategy:=exact

# uses only the top face, as a real overhead camera would: mean 1.67, worst 3
ros2 launch dice_task dice_challenge.launch.py target_face:=6 strategy:=deduce
```

### 3.3 The colour sweep

Change the die's colour live, with nothing restarting:

```bash
ros2 param set /fake_camera die_colour blue
ros2 param set /fake_camera die_colour red
ros2 param set /fake_camera die_colour black
ros2 param set /fake_camera die_colour white
ros2 param set /fake_camera die_colour green_dark
```

Watch `/dice_vision/mosaic` in RViz. The readout names each colour and the swatch
matches. **`green_dark` is the one to record** — a green die on a green board is
the case a fixed hue threshold cannot survive, and the reason the detector
segments the *board* and takes everything that is not board-coloured.

### 3.4 Harsher conditions

```bash
ros2 param set /fake_camera shadow_strength 0.7
ros2 param set /fake_camera light_gradient 0.45
```

And a wide-angle lens with real barrel distortion, which needs a relaunch:

```bash
ros2 launch dice_vision vision_in_simulation.launch.py \
    die_colour:=orange dist_coeffs:="[-0.28, 0.09, 0.0, 0.0, 0.0]"
```

### 3.5 Every starting face

```bash
for f in 1 2 3 4 5 6; do
  ros2 launch drims_dice_simulator spawn_dice.launch.py face_up:=$f position:="[-0.1, 0.58, -0.04]"
  ros2 launch dice_task dice_challenge.launch.py target_face:=4 strategy:=deduce
done
```

The point of the challenge is that it works from *any* start, not one.

---

## 4. The generated trajectories

Everything above plans each waypoint to a full stop through MoveIt. To run the
trajectory generator instead — LIN path, trapezoidal profile, SLERP, blended
corners, IK per sample, one timed `JointTrajectory`:

```bash
ros2 launch dice_task dice_challenge.launch.py target_face:=2 motion_mode:=trajectory
```

It prints per segment:

```
executing 34 points over 1.68 s (2 stops, peak joint rate 1.21 rad/s)
```

`trajectory rejected: ...` means it fell back to MoveIt and said why — an
unreachable sample or an IK branch change. That is intended behaviour, not a
failure: the alternative is executing a joint path that jumps.

> **This mode bypasses MoveIt's planning-scene collision checks.** Get the task
> working in `moveit` mode first.

---

## 5. When something goes wrong

**Restart terminal 2 after any failed run.** An aborted cycle can leave the die
attached to the gripper in the planning scene, and every later run then fails for
reasons that have nothing to do with what you are testing. The symptom is
`Dice object 'dice' not found in planning scene` or `Timeout while waiting for
planning scene` in terminal 2, and `success=False` from the identification
service.

Check before blaming anything else:

```bash
ros2 service call /dice_identification easy_motion_msgs/srv/DiceIdentification "{}"
```

You need `success=True` and a sensible `face_number`. If not, nothing downstream
can work.

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Package 'dice_task' not found` | The shell never sourced the workspace. Check the error's search path — if it lists only `/home/drims/static/drims2_ws/...`, that is it. | `source ~/drims_ws/install/setup.bash`. Note that appending to `.bashrc` does **not** affect the shell you typed it in. |
| `motion failed with MoveIt error -31` | No IK solution. Usually the die is too far out for the grasp orientation. | The node now tries other leans, wrist rolls and alternative turns, and reports the flange distance. If it still fails, move the die nearer: `y = 0.58`. |
| `motion failed with MoveIt error 99999` on the *place* | The Cartesian planner could not complete the descent — the gripper's own body reaches the board first. | Handled: the node computes the lowest height the gripper can reach and releases there. Seeing this means a lean was refused *and* the release height was still rejected. |
| `InvalidParameterTypeException ... 'gripper_close_axis' to 'True'` | YAML 1.1 reads a bare `y` as a boolean. | Quote it: `gripper_close_axis: "y"`. |
| `cannot look up dice_tf: "base_link" does not exist` | TF buffer empty. | Check terminal 1 is running; the node now fails loudly here instead of silently degrading. |
| `colcon build` → `PermissionError: ... 'log'` | `drims_ws` is bind-mounted and owned by your host user; the container runs as uid 1001. | On the **host**: `sudo chgrp -R 42042 drims_ws bags && sudo chmod -R g+rwX drims_ws bags` |
| RViz shows the cell but no die | The spawner died — look at terminal 2, not RViz. | Respawn. |

---

## 6. Regenerating the documentation figures

Every image in the README is built from the code, so it cannot drift:

```bash
python3 scripts/make_docs_images.py
```

Writes `docs/images/{pipeline,topics,readout,colours,faces,colour_sweep,top_face}.png`.
