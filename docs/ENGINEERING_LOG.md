# Engineering log

The README says what the system does. This says what it took, because most of
the interesting work was in the gap between "the tests pass" and "the robot
moved."

The perception and planning modules were written test-first and were correct
before they ever met the simulator — 355 offline tests, exhaustive over all 24
die orientations and all 8 die colours. That bought a lot, and it bought nothing
at all against the second half of this list. Every failure below survived a green
test suite, and most were only visible with the whole cell running.

Each entry is: what it looked like, what it actually was, and how it was proved
rather than guessed.

---

## Part 1 — found offline, by tests

These were caught before any robot was involved. They are here because each one
is a case where the obvious implementation is wrong in a way that looks right.

### A green die on a green board

**Symptom.** A dark green die was invisible to the detector.

**Cause.** Segmenting the die by colour. Any fixed hue window that finds the
board also finds a green die sitting on it.

**Fix.** Stop looking for the die. Segment the **board**, then treat everything
on the board that is not board-coloured as foreground. The board's colour is not
hard-coded either: a wide green window locates it once, then its own median
chromaticity becomes the classifier. A dark green die differs from the board in
chromaticity even while sharing a hue bin, so it separates cleanly.

This is why the detector works on all eight palette colours with identical
parameters, and it is the single decision the whole perception design rests on.

### The silhouette is not the top face

**Symptom.** Face values misread away from the image centre; the die's reported
position biased toward the middle of the board.

**Cause.** Away from the principal point a cube shows its side faces, so its
blob is a hexagon, not the top square. Using it shifts the centroid and feeds
shaded side-face pixels into the pip-reading patch.

**Fix.** `top_face_from_hull` recovers the top square from the hexagonal
silhouette using the Minkowski relationship between the two.

### Counting pips does not read a die

**Symptom.** Face 6 misread in 13 of 28 test scenes.

**Cause.** At 25 px across, adjacent pips touch and blob-counting merges them.

**Fix.** Stop counting. Fit the 3×3 pip grid and score the six canonical layouts
against it. Reading a die is a template-matching problem wearing a
blob-detection costume.

**And then a second-order version of the same mistake.** Faces 3 and 5 read as
1, and 4 as 2 — because the grid scale was being re-estimated per hypothesis, so
a one-pip hypothesis could shrink the grid until it fitted. Estimating the scale
**once**, from maximum variance, fixed it. Final accuracy: 98.5 % over 200
randomised scenes.

### A blue die that read as white — but only on the six

**Symptom.** Colour naming correct on faces 1–5, wrong on 6, for exactly the
dice whose pips contrast strongly with the body.

**Cause.** The body colour was taken as the larger of two colour clusters. On a
six, the pips cover more of the sampled area than the body between them.

**Fix.** Choose the cluster with the largest **connected component** instead of
the most pixels. One pip is never bigger than the face around it. 384/384 correct
afterwards.

### A measurement bug that flattered the wrong method

**Symptom.** PnP appeared four times worse than the homography for position.

**Cause.** Mine. I scored PnP against an assumed nadir camera while the test
scenes deliberately tilt and offset it.

**Fix.** Score against the renderer's true extrinsics. PnP went from "4× worse"
to correctly characterised: still not used for position (its range comes from
apparent size, biased ~9 % on a 25 px die) but excellent for orientation and as a
reprojection-residual quality gate. **The negative result stayed in the README.**

### Two policies, and one that was needlessly slow

The camera-only re-orientation policy originally walked the 4-cycle: turn, look,
turn again, switch axis when exhausted. Mean 2.0 re-grasps, worst case 4.

Replaced by **deduction**. A die is chiral — 1, 2 and 3 run anticlockwise about
their shared vertex — so the 24 orientations produce 24 distinct
`(top, one lateral)` pairs. One 90° probe turn therefore pins down the entire
configuration, and the exact planner takes it from there. Mean 5/3 ≈ 1.67, worst
case 3, proved exhaustively over 24 orientations × 6 targets × 2 axes.

The insight is that the four side faces are indistinguishable *until one is
turned up*, so any first probe leaves them at 1, 2, 2 and 3 turns — 5/3 is
optimal, not merely better.

---

## Part 2 — found only by running the robot

This is the half a test suite cannot reach. Every one of these looked fine in
simulation-free tests and failed on contact with the cell.

### YAML 1.1 turns `y` into `True`

**Symptom.** `InvalidParameterTypeException: Trying to set parameter
'gripper_close_axis' to 'True' of type 'BOOL', expecting type 'STRING'`.

**Cause.** In YAML 1.1 a bare `y` is a boolean. `gripper_close_axis: y` becomes
`True` before ROS ever sees it.

**Fix.** Quote it — and, more usefully, declare the parameter with
`dynamic_typing=True` and normalise in code, so the node explains the problem
instead of dying with a type error that never mentions YAML.

### A TF listener that never listened

**Symptom.** `cannot look up dice_tf: "base_link" does not exist`, then a silent
fall back to the camera-only policy.

**Cause.** A TF listener only fills its buffer while its node is being spun.
Nothing was spinning the task node — `MotionClient` and `VisionClient` spin
themselves during calls, so the omission was invisible until a TF lookup was
needed.

**Fix.** Give the node its own executor on a background thread, and add
`wait_for_tf` so the failure is loud instead of degrading quietly. **Silent
degradation is worse than a crash**: it produced a working-looking run using the
wrong strategy.

### `dice_tf` is on the die's top face

**Symptom.** None, directly — which is what makes it interesting. The grasp
"worked" because the simulator attaches the die to the gripper rather than
simulating contact.

**Cause.** Found by arithmetic on the spawner's own log, not by observation. It
reports `surface_height -0.020` and `size 0.03`, so the die's centre is at
−0.005 in `base_link` — but the frame arrives at **+0.011**, half a die higher.

**Fix.** Subtract half the die size. Every height downstream — grasp, lift,
place, board clearance — had been out by 15 mm. The same log also showed the
simulator spawns a **30 mm** die while the default config says 27 mm.

The lesson: in a simulator that fakes the physics, a geometric error produces no
symptom at all. It would have produced one immediately on the real cell.

### The place move that stopped at 75 %

**Symptom.**
```
Computed Cartesian path with 10 points (followed 75.000000% of requested trajectory)
Planning failed! Cartesian planner completed 0.75, less than the threshold 0.99.
motion failed with MoveIt error 99999
```
Always on the *place*, always after a clean pick, lift and rotate.

**Cause.** A 90° flip rotates the **tool** by the same 90° as the die, because
the fingers must stay on the two faces the rotation leaves in place. So a grasp
taken straight down finishes pointing **horizontally** — and a horizontal
gripper cannot be lowered to the board, because its own body arrives first. 75 %
of the descent is exactly where the gripper reaches the board.

**Fix, and its limits.** Lean the grasp −45° so it finishes at +45°, centring the
excursion on vertical so neither end is horizontal. Leaning is geometrically
free: the fingers close along the grasp axis whatever the roll about that axis.

Where a lean cannot be used, the die is **released** instead of placed, from the
lowest height the gripper's own envelope allows. Inelegant, and exactly what a
parallel-jaw gripper leaves you after a 90° flip. It is self-correcting: the die
lands on the face the turn chose, and the loop re-reads it regardless.

### Reach is spent on orientation, not just position

**Symptom.** The lean fixed the place and broke the pick: `-31`, no IK solution,
on the free-space move before the arm had gone anywhere.

**Cause.** I had changed the pre-grasp to stand off 10 cm *along the tool axis* —
the textbook approach, which slides the fingers on along their own length. With
a lean that moves sideways as well as up, and sideways is the expensive
direction.

| pre-grasp, same grasp pose | flange distance | of a UR5e's 850 mm |
| --- | --- | --- |
| 10 cm along the tool axis | 0.847 m | **100 %** |
| 10 cm straight up | 0.789 m | 93 % |

**Fix.** Stand off vertically. Nothing is lost: the open fingers straddle the die
along the grasp axis, so a vertical descent never touches it however far the
gripper leans.

The general point is worth more than the fix. An arm's reach is quoted **to its
flange**; a 150 mm gripper is 150 mm the arm does not have. So at a *fixed* grasp
point, changing the tool's orientation changes whether the pose is reachable —
a 45° lean swings the flange through more than 10 cm.

### Checking IK against the wrong oracle

**Symptom.** A pre-flight IK check that passed every candidate and rejected
nothing, followed immediately by `-31` from the real move.

**Cause.** The check asked `easy_motion`'s `get_ik`. `move_to_pose` applies
constraints `get_ik` does not, so the two disagree about what is reachable.

**Fix.** Try the real move. Each lean is attempted as the actual approach motion,
in order, ending with straight down. Slower, and correct — **the only oracle
worth consulting is the one that will run the motion.**

### The one that was never the code

**Symptom.** `-31` at `y = 0.65`, no matter what the lean was.

**Cause.** Workspace, not geometry. At `y = 0.65` the UR5e is extended far
enough that its wrist is near a singularity and whole families of orientations
have no IK solution at all.

**Proof.** Same code, same target, same lean, die moved to `y = 0.58`:

```
grasping at (-0.100, 0.580, 0.011), yaw 28.6 deg, lean -45 deg, rotate +90deg about X
face 3 is up (target 3, bottom 4)
SUCCESS: face 3 up (target 3) after 1 re-grasp(s) in 149.9 s
```

A single controlled variable, changed once. That is the whole diagnosis, and it
took far longer to reach than it should have, because I kept proposing fixes to
the grasp geometry when the evidence — IK timing out rather than answering, the
flange check passing — already said the pose was fine and the *place* was not.

**The transferable lesson.** `-31` says "no IK solution" and nothing about why.
The node now measures the flange distance for every pose it is about to command
and names the one that is out of range, so the next person gets a sentence
instead of a number.

---

### Reachability as a first-class input

The last fix is the only one that changed the *architecture* rather than a
number, so it is worth stating separately.

Every earlier attempt treated the planner's output as fixed and tried to make the
arm accept it. The planner chooses a turn, the executor works out a grasp, and if
the arm refuses, the cycle fails. But the planner's choice is frequently
arbitrary — four different routes reach the bottom face, and the deductive
policy's probe turn can be *any* of the four primitives, since with the sides
indistinguishable all are equally informative.

Throwing that arbitrariness away was the mistake. `equivalent_first_moves` and
`DeductivePolicy.options` now surface it, the executor tries the alternatives
when a grasp is refused, and `DeductivePolicy.substitute` tells the policy which
route was actually taken so its deduced model of the die stays true.

The ordering of the search encodes a real trade-off: lean, then alternative turn,
then wrist roll. A lean of 32° or more lets the die be *placed* while straight
down forces it to be *dropped*, so a good lean is worth more than the preferred
turn.

And where strict equivalence offers nothing — a side-face target has exactly one
shortest route — the search may take a route one turn longer. Paying an extra
re-grasp beats not moving. Tests pin both the equivalence and the bound: always
taking the least-preferred option still reaches the target from every start and
costs at most one extra re-grasp.

## What the failures have in common

Three of the six field bugs were **diagnostic** failures rather than logic
failures — the system knew something was wrong and could not say what:

* a TF listener that degraded silently to a worse strategy instead of failing;
* a YAML boolean surfacing as an rclpy type error that never mentions YAML;
* `-31` and `99999`, which name an error class and no cause.

So the fixes are not only geometric. `wait_for_tf` fails loudly, the parameter
handler explains YAML 1.1, the reach check turns `-31` into a distance in metres,
and the re-orientation policy logs its reasoning:

```
reasoning: top was 5, X-turn showed 1: configuration is top=1, bottom=6, +X=4, -X=3, +Y=2, -Y=5
```

The other consistent theme: **a simulator that fakes the physics hides geometric
errors.** The 15 mm grasp offset produced no symptom whatsoever, because the die
is attached to the gripper by the planning scene rather than held by friction.
It was found by arithmetic on a log, and it would have been found in one second
by a real gripper closing on air.

---

## Not done

Honest list, so the README's claims stay bounded.

* **No hardware run.** Everything here is simulation plus offline tests against a
  synthetic camera. The provided rosbags are the next validation step, and the
  real cell after that.
* **No LabVIEW sensor/actuator integration** for the gripper — that part of the
  school's brief needs the physical cell.
* **The workspace limitation is mitigated, not eliminated.** Choosing among
  equally good re-grasps by reachability *is* now written — see "Reachability as
  a first-class input" below — but nothing moves the die to a friendlier part of
  the board when no grasp at its current position works.
* **One die per board.** The detector returns the single most die-like blob. The
  course's multi-dice photo would need the scorer turned into a ranked list; the
  pipeline supports it, the service signature does not.
