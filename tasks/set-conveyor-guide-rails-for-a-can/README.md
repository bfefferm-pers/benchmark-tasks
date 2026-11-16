# Change a conveyor lane over to a 66 mm can

Created 11_16_26 11:25 EST.

A 56 cm section of tabletop-chain conveyor lies across the table, its lane set for a 76 mm container (78 mm wide). The far
guide rail is the fixed datum. The near rail is adjustable: a 1/2 in rod at each end slides in a cross block on a bracket
post, and a 40 mm clamp knob on each block locks its rod. Two 68.0 mm setting plugs (the 66 mm can plus 2 mm of running
clearance) stand in the lane against the datum rail, one at each bracket. A sample can stands at the infeed end.

The robot slackens both knobs, brings the rail in against both plugs and holds it there with one hand while the other
locks a knob, locks the second knob, takes the plugs out to their rests and pushes the can through the set lane.

Install once, from this folder: `pip install -e . --no-deps`

Run a trial:

    inspect-robots run --embodiment yam_guide_rail -E task=task.py -E object_jitter=0 -E control_hz=20 \
      -E cam_width=320 -E cam_height=176 --policy yam_scripted --no-prompt \
      --instruction "Set this lane for the 66 mm can and run the sample can through it, leaving the setting plugs on their rests." \
      --scorer task_score --max-steps 2400 --seed 0

The score is printed as `task_score`. Use `--policy dream` for Dream-1.

- **Runtime per trial:** about 3.5 minutes on one cloud GPU for the scripted demo (1,406 steps, 70 s of simulated
  time; it ends on success), plus about 25 s to start the simulator. Two trials took 423 s in all.
- **Seeds:** none. The start is fixed and `object_jitter=0`; `--seed 0` was used.
- **Scoring tests, no simulator:** `python -m pytest -q test_score.py` (20 tests, NumPy only: every band and each
  shortcut below). `python task.py` (in the Python environment that has `yam_arena`) confirms that every waypoint of the
  demo and of each shortcut variant is reachable inside the joint limits and fits the run limit. `python dev/clearance.py`
  plays the plan with perfect tracking and reports how close each arm comes to the parts and to the other arm.
- **The scripted demo** is open loop. It uses only the fixed starting layout (where the brackets, knobs, rail, plugs,
  rests and can stand) and reads nothing from the simulator. Both hands slacken their knobs by a 95 degree turn; the
  right hand cages the rail between its fingers 55 mm right of its middle (between the two plugs) and is aimed 16 mm past the plugs, so the
  arm's compliance holds it against both; the left hand locks its knob by a 110 degree turn; the right hand lets go and
  locks its own knob; the left hand slides its plug out along the lane past the rails' left end and stands it on a rest;
  the right hand pushes its plug along the lane to mid-lane with closed fingers and the left takes that one out the same
  way (the can stands at the infeed end, so both plugs leave by the outfeed end); finally the right hand pushes the can
  in from the infeed end with its closed fingers above the rails, and the left takes it on and pushes it past the left
  bracket.
- **Shortcuts tried in the simulator** (`RAIL_DEMO=<name>`, see Results): `none`; `shove` (the rail pushed at the plugs
  with the knobs still locked); `loose` (set and let go, knobs never locked again); `one` (only the left knob locked);
  `plugs` (plugs left in the lane); `can` (plugs taken out first, the can pushed in mid-lane and the rail set against the
  can itself); `norun` (everything but running the can through). `RAIL_DEBUG=1` prints each step's measurements;
  `python dev/sim.py demo none ...` runs several in one simulator session.
- **Third-party assets:** none. Every part is built from boxes, cylinders and prisms in `guide_rails/mech.py`.

## How this differs from earlier tasks

The nearest submission, a guide-rail changeover with loose bars on an inert bed (#342), was closed because nothing was
adjustable or securable. Here the rail is carried on real sliding rods, each locked by a knob that must be turned at its
real torque, and the setting is proved by running a can through it. The pending proximity-sensor task (#2229) sets one
small sensor's gap with a feeler leaf; this task sets a 34 cm rail at two stations at once, against product-size plugs,
which needs one hand to hold the rail while the other locks. Paper-guide (#1782, #1781) and conveyor jam (#1189) tasks
neither adjust nor lock a rail. No conveyor guide adjustment appears in LIBERO, RoboCasa, SimplerEnv, BEHAVIOR-1K,
DataCenterBench, RoboDojo, kinder, dexhand or embodied-safety-exam.

## Results

Simulator runs of 2026-11-16 on a cloud GPU, scripted policy, `--seed 0`, `object_jitter=0`. A further demo on an aarch64 host also scored 1.0. One Dream-1 run (`--policy dream`, 2,400 steps, with the instruction above) scored 0.0: the left arm reached to a setting plug and the right arm toward the can, but no knob was turned; an earlier run with a longer wording of the instruction also scored 0.0.

| Run | What the plan does | Score |
|---|---|---|
| demo, two trials in a row (the command above with `--epochs 2`) | the full demo | 1.0 and 1.0 |
| `none` | nothing | 0.0 |
| `loose` | rail set against the plugs and let go, knobs left slack | 0.25 |
| `one` | only the left knob locked again | 0.5 |
| `plugs` | rail set and both knobs locked, plugs left in the lane | 0.75 |
| `norun` | everything but running the can through | 0.75 |
| `can` | plugs taken out first, the can pushed in mid-lane and the rail set against it: the rail pivots on the one can and locks at 0.45 mm at the left bracket and -3.1 mm at the right | 0.0 |

The shortcuts were run with `python dev/sim.py none loose one plugs norun can` (the same embodiment, policy and scorer in
one simulator session). In two further demo episodes run the same way (`python dev/sim.py demo demo`, both 1.0) the
lane was set to 68.00 mm at both brackets (2.00 mm of clearance), the knobs ended 74 degrees past the seat (0.63 N m to
turn; 115 N of hold), the plugs stood about 5 mm from their rests' centres and the can ended upright 75 mm past the left
bracket. In development runs the wrist-roll torque while turning a knob peaked at 0.93 N m (the joint allows 7 N m). Earlier development runs showed two things the demo is built around: a rail pushed open
handed seats on one plug only (the rail turns), so the hand cages it; and with the rail pressed in this firmly the locked rail
preloads the plugs, so lifting a plug straight up fails and both plugs are slid out along the lane instead.

## The clamp and its numbers

- **Knob and screw:** a 40 mm six-lobe knob on an M8 clamp screw closing a split acetal block on a 1/2 in rod. Hand
  tight is about 0.5 N m; the knob turns an eighth to a quarter turn from snug to hand tight (K = 0.5 N m per 60 degrees).
- **Holding force:** an M8 screw with a nut factor of 0.2 makes 625 N of clamp load per N m; two lines of contact at a
  friction coefficient of 0.15 give 187 N of axial hold per N m. As found (0.5 N m) each rod holds 94 N. A rod holds the
  10 N test once its knob is about 6 degrees past the seat.
- **Loose:** below the seat the rod slides against 0.6 N of drag (half the 0.63 kg rail assembly's weight at 0.2).
- **In the simulation** each knob is a hinge whose friction limit is the turning torque above, so the gripper must
  supply it and a knob let go stays where it is. Each rod is a slide whose friction limit is the holding force; once a
  rod holds 10 N or more, a lock joint fixes it where it stands (a real rod slips only at its holding force, far above
  anything the arms apply here), and it is freed again when the knob is slackened. The thread and the block's elastic
  closing are not meshed.
- **Loads:** the heaviest part moved is the 0.31 kg plug; knob torques stay at or under 1.3 N m (the wrist allows 7 N m).

## Real-world setup

- Conveyor: a 56 cm section of 82.6 mm (3.25 in, 820-series K325) straight-running acetal tabletop chain on a steel
  frame, motor disconnected. Guide rails: two 34 cm stainless flat bars 17.5 x 25.4 mm with UHMW wear strips, centred 40 mm
  above the chain. The datum rail is bolted to its brackets. The adjustable rail carries two 1/2 in rods in split acetal
  cross blocks on 1/2 in posts, each closed by a 40 mm six-lobe knob on an M8 screw.
- Setting plugs: two acetal cylinders turned to 68.0 +/- 0.05 mm, 60 mm tall. Rests: two 100 mm green discs taped to the
  table at the positions in the setup.
- Can: a full, sealed 355 ml aluminium drinks can.
- Before each trial: lane set to 78 mm with a gauge block, both knobs locked to 0.5 N m with a torque screwdriver and knob
  adapter, plugs and can placed as in the setup photo (positions marked on the table with tape).
- Safety: the conveyor is not powered; nothing is heavier than 0.4 kg; knob torques are 1.3 N m or less; no button,
  trigger or pinch point beyond the lane itself. An operator with the emergency stop watches the trial.

## Scoring: real measurement and simulated stand-in

| Criterion | Real measurement (how) | Simulated stand-in |
|---|---|---|
| Lane clearance at a bracket | Calipers (or a ruler reading 0.5 mm) across the lane at each rod, datum face to adjustable face, minus 66.0 mm; in band if 0.5 to 3.0 mm | The adjustable rail's face at each bracket from its pose, against the fixed datum face, minus 66.0 mm |
| A clamp holds | A 10 N push along the rod with a spring scale on the rail at that bracket: no slip (a pencil mark on the rod against the block) | The knob's tightness past the seat gives a holding force of 10 N or more (clamp law above) |
| Rail let go and still | Last frame: both grippers 6 cm or more from the rail; the rail does not move in the last second of video | Fingertip midpoints 6 cm from the rail's surface; rail under 1 cm/s and 0.5 rad/s |
| Plugs on rests | Each plug upright, its centre within 15 mm of a rest's centre (ruler), one per rest | The same from the poses; tilt under 5 degrees, down on the rest |
| Can run through | The can upright on the chain with its centre 30 mm or more past the left bracket (ruler), and in the video it passes the right bracket only after both knobs were locked | The same from the pose, y of the can tracked every step against the clamps' state |
| Hands clear, everything still | Both grippers 6 cm from rail, knobs and plugs; nothing moves in the last second of video | The same from the poses and velocities |
| Off the table (score 0) | Any part on the floor | Any loose part 1 cm below the table top |
