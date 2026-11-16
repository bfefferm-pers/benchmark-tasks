# Set an impulse sprinkler's two trip collars so it waters only the quarter between two flags

<!-- Created 11_16_26 (America/New_York). -->

A plastic impulse sprinkler on a spike waters 220 degrees. The robot has to make it water only the 90-degree bed between
two flags: slide each of the two friction trip collars round the bearing under the head, with a fingertip on the collar's
small tab, to the place where the head's trip lever will meet it. The head's drive arm, counterweight and nozzle overhang
the collars, so a gripper has to come in low, under them, and the free-turning head has to be clear of the hand (turned
aside, or pushed aside on the way in). Then the robot takes its hands away and the head is run: the score is the sweep the
head actually makes, its two reversal headings against the flags.

Instruction given to the policy:

`Set the sprinkler so it waters only the bed between the two orange flags, then take your hands away.`

## Why it matters, and the real mechanism

- Everyday relevance (category: Societal relatability): hose-end impulse sprinklers are a common sight on US lawns and
  sold at every home-improvement store for under $35, and setting one so it waters the bed and not the path, the
  driveway or the house is a familiar household chore; groundskeepers do it too (O*NET 37-3011.00 lists "Water lawns,
  trees, or plants, using portable sprinkler systems, hoses, or watering cans" as a supplemental task), but the claim
  here is recognisability, not a worker count.
- Practice: Rain Bird's impact-sprinkler guide: arc adjusters (trip collars) are "Used to adjust degree of arc sprinkler
  will rotate. Position one arc adjuster for each side of the desired arc ... plastic models move by just a push of the
  thumb"; the trip pin must be down "for part circle operation (then slide arc adjusters into position to set degree of
  arc)". Rain Bird's Maxi-Paw sheet: "slowly move the two pattern adjusters ... The sprinkler will water the angle you set
  with the pattern adjusters (on the opposite side of the trip lever)." The P5R's catalog entry gives 4.38 x 4.5 in
  (111 x 114 mm) and a 20 to 340 degree part-circle range.
- The mechanism modelled is the real one: the collars ride on the stationary bearing and do not turn with the head; the
  head's trip lever hangs beside their tabs, so a running head turns until the lever meets a tab and then reverses. The
  arc is on the side opposite the collars.

## How it differs from the nearest tasks

The nearest is #2005 (another team's sprinkler arc task, closed): its collars rode on and turned with the head and their
pins tripped nothing, and the reviewer judged that construct invalid. Here the collars are stationary stops on the bearing,
the head's lever strikes them, and the score is the sweep the head then makes. #2112 (pending) clicks a hose wand's dial
between detents; #1825 and #2052 (closed / revising) turn and lock a guard or a saw shoe to a mark. None sets an angle by
sliding friction stops under an overhanging, free-turning part, and none scores the motion those stops then produce.
BEHAVIOR-1K's lawn activities (`water_your_lawn_efficiently` and others) use a watering can, not a sprinkler.

## Files

| Path | What |
|---|---|
| `task.py` | the task file: the water valve and the head's reversal bookkeeping (`note_step`, `drive_command`), the scoring, the scripted open-loop demo and its shortcut variants (`SPRINKLER_DEMO`) |
| `sprinkler_arc/mech.py` | geometry, masses and friction, and the USD builder of the sprinkler (one fixed-base articulation) |
| `sprinkler_arc/embodiment.py` | `yam_sprinkler`: the stock `yam_arena` embodiment plus the sprinkler, read and driven through the PhysX tensor view |
| `test_score.py` | scorer unit tests (30), pure Python |
| `dev/run.sh`, `dev/batch.sh`, `dev/media.sh` | one simulator trial; the demo twice, every variant and (`--cal`) the stall test; the media run (H.264 videos and the setup screenshot) |
| `dev/clearance.py` | offline: steps the demo and reports each gripper's closest approach to every sprinkler collider |
| `dev/sheet.py` | contact sheet of stored camera frames |

## Requirements and install

inspect-robots 1.0.0, yam-arena `bd62771e6032` (Isaac Lab-Arena 0.3.1, Isaac Sim 6.1, Isaac Lab 3.0), one L40S.

```bash
source ~/IsaacLab-Arena/yam.env
uv pip install -e . --no-deps --python "$(which python)"   # registers the embodiment yam_sprinkler
```

## Run and score one trial

```bash
inspect-robots run --embodiment yam_sprinkler -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --policy yam_scripted -P hold_s=3 --no-prompt \
  --instruction "Set the sprinkler so it waters only the bed between the two orange flags, then take your hands away." \
  --scorer task_score --max-steps 2400 --seed 0 --store-frames --log-dir ~/yam-eval/sprinkler
```

`task_score` prints at the end. For another policy, replace `--policy yam_scripted -P hold_s=3`. Shortcut variants:
`SPRINKLER_DEMO=nothing|lower_only|short|head_only|no_head` before the command. Offline:
`python -m pytest -q test_score.py` (yam_arena importable) and `python task.py` (the demo's length and IK).

Runtime per trial on an L40S: about 120 s wall clock including Isaac Sim's start; the demo's episode is about 986 steps
(49.3 s) and ends on success. Seeds: the layout is fixed (`-E object_jitter=0`, `--seed 0`); nothing in the task
is random.

## Scoring

The sweep's ends are the nozzle headings where the head reverses. Each end earns 1 within 10 degrees of its flag
(-45 degrees for the clockwise end, +45 for the counter-clockwise end), 0.5 within 20, 0.25 within 30, else 0; the score is
the mean of the two ends, once each end has been reached (the latest reading each way counts).

| Outcome | Score | Real trial reading |
|---|---|---|
| Both ends within 10 degrees of their flags | 1.0 | protractor disc, overhead photo of each stop |
| One end within 10, the other within 20 / 30 / beyond | 0.75 / 0.625 / 0.5 | same |
| Both ends within 20 (arc too wide or narrow by up to 40) | 0.5 | same |
| Both within 30 | 0.25 | same |
| Ends where they started (-110, +110), or the right width aimed elsewhere | 0 | same |
| Never parked clear and still for 2 s after reaching in | 0 | stopwatch on the video |

Doing nothing scores 0 (the readings need the robot to have reached in, and the ends stay 65 degrees off). A gripper
returning within the clear zone shuts the water off and the readings start again, so no hand can hold the head or act as a
stop while the sweep is read. In the simulation the episode ends with success once the score is 1 and has held for 1 s.

Simulation: the "water" is a scripted velocity drive on the head's joint, off until the grippers have been parked
(17 cm or more from the axis, or 6.5 cm above the head's top, each moving under 1 mm a step) for 2 s after one of them
has been within that zone. It turns the head counter-clockwise first at 20 deg/s with at most 0.015 N m, and reverses
when the head has turned slower than 6 deg/s for 0.1 s (0.5 s after a start or a reversal): its lever is against a tab.
Each reversal's heading is recorded. The collars slip at 0.04 N m, so the head cannot push them.

## Real-world setup and safety

- A Rain Bird P5R plastic impulse sprinkler (or one with the dimensions above) on a steel hose-end spike, its trip pin down
  (part circle), in a turf mat on the table, the spike in a taped outline; a printed 360-degree disc (1-degree marks, 0 at
  the robot's forward direction) under the head; two marker flags 24 cm out at -45 and +45 degrees.
- No water runs during a robot trial and no hose is connected, so nothing sprays near the robots. The supervisor reads the
  stops by turning the head by hand (a fingertip touch at the lever, not pushing through) after the robot has parked.
  Optionally the setting is confirmed outdoors on a hose at low pressure, robots away.
- Forces are small: a collar slips at about 1.3 N at its tab's end, the head turns at about 0.1 N at its nozzle; no
  button or trigger is pressed.

## Equipment and cost for real trials

Prices seen in US retailer listings on 11_16_26; they change often.

| Item | Product | Price |
|---|---|---|
| Impulse sprinkler | Rain Bird P5R Mini Bird plastic impact sprinkler, 1/2 in inlet ([Rain Bird store](https://store.rainbird.com/p5r-professional-grade-plastic-impact-sprinkler.html); [Home Depot](https://homedepot.com/p/Rain-Bird-25-41-ft-Professional-Grade-Riser-Mounted-Polymer-Impact-Sprinkler-P5R/202078355)) | $7.97 to $9.49 |
| Spike | 1/2 in metal hose-end sprinkler spike base, e.g. [Orbit 58034N](https://www.walmart.com/ip/3926340550); or the same sprinkler sold on a spike, Rain Bird P5RLSP ([Walmart](https://www.walmart.com/ip/377376993)) | about $10 to $33 |
| Marker flags | wire stake flags, orange, 100-pack, e.g. [Anley 4 x 5 in](https://www.walmart.com/ip/25483812) (two used, posts cut to 11 cm) | about $16 |
| Turf mat | synthetic turf offcut, 30 x 38 cm or larger, 10 mm pile | about $10 |
| Spring scale | 500 g / 5 N push-pull scale, e.g. [Eisco at Wards](https://www.wardsci.com/us/en/product/NA5282255/spring-scale-500g-x-10g--5n-x-010n) | about $10 |
| Heading disc | printed 360-degree protractor disc, 1-degree marks, on paper or card | about $0 |
| **Total** | | **about $55 to $80** |

Nothing is consumed between trials.

## Real measurement vs simulated stand-in

| Quantity | Real trial | Simulation |
|---|---|---|
| Sweep ends | head turned by hand to each stop, heading read on the disc (1 degree), photo | scripted drive, heading at each stall-reversal from the joint angle |
| Collar grip | spring scale at the tab's end: 1.3 N +- 0.3 N | Coulomb joint friction 0.04 N m |
| Head bearing | spring scale at the nozzle tip: 0.2 N or less | Coulomb joint friction 0.005 N m |
| Hands clear and still | video: both grippers 17 cm from the axis or 6.5 cm over the head, still 2 s | fingertip midpoints from the joint angles |
| Water | not run in the trial (optional outdoor confirmation) | not simulated |

## Measured results (scripted open-loop demo and shortcut variants)

All on an L40S with the files in this folder, `dev/batch.sh` (logs in `~/yam-eval/sprinkler-<label>/`). The variants run
with `-P hold_s=70`, so the scripted policy does not stop the trial before the run limit.

| Run | What it does | Sweep ends (deg) | task_score |
|---|---|---|---|
| demo, run 1 | as described below | -44.9, +44.7 | 1.0 (success at 49.3 s) |
| demo, run 2 | the same | -44.9, +44.7 | 1.0 (success at 49.3 s) |
| demo at 640 x 352 (media) | the same | | 1.0 |
| `nothing` | no motion | no water (never reached in) | 0.0 |
| `lower_only` | only the lower collar set | -45.6, +111.0 | 0.5 |
| `short` | each collar stops 15 deg short | -61.8, +62.3 | 0.5 |
| `head_only` | head turned to point between the flags, collars left | -110.7, +110.6 | 0.0 |
| `no_head` | collars pushed, the head not turned between them | -44.8, +44.6 | 1.0 |

`no_head` scores 1.0: the right hand, coming in for the upper collar, pushed the free head aside on its way. Turning the
head deliberately is the demo's choice, not a requirement; the score depends only on where the collars end up.

The demo reads no simulator state: it plays fixed waypoints from the starting layout. The left hand comes in low under
the head's top and pushes the lower collar's tab counter-clockwise from 47 to 126 degrees with the side of a finger, aimed
8 degrees past the goal for the arm's give, then pushes the head round by its counterweight until the nozzle points at
about 23 degrees (square to the upper collar's whole travel) and parks; the right hand then pushes the upper collar's tab
clockwise from 313 to 234 degrees and parks. Both fingertips press just onto the turf while pushing a tab, so they do not
ride up it.

## Stall test of the collar grip

The collars' grip and the head's drive are joint friction and a torque-limited drive in the simulator, so they were measured
in it: `SPRINKLER_CAL=1 SPRINKLER_DEMO=nothing` runs the water at once and raises the head's stall torque by 0.01 N m at
each reversal, with the collars as they start. Measured (`~/yam-eval/sprinkler-cal/`): at 0.01 and 0.02 N m
the head stopped against the tabs and reversed (ends 110.1 and -110.2 degrees, collars unmoved); at 0.03 and 0.04 N m it
still reversed, at 110.8 and -111.6 (each collar crept under 1.5 degrees); at 0.05 N m it carried the lower collar round
with it. So a collar slips between 0.04 and 0.05 N m, as set (0.04). The demo's drive, 0.015 N m, moved the collars 0.2
degrees or less a stroke.

An earlier version used free joints with drive force limits, and those did not act as torque limits (a 0.04 limit gave the
head only 1.7e-4 N m); that is why the sprinkler is an articulation with joint friction.

## Fidelity and open weaknesses

- Water is not simulated; the drive and its stall-triggered reversal stand in for the drive arm's impulses and the trip
  lever's toggle. The trial reads the stops directly instead.
- The head's overall size is from the P5R catalog entry; the widths of its parts, the collar and tab sizes, the masses and
  the slip torques are estimates (no dimensioned drawing or force figure is published). The lever is opposite the nozzle
  as Rain Bird's Maxi-Paw sheet describes; the P5R's own lever position was not confirmed.
- The collar slip torque, 0.04 N m (1.3 N at the tab's end), is a light thumb push. At 0.10 N m (3.7 N) the YAM's finger
  slid off the tab in the simulation instead of moving it.

## Licence

MIT. Copyright (c) 2026 the task authors. All code here is our own; the scene uses no third-party assets.
