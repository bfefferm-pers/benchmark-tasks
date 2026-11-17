# Pull a pocket radio's telescopic antenna out to full length and stand it upright

Created 11_16_26 (America/New_York).

> Get this radio ready for FM: pull the antenna all the way out and stand it straight up, and leave the radio standing where it is.

A Sony ICF-P26 pocket radio stands on the table facing the robot, its five-section telescopic FM antenna collapsed and
tilted 60 degrees back on its swivel. The robot has to draw the antenna out to its full 43.2 cm and stand it within 15
degrees of vertical, then take both hands away with the radio still standing where it was.

What makes it a physical skill rather than a choice: the radio weighs 1.9 N, less than the 2.4 N the stiffest section
needs to slide and far less than its swivel needs to turn at the radio's top (0.10 N m against the 0.035 N m that keeps
a 38 mm deep radio from tipping). Pulled or swung on its own, the antenna lifts the radio or tips it over. As people do
it, one hand holds the radio upright and down on the table while the other pinches the 6 mm tip knob, swings the antenna
up and draws it out along its own axis, through four friction joints that give one after another, to the last stop.

## How it differs from the nearest tasks

- **#2107 (merged)**, extend a suitcase's handle: one rigid twin-rail handle with a release button and lock pins, on a
  2 kg case lying flat that one hand holds down. Here there is no button or lock: a thin knob is pinched and drawn out
  through four nested friction sections in series (each gives at its own force, up to its own stop) on a 190 g radio
  that stands on a 38 mm deep base and tips or lifts if not held, and the antenna is then on a friction swivel whose
  angle is scored.
- **#1437 (closed, final)**, a shock-corded antenna mast: assembled from loose sections and closed for miniaturisation.
  Here nothing is assembled and the radio and antenna are the real product at its published size. **#1434 (closed)**,
  re-pinning an antenna panel's downtilt, was also a miniature and a different skill (pin and bracket).
- **#933 and #187 (closed)**, a generic training handle with a detent: no real product. **#2264 (revising)** closes a
  telescoping-lid box: a lid, not a sliding mast.
- BEHAVIOR-1K's radio activity (turning_on_radio) is a toggle; RoboCasa, LIBERO, SimplerEnv, DataCenterBench, RoboDojo and
  the Embodied First Exam lists (dexhand, kinder, embodied-safety-exam) have no antenna or telescoping-rod task. This team
  has no telescoping, radio or antenna task.

## Run and score

From this folder, with the simulator's Python environment active and this package installed
(`uv pip install -e . --no-deps --python "$(which python)"`):

    inspect-robots run --embodiment yam_radio_antenna -E task=task.py -E object_jitter=0 -E control_hz=20 \
      -E cam_width=320 -E cam_height=176 --policy yam_scripted --no-prompt \
      --instruction "Get this radio ready for FM: pull the antenna all the way out and stand it straight up, and leave the radio standing where it is." \
      --scorer task_score --max-steps 1200 --seed 0 --store-frames --log-dir radio-demo-logs

`dev/run.sh <label>` wraps the same command. The run prints `task_score`. The layout has no randomness (`object_jitter=0`,
`--seed 0`); the demo is open loop from the fixed start. A trial takes about 3 minutes on an L40S cloud GPU including
Isaac Sim's start (about 29 s of simulated time for the demo, which ends on success).

Shortcut variants of the demo: `RADIO_DEMO=nothing|one_hand|collapsed|half|hold`. Scorer unit tests:
`python -m pytest test_score.py`. `RADIO_DEBUG=1` prints the antenna's length, each slide, the antenna's and the radio's
tilt, how far the radio moved and the score every step.

## Files

| File | What it is |
|---|---|
| `task.json` | the task card |
| `task.py` | layout, start pose, the scorer and the scripted demo |
| `radio_antenna/mech.py` | the radio and antenna: dimensions with sources, geometry, USD builder |
| `radio_antenna/embodiment.py` | `yam_radio_antenna`: the stock embodiment plus the radio articulation, read through PhysX tensor views |
| `radio_antenna/grip.py` | lets a demo waypoint carry a full hand orientation (the hands work level, up to 54 cm above the table) |
| `test_score.py` | scorer unit tests on synthetic states |
| `dev/` | `run.sh` (one trial), `sheet.py` (contact sheet of camera frames), `media.sh` (video and screenshot) |

## Where the numbers come from

| Quantity | Value | Source |
|---|---|---|
| Body | 69.5 x 119 x 38 mm (W x H x D), antenna folded | Sony ICF-P26 specifications: https://www.sony.com.au/electronics/radios/icf-p26/specifications (also sony.com.hk, sony.co.nz; sony.jp: "about 69.5 x 119 x 38 mm, including protrusions") |
| Mass | 190 g with batteries | Sony specification pages; BuyDig listing "6.7 oz (190 g)" (sony.jp gives about 180 g) |
| Antenna length | 11.1 cm fully retracted, 43.2 cm fully extended | BuyDig specification tab: https://www.buydig.com/shop/product/SNICFP26/Sony-ICF-P26-Portable-Analog-Display-AMFM-Radio-with-Speaker ("Fully extended: 17.0 in (43.2 cm); Fully retracted: 4.5 in (11.1 cm)"). Not on Sony's pages. |
| Swivel position | top right of the back (seen from the front), 6 mm behind the back face, 6 mm below the top, 25.5 mm right of centre | position from Sony's product photo (https://www.sony.jp/products/picture/ICF-P26.jpg); offsets estimated |
| Sections | 5 tubes of 100 mm, each inner one sliding 80.25 mm (19.75 mm overlap at full extension) | not published; the count and lengths are the simplest fit to the two published lengths with the overlap typical of such antennas. A purchased unit's count is noted at the first trial |
| Tube diameters | 6.0, 5.0, 4.0, 3.2, 2.5 mm | not published; typical pocket-radio antenna |
| Tip knob | 6 mm across, 7 mm long, cylindrical | Sony's photo shows a plain cylindrical cap wider than the thinnest tube; size estimated |
| Antenna mass | 15.5 g in all | estimate (thin steel tubes) |
| Slide friction | 2.4, 2.1, 1.8, 1.5 N (outer joint to tip joint) | estimates; the reset notes give the spring-scale test |
| Swivel friction | 0.10 N m | estimate; holds the extended antenna (0.03 N m of weight when level) at any angle |
| Swivel range | 60 degrees forward to 180 degrees back (folded down the back) | estimate |

## The scripted demo

Open loop from the fixed start; it reads no simulator state. The left hand comes in from the robot's left, level and
dipped 20 degrees, and pinches the radio's case front to back near its left side (fingers closing toward a 34 mm gap on
the 38 mm case: a firm pinch, about 10 N). The right hand comes in level from the robot's right and pinches the tip knob,
then swings the collapsed antenna up on its swivel to 1.5 degrees past vertical, its wrist turning with the antenna. It
then draws the knob straight up along the antenna's axis for 8 s; the sections give one after another (the tip section
first, then the outer joint, then the middle two together) until each is at its stop, aiming 8 mm past the last stop and
then easing back to 3 mm past it so the hand lets go without a flick. The right hand opens slowly and backs away along
its fingertips, then the left. About 29 s of simulated time; the trial ends on success 1 s after both hands are clear.

## Measured in the simulator

`object_jitter=0`, `--seed 0`, 20 Hz, 320 x 176 cameras.

| Run (label) | Host | `RADIO_DEMO` | task_score | Final reading |
|---|---|---|---|---|
| g4 | cloud L40S | demo | 1.0 | 431.8 mm, antenna 0.6 degrees from vertical, radio moved 1.0 mm, upright; success at 28.9 s |
| g5 | cloud L40S | demo | 1.0 | identical |
| local1 | local aarch64 | demo | 1.0 | 431.9 mm, 0.6 degrees, radio moved 0.9 mm |
| v-nothing | cloud L40S | `nothing`: hands stay at the ready pose 60 s | 0.0 | 111.0 mm, 60.0 degrees |
| v-one_hand | cloud L40S | `one_hand`: the radio is never held | 0.0 | the swing tipped the radio onto its back (90 degrees) and dragged it 44 mm; the sections barely moved (118.6 mm) |
| v-collapsed | cloud L40S | `collapsed`: swung upright, never pulled out | 0.4 | upright, length below the first step |
| v-half | cloud L40S | `half`: pulled partway | 0.55 | upright, three of the four length steps (designed 0.7; the hand stopped short of the third step) |
| v-hold | cloud L40S | `hold`: the demo, but the hand stays on the knob | 0.0 | hands never clear |
| dream | local aarch64 | Dream-1, 1,200 steps | 0.0 | the arms barely moved and never took the knob; the antenna stayed at its start length |

Earlier tuning runs (g1 to g3) found and fixed the released antenna's creep (see Simulation model) and a flick on
release (the ease-off step).

## Simulation model

- One articulation whose free root is the radio's body: a 69.5 x 119 x 38 mm box of 175 g (the radio less the antenna)
  standing on the table on its base (Coulomb friction 0.5), with its dial, grille, tuning wheel and the antenna's
  bracket drawn. The antenna hangs from it: the outer tube on a revolute swivel about the radio's left-right axis, and
  each of the four inner tubes on a prismatic joint along the axis of the tube below it, from 0 to its hard stop at
  80.25 mm. Every tube and the knob collide with the hands; the radio's own parts do not collide with each other.
- Friction: PhysX Coulomb joint friction at every joint (static as above, dynamic 80% of it). With the radio pinched
  and the knob pulled along the axis, the sections give one after another, as on a real antenna.
- The tubes keep their mass and inertia but carry no weight. Their weight (at most 0.03 N along a section, 0.03 N m at
  the swivel with the antenna level) is far below the friction that holds them, but PhysX joint friction lets a joint
  creep slowly under a steady load: with weight on, the released antenna sagged about 1 degree and 2 mm a second in the
  first trial, which a real antenna does not.
- Not modelled: the swivel's second (sideways) axis, and the clip that holds the folded antenna down the back (the task
  starts with the antenna out of the clip). The tubes are rigid; a real antenna flexes a little when pulled off axis.

## Scoring, and how a real trial measures it

All criteria are judged on the final state.

| Criterion | Simulation | Real trial |
|---|---|---|
| Hands clear | each fingertip midpoint at least 4 cm from the case and from the antenna's axis, for the last 1 s | the robot stopped with its hands visibly away; the supervisor waits 1 s |
| Antenna length | swivel pin to the top of the tip knob, from the four slide positions: 0.15 for each of 18.8, 26.9, 34.9 and 42.9 cm reached | tape measure from the swivel pin to the top of the knob |
| Antenna upright | the outer tube's axis within 15 degrees of vertical: 0.40 | phone inclinometer held against the outer section |
| Radio standing | the case's vertical axis within 10 degrees of vertical, else 0 | phone inclinometer against the case's side; side photo |
| Radio in place | the middle of its base within 3 cm of its start; 3 to 10 cm caps the score at 0.5; more than 10 cm scores 0 | masking-tape outline round the base; ruler from the outline |

Every outcome is covered: credit is the sum of the length steps reached and the upright bonus, gated and capped by the
radio and hands criteria. Doing nothing scores 0 (collapsed, 60 degrees from vertical). Holding the antenna up at the
end scores 0, as does a radio left lying down, tipped, or hanging from the antenna. Swinging the collapsed antenna
upright alone earns 0.40; each section's worth of length then adds 0.15. A wobble or a slide of the radio during the
trial costs nothing if the final state is right (one unit test per such case).

## Real-world setup and safety

The radio stands, switched off, on a table at least 60 x 40 cm, inside a masking-tape outline of its base. The antenna is
set collapsed and 60 degrees back from vertical with a printed angle template. Nothing is powered, hot, sharp or heavy:
the largest force the demo needs is under 3 N at the knob, and the radio's two AA cells stay inside. The extended
antenna's tip stands 55 cm above the table; the supervisor keeps their face at least 1 m from it while the robot moves.

## Items for a real trial

| Item | Product | Price | Where |
|---|---|---|---|
| Radio | Sony ICF-P26 portable AM/FM radio (two AA cells supplied) | $21.99 | https://www.bestbuy.com/site/sony-portable-am-fm-radio-black/5784509.p (price seen 11/2/26 per https://shopsavvy.com/products/sony-icfp26-radio, where 13 retailers average $18.95) |
| Section and swivel friction | Arbor Scientific 1000 g (10 N) push-pull spring scale | $10.95 | https://arborsci.com/products/1000g-push-pull-spring-scale |
| Antenna length | Stanley 16 ft PowerLock tape measure | $11.97 | https://www.homedepot.com/p/Stanley-16-ft-PowerLock-Tape-Measure-33-116L/100654911 |
| Radio's position | Westcott 10415 12 in stainless steel ruler | $8.49 | https://www.quill.com/westcott-stainless-steel-rulers-12/cbs/001457.html |
| Outline | ScotchBlue 2090 painter's tape, 0.94 in x 60 yd | $10.27 | https://www.walmart.com/ip/ScotchBlue-Original-Multi-Surface-Painters-Tape-94-inch-x-60-yard-2090-1-Roll/471970548 |
| Angles | any smartphone's built-in level / inclinometer (iPhone Measure app, Android level apps) | free | already owned |

Total about $64, all reusable; nothing is consumed per trial.

## Reset

Collapse the antenna fully, set it 60 degrees back with the angle template, and stand the radio in its outline. Before a
session, with the radio held, pull the tip knob along the antenna's axis with the spring scale: the sections should start
to slide between 1.5 and 2.4 N. With the antenna collapsed, a sideways pull at the knob should swing it at about 0.9 N
(0.10 N m at the 11 cm knob radius). If a real unit differs, set `SLIDE_FRICTION_N` and `PIVOT_FRICTION_NM` in
`radio_antenna/mech.py` to match.
