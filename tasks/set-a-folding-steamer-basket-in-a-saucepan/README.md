# Set a folding steamer basket in a saucepan, ready to steam

Created 11_17_26 (America/New_York). Revised 11_17_26 10:45: leaves modelled as overlapping sheets that collide with each other; new shortcut runs.

> Put the steamer basket in the saucepan ready to steam: standing on its feet with every leaf inside the pan, and its handle pushed down so the lid will fit.

A folding stainless steamer basket (OXO Good Grips, model 1067247) stands open on the counter beside an empty, cold 4.5 qt
sauce pan (Vigor SS1). Open, the basket is 26.8 cm across; the pan's mouth is 20.3 cm. The robot has to carry the basket
by its telescoping handle, lower it into the pan so that its 16 hinged leaves fold up on the rim and slide down the wall
as it goes in, stand it on its feet on the bottom, and push the handle down until its T-bar is below the rim, where the
flat lid will sit.

What makes it a physical skill: the basket's 168 mm base plate has only about 9 mm of clearance to the pan's wall at the
height it stands, and the open leaves reach 33 mm past the rim all round. Brought over the pan within about 3 cm of its
axis, the leaves meet the rim together, fold up and funnel the basket in (in the simulator the demo still scored 1.0
lowered 30 mm off the axis, and letting go 3 cm above the rim over the centre also lands it on its feet); 40 mm or more
off, the plate lands on the rim and the basket stays perched and tipped (0.25). So the deciding skill is carrying a
313 g articulated basket by a small T-bar, 20 cm up, and placing it over the pan's axis within about 3 cm; the speed of
the lowering matters little. The handle then has to be pushed down 38 mm against its lock's friction while the basket
stays on its feet (left up, 0.5).

What it is not: a two-handed task. The pan weighs 1.3 kg and the push is about 3 N, so nothing needs holding; the
scripted demo uses the left hand only and the right hand never moves. A person usually does the same (one hand on the
handle).

## Why it is recognisable

Folding "petal" steamer baskets are the standard stovetop steamer in US kitchens: sold by Target, Walmart, Williams-Sonoma
and every cookware brand, and reviewed as a category by America's Test Kitchen ("You open the leaves so that they form a
fairly flat surface, add your food, and then lower the basket using the handle",
https://www.americastestkitchen.com/equipment_reviews/1885-collapsible-steamer-baskets). Setting one in a pot is what
anyone who has steamed vegetables has done; a 5-second clip of a steel petal basket going into a saucepan reads without
explanation. The task ends at the state a cook leaves before adding water and food: basket standing in the pan, handle
down for the lid.

## How it differs from the nearest tasks

- **#2058 (closed, final)**, bridge a telescoping over-sink colander across a basin: closed as a specialised implement
  whose setup most people do not know firsthand, and because rinsing was only context. A petal steamer is a far
  commoner item, and the scored end state is the cooking set-up itself (in the pan, lid able to go on), not context.
  The mechanism also differs: hinged leaves folded by the pan's own rim and wall, not telescoping arms set on a sink.
- **#1853 (pending)**, lower a hotel pan level into a steam-table well: a rigid pan lowered by its edges into a
  rectangular well. Here the part is articulated (16 free leaves) and wider than the opening, and a telescoping handle
  must then be collapsed.
- **#2319 (pending)**, a reusable coffee filter into a swivel brew basket with a bail folded flat; **#723 (merged)**, open
  an umbrella until its runner latches; **#1804 (merged)**, erect a folding crate. None puts a folding part through a
  narrower round opening.
- RoboCasa's steaming tasks (SteamVeggiesWithWater, PrepareVeggiesForSteaming, MultistepSteaming, SteamFish) move
  vegetables into a pot and a pot onto a burner; no steamer basket. BEHAVIOR-1K's activity list, LIBERO, SimplerEnv,
  DataCenterBench, RoboDojo and the Embodied First Exam lists (dexhand, kinder, embodied-safety-exam) have no steamer
  task. This team has no steamer, pan-insert or folding-basket task.

## Run and score

From this folder, with the simulator's Python environment active and this package installed
(`uv pip install -e . --no-deps --python "$(which python)"`):

    inspect-robots run --embodiment yam_steamer_basket -E task=task.py -E object_jitter=0 -E control_hz=20 \
      -E cam_width=320 -E cam_height=176 --policy yam_scripted --no-prompt \
      --instruction "Put the steamer basket in the saucepan ready to steam: standing on its feet with every leaf inside the pan, and its handle pushed down so the lid will fit." \
      --scorer task_score --max-steps 900 --seed 0 --store-frames --log-dir steamer-demo-logs

`dev/run.sh <label>` wraps the same command. The run prints `task_score`. The layout has no randomness
(`object_jitter=0`, `--seed 0`); the demo is open loop from the fixed start. A trial takes about 3 minutes on an L40S cloud GPU, and about 3 minutes on the local aarch64 machine, including Isaac
Sim's start (about 34 s of simulated time for the demo, which ends on success 1 s after the hand is clear).

Shortcut variants of the demo: `STEAMER_DEMO=nothing|handle_up|off_centre|beside|hold|drop|drop_push|tilted` (`STEAMER_OFF_CENTRE` sets the off-centre distance, default 0.05 m; `STEAMER_TILT` the tilt, default 20 degrees; `STEAMER_LOWER_S` the lowering time, default 8 s). Scorer unit tests:
`python -m pytest test_score.py`. `STEAMER_DEBUG=1` prints the plate's offset, height and tilt, the leaf angles, the
handle's slide and the score every step.

## Files

| File | What it is |
|---|---|
| `task.json` | the task card |
| `task.py` | layout, start pose, the scorer and the scripted demo |
| `steamer_basket/mech.py` | the steamer and the pan: dimensions with sources, geometry, USD builders |
| `steamer_basket/embodiment.py` | `yam_steamer_basket`: the stock embodiment plus the steamer articulation and the pan, read through PhysX tensor views |
| `steamer_basket/grip.py` | lets a demo waypoint carry a tilted hand (a straight-down hand cannot reach 30 cm above the table) |
| `test_score.py` | scorer unit tests on synthetic states |
| `dev/` | `run.sh` (one trial), `sheet.py` (contact sheet of camera frames), `media.sh` (video and screenshot) |

## Where the numbers come from

| Quantity | Value | Source |
|---|---|---|
| Steamer, closed and open | 7 in (177.8 mm) across x 3 in closed; 10.5 in (266.7 mm) across x 5 in open | OXO's answer on the Target listing (https://www.target.com/p/oxo-stainless-steel-steamer-with-extendable-handle/-/A-52379834); the same on Fante's (https://fantes.com/oxo-good-grips-stainless-steel-steamer-with-extendable-handle/, model 1067247V1) |
| Steamer mass | 0.69 lb (313 g) | Target specification tab |
| Leg height | 1.5 in (38.1 mm) | Target Q&A; America's Test Kitchen specification list (some listings say 1 3/4 in) |
| Handle | T-bar 4 in above the basket extended, 2.5 in collapsed (38.1 mm of travel); locks up when pulled, pressed down to lock | America's Test Kitchen review; Fante's ("Press the handle again to lock it down") |
| Pan fit | "pots 8 in diameter or larger" | Williams-Sonoma and Sizzle and Sear listings |
| Pan | 8 in (203.2 mm) top diameter, 7 1/16 in (179.4 mm) bottom diameter, 5 1/4 in (133.4 mm) tall, 7 7/8 in handle, 22 gauge stainless, 4 mm aluminium-clad bottom, flat cover | Vigor sauce pan specification sheet (https://www.webstaurantstore.com/documents/specsheets/sauce_pans.pdf) and listing 473SSAUCP4 |
| Leaves | 16 steel sheets, each 47 mm long, 36 mm wide at the hinge and 54 mm at the tip, twisted 12 degrees about their length so each overlaps the next; hinged on a 168 mm circle; open, they rest 15 degrees above level (267 mm across the tip corners) | not published: the count and the overlap from the retailer's photo; the length, widths and circle the simplest fit to the published closed and open diameters; the twist the least that lets neighbours fold fully closed without meeting (unit test) |
| T-bar | 74 x 14 x 12 mm | scaled from the product photo (the published open diameter as the scale) |
| Feet | three wire loops, 30 mm foot bar at 60 mm radius | from the product photo; estimate |
| Part masses | plate 75 g, post 30 g, feet 35 g, handle 45 g, leaves 8 g each | estimates summing to the published 313 g |
| Pan mass | 1.3 kg | estimate (4 lb shipped with cover and box) |
| Frictions | leaf hinge 0.0004 N m; handle slide 3.0 N; steel on steel 0.3; pan on table 0.4 | estimates; the reset notes give the spring-scale test |

## The scripted demo

Open loop from the fixed start; it reads no simulator state. The left hand comes in tilted 40 degrees along the T-bar
and pinches it over the post, lifts the open basket by its handle until the feet are 15 cm up (the rim is 13.3 cm),
carries it over the pan, waits 2 s, and lowers it over 8 s: the leaves meet the rim, fold up and slide down the wall
until the basket stands on its feet on the bottom. Still pinching, the hand pushes the T-bar down 42 mm (4 mm past the
collapsed stop), opens, backs off along its fingertips and parks. The right hand stays at its ready pose. Two open-loop
corrections measured in earlier simulator runs are built in (the demo reads nothing at run time): the loaded hand aims
14 mm higher, and over the pan it aims 15 mm forward and 13 mm right of the pan's axis, because carried by the T-bar the
basket hangs a few degrees off plumb (held by the pads' friction) and the loaded arm falls short.

## Measured in the simulator

`object_jitter=0`, `--seed 0`, 20 Hz, 320 x 176 cameras unless noted.

| Run (label) | Host | `STEAMER_DEMO` | task_score | Final reading |
|---|---|---|---|---|
| s2-demo | local aarch64 | demo | 1.0 | plate on its feet (38.1 mm above the floor), 3 mm off the axis, level; leaves folded 67 to 78 degrees, all inside; T-bar 26.8 mm below the rim; pan moved 1.4 mm |
| c-demo | cloud L40S | demo | 1.0 | as s2-demo |
| media | cloud L40S | demo, 640 x 352 cameras | 1.0 | the run rendered for the videos |
| c-nothing | cloud L40S | `nothing`: hands at the ready pose 60 s | 0.0 | basket still on the table |
| c-handle_up | cloud L40S | `handle_up`: set in, handle never pushed | 0.5 | standing, T-bar above the rim |
| c-beside | cloud L40S | `beside`: lifted and set back down on the table | 0.0 | not in the pan |
| c-hold | cloud L40S | `hold`: the demo, but the hand stays on the T-bar | 0.0 | hand never clear |
| c-off30 | cloud L40S | `off_centre`, 30 mm | 1.0 | the leaves funnelled it in |
| s2-off40 | local aarch64 | `off_centre`, 40 mm | 0.25 | perched on the wall 64 mm up, tipped 29 degrees, 20 mm off the axis; pan pushed 14 mm |
| c-off50 | cloud L40S | `off_centre`, 50 mm (the variant's default) | 0.25 | perched |
| c-tilted | cloud L40S | `tilted`: lowered with the hand turned 20 degrees | 0.25 | wedged 72 mm up, tipped 55 degrees; pan pushed 8 mm |
| c-fast | cloud L40S | demo, lowered in 1 s instead of 8 s | 1.0 | the leaves still fold and funnel it in |
| s2-drop | local aarch64 | `drop`: let go over the centre, feet 3 cm above the rim, never pushed | 0.5 | landed on its feet, 3 mm off the axis; T-bar 3 mm above the rim (the landing pushed the handle 8 mm) |
| c-drop | cloud L40S | `drop` | 0.5 | landed on its feet, 3.5 mm off the axis, level; T-bar 4 mm above the rim |
| s2-drop_push | local aarch64 | `drop_push`: the drop, then re-pinch the T-bar and push it down | 1.0 | as the demo |

What the variants show: the score turns on where the basket comes down, not how. Over the axis within about 3 cm it
funnels in whether lowered slowly, quickly or let go just above the rim; 4 cm or more off, or tilted 20 degrees, it
stays perched (0.25). Leaving the handle up costs half. Dream-1 scored 0.0 on this revision in one 900-step run
(it did not put the basket in the pan). Earlier tuning runs (g1 to g7, on the previous leaf model) found and fixed an
off-centre pinch that let the basket hang tilted, and left-handed rotations that misplaced the pan's wall colliders;
the demo's open-loop offsets were measured in g4 and still hold.

## Simulation model

- The steamer is one articulation whose free root is the base plate (a 168 mm disc with the fixed post and three wire
  feet). Each of the 16 leaves hangs from the plate's rim on a revolute hinge from its open stop (15 degrees above level)
  to 100 degrees above level, with Coulomb friction small enough that an unsupported leaf falls open. The handle slides
  on the post on a prismatic joint with 3 N of Coulomb friction, from collapsed to extended; the real push lock is
  modelled as that friction (it holds the basket's weight carried either way and gives to a 3 N push).
- The leaves collide with each other, as drawn. Each is a thin trapezoidal sheet (convex-hull collider identical to its
  render), twisted 12 degrees about its length so its leading edge lies over the next leaf's trailing edge all round,
  as the real leaves are shingled. Folded together they never meet, from open to fully closed (unit test
  `test_leaves_fold_closed_without_touching`); when the rim folds one leaf ahead of its neighbour, it presses on and
  slides against it and drags it up, and nothing passes through anything (largest overlap between neighbours in the
  demo runs, from the logged angles: under 1 mm, PhysX contact depth). Each leaf, the plate, post, feet and T-bar
  collide with the hands and the pan.
- The pan is one rigid body: a floor disc, 48 flat wall segments on the published taper (5.1 degrees out from vertical),
  and the welded handle. The cover is not modelled; the handle criterion stands in for it (see below).
- Not modelled: the folding of the feet (they start, and stay, unfolded), perforations, and the steel's flex.

## Scoring, and how a real trial measures it

All criteria are judged on the final state; nothing that happens earlier in the trial is scored except that both hands
must have been clear for the last second.

| Criterion | Simulation | Real trial |
|---|---|---|
| Hands clear | each fingertip midpoint at least 4 cm from the steamer's post and handle and not inside the pan (within the rim's circle and below 3 cm above it), for the last 1 s | robot stopped, hands visibly away; the supervisor waits 1 s |
| Pan in place | its base moved at most 3 cm and tilted at most 5 degrees, else 0 | masking-tape outline round the pan's base; ruler |
| In the pan | the plate's centre inside the rim's circle and below the rim, else 0 | look: the basket's base is inside the pan |
| Standing | the plate's underside 38 +- 6 mm above the inside floor (on its feet on the bottom) and its centre within 15 mm of the pan's axis | ruler from the rim down to the plate (published depth less 38 mm, +- 6 mm); a look that all three feet are on the floor |
| Leaves inside | both corners of every leaf's tip edge below the rim's plane and inside the wall | a ruler laid across the rim touches no leaf; no leaf outside the pan |
| Level | plate within 5 degrees of the pan's axis for full credit; 5 to 15 degrees half | phone inclinometer on the plate beside the post |
| Handle down | T-bar top at least 5 mm below the rim's plane | the pan's flat cover set on: it sits flat on the rim (a ruler across the rim clears the T-bar) |

Bands: 1.0 standing, leaves inside, level, handle down. 0.5 standing with leaves inside but the handle up (or not 5 mm
below the rim) or the plate 5 to 15 degrees off level. 0.25 in the pan but not standing (perched on the wall, a leaf over
the rim, tilted more than 15 degrees, or on its side). 0 out of the pan, pan moved or tipped, or a hand not clear.
Doing nothing scores 0 (the basket starts on the table). Every outcome falls in one band; a transient tilt, a leaf
catching the rim on the way in, or the handle pushed early costs nothing if the final state is right (unit tests for
each such case).

## Real-world setup and safety

The pan stands empty, cold and off any burner on a table at least 60 x 45 cm, handle pointing away from the robot,
inside a masking-tape outline. The basket stands open beside it on its feet with the handle pulled up. Nothing is hot,
sharp, powered or heavy: the heaviest item moved is the 313 g basket, and the largest force the demo needs is the 3 N
push on the handle. No water is used.

## Items for a real trial

| Item | Product | Price | Where |
|---|---|---|---|
| Steamer basket | OXO Good Grips Stainless Steel Steamer with Extendable Handle (1067247) | $26.99 | https://www.target.com/p/oxo-stainless-steel-steamer-with-extendable-handle/-/A-52379834 |
| Pan and flat cover | Vigor SS1 Series 4.5 qt stainless sauce pan with aluminium-clad bottom and cover (473SSAUCP4) | $21.99 | https://www.webstaurantstore.com/vigor-4-1-2-qt-stainless-steel-aluminum-clad-straight-sided-sauce-pan/473SSAUCP4.html |
| Handle friction | Arbor Scientific 1000 g (10 N) push-pull spring scale | $10.95 | https://arborsci.com/products/1000g-push-pull-spring-scale |
| Plate height, pan position | Westcott 10415 12 in stainless steel ruler | $8.49 | https://www.quill.com/westcott-stainless-steel-rulers-12/cbs/001457.html |
| Outline | ScotchBlue 2090 painter's tape, 0.94 in x 60 yd | $10.27 | https://www.walmart.com/ip/ScotchBlue-Original-Multi-Surface-Painters-Tape-94-inch-x-60-yard-2090-1-Roll/471970548 |
| Plate level | any smartphone's built-in level / inclinometer | free | already owned |

Total about $79, all reusable; nothing is consumed per trial.

## Reset

Lift the basket out by its handle, pull the handle up until it locks, open every leaf flat to its resting angle, and
stand it on its feet at its spot beside the pan (a second tape outline, round its three feet). Before a session, with
the basket standing on the table, push the T-bar down with the spring scale: the handle should give at about 3 N (if a
real unit differs, set `HANDLE_FRICTION_N` in `steamer_basket/mech.py` to match), and each leaf, lifted by its tip to
vertical and let go, should fall open.
