# Solder a wire into a terminal lug with an iron and solder wire

An 8 AWG copper soldering lug (a solder-type tube lug) is screwed by its tongue to a post on a bench jig, and a stripped, pre-tinned 8 AWG wire lies loose in its barrel. A hot soldering iron lies on its stand and a stick of solder wire on its rest. The robot has to take the iron in one hand and the solder in the other, heat the lug through the tip until the joint itself melts solder, feed solder into the joint and not onto the iron, take the solder away and then the iron, leave the joint alone while it freezes, and put both tools back. Instruction: "Solder the wire into the lug with the iron and the solder, let the joint set, and put the iron and the solder back."

Install once, from this folder: `pip install -e . --no-deps`

Run a trial:

    inspect-robots run --embodiment yam_solder -E task=task.py -E object_jitter=0 -E control_hz=20 \
      -E cam_width=320 -E cam_height=176 --policy yam_scripted --no-prompt \
      --instruction "Solder the wire into the lug with the iron and the solder, let the joint set, and put the iron and the solder back." \
      --scorer task_score --max-steps 2000 --seed 0

The score is printed as `task_score`. Use `--policy dream` for Dream-1.

- **Runtime per trial:** 3-7 minutes on one GPU (the demo is 919 steps plus a 3 s hold).
- **Seeds:** none. The start is fixed and `object_jitter=0`; `--seed 0` was used for the demo.
- **Scoring checks, no simulator:** `python -m pytest -q test_score.py` (27 checks: every band and each shortcut). `python task.py` checks the demo's length, reach and joint ranges.
- **The scripted demo** is open loop. It uses only the fixed starting layout, which a policy can see, and two aiming offsets measured from a first logged run, which was aimed with none and still made a sound joint (the tip landed 1.5 mm further along the barrel than aimed, the stick 2.5-3.4 mm nearer the insulation; the arms' 2-5 mm of sag is taken up by bearing down on the parts). The lug leaves about 3 mm either way for the tip along its 8.7 mm barrel and for the stick along the 8 mm of bare conductor, and 9 mm either way along the tip. The left arm takes the iron and lays the tip across the barrel, 5.5 mm inside the mouth, bearing down with about 1 N. After 10.5 s the right arm lays the solder stick across the bare conductor at the mouth (about 0.2 N), holds it there 5.25 s, and lifts it away; 1 s later the iron lifts off. Both tools go back on their rests, both arms move away, and the run waits until the joint has set.
- **What the physics does:** jig, lug, wire, iron, stick and hands are rigid bodies with real contact. The joint's heat and the solder are bookkeeping driven by the simulator's contact forces (below). When a joint sets, a fixed joint ties the wire to the lug; before that, and after any cold joint, the wire is loose in the barrel and can be pulled out. The fillet shown at the mouth (lean, full, fat, dull grey if cold, a black ring if overheated, a white ring while liquid) is drawn from the record.
- **Shortcut variants** (`SOLDER_DEMO=...`, for checking the scorer only) and what each scored in the simulator:

  | Variant | What the arms do | Score | Why |
  |---|---|---|---|
  | (none set) | the demo, run twice in a row | 1.0, 1.0 | 44.1 mm fed, peak 279 C, freezing from 8.9 s and set and joined 14.4 s after the tip left |
  | `none` | nothing | 0.0 | joint stays at 25 C |
  | `hover` | the same motions with the tip held 3 mm above the barrel | 0.0 | no contact force, so no heat; the solder lies on the cold wire and does not melt |
  | `tip` | the solder is fed onto the iron's tip | 0.0 | 45 mm melted onto the iron, none in the joint |
  | `ironfirst` | the iron lifts off 2.5 s before the solder does | 0.0 | "solder fed with the iron off the lug": cold joint, wire left loose |
  | `yank` | after the tools are back, a hand takes the wire and pulls it as the joint freezes | 0.0 | "the wire moved while the joint froze": the wire ends 38.5 mm out of the barrel, loose |
  | `short` | 2 s of feed | 0.5 | 18.1 mm fed: joined but starved |
  | `untidy` | a sound joint, then both tools put down on the table | 0.75 | not on their rests |

- `SOLDER_DEBUG=1` prints the record every step. `dev/` holds the run scripts.
- **Third-party assets:** none. Every part is built from boxes, cylinders and cones in this folder's code.

## How this differs from sweating a copper pipe joint

A pending task has a torch heat a copper water fitting until solder is drawn in. This one is an electrical termination made with a contact iron. Heat arrives only by conduction through a tip that must actually bear on the lug, so where and whether the tip touches matters, not how near a flame is. The work is ruined by the order of withdrawal (solder away first, then the iron) and by any movement of a loose wire while the solder freezes, which a clamped pipe joint doesn't have. The result is tested as a termination: a pull on the wire, the fillet at the barrel mouth, the state of the insulation.

## The heat and solder numbers

- **Parts.** The lug is a solder-type lug, made to be soldered: the ILSCO copper soldering lug, catalogue number 1/4 (8 AWG, one hole, #10 bolt, seamless copper tube, UL 486A-486B listed), with the dimensions of the maker's catalogue table: 1/4 in tube (6.35 mm outside), 3/16 in bore (4.76 mm), barrel 11/32 in (8.73 mm), tongue 3/8 in wide (9.5 mm) and 0.062 in thick (1.6 mm), 1-1/32 in overall (26.2 mm), 13/64 in hole. The wire is 8 AWG 19-strand THHN (conductor 0.143 in, 0.217 in over the insulation, from a maker's data sheet). The iron is a 67 W temperature-controlled pencil iron with a 6.5 mm chisel tip, set to 400 C, the usual top of the range for lead-free hand soldering. The solder is 1.5 mm (0.062 in) rosin-core SAC305, which melts at 217-220 C.
- **Heat capacity, 2.2 J/K:** the lug is 330 mm3 of copper (3.0 g, 1.1 J/K); the 30 mm of conductor that heats with it is 2.3 g (0.9 J/K); 0.5 g of solder adds 0.1 J/K; the rest is the screw's and washer's share.
- **Heat in, min(67 W, 0.25 W/K x (400 C - T)):** the station's rated power caps it. 0.25 W/K is the whole path from the heater's sensor to the joint through a tinned tip wetted onto the copper: the 6.5 mm copper tip core alone conducts about 0.7 W/K over its 18 mm, and the heater-to-tip joint and the solder bridge take that down by about two thirds. A dry or oxidised tip conducts several times less.
- **Heat out, 0.07 W/K x (T - 25 C):** the 8.4 mm2 conductor acts as a cooling fin (about 0.02-0.04 W/K while it is still warming up), the lug's 6 cm2 of surface loses about 0.01 W/K to the air, and the screw and phenolic post take about 0.02 W/K.
- **What that gives:** solder flows (227 C) after 8.7 s of contact, and the joint passes 300 C after 19.8 s. Left alone it settles at 318 C, so overheating is reachable by dwelling. As a check on the size of these numbers: taking 2.2 J/K from 25 C to 227 C needs 445 J, which is 7 s of the station's 67 W, so a 60-80 W iron does this joint with room to spare and no larger iron is needed (a 4 AWG lug, four times the copper, would need a 150 W class iron). On a real bench expect 8 to 25 s of dwell before solder flows on this lug: the short end with a freshly tinned, wetted tip, as modelled; the long end with a dry tip. The three constants are estimates from the parts' masses and conductances, not bench measurements; before real trials the dwell is timed on a scrap lug with a thermocouple under the screw and the constants are adjusted to it.
- **Cooling:** after the tip leaves at about 279 C (the demo), the joint is back at 217 C in 9 s, freezes over 2-3 s and is at 200 C about 14 s after the tip left.
- **Solder, 8 mm/s and 1.37 J/mm:** a 1.5 mm wire is 13 mg/mm; warming it to 217 C takes 0.58 J/mm and melting it 0.79 J/mm, so 8 mm/s draws 11 W from the joint, which a 67 W iron keeps up with. The space to fill is the bore less the conductor (17.8 - 10.3 mm2) over 8.2 mm, plus the fillet: about 70 mm3, or 40 mm of 1.5 mm wire. **Sound: 30 to 55 mm.** Under 30 mm leaves the mouth unbridged somewhere; over 55 mm has nowhere to go but out. Under 10 mm doesn't hold the wire.
- **Flow at 227 C,** 10 C over the melting point, so that the lug and not the iron melts the solder. **Overheating at 300 C,** where rosin flux chars quickly and PVC insulation near the joint softens and shrinks back. **Set at 200 C,** solid with a margin.
- **Cold joint.** Solder fed more than 0.5 s after the tip has left goes into a pool that is no longer being heated. From 217 C down to 200 C the solder is freezing: the wire moving more than 0.5 mm or 1 degree, or the stick or the iron touching the joint, leaves it cracked or with the stick frozen in. Reheating the joint to 240 C with the iron reflows it and the fault is gone, as on the bench.

## Real-world setup and safety

These are fixed before a trial and not changed during it.

- **Iron:** a 67 W (60-80 W) temperature-controlled pencil iron with a 6.5 mm chisel tip, set to 400 C, switched on and at temperature on its stand before the trial; the robot operates no switch. The tip is wiped and tinned before each trial. A glass-filled nylon (or PEEK) grip block is clamped on top of the handle, behind the finger guard, so the fingers never come within 60 mm of the heater. The cord is silicone, hung from an overhead spring balancer above the jig so that a spring scale at the grip reads under 0.5 N of drag anywhere on the path.
- **Stand:** a steel two-fork cradle on a steel foot holding the iron level by its handle with the tip in free air over the soldering mat, 10 cm from the jig. Nothing combustible within 15 cm of the tip.
- **Solder:** lead-free only, 1.5 mm rosin-core SAC305, a straight 130 mm stick held in a 25 mm nylon sleeve by a set screw, new and measured for each trial. No lead-bearing solder is used, and gloves are worn when handling used parts.
- **Solder stick follow-in:** the real stick shortens by about 40 mm as it feeds, so the real hand moves in along the stick to keep it on the joint, and it must bear lightly (a 1.5 mm solder wire bends above about 0.3 N at this length). The simulated stick is rigid and doesn't shorten; this is a known difference.
- **Lug and wire:** a new 8 AWG copper soldering lug (ILSCO catalogue number 1/4 or the same size from another maker) held to the phenolic post by a #10 screw and washer (phenolic so the post doesn't draw the heat away); a new wire, stripped 16.2 mm, twisted, fluxed and tinned beforehand, pushed home and left lying in the cradle. Ink marks on the conductor at the mouth and at the insulation's end.
- **Fume extraction:** a bench fume extractor with a carbon filter (or a ducted arm), its inlet 15-20 cm behind and above the joint, drawing away from the cameras, running throughout. Rosin fume is a respiratory sensitiser; nobody leans over the joint during the trial.
- **Heat protection for the grippers:** silicone-coated glass-fibre sleeves over both grippers and wrists; silicone finger pads rated to 230 C. The hand holding the solder is the one nearer the heat (about 12 cm from the joint); the stick's nylon sleeve keeps the fingers off the metal.
- **Bench:** a silicone soldering mat under the jig, the stand and the solder rest, covering everything the tip can reach; a fire blanket and a CO2 extinguisher at the cell.
- **Stops:** the iron's station is on a switched outlet beside the trial supervisor, with the robots' emergency stop. If the iron is dropped or left anywhere but its stand, the supervisor cuts the outlet and the robots, and the iron is picked up by its handle with a heat-resistant glove. Nobody reaches into the cell while the robots are live.
- **After the trial:** the joint is left 2 minutes, then checked under 50 C with the back of a gloved hand or an IR thermometer before the pull test. The iron stays on its stand, switched off at the outlet.

## Scoring: real measurement and simulated stand-in

Tools: a 10 N (1 kg) spring scale, a 150 mm steel ruler, a 10x loupe, a fine marker, a camera, and one standard photo of a sound joint on this lug. The joint is photographed from above and from the mouth end with the ruler in view.

| Criterion | Real measurement (how) | Simulated stand-in |
|---|---|---|
| Wire joined to the lug | Hook the spring scale to a cord loop tied round the insulation 5 cm from the lug and pull straight along the wire, away from the barrel, to 10 N for 5 s. The ink mark on the conductor at the mouth moves under 0.5 mm. | A fixed joint between lug and wire, switched on when a joint sets that is not cold, has 10 mm or more of solder in it, and has the wire within 2.5 mm (along) and 1.5 mm (across) of its seat. Otherwise the wire stays loose. |
| Not a cold or disturbed joint | From the video: in the 30 s after the iron's tip leaves the lug, nothing touches the lug or the wire and the wire doesn't move; the solder stick left the joint before the iron did. On the part: no solder stick frozen in, no crack or ring round the conductor in the fillet under the loupe. Any of these counts as not joined. | Solder fed more than 0.5 s after the tip's contact with the lug ended; or, while the modelled joint cools from 217 C to 200 C (9 to 14 s after the tip leaves in the demo), the wire moving over 0.5 mm or 1 degree from where it lay, or a reported contact of the stick or the iron with the joint. Reheating to 240 C with the iron clears it. |
| Heat really got into the joint | Solder has flowed and wetted the barrel mouth and the conductor (a concave fillet, not a ball sitting on the copper). | Heat is added only in physics steps where the simulator reports a contact force over 0.02 N between the iron and the lug, with the iron in a hand. Holding the tip near the lug adds nothing. |
| Solder went into the joint, not onto the iron | The fillet is at the mouth; a blob left on the iron's tip, or solder on the tongue or the outside of the barrel away from the mouth, is not in the joint. | Solder melts only in steps where the simulator reports the stick touching the lug or the wire, the stick's bare length is within 5 mm of the wire's axis between the insulation's end and 3 mm inside the mouth, the stick is in the other hand, and the joint is at 227 C or more. Touching the iron melts it onto the iron (recorded, not counted); touching hot copper elsewhere is recorded as stray. |
| Amount: not starved | Under the loupe, and against the standard photo, solder bridges conductor and barrel all the way round the mouth, with no gap or dark line. Cross-check: the stick is at least 30 mm shorter. | 30 mm or more fed. |
| Amount: not excess | No solder bulging beyond the barrel's outside diameter at the mouth, no drip or icicle, none on the tongue's face; the strands' outline still shows under the solder on the bare conductor. Cross-check: the stick is at most 55 mm shorter. | 55 mm or less fed. |
| Not overheated | Flux residue amber or brown, not black; the insulation's end within 1 mm of its ink mark, not scorched, and with no melted groove; no blue or purple tint on the tongue. | The modelled joint never above 300 C, and no reported contact of the held iron's tip with the wire where the nearest point of the wire is insulation. |
| Joint set by the end | The joint is solid when the run ends: touched with a wooden stick at the end of the run, the fillet doesn't smear. | The modelled joint is below 200 C at the last step. A joint still liquid or freezing scores 0. |
| Tools back, hands clear and still | Ruler against marks on the rests: each grip centre within 12 mm in plan and 6 mm in height of its rest position, tilted under 10 degrees, turned under 20 degrees; both hands 6 cm from both tools and from the joint; nothing moving in the last 1 s of the video. | The same numbers from the tool and gripper poses; at rest means under 1 cm/s and 0.5 rad/s for the iron, the stick and the wire. |
| Nothing dropped | Iron, solder stick and wire all still on the table. | None of them more than 1 cm below the table top at any step. |

Other outcomes:
- **Iron laid on the lug and let go:** it adds no heat in the simulation (the iron must be in a hand) and is not back on its stand. A real iron left lying on the work would go on heating it; the supervisor stops the trial.
- **Solder stuck to the joint:** the simulated stick never sticks; a stick touching the joint while it freezes is a cold joint. A real stick frozen in is left as it is and is not on its rest.
- **Reworked joint:** judged as it ends. Reflowing a cold joint with the iron is allowed and can repair it; solder added then counts toward the amount, and the heat limit still applies.
