"""Set a folding steamer basket in a saucepan, ready to steam. Created 11_17_26 (America/New_York).

On the station table: an empty, cold Vigor SS1 4.5 qt stainless sauce pan (8 in across the top, 7 1/16 in across the
bottom, 5 1/4 in tall), standing on its bottom with its handle pointing away from the robot, the middle of its bottom at
(0.28, -0.01). To its left, at (0.27, 0.25), an OXO stainless steel folding steamer basket stands on its three wire feet
(base plate 38 mm above the table), its 16 hinged leaves lying open (26.8 cm across, wider than the pan's 20.3 cm mouth)
and its telescoping handle pulled up, the T-bar's top 14 cm above the table. The leaves fold up on their hinges when
pushed; the handle slides down its post against 3 N of friction (steamer_basket/mech.py).

Run it:  inspect-robots run --embodiment yam_steamer_basket -E task=task.py -E object_jitter=0 --policy yam_scripted \\
           --no-prompt --instruction "<INSTRUCTION below>" --scorer task_score

Everything is judged on the final state, as in the real trial (see score()).
"""

import math
import os

import numpy as np

from steamer_basket import grip as G
from steamer_basket import mech as M
from yam_arena import ik
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, READY
from yam_arena.demo import Waypoint
from yam_arena.station import BASES
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]
INSTRUCTION = ("Put the steamer basket in the saucepan ready to steam: standing on its feet with every leaf inside the pan, "
               "and its handle pushed down so the lid will fit.")
PAN_XY = np.array((0.28, -0.01))
STEAMER_XY = np.array((0.27, 0.25))
STEAMER_YAW = math.radians(135.0)  # the T-bar runs along this heading

OBJECTS = [Box("steamer", pos=(float(PAN_XY[0]), float(PAN_XY[1]), TOP), size=(0.01, 0.01, 0.01))]  # built by mech

START_Q = {"handle": M.HANDLE_TRAVEL, **{k: 0.0 for k in range(M.N_LEAVES)}}

# -- criteria (all judged on the final state) ----------------------------------------------------------------------
CENTRE_TOL = 0.015  # the base plate's centre within this of the pan's axis
FOOT_TOL = 0.006  # the plate's underside within this of LEG_H above the pan's floor: standing on its feet on the bottom
TILT_OK = 5.0  # the plate within this of level (relative to the pan)
TILT_PART = 15.0  # 5 to 15 degrees: half credit
RIM_MARGIN = 0.002  # every leaf tip at least this far below the rim's plane and inside the wall
HANDLE_BELOW = 0.005  # the T-bar's top at least this far below the rim's plane: the flat lid sits on the rim
PAN_MOVED_MAX = 0.030
PAN_TILT_MAX = 5.0
CLEAR = 0.040  # a hand is clear with its fingertip midpoint this far from the steamer's post and handle ...
CLEAR_ABOVE = 0.030  # ... and not inside the pan (within the rim's circle and below this much above the rim)
CLEAR_S = 1.0  # both hands clear for at least the last second of the trial


def _fresh():
    return {"clear": 0.0}


_trial = _fresh()


def reset_latch():
    """Forget the trial: the embodiment calls this at every reset."""
    _trial.clear()
    _trial.update(_fresh())


def _rot(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _angle_deg(a, b):
    return math.degrees(math.acos(float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))))


def in_pan_frame(objects, p):
    """World point p in the pan's frame (origin at the middle of its outside bottom, z up its axis)."""
    pan = objects["pan"]
    return _rot(pan.quat_xyzw).T @ (np.asarray(p, float) - np.asarray(pan.pos, float))


def pan_moved(objects):
    return float(np.hypot(*(np.asarray(objects["pan"].pos, float)[:2] - PAN_XY)))


def pan_tilt_deg(objects):
    return _angle_deg(_rot(objects["pan"].quat_xyzw)[:, 2], (0.0, 0.0, 1.0))


def steamer_tilt_deg(objects):
    """The base plate's tilt from the pan's axis."""
    return _angle_deg(_rot(objects["steamer"].quat_xyzw)[:, 2], _rot(objects["pan"].quat_xyzw)[:, 2])


def plate_offset(objects):
    """(horizontal distance from the pan's axis, height above the pan's inside floor) of the plate's underside centre."""
    p = in_pan_frame(objects, objects["steamer"].pos)
    return float(np.hypot(p[0], p[1])), float(p[2] - M.PAN_FLOOR)


def leaves_inside(objects):
    """Every corner of every leaf's tip edge below the rim's plane and inside the pan's wall."""
    for tip in np.asarray(objects["leaves"].high, float).reshape(-1, 3):
        p = in_pan_frame(objects, tip)
        if p[2] > M.PAN_H - RIM_MARGIN or math.hypot(p[0], p[1]) > M.pan_inner_r(p[2]) + 0.004:
            return False
    return True


def handle_top_below_rim(objects):
    """How far (m) the T-bar's top is below the rim's plane (negative: above it)."""
    return float(M.PAN_H - in_pan_frame(objects, objects["handle"].high)[2])


def _seg_dist(p, a, b):
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def hand_clear(objects, hand):
    p = np.asarray(objects[hand].pos, float)
    if _seg_dist(p, np.asarray(objects["steamer"].pos, float), np.asarray(objects["handle"].high, float)) < CLEAR:
        return False
    q = in_pan_frame(objects, p)
    return not (math.hypot(q[0], q[1]) < M.RIM_R and q[2] < M.PAN_H + CLEAR_ABOVE)


def note_step(objects, dt):
    """After every control step: how long both hands have been clear without a break (the only thing tracked over time;
    every other criterion is read from the final state)."""
    if all(hand_clear(objects, h) for h in ("left_tcp", "right_tcp")):
        _trial["clear"] += dt
    else:
        _trial["clear"] = 0.0


def score(objects):
    """From the final state. 0 unless both hands have been clear for the last 1 s, the pan has moved at most 30 mm and
    tilted at most 5 degrees, and the steamer's base plate is in the pan (its centre inside the rim's circle and below
    the rim). Standing on its feet on the bottom (plate 38 +- 6 mm above the floor, centre within 15 mm of the pan's
    axis) with every leaf tip inside and below the rim: 1.0 if the plate is within 5 degrees of level and the T-bar's top
    at least 5 mm below the rim; 0.5 if the handle is higher or the plate tilts 5 to 15 degrees. Otherwise, in the pan
    (a leaf over the rim, perched, tilted more than 15 degrees or on its side): 0.25."""
    if _trial["clear"] < CLEAR_S - 1e-9:
        return 0.0
    if pan_moved(objects) > PAN_MOVED_MAX or pan_tilt_deg(objects) > PAN_TILT_MAX:
        return 0.0
    d, h = plate_offset(objects)
    if d > M.RIM_R or h + M.PAN_FLOOR > M.PAN_H:
        return 0.0
    tilt = steamer_tilt_deg(objects)
    standing = d <= CENTRE_TOL and abs(h - M.LEG_H) <= FOOT_TOL and tilt <= TILT_PART and leaves_inside(objects)
    if not standing:
        return 0.25
    if tilt <= TILT_OK and handle_top_below_rim(objects) >= HANDLE_BELOW:
        return 1.0
    return 0.5


def success(objects):
    return score(objects) >= 1.0


def debug_line(objects, t):
    d, h = plate_offset(objects)
    ang = np.degrees(np.asarray(objects["leaves"].low, float))
    r = lambda v: np.round(1000 * np.asarray(v), 0).tolist()  # noqa: E731
    tips = [in_pan_frame(objects, p) for p in np.asarray(objects["leaves"].high, float).reshape(-1, 3)]
    per = max(1, len(tips) // len(ang))  # corners per leaf
    tipr = " ".join(f"{a:.0f}/{1000 * max(math.hypot(p[0], p[1]) for p in tips[i * per:(i + 1) * per]):.0f}/"
                    f"{1000 * max(p[2] for p in tips[i * per:(i + 1) * per]):.0f}" for i, a in enumerate(ang))
    return (f"STEAMER t={t:6.2f} off {1000 * d:5.1f} h {1000 * h:6.1f} tilt {steamer_tilt_deg(objects):5.1f} "
            f"leaves {ang.min():5.1f}-{ang.max():5.1f} in {int(leaves_inside(objects))} slide "
            f"{1000 * float(objects['handle'].low[0]):5.1f} below {1000 * handle_top_below_rim(objects):6.1f} "
            f"pan {1000 * pan_moved(objects):4.1f} clear {_trial['clear']:.2f} score {score(objects):.2f} "
            f"fingers {r(objects['left_tcp'].high[:2])} L {r(objects['left_tcp'].pos)} R {r(objects['right_tcp'].pos)} plate {r(objects['steamer'].pos)} tips {tipr}")


# -- scripted demo --------------------------------------------------------------------------------------------------
# Open loop from the fixed start; it reads no simulator state. The left hand comes in tilted 40 degrees along the T-bar
# and pinches it over the post, lifts the open steamer by its handle until the feet clear the rim, carries it over
# the pan, centres it and lowers it: the leaves meet the rim, fold up on their hinges and slide down the wall as
# the steamer goes in, until it stands on its feet on the bottom. Still pinching, the hand pushes the T-bar down until the
# handle is collapsed, then opens, backs off along its fingertips and parks. The right hand does not move: nothing needs
# holding (the pan weighs 1.3 kg and the push is 3 N). STEAMER_DEMO switches the shortcut variants: "nothing": no motion.
# "handle_up": no push. "off_centre": lowered 50 mm off the pan's axis. "beside": set down on the table beside the pan.
# "hold": the hand never lets go. "drop": let go over the pan from the carrying height (feet 3 cm above the rim) and
# never push. "drop_push": the same, then the open hand comes back, pinches the T-bar and pushes it down as the demo does.
# "tilted": lowered with the hand turned STEAMER_TILT degrees (default 20) about the pads' closing axis, so
# the steamer comes down tilted. STEAMER_LOWER_S sets how long the lowering takes (8 s; 1 s is a fast drop-in).
VARIANT = os.environ.get("STEAMER_DEMO", "")


def _env(name, default):
    return float(os.environ.get(name, default))


PITCH = math.radians(_env("STEAMER_PITCH", 40.0))
GRIP_OFF = _env("STEAMER_GRIP_OFF", 0.0)  # along the T-bar from the post (0: right over the post, so the steamer hangs plumb)
GRIP_BELOW_TOP = _env("STEAMER_GRIP_BELOW_TOP", 0.002)  # TCP this far below the T-bar's top
LIFT_FEET = _env("STEAMER_LIFT_FEET", 0.165)  # carry with the feet this high above the table (rim at 133 mm)
LOWER_S = _env("STEAMER_LOWER_S", 8.0)
PUSH_OVER = _env("STEAMER_PUSH_OVER", 0.004)  # the push aims this far past the collapsed stop
SET_OVER = _env("STEAMER_SET_OVER", 0.015)  # the lowering aims this far below the feet touching the floor
# The arm sags under the steamer's weight at the raised carry pose: in the first simulator run the TCP sat 10 to 18 mm
# below its target while carrying (none with an empty hand). The open-loop demo aims that much higher while loaded.
SAG_EMPTY = _env("STEAMER_SAG_EMPTY", 0.0)
SAG_LOADED = _env("STEAMER_SAG_LOADED", 0.014)
# Carried by its T-bar, the steamer also hangs a few degrees off plumb (held by the pads' friction) and the loaded hand
# falls short in x and y: in the fourth run the plate hung 11 mm behind and 9 mm left of the hand, and the hand 4 mm short
# in each of x and y. The demo aims the hand this far off the pan's axis so the plate comes down over it.
SAG_XY = np.array((_env("STEAMER_SAG_X", 0.015), _env("STEAMER_SAG_Y", -0.013)))
OFF_CENTRE = _env("STEAMER_OFF_CENTRE", 0.050)  # the off_centre variant lowers this far off the axis
TILT = math.radians(_env("STEAMER_TILT", 20.0))  # the tilted variant turns the hand this far
BAR = np.array((math.cos(STEAMER_YAW), math.sin(STEAMER_YAW), 0.0))
TIPS = math.cos(PITCH) * np.array((0.0, 0.0, -1.0)) - math.sin(PITCH) * BAR
CLOSING = np.array((-BAR[1], BAR[0], 0.0))
HAND = G.frame(TIPS, CLOSING)
BACK = 0.07
PARK = {"left": (0.15, 0.32, TOP + 0.16), "right": (0.15, -0.32, TOP + 0.16)}
DOWN = G.frame((0.0, 0.0, -1.0), (0.0, -1.0, 0.0))  # the stock straight-down grip, for parking


def tcp_for(axis_xy, plate_z, slide, sag=0.0):
    """The left TCP target pinching the T-bar of a steamer standing upright with its axis at axis_xy, its plate's
    underside at absolute height plate_z and its handle at `slide`, aimed `sag` (m) higher for the arm's sag."""
    top = plate_z + M.tbar_top(slide)
    return np.array((axis_xy[0], axis_xy[1], top - GRIP_BELOW_TOP + sag)) + GRIP_OFF * BAR


class _Plan:
    def __init__(self):
        self.wps = {"left": [], "right": []}
        self.t = {"left": 0.0, "right": 0.0}
        self.last = {}
        for a in self.wps:
            pos, r = ik.tcp_pose(np.array(READY, float), np.array(BASES[a]))
            self.last[a] = (np.asarray(pos, float), FINGER_OPEN, np.asarray(r, float))

    def go(self, arm, dur, pos=None, finger=None, rot=None, label=""):
        p0, f0, r0 = self.last[arm]
        p = p0 if pos is None else np.asarray(pos, float)
        f = f0 if finger is None else finger
        r = r0 if rot is None else np.asarray(rot, float)
        self.wps[arm].append(Waypoint(dur, tuple(float(v) for v in p), f, label, G.tilted(r)))
        self.last[arm] = (p, f, r)
        self.t[arm] += dur

    def until(self, arm, t):
        if t > self.t[arm] + 1e-9:
            self.go(arm, t - self.t[arm], label="wait")


def demo():
    global LAST_PLAN
    plan = _Plan()
    if VARIANT == "nothing":  # both hands stay at the ready pose for a full minute
        for a in plan.wps:
            plan.go(a, 60.0, label="wait")
        LAST_PLAN = plan
        return plan.wps
    up = M.HANDLE_TRAVEL
    floor_plate = TOP + M.PAN_FLOOR + M.LEG_H  # the plate standing on the pan's floor
    grasp = tcp_for(STEAMER_XY, TOP + M.LEG_H, up, SAG_EMPTY)
    plan.go("left", 3.0, grasp - BACK * TIPS + (0.0, 0.0, 0.03), FINGER_OPEN, HAND, "above the T-bar")
    plan.go("left", 1.0, grasp - BACK * TIPS, label="in line with the T-bar")
    plan.go("left", 1.5, grasp, label="pads either side of the T-bar")
    plan.go("left", 1.0, finger=FINGER_CLOSED, label="pinch the T-bar")
    plan.go("left", 0.5, label="settle")
    carry_plate = TOP + LIFT_FEET + M.LEG_H
    plan.go("left", 4.0, tcp_for(STEAMER_XY, carry_plate, up, SAG_LOADED), label="lift until the feet clear the rim")
    if VARIANT == "beside":
        target = STEAMER_XY + np.array((0.0, -0.005))
        plan.go("left", 1.5, tcp_for(target, carry_plate, up, SAG_LOADED), label="hold")
        plan.go("left", LOWER_S, tcp_for(target, TOP + M.LEG_H - SET_OVER, up, SAG_LOADED), label="set it back down")
        set_plate = TOP + M.LEG_H - SET_OVER
    else:
        target = PAN_XY + SAG_XY + (np.array((0.0, OFF_CENTRE)) if VARIANT == "off_centre" else 0.0)
        plan.go("left", 4.0, tcp_for(target, carry_plate, up, SAG_LOADED), label="carry over the pan")
        plan.go("left", 2.0, label="steady")
        set_plate = floor_plate - SET_OVER
        if VARIANT in ("drop", "drop_push"):
            here = tcp_for(target, carry_plate, up, SAG_LOADED)
            plan.go("left", 1.0, finger=FINGER_OPEN, label="let go over the pan")
            plan.go("left", 1.2, here - BACK * TIPS, label="back off the T-bar")
            if VARIANT == "drop":
                plan.go("left", 2.0, PARK["left"], rot=DOWN, label="away")
                end = max(plan.t.values()) + 5.0
                for a in plan.wps:
                    plan.until(a, end)
                LAST_PLAN = plan
                return plan.wps
            plan.go("left", 2.0, label="let it settle")
            set_plate = floor_plate  # come back to the T-bar of the landed steamer and push as the demo does
            landed = tcp_for(target, set_plate, up, SAG_EMPTY)
            plan.go("left", 2.0, landed - BACK * TIPS, label="in line with the landed T-bar")
            plan.go("left", 1.5, landed, label="pads either side of the T-bar")
            plan.go("left", 1.0, finger=FINGER_CLOSED, label="pinch the T-bar")
        if VARIANT == "tilted":
            c = CLOSING / np.linalg.norm(CLOSING)
            k = np.array(((0.0, -c[2], c[1]), (c[2], 0.0, -c[0]), (-c[1], c[0], 0.0)))
            turn = np.eye(3) + math.sin(TILT) * k + (1 - math.cos(TILT)) * (k @ k)  # Rodrigues: TILT about CLOSING
            plan.go("left", 2.0, rot=turn @ HAND, label="tilt the hand")
        if VARIANT != "drop_push":
            plan.go("left", LOWER_S, tcp_for(target, set_plate, up, SAG_LOADED), label="lower it in")
    plan.go("left", 0.8, label="settle")
    slide = up
    if VARIANT not in ("handle_up",):
        slide = -PUSH_OVER
        plan.go("left", 3.0, tcp_for(target, set_plate, slide, SAG_LOADED), label="push the handle down")
        plan.go("left", 0.5, label="steady")
    if VARIANT != "hold":
        here = tcp_for(target, set_plate, slide, SAG_LOADED)
        plan.go("left", 1.0, finger=FINGER_OPEN, label="open")
        plan.go("left", 1.2, here - BACK * TIPS, label="back off the T-bar")
        plan.go("left", 1.5, here - BACK * TIPS + (0.0, 0.0, 0.10), label="up")
        plan.go("left", 2.0, PARK["left"], rot=DOWN, label="away")
    end = max(plan.t.values()) + 3.0
    for a in plan.wps:
        plan.until(a, end)
    LAST_PLAN = plan
    return plan.wps


LAST_PLAN = None


def ik_errors(wps):
    from yam_arena.demo import ArmPlan

    bad = []
    worst = (0.0, 0.0)
    for arm, ws in wps.items():
        ap = ArmPlan(list(ws), BASES[arm])
        t = 0.0
        labels = {}
        while t <= ap.duration:
            before = (ap.worst_ik_mm, ap.worst_ik_deg)
            _, label = ap.action7(t)
            if (ap.worst_ik_mm, ap.worst_ik_deg) != before and (ap.worst_ik_mm > 1.0 or ap.worst_ik_deg > 1.0):
                labels[label] = (round(ap.worst_ik_mm, 1), round(ap.worst_ik_deg, 2))
            t += 0.05
        worst = (max(worst[0], ap.worst_ik_mm), max(worst[1], ap.worst_ik_deg))
        bad += [(arm, k, v) for k, v in labels.items()]
    return worst, bad


if __name__ == "__main__":
    print("open mm", round(1000 * M.open_diameter(), 1), "closed mm", round(1000 * M.closed_diameter(), 1),
          "leaf g", round(1000 * M.MASS["leaf"], 2), "rim r", M.RIM_R, "inner r at plate",
          round(M.pan_inner_r(M.PAN_FLOOR + M.LEG_H), 4))
    wps = demo()
    t = max((sum(w.duration for w in ws) for ws in wps.values()), default=0.0)
    print(f"demo {t:.1f} s ({t * 20:.0f} steps), IK worst mm/deg, bad:", ik_errors(wps))
