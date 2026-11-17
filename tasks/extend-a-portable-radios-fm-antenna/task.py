"""Get a portable radio ready for FM: antenna all the way out and standing straight up. Created 11_16_26
(America/New_York).

On the station table: a Sony ICF-P26 portable AM/FM radio (69.5 x 119 x 38 mm, 190 g with batteries) standing upright on
its flat base, its front (speaker and dial) facing the robot, the middle of its base at (0.30, 0.00). Its telescopic FM
antenna (five sections, 11.1 cm collapsed and 43.2 cm extended from its swivel) is collapsed and tilted 60 degrees back
from vertical on the swivel at the top right of the radio's back. Every inner section slides out of the one below it
against its own friction, up to a stop; the swivel holds any angle by friction. The radio weighs 1.9 N; pulling a
section out takes up to 2.4 N and swinging the antenna takes about 0.06 N m at the swivel, so a radio that is not held
lifts or tips over instead (radio_antenna/mech.py).

Run it:  inspect-robots run --embodiment yam_radio_antenna -E task=task.py -E object_jitter=0 --policy yam_scripted \\
           --no-prompt --instruction "Get this radio ready for FM: pull the antenna all the way out and stand it straight up, and leave the radio standing where it is." \\
           --scorer task_score

Everything is judged on the final state, as in the real trial (see score()).
"""

import math
import os

import numpy as np

from radio_antenna import grip as G
from radio_antenna import mech as M
from yam_arena import ik
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, READY
from yam_arena.demo import Waypoint
from yam_arena.station import BASES
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]
RADIO_XY = np.array((0.30, 0.00))

OBJECTS = [Box("radio", pos=(float(RADIO_XY[0]), float(RADIO_XY[1]), TOP), size=(0.01, 0.01, 0.01))]  # built by radio_antenna.mech

START_THETA = math.radians(60.0)  # the collapsed antenna leans 60 degrees back from vertical
START_Q = {"pivot": START_THETA, "slide2": 0.0, "slide3": 0.0, "slide4": 0.0, "slide5": 0.0}

# -- criteria (all judged on the final state) ----------------------------------------------------------------------
LENGTH_STEPS = tuple(M.RETRACTED + k * M.TRAVEL - 0.003 for k in range(1, M.N_SECTIONS))  # 188, 269, 349, 429 mm
LENGTH_CREDIT = 0.15  # per step of LENGTH_STEPS reached (one section's travel each), 0.60 in all
UPRIGHT_DEG = 15.0  # antenna within this of vertical ...
UPRIGHT_CREDIT = 0.40  # ... earns this
RADIO_TIPPED_DEG = 10.0  # the radio tilted more than this from standing upright: 0
MOVED_OK = 0.030  # the radio's base within this (m) of its start: no cap
MOVED_CAP = 0.50  # moved more than MOVED_OK but not more than MOVED_MAX: the score is capped at this
MOVED_MAX = 0.100  # moved more than this: 0
CLEAR = 0.040  # a hand is clear with its fingertip midpoint this far from the radio's body and from the antenna's axis
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


def _tilt_deg(q):
    """Angle (deg) between a frame's z axis and the world vertical."""
    return math.degrees(math.acos(float(np.clip(_rot(q)[2, 2], -1.0, 1.0))))


def antenna_tilt_deg(objects):
    return _tilt_deg(objects["antenna"].quat_xyzw)


def antenna_length(objects):
    return float(objects["antenna"].low[1])


def radio_tilt_deg(objects):
    return _tilt_deg(objects["radio"].quat_xyzw)


def radio_moved(objects):
    return float(np.hypot(*(np.asarray(objects["radio"].pos, float)[:2] - RADIO_XY)))


def _seg_dist(p, a, b):
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _box_dist(p, objects):
    """Distance from world point p to the radio's case (a box in the radio's frame)."""
    r = objects["radio"]
    local = _rot(r.quat_xyzw).T @ (np.asarray(p, float) - np.asarray(r.pos, float))
    half = np.array((M.BODY["d"] / 2, M.BODY["w"] / 2, M.BODY["h"] / 2))
    d = np.abs(local - np.array((0.0, 0.0, M.BODY["h"] / 2))) - half
    return float(np.linalg.norm(np.maximum(d, 0.0)))


def hand_clear(objects, hand):
    p = np.asarray(objects[hand].pos, float)
    a = objects["antenna"]
    return _box_dist(p, objects) >= CLEAR and _seg_dist(p, np.asarray(a.pos, float), np.asarray(a.high, float)) >= CLEAR


def note_step(objects, dt):
    """After every control step: how long both hands have been clear without a break (the only thing tracked over time;
    every other criterion is read from the final state)."""
    if all(hand_clear(objects, h) for h in ("left_tcp", "right_tcp")):
        _trial["clear"] += dt
    else:
        _trial["clear"] = 0.0


def length_credit(length):
    return LENGTH_CREDIT * sum(1 for s in LENGTH_STEPS if length >= s - 1e-9)


def raw_credit(objects):
    """Antenna credit before the radio and hands criteria: 0.15 per section's travel of length, 0.40 if upright."""
    c = length_credit(antenna_length(objects))
    if antenna_tilt_deg(objects) <= UPRIGHT_DEG + 1e-9:
        c += UPRIGHT_CREDIT
    return round(c, 6)


def score(objects):
    """From the final state. 0 unless both hands have been clear of the radio and antenna for the last 1 s and the radio
    stands within 10 degrees of upright. Then 0.15 for each 80.25 mm section's travel the antenna is out (pivot to tip
    at least 188, 269, 349, 429 mm) plus 0.40 if the antenna is within 15 degrees of vertical. The radio's base more than
    30 mm from its start caps the score at 0.5; more than 100 mm scores 0."""
    if _trial["clear"] < CLEAR_S - 1e-9 or radio_tilt_deg(objects) > RADIO_TIPPED_DEG:
        return 0.0
    moved = radio_moved(objects)
    if moved > MOVED_MAX:
        return 0.0
    c = raw_credit(objects)
    return min(c, MOVED_CAP) if moved > MOVED_OK else c


def success(objects):
    return score(objects) >= 1.0


def debug_line(objects, t):
    s = objects["slides"]
    sl = [*np.asarray(s.pos, float).tolist(), float(s.low[0])]
    r = lambda v: np.round(1000 * np.asarray(v), 0).tolist()  # noqa: E731
    return (f"RADIO t={t:6.2f} len {1000 * antenna_length(objects):6.1f} slides {r(sl)} tilt {antenna_tilt_deg(objects):5.1f} "
            f"radio tilt {radio_tilt_deg(objects):5.1f} moved {1000 * radio_moved(objects):5.1f} clear {_trial['clear']:.2f} "
            f"score {score(objects):.3f} L {r(objects['left_tcp'].pos)} R {r(objects['right_tcp'].pos)} "
            f"tip {r(objects['antenna'].high)} radio {r(objects['radio'].pos)}")


# -- scripted demo --------------------------------------------------------------------------------------------------
# Open loop from the fixed start; it reads no simulator state. The left hand comes in level from the robot's left and
# pinches the radio's body front to back near its left side, holding it down and upright. The right hand comes in level
# from the robot's right and pinches the tip knob, swings the collapsed antenna up on its swivel to vertical (its wrist
# turning with the antenna), then draws the knob straight up along the antenna's axis until every section is at its stop.
# The right hand opens and backs away, then the left. RADIO_DEMO switches the shortcut variants: "nothing": no motion.
# "one_hand": the left hand never holds the radio. "collapsed": swing upright only, no pull. "half": pull two sections'
# worth only. "hold": the right hand never lets go of the knob.
VARIANT = os.environ.get("RADIO_DEMO", "")


def _env(name, default):
    return float(os.environ.get(name, default))


PINCH_OFF_Y = _env("RADIO_PINCH_OFF_Y", -0.003)  # the right TCP sits this far along y from the knob's centre
SWING_END_DEG = _env("RADIO_SWING_END", -1.5)  # the swing goes this far past vertical, for the give in the pinch
OVERPULL = _env("RADIO_OVERPULL", 0.008)  # the pull aims this far past the last stop, then eases back to 3 mm past it
HOLD_GAP = _env("RADIO_HOLD_GAP", 0.034)  # the left fingers close toward this gap on the 38 mm case: a firm, not crushing, pinch
SWING_S = _env("RADIO_SWING_S", 4.0)
PULL_S = _env("RADIO_PULL_S", 8.0)
HOLD_Z = _env("RADIO_HOLD_Z", 0.065)  # the left pinch's height above the table
HOLD_TIP_Y = _env("RADIO_HOLD_TIP_Y", 0.006)  # the left pad tips reach this y (the case spans y -0.035 to 0.035)
HOLD_DIP = math.radians(20.0)  # the left fingertips point toward -y, dipped this much below level
BACK = 0.09
PARK = {"left": (0.12, 0.34, TOP + 0.18), "right": (0.12, -0.34, TOP + 0.18)}
DOWN = G.frame((0.0, 0.0, -1.0), (0.0, -1.0, 0.0))
LEFT_TIPS = np.array((0.0, -math.cos(HOLD_DIP), -math.sin(HOLD_DIP)))
LEFT_ROT = G.frame(LEFT_TIPS, (-1.0, 0.0, 0.0))  # closing front to back


def world(p_radio):
    return np.array((RADIO_XY[0], RADIO_XY[1], TOP)) + np.asarray(p_radio, float)


def right_pinch(theta, ext_total=0.0, out=0.0):
    """(TCP, rotation) of the right hand pinching the knob with the antenna at swivel angle theta, `ext_total` (m) of
    slide in all, backed off `out` toward -y. Fingertips toward +y, closing square to the antenna in its swing plane."""
    k = M.PIVOT + (M.RETRACTED + ext_total - M.KNOB["len"] / 2) * M.axis_dir(theta)
    rot = G.frame((0.0, 1.0, 0.0), (math.cos(theta), 0.0, -math.sin(theta)))
    return world(k) + np.array((0.0, PINCH_OFF_Y - out, 0.0)), rot


class _Plan:
    def __init__(self):
        self.wps = {"left": [], "right": []}
        self.t = {"left": 0.0, "right": 0.0}
        self.last = {}
        for a in self.wps:
            pos, rot = ik.tcp_pose(np.array(READY, float), np.array(BASES[a]))
            self.last[a] = (np.asarray(pos, float), FINGER_OPEN, np.asarray(rot, float))

    def go(self, arm, dur, pos=None, finger=None, rot=None, label=""):
        p0, f0, r0 = self.last[arm]
        p = p0 if pos is None else np.asarray(pos, float)
        f = f0 if finger is None else finger
        r = r0 if rot is None else np.asarray(rot, float)
        self.wps[arm].append(Waypoint(dur, tuple(float(v) for v in p), f, label, G.turned(r)))
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
    hold = VARIANT != "one_hand"
    hold_in = world((0.0, HOLD_TIP_Y, HOLD_Z)) - 0.013 * LEFT_TIPS
    if hold:
        plan.go("left", 2.5, hold_in - BACK * LEFT_TIPS + (0.0, 0.0, 0.03), FINGER_OPEN, LEFT_ROT, "to the radio's left side")
        plan.go("left", 1.0, hold_in - BACK * LEFT_TIPS, label="level with the radio")
        plan.go("left", 1.5, hold_in, label="pads either side of the case")
        plan.go("left", 1.0, finger=-HOLD_GAP / 2, label="pinch the radio")
        plan.go("left", 0.5, label="settle")
    p_out, r0 = right_pinch(START_THETA, out=BACK)
    p_in, _ = right_pinch(START_THETA)
    plan.go("right", 2.5, p_out + (0.0, 0.0, 0.03), FINGER_OPEN, r0, "to the knob")
    plan.go("right", 1.0, p_out, label="level with the knob")
    plan.go("right", 1.5, p_in, label="pads either side of the knob")
    plan.go("right", 1.0, finger=FINGER_CLOSED, label="pinch the knob")
    plan.until("right", max(plan.t.values()) + 0.5)
    th1 = math.radians(SWING_END_DEG)
    n = 20
    for i in range(1, n + 1):
        th = START_THETA + (th1 - START_THETA) * ik.smoothstep(i / n)
        p, r = right_pinch(th)
        plan.go("right", SWING_S / n, p, rot=r, label="swing the antenna up")
    plan.go("right", 0.5, label="steady")
    full = (M.N_SECTIONS - 1) * M.TRAVEL
    ext = {"collapsed": 0.0, "half": 2 * M.TRAVEL - 0.010}.get(VARIANT, full + OVERPULL)
    if ext > 0.0:
        p, r = right_pinch(th1, ext)
        plan.go("right", PULL_S * ext / full, p, rot=r, label="draw the sections out")
        plan.go("right", 0.5, label="steady")
        if ext > full:
            ext = full + 0.003
            plan.go("right", 0.5, right_pinch(th1, ext)[0], label="ease off the stop")
            plan.go("right", 0.3, label="steady")
    if VARIANT != "hold":
        plan.go("right", 1.2, finger=FINGER_OPEN, label="open")
        plan.go("right", 1.2, right_pinch(th1, ext, out=BACK)[0], label="back off the knob")
        plan.go("right", 2.0, right_pinch(th1, ext, out=BACK)[0] + (-0.08, 0.0, -0.10), label="away")
    if hold:
        plan.until("left", plan.t["right"])
        plan.go("left", 0.8, finger=FINGER_OPEN, label="open")
        plan.go("left", 1.2, hold_in - BACK * LEFT_TIPS, label="back off the radio")
        plan.go("left", 2.0, PARK["left"], rot=DOWN, label="away")
    end = max(plan.t.values()) + 4.0
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
    print("length steps mm", [round(1000 * s, 1) for s in LENGTH_STEPS], "travel mm", round(1000 * M.TRAVEL, 2),
          "overlap mm", round(1000 * M.OVERLAP, 2), "antenna g", round(1000 * M.ANTENNA_MASS, 1))
    print("knob start", np.round(world(M.knob_centre(START_THETA, [0] * 4)), 3), "tip full up",
          np.round(world(M.tip_point(0.0, [M.TRAVEL] * 4)), 3))
    wps = demo()
    t = max((sum(w.duration for w in ws) for ws in wps.values()), default=0.0)
    print(f"demo {t:.1f} s ({t * 20:.0f} steps), IK worst mm/deg, bad:", ik_errors(wps))
