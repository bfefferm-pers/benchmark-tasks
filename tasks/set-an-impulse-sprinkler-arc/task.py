"""Set a plastic impulse sprinkler to water one quarter of a lawn. Created 11_16_26 (America/New_York).

A plastic impulse sprinkler (the size of a Rain Bird P5R: 111 mm long, 114 mm tall) stands on a spike in a turf mat in
front of the arms, its axis at (0.27, 0.00). Two orange marker flags stand 24 cm from it, at headings -45 and +45 degrees
(+x = 0, counter-clockwise seen from above): the quarter between them, away from the robot, is the bed to water. Under the
head, two friction trip collars ride on the stationary bearing sleeve, each with a tab; the head's trip lever hangs behind
the head (opposite the nozzle) beside both tabs. With the water on, the head turns until its lever strikes a tab, then
reverses, so the head waters the arc between the collars on the side opposite them. At the start it waters 220 degrees
(from -110 to +110).

Run it:  inspect-robots run --embodiment yam_sprinkler -E task=task.py -E object_jitter=0 --policy yam_scripted --no-prompt \\
           --instruction "Set the sprinkler so it waters only the bed between the two orange flags, then take your hands away." \\
           --scorer task_score

`yam_sprinkler` (sprinkler_arc/embodiment.py) builds the sprinkler, reports each gripper's fingertip midpoint and the
head's and collars' joint angles, and sets the head's drive from drive_command() after every control step.

The water (a scripted drive: water is not simulated) is off until a hand has been within 17 cm of the sprinkler's axis and
then both hands have been clear (17 cm or more from the axis, or 6.5 cm above the head's top) and still for 2 s. Then the
head turns counter-clockwise at 20 deg/s, with at most 0.015 N m, and reverses each time it stalls: its lever has struck a
collar's tab (a collar slips only at 0.04 N m, so the tab holds). A hand coming back within the clear zone shuts the water
off and clears the sweep record. Each reversal's nozzle heading is recorded; the sweep's ends are the latest clockwise and
counter-clockwise reversals, once each way has had one.
"""

import math
import os

import numpy as np

from sprinkler_arc import mech as M
from yam_arena import ik
from yam_arena.constants import FINGER_OPEN, READY
from yam_arena.demo import Waypoint, grip_rotation
from yam_arena.station import BASES
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]
AX = np.array((M.AXIS_X, M.AXIS_Y))
HEAD_Z = TOP + M.Z0  # world height of the head's inlet bottom (all three moving parts' origin)


OBJECTS = [Box("sprinkler", pos=(M.AXIS_X, M.AXIS_Y, TOP), size=(0.01, 0.01, 0.01))]  # the whole articulation (mech.py)

# -- criteria ----------------------------------------------------------------------------------------------------------
CLEAR_R = 0.17  # a hand is clear of the sprinkler with its fingertip midpoint this far from the axis (m, in plan) ...
CLEAR_ABOVE = 0.065  # ... or this far above the head's top
STILL_STEP = 0.001  # a hand is still when it moves under 1 mm in a control step (20 mm/s)
OPEN_AFTER = 2.0  # s clear and still before the water comes on
STALL_FRAC, STALL_S, REV_GRACE = 0.3, 0.10, 0.5  # stalled: under 30% of the set speed for 0.1 s, 0.5 s after a start or reversal
NEED_REVERSALS = 1  # each end must have been reached at least once (the latest reading each way counts)
BANDS = ((10.0, 1.0), (20.0, 0.5), (30.0, 0.25))  # an end within this many degrees of its flag earns this much


def wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


def yaw_deg(state):
    x, y, z, w = (float(v) for v in state.quat_xyzw)
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _fresh():
    return {"water": False, "dir": 1, "since": 0.0, "stall": 0.0, "clear": 0.0, "touched": os.environ.get("SPRINKLER_CAL") == "1", "rev": [], "last": {},
            "t": 0.0, "opened": 0}


_trial = _fresh()


def reset_latch():
    """Forget the trial: the embodiment calls this at every reset."""
    _trial.clear()
    _trial.update(_fresh())


def hand_clear(objects, hand):
    p = np.asarray(objects[hand].pos, float)
    return bool(np.hypot(*(p[:2] - AX)) >= CLEAR_R or p[2] >= HEAD_Z + M.HEAD_TOP + CLEAR_ABOVE)


def note_step(objects, dt):
    """After every control step: the hands, the water valve, the head's stall and reversals."""
    tr = _trial
    tr["t"] += dt
    clear = all(hand_clear(objects, h) for h in ("left_tcp", "right_tcp"))
    still = True
    for h in ("left_tcp", "right_tcp"):
        p = np.asarray(objects[h].pos, float)
        if h in tr["last"] and np.linalg.norm(p - tr["last"][h]) > STILL_STEP:
            still = False
        tr["last"][h] = p
    if not clear:
        tr["touched"] = True
        if tr["water"]:  # the supervisor shuts the valve when a hand reaches in again; the sweep record starts over
            tr["water"], tr["rev"] = False, []
    tr["clear"] = tr["clear"] + dt if (clear and still) else 0.0
    if not tr["water"] and tr["touched"] and tr["clear"] >= OPEN_AFTER:
        tr["water"], tr["dir"], tr["since"], tr["stall"], tr["rev"] = True, 1, 0.0, 0.0, []
        tr["opened"] += 1
    if tr["water"]:
        tr["since"] += dt
        w = math.degrees(float(objects["head"].ang_vel[2]))
        slow = tr["dir"] * w < STALL_FRAC * M.DRIVE_SPEED
        tr["stall"] = tr["stall"] + dt if (slow and tr["since"] >= REV_GRACE) else 0.0
        if tr["stall"] >= STALL_S - 1e-9:
            tr["rev"].append(("ccw" if tr["dir"] > 0 else "cw", yaw_deg(objects["head"]), tr["t"]))
            tr["dir"], tr["since"], tr["stall"] = -tr["dir"], 0.0, 0.0


CAL = os.environ.get("SPRINKLER_CAL") == "1"  # dev: the water comes on at once and the stall torque rises 0.01 N m a reversal


def drive_command():
    """(target deg/s, torque limit N m) for the head joint's velocity drive: off (no torque) with the water off; with it on,
    held at DRIVE_TORQUE until the head nears DRIVE_SPEED. The bearing's friction acts either way."""
    if _trial["water"]:
        torque = 0.01 * (1 + len(_trial["rev"])) if CAL else M.DRIVE_TORQUE
        return (_trial["dir"] * M.DRIVE_SPEED, torque)
    return (0.0, 0.0)


def sweep_ends(rev=None):
    """(clockwise end, counter-clockwise end) nozzle headings in deg: the latest reversal each way, or None until each way
    has had one."""
    rev = _trial["rev"] if rev is None else rev
    cw = [h for s, h, _ in rev if s == "cw"]
    ccw = [h for s, h, _ in rev if s == "ccw"]
    if len(cw) < NEED_REVERSALS or len(ccw) < NEED_REVERSALS:
        return None
    return cw[-1], ccw[-1]


def end_credit(err):
    for lim, credit in BANDS:
        if err <= lim + 1e-9:
            return credit
    return 0.0


def score_ends(ends):
    """Mean of the two ends' credits: each end's error is its distance from its flag's heading."""
    if ends is None:
        return 0.0
    e_cw, e_ccw = abs(wrap(ends[0] - M.TARGET[0])), abs(wrap(ends[1] - M.TARGET[1]))
    return 0.5 * (end_credit(e_cw) + end_credit(e_ccw))


def success(objects):
    return bool(_trial["water"] and score_ends(sweep_ends()) >= 1.0)


def score(objects):
    """Read at the end of the episode (or at success): 0 until the water has run long enough for each end of the sweep to be
    reached with both hands clear; then the mean of the two ends' credits (1 within 10 degrees of its flag, 0.5 within
    20, 0.25 within 30, else 0)."""
    return score_ends(sweep_ends()) if _trial["water"] else 0.0


def debug_line(objects, t):
    tr = _trial
    r = lambda v: np.round(1000 * np.asarray(v), 1).tolist()  # noqa: E731
    return (f"SPRINKLER t={t:6.2f} head {yaw_deg(objects['head']):7.2f} lower {yaw_deg(objects['lower']):7.2f} upper {yaw_deg(objects['upper']):7.2f} "
            f"w {math.degrees(float(objects['head'].ang_vel[2])):6.1f} water {int(tr['water'])} dir {tr['dir']} clear {tr['clear']:.2f} "
            f"L {r(objects['left_tcp'].pos)} R {r(objects['right_tcp'].pos)} rev {[(s, round(h, 1)) for s, h, _ in tr['rev']]} score {score(objects):.3f}")


# -- scripted demo -----------------------------------------------------------------------------------------------------
# Open loop from the known start: the plan reads no simulator state. With the nozzle at -23 degrees the head's top (drive
# arm, spoon, counterweight, nozzle) is clear of the lower collar's whole travel: the left hand comes in low, under the
# head's top, and pushes that collar's tab counter-clockwise with the side of a finger. It then pushes the head round by
# its counterweight until the nozzle points at 23 degrees, clear of the upper collar's travel, and parks; the right hand
# pushes the upper collar's tab clockwise and parks. SPRINKLER_DEMO switches the shortcut variants:
# "nothing": no motion. "lower_only": only the lower collar is set. "short": each collar stops 15 degrees short of its
# place. "head_only": the head is turned to point between the flags; the collars are left alone. "no_head": the collars are
# pushed as in the demo but the head is not turned between them.
VARIANT = os.environ.get("SPRINKLER_DEMO", "")


def _env(name, default):
    return float(os.environ.get(name, default))


SIDE = {"left": 1, "right": -1}
PARK = 0.12, 0.34, TOP + 0.18
DT = 0.05
PUSH_SPEED = _env("SPRINKLER_SPEED", 12.0)  # deg/s round the axis while pushing
BACKOFF = 6.0  # deg: a finger comes down this far behind the face it will push
HEAD_END = 23.0  # deg: the head is turned to point here before the upper collar, square to that collar's whole travel
HEAD_OVER = _env("SPRINKLER_HEAD_OVER", 7.0)  # deg: the head is pushed this much further, for the arm's give
OVER = {"lower": _env("SPRINKLER_OVER_LOWER", 8.3), "upper": _env("SPRINKLER_OVER_UPPER", 8.3)}  # deg past the goal: the arm's give
SAG = _env("SPRINKLER_SAG", 0.0)  # every TCP target is aimed this much lower
PRESS = _env("SPRINKLER_PRESS", 0.007)  # a collar-pushing finger is aimed this much lower still, so it doesn't ride up the tab
# How a finger meets each part (TCP frame of an open grip: the inner finger occupies 47-71 mm from the TCP toward the axis;
# its lower blade, up to 20 mm above the TCP, is 40 mm across and 10 mm thick; its two lobes, 20-44 mm above the TCP, reach
# 66 mm from the TCP and 28 mm to either side).
TOUCH = {
    # part: (TCP's distance from the axis, the touching edge's offset across the finger, TCP height above the inlet bottom)
    "lower": (0.090, 0.028, sum(M.COLLAR_Z["lower"]) / 2 - 0.032),  # the lobes on the tab: r 24-42 mm against the tab's 15.5-30
    "upper": (0.090, 0.028, sum(M.COLLAR_Z["upper"]) / 2 - 0.032),
    "spoon": (0.100, 0.020, 0.098),  # the lower blade on the spoon: r 43-53 mm against 44-58, z 85-118 mm against 78-110
    "weight": (0.092, 0.020, 0.105),  # the lower blade on the counterweight: r 35-45 mm against 36-53, z 92-125 against 92-110
}
TRANSIT_Z = HEAD_Z + M.HEAD_TOP + 0.013 + 0.015  # every move between parts: fingertips 15 mm over the head's top
HALF = {"lower": M.TAB["w"] / 2, "upper": M.TAB["w"] / 2, "spoon": M.SPOON["w"] / 2, "weight": M.WEIGHT["w"] / 2}


def push_tcp(part, face_deg, push_dir, prev_yaw, dr=0.0):
    """TCP position and grip yaw for an open grip whose inner finger touches, from behind, the face of a radial part whose
    centre line is at face_deg; push_dir +1 pushes the part counter-clockwise (finger on its clockwise side)."""
    r_t, edge, z = TOUCH[part]
    a = math.radians(face_deg)
    u, v = np.array((math.cos(a), math.sin(a))), np.array((-math.sin(a), math.cos(a)))
    p = AX + (r_t + dr) * u - push_dir * (HALF[part] + edge) * v
    yaw = min((a - math.pi / 2, a + math.pi / 2), key=lambda y: abs(_unwrap_near(y, prev_yaw) - prev_yaw))
    yaw = _unwrap_near(yaw, prev_yaw)
    if part == "weight" and yaw > 0.0:  # the left wrist turns the other way round
        yaw -= math.pi
    return np.array((p[0], p[1], HEAD_Z + z - SAG - (PRESS if part in ("lower", "upper") else 0.0))), yaw


def _unwrap_near(yaw, ref):
    while yaw - ref > math.pi:
        yaw -= 2 * math.pi
    while yaw - ref < -math.pi:
        yaw += 2 * math.pi
    return yaw


class _Plan:
    def __init__(self):
        self.wps = {a: [] for a in SIDE}
        self.t = {a: 0.0 for a in SIDE}
        self.pushes = []  # (part, arm, [(t, face deg)]) for dev/clearance.py
        self.last = {}
        for a in SIDE:
            pos, _ = ik.tcp_pose(np.array(READY, float), np.array(BASES[a]))
            self.last[a] = (tuple(float(v) for v in pos), FINGER_OPEN, 0.0)

    def go(self, arm, dur, pos=None, finger=None, label="", yaw=None):
        p0, f0, y0 = self.last[arm]
        p = p0 if pos is None else tuple(float(v) for v in pos)
        f = f0 if finger is None else finger
        y = y0 if yaw is None else yaw
        n = 1 if label == "down" else max(1, math.ceil(np.linalg.norm(np.subtract(p, p0)) / 0.03), math.ceil(abs(y - y0) / math.radians(15)))
        for k in range(1, n + 1):
            q = tuple(float(v) for v in np.asarray(p0) + (np.asarray(p) - np.asarray(p0)) * k / n)
            self.wps[arm].append(Waypoint(dur / n, q, f0 + (f - f0) * k / n, label, y0 + (y - y0) * k / n))
        self.last[arm] = (p, f, y)
        self.t[arm] += dur

    def until(self, arm, t):
        if t > self.t[arm] + 1e-9:
            self.go(arm, t - self.t[arm], label="wait")

    def sync(self):
        t = max(self.t.values())
        for a in SIDE:
            self.until(a, t)


OUT = 0.035  # a finger comes down this far out from its working place and slides in under the head's top


def _approach(plan, arm, part, face_deg, d, label):
    p, y = push_tcp(part, face_deg, d, plan.last[arm][2])
    head_part = part in ("spoon", "weight")
    q, _ = push_tcp(part, face_deg, d, plan.last[arm][2], 0.015 if head_part else OUT)
    if head_part:  # come in low from well outside the head's top
        w, _ = push_tcp(part, face_deg, d, plan.last[arm][2], 0.06)
        plan.go(arm, 1.5, w + (0.0, 0.0, 0.03), FINGER_OPEN, f"{label}: outside", y)
    plan.go(arm, 2.0 if not head_part else 1.0, q + (0.0, 0.0, 0.0 if head_part else 0.06), FINGER_OPEN, f"{label}: above", y)
    plan.go(arm, 0.3, label="settle")
    plan.go(arm, 1.0, q, label=f"{label}: down", yaw=y)
    plan.go(arm, 0.8, p, label=f"{label}: in", yaw=y)


def _leave(plan, arm, part, face_deg, d):
    p, y = push_tcp(part, face_deg - d * 5.0, d, plan.last[arm][2])
    q, _ = push_tcp(part, face_deg - d * 5.0, d, plan.last[arm][2], OUT)
    plan.go(arm, 0.5, p, label="back off", yaw=y)
    plan.go(arm, 0.6, q, label="out", yaw=y)
    plan.go(arm, 0.8, q + (0.0, 0.0, 0.06), label="lift", yaw=y)


def _phase(plan, collar, arm, start, goal):
    """One collar's tab pushed from just behind its start to goal, plus OVER for the arm's give."""
    d = 1 if goal > start else -1
    goal += d * OVER[collar]
    th = start - d * BACKOFF
    _approach(plan, arm, collar, th, d, f"{collar} collar")
    plan.sync()
    plan.go(arm, 0.4, label="settle")
    plan.sync()
    track = [(plan.t[arm], th)]
    step = PUSH_SPEED * DT * 2  # one waypoint every 2 control steps
    while (goal - th) * d > 1e-9:
        th += d * min(step, abs(goal - th))
        p, y = push_tcp(collar, th, d, plan.last[arm][2])
        plan.go(arm, 2 * DT, p, label=f"{collar} collar: push", yaw=y)
        track.append((plan.t[arm], th))
    plan.pushes.append((collar, arm, track))
    plan.go(arm, 0.4, label="hold")
    _leave(plan, arm, collar, goal, d)


def _turn_head(plan, arm, start, goal):
    """Push the head round by its counterweight from nozzle heading start to goal (deg)."""
    d = 1 if goal > start else -1
    th = start + 180.0 - d * BACKOFF
    _approach(plan, arm, "weight", th, d, "turn the head")
    track = [(plan.t[arm], th - 180.0)]
    while (goal + 180.0 - th) * d > 1e-9:
        th += d * min(PUSH_SPEED * DT * 2, abs(goal + 180.0 - th))
        p, y = push_tcp("weight", th, d, plan.last[arm][2])
        plan.go(arm, 2 * DT, p, label="turn the head: push", yaw=y)
        track.append((plan.t[arm], th - 180.0))
    plan.pushes.append(("head", arm, track))
    plan.go(arm, 0.3, label="hold")
    _leave(plan, arm, "weight", goal + 180.0, d)


def _park(plan, arm, dur=1.6):
    plan.go(arm, dur, (PARK[0], SIDE[arm] * PARK[1], PARK[2]), None, "away", 0.0)


def demo():
    global LAST_PLAN
    if VARIANT == "nothing":
        return {}
    plan = _Plan()
    for arm in SIDE:
        p0 = plan.last[arm][0]
        plan.go(arm, 1.0, (p0[0], p0[1], TOP + 0.20), label="down")
    short = 15.0 if VARIANT == "short" else 0.0
    head = VARIANT != "no_head"
    if VARIANT == "head_only":
        th = M.START["head"] - BACKOFF
        _approach(plan, "right", "spoon", th, 1, "spoon")
        track = [(plan.t["right"], th)]
        while th < 0.0:
            th = min(0.0, th + PUSH_SPEED * DT * 2)
            p, y = push_tcp("spoon", th, 1, plan.last["right"][2])
            plan.go("right", 2 * DT, p, label="spoon: push", yaw=y)
            track.append((plan.t["right"], th))
        plan.pushes.append(("head", "right", track))
        _leave(plan, "right", "spoon", 0.0, 1)
    else:
        _phase(plan, "lower", "left", M.START["lower"], M.GOAL_TAB["lower"] - short)
        if head:
            _turn_head(plan, "left", M.START["head"], HEAD_END + HEAD_OVER)
        _park(plan, "left")
        if VARIANT != "lower_only":
            plan.until("right", plan.t["left"] - 2.6)  # the head is turned and the left hand on its way out
            _phase(plan, "upper", "right", M.START["upper"], M.GOAL_TAB["upper"] + short)
    plan.sync()
    if plan.last["left"][0][1] < 0.3:
        _park(plan, "left")
    plan.until("right", plan.t["left"])
    if plan.last["right"][0][1] > -0.3:
        _park(plan, "right")
    plan.sync()
    for arm in SIDE:  # hands away and still while the water runs: the sweep is scored, and success ends the trial
        plan.go(arm, 20.0, label="wait")
    LAST_PLAN = plan
    return plan.wps


LAST_PLAN = None


def ik_errors(wps):
    worst_p = worst_r = 0.0
    bad = []
    for arm, ws in wps.items():
        q = np.array(READY, float)
        for w in ws:
            q, pe, re = ik.solve(np.asarray(w.pos, float), ik.matrix_to_quat_xyzw(grip_rotation(w.yaw)), q, base_pos=BASES[arm])
            worst_p, worst_r = max(worst_p, pe), max(worst_r, re)
            if pe > 1e-3 or re > math.radians(1.0):
                bad.append((arm, w.label[:24], tuple(round(v, 3) for v in w.pos), round(pe * 1000, 1), round(math.degrees(re), 2)))
    return worst_p, worst_r, bad


if __name__ == "__main__":
    wps = demo()
    t = max((sum(w.duration for w in ws) for ws in wps.values()), default=0.0)
    pe, re, bad = ik_errors(wps)
    print(f"demo {t:.1f} s ({t * 20:.0f} steps), IK {pe * 1000:.2f} mm / {math.degrees(re):.2f} deg, {len(bad)} bad", bad[:6])
    print("start tabs", round(M.START["lower"], 1), round(M.START["upper"], 1), "goal tabs", round(M.GOAL_TAB["lower"], 1), round(M.GOAL_TAB["upper"], 1),
          "stop", round(M.STOP_DEG, 2))
