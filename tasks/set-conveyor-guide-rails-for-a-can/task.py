"""Change a conveyor lane over to a 66 mm can. Created 11_16_26 11:10 EST.

On the station table stands a 560 mm section of 82.6 mm (3.25 in) tabletop-chain conveyor, not running, its lane across
the table. The far guide rail is the fixed datum. The near guide rail is adjustable: a 1/2 in rod at each end slides in a
cross block on a bracket post, and a 40 mm star knob on top of each block clamps its rod. As found, both knobs are locked
and the lane is 78 mm wide. A 68.0 mm setting plug (the can plus the 2 mm running clearance) stands in the lane in front
of each bracket, against the datum rail, and a full 66 mm drinks can stands on the chain at the infeed (right-hand) end,
outside the rails. The job: slacken both knobs, bring the rail in against the plugs, lock both knobs while the rail is
held there, stand the plugs on their two green rests, and push the can through the set lane past the left bracket.
(Edited 11_16_26 11:20 EST: the can is run through the lane; one hand holds the rail while the knobs are locked.)

Run it:  inspect-robots run --embodiment yam_guide_rail -E task=task.py -E object_jitter=0 --policy yam_scripted \\
           --no-prompt --instruction "..." --scorer task_score

`yam_guide_rail` (guide_rails/embodiment.py) spawns the parts and their joints, reports each gripper's fingertip
midpoint, and sets each joint's friction limit from the clamp law in guide_rails/mech.py after every control step: a
knob's turning torque and its rod's holding force both follow the knob's tightness. The score is in
guide_rails/scoring.py.
"""

import math
import os

import numpy as np

from guide_rails import mech as M
from guide_rails import scoring as S
from yam_arena import ik
from yam_arena.constants import FINGER_OPEN, READY
from yam_arena.demo import Waypoint, grip_rotation
from yam_arena.station import BASES
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]
_TINY = (0.01, 0.01, 0.01)


def _at(p):
    return (p[0], p[1], TOP + p[2])


OBJECTS = [
    Box("conveyor", pos=(0.0, 0.0, TOP), size=_TINY),
    Box("rail", pos=_at(M.RAIL_START), size=_TINY, mass=M.RAIL_MASS),
    Box("knob_left", pos=_at((M.XB, M.STATIONS["left"], M.KNOB_Z)), size=_TINY, mass=M.KNOB_MASS),
    Box("knob_right", pos=_at((M.XB, M.STATIONS["right"], M.KNOB_Z)), size=_TINY, mass=M.KNOB_MASS),
    Box("plug_left", pos=_at(M.PLUG_START["left"]), size=_TINY, mass=M.PLUG_MASS),
    Box("plug_right", pos=_at(M.PLUG_START["right"]), size=_TINY, mass=M.PLUG_MASS),
    Box("can", pos=_at(M.CAN_START), size=_TINY, mass=M.CAN_MASS),
]
TRACE = None  # dev only: a list here gets each step's readings (dev/sim.py)


def reset_trial():
    S.reset()


def note_step(objects, dt):
    S.note_step(objects, dt, TOP)
    if TRACE is not None:
        c = S.clearances(objects)
        R = S.rot(objects["rail"].quat_xyzw)
        yaw = {s: math.degrees(math.atan2(*S.rot(objects[f"plug_{s}"].quat_xyzw)[[1, 0], 0])) for s in S.SIDES}
        TRACE.append(dict(c=dict(c), a={s: S.tightness(s) for s in S.SIDES}, tcp={h: np.array(objects[f"{h}_tcp"].pos) for h in S.SIDES},
                          wrist={h: np.array(objects[f"{h}_wrist"].pos) for h in S.SIDES if f"{h}_wrist" in objects},
                          roll=math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0])))), skew=math.degrees(math.atan2(R[1, 0], R[0, 0])), plug_yaw=yaw,
                          rail=np.array(objects["rail"].pos), plug={s: np.array(objects[f"plug_{s}"].pos) for s in S.SIDES},
                          can=np.array(objects["can"].pos), score=S.score(objects, TOP)))


def take_commands():
    return S.take_commands()


def score(objects):
    return S.score(objects, TOP)


def success(objects):
    return S.score(objects, TOP) >= 1.0


def debug_line(objects):
    return S.debug_line(objects, TOP)


# -- scripted demo -----------------------------------------------------------------------------------------------------
# Open loop from the fixed start: it uses only where the brackets, knobs, rail, plugs and rests stand, which a policy can
# see, and reads no simulator state. RAIL_DEMO picks a shortcut, for testing the score:
#   "none"   nothing is done;
#   "shove"  the rail is pushed at the plugs without slackening the knobs;
#   "loose"  the rail is set against the plugs and let go, the knobs never locked again;
#   "one"    only the left knob is locked;
#   "plugs"  everything but the plugs, which are left in the lane;
#   "can"    the plugs are taken out first, the can pushed in mid-lane and the rail set against the can itself;
#   "norun"  everything but running the can through.
FINGER_PER_M = 0.04695 / 0.094
ARMS = {"left": 1, "right": -1}
PARK = (0.10, 0.36, 0.17)
SEGMENT = 0.03
KNOB_OPEN, KNOB_GRIP = 0.062, M.KNOB_D - 0.010  # finger openings round a knob: clear, and squeezed 5 mm a side
KNOB_DZ = 0.002  # TCP above the knob's centre
YAW_SLACK = {"left": (math.radians(-45.0), math.radians(50.0)), "right": (math.radians(5.0), math.radians(100.0))}  # a 95 degree turn out
YAW_LOCK = {"left": (math.radians(60.0), math.radians(-50.0)), "right": (math.radians(125.0), math.radians(15.0))}  # 110 degrees in
TURN_S = 2.4
HOLD_Y = 0.055  # the rail is held this far from its middle toward the holding arm, inside the two plugs (the arms clear)
HOLD_YAW = {"left": -0.5 * math.pi, "right": 0.5 * math.pi}  # fingers close along x, across the rail
HOLD_OPEN, HOLD_GRIP = 0.030, M.RAIL_T + 0.002  # the fingers cage the rail, 1 mm clear a side, and push it in
# (an open-handed push, tried, let the rail turn and seat on one plug only)
HOLD_DZ = 0.003  # TCP above the rail's centre line
PRESS = 0.016  # the holding hand is aimed this far past where the rail meets the plugs: the arm's give makes the force (20 mm also sets it, but locks in more preload on the plugs)
PLUG_OPEN, PLUG_GRIP = 0.088, M.PLUG_D - 0.010
PLUG_DZ = M.PLUG_H / 2 - 0.022  # TCP above the plug's centre
PLUG_LIFT = 0.020
PLUG_OVER = 0.012
TUG = 0.03  # dev: how far back the "tug" test aims
CAL = dict(hold_x=0.0, rest_x=0.0, rest_y=0.0)  # fixed offsets against the arms' give (dev/sim.py measures them)


def set_variant(name):
    global VARIANT
    VARIANT = name


set_variant(os.environ.get("RAIL_DEMO", ""))


def _fingers(width):
    return -FINGER_PER_M * width


class _Plan:
    def __init__(self):
        self.wps = {a: [] for a in ARMS}
        self.t = {a: 0.0 for a in ARMS}
        self.last = {}
        for a in ARMS:
            pos, _ = ik.tcp_pose(np.array(READY, float), np.array(BASES[a]))
            self.last[a] = (tuple(float(v) for v in pos), FINGER_OPEN, 0.0)

    def go(self, arm, dur, pos=None, finger=None, label="", yaw=None, whole=False):
        p0, f0, y0 = self.last[arm]
        p = p0 if pos is None else tuple(float(v) for v in pos)
        f = f0 if finger is None else finger
        y = y0 if yaw is None else yaw
        n = 1 if whole else max(1, math.ceil(np.linalg.norm(np.subtract(p, p0)) / SEGMENT), math.ceil(abs(y - y0) / math.radians(15)))
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
        for a in ARMS:
            self.until(a, t)
        return t


def _w(x, y, z):
    return (x, y, TOP + z)


def _park(plan, arm, dur=1.6):
    plan.go(arm, dur, _w(PARK[0], ARMS[arm] * PARK[1], PARK[2]), FINGER_OPEN, "away", yaw=0.0)


def _turn_knob(plan, arm, yaws, what):
    """Take that arm's knob from above, turn it from yaws[0] to yaws[1] (counter-clockwise positive), and let go."""
    x, y, z = M.XB, M.STATIONS[arm], M.KNOB_Z + KNOB_DZ
    plan.go(arm, 1.8, _w(x, y, z + 0.045), _fingers(KNOB_OPEN), f"knob: over it ({what})", yaw=yaws[0])
    plan.go(arm, 0.3, label="settle")
    plan.go(arm, 0.8, _w(x, y, z), label="knob: down")
    plan.go(arm, 0.6, finger=_fingers(KNOB_GRIP), label="knob: grip")
    plan.go(arm, TURN_S, yaw=yaws[1], label=f"knob: {what}")
    plan.go(arm, 0.5, label=f"knob: {what} hold")
    plan.go(arm, 0.5, finger=_fingers(KNOB_OPEN), label="knob: let go")
    plan.go(arm, 0.7, _w(x, y, z + 0.055), label="knob: up")


def hold_point(arm, x_face, press=0.0):
    """The holding hand's TCP with the rail's product face at ``x_face``, aimed ``press`` further in."""
    y = ARMS[arm] * HOLD_Y
    return _w(x_face - M.RAIL_T / 2 + press + CAL["hold_x"], y, M.RAIL_Z + HOLD_DZ)


def _take_rail(plan, arm, x_face, press):
    """Cage the rail (its face at x_face) between the fingers and press it toward the datum rail."""
    p = np.array(hold_point(arm, x_face))
    plan.go(arm, 1.8, p + (0, 0, 0.055), _fingers(HOLD_OPEN), "rail: over it", yaw=HOLD_YAW[arm])
    plan.go(arm, 0.3, label="settle")
    plan.go(arm, 0.8, p, label="rail: down")
    plan.go(arm, 0.3, label="settle")
    plan.go(arm, 0.5, finger=_fingers(HOLD_GRIP), label="rail: grip")
    plan.go(arm, 1.5, hold_point(arm, x_face_target(), press), label="rail: press in")
    plan.go(arm, 0.5, label="rail: hold")


def x_face_target():
    """Where the rail's face stops: against the plugs, or against the can for the 'can' shortcut."""
    return M.X_DATUM - (M.CAN_D if VARIANT == "can" else M.PLUG_D)


def _leave_rail(plan, arm):
    p = np.array(plan.last[arm][0])
    plan.go(arm, 0.5, finger=_fingers(HOLD_OPEN), label="rail: let go")
    plan.go(arm, 0.3, label="settle")
    plan.go(arm, 0.6, p + (0, 0, 0.045), label="rail: up")  # straight up first (the wrist is near its limit here)
    plan.go(arm, 0.5, p + (-0.025, 0, 0.065), label="rail: up")


PLUG_HANDOVER = 0.020  # the right hand pushes its plug along the lane until the plug's centre is here (y)
PUSH_LEAD = 0.012  # the pushing hand is aimed this far past where the plug should stop: it lags against the plug's drag
PLUG_EXIT = M.RAIL_LEN / 2 + 0.085  # a plug slid this far along the lane (y) is clear of the rails' ends


def _plug_out(plan, y, rest, opening=PLUG_OPEN):
    """The left hand takes the plug standing in the lane at ``y``, slides it out along the lane past the rails' left
    end, and stands it on ``rest``."""
    x, z = M.XC, M.CHAIN_TOP + M.PLUG_H / 2
    rx, ry = rest
    rx, ry = rx + CAL["rest_x"], ry + CAL["rest_y"]
    g = np.array(_w(x, y, z + PLUG_DZ))
    down = M.REST_T + M.PLUG_H / 2 + PLUG_DZ
    if y < M.STATIONS["left"]:  # over the left knob, not past it
        plan.go("left", 1.0, _w(0.30, 0.17, 0.175), _fingers(opening), "plug: over it", yaw=0.0)
    plan.go("left", 1.8, g + (0, 0, 0.05), _fingers(opening), "plug: over it", yaw=0.0)
    plan.go("left", 0.3, label="settle")
    plan.go("left", 0.9, g, label="plug: down")
    plan.go("left", 0.6, finger=_fingers(PLUG_GRIP), label="plug: grip")
    plan.go("left", 0.6, g + (0, 0, PLUG_LIFT), label="plug: lift")
    plan.go("left", 1.8, _w(x, PLUG_EXIT, z + PLUG_DZ + PLUG_LIFT), label="plug: out along the lane")
    plan.go("left", 2.2, _w(rx, ry, down + 0.03), label="plug: to a rest")
    plan.go("left", 1.0, _w(rx, ry, down + 0.001), label="plug: stand it down")
    plan.go("left", 0.5, finger=_fingers(PLUG_OPEN), label="plug: let go")
    plan.go("left", 0.8, _w(rx, ry, down + 0.07), label="up")


def _stow_plugs(plan):
    """Both plugs leave the lane by its left end: the can stands at the right-hand (infeed) end. The left hand takes its
    own plug out first; meanwhile the right hand, closed fingers behind its plug in the lane, pushes that plug along the
    lane to mid-lane and backs away, and the left hand then takes it out too."""
    _plug_out(plan, M.PLUG_START["left"][1], M.REST["left"])
    x, y, z = M.PLUG_START["right"]
    zp = z + PLUG_DZ
    behind = y - M.PLUG_D / 2 - 0.012  # the closed fingertips 6 mm short of the plug's back; they widen to 20 mm a side
    # 45 mm up, which still leaves 25 mm to the can
    plan.go("right", 1.2, _w(0.22, -0.13, 0.20), 0.0, "plug: behind it", yaw=0.0)  # clear of the can
    plan.go("right", 1.0, _w(x, behind, zp + 0.06), label="plug: behind it")
    plan.go("right", 0.8, _w(x, behind, zp), label="plug: down behind")
    plan.go("right", 3.0, _w(x, PLUG_HANDOVER - M.PLUG_D / 2 - 0.006 + PUSH_LEAD, zp), label="plug: push along")
    plan.go("right", 0.6, _w(x, PLUG_HANDOVER - M.PLUG_D / 2 - 0.04, zp), label="plug: back off")
    plan.go("right", 0.8, _w(x - 0.03, PLUG_HANDOVER - 0.08, zp + 0.06), label="plug: away")
    plan.go("right", 0.6, _w(x - 0.07, PLUG_HANDOVER - 0.10, 0.20), label="plug: away")  # up, clear of the right knob
    _park(plan, "right")
    plan.until("left", plan.t["right"] - 1.4)
    _plug_out(plan, PLUG_HANDOVER, M.REST["right"], opening=0.094)  # wide open: the pushed plug's stop varies


RUN_X, RUN_Z = 0.320, 0.125  # the pushing hand's TCP: 15 mm short of the lane's centre line, fingertips 9 mm over the rails
CAN_BACK = 0.040  # the closed fingers' back face to the TCP (6 mm) plus the can's radius
RUN_HANDOVER = 0.005  # the right hand pushes the can this far (TCP y); the left takes it on from behind
RUN_END = 0.135  # the left hand's last TCP y: the can's centre about 175 mm, past the left bracket, short of the end


def _push_in(plan):
    """The right hand pushes the sample can in from the infeed end to mid-lane, closed fingers on its back above the
    rails' tops, and goes back to its park."""
    cy = M.CAN_START[1]
    plan.go("right", 1.8, _w(RUN_X, cy - CAN_BACK - 0.03, RUN_Z + 0.05), 0.0, "can: behind it", yaw=0.0)
    plan.go("right", 0.8, _w(RUN_X, cy - CAN_BACK - 0.03, RUN_Z), label="can: down")
    plan.go("right", 4.5, _w(RUN_X, RUN_HANDOVER, RUN_Z), label="can: push in")
    plan.go("right", 0.6, _w(RUN_X, RUN_HANDOVER - 0.035, RUN_Z), label="can: back off")
    plan.go("right", 0.8, _w(0.25, RUN_HANDOVER - 0.05, RUN_Z + 0.045), label="can: out")
    _park(plan, "right")


def _run_can(plan):
    """Run the sample can through the set lane: the right hand pushes it in, the left takes it on from behind and
    pushes it out past the left bracket."""
    _push_in(plan)
    y0 = RUN_HANDOVER - 0.025  # the fingers widen to 20 mm a side 45 mm up: well behind the can
    plan.until("left", plan.t["right"] - 1.6)
    plan.go("left", 1.2, _w(0.16, 0.03, RUN_Z + 0.07), 0.0, "can: behind it", yaw=0.0)  # inside the left knob
    plan.go("left", 0.8, _w(0.25, y0, RUN_Z + 0.03), label="can: behind it")
    plan.go("left", 0.8, _w(RUN_X, y0, RUN_Z), label="can: in behind")
    plan.go("left", 3.5, _w(RUN_X, RUN_END, RUN_Z), label="can: push through")
    plan.go("left", 0.6, _w(RUN_X, RUN_END - 0.035, RUN_Z), label="can: back off")
    plan.go("left", 0.8, _w(0.25, RUN_END - 0.05, RUN_Z + 0.045), label="can: out")
    _park(plan, "left")


def demo():
    plan = _Plan()
    for arm in ARMS:  # down from the ready pose first, so the wrists turn low
        p0 = plan.last[arm][0]
        plan.go(arm, 1.0, (p0[0], p0[1], TOP + 0.19), label="down", whole=True)
    if VARIANT == "none":
        for arm in ARMS:
            plan.go(arm, 3.0, label="wait")
        return plan.wps
    if VARIANT == "can":  # shortcut: no plugs; the can is pushed in mid-lane and the rail set against it
        _stow_plugs(plan)
        plan.sync()
        _push_in(plan)
        plan.sync()
    if VARIANT != "shove":
        for arm in ARMS:  # slacken both knobs, each arm its own
            _turn_knob(plan, arm, YAW_SLACK[arm], "slacken")
        plan.sync()
    # the right hand brings the rail in and holds it while the left locks its knob
    _take_rail(plan, "right", M.X_FACE_START, PRESS)
    if VARIANT in ("shove", "loose"):
        plan.go("right", 2.0, label="rail: hold")
        _leave_rail(plan, "right")
        _park(plan, "right")
        _park(plan, "left")
    else:
        # the right hand holds the rail against both plugs while the left locks its knob; then the right lets go (the
        # rail now hangs on the locked left rod, its right end resting against the right plug) and locks its own knob
        plan.until("left", plan.t["right"] - 2.6)
        _turn_knob(plan, "left", YAW_LOCK["left"], "lock")
        plan.until("right", plan.t["left"] - 0.5)
        _leave_rail(plan, "right")
        if VARIANT != "one":
            _turn_knob(plan, "right", YAW_LOCK["right"], "lock")
        if VARIANT == "tug":  # dev: with both knobs locked, the right hand takes the rail again and pulls it back
            plan.sync()
            p = np.array(hold_point("right", x_face_target()))
            plan.go("right", 1.8, p + (0, 0, 0.055), _fingers(HOLD_OPEN), "tug: over the rail", yaw=HOLD_YAW["right"])
            plan.go("right", 0.8, p, label="tug: down")
            plan.go("right", 0.5, finger=_fingers(M.RAIL_T - 0.008), label="tug: grip")
            plan.go("right", 1.5, p - (TUG, 0, 0), label="tug: pull back")
            plan.go("right", 2.0, label="tug: pull hold")
            plan.go("right", 1.0, p, label="tug: ease")
            _leave_rail(plan, "right")
        if VARIANT in ("one", "plugs", "can", "tug"):
            for arm in ARMS:
                _park(plan, arm)
        else:
            plan.sync()
            _stow_plugs(plan)
            if VARIANT == "norun":
                for arm in ARMS:
                    _park(plan, arm)
            else:
                _park(plan, "left", 1.2)
                plan.sync()
                _run_can(plan)
    end = max(plan.t.values()) + 2.0
    for a in ARMS:
        plan.until(a, end)
    return plan.wps


def ik_errors(wps):
    worst_p = worst_r = jump = 0.0
    bad = []
    for arm, ws in wps.items():
        q = np.array(READY, float)
        for w in ws:
            q0 = q
            q, pe, re = ik.solve(np.asarray(w.pos, float), ik.matrix_to_quat_xyzw(grip_rotation(w.yaw)), q, base_pos=BASES[arm])
            worst_p, worst_r, jump = max(worst_p, pe), max(worst_r, re), max(jump, float(np.abs(q - q0).max()))
            edge = float(np.minimum(q - ik.LOWER, ik.UPPER - q).min())
            if pe > 1e-3 or re > math.radians(1.0) or edge < math.radians(1.0):
                bad.append((arm, w.label[:24], tuple(round(v, 3) for v in w.pos), round(pe * 1000, 1), round(math.degrees(re), 2), round(math.degrees(edge), 1)))
    return worst_p, worst_r, bad, jump


# -- self-test (no simulator): the demo fits the run limit and IK reaches every waypoint inside the joint limits -----------
if __name__ == "__main__":
    for v in ("", "none", "shove", "loose", "one", "plugs", "can", "norun", "tug"):
        set_variant(v)
        wps = demo()
        t = max(sum(w.duration for w in ws) for ws in wps.values())
        pe, re, bad, jump = ik_errors(wps)
        print(f"{v or 'demo':6s} {t:5.1f} s ({t * 20:.0f} steps), IK {pe * 1000:.2f} mm / {math.degrees(re):.2f} deg, {len(bad)} bad, largest joint step {math.degrees(jump):.0f} deg", bad[:4])
        assert t * 20 < 2400 and not bad
