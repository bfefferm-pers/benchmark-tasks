"""Solder a wire into a terminal lug with an iron and solder wire. Created 11_15_26 (local time).

On the station table: an 8 AWG copper soldering lug (solder-type tube lug) screwed by its tongue to a post on a steel jig, its barrel pointing at the
robot along -x, with a stripped, pre-tinned 8 AWG wire lying loose in the barrel and in a cradle. A hot soldering iron
(orange grip block) lies on its stand to the left; a straight stick of solder wire in a green grip sleeve lies on its
rest to the right.

Run it:  inspect-robots run --embodiment yam_solder -E task=task.py -E object_jitter=0 --policy yam_scripted --no-prompt \\
           --instruction "Solder the wire into the lug with the iron and the solder, let the joint set, and put the iron and the solder back." --scorer task_score

`yam_solder` (lug_solder/embodiment.py) spawns the parts, reports each gripper's fingertip midpoint and the seconds each
pair of parts touched in the step (PhysX contact forces), switches the fixed joint between lug and wire, and shows the joint.

The joint is a lumped heat balance, kept in note_step (numbers and their reasons: README):
- Heat goes in only while the iron is in a hand and its tip touches the lug (contact force, not distance):
  min(67 W, 0.25 W/K x (400 C - T)). It leaves all the time at 0.07 W/K x (T - 25 C). Heat capacity 2.2 J/K.
- Solder melts off the stick at 8 mm/s while the stick, in the other hand, touches the lug or the wire at the barrel
  mouth and the joint is at 227 C or more; each mm takes 1.37 J. Solder touched to the iron melts onto the iron.
- Solder fed more than 0.5 s after the tip left the lug, anything touching the joint or the wire moving while it
  freezes (from 217 C down to 200 C) leaves a cold joint; reheating to 240 C with the iron reflows it.
- When a joint that is not cold sets with 10 mm or more of solder in it and the wire seated, lug and wire are joined.
"""

import math
import os

import numpy as np

from lug_solder import mech as M
from yam_arena import ik
from yam_arena.constants import FINGER_OPEN, READY
from yam_arena.demo import Waypoint, grip_rotation
from yam_arena.station import BASES
from yam_arena.task_file import TABLE_TOP, Box

SCENE = "station"
TOP = TABLE_TOP[SCENE]
AXIS = np.array((M.MOUTH[0], M.MOUTH[1], TOP + M.AXIS_Z))  # the lug frame's origin (mouth centre), world
WIRE0 = AXIS - (0.0, 0.0, M.WIRE_DROP)
IRON0 = np.array((M.IRON_REST["x"], M.IRON_REST["y"], TOP + M.IRON_REST_Z))
SOLDER0 = np.array((M.SOLDER_REST["x"], M.SOLDER_REST["y"], TOP + M.SOLDER_REST_Z))
_S = (0.01, 0.01, 0.01)
OBJECTS = [
    Box("bench", pos=(0.0, 0.0, TOP), size=_S),
    Box("lug", pos=tuple(AXIS), size=_S),
    Box("wire", pos=tuple(WIRE0 + (0.0, 0.0, 0.0002)), size=_S, mass=M.WIRE_MASS),
    Box("iron", pos=tuple(IRON0 + (0.0, 0.0, 0.0003)), size=_S, mass=M.IRON_MASS, yaw=M.IRON_REST["heading"]),
    Box("solder", pos=tuple(SOLDER0 + (0.0, 0.0, 0.0003)), size=_S, mass=M.SOLDER_MASS, yaw=M.SOLDER_REST["heading"]),
]
TOOLS = ("iron", "solder")

# -- the heat balance (README: where each number comes from) -------------------------------------------------------------
T_AMB, T_TIP = 25.0, 400.0  # C: room; the iron's set tip temperature
P_MAX, G_TIP, G_LOSS, CAP = 67.0, 0.25, 0.07, 2.2  # W, W/K, W/K, J/K
T_MELT, T_FLOW, T_REFLOW, T_SET, T_BURN = 217.0, 227.0, 240.0, 200.0, 300.0  # C
RATE = 8.0  # mm/s of 1.5 mm solder melting off the stick on contact
J_SENS, J_LAT = 0.58, 0.79  # J per mm of solder: to bring it to 217 C, and to melt it
F_BOND, F_MIN, F_MAX = 10.0, 30.0, 55.0  # mm of solder in the joint: enough to hold; the sound range
IRON_OFF = 0.5  # s: solder fed longer than this after the tip left the lug is a cold feed
ZONE = (-M.BARE, 0.003, 0.005)  # the joint: along the wire from the insulation's end to 3 mm inside the mouth, within 5 mm of the axis
MOVE, TURN = 0.0005, math.radians(1.0)  # the wire moving this much while the joint freezes disturbs it
SEATED = (0.0025, 0.0015)  # the wire no more than this far out of its seat (along, across) when the joint sets
HELD = 0.025
HANDS_CLEAR, LIN, ANG, STILL_S = 0.06, 0.01, 0.5, 1.0
REST_XY, REST_Z, REST_TILT, REST_TURN = 0.012, 0.006, math.radians(10.0), math.radians(20.0)


def _fresh():
    return {"E": 0.0, "fed": 0.0, "blob": 0.0, "stray": 0.0, "peak": T_AMB, "off": 99.0, "cold": False, "phase": "none", "ref": None,
            "joined": False, "burnt": False, "ins": False, "bad": False, "still": 0.0, "heat_s": 0.0, "why": ""}


_trial = _fresh()


def reset_latch():
    """Forget the trial: the embodiment calls this at every reset. Nothing a robot does clears the record."""
    _trial.clear()
    _trial.update(_fresh())


def temp(tr=None):
    """The joint's temperature (C) and the liquid fraction of the solder in it, from its heat content."""
    tr = _trial if tr is None else tr
    e_m, lat = CAP * (T_MELT - T_AMB), J_LAT * tr["fed"]
    if tr["E"] <= e_m:
        return T_AMB + tr["E"] / CAP, 0.0
    if tr["E"] < e_m + lat:
        return T_MELT, (tr["E"] - e_m) / lat
    return T_MELT + (tr["E"] - e_m - lat) / CAP, 1.0


def _rot(q):
    x, y, z, w = q
    return np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                     (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))))


def _world(part, p):
    return np.asarray(part.pos, float) + _rot(part.quat_xyzw) @ np.asarray(p, float)


def holder(objects, name):
    """Which hand ('left_tcp' or 'right_tcp') holds a tool: its fingertip midpoint within HELD of the grip's centre."""
    p = np.asarray(objects[name].pos, float)
    best = min(("left_tcp", "right_tcp"), key=lambda a: np.linalg.norm(np.asarray(objects[a].pos, float) - p))
    return best if np.linalg.norm(np.asarray(objects[best].pos, float) - p) <= HELD else None


def _seg_seg(p0, p1, q0, q1, n=41):
    a = np.linspace(0.0, 1.0, n)[:, None]
    P, Q = p0 + a * (p1 - p0), q0 + a * (q1 - q0)
    d = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)
    i, j = np.unravel_index(int(np.argmin(d)), d.shape)
    return float(d[i, j]), P[i], Q[j]


def stick_at_joint(objects):
    """Is the solder stick's bare length at the joint: within ZONE[2] of the wire's axis between the insulation's end and
    3 mm inside the mouth? (Where on the parts the reported contact is; the contact itself comes from the simulator.)"""
    s = objects["solder"]
    d, _, _ = _seg_seg(_world(s, M.S_EXIT), _world(s, M.S_TIP), AXIS + (ZONE[0], 0, -M.WIRE_DROP), AXIS + (ZONE[1], 0, -M.WIRE_DROP))
    return d <= ZONE[2]


def tip_on_insulation(objects):
    """Is the iron's tip over the wire's insulation (nearest point of the wire's axis to the tip, in the wire's frame)?"""
    i, w = objects["iron"], objects["wire"]
    _, _, q = _seg_seg(_world(i, M.TIP_A), _world(i, M.TIP_B), _world(w, (M.SEAT - M.WIRE_LEN, 0, 0)), _world(w, (M.SEAT, 0, 0)), 81)
    return float((_rot(w.quat_xyzw).T @ (q - np.asarray(w.pos, float)))[0]) < -M.BARE


def wire_seat(objects):
    """How far the wire is out of its seat: (along the barrel, across it), m."""
    d = np.asarray(objects["wire"].pos, float) - WIRE0
    return float(max(0.0, -d[0])), float(math.hypot(d[1], d[2]))


def _box_dist(objects, part, lo, hi, hand):
    R = _rot(objects[part].quat_xyzw)
    d = R.T @ (np.asarray(objects[hand].pos, float) - np.asarray(objects[part].pos, float))
    return float(np.linalg.norm(d - np.clip(d, lo, hi)))


def let_go(objects, hand):
    """A hand 6 cm or more from both tools (grip, body and stick) and from the joint."""
    h = np.asarray(objects[hand].pos, float)
    if _box_dist(objects, "iron", *M.IRON_HULL, hand) < HANDS_CLEAR or _box_dist(objects, "solder", *M.SLEEVE_BOX, hand) < HANDS_CLEAR:
        return False
    s = objects["solder"]
    a, b = _world(s, M.S_EXIT), _world(s, M.S_TIP)
    t = min(max(float(np.dot(h - a, b - a) / np.dot(b - a, b - a)), 0.0), 1.0)
    return bool(np.linalg.norm(h - (a + t * (b - a))) >= HANDS_CLEAR and np.linalg.norm(h - AXIS) >= HANDS_CLEAR)


def hands_clear(objects):
    return let_go(objects, "left_tcp") and let_go(objects, "right_tcp")


def _rested(part, p0, heading):
    R = _rot(part.quat_xyzw)
    p = np.asarray(part.pos, float)
    a = math.atan2(R[1, 0], R[0, 0]) - heading
    return bool(math.hypot(p[0] - p0[0], p[1] - p0[1]) <= REST_XY and abs(p[2] - p0[2]) <= REST_Z and R[2, 2] >= math.cos(REST_TILT)
                and abs(math.atan2(math.sin(a), math.cos(a))) <= REST_TURN)


def tools_rested(objects):
    return _rested(objects["iron"], IRON0, M.IRON_REST["heading"]) and _rested(objects["solder"], SOLDER0, M.SOLDER_REST["heading"])


def note_step(objects, dt, touch):
    """Called by the embodiment after every control step. ``touch``: seconds in this step that each pair of parts touched
    (PhysX contact force over 0.02 N): iron_lug, iron_wire, iron_solder, solder_lug, solder_wire."""
    tr = _trial
    c = lambda k: min(max(float(touch.get(k, 0.0)), 0.0), dt)  # noqa: E731
    if any(objects[n].pos[2] < TOP - 0.01 for n in TOOLS + ("wire",)):
        tr["bad"] = True
    ih, sh = holder(objects, "iron"), holder(objects, "solder")
    T, liq = temp()
    # heat in through the tip on the lug, out to the wire, the jig and the air
    on = c("iron_lug") if ih is not None else 0.0
    tr["E"] = max(0.0, tr["E"] + on * min(P_MAX, G_TIP * (T_TIP - T)) - dt * G_LOSS * (T - T_AMB))
    tr["off"] = 0.0 if on > 0 else tr["off"] + dt
    tr["heat_s"] += on
    if ih is not None and c("iron_wire") > 0 and tip_on_insulation(objects):
        tr["ins"] = True
    # solder: onto the iron, or into the joint if the joint is hot enough to melt it
    at_joint = max(c("solder_lug"), c("solder_wire")) if stick_at_joint(objects) else 0.0
    stray = max(c("solder_lug"), c("solder_wire")) - at_joint
    two = sh is not None and sh != ih
    if c("iron_solder") > 0:
        tr["blob"] += RATE * c("iron_solder")
    elif two and T >= T_FLOW and at_joint > 0:
        d = RATE * at_joint
        tr["fed"] += d
        tr["E"] = max(0.0, tr["E"] - J_SENS * d)
        if tr["off"] > IRON_OFF:
            tr["cold"], tr["why"] = True, "solder fed with the iron off the lug"
    elif two and T >= T_FLOW and stray > 0:
        tr["stray"] += RATE * stray  # melted onto the outside of the lug, away from the mouth: not in the joint
    T, liq = temp()
    tr["peak"] = max(tr["peak"], T)
    if T > T_BURN:
        tr["burnt"] = True
    # the joint's state: liquid, freezing, set
    if tr["fed"] > 0:
        w = objects["wire"]
        if liq >= 1.0:
            tr["phase"], tr["ref"], tr["joined"] = "liquid", None, False
            if T >= T_REFLOW and on > 0:
                tr["cold"], tr["why"] = False, ""
        elif tr["phase"] in ("liquid", "freezing") and T >= T_SET:
            tr["phase"], tr["joined"] = "freezing", False
            if tr["ref"] is None:
                tr["ref"] = (np.asarray(w.pos, float).copy(), _rot(w.quat_xyzw))
            moved = float(np.linalg.norm(np.asarray(w.pos, float) - tr["ref"][0]))
            turned = math.acos(min(1.0, max(-1.0, (np.trace(tr["ref"][1].T @ _rot(w.quat_xyzw)) - 1) / 2)))
            if moved > MOVE or turned > TURN:
                tr["cold"], tr["why"] = True, "the wire moved while the joint froze"
            elif at_joint > 0 or c("iron_wire") > 0:
                tr["cold"], tr["why"] = True, "the joint was touched while it froze"
        elif tr["phase"] in ("liquid", "freezing") and T < T_SET:
            out = wire_seat(objects)
            seated = out[0] <= SEATED[0] and out[1] <= SEATED[1]
            tr["phase"] = "set"
            tr["joined"] = bool(tr["fed"] >= F_BOND and not tr["cold"] and seated)
            if not tr["joined"] and not tr["why"]:
                tr["why"] = "too little solder to hold" if tr["fed"] < F_BOND else "the wire was out of the barrel"
        elif tr["phase"] == "set" and liq > 0.0:
            tr["phase"], tr["joined"] = "freezing", False  # remelting: it must freeze undisturbed again
    still = all(np.linalg.norm(objects[n].lin_vel) <= LIN and np.linalg.norm(objects[n].ang_vel) <= ANG for n in TOOLS + ("wire",)) and hands_clear(objects)
    tr["still"] = tr["still"] + dt if still else 0.0


def joined():
    """Lug and wire are soldered together now (the embodiment keeps the fixed joint in step with this)."""
    return bool(_trial["joined"])


def defects():
    tr = _trial
    out = []
    if tr["fed"] < F_MIN:
        out.append("starved")
    if tr["fed"] > F_MAX:
        out.append("excess")
    if tr["burnt"] or tr["ins"]:
        out.append("overheated")
    return out


def shown():
    """The look prims to show now."""
    tr = _trial
    on = set()
    if tr["fed"] >= 2.0:
        if tr["phase"] == "set" and not tr["joined"]:
            on.add("fx_cold")
        else:
            on.add("fx_lean" if tr["fed"] < F_MIN else "fx_fat" if tr["fed"] > F_MAX else "fx_fill")
        if tr["phase"] == "liquid":
            on.add("fx_wet")
    if tr["burnt"]:
        on.add("fx_char")
    if tr["ins"]:
        on.add("fx_ins")
    if tr["blob"] >= 2.0:
        on.add("fx_blob")
    return on


def score(objects):
    """Read at the end of the run; there is no early end. 0 if a tool or the wire is off the table, or lug and wire are not
    soldered together (nothing flowed, a cold or disturbed joint, the wire out of the barrel, or a joint not yet set).
    Joined with two faults (solder amount; overheating) 0.25, with one 0.5. A sound joint 0.75, and 1.0 with the iron on its
    stand, the solder on its rest, both hands 6 cm clear and everything still for the last 1 s."""
    tr = _trial
    if tr["bad"] or not tr["joined"] or tr["phase"] != "set":
        return 0.0
    n = len(defects())
    if n:
        return 0.5 if n == 1 else 0.25
    if tools_rested(objects) and hands_clear(objects) and tr["still"] >= STILL_S:
        return 1.0
    return 0.75


def debug_line(objects, touch, t):
    tr = _trial
    T, liq = temp()
    r = lambda v: np.round(1000 * np.asarray(v, float), 1).tolist()  # noqa: E731
    i, s, w = objects["iron"], objects["solder"], objects["wire"]
    tilt = lambda o: math.degrees(math.acos(min(1.0, _rot(o.quat_xyzw)[2, 2])))  # noqa: E731
    tm = _world(i, M.TIP_MID) - AXIS
    st = _world(s, M.S_TIP) - AXIS
    tc = {k: round(v, 3) for k, v in touch.items()}
    return (f"SOL t={t:6.2f} T {T:5.1f} liq {liq:.2f} fed {tr['fed']:5.1f} blob {tr['blob']:4.1f} stray {tr['stray']:4.1f} phase {tr['phase']} cold {int(tr['cold'])} "
            f"joined {int(tr['joined'])} tipmid {r(tm)} tilt {tilt(i):4.1f} stick {r(st)} tilt {tilt(s):4.1f} zone {int(stick_at_joint(objects))} "
            f"wire {r(np.asarray(w.pos) - WIRE0)} hands {holder(objects, 'iron')},{holder(objects, 'solder')} L {r(objects['left_tcp'].pos)} R {r(objects['right_tcp'].pos)} "
            f"iron {r(i.pos)} solder {r(s.pos)} touch {tc} rested {int(tools_rested(objects))} clear {int(hands_clear(objects))} still {tr['still']:.2f} "
            f"why '{tr['why']}' score {score(objects):.2f}")


# -- scripted demo -------------------------------------------------------------------------------------------------------
# Open loop from the fixed starting layout, which a policy can see; the plan reads no simulator state.
# SOLDER_DEMO: shortcut variants for checking the scorer. "none": nothing moves. "hover": the tip is held 3 mm above the
# barrel. "tip": the solder is fed onto the iron's tip. "short": 2 s of feed. "ironfirst": the iron leaves before the
# solder. "yank": the wire is pulled out while the joint freezes. "untidy": the tools are put down on the table.
VARIANT = os.environ.get("SOLDER_DEMO", "")
FINGER_PER_M = 0.04695 / 0.094
SIDE = {"left": 1, "right": -1}
PARK = 0.12, 0.34, TOP + 0.18
SEGMENT = 0.03
GRIP_DZ, SLEEVE_DZ = 0.003, 0.0025  # TCP above each grip's centre when holding
IRON_AT = 0.0055  # the tip crosses the barrel this far inside the mouth
STICK_AT = -0.0050  # the stick crosses the bare conductor this far outside the mouth
STICK_PAST = 0.004  # and its tip ends this far past the wire's axis
# The arms' give, measured from logged runs (dev/): how far each tool sat from its target. The plan aims the other way.
# First logged run (on an earlier, 2.5 mm longer barrel), aimed with no offsets: the tip crossed the barrel 1.5 mm further
# in than aimed, and the stick lay 2.5-3.4 mm nearer the insulation than aimed; both bore down as aimed.
IRON_CAL = np.array([float(v) for v in os.environ.get("SOLDER_IRON_CAL", "0.0015,0,0").split(",")])
STICK_CAL = np.array([float(v) for v in os.environ.get("SOLDER_STICK_CAL", "-0.003,0,0").split(",")])
IRON_PRESS, STICK_PRESS = 0.002, 0.0015  # aimed this far into the part, so the hand bears on it lightly
HEAT_S, FEED_S, FLOW_S, SET_S = 10.5, 5.25, 1.0, 17.0


def _fingers(width):
    return -FINGER_PER_M * width


IRON_OPEN, IRON_GRIP = _fingers(M.GRIP["x"] + 0.03), _fingers(M.GRIP["x"] - 0.004)
SOLDER_OPEN, SOLDER_GRIP = _fingers(M.SLEEVE + 0.03), _fingers(M.SLEEVE - 0.004)


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def iron_tcp(tip_mid, heading):
    """The iron hand's TCP and yaw for the middle of the tip at ``tip_mid`` (world), the iron pointing along ``heading``."""
    return np.asarray(tip_mid, float) - _rz(heading) @ M.TIP_MID + (0.0, 0.0, GRIP_DZ), heading + math.pi / 2


def solder_tcp(tip, heading):
    return np.asarray(tip, float) - _rz(heading) @ M.S_TIP + (0.0, 0.0, SLEEVE_DZ), heading - math.pi / 2


class _Plan:
    def __init__(self):
        self.wps = {a: [] for a in SIDE}
        self.t = {a: 0.0 for a in SIDE}
        self.last = {}
        for a in SIDE:
            pos, _ = ik.tcp_pose(np.array(READY, float), np.array(BASES[a]))
            self.last[a] = (tuple(float(v) for v in pos), FINGER_OPEN, 0.0)

    def go(self, arm, dur, pos=None, finger=None, label="", yaw=None):
        p0, f0, y0 = self.last[arm]
        p = p0 if pos is None else tuple(float(v) for v in pos)
        f = f0 if finger is None else finger
        y = y0 if yaw is None else yaw
        n = 1 if label == "down" else max(1, math.ceil(np.linalg.norm(np.subtract(p, p0)) / SEGMENT), math.ceil(abs(y - y0) / math.radians(15)))
        for k in range(1, n + 1):
            q = tuple(float(v) for v in np.asarray(p0) + (np.asarray(p) - np.asarray(p0)) * k / n)
            self.wps[arm].append(Waypoint(dur / n, q, f0 + (f - f0) * k / n, label, y0 + (y - y0) * k / n))
        self.last[arm] = (p, f, y)
        self.t[arm] += dur

    def until(self, arm, t):
        if t > self.t[arm] + 1e-9:
            self.go(arm, t - self.t[arm], label="wait")

    def park(self, arm, dur=1.6):
        self.go(arm, dur, (PARK[0], SIDE[arm] * PARK[1], PARK[2]), None, "away")


def _pick(plan, arm, tcp, yaw, opened, grip, name):
    plan.go(arm, 1.6, tcp + (0, 0, 0.04), opened, f"{name}: above it", yaw)
    plan.go(arm, 0.3, label="settle")
    plan.go(arm, 0.8, tcp, label="descend")
    plan.go(arm, 0.6, finger=grip, label="grip")
    plan.go(arm, 0.8, tcp + (0, 0, 0.035), label="lift")


def _put(plan, arm, tcp, yaw, opened, name, dur=3.0):
    plan.go(arm, 0.8, np.array(plan.last[arm][0]) + (0, 0, 0.03), label=f"{name}: up")
    plan.go(arm, dur, tcp + (0, 0, 0.04), label=f"{name}: over its rest", yaw=yaw)
    plan.go(arm, 1.0, tcp + (0, 0, 0.001), label=f"{name}: down")
    plan.go(arm, 0.6, finger=opened, label=f"{name}: let go")
    plan.go(arm, 0.8, tcp + (0, 0, 0.05), label="up")
    plan.park(arm)


IRON_HEAD, STICK_HEAD = M.IRON_REST["heading"], M.SOLDER_REST["heading"]


def iron_target(lift=0.0):
    """Where the middle of the tip is aimed: across the barrel, its underside IRON_PRESS into the barrel's top."""
    return AXIS + (IRON_AT, 0.0, M.B_OD / 2 + M.TIP_D / 2 - IRON_PRESS + lift) - IRON_CAL


def stick_target(lift=0.0, fed=0.0):
    """Where the stick's tip is aimed: the stick across the bare conductor, its underside STICK_PRESS into the conductor's top.
    The stick's own length doesn't change in the simulation, so the hand holds still while it feeds."""
    top = -M.WIRE_DROP + M.COND_SQ / 2 + M.SOLDER_D / 2 / math.cos(M.SLOPE) - STICK_PRESS
    return AXIS + (STICK_AT, STICK_PAST, top - STICK_PAST * math.tan(M.SLOPE) + lift) - STICK_CAL


def demo():
    plan = _Plan()
    if VARIANT == "none":
        for a in SIDE:
            plan.go(a, 5.0, label="wait")
        return plan.wps
    for a in SIDE:
        p0 = plan.last[a][0]
        plan.go(a, 1.0, (p0[0], p0[1], TOP + 0.15), label="down")
    irest, iyaw = iron_tcp(IRON0 + _rz(IRON_HEAD) @ M.TIP_MID, IRON_HEAD)
    srest, syaw = solder_tcp(SOLDER0 + _rz(STICK_HEAD) @ M.S_TIP, STICK_HEAD)
    _pick(plan, "left", irest, iyaw, IRON_OPEN, IRON_GRIP, "iron")
    _pick(plan, "right", srest, syaw, SOLDER_OPEN, SOLDER_GRIP, "solder")
    hover = 0.003 + IRON_PRESS if VARIANT == "hover" else 0.0
    plan.go("left", 2.5, iron_tcp(iron_target(0.025), IRON_HEAD)[0], label="iron: over the lug")
    plan.go("left", 1.0, iron_tcp(iron_target(hover), IRON_HEAD)[0], label="iron: on the barrel")
    t_on = plan.t["left"]
    plan.go("right", 2.5, solder_tcp(stick_target(0.02), STICK_HEAD)[0], label="solder: over the joint")
    feed_s = 2.0 if VARIANT == "short" else FEED_S
    plan.until("right", t_on + HEAT_S - 0.8)
    if VARIANT == "tip":  # onto the iron's tip, past the barrel
        over = iron_target() + (0.0, -0.005, M.TIP_D / 2 + M.SOLDER_D / 2 - 0.001) + IRON_CAL - STICK_CAL
        plan.go("right", 0.8, solder_tcp(over, STICK_HEAD)[0], label="solder: onto the tip")
    else:
        plan.go("right", 0.8, solder_tcp(stick_target(), STICK_HEAD)[0], label="solder: into the joint")
    plan.go("right", feed_s, label="feed")
    if VARIANT == "ironfirst":
        plan.until("left", plan.t["right"] - 2.5)
        plan.go("left", 0.8, iron_tcp(iron_target(0.03), IRON_HEAD)[0], label="iron: off (too soon)")
    plan.go("right", 0.8, solder_tcp(stick_target(0.03), STICK_HEAD)[0], label="solder: out")
    t_out = plan.t["right"]
    if VARIANT != "ironfirst":
        plan.until("left", t_out + FLOW_S)
        plan.go("left", 0.8, iron_tcp(iron_target(0.03), IRON_HEAD)[0], label="iron: off")
    t_off = plan.t["left"]
    if VARIANT == "untidy":
        for a, dy in (("left", 0.10), ("right", -0.10)):
            p = np.array(plan.last[a][0]) + (-0.06, dy, 0.0)
            plan.go(a, 2.0, p, label="aside")
            plan.go(a, 1.0, (p[0], p[1], TOP + 0.045), label="down")
            plan.go(a, 0.6, finger=IRON_OPEN, label="let go")
            plan.go(a, 0.8, p + (0, 0, 0.05), label="up")
            plan.park(a)
    else:
        _put(plan, "left", irest, iyaw, IRON_OPEN, "iron")
        _put(plan, "right", srest, syaw, SOLDER_OPEN, "solder")
    if VARIANT == "yank":  # the right hand takes the wire by its insulation and pulls it out while the joint freezes
        g = WIRE0 + (-0.045, 0.0, 0.004)
        plan.go("right", 2.0, g + (0, 0, 0.05), _fingers(0.03), "wire: above it", 0.0)
        plan.go("right", 0.8, g, label="descend")
        plan.go("right", 0.6, finger=_fingers(M.INS_D - 0.003), label="grip")
        plan.go("right", 1.5, g + (-0.04, 0, 0.0), label="pull")
        plan.go("right", 0.6, finger=_fingers(0.03), label="let go")
        plan.go("right", 0.8, g + (-0.04, 0, 0.06), label="up")
        plan.park("right")
    end = max(max(plan.t.values()), t_off + SET_S) + 2.0
    for a in SIDE:
        plan.until(a, end)
    return plan.wps


def ik_errors(wps):
    worst_p = worst_r = 0.0
    bad, span = [], {}
    for arm, ws in wps.items():
        q = np.array(READY, float)
        lo, hi = q.copy(), q.copy()
        for w in ws:
            q, pe, re = ik.solve(np.asarray(w.pos, float), ik.matrix_to_quat_xyzw(grip_rotation(w.yaw)), q, base_pos=BASES[arm])
            lo, hi = np.minimum(lo, q), np.maximum(hi, q)
            worst_p, worst_r = max(worst_p, pe), max(worst_r, re)
            if pe > 1e-3 or re > math.radians(1.0):
                bad.append((arm, w.label[:24], tuple(round(v, 3) for v in w.pos), round(pe * 1000, 1), round(math.degrees(re), 2)))
        span[arm] = (lo, hi)
    return worst_p, worst_r, bad, span


if __name__ == "__main__":  # self-check, no simulator: the demo fits the run limit and IK reaches every waypoint inside the joint limits
    wps = demo()
    t = max(sum(w.duration for w in ws) for ws in wps.values())
    pe, re, bad, span = ik_errors(wps)
    print(f"demo {t:.1f} s ({t * 20:.0f} steps), IK {pe * 1000:.2f} mm / {math.degrees(re):.2f} deg, {len(bad)} bad", bad[:6])
    for arm, (lo, hi) in span.items():
        margin = np.minimum(lo - ik.LOWER, ik.UPPER - hi)
        print(arm, "joint range used (deg):", np.round(np.degrees(lo), 0).tolist(), np.round(np.degrees(hi), 0).tolist(), "least margin to a limit (deg):", np.round(np.degrees(margin), 0).tolist())
    assert t * 20 < 2000 and not bad
