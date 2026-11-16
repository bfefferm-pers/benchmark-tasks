"""The score and the clamp's bookkeeping. Created 11_16_26 10:40 EST. NumPy only: no simulator and no robot model, so the
tests run in any Python.

Every function takes ``objects``: name -> a state with ``pos`` (the part's own origin, world, m), ``quat_xyzw``,
``lin_vel`` and ``ang_vel``. Parts: ``rail``, ``knob_left``, ``knob_right``, ``plug_left``, ``plug_right``, ``can``; and
``left_tcp``, ``right_tcp``, each gripper's fingertip midpoint. ``top`` is the table top's height (world z).

The embodiment calls ``note_step`` after every control step. It follows each knob's turn, applies the clamp law of
``mech`` (the knob's turning torque and the rod's holding force, both set by the knob's tightness) and keeps the clock
for everything at rest. ``take_commands`` hands the embodiment the friction limits to set on the real joints.
"""

import math

import numpy as np

from . import mech as M

SIDES = ("left", "right")
PARTS = ("rail", "knob_left", "knob_right", "plug_left", "plug_right", "can")
CLEAR_LO, CLEAR_HI = 0.0005, 0.003  # the lane at a bracket: the can's 66.0 mm plus this much
HOLD_TEST = 10.0  # N: a locked clamp holds this push on the rail at its bracket
HANDS_CLEAR = 0.06  # m from a part's surface to a gripper's fingertip midpoint: open fingers sit 4.7 cm out
LIN, ANG, STILL_S = 0.01, 0.5, 1.0  # at rest: under 1 cm/s and 0.5 rad/s, for the last second
REST_XY, REST_Z, UPRIGHT = 0.015, 0.003, math.radians(5.0)  # a plug on its rest; a part standing upright
CAN_Z = 0.003  # the can stands on the chain
RUN_Y = 0.030  # the can has run through: its centre this far past the left bracket (its trailing edge past the bracket)
OFF_TABLE = 0.01  # a loose part this far below the table top has left the table

_trial = {}


def reset():
    """Forget the trial: the embodiment calls this at every reset. Nothing a robot does clears it."""
    _trial.clear()
    _trial.update(turn={s: 0.0 for s in SIDES}, yaw={s: None for s in SIDES}, sent={}, commands=[], still=0.0, bad=False,
                  locked={s: True for s in SIDES}, early=False)


reset()


def rot(q):
    x, y, z, w = (float(v) for v in q)
    return np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                     (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))))


def tightness(side):
    """Degrees the knob stands clockwise past its seat (negative: backed off)."""
    return M.A_START - math.degrees(_trial["turn"][side])


def faces(objects):
    """The adjustable rail's product face at each bracket: world (x, y, z)."""
    r = objects["rail"]
    R, p = rot(r.quat_xyzw), np.asarray(r.pos, float)
    return {s: p + R @ np.array((M.RAIL_T / 2, y, 0.0)) for s, y in M.STATIONS.items()}


def clearances(objects):
    """At each bracket: the lane's width less the can's diameter (m). The datum rail's face is fixed at X_DATUM."""
    return {s: float(M.X_DATUM - f[0] - M.CAN_D) for s, f in faces(objects).items()}


def in_band(c):
    return CLEAR_LO - 1e-9 <= c <= CLEAR_HI + 1e-9


def holds(side):
    """Does that bracket's clamp hold HOLD_TEST newtons on the rod?"""
    return M.hold_force(tightness(side)) >= HOLD_TEST


def _tilt(q):
    return math.acos(max(-1.0, min(1.0, rot(q)[2, 2])))


def _speed_ok(o):
    return np.linalg.norm(o.lin_vel) <= LIN and np.linalg.norm(o.ang_vel) <= ANG


def _box_dist(o, half, p):
    d = rot(o.quat_xyzw).T @ (np.asarray(p, float) - np.asarray(o.pos, float))
    return float(np.linalg.norm(d - np.clip(d, -np.asarray(half), np.asarray(half))))


def hand_distance(objects, part):
    """The nearer gripper's fingertip midpoint to the part's surface (m)."""
    if part == "rail":
        half = (M.RAIL_T / 2, M.RAIL_LEN / 2, M.RAIL_H / 2)
    elif part.startswith("knob"):
        half = (M.KNOB_D / 2, M.KNOB_D / 2, M.KNOB_H / 2)
    elif part.startswith("plug"):
        half = (M.PLUG_D / 2, M.PLUG_D / 2, M.PLUG_H / 2)
    else:
        half = (M.CAN_D / 2, M.CAN_D / 2, M.CAN_H / 2)
    return min(_box_dist(objects[part], half, objects[h].pos) for h in ("left_tcp", "right_tcp"))


def clear_of(objects, part):
    return hand_distance(objects, part) >= HANDS_CLEAR


def rail_let_go(objects):
    """The rail stands by its clamps alone: both grippers clear of it, and it is not moving."""
    return clear_of(objects, "rail") and _speed_ok(objects["rail"])


def plug_rested(objects, side, top):
    """A plug standing on a rest (either rest): centre within REST_XY of the rest's centre, upright, down on it, still."""
    p = objects[f"plug_{side}"]
    pos = np.asarray(p.pos, float)
    on = any(math.hypot(pos[0] - rx, pos[1] - ry) <= REST_XY for rx, ry in M.REST.values())
    down = abs(pos[2] - (top + M.REST_T + M.PLUG_H / 2)) <= REST_Z
    return bool(on and down and _tilt(p.quat_xyzw) <= UPRIGHT and _speed_ok(p))


def plugs_rested(objects, top):
    if not all(plug_rested(objects, s, top) for s in SIDES):
        return False
    a, b = (np.asarray(objects[f"plug_{s}"].pos, float) for s in SIDES)
    return bool(np.linalg.norm(a[:2] - b[:2]) > M.PLUG_D)  # one on each rest


def can_on_chain(objects, top):
    """The sample can upright on the chain (anywhere along the section)."""
    c = objects["can"]
    pos = np.asarray(c.pos, float)
    return bool(_tilt(c.quat_xyzw) <= UPRIGHT and abs(pos[2] - (top + M.CHAIN_TOP + M.CAN_H / 2)) <= CAN_Z
                and abs(pos[0] - M.XC) <= M.CHAIN_W / 2 and abs(pos[1]) <= M.SEC_LEN / 2)


def can_run_through(objects, top):
    """The sample can has been run through the lane: upright on the chain, its centre at least RUN_Y past the left
    bracket, and it entered the set stretch (past the right bracket) only once both clamps held."""
    y = float(objects["can"].pos[1])
    return can_on_chain(objects, top) and y >= M.STATIONS["left"] + RUN_Y and not _trial["early"]


def note_step(objects, dt, top):
    tr = _trial
    for s in SIDES:
        R = rot(objects[f"knob_{s}"].quat_xyzw)
        yaw = math.atan2(R[1, 0], R[0, 0])
        if tr["yaw"][s] is not None:
            d = yaw - tr["yaw"][s]
            tr["turn"][s] += math.atan2(math.sin(d), math.cos(d))  # counter-clockwise turn since the start (rad)
        tr["yaw"][s] = yaw
        a = tightness(s)
        bites = M.hold_force(a) >= M.LOCK_N
        if bites != tr["locked"][s]:  # the clamp bites, or lets go: the rod is locked where it stands, or freed
            tr["locked"][s] = bites
            r = objects["rail"]
            at = np.asarray(r.pos, float) + rot(r.quat_xyzw) @ np.array((0.0, M.STATIONS[s], 0.0))
            tr["commands"].append((f"lock_{s}", float(at[0]) if bites else None))
        for name, value, step in ((f"knob_{s}", M.knob_torque(a), 0.005), (f"slide_{s}", M.hold_force(a), 0.25)):
            q = round(value / step) * step
            if tr["sent"].get(name) != q:
                tr["sent"][name] = q
                tr["commands"].append((name, q))
    if objects["can"].pos[1] > M.STATIONS["right"] and not all(holds(s) for s in SIDES):
        tr["early"] = True  # the can got into the set stretch before both clamps held
    if any(objects[n].pos[2] < top - OFF_TABLE for n in PARTS):
        tr["bad"] = True
    still = all(_speed_ok(objects[n]) for n in PARTS) and all(clear_of(objects, n) for n in PARTS if n != "can")
    tr["still"] = tr["still"] + dt if still else 0.0


def take_commands():
    """What to set on the joints since the last call: (joint, value). A knob's friction limit in N m; a slide's in N;
    for ``lock_left`` / ``lock_right`` the x at which to lock that rod, or None to free it."""
    out = _trial["commands"]
    _trial["commands"] = []
    return out


def score(objects, top):
    """Read at the end of the run.
    0: a loose part off the table; or the lane at neither bracket is the can plus 0.5 to 3.0 mm.
    0.25: the lane is in that band at one bracket only; or at both, but a gripper is still on the rail, the rail is
          moving, or neither clamp holds 10 N.
    0.5: in the band at both brackets, the rail let go and still, one clamp holding 10 N.
    0.75: the same with both clamps holding.
    1.0: and both setting plugs standing on their rests, the sample can run through (upright on the chain, its centre
         30 mm or more past the left bracket, having passed the right bracket only after both clamps held), both
         grippers 6 cm clear of the rail, the knobs and the plugs, everything at rest for the last 1 s."""
    tr = _trial
    if tr["bad"]:
        return 0.0
    c = clearances(objects)
    n = sum(in_band(v) for v in c.values())
    if n == 0:
        return 0.0
    if n < 2 or not rail_let_go(objects):
        return 0.25
    k = sum(holds(s) for s in SIDES)
    if k == 0:
        return 0.25
    if k == 1:
        return 0.5
    if plugs_rested(objects, top) and can_run_through(objects, top) and tr["still"] >= STILL_S:
        return 1.0
    return 0.75


def debug_line(objects, top):
    c = clearances(objects)
    mm = lambda v: np.round(1000 * np.asarray(v, float), 1).tolist()  # noqa: E731
    R = rot(objects["rail"].quat_xyzw)
    return (f"roll {math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0])))):+.2f} clear L {1000 * c['left']:+6.2f} R {1000 * c['right']:+6.2f} mm | tight L {tightness('left'):+7.1f} R {tightness('right'):+7.1f} deg "
            f"hold L {M.hold_force(tightness('left')):5.1f} R {M.hold_force(tightness('right')):5.1f} N | tcp L {mm(objects['left_tcp'].pos)} R {mm(objects['right_tcp'].pos)} "
            f"| plug L {mm(objects['plug_left'].pos)} R {mm(objects['plug_right'].pos)} can {mm(objects['can'].pos)} | rested {int(plugs_rested(objects, top))} "
            f"can {int(can_run_through(objects, top))} let_go {int(rail_let_go(objects))} still {_trial['still']:.2f} score {score(objects, top):.2f}")
