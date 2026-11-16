"""Dev only, no simulator: step the scripted demo at 20 Hz and report the closest approach of each gripper's collision hulls
to every collider of the sprinkler, with the head and collars where the plan expects them (a pushed part follows the
finger once the finger reaches it). Created 11_16_26.

    python dev/clearance.py            (SPRINKLER_DEMO selects a variant, as in the simulator)
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import task as T  # noqa: E402
from sprinkler_arc import mech as M  # noqa: E402
from yam_arena import analysis  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402
from yam_arena.station import BASES  # noqa: E402

LINKS = ["link4", "link5", "gripper", "tip_left", "tip_right"]


def _rz(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def _hull(verts):
    """Convex hull of a prism's vertices (box or n-gon): planes from its faces."""
    from scipy.spatial import ConvexHull

    h = ConvexHull(verts)
    planes = np.c_[h.equations[:, :3], -h.equations[:, 3]]
    return analysis.Hull(vertices=verts, planes=planes)


def box(lo, hi, R=np.eye(3), t=np.zeros(3)):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    v = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return _hull(v @ R.T + t)


def cyl(c, r, h, R=np.eye(3), t=np.zeros(3), n=12, axis=None):
    ring = np.array([[r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n)] for k in range(n)])
    v = np.r_[np.c_[ring, np.full(n, -h / 2)], np.c_[ring, np.full(n, h / 2)]]
    if axis is not None:
        v = v @ axis.T
    v = v + c
    return _hull(v @ R.T + t)


def parts(head_deg, lower_deg, upper_deg):
    o = np.array((M.AXIS_X, M.AXIS_Y, T.HEAD_Z))
    out = {}
    Rh = _rz(head_deg)
    out["column"] = cyl((0, 0, sum(M.COLUMN["z"]) / 2), M.COLUMN["r"], M.COLUMN["z"][1] - M.COLUMN["z"][0], Rh, o)
    for name, d in (("lever", M.LEVER), ("bridge", M.BRIDGE), ("arm", M.DRIVE_ARM), ("spoon", M.SPOON), ("weight", M.WEIGHT)):
        out[name] = box((d["x"][0], -d["w"] / 2, d["z"][0]), (d["x"][1], d["w"] / 2, d["z"][1]), Rh, o)
    n = M.NOZZLE
    tilt = 90.0 - math.degrees(n["tilt"])
    a = math.radians(tilt)
    Ry = np.array(((math.cos(a), 0, math.sin(a)), (0, 1, 0), (-math.sin(a), 0, math.cos(a))))
    mid = np.array(n["root"]) + 0.5 * n["length"] * np.array((math.cos(n["tilt"]), 0.0, math.sin(n["tilt"])))
    out["nozzle"] = cyl(mid, n["r"], n["length"], Rh, o, axis=Ry)
    out["hub"] = cyl((0, 0, sum(M.HUB["z"]) / 2), M.HUB["r"], M.HUB["z"][1] - M.HUB["z"][0], Rh, o)
    for which, deg in (("lower", lower_deg), ("upper", upper_deg)):
        z0, z1 = M.COLLAR_Z[which]
        out[f"tab_{which}"] = box((M.TAB["r"][0], -M.TAB["w"] / 2, z0), (M.TAB["r"][1], M.TAB["w"] / 2, z1), _rz(deg), o)
    ob = np.array((M.AXIS_X, M.AXIS_Y, T.TOP))
    out["nut"] = cyl((0, 0, M.Z0 + sum(M.NUT["z"]) / 2), M.NUT["r"], M.NUT["z"][1] - M.NUT["z"][0], np.eye(3), ob)
    out["sleeve"] = cyl((0, 0, M.Z0 + sum(M.SLEEVE["z"]) / 2), M.SLEEVE["r"], M.SLEEVE["z"][1] - M.SLEEVE["z"][0], np.eye(3), ob)
    (x0, x1), (y0, y1) = M.TURF["x"], M.TURF["y"]
    out["turf"] = box((x0, y0, 0), (x1, y1, M.TURF["h"]), np.eye(3), ob)
    for k, deg in enumerate(M.TARGET):
        fx, fy = M.FLAG_R * math.cos(math.radians(deg)), M.FLAG_R * math.sin(math.radians(deg))
        out[f"flag{k}"] = box((fx - 0.004, fy - 0.004, 0), (fx + 0.004, fy + 0.004, M.TURF["h"] + M.FLAG_POST["h"]), np.eye(3), ob)
    return out


def expected(plan, t):
    """The head's and collars' headings at time t as the plan expects them."""
    ang = {"head": M.START["head"], "lower": M.START["lower"], "upper": M.START["upper"]}
    for what, _arm, track in plan.pushes:
        d = 1 if track[-1][1] > track[0][1] else -1
        reached = None
        for tt, deg in track:
            if tt <= t:
                reached = deg
        if reached is None:
            continue
        if T.wrap(reached - ang[what]) * d > 0:
            ang[what] = reached
    return ang


def main():
    wps = T.demo()
    plan = T.LAST_PLAN
    arms = {a: ArmPlan(list(w), BASES[a]) for a, w in wps.items()}
    dur = max(p.duration for p in arms.values())
    worst = {}
    for k in range(int(dur / 0.05) + 1):
        t = k * 0.05
        ang = expected(plan, t)
        prts = parts(ang["head"], ang["lower"], ang["upper"])
        pushing = {}
        for what, arm, track in plan.pushes:
            if track[0][0] - 0.2 <= t <= track[-1][0] + 0.3:
                pushing[arm] = ("tab_" + what,) if what != "head" else ("spoon", "arm", "weight")
        both = {}
        for arm, p in arms.items():
            a7, label = p.action7(t)
            q = {f"joint{i + 1}": float(a7[i]) for i in range(6)}
            f = -0.04695 * float(a7[6])
            q["joint7"] = q["joint8"] = f
            hulls = analysis.world_hulls(analysis.fk_link_poses_world(q, BASES[arm]), LINKS)
            both[arm] = (hulls, label)
            for name, h in prts.items():
                dist, pair = analysis.hull_set_distance(hulls, [(name, h)])
                if name in pushing.get(arm, ()):
                    continue
                key = (arm, name)
                if key not in worst or dist < worst[key][0]:
                    worst[key] = (dist, round(t, 2), label, pair[0] if pair else None, {k2: round(v, 1) for k2, v in ang.items()})
        if len(both) == 2:  # the two arms against each other
            dist, pair = analysis.hull_set_distance(both["left"][0], both["right"][0])
            if ("both", "arms") not in worst or dist < worst[("both", "arms")][0]:
                worst[("both", "arms")] = (dist, round(t, 2), both["left"][1] + " / " + both["right"][1], pair, {})
    for key, (dist, t, label, link, ang) in sorted(worst.items(), key=lambda kv: kv[1][0]):
        if dist < 0.02:
            print(f"{key[0]:5s} {key[1]:10s} {dist * 1000:7.1f} mm at t={t:5.2f} ({label}; {link}) {ang}")


if __name__ == "__main__":
    main()
