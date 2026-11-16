"""Dev only, no simulator: play the demo's plan with perfect tracking and report how close each arm's collision hulls
come to the conveyor, the brackets, the knobs, the rail, the plugs, the can and the other arm. Created 11_16_26 12:00 EST.

    python dev/clearance.py [variant] [step_s]

The rail is taken at its start until the holding hand has pressed it in, the plugs where they start. A hand's own
fingers are left out against the part it is working on (they are meant to touch it)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import task as t  # noqa: E402
from guide_rails import mech as M  # noqa: E402
from yam_arena import analysis as A  # noqa: E402
from yam_arena.constants import ARM_JOINTS  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402
from yam_arena.station import BASES  # noqa: E402

t.set_variant(sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "demo" else "")
STEP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
Z = t.TOP
FINGERS = ("tip_left", "tip_right")


def box(lo, hi):
    lo, hi = np.array(lo, float) + (0, 0, Z), np.array(hi, float) + (0, 0, Z)
    return (lo + hi) / 2, hi - lo


def around(c, half):
    c = np.asarray(c, float)
    return box(c - half, c + half)


x0, x1 = M.XC - M.CHAIN_W / 2, M.XC + M.CHAIN_W / 2
r0, r1 = M.RAIL_Z - M.RAIL_H / 2, M.RAIL_Z + M.RAIL_H / 2
FIXED = {
    "table": box((-0.3, -0.7, -0.05), (0.9, 0.7, 0.0)),
    "conveyor": box((x0 - M.FRAME_T, -M.SEC_LEN / 2, 0.0), (x1 + M.FRAME_T, M.SEC_LEN / 2, M.CHAIN_TOP)),
    "datum rail": box((M.X_DATUM, -M.RAIL_LEN / 2, r0), (M.X_DATUM + M.RAIL_T, M.RAIL_LEN / 2, r1)),
    "can": around((M.CAN_START[0], M.CAN_START[1], M.CAN_START[2]), np.array((M.CAN_D / 2 * 0.71, M.CAN_D / 2 * 0.71, M.CAN_H / 2))),
    "can (across)": around((M.CAN_START[0], M.CAN_START[1], M.CAN_START[2]), np.array((M.CAN_D / 2, 0.012, M.CAN_H / 2))),
    "can (along)": around((M.CAN_START[0], M.CAN_START[1], M.CAN_START[2]), np.array((0.012, M.CAN_D / 2, M.CAN_H / 2))),
}
for s, y in M.STATIONS.items():
    FIXED[f"block {s}"] = around((M.XB, y, M.RAIL_Z), np.array(M.BLOCK) / 2)
    FIXED[f"rod {s} (start)"] = box((M.X_FACE_START - M.RAIL_T - M.ROD_LEN, y - M.ROD_D / 2, M.RAIL_Z - M.ROD_D / 2), (M.X_FACE_START - M.RAIL_T, y + M.ROD_D / 2, M.RAIL_Z + M.ROD_D / 2))
KNOB = {s: around((M.XB, y, M.KNOB_Z), np.array((M.KNOB_D / 2, M.KNOB_D / 2, M.KNOB_H / 2))) for s, y in M.STATIONS.items()}
PLUG = {s: around(M.PLUG_START[s], np.array((M.PLUG_D / 2 * 0.71, M.PLUG_D / 2 * 0.71, M.PLUG_H / 2))) for s in M.STATIONS}


def rail_box(x_face):
    return box((x_face - M.RAIL_T, -M.RAIL_LEN / 2, r0), (x_face, M.RAIL_LEN / 2, r1))


def main():
    wps = t.demo()
    plans = {a: ArmPlan(wps[a], BASES[a]) for a in wps}
    T = max(p.duration for p in plans.values())
    worst = {}

    def note(key, d, who, tt, label):
        if key not in worst or d < worst[key][0]:
            worst[key] = (d, who, tt, label)

    pressed, gone = False, set()
    for k in range(int(round(T / STEP)) + 1):
        tt = k * STEP
        hulls, labels = {}, {}
        for a, pl in plans.items():
            _, _, finger, _ = pl.target(tt)
            q, label = pl.action7(tt)
            joints = {n: float(v) for n, v in zip(ARM_JOINTS, q[:6])}
            joints.update(joint7=finger, joint8=finger)
            hulls[a] = A.world_hulls(A.fk_link_poses_world(joints, BASES[a]))
            labels[a] = label
            pressed = pressed or label == "rail: hold"
            if label == "plug: lift":
                gone.add(a)
        for a in plans:
            lab = labels[a]
            body = [h for h in hulls[a] if h[0] not in FINGERS]
            for name, (c, size) in FIXED.items():
                if name.startswith("rod") and pressed:
                    continue
                d, who = A.hulls_to_box(hulls[a], c, np.eye(3), size)
                note(f"{a} arm - {name}", d, who, tt, lab)
            for s, (c, size) in KNOB.items():
                d, who = A.hulls_to_box(body if lab.startswith("knob") and s == a else hulls[a], c, np.eye(3), size)
                note(f"{a} arm{' (not its fingers)' if s == a else ''} - knob {s}", d, who, tt, lab)
            for s, (c, size) in PLUG.items():
                if s in gone:
                    continue
                d, who = A.hulls_to_box(body if lab.startswith("plug") and s == a else hulls[a], c, np.eye(3), size)
                note(f"{a} arm{' (not its fingers)' if s == a else ''} - plug {s}", d, who, tt, lab)
            c, size = rail_box(t.x_face_target() if pressed else M.X_FACE_START)
            d, who = A.hulls_to_box(body if lab.startswith("rail") else hulls[a], c, np.eye(3), size)
            note(f"{a} arm (not its fingers when on it) - rail", d, who, tt, lab)
        d, pair = A.hull_set_distance(hulls["left"], hulls["right"])
        if pair:
            note("left arm - right arm", d, pair, tt, labels["left"] + " / " + labels["right"])
    for key in sorted(worst, key=lambda k: worst[k][0]):
        d, who, tt, label = worst[key]
        if d < 0.015:
            print(f"{1000 * d:7.1f} mm  {key}  [{who}] at t={tt:.1f} ({label})")


main()
