"""Dev only: several episodes in one simulator session, to measure the mechanism and to try the shortcuts.
Created 11_16_26 11:30 EST.

    python dev/sim.py demo demo none loose one plugs shove can demo:hold_x=0.003     (same environment as an inspect-robots run)

Each word is one episode of the scripted plan, played open loop by the stock yam_scripted policy on the yam_guide_rail
embodiment: a task.set_variant name ("demo" is the full demo), optionally followed by ":key=value,..." which shifts an
aim for that episode (a task.CAL entry, or PRESS), to see how far the result moves when the arm is off by that much.
After each phase of the plan it prints the lane's clearance at both brackets, both knobs' tightness and where the hands
are against where they were aimed. The demo itself reads nothing from the simulator.
"""
import json
import os
import sys

os.environ["RAIL_DEV"] = "1"

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from guide_rails.embodiment import GuideRailEmbodiment  # noqa: E402
from inspect_robots import Scene  # noqa: E402
from yam_arena.scripted import YamScriptedPolicy  # noqa: E402

INSTR = json.load(open(os.path.join(HERE, "task.json")))["instruction"] if os.path.exists(os.path.join(HERE, "task.json")) else "set the lane"
MAX_STEPS = int(os.environ.get("RAIL_STEPS", "2400"))


def phases(m):
    """(arm, label, end time, aimed TCP) for each labelled stretch of the plan."""
    out = []
    for arm, ws in m.demo().items():
        t = 0.0
        for i, w in enumerate(ws):
            t += w.duration
            if i + 1 == len(ws) or ws[i + 1].label != w.label:
                out.append((t, arm, w.label, np.array(w.pos)))
    return sorted(out, key=lambda r: r[0])


def main():
    emb = GuideRailEmbodiment(task=os.path.join(HERE, "task.py"), object_jitter=0, control_hz=20, cam_width=320, cam_height=176)
    pol = YamScriptedPolicy()
    pol.bind(emb.info)
    m = emb.task.module
    dt = 1.0 / emb.info.control_hz
    base_cal, base_press = dict(m.CAL), m.PRESS
    for n, word in enumerate(sys.argv[1:]):
        name, _, opts = word.partition(":")
        m.CAL.update(base_cal)
        m.PRESS = base_press
        for kv in filter(None, opts.split(",")):
            k, v = kv.split("=")
            if k in ("press", "tug"):
                setattr(m, k.upper(), float(v))
            else:
                m.CAL[k] = base_cal[k] + float(v)
        m.set_variant("" if name == "demo" else name)
        m.TRACE = []
        scene = Scene(id=f"ep{n}", instruction=INSTR)
        print(f"EPISODE {n} {word} start", flush=True)
        obs = emb.reset(scene, seed=0)
        pol.reset(scene)
        info, steps, done = {}, 0, "limit"
        for steps in range(1, MAX_STEPS + 1):
            action = pol.act(obs).actions[0]
            res = emb.step(action)
            obs, info = res.observation, res.info
            if res.terminated:
                done = res.termination_reason or "terminated"
                break
            if action.meta.get("request_stop"):
                done = "plan ended"
                break
        tr = m.TRACE
        for t, arm, label, aim in phases(m):
            k = min(len(tr) - 1, int(round(t / dt)) - 1)
            if k < 0 or label in ("wait", "settle"):
                continue
            r = tr[k]
            off = 1000 * (r["tcp"][arm] - aim)
            j6 = ""
            if label.startswith("knob:") and r["wrist"]:
                k0 = max(0, k - int(round(2.9 / dt)))
                peak = max((abs(q["wrist"][arm][0]) for q in tr[k0:k + 1]), default=0.0)
                j6 = f"  wrist-roll torque now {r['wrist'][arm][0]:+.2f}, peak in the last 2.9 s {peak:.2f} N m"
            print(f"  {t:5.1f}s {arm:5s} {label:28s} roll {r['roll']:+5.2f} skew {r['skew']:+5.2f} clear L {1000 * r['c']['left']:+6.2f} R {1000 * r['c']['right']:+6.2f} mm  tight L {r['a']['left']:+6.1f} R {r['a']['right']:+6.1f} deg"
                  f"  tcp-aim ({off[0]:+5.1f},{off[1]:+5.1f},{off[2]:+5.1f}) mm  score {r['score']:.2f}{j6}", flush=True)
        lim = {k: (round(float(a.Get()), 3) if not k.startswith("lock") else bool(a.GetJointEnabledAttr().Get())) for k, a in emb._limits.items()}
        r = tr[-1]
        print(f"  joint friction limits read back: {lim}", flush=True)
        print(f"EPISODE {n} {word} end: steps {steps} ({done}) score {info.get('score')} success {info.get('success')} | clear L {1000 * r['c']['left']:+.2f} R {1000 * r['c']['right']:+.2f} mm"
              f" tight L {r['a']['left']:+.1f} R {r['a']['right']:+.1f} | plug L {np.round(1000 * r['plug']['left'], 1).tolist()} R {np.round(1000 * r['plug']['right'], 1).tolist()}"
              f" can {np.round(1000 * r['can'], 1).tolist()} plug yaw {r['plug_yaw']} | cmd: CAL {m.CAL} PRESS {m.PRESS}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
