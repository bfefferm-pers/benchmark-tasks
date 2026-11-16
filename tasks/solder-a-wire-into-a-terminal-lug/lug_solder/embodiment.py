"""`yam_solder`: the stock yam_arena embodiment plus the soldering bench (lug_solder.mech). Created 11_15_26 (local time).

What it adds to the stock embodiment, and nothing else:
- spawns the jig, lug, wire, iron and solder stick;
- gives the scorer `left_tcp` and `right_tcp` (each gripper's fingertip midpoint) and the wire, iron and solder poses;
- reads PhysX's contact reports after every physics step (240 Hz): the contact force between the iron and the lug, the
  wire and the solder stick, and between the solder stick and the lug and the wire. A pair counts as touching for a
  physics step when that force is over 0.02 N. The seconds of touching per control step go to the task's note_step;
- switches the fixed joint between lug and wire on when the task's record says a joint has set, off when it is molten;
- shows the joint's look (fillet, cold, charred) from the task's record.
"""

import math
import os

import numpy as np
from yam_arena.inspect_embodiment import YamArenaEmbodiment

DEBUG = os.environ.get("SOLDER_DEBUG") == "1"
BENCH, FIXED, LOOSE = "bench", ("bench", "lug"), ("wire", "iron", "solder")
TOUCH_N = 0.02
PAIRS = {"iron": ("lug", "wire", "solder"), "solder": ("lug", "wire")}
_MON = {"views": None, "paths": {}, "acc": {}, "err": None, "on": False, "hooked": False}


def _np(v):
    if hasattr(v, "detach"):
        return v.detach().cpu().numpy()
    return np.asarray(v.numpy() if hasattr(v, "numpy") else v)


def _poll(mgr):
    """After a physics step: add this step's touching time for every pair."""
    if not _MON["on"]:
        return
    try:
        dt = float(mgr.get_physics_dt())
        if _MON["views"] is None:
            sv = mgr.get_physics_sim_view()
            p = _MON["paths"]
            if sv is None or len(p) < 4:
                return
            _MON["views"] = {s: sv.create_rigid_contact_view([p[s]], filter_patterns=[[p[f] for f in fs]], max_contact_data_count=256)
                             for s, fs in PAIRS.items()}
        for s, fs in PAIRS.items():
            f = np.asarray(_np(_MON["views"][s].get_contact_force_matrix(dt=dt)), float).reshape(-1, len(fs), 3)[0]
            for name, n in zip(fs, np.linalg.norm(f, axis=1)):
                key = f"{s}_{name}"
                _MON["acc"][key] = _MON["acc"].get(key, 0.0) + (dt if n > TOUCH_N else 0.0)
                _MON["acc"]["F_" + key] = max(_MON["acc"].get("F_" + key, 0.0), float(n))
    except Exception as exc:  # reported once by the embodiment; a silent monitor would score every trial 0
        _MON["err"] = repr(exc)


def _hook():
    import carb
    from isaaclab_physx.physics.physx_manager import PhysxManager

    if _MON["hooked"]:
        return
    base_cfg, base_step = PhysxManager._configure_physics.__func__, PhysxManager.step.__func__

    def _configure_physics(cls):
        base_cfg(cls)
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)

    def step(cls):
        base_step(cls)
        _poll(cls)

    PhysxManager._configure_physics = classmethod(_configure_physics)
    PhysxManager.step = classmethod(step)
    _MON["hooked"] = True


def _place(stage, prim_path, translation, orientation):
    from pxr import Gf, UsdGeom

    x = UsdGeom.Xform.Define(stage, prim_path)
    x.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    qx, qy, qz, qw = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)
    x.AddOrientOp().Set(Gf.Quatf(float(qw), Gf.Vec3f(float(qx), float(qy), float(qz))))
    return x.GetPrim()


class SolderEmbodiment(YamArenaEmbodiment):
    _fx = None
    _joint = None
    _joined = None

    def reset(self, scene, *, seed=None):
        self.task.module.reset_latch()
        _MON["on"] = False
        if self._joint is not None:
            self._set_joint(False)
        out = super().reset(scene, seed=seed)
        _MON["acc"] = {}
        _MON["on"] = True
        self._show(set())
        return out

    def _ensure_sim(self, goal):
        import atexit

        from yam_arena import inspect_embodiment as base

        if base._APP is None:  # Kit must be up before Arena is imported
            from isaaclab.app import AppLauncher

            os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
            base._APP = AppLauncher(headless=self.headless, device=self.device, enable_cameras=True).app
            atexit.register(base._close_app_at_exit)
        _hook()
        from yam_arena import scene

        original = scene.task_object

        def spawn(prim_path, cfg, translation=None, orientation=None, **kwargs):
            import omni.usd

            from . import mech

            stage = omni.usd.get_context().get_stage()
            _place(stage, prim_path, translation, orientation)
            name = prim_path.rstrip("/").rsplit("/", 1)[-1]
            if name == "wire":
                return mech.build_wire(stage, prim_path, prim_path.rsplit("/", 1)[0] + "/lug")
            return getattr(mech, f"build_{name}")(stage, prim_path)

        def make_object(spec):
            import isaaclab.sim as sim_utils
            from isaaclab_arena.assets.object import Object
            from isaaclab_arena.assets.object_type import ObjectType
            from isaaclab_arena.utils.pose import Pose

            if spec.name not in FIXED + LOOSE:
                return original(spec)
            cfg = sim_utils.CuboidCfg(size=(0.01, 0.01, 0.01))
            cfg.func = spawn
            kind = ObjectType.BASE if spec.name in FIXED else ObjectType.RIGID
            return Object(name=spec.name, object_type=kind, spawner_cfg=cfg,
                          initial_pose=Pose(position_xyz=tuple(float(v) for v in spec.pos),
                                            rotation_xyzw=(0.0, 0.0, math.sin(spec.yaw / 2), math.cos(spec.yaw / 2))))

        scene.task_object = make_object
        try:
            out = super()._ensure_sim(goal)
        finally:
            scene.task_object = original
        self._find()
        return out

    def _find(self):
        """The bodies' prim paths (for the contact views), the joint and the look prims."""
        import omni.usd
        from pxr import UsdGeom, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        self._fx, paths = {}, {}
        for prim in stage.Traverse():
            n, p = prim.GetName(), str(prim.GetPath())
            if "/env_0/" not in p:
                continue
            if n in ("lug", "wire", "iron", "solder") and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                paths[n] = p
            elif n == "soldered":
                self._joint = UsdPhysics.Joint(prim).GetJointEnabledAttr()
            elif n.startswith("fx_"):
                self._fx[n] = [UsdGeom.Imageable(prim), None]
        if len(paths) != 4 or self._joint is None:
            raise RuntimeError(f"yam_solder: bodies {sorted(paths)} joint {self._joint}")
        if _MON["paths"] != paths:
            _MON.update(paths=paths, views=None)

    def _set_joint(self, on):
        if self._joined != on:
            self._joint.Set(bool(on))
            self._joined = on

    def _show(self, on):
        if not self._fx:
            return
        for n, rec in self._fx.items():
            want = n in on
            if rec[1] != want:
                rec[0].MakeVisible() if want else rec[0].MakeInvisible()
                rec[1] = want

    def _extra_states(self):
        from yam_arena import ik
        from yam_arena.station import BASES
        from yam_arena.task_file import ObjectState

        base = self._base
        origin = base.scene.env_origins[0].cpu().numpy().astype(float)
        arr = lambda t: t.detach().cpu().numpy().astype(float)  # noqa: E731
        zero, still = np.zeros(3), np.array((0.0, 0.0, 0.0, 1.0))
        out = {}
        for arm, robot in (("left", "left_robot"), ("right", "right_robot")):
            arm_ids, _ = self._ids[robot]
            p = np.asarray(ik.tcp_pose(arr(base.scene[robot].data.joint_pos.torch[0, arm_ids]), np.array(BASES[arm]))[0], float)
            out[f"{arm}_tcp"] = ObjectState(pos=p, quat_xyzw=still, lin_vel=zero, ang_vel=zero, low=p, high=p)
        world = base.arena_world
        for name in LOOSE:
            pose = arr(world.get_pose_w(name)[0])
            p = pose[:3] - origin
            out[name] = ObjectState(pos=p, quat_xyzw=pose[3:7], lin_vel=arr(world.get_root_linear_velocity_w(name)[0]),
                                    ang_vel=arr(world.get_root_angular_velocity_w(name)[0]), low=p, high=p)
        return out

    def _task_step(self):
        from yam_arena import scene

        if _MON["err"]:
            raise RuntimeError("yam_solder contact monitor failed: " + _MON["err"])
        original = scene.object_states
        last = {}
        m = self.task.module
        touch, _MON["acc"] = _MON["acc"], {}

        def object_states(env, names):
            if "s" not in last:
                st = {**original(env, [n for n in names if n not in FIXED + LOOSE]), **self._extra_states()}
                m.note_step(st, 1.0 / self.info.control_hz, touch)
                last["s"] = st
            return last["s"]

        scene.object_states = object_states
        try:
            result = super()._task_step()
        finally:
            scene.object_states = original
        self._set_joint(m.joined())
        self._show(m.shown())
        if DEBUG and "s" in last:
            print(m.debug_line(last["s"], touch, self._t), flush=True)
        return result


def make_embodiment(**kwargs):
    return SolderEmbodiment(**kwargs)
