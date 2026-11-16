"""`yam_guide_rail`: the stock yam_arena embodiment plus the conveyor section, its adjustable guide rail, the two clamp
knobs, the setting plugs and the can (guide_rails.mech). Created 11_16_26 10:55 EST.

The scorer also gets `left_tcp` and `right_tcp` (each gripper's fingertip midpoint) and every loose part in its own
frame. After each control step the task's note_step runs; the friction limits it reports (each knob's turning torque
and each rod's holding force, from the clamp law) are set on the mechanism's joints. The robot, cameras, action and
observations are the stock ones."""

import math
import os

import numpy as np
from yam_arena.inspect_embodiment import YamArenaEmbodiment

from . import mech

DEBUG = os.environ.get("RAIL_DEBUG") == "1"
DEV = os.environ.get("RAIL_DEV") == "1"
FIXED = "conveyor"
PARTS = ("rail", "knob_left", "knob_right", "plug_left", "plug_right", "can")


def _place(stage, prim_path, translation, orientation):
    from pxr import Gf, UsdGeom

    x = UsdGeom.Xform.Define(stage, prim_path)
    x.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    qx, qy, qz, qw = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)  # Isaac Lab spawners: xyzw
    x.AddOrientOp().Set(Gf.Quatf(float(qw), Gf.Vec3f(float(qx), float(qy), float(qz))))
    return x.GetPrim()


class GuideRailEmbodiment(YamArenaEmbodiment):
    _limits = None  # joint name -> the friction limit's USD attribute

    def reset(self, scene, *, seed=None):
        m = self.task.module
        m.reset_trial()
        if self._limits is not None:
            self._set_as_found()
        out = super().reset(scene, seed=seed)
        if self._limits is None:
            import omni.usd

            self._limits = mech.controls(omni.usd.get_context().get_stage())
            assert set(self._limits) == {*mech.JOINTS, "lock_left", "lock_right"}, self._limits
        self._set_as_found()
        m.reset_trial()
        return out

    def _set_as_found(self):
        for name, attr in self._limits.items():  # as found: both clamps locked
            if name.startswith("lock"):
                mech.set_lock(attr, name.split("_")[1], mech.RAIL_START[0])
            else:
                attr.Set(float(mech.hold_force(mech.A_START) if name.startswith("slide") else mech.knob_torque(mech.A_START)))

    def _ensure_sim(self, goal):
        import atexit

        from yam_arena import inspect_embodiment as base

        if base._APP is None:  # Kit must be up before Arena is imported
            from isaaclab.app import AppLauncher

            os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
            base._APP = AppLauncher(headless=self.headless, device=self.device, enable_cameras=True).app
            atexit.register(base._close_app_at_exit)
        from yam_arena import scene

        original = scene.task_object

        def spawn(prim_path, cfg, translation=None, orientation=None, **kwargs):
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            _place(stage, prim_path, translation, orientation)
            name = prim_path.rstrip("/").rsplit("/", 1)[-1]
            prim = mech.BUILDERS[name](stage, prim_path)
            mech.add_joints(stage, name, prim_path.rstrip("/"))
            return prim

        def make_object(spec):
            import isaaclab.sim as sim_utils
            from isaaclab_arena.assets.object import Object
            from isaaclab_arena.assets.object_type import ObjectType
            from isaaclab_arena.utils.pose import Pose

            if spec.name != FIXED and spec.name not in PARTS:
                return original(spec)
            cfg = sim_utils.CuboidCfg(size=(0.01, 0.01, 0.01))
            cfg.func = spawn
            kind = ObjectType.BASE if spec.name == FIXED else ObjectType.RIGID
            return Object(name=spec.name, object_type=kind, spawner_cfg=cfg,
                          initial_pose=Pose(position_xyz=tuple(float(v) for v in spec.pos),
                                            rotation_xyzw=(0.0, 0.0, math.sin(spec.yaw / 2), math.cos(spec.yaw / 2))))

        scene.task_object = make_object
        try:
            return super()._ensure_sim(goal)
        finally:
            scene.task_object = original

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
            if DEV:  # the wrist roll joint's torque and angle, for dev/sim.py
                d = base.scene[robot].data
                w = np.array((float(arr(d.applied_torque.torch[0, arm_ids])[5]), float(arr(d.joint_pos.torch[0, arm_ids])[5]), 0.0))
                out[f"{arm}_wrist"] = ObjectState(pos=w, quat_xyzw=still, lin_vel=zero, ang_vel=zero, low=w, high=w)
        world = base.arena_world
        for name in PARTS:
            pose = arr(world.get_pose_w(name)[0])
            p = pose[:3] - origin
            out[name] = ObjectState(pos=p, quat_xyzw=pose[3:7], lin_vel=arr(world.get_root_linear_velocity_w(name)[0]),
                                    ang_vel=arr(world.get_root_angular_velocity_w(name)[0]), low=p, high=p)
        return out

    def _task_step(self):
        from yam_arena import scene

        original = scene.object_states
        last = {}
        m = self.task.module

        def object_states(env, names):
            names = [n for n in names if n != FIXED and n not in PARTS]  # these come from _extra_states
            last["s"] = {**original(env, names), **self._extra_states()}
            m.note_step(last["s"], 1.0 / self.info.control_hz)  # before the score is read
            return last["s"]

        scene.object_states = object_states
        try:
            result = super()._task_step()
        finally:
            scene.object_states = original
        if "s" in last:
            for name, value in m.take_commands():
                if name.startswith("lock"):
                    mech.set_lock(self._limits[name], name.split("_")[1], value)
                else:
                    self._limits[name].Set(float(value))
            if DEBUG:
                print(f"RAIL t={self._t:6.2f} " + m.debug_line(last["s"]), flush=True)
        return result


def make_embodiment(**kwargs):
    return GuideRailEmbodiment(**kwargs)
