"""`yam_sprinkler`: the stock yam_arena embodiment plus the impulse sprinkler (sprinkler_arc.mech), one fixed-base
articulation whose head and two trip collars turn on revolute joints with Coulomb joint friction. Created 11_16_26
(America/New_York).

The scorer also gets `left_tcp` and `right_tcp` (each gripper's fingertip midpoint) and `head`, `lower` and `upper`, each
on the axis with its heading as a yaw (the joint angle) and its joint speed as the z spin. They are read through the PhysX
tensor view of the articulation. After each control step the task's note_step runs, and the head joint's drive is set to
what drive_command() returns (water off: no drive; water on: a velocity target and a torque limit). The harness is
unchanged."""

import math
import os

import numpy as np
from yam_arena.inspect_embodiment import YamArenaEmbodiment

DEBUG = os.environ.get("SPRINKLER_DEBUG") == "1"
ROOT = "sprinkler"
PARTS = ("head", "lower", "upper")


def _place(stage, prim_path, translation, orientation):
    from pxr import Gf, UsdGeom

    x = UsdGeom.Xform.Define(stage, prim_path)
    x.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    qx, qy, qz, qw = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)  # Isaac Lab spawners: xyzw
    x.AddOrientOp().Set(Gf.Quatf(float(qw), Gf.Vec3f(float(qx), float(qy), float(qz))))
    return x.GetPrim()


class SprinklerEmbodiment(YamArenaEmbodiment):
    _view = None
    _path = None
    _dof = None
    _last_cmd = None

    def reset(self, scene, *, seed=None):
        self._ensure_sim(None)
        m = self.task.module
        m.reset_latch()
        self._start_pose()
        self._last_cmd = None
        self._set_drive(m.drive_command())
        return super().reset(scene, seed=seed)

    def _ensure_sim(self, goal):
        import atexit

        from yam_arena import inspect_embodiment as base

        if self._env is not None:
            return super()._ensure_sim(goal)
        if base._APP is None:  # Kit must be up before Arena is imported
            from isaaclab.app import AppLauncher

            os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
            base._APP = AppLauncher(headless=self.headless, device=self.device, enable_cameras=True).app
            atexit.register(base._close_app_at_exit)
        from yam_arena import scene

        original = scene.task_object
        owner = self

        def spawn(prim_path, cfg, translation=None, orientation=None, **kwargs):
            import omni.usd

            from . import mech

            stage = omni.usd.get_context().get_stage()
            _place(stage, prim_path, translation, orientation)
            owner._path = prim_path
            return mech.build_sprinkler(stage, prim_path)

        def make_object(spec):
            import isaaclab.sim as sim_utils
            from isaaclab_arena.assets.object import Object
            from isaaclab_arena.assets.object_type import ObjectType
            from isaaclab_arena.utils.pose import Pose

            if spec.name != ROOT:
                return original(spec)
            cfg = sim_utils.CuboidCfg(size=(0.01, 0.01, 0.01))
            cfg.func = spawn
            return Object(name=spec.name, object_type=ObjectType.BASE, spawner_cfg=cfg,
                          initial_pose=Pose(position_xyz=tuple(float(v) for v in spec.pos), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

        scene.task_object = make_object
        try:
            return super()._ensure_sim(goal)
        finally:
            scene.task_object = original

    # -- the articulation, through the PhysX tensor view -----------------------------------------------------------------
    def _art(self):
        if self._view is None:
            import omni.physics.tensors as tensors

            from . import mech

            sim_view = None
            try:
                from isaaclab.sim import SimulationContext

                sim_view = SimulationContext.instance().physics_sim_view
            except Exception:  # noqa: BLE001
                sim_view = None
            if sim_view is None:
                sim_view = tensors.create_simulation_view("torch")
                sim_view.set_subspace_roots("/")
            self._view = sim_view.create_articulation_view(self._path)
            names = list(self._view.shared_metatype.dof_names)
            self._dof = {part: names.index(f"j_{part}") for part in PARTS}
            fr = self._get(self._view.get_dof_friction_properties)
            for part, torque in (("head", mech.BEARING_FRICTION), ("lower", mech.COLLAR_SLIP), ("upper", mech.COLLAR_SLIP)):
                fr[0, self._dof[part], :] = (torque, torque, 0.0)
            self._put(self._view.set_dof_friction_properties, fr, self._view.get_dof_friction_properties)
            if DEBUG:
                print("SPRINKLER dofs", names, "friction", self._get(self._view.get_dof_friction_properties).tolist(), flush=True)
        return self._view

    @staticmethod
    def _get(getter):
        """A tensor-view getter's result as a NumPy copy (the view may hand back warp, torch or NumPy arrays)."""
        a = getter()
        if hasattr(a, "numpy"):
            try:
                return np.array(a.numpy(), dtype=np.float32)
            except TypeError:
                pass
        return np.array(a.detach().cpu().numpy() if hasattr(a, "detach") else a, dtype=np.float32)

    @staticmethod
    def _put(setter, data, getter):
        """Set from a NumPy array, in the same kind of array and on the same device as the getter returns."""
        like = getter()
        idx = np.array([0], dtype=np.int32)
        if type(like).__module__.startswith("warp"):
            import warp as wp

            setter(wp.from_numpy(data.astype(np.float32), dtype=wp.float32, device=like.device),
                   wp.from_numpy(idx, dtype=wp.int32, device=like.device))
        elif hasattr(like, "detach"):
            import torch

            setter(torch.tensor(data, dtype=torch.float32, device=like.device), torch.tensor(idx, dtype=torch.int32, device=like.device))
        else:
            setter(data.astype(np.float32), idx)

    def _start_pose(self):
        from . import mech

        v = self._art()
        pos = self._get(v.get_dof_positions)
        for part in PARTS:
            pos[0, self._dof[part]] = math.radians(mech.START[part])
        self._put(v.set_dof_positions, pos, v.get_dof_positions)
        self._put(v.set_dof_velocities, pos * 0.0, v.get_dof_velocities)

    def _set_drive(self, cmd):
        """cmd = (target deg/s, max torque N m) for the head joint's drive."""
        if cmd == self._last_cmd:
            return
        v = self._art()
        tgt = self._get(v.get_dof_velocity_targets) * 0.0
        tgt[0, self._dof["head"]] = math.radians(cmd[0])
        mf = self._get(v.get_dof_max_forces)
        mf[0, self._dof["head"]] = float(cmd[1])
        self._put(v.set_dof_max_forces, mf, v.get_dof_max_forces)
        self._put(v.set_dof_velocity_targets, tgt, v.get_dof_velocity_targets)
        self._last_cmd = cmd

    def _extra_states(self):
        from yam_arena import ik
        from yam_arena.station import BASES
        from yam_arena.task_file import ObjectState

        from . import mech

        base = self._base
        arr = lambda t: t.detach().cpu().numpy().astype(float)  # noqa: E731
        zero, still = np.zeros(3), np.array((0.0, 0.0, 0.0, 1.0))
        out = {}
        for arm, robot in (("left", "left_robot"), ("right", "right_robot")):
            arm_ids, _ = self._ids[robot]
            q = arr(base.scene[robot].data.joint_pos.torch[0, arm_ids])
            p = np.asarray(ik.tcp_pose(q, np.array(BASES[arm]))[0], float)
            out[f"{arm}_tcp"] = ObjectState(pos=p, quat_xyzw=still, lin_vel=zero, ang_vel=zero, low=p, high=p)
        v = self._art()
        pos, vel = self._get(v.get_dof_positions)[0], self._get(v.get_dof_velocities)[0]
        axis = np.array((mech.AXIS_X, mech.AXIS_Y, 0.0))
        for part in PARTS:
            a = float(pos[self._dof[part]])
            quat = np.array((0.0, 0.0, math.sin(a / 2), math.cos(a / 2)))
            w = np.array((0.0, 0.0, float(vel[self._dof[part]])))
            out[part] = ObjectState(pos=axis, quat_xyzw=quat, lin_vel=zero, ang_vel=w, low=axis, high=axis)
        return out

    def _task_step(self):
        from yam_arena import scene

        original = scene.object_states
        m = self.task.module

        def object_states(env, names):
            names = [n for n in names if n != ROOT and n not in PARTS]
            return {**(original(env, names) if names else {}), **self._extra_states()}

        scene.object_states = object_states
        try:
            st0 = object_states(self._env, [])
            m.note_step(st0, 1.0 / self.info.control_hz)
            result = super()._task_step()
        finally:
            scene.object_states = original
        self._set_drive(m.drive_command())
        if DEBUG:
            print(m.debug_line(st0, self._t), flush=True)
        return result


def make_embodiment(**kwargs):
    return SprinklerEmbodiment(**kwargs)
