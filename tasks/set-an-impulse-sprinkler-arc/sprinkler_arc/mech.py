"""A plastic impulse sprinkler on a spike in a turf tray: the stationary bearing with its two friction trip collars, and the
rotating head whose trip lever strikes the collars' tabs. Created 11_16_26 (America/New_York).

Geometry constants are NumPy only, so the task file and the tests can use them without Isaac Sim; the USD builders import
pxr when called.

Frames (m). All three moving parts turn about one vertical axis at (AXIS_X, AXIS_Y) on the table.
- The sprinkler is one fixed-base articulation whose root prim sits on the axis at the table's top. Its root link: the turf
  mat, the spike's upright, the stationary bearing (the head's inlet nut and the sleeve the collars ride on) and two marker
  flags. Three links turn on revolute joints about the axis, each with Coulomb joint friction (PhysX static and dynamic
  friction effort, N m). Collisions between jointed links are off (a PhysX joint's default), so the collars and the head
  turn on their joints, not on contact with the sleeve.
- Head link: origin on the axis at the inlet's bottom (Z0 above the table), +x along the nozzle at joint angle 0. A column,
  the nozzle barrel pointing forward and 25 degrees up, the drive arm on top with its spoon in front and counterweight
  behind, the spring hub, and at the back the trip lever, which hangs down beside both collars' tabs. Its joint carries the
  bearing's friction and, with the water on, a velocity drive with a torque limit (the water's push, scripted).
- Collar links ("lower" and "upper"): origin on the axis at Z0, +x along the tab. A plastic ring on the sleeve (drawn
  only) and a radial tab (the collider the lever strikes and a finger pushes). Joint friction COLLAR_SLIP is the ring's
  grip on the sleeve.

Sources and estimates (see the README): the head's overall size is the catalog's 4.38 x 4.5 in (111 x 114 mm) for the
Rain Bird P5R; widths of individual parts, the collar and tab sizes, the slip torque and the head's mass are estimates.
"""

import math

import numpy as np

# -- layout (base frame: table top on the axis) ---------------------------------------------------------------------
AXIS_X, AXIS_Y = 0.270, 0.000
TURF = dict(x=(-0.110, 0.190), y=(-0.190, 0.190), h=0.010)  # turf mat, 30 x 38 cm, 10 mm thick
Z0 = TURF["h"] + 0.030  # the head's inlet bottom: on the spike's upright, 30 mm above the turf
SPIKE_R = 0.0105  # 1/2 in pipe upright, 21 mm across
NUT = dict(r=0.0130, z=(0.000, 0.014))  # the head's inlet nut (stationary)
SLEEVE = dict(r=0.0105, z=(0.014, 0.039))  # stationary bearing sleeve the collars ride on
FLAG_R = 0.240  # the marker flags stand this far from the axis
FLAG_POST = dict(r=0.003, h=0.110)
FLAG_CLOTH = dict(w=0.040, h=0.028, t=0.002)
TARGET = (-45.0, 45.0)  # world headings (deg, +x = 0, counter-clockwise seen from above) of the flags: the arc to water

# -- collars (collar frame: z from the inlet bottom, +x along the tab) ------------------------------------------------
RING = dict(r_in=0.0108, r_out=0.0155)
TAB = dict(r=(0.0155, 0.0300), w=0.006)  # the tab: radial lug the trip lever strikes, 6 mm wide
COLLAR_Z = {"lower": (0.017, 0.025), "upper": (0.027, 0.035)}
COLLAR_MASS = 0.004
COLLAR_INERTIA = (2.0e-6, 2.0e-6, 2.0e-6)
COLLAR_SLIP = 0.04  # N m: a collar slips on its sleeve at this torque (1.3 N on the tab's outer end): Coulomb joint friction

# -- head (head frame: z from the inlet bottom, +x along the nozzle) ---------------------------------------------------
COLUMN = dict(r=0.015, z=(0.040, 0.100))
LEVER = dict(x=(-0.025, -0.017), w=0.004, z=(0.015, 0.040))  # trip lever: hangs beside both tabs (z 17-35 mm)
BRIDGE = dict(x=(-0.025, -0.013), w=0.004, z=(0.037, 0.042))  # ties the lever to the column
NOZZLE = dict(r=0.006, root=(0.0, 0.0, 0.055), tilt=math.radians(25.0), length=0.050)
DRIVE_ARM = dict(x=(-0.053, 0.058), w=0.010, z=(0.100, 0.110))
SPOON = dict(x=(0.044, 0.058), w=0.020, z=(0.078, 0.110))
WEIGHT = dict(x=(-0.053, -0.036), w=0.016, z=(0.092, 0.110))
HUB = dict(r=0.009, z=(0.100, 0.114))
HEAD_MASS = 0.150
HEAD_COM = (0.0, 0.0, 0.075)
HEAD_INERTIA = (1.6e-4, 1.6e-4, 1.2e-4)
HEAD_LEN = DRIVE_ARM["x"][1] - DRIVE_ARM["x"][0]  # 111 mm
HEAD_TOP = HUB["z"][1]  # 114 mm above the inlet bottom
LEVER_R = -0.5 * (LEVER["x"][0] + LEVER["x"][1])  # 21 mm behind the axis

# -- the head's drive --------------------------------------------------------------------------------------------------
BEARING_FRICTION = 0.005  # N m: the head turns on its bearing against this Coulomb joint friction
DRIVE_SPEED = 20.0  # deg/s with the water on (a real head steps round more slowly; see the README)
DRIVE_TORQUE = 0.015  # N m, under half of COLLAR_SLIP, so a tab stops the head and the collar stays put
DRIVE_DAMP = 0.1  # N m s / deg: the drive is at its torque limit until the head is within 0.4 deg/s of the set speed

# Contact geometry: where the lever stops against a tab. They first touch at the lever's inner edge (r = 17 mm), where
# their half-widths, (TAB w + LEVER w) / 2 = 5 mm, subtend the widest angle.
LEVER_R_IN = -LEVER["x"][1]
STOP_DEG = math.degrees((TAB["w"] + LEVER["w"]) / 2 / LEVER_R_IN)  # 16.9 deg between a tab's centre line and the lever's


def lever_heading(nozzle_deg):
    return nozzle_deg + 180.0


def tab_for_limit(nozzle_limit_deg, side):
    """The tab heading (deg) that stops the head with its nozzle at nozzle_limit_deg. side 'cw' (the clockwise limit,
    the lever moving clockwise into the tab) or 'ccw'."""
    lever = lever_heading(nozzle_limit_deg)
    return lever - STOP_DEG if side == "cw" else lever + STOP_DEG


def limit_for_tab(tab_deg, side):
    """The nozzle heading (deg) at which a tab at tab_deg stops the head."""
    return tab_deg + STOP_DEG - 180.0 if side == "cw" else tab_deg - STOP_DEG - 180.0


# Which collar stops which way: the lower collar is the clockwise stop (it stops the lever turning clockwise, so it sets
# the clockwise end of the arc), the upper collar the counter-clockwise stop.
SIDE = {"lower": "cw", "upper": "ccw"}
START_LIMITS = (-110.0, 110.0)  # start: the head waters a 220 degree arc, each end 65 degrees past its flag
START = {
    "head": -23.0,  # nozzle heading at the start
    "lower": tab_for_limit(START_LIMITS[0], "cw"),  # 56.4
    "upper": tab_for_limit(START_LIMITS[1], "ccw"),  # 303.6
}
GOAL_TAB = {"lower": tab_for_limit(TARGET[0], "cw"), "upper": tab_for_limit(TARGET[1], "ccw")}  # 121.4, 238.6

COLORS = {  # sRGB base colour, metallic, roughness
    "turf": ((0.20, 0.38, 0.14), 0.0, 0.95), "spike": ((0.25, 0.26, 0.28), 0.8, 0.4),
    "nut": ((0.10, 0.10, 0.10), 0.0, 0.6), "sleeve": ((0.12, 0.12, 0.12), 0.0, 0.5),
    "lower": ((0.15, 0.15, 0.15), 0.0, 0.5), "upper": ((0.15, 0.15, 0.15), 0.0, 0.5),
    "tab_lower": ((0.15, 0.15, 0.15), 0.0, 0.5), "tab_upper": ((0.15, 0.15, 0.15), 0.0, 0.5),
    "head": ((0.10, 0.10, 0.11), 0.0, 0.45), "steel": ((0.70, 0.71, 0.73), 1.0, 0.3),
    "lever": ((0.70, 0.71, 0.73), 1.0, 0.3), "flag": ((0.95, 0.45, 0.05), 0.0, 0.7), "post": ((0.85, 0.85, 0.85), 0.2, 0.5),
}
MU = {"plastic": 0.5, "steel": 0.4, "turf": 0.8}


# -- USD ---------------------------------------------------------------------------------------------------------------
def _schema(prim, schema, **attrs):
    from pxr import Sdf

    if not prim.ApplyAPI(schema):
        prim.AddAppliedSchema(schema)
    types = {bool: Sdf.ValueTypeNames.Bool, int: Sdf.ValueTypeNames.Int, float: Sdf.ValueTypeNames.Float}
    for name, value in attrs.items():
        prim.CreateAttribute(name, types[type(value)]).Set(value)


def _lin(c):
    return tuple(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c)


class _Builder:
    def __init__(self, stage, root):
        self.stage, self.root = stage, root
        self.looks, self.phys = {}, {}

    def look(self, name):
        from pxr import Gf, Sdf, UsdShade

        if name not in self.looks:
            color, metal, rough = COLORS[name]
            mat = UsdShade.Material.Define(self.stage, f"{self.root}/Looks/{name}")
            sh = UsdShade.Shader.Define(self.stage, f"{self.root}/Looks/{name}/shader")
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*_lin(color)))
            sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metal))
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(rough))
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            self.looks[name] = mat
        return self.looks[name]

    def surface(self, name):
        from pxr import UsdPhysics, UsdShade

        if name not in self.phys:
            mat = UsdShade.Material.Define(self.stage, f"{self.root}/Physics/{name}")
            pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
            pm.CreateStaticFrictionAttr(MU[name])
            pm.CreateDynamicFrictionAttr(MU[name])
            self.phys[name] = mat
        return self.phys[name]

    def _finish(self, g, look, collide):
        from pxr import UsdPhysics, UsdShade

        prim = g.GetPrim()
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.look(look))
        if collide:
            UsdPhysics.CollisionAPI.Apply(prim)
            _schema(prim, "PhysxCollisionAPI", **{"physxCollision:contactOffset": 0.001, "physxCollision:restOffset": 0.0})
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.surface(collide), UsdShade.Tokens.weakerThanDescendants, "physics")
        return g

    def box(self, path, lo, hi, look, collide=None, yaw=0.0):
        """An axis-aligned box from lo to hi, turned yaw (rad) about the frame's z axis through the origin."""
        from pxr import Gf, UsdGeom

        lo, hi = np.array(lo, float), np.array(hi, float)
        c = UsdGeom.Cube.Define(self.stage, path)
        c.CreateSizeAttr(1.0)
        ctr = (lo + hi) / 2
        cz, sz = math.cos(yaw), math.sin(yaw)
        c.AddTranslateOp().Set(Gf.Vec3d(cz * ctr[0] - sz * ctr[1], sz * ctr[0] + cz * ctr[1], float(ctr[2])))
        if yaw:
            c.AddRotateZOp().Set(float(math.degrees(yaw)))
        c.AddScaleOp().Set(Gf.Vec3f(*map(float, hi - lo)))
        return self._finish(c, look, collide)

    def cyl(self, path, center, r, h, look, collide=None, tilt_y=None):
        """A cylinder along z, or tipped about y by tilt_y degrees (its axis then runs toward +x as it rises)."""
        from pxr import Gf, UsdGeom

        c = UsdGeom.Cylinder.Define(self.stage, path)
        c.CreateAxisAttr("Z")
        c.CreateRadiusAttr(float(r))
        c.CreateHeightAttr(float(h))
        c.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
        if tilt_y is not None:
            c.AddRotateYOp().Set(float(tilt_y))
        return self._finish(c, look, collide)

    def ring(self, path, r_in, r_out, z0, z1, look, n=16):
        """A ring drawn as n short boxes (no collider)."""
        rm, t = (r_in + r_out) / 2, r_out - r_in
        w = 2 * rm * math.tan(math.pi / n) + 0.0005
        for k in range(n):
            a = 2 * math.pi * k / n
            self.box(f"{path}_{k:02d}", (rm - t / 2, -w / 2, z0), (rm + t / 2, w / 2, z1), look, None, yaw=a)

    def rigid(self, prim, kinematic=False, mass=None, com=None, inertia=None):
        from pxr import Gf, UsdPhysics

        UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(kinematic)
        _schema(prim, "PhysxRigidBodyAPI", **{"physxRigidBody:solverPositionIterationCount": 32,
                                              "physxRigidBody:solverVelocityIterationCount": 4,
                                              "physxRigidBody:maxDepenetrationVelocity": 0.2,
                                              "physxRigidBody:angularDamping": 0.0})
        if mass is not None:
            m = UsdPhysics.MassAPI.Apply(prim)
            m.CreateMassAttr(float(mass))
            if com is not None:
                m.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, com)))
            if inertia is not None:
                m.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, inertia)))


def _link(stage, path, mass, com, inertia):
    from pxr import Gf, UsdGeom

    x = UsdGeom.Xform.Define(stage, path)
    x.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(Z0)))
    return x.GetPrim()


def _joint(stage, path, root, body, friction, drive_damping=None):
    """Revolute joint about z from the root link (at the inlet bottom on the axis) to a link whose origin is there, with
    Coulomb friction `friction` N m (set by the embodiment), and optionally an angular velocity drive (stiffness 0) set at run time."""
    from pxr import Gf, Sdf, UsdPhysics

    j = UsdPhysics.RevoluteJoint.Define(stage, path)
    j.CreateBody0Rel().SetTargets([Sdf.Path(root)])
    j.CreateBody1Rel().SetTargets([Sdf.Path(body)])
    j.CreateAxisAttr("Z")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(Z0)))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    prim = j.GetPrim()  # the friction itself is set through the tensor view once the simulation runs (embodiment.py)
    if drive_damping is not None:
        d = UsdPhysics.DriveAPI.Apply(prim, "angular")
        d.CreateTypeAttr("force")
        d.CreateStiffnessAttr(0.0)
        d.CreateDampingAttr(float(drive_damping))
        d.CreateMaxForceAttr(0.0)
        d.CreateTargetVelocityAttr(0.0)
    return prim


def build_sprinkler(stage, path):
    """The whole sprinkler under `path` (already placed on the axis at the table's top): a fixed-base articulation."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    root_prim = stage.GetPrimAtPath(path)
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    _schema(root_prim, "PhysxArticulationAPI", **{"physxArticulation:solverPositionIterationCount": 32,
                                                  "physxArticulation:solverVelocityIterationCount": 4,
                                                  "physxArticulation:enabledSelfCollisions": True})
    b = _Builder(stage, path)
    root = f"{path}/root"
    UsdGeom.Xform.Define(stage, root)
    rp = stage.GetPrimAtPath(root)
    b.rigid(rp, mass=5.0, com=(0.0, 0.0, 0.02), inertia=(0.05, 0.05, 0.05))
    (x0, x1), (y0, y1), h = TURF["x"], TURF["y"], TURF["h"]
    b.box(f"{root}/turf", (x0, y0, 0.0), (x1, y1, h), "turf", "turf")
    b.cyl(f"{root}/spike", (0.0, 0.0, (h + Z0) / 2), SPIKE_R, Z0 - h, "spike", "steel")
    b.cyl(f"{root}/nut", (0.0, 0.0, Z0 + sum(NUT["z"]) / 2), NUT["r"], NUT["z"][1] - NUT["z"][0], "nut", "plastic")
    b.cyl(f"{root}/sleeve", (0.0, 0.0, Z0 + sum(SLEEVE["z"]) / 2), SLEEVE["r"], SLEEVE["z"][1] - SLEEVE["z"][0], "sleeve", "plastic")
    for k, deg in enumerate(TARGET):
        a = math.radians(deg)
        fx, fy = FLAG_R * math.cos(a), FLAG_R * math.sin(a)
        b.cyl(f"{root}/flag{k}_post", (fx, fy, h + FLAG_POST["h"] / 2), FLAG_POST["r"], FLAG_POST["h"], "post", "plastic")
        c = FLAG_CLOTH
        b.box(f"{root}/flag{k}_cloth", (fx - c["t"] / 2, fy, h + FLAG_POST["h"] - c["h"]), (fx + c["t"] / 2, fy + (c["w"] if k else -c["w"]), h + FLAG_POST["h"]), "flag")
    # fixed to the world where it stands
    world = UsdGeom.Xformable(root_prim).ComputeLocalToWorldTransform(0).ExtractTranslation()
    fix = UsdPhysics.FixedJoint.Define(stage, f"{path}/fix")
    fix.CreateBody1Rel().SetTargets([Sdf.Path(root)])
    fix.CreateLocalPos0Attr(Gf.Vec3f(*map(float, world)))
    fix.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    # the head
    head = f"{path}/head"
    hp = _link(stage, head, HEAD_MASS, HEAD_COM, HEAD_INERTIA)
    b.rigid(hp, mass=HEAD_MASS, com=HEAD_COM, inertia=HEAD_INERTIA)
    b.cyl(f"{head}/column", (0.0, 0.0, sum(COLUMN["z"]) / 2), COLUMN["r"], COLUMN["z"][1] - COLUMN["z"][0], "head", "plastic")
    for name, d, look in (("lever", LEVER, "lever"), ("bridge", BRIDGE, "lever"), ("arm", DRIVE_ARM, "head"), ("spoon", SPOON, "head"), ("weight", WEIGHT, "steel")):
        b.box(f"{head}/{name}", (d["x"][0], -d["w"] / 2, d["z"][0]), (d["x"][1], d["w"] / 2, d["z"][1]), look, "steel" if look != "head" else "plastic")
    n = NOZZLE
    mid = np.array(n["root"]) + 0.5 * n["length"] * np.array((math.cos(n["tilt"]), 0.0, math.sin(n["tilt"])))
    b.cyl(f"{head}/nozzle", mid, n["r"], n["length"], "head", "plastic", tilt_y=90.0 - math.degrees(n["tilt"]))
    b.cyl(f"{head}/hub", (0.0, 0.0, sum(HUB["z"]) / 2), HUB["r"], HUB["z"][1] - HUB["z"][0], "steel", "steel")
    _joint(stage, f"{path}/j_head", root, head, BEARING_FRICTION, DRIVE_DAMP)
    # the collars
    for which in ("lower", "upper"):
        cp_path = f"{path}/{which}"
        cp = _link(stage, cp_path, COLLAR_MASS, None, COLLAR_INERTIA)
        z0, z1 = COLLAR_Z[which]
        b.rigid(cp, mass=COLLAR_MASS, com=(0.02, 0.0, (z0 + z1) / 2), inertia=COLLAR_INERTIA)
        b.ring(f"{cp_path}/ring", RING["r_in"], RING["r_out"], z0, z1, which)
        b.box(f"{cp_path}/tab", (TAB["r"][0], -TAB["w"] / 2, z0), (TAB["r"][1], TAB["w"] / 2, z1), f"tab_{which}", "plastic")
        _joint(stage, f"{path}/j_{which}", root, cp_path, COLLAR_SLIP)
    return root_prim


DOFS = ("j_head", "j_lower", "j_upper")
