"""A Sony ICF-P26 portable AM/FM radio with its telescopic FM antenna, built true size. Created 11_16_26
(America/New_York).

Geometry is NumPy only, so the task file and the tests use it without Isaac Sim; the USD builder imports pxr when called.

Sources (README "Where the numbers come from"):
- Body 69.5 x 119 x 38 mm (W x H x D) with the antenna folded: Sony's ICF-P26 specification pages (sony.com.au,
  sony.com.hk, sony.co.nz; sony.jp gives the same, "including protrusions").
- Mass 190 g with batteries: Sony's specification pages; the same 6.7 oz (190 g) on the BuyDig listing (sony.jp gives
  about 180 g).
- Antenna 11.1 cm fully retracted and 43.2 cm fully extended: the BuyDig listing's specification tab ("Antenna length:
  Fully extended: 17.0 in (43.2 cm); Fully retracted: 4.5 in (11.1 cm)"). Sony does not publish it.
- Not published, chosen and stated: five sections (five tubes of 100 mm, each inner one sliding 80.25 mm, fit the two
  published lengths with about 20 mm of overlap at every joint, typical of such antennas); tube diameters 6.0, 5.0, 4.0,
  3.2 and 2.5 mm; a 6 mm by 7 mm cylindrical tip knob (Sony's product photo shows a plain cylindrical cap wider than the
  thinnest tube); the swivel 6 mm behind the back face, 6 mm below the top and 25.5 mm right of centre (seen from the
  front, the antenna rises from the top right of the back in Sony's photo); every friction value.

Frames (m). The radio's frame sits at the middle of its bottom face on the table: x toward its back (away from the
robot), y toward its left side as seen from the front (world +y), z up. The antenna is one chain on the radio's
articulation: the outer tube turns on the swivel (a revolute joint about the radio's y axis; angle 0 is straight up and
positive tilts the antenna toward the back), and each inner tube slides out of the one below it on a prismatic joint
along the antenna's axis, from 0 (collapsed) to TRAVEL (at its stop). Links are built in the upright, collapsed pose.
"""

import math

import numpy as np

# -- the radio (Sony ICF-P26) -----------------------------------------------------------------------------------------
BODY = dict(w=0.0695, h=0.119, d=0.038)  # Sony's published size
MASS_TOTAL = 0.190  # with two AA cells, Sony
# -- the antenna ------------------------------------------------------------------------------------------------------
RETRACTED = 0.111  # pivot to tip, fully retracted (BuyDig)
EXTENDED = 0.432  # fully extended (BuyDig)
N_SECTIONS = 5
SECTION_LEN = 0.100
TUBE_R = (0.0030, 0.0025, 0.0020, 0.0016, 0.00125)  # outer tube first
KNOB = dict(r=0.0030, len=0.007)
TUBE_START = 0.004  # the outer tube starts this far from the pivot (the swivel knuckle)
TRAVEL = (EXTENDED - RETRACTED) / (N_SECTIONS - 1)  # each inner tube slides 80.25 mm
OVERLAP = SECTION_LEN - TRAVEL  # 19.75 mm left inside the tube below at full extension
PIVOT = np.array((BODY["d"] / 2 + 0.006, -0.0255, BODY["h"] - 0.006))  # in the radio's frame
PIVOT_LIMITS_DEG = (-60.0, 180.0)  # forward (over the top) to folded down the back
TUBE_DENSITY_KG_PER_M = (0.0037 / 0.1, 0.0030 / 0.1, 0.0024 / 0.1, 0.0019 / 0.1, 0.0015 / 0.1)  # thin steel/brass tubes
KNOB_MASS = 0.001
ANTENNA_MASS = sum(d * SECTION_LEN for d in TUBE_DENSITY_KG_PER_M) + KNOB_MASS + 0.002  # + the swivel knuckle
BODY_MASS = MASS_TOTAL - ANTENNA_MASS
# Coulomb friction (PhysX joint friction, static and dynamic efforts). Estimates: each joint of a new pocket-radio antenna
# needs a pull of a newton or two; the tip section, pulled directly, comes out first. Measured on a real unit with a
# 10 N spring scale (README, Reset).
SLIDE_FRICTION_N = (2.4, 2.1, 1.8, 1.5)  # joints 2..5 (outer joint first)
PIVOT_FRICTION_NM = 0.10  # holds the fully extended antenna (about 0.03 N m of weight at horizontal) at any angle, with margin
DYNAMIC_FRACTION = 0.8
# The tubes keep their mass and inertia but carry no weight. Their weight (15 g in all: at most 0.03 N along a section
# and 0.03 N m at the swivel with the antenna level) is far below the friction that holds them, but PhysX joint friction
# lets a joint creep slowly under any steady load, which no real antenna does; with weight on, the released antenna
# sagged about 1 degree and 2 mm per second in the first trial.
ANTENNA_GRAVITY = False
DOFS = ("pivot", "slide2", "slide3", "slide4", "slide5")
MU_TABLE = 0.5  # the radio's plastic base on the table


def tube_span(k, ext=0.0):
    """(start, end) of section k (1-5) along the antenna's axis from the pivot, with every slide at `ext` (m)."""
    if k == 1:
        return TUBE_START, TUBE_START + SECTION_LEN
    top = TUBE_START + SECTION_LEN + (k - 1) * ext
    return top - SECTION_LEN, top


def axis_dir(theta):
    """The antenna's unit axis in the radio's frame at swivel angle theta (rad, 0 up, positive toward the back)."""
    return np.array((math.sin(theta), 0.0, math.cos(theta)))


def antenna_length(slides):
    """Pivot to tip (m) for the four slide positions (m)."""
    return RETRACTED + float(np.sum(np.clip(np.asarray(slides, float), 0.0, TRAVEL)))


def tip_point(theta, slides):
    """The tip knob's top centre in the radio's frame."""
    return PIVOT + antenna_length(slides) * axis_dir(theta)


def knob_centre(theta, slides):
    return PIVOT + (antenna_length(slides) - KNOB["len"] / 2) * axis_dir(theta)


# -- USD ----------------------------------------------------------------------------------------------------------
COLORS = {  # sRGB base colour, metallic, roughness
    "body": ((0.10, 0.10, 0.11), 0.0, 0.5), "grille": ((0.05, 0.05, 0.055), 0.0, 0.8),
    "dial": ((0.80, 0.80, 0.78), 0.0, 0.4), "wheel": ((0.22, 0.22, 0.24), 0.0, 0.6),
    "chrome": ((0.80, 0.81, 0.82), 1.0, 0.18), "knob": ((0.75, 0.76, 0.78), 1.0, 0.3),
}
MU = {"plastic": MU_TABLE, "metal": 0.5}


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
            _schema(prim, "PhysxCollisionAPI", **{"physxCollision:contactOffset": 0.0015, "physxCollision:restOffset": 0.0})
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.surface(collide), UsdShade.Tokens.weakerThanDescendants, "physics")
        return g

    def box(self, path, center, size, look, collide=None):
        from pxr import Gf, UsdGeom

        c = UsdGeom.Cube.Define(self.stage, path)
        c.CreateSizeAttr(1.0)
        c.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
        c.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
        return self._finish(c, look, collide)

    def cyl(self, path, center, r, h, look, collide=None, axis="Z"):
        from pxr import Gf, UsdGeom

        c = UsdGeom.Cylinder.Define(self.stage, path)
        c.CreateAxisAttr(axis)
        c.CreateRadiusAttr(float(r))
        c.CreateHeightAttr(float(h))
        c.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
        return self._finish(c, look, collide)

    def rigid(self, prim, mass, com, inertia, gravity=True):
        from pxr import Gf, UsdPhysics

        UsdPhysics.RigidBodyAPI.Apply(prim)
        _schema(prim, "PhysxRigidBodyAPI", **{"physxRigidBody:solverPositionIterationCount": 32,
                                              "physxRigidBody:solverVelocityIterationCount": 4,
                                              "physxRigidBody:maxDepenetrationVelocity": 0.2,
                                              "physxRigidBody:disableGravity": not gravity,
                                              "physxRigidBody:angularDamping": 0.05, "physxRigidBody:linearDamping": 0.05})
        m = UsdPhysics.MassAPI.Apply(prim)
        m.CreateMassAttr(float(mass))
        m.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, com)))
        m.CreateDiagonalInertiaAttr(Gf.Vec3f(*(max(float(v), 1e-7) for v in inertia)))


def _xform(stage, path, pos=(0.0, 0.0, 0.0)):
    from pxr import Gf, UsdGeom

    x = UsdGeom.Xform.Define(stage, path)
    x.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    return x.GetPrim()


def _joint(stage, kind, path, body0, body1, pos0, pos1, axis, lo, hi):
    from pxr import Gf, Sdf, UsdPhysics

    j = (UsdPhysics.RevoluteJoint if kind == "revolute" else UsdPhysics.PrismaticJoint).Define(stage, path)
    j.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    j.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    j.CreateAxisAttr(axis)
    j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, pos0)))
    j.CreateLocalPos1Attr(Gf.Vec3f(*map(float, pos1)))
    j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    j.CreateLowerLimitAttr(float(lo))
    j.CreateUpperLimitAttr(float(hi))
    return j.GetPrim()


def build_radio(stage, path):
    """The radio under `path`, whose Xform stands on the table at the middle of the radio's bottom face. The radio's body
    is the articulation's free root; the antenna's five tubes hang from it (frames: every tube's frame at the pivot, the
    antenna upright and collapsed). The embodiment sets the start pose through the joints. Returns the root prim."""
    from pxr import UsdPhysics

    root_prim = stage.GetPrimAtPath(path)
    b = _Builder(stage, path)
    W, H, D = BODY["w"], BODY["h"], BODY["d"]

    body = f"{path}/radio"
    bp = _xform(stage, body)
    b.rigid(bp, BODY_MASS, (0.0, 0.0, H * 0.45), (BODY_MASS * (W * W + H * H) / 12, BODY_MASS * (D * D + H * H) / 12,
                                                  BODY_MASS * (W * W + D * D) / 12))
    UsdPhysics.ArticulationRootAPI.Apply(bp)
    _schema(bp, "PhysxArticulationAPI", **{"physxArticulation:solverPositionIterationCount": 32,
                                           "physxArticulation:solverVelocityIterationCount": 4,
                                           "physxArticulation:enabledSelfCollisions": False})
    b.box(f"{body}/case", (0.0, 0.0, H / 2), (D, W, H), "body", "plastic")
    b.box(f"{body}/grille", (-D / 2 - 0.0004, 0.0, 0.045), (0.001, W - 0.012, 0.075), "grille", None)
    b.box(f"{body}/dial", (-D / 2 - 0.0004, 0.0, 0.097), (0.001, W - 0.014, 0.012), "dial", None)
    b.cyl(f"{body}/tuning", (0.0, -W / 2 - 0.001, 0.090), 0.009, 0.003, "wheel", None, "Y")
    b.box(f"{body}/bracket", (D / 2 + 0.003, PIVOT[1], PIVOT[2]), (0.006, 0.008, 0.012), "body", None)

    prev = body
    for k in range(1, N_SECTIONS + 1):
        link = f"{path}/s{k}"
        lp = _xform(stage, link, PIVOT)
        s0, s1 = tube_span(k)
        r = TUBE_R[k - 1]
        m = TUBE_DENSITY_KG_PER_M[k - 1] * SECTION_LEN + (KNOB_MASS if k == N_SECTIONS else 0.0) + (0.002 if k == 1 else 0.0)
        b.rigid(lp, m, (0.0, 0.0, (s0 + s1) / 2), (m * SECTION_LEN ** 2 / 12, m * SECTION_LEN ** 2 / 12, m * r * r),
                gravity=ANTENNA_GRAVITY)
        b.cyl(f"{link}/tube", (0.0, 0.0, (s0 + s1) / 2), r, SECTION_LEN, "chrome", "metal")
        b.cyl(f"{link}/ferrule", (0.0, 0.0, s1 - 0.0015), r + 0.0003, 0.003, "chrome", None)
        if k == 1:
            b.cyl(f"{link}/knuckle", (0.0, 0.0, 0.0), 0.004, 0.008, "chrome", "metal", "Y")
        if k == N_SECTIONS:
            b.cyl(f"{link}/knob", (0.0, 0.0, s1 + KNOB["len"] / 2), KNOB["r"], KNOB["len"], "knob", "metal")
        if k == 1:
            _joint(stage, "revolute", f"{path}/j_pivot", body, link, PIVOT, (0.0, 0.0, 0.0), "Y", *PIVOT_LIMITS_DEG)
        else:
            _joint(stage, "prismatic", f"{path}/j_slide{k}", prev, link, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "Z", 0.0, TRAVEL)
        prev = link
    return root_prim
