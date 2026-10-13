#!/usr/bin/env python3
"""
Generates app/src/main/res/raw/elephant.json - a Lottie animation of a jumping elephant.

The elephant is a rig of parented layers (body -> head -> ears/trunk, body -> thigh -> shin -> foot,
body -> tail). Every frame is computed procedurally:

  * the body follows real physics: anticipation crouch, push-off, ballistic flight
    (h = v0*t - g*t^2/2) and a damped spring landing;
  * legs are solved with two-bone inverse kinematics, so feet stay planted on the ground while
    the knees bend in the crouch / landing, and tuck in during the flight;
  * head, ears, trunk, tail and body pitch are damped springs that track a "pose target"
    and are additionally driven by the body's vertical acceleration (inertia), so they lag,
    overshoot and settle naturally.

The composition contains two markers:
  "jump" - the one-shot jump (starts and ends in the rest pose),
  "idle" - a seamless idle loop (breathing, ear flaps, tail sway).

Run:  python3 tools/generate_elephant.py
"""
import json
import math
import os

FPS = 60
W, H = 580, 500
GROUND_Y = 430.0

# ----------------------------------------------------------------------------------------------
# Timeline (frames)
# ----------------------------------------------------------------------------------------------
F_CROUCH_END = 22      # anticipation: elephant squats, head pressed down
F_TAKEOFF = 30         # legs fully extended, feet leave the ground
FLIGHT = 46            # frames in the air
F_LAND = F_TAKEOFF + FLIGHT
F_APEX = F_TAKEOFF + FLIGHT / 2
JUMP_END = F_LAND + 64 # landing absorption + settling
IDLE_LEN = 120
TOTAL = JUMP_END + IDLE_LEN

CROUCH = 36.0          # crouch depth, px
JUMP_HEIGHT = 150.0    # apex height, px
G = 8 * JUMP_HEIGHT / FLIGHT ** 2      # gravity, px/frame^2
V0 = 4 * JUMP_HEIGHT / FLIGHT          # take-off speed, px/frame

# ----------------------------------------------------------------------------------------------
# Rig geometry (body local space, origin = torso centre, elephant faces right, y goes down)
# ----------------------------------------------------------------------------------------------
BODY_X = 262.0
L1, L2 = 58.0, 54.0                    # thigh / shin length
STAND = 104.0                          # hip-to-foot distance in rest pose (knees slightly bent)
HIP_Y = 40.0
BODY_REST_Y = GROUND_Y - HIP_Y - STAND
NECK = (100.0, -28.0)

LEGS = {
    # name: (hip local x, hip local y, ground y, is_front, is_far, delay)
    "near_front": (58.0, HIP_Y, GROUND_Y, True, False, 0),
    "far_front": (82.0, HIP_Y - 10, GROUND_Y - 10, True, True, 2),
    "near_hind": (-68.0, HIP_Y, GROUND_Y, False, False, 0),
    "far_hind": (-44.0, HIP_Y - 10, GROUND_Y - 10, False, True, 2),
}

# ----------------------------------------------------------------------------------------------
# Colours
# ----------------------------------------------------------------------------------------------
def rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)] + [1]

SKIN = rgb("#9AA5B8")
SKIN_FAR = rgb("#7D879B")
SKIN_DARK = rgb("#6E788C")
OUTLINE = rgb("#3C4250")
EAR_INNER = rgb("#E9A9B6")
EAR_INNER_FAR = rgb("#C98C99")
TUSK = rgb("#FFF8E7")
WHITE = rgb("#FFFFFF")
BLACK = rgb("#1E2129")
GROUND = rgb("#7CC36A")
GROUND_DARK = rgb("#5FA552")
SHADOW = [0, 0, 0, 1]
CHEEK = rgb("#F2A7B3")


def smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def lerp(a, b, t):
    return a + (b - a) * t


# ----------------------------------------------------------------------------------------------
# Body height h(f) (up is positive), sampled per frame with sub-frame physics on landing
# ----------------------------------------------------------------------------------------------
def body_height():
    h = []
    # landing spring (simulated once, sampled per frame)
    land = []
    y, v = 0.0, -V0
    w, z = 0.25, 0.40
    sub = 20
    for _ in range(JUMP_END - F_LAND + 1):
        land.append(y)
        for _ in range(sub):
            dt = 1.0 / sub
            a = -w * w * y - 2 * z * w * v
            v += a * dt
            y += v * dt
    for f in range(TOTAL + 1):
        if f <= F_CROUCH_END:
            s = f / F_CROUCH_END
            val = -CROUCH * (0.5 - 0.5 * math.cos(math.pi * s))
        elif f <= F_TAKEOFF:
            # cubic Hermite: from -CROUCH with v=0 to 0 with v=V0 (legs push and straighten)
            dur = F_TAKEOFF - F_CROUCH_END
            s = (f - F_CROUCH_END) / dur
            h00 = 2 * s ** 3 - 3 * s ** 2 + 1
            h11 = s ** 3 - s ** 2
            val = -CROUCH * h00 + V0 * dur * h11
        elif f <= F_LAND:
            t = f - F_TAKEOFF
            val = V0 * t - 0.5 * G * t * t
        elif f <= JUMP_END:
            val = land[f - F_LAND]
            # make sure we end exactly in the rest pose
            val *= 1 - smoothstep((f - (JUMP_END - 14)) / 14)
        else:
            t = (f - JUMP_END) / IDLE_LEN
            val = 1.6 * math.sin(2 * math.pi * t)  # gentle breathing
        h.append(val)
    return h


def acceleration(h):
    a = [0.0] * len(h)
    for f in range(1, len(h) - 1):
        a[f] = h[f + 1] - 2 * h[f] + h[f - 1]
    return a


def spring(targets, drive, w, z, x0=0.0, sub=10, lo=None, hi=None):
    """Damped spring x'' = w^2 (target - x) - 2 z w x' + drive  (per frame units)."""
    out = []
    x, v = x0, 0.0
    for f in range(len(targets)):
        out.append(x)
        for _ in range(sub):
            dt = 1.0 / sub
            acc = w * w * (targets[f] - x) - 2 * z * w * v + drive[f]
            v += acc * dt
            x += v * dt
            if lo is not None and x < lo:
                x, v = lo, 0.0
            if hi is not None and x > hi:
                x, v = hi, 0.0
    return out


def phase_target(f, crouch, push, ascent, descent, land, rest=0.0):
    """Piecewise pose target for a secondary part over the jump."""
    if f <= F_CROUCH_END:
        return lerp(rest, crouch, smoothstep(f / (F_CROUCH_END * 0.8)))
    if f <= F_TAKEOFF:
        return lerp(crouch, push, smoothstep((f - F_CROUCH_END) / (F_TAKEOFF - F_CROUCH_END)))
    if f <= F_APEX:
        return lerp(push, ascent, smoothstep((f - F_TAKEOFF) / 8))
    if f <= F_LAND:
        # just after the top of the arc the elephant stretches out
        return lerp(ascent, descent, smoothstep((f - F_APEX + 2) / 10))
    if f <= JUMP_END:
        return lerp(land, rest, smoothstep((f - F_LAND - 10) / 30))
    return rest


def blend_to_rest(values, rest, start=JUMP_END - 16, end=JUMP_END):
    """Force the jump to end exactly in the rest pose (seamless hand-off to the idle loop)."""
    out = list(values)
    for f in range(start, end + 1):
        k = smoothstep((f - start) / (end - start))
        out[f] = lerp(values[f], rest, k)
    return out


# ----------------------------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------------------------
h = body_height()
acc = acceleration(h)
frames = range(TOTAL + 1)
idle_t = [(f - JUMP_END) / IDLE_LEN if f > JUMP_END else 0.0 for f in frames]


def idle(f, amp, cycles=1, phase=0.0):
    if f <= JUMP_END:
        return 0.0
    return amp * math.sin(2 * math.pi * (cycles * idle_t[f]) + phase) - amp * math.sin(phase)


def in_jump(f):
    return f <= JUMP_END


# Head pose p: -1 = pressed into the shoulders, +1 = stretched forward/up
head_target = [phase_target(f, -1.0, -0.7, 0.1, 1.0, -0.2) if in_jump(f) else 0.0 for f in frames]
head_p = spring(head_target, [-0.035 * a if in_jump(f) else 0 for f, a in enumerate(acc)],
                w=0.28, z=0.55, lo=-1.5, hi=1.4)
head_p = blend_to_rest(head_p, 0.0)

# Ears e: + = flared back/up, - = flopped forward/down (floppy, under-damped)
ear_target = [phase_target(f, 0.55, 0.7, 0.2, 0.9, 0.0) if in_jump(f) else 0.0 for f in frames]
ear_e = spring(ear_target, [-0.05 * a if in_jump(f) else 0 for f, a in enumerate(acc)],
               w=0.22, z=0.22, lo=-1.0, hi=1.5)
ear_e = blend_to_rest(ear_e, 0.0)

# Trunk r: + = curled in towards the chest, - = raised forward
trunk_target = [phase_target(f, 0.9, 0.8, -0.3, -1.0, 0.2) if in_jump(f) else 0.0 for f in frames]
trunk_r = spring(trunk_target, [0.04 * a if in_jump(f) else 0 for f, a in enumerate(acc)],
                 w=0.22, z=0.35, lo=-1.6, hi=1.4)
trunk_r = blend_to_rest(trunk_r, 0.0)
trunk_tip = spring(trunk_r, [0.0] * len(trunk_r), w=0.32, z=0.4)  # tip lags behind the base
trunk_tip = blend_to_rest(trunk_tip, 0.0)

# Tail t: + = lifted up/back
tail_target = [phase_target(f, -0.3, 0.2, 0.3, 0.9, 0.0) if in_jump(f) else 0.0 for f in frames]
tail_t = spring(tail_target, [-0.06 * a if in_jump(f) else 0 for f, a in enumerate(acc)],
                w=0.24, z=0.25, lo=-1.0, hi=1.6)
tail_t = blend_to_rest(tail_t, 0.0)
tail_tip = spring(tail_t, [0.0] * len(tail_t), w=0.3, z=0.3)
tail_tip = blend_to_rest(tail_tip, 0.0)

# Body pitch q (deg, + = nose down)
pitch_target = [phase_target(f, 4.0, -7.0, -5.0, 5.0, -2.0) if in_jump(f) else 0.0 for f in frames]
pitch = spring(pitch_target, [0.0] * len(frames), w=0.2, z=0.8)
pitch = blend_to_rest(pitch, 0.0)

# Add idle motion on top of the rest pose
for f in frames:
    head_p[f] += idle(f, 0.12, 1, 0.0)
    ear_e[f] += idle(f, 0.28, 3, 0.0)
    trunk_r[f] += idle(f, 0.35, 1, 0.0)
    trunk_tip[f] += idle(f, 0.35, 1, -0.6)
    tail_t[f] += idle(f, 0.45, 2, 0.0)
    tail_tip[f] += idle(f, 0.45, 2, -0.8)


def velocity(f):
    if 0 < f < len(h) - 1:
        return (h[f + 1] - h[f - 1]) / 2
    return 0.0


def torso_squash(f):
    """Squash & stretch: stretch with vertical speed in the air, squash when compressed."""
    if F_TAKEOFF - 3 <= f <= F_LAND:
        s = 0.006 * abs(velocity(f))
    elif f <= JUMP_END:
        s = 0.0035 * min(0.0, h[f])
    else:
        s = 0.008 * math.sin(2 * math.pi * idle_t[f])
    return s


# ----------------------------------------------------------------------------------------------
# Leg IK
# ----------------------------------------------------------------------------------------------
def rot(v, deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return (v[0] * c - v[1] * s, v[0] * s + v[1] * c)


def ang_of(dx, dy):
    """Lottie rotation (deg, clockwise) that turns the 'down' vector (0,1) into (dx,dy)."""
    return math.degrees(math.atan2(-dx, dy))


def solve_ik(dx, dy):
    d = math.hypot(dx, dy)
    d = max(abs(L1 - L2) + 1, min(L1 + L2 - 0.5, d))
    phi = ang_of(dx, dy)
    cos_a = (L1 * L1 + d * d - L2 * L2) / (2 * L1 * d)
    a = math.degrees(math.acos(max(-1.0, min(1.0, cos_a))))
    t1 = phi - a  # knee points forward (+x), like an elephant's knee / wrist
    kx = -L1 * math.sin(math.radians(t1))
    ky = L1 * math.cos(math.radians(t1))
    t2 = ang_of(dx - kx, dy - ky)
    return t1, t2


def flight_phase(f, delay):
    s = (f - F_TAKEOFF - delay) / FLIGHT
    return s


def leg_angles(name, f):
    hx, hy, gy, front, far, delay = LEGS[name]
    body_y = BODY_REST_Y - h[f]
    q = pitch[f]
    ox, oy = rot((hx, hy), q)
    hip = (BODY_X + ox, body_y + oy)
    stance = (BODY_X + hx, gy)

    foot = stance
    if F_TAKEOFF < f < F_LAND:
        s = flight_phase(f, delay)
        s = max(0.0, min(1.0, s))
        # tuck the feet under the belly early in the flight, reach for the ground before landing
        bump = math.sin(math.pi * min(1.0, s / 0.88)) ** 2 if s < 0.88 else 0.0
        tuck = (46 if front else 40) * bump
        dx = (-14 if front else 16) * bump
        rest_d = gy - (BODY_REST_Y + hy)
        air = (hip[0] + dx, hip[1] + rest_d - tuck)
        # blend from ground contact into the airborne pose and back
        w = smoothstep((f - F_TAKEOFF) / 5) * (1 - smoothstep((f - (F_LAND - 5)) / 5))
        foot = (lerp(stance[0], air[0], w), lerp(stance[1], air[1], w))
        # never go below the ground
        foot = (foot[0], min(foot[1], gy))

    t1, t2 = solve_ik(foot[0] - hip[0], foot[1] - hip[1])
    thigh_local = t1 - q
    shin_local = t2 - t1
    foot_local = 0.0 - t2  # foot stays level with the ground
    return thigh_local, shin_local, foot_local


# ----------------------------------------------------------------------------------------------
# Lottie helpers
# ----------------------------------------------------------------------------------------------
def static(v):
    return {"a": 0, "k": v}


def simplify(values, eps):
    """Drop samples that linear interpolation reproduces within eps (values: list of lists)."""
    n = len(values)
    keep = [0]
    i = 0
    while i < n - 1:
        j = i + 1
        while j + 1 < n:
            ok = True
            for k in range(i + 1, j + 2):
                t = (k - i) / (j + 1 - i)
                for d in range(len(values[0])):
                    if abs(lerp(values[i][d], values[j + 1][d], t) - values[k][d]) > eps:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
            j += 1
        keep.append(j)
        i = j
    return keep


def anim(values, eps=0.05, nd=2):
    """values: list (per frame) of scalars or lists -> Lottie animated property."""
    vals = [v if isinstance(v, (list, tuple)) else [v] for v in values]
    vals = [[round(x, nd) for x in v] for v in vals]
    if all(v == vals[0] for v in vals):
        return static(vals[0] if len(vals[0]) > 1 else vals[0][0])
    keep = simplify(vals, eps)
    kfs = []
    for idx, f in enumerate(keep):
        kf = {"t": f, "s": vals[f]}
        if idx < len(keep) - 1:
            kf["i"] = {"x": [1], "y": [1]}
            kf["o"] = {"x": [0], "y": [0]}
        kfs.append(kf)
    return {"a": 1, "k": kfs}


def transform(p=(0, 0), a=(0, 0), s=(100, 100), r=0, o=100):
    def prop(v):
        return v if isinstance(v, dict) else static(v)

    return {
        "o": prop(o),
        "r": prop(r),
        "p": prop(list(p) + [0] if isinstance(p, (list, tuple)) else p),
        "a": prop(list(a) + [0]),
        "s": prop(list(s) + [100] if isinstance(s, (list, tuple)) else s),
    }


def group(name, items, fill=None, stroke=None, stroke_w=3.0):
    it = list(items)
    if stroke is not None:
        it.append({"ty": "st", "nm": "Stroke", "c": static(stroke), "o": static(100),
                   "w": static(stroke_w), "lc": 2, "lj": 2, "ml": 4})
    if fill is not None:
        it.append({"ty": "fl", "nm": "Fill", "c": static(fill), "o": static(100), "r": 1})
    it.append({"ty": "tr", "p": static([0, 0]), "a": static([0, 0]), "s": static([100, 100]),
               "r": static(0), "o": static(100), "sk": static(0), "sa": static(0)})
    return {"ty": "gr", "nm": name, "it": it, "np": len(it)}


def ellipse(cx, cy, w, h):
    return {"ty": "el", "nm": "Ellipse", "p": static([cx, cy]), "s": static([w, h]), "d": 1}


def rect(cx, cy, w, h, r):
    return {"ty": "rc", "nm": "Rect", "p": static([cx, cy]), "s": static([w, h]), "r": static(r), "d": 1}


def path(points, closed=True):
    """points: list of (v, in_tangent, out_tangent) with relative tangents."""
    return {"ty": "sh", "nm": "Path", "d": 1, "ks": static({
        "c": closed,
        "v": [list(p[0]) for p in points],
        "i": [list(p[1]) for p in points],
        "o": [list(p[2]) for p in points],
    })}


_layers = []
_ind = [0]


def layer(name, shapes=None, parent=None, ks=None, null=False):
    _ind[0] += 1
    lay = {
        "ddd": 0, "ind": _ind[0], "ty": 3 if null else 4, "nm": name, "sr": 1,
        "ks": ks or transform(), "ao": 0, "ip": 0, "op": TOTAL + 1, "st": 0, "bm": 0,
    }
    if parent is not None:
        lay["parent"] = parent
    if not null:
        lay["shapes"] = shapes or []
    _layers.append(lay)
    return _ind[0]


# ----------------------------------------------------------------------------------------------
# Build the rig. Layers are created parent-first; drawing order is fixed at the end.
# ----------------------------------------------------------------------------------------------
order = {}  # layer index -> z (higher = on top)

body = layer("Body (root)", null=True, ks=transform(
    p=anim([[BODY_X, BODY_REST_Y - h[f], 0] for f in frames]),
    r=anim([pitch[f] for f in frames]),
))

# torso
torso = layer("Torso", parent=body, ks=transform(
    p=(0, 0),
    s=anim([[100 * (1 - 0.6 * torso_squash(f)), 100 * (1 + torso_squash(f)), 100] for f in frames], eps=0.1),
), shapes=[
    group("Belly shade", [ellipse(5, 38, 190, 70)], fill=rgb("#8D98AC")),
    group("Torso", [ellipse(0, 0, 236, 164)], fill=SKIN, stroke=OUTLINE, stroke_w=3.5),
])
order[torso] = 50

# head: position = neck + offset (pressed back/down or stretched forward/up)
head = layer("Head", parent=body, ks=transform(
    p=anim([[NECK[0] + 12 * head_p[f], NECK[1] - 9 * head_p[f], 0] for f in frames]),
    r=anim([-15 * head_p[f] for f in frames]),
), shapes=[
    group("Eye", [
        group("Glint", [ellipse(67, -24, 5, 5)], fill=WHITE),
        group("Pupil", [ellipse(64, -21, 11, 14)], fill=BLACK),
    ]),
    group("Brow", [path([((54, -36), (0, 0), (6, -4)), ((72, -35), (-6, -4), (0, 0))], closed=False)],
          stroke=OUTLINE, stroke_w=3),
    group("Cheek", [ellipse(72, 4, 18, 11)], fill=CHEEK),
    group("Tusk", [path([
        ((70, 22), (0, 0), (2, 10)),
        ((96, 42), (-12, 2), (4, -1)),
        ((100, 34), (0, 0), (0, 0)),
        ((82, 18), (6, 8), (0, 0)),
    ])], fill=TUSK, stroke=OUTLINE, stroke_w=2.5),
    group("Skull", [ellipse(42, -6, 118, 112)], fill=SKIN, stroke=OUTLINE, stroke_w=3.5),
])
order[head] = 80

# ears (rotate around their root)
def ear_shape(inner, outer, scale=1.0):
    k = scale
    outline = path([
        ((0, 0), (0, 0), (-30 * k, -12 * k)),
        ((-62 * k, 30 * k), (2 * k, -30 * k), (-2 * k, 34 * k)),
        ((-20 * k, 92 * k), (-30 * k, 2 * k), (16 * k, 0)),
        ((8 * k, 50 * k), (0, 24 * k), (0, 0)),
    ])
    inner_p = path([
        ((-6 * k, 10 * k), (0, 0), (-20 * k, -6 * k)),
        ((-48 * k, 32 * k), (2 * k, -20 * k), (-2 * k, 24 * k)),
        ((-20 * k, 78 * k), (-20 * k, 2 * k), (10 * k, 0)),
        ((-2 * k, 48 * k), (0, 18 * k), (0, 0)),
    ])
    return [group("Inner", [inner_p], fill=inner), group("Ear", [outline], fill=outer, stroke=OUTLINE, stroke_w=3.5)]


near_ear = layer("Ear near", parent=head, ks=transform(
    p=(22, -26),
    r=anim([38 * ear_e[f] for f in frames]),
    s=anim([[100 - 14 * max(0.0, ear_e[f]) + 8 * min(0.0, ear_e[f]), 100, 100] for f in frames], eps=0.2),
), shapes=ear_shape(EAR_INNER, SKIN))
order[near_ear] = 90

far_ear = layer("Ear far", parent=head, ks=transform(
    p=(52, -50),
    r=anim([-20 + 30 * ear_e[f] for f in frames]),
), shapes=ear_shape(EAR_INNER_FAR, SKIN_FAR, 0.8))
order[far_ear] = 30

# trunk: 3 segments, each parented to the previous one
TRUNK = [(34, 34), (30, 28), (28, 22)]  # (length, width)
TRUNK_REST = [-28, 12, 26]
trunk_parent = head
trunk_pos = (90, 14)
for i, (ln, wd) in enumerate(TRUNK):
    gain = [18, 26, 38][i]
    src = trunk_r if i == 0 else trunk_tip
    shapes = []
    if i == len(TRUNK) - 1:
        shapes.append(group("Nostril", [ellipse(0, ln + 1, wd * 0.8, 9)], fill=SKIN_DARK, stroke=OUTLINE, stroke_w=2.5))
    shapes.append(group("Wrinkles", [
        path([((-wd / 2 + 4, ln * 0.45), (0, 0), (0, 0)), ((wd / 2 - 4, ln * 0.45), (0, 0), (0, 0))], closed=False)
    ], stroke=SKIN_DARK, stroke_w=2))
    shapes.append(group("Segment", [rect(0, ln / 2, wd, ln + 14, wd / 2)], fill=SKIN, stroke=OUTLINE, stroke_w=3))
    seg = layer(f"Trunk {i + 1}", parent=trunk_parent, ks=transform(
        p=trunk_pos,
        r=anim([TRUNK_REST[i] + gain * src[f] for f in frames]),
    ), shapes=shapes)
    order[seg] = 79 - i  # base segment over the next one, all under the head
    trunk_parent = seg
    trunk_pos = (0, ln)

# tail
tail1 = layer("Tail 1", parent=body, ks=transform(
    p=(-112, -18),
    r=anim([38 + 45 * tail_t[f] for f in frames]),
), shapes=[group("Tail", [rect(0, 16, 9, 38, 4.5)], fill=SKIN, stroke=OUTLINE, stroke_w=2.5)])
order[tail1] = 45
tail2 = layer("Tail 2", parent=tail1, ks=transform(
    p=(0, 32),
    r=anim([-6 + 40 * (tail_tip[f] - tail_t[f]) for f in frames]),
), shapes=[
    group("Tuft", [path([
        ((-7, 26), (0, 0), (-4, 8)),
        ((0, 44), (-6, 0), (6, 0)),
        ((7, 26), (4, 8), (0, 0)),
    ])], fill=SKIN_DARK, stroke=OUTLINE, stroke_w=2.5),
    group("Tail", [rect(0, 14, 7, 30, 3.5)], fill=SKIN, stroke=OUTLINE, stroke_w=2.5),
])
order[tail2] = 44

# legs
leg_data = {name: [leg_angles(name, f) for f in frames] for name in LEGS}
for name, (hx, hy, gy, front, far, delay) in LEGS.items():
    skin = SKIN_FAR if far else SKIN
    base_z = 20 if far else 60
    data = leg_data[name]
    thigh = layer(f"Thigh {name}", parent=body, ks=transform(
        p=(hx, hy), r=anim([d[0] for d in data], eps=0.08),
    ), shapes=[group("Thigh", [rect(0, L1 / 2 - 4, 46, L1 + 26, 22)], fill=skin, stroke=OUTLINE, stroke_w=3)])
    shin = layer(f"Shin {name}", parent=thigh, ks=transform(
        p=(0, L1), r=anim([d[1] for d in data], eps=0.08),
    ), shapes=[group("Shin", [rect(0, L2 / 2 - 4, 40, L2 + 16, 18)], fill=skin, stroke=OUTLINE, stroke_w=3)])
    foot = layer(f"Foot {name}", parent=shin, ks=transform(
        p=(0, L2), r=anim([d[2] for d in data], eps=0.08),
    ), shapes=[
        group("Toes", [ellipse(x, 4, 8, 6) for x in (-11, 0, 11)], fill=TUSK),
        group("Foot", [rect(0, -2, 46, 16, 8)], fill=skin, stroke=OUTLINE, stroke_w=3),
    ])
    order[thigh] = base_z + 3
    order[shin] = base_z + 2
    order[foot] = base_z + 1

# shadow on the ground (shrinks and fades when the elephant is in the air)
shadow = layer("Shadow", ks=transform(
    p=(BODY_X - 5, GROUND_Y + 2),
    s=anim([[100 * (1 - 0.0028 * max(0.0, h[f])), 100 * (1 - 0.0028 * max(0.0, h[f])), 100] for f in frames], eps=0.3),
    o=anim([30 * (1 - 0.0035 * max(0.0, h[f])) for f in frames], eps=0.3),
), shapes=[group("Shadow", [ellipse(0, 0, 260, 26)], fill=SHADOW)])
order[shadow] = 5

ground = layer("Ground", shapes=[
    group("Grass tufts", [
        path([((x - 8, GROUND_Y + 2), (0, 0), (0, 0)), ((x - 3, GROUND_Y - 9), (0, 0), (0, 0)),
              ((x, GROUND_Y + 1), (0, 0), (0, 0)), ((x + 4, GROUND_Y - 11), (0, 0), (0, 0)),
              ((x + 8, GROUND_Y + 2), (0, 0), (0, 0))])
        for x in (40, 120, 380, 455)
    ], fill=GROUND_DARK),
    group("Ground", [rect(W / 2, GROUND_Y + 40, W + 40, 80, 0)], fill=GROUND),
])
order[ground] = 1

# Lottie draws the first layer on top
_layers.sort(key=lambda l: -order.get(l["ind"], 0))

anim_json = {
    "v": "5.7.4",
    "fr": FPS,
    "ip": 0,
    "op": TOTAL + 1,
    "w": W,
    "h": H,
    "nm": "Jumping elephant",
    "ddd": 0,
    "assets": [],
    "layers": _layers,
    "markers": [
        {"tm": 0, "cm": "jump", "dr": JUMP_END},
        {"tm": JUMP_END, "cm": "idle", "dr": IDLE_LEN},
    ],
}

out = os.path.join(os.path.dirname(__file__), "..", "app", "src", "main", "res", "raw", "elephant.json")
out = os.path.normpath(out)
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as fh:
    json.dump(anim_json, fh, separators=(",", ":"))

print(f"wrote {out}: {os.path.getsize(out) / 1024:.1f} KiB, {TOTAL + 1} frames, "
      f"jump 0..{JUMP_END}, idle {JUMP_END}..{TOTAL}")
print(f"take-off v0={V0:.2f}px/f, g={G:.3f}px/f^2, min landing h={min(h[F_LAND:JUMP_END]):.1f}px")
