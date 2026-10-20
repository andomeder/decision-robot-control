"""Obstacle maze. No dotted path.

Six boxes. North and south walls close the outside. Three plugs leave a gap
that switches from the high side to the low side and back. A purple spur
dead-ends on the south wall. A local rule casts straight rays and puts a
white disc on the next open step. d1-3B reads that picture. It does not
receive meters or degrees.

The disc is recomputed from where the duck is. It is not a waypoint list.

    MUJOCO_GL=egl .venv/bin/python packages/microduck-harness/maze_clip.py --sim
    MUJOCO_GL=egl .venv/bin/python packages/microduck-harness/maze_clip.py --smoke
    MUJOCO_GL=egl .venv/bin/python packages/microduck-harness/maze_clip.py --out clips/maze_drift.mp4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import subprocess
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "decision-client"))
sys.path.insert(0, os.path.dirname(__file__))

from decision_client import DecisionClient
from full_demo import DECISION_URL, FRAME_DT, Plate, W, controller_speed
from goal_clip import HOLD_S, _choose, write_bed
from policy_runner import CONTROL_DT, DuckRuntime
from spatial import depth_picture, forward_depth_picture

# Half-sizes are meters. North and south bars run off the chase view.
# Slots between plugs are 0.64 m. Each plug stops 2 cm short of the wall and
# of the next plug, so the tops do not share a face.
# Open gaps stay put: clay high 0.71 m, amber low 0.79 m, olive high 0.73 m.
# South wall inner face y=-0.70. North wall inner face y=1.05.
OBSTACLES = (
    {"pos": (1.20, 2.15, 0.16), "size": (3.40, 1.10, 0.16), "rgba": (0.28, 0.38, 0.55, 1.0)},
    {"pos": (1.20, -1.85, 0.16), "size": (3.40, 1.15, 0.16), "rgba": (0.30, 0.32, 0.36, 1.0)},
    # Clay. Bottom y=-0.68, top y=0.34. The open gap is above it.
    {"pos": (0.50, -0.17, 0.16), "size": (0.16, 0.51, 0.16), "rgba": (0.62, 0.32, 0.22, 1.0)},
    # Amber. Top y=1.03, bottom y=0.09. The open gap is below it.
    {"pos": (1.46, 0.56, 0.16), "size": (0.16, 0.47, 0.16), "rgba": (0.72, 0.48, 0.18, 1.0)},
    # Olive. Bottom y=-0.68, top y=0.32. The open gap is above it.
    {"pos": (2.42, -0.18, 0.16), "size": (0.16, 0.50, 0.16), "rgba": (0.42, 0.48, 0.26, 1.0)},
    # Purple spur. Right face x=0.32, 2 cm clear of the clay. Bottom y=-0.68.
    {"pos": (0.00, -0.49, 0.16), "size": (0.32, 0.19, 0.16), "rgba": (0.40, 0.26, 0.48, 1.0)},
)
GOAL = (2.95, 0.62)
CELL = 0.45
COLS = 5
ROWS = 6
CENTER = 2
INFLATE = 0.10
STAND_M = 0.42
BODY_R = 0.12
RAY_CLEAR = 0.14

WALL_RGB = (128, 64, 176)
OPEN_RGB = (32, 40, 54)
BG = (27, 34, 46)
CREAM = (236, 230, 218)
INK = (18, 22, 30)
ORANGE = (232, 120, 48)

MAZE_QUESTIONS = {
    "skill": {
        "type": "choice",
        "instructions": (
            "Choose the one skill whose criterion matches the GRID picture. "
            "You are the triangle at the bottom, facing up. "
            "Violet cells are walls. The white disc is the next open step. "
            "The orange ring is the final post. Do not walk into violet. "
            "No distances or angles are given."
        ),
        "criteria": {
            "turn_left": "the white disc is left of the CENTER column",
            "turn_right": "the white disc is right of the CENTER column",
            "walk_forward": "the white disc is in the CENTER column and not in the bottom row",
            "stand": "the white disc is in the bottom CENTER cell",
        },
    },
}


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/TTF/DejaVuSans.ttf"
    return ImageFont.truetype(path, size)


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _to_robot(x: float, y: float, yaw: float, px: float, py: float) -> tuple[float, float]:
    dx, dy = px - x, py - y
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def _to_world(x: float, y: float, yaw: float, forward: float, left: float) -> tuple[float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * forward - s * left, y + s * forward + c * left


def _hits_box(px: float, py: float, inflate: float) -> bool:
    for obs in OBSTACLES:
        ox, oy, _ = obs["pos"]
        hx, hy, _ = obs["size"]
        if abs(px - ox) <= hx + inflate and abs(py - oy) <= hy + inflate:
            return True
    return False


def _cell_center(col: int, row: int) -> tuple[float, float]:
    """Robot frame. Column 0 is far left. Row 0 is the nearest row."""
    left = (CENTER - col) * CELL
    forward = (row + 0.5) * CELL
    return forward, left


def _clamp_cell(forward: float, left: float) -> tuple[int, int]:
    col = CENTER - int(math.floor(left / CELL + 0.5))
    row = int(math.floor(forward / CELL))
    return max(0, min(COLS - 1, col)), max(0, min(ROWS - 1, row))


def occupancy(x: float, y: float, yaw: float) -> set[tuple[int, int]]:
    """A cell is a wall if its center or a corner sits in a box."""
    blocked = set()
    pad = CELL * 0.38
    for col in range(COLS):
        for row in range(ROWS):
            forward, left = _cell_center(col, row)
            for df, dl in ((0.0, 0.0), (pad, pad), (pad, -pad), (-pad, pad), (-pad, -pad)):
                wx, wy = _to_world(x, y, yaw, forward + df, left + dl)
                if _hits_box(wx, wy, INFLATE):
                    blocked.add((col, row))
                    break
    return blocked


def _segment_hits(x: float, y: float, gx: float, gy: float, box: int | None, clearance: float) -> bool:
    """True if the segment meets one box, or any box when box is None."""
    dist = math.hypot(gx - x, gy - y)
    steps = max(1, int(dist / 0.04))
    for i in range(1, steps + 1):
        t = i / steps
        px = x + (gx - x) * t
        py = y + (gy - y) * t
        if box is None:
            if _hits_box(px, py, clearance):
                return True
            continue
        obs = OBSTACLES[box]
        ox, oy, _ = obs["pos"]
        hx, hy, _ = obs["size"]
        if abs(px - ox) <= hx + clearance and abs(py - oy) <= hy + clearance:
            return True
    return False


def _corridor() -> tuple[float, float]:
    """Inner edges of the north and south bars."""
    south = max(o["pos"][1] + o["size"][1] for o in OBSTACLES if o["size"][1] > 1.0 and o["pos"][1] < 0)
    north = min(o["pos"][1] - o["size"][1] for o in OBSTACLES if o["size"][1] > 1.0 and o["pos"][1] > 0)
    return south, north


def _samples() -> list[tuple[float, float]]:
    """Gap centers between the plugs. Rebuilt from the boxes on every call."""
    south, north = _corridor()
    gaps = []
    for obs in OBSTACLES:
        hx, hy, _ = obs["size"]
        if hy > 1.0:
            continue
        ox, oy, _ = obs["pos"]
        top, bot = oy + hy, oy - hy
        above, below = north - top, bot - south
        # A spur with a corridor-sized opening is a dead end, not a gate.
        if 0.40 <= above <= 1.0:
            gaps.append((ox, (top + north) / 2.0, hx))
        elif 0.40 <= below <= 1.0:
            gaps.append((ox, (bot + south) / 2.0, hx))
    gaps.sort()
    pts: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()

    def add(px: float, py: float) -> None:
        if py > north - 0.12 or py < south + 0.12:
            return
        if _hits_box(px, py, BODY_R + 0.04):
            return
        key = (round(px, 2), round(py, 2))
        if key in seen:
            return
        seen.add(key)
        pts.append((px, py))

    for ox, gy, hx in gaps:
        for shift in (-0.34, 0.0, 0.34):
            add(ox + shift, gy)
    for (x0, y0, _), (x1, y1, _) in zip(gaps, gaps[1:]):
        mid = (x0 + x1) / 2.0
        add(mid, y0)
        add(mid, (y0 + y1) / 2.0)
        add(mid, y1)
    add(GOAL[0], GOAL[1])
    return pts


def _waypoint(x: float, y: float) -> tuple[float, float]:
    """First open gap on a shortest clear chain to the post.

    The chain is rebuilt from the duck's pose. Nothing is stored on the world.
    """
    pts = _samples()
    nodes = [(x, y), *pts]
    goal_i = len(nodes) - 1
    n = len(nodes)
    dist = [1e9] * n
    prev = [-1] * n
    dist[0] = 0.0
    used = [False] * n
    for _ in range(n):
        u = min((i for i in range(n) if not used[i]), key=lambda i: dist[i])
        if dist[u] >= 1e8:
            break
        used[u] = True
        if u == goal_i:
            break
        ux, uy = nodes[u]
        for v in range(n):
            if used[v] or v == u:
                continue
            vx, vy = nodes[v]
            step = math.hypot(vx - ux, vy - uy)
            if step > 0.08 and _segment_hits(ux, uy, vx, vy, None, RAY_CLEAR):
                continue
            w = 0.0 if step < 0.08 else step
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
    if dist[goal_i] >= 1e8:
        return GOAL
    chain = []
    v = goal_i
    while v > 0:
        chain.append(nodes[v])
        v = prev[v]
    chain.reverse()
    for px, py in chain:
        if math.hypot(px - x, py - y) >= 0.16:
            return px, py
    return GOAL


def _open_disc(blocked: set[tuple[int, int]], skill: str) -> tuple[int, int]:
    """A cell whose column matches the skill, preferring one that is not violet."""
    if skill == "stand":
        return (CENTER, 0)
    if skill == "walk_forward":
        for row in (2, 3, 1, 4, 5):
            if (CENTER, row) not in blocked:
                return (CENTER, row)
        return (CENTER, 2)
    cols = (1, 0) if skill == "turn_left" else (3, 4)
    for col in cols:
        for row in (2, 3, 1, 4):
            if (col, row) not in blocked:
                return (col, row)
    return (cols[0], 2)


def plan(x: float, y: float, yaw: float, prev: dict | None = None) -> dict:
    """One step. The disc sits on the next open gap the rule can still see.

    A gap is held until the duck reaches it, so a few centimetres of motion
    cannot swap the step for the one behind.
    """
    gx, gy = GOAL
    forward, left = _to_robot(x, y, yaw, gx, gy)
    dist = math.hypot(gx - x, gy - y)
    bearing = math.degrees(_wrap(math.atan2(left, forward)))
    blocked = occupancy(x, y, yaw)
    gcol, grow = _clamp_cell(forward, left)
    if forward < 0:
        grow = 0
    prev_skill = None if prev is None else prev.get("skill")

    aligned = forward > 0.0 and abs(bearing) <= 22.0
    if dist <= STAND_M and aligned and not _segment_hits(x, y, gx, gy, None, RAY_CLEAR):
        skill = "stand"
        aim = yaw
        tx, ty = gx, gy
    else:
        tx, ty = _waypoint(x, y)
        if prev is not None:
            px, py = prev["disc_xy"]
            held = math.hypot(px - x, py - y)
            if held > 0.22 and not _segment_hits(x, y, px, py, None, RAY_CLEAR):
                tx, ty = px, py
        elif math.hypot(tx - x, ty - y) < 0.12 and dist > STAND_M:
            nx, ny = _waypoint(tx, ty)
            if math.hypot(nx - tx, ny - ty) > 0.08:
                tx, ty = nx, ny
        aim = math.atan2(ty - y, tx - x)
        err = _wrap(aim - yaw)
        err_deg = abs(math.degrees(err))
        if prev_skill in ("turn_left", "turn_right"):
            turning = err_deg > 8.0
        else:
            turning = err_deg > 14.0
        if turning:
            skill = "turn_left" if err > 0.0 else "turn_right"
        else:
            skill = "walk_forward"

    px = x + math.cos(aim) * 0.70
    py = y + math.sin(aim) * 0.70
    af, al = _to_robot(x, y, yaw, px, py)
    col, row = _clamp_cell(af, al)
    disc = _open_disc(blocked, skill)
    if skill == "walk_forward" and col == CENTER and row > 0 and (col, row) not in blocked:
        disc = (CENTER, row)
    elif skill == "turn_left" and col < CENTER and (col, row) not in blocked:
        disc = (col, max(row, 1))
    elif skill == "turn_right" and col > CENTER and (col, row) not in blocked:
        disc = (col, max(row, 1))

    return {
        "skill": skill,
        "disc": disc,
        "disc_xy": (tx, ty),
        "blocked": blocked,
        "goal_cell": (gcol, grow),
        "dist": dist,
        "bearing": bearing,
        "aim": aim,
    }


def maze_grid(x: float, y: float, yaw: float, choice: dict) -> Image.Image:
    width, height = 640, 760
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)
    title = _font(26, bold=True)
    label = _font(20, bold=True)
    draw.text((16, 12), "GRID", fill=CREAM, font=title)
    draw.rectangle([120, 16, 148, 40], fill=WALL_RGB)
    draw.text((156, 14), "wall", fill=CREAM, font=label)
    draw.ellipse([250, 14, 274, 38], fill=(250, 250, 250), outline=INK, width=3)
    draw.text((282, 14), "step", fill=CREAM, font=label)
    draw.ellipse([390, 14, 414, 38], outline=ORANGE, width=4)
    draw.text((422, 14), "post", fill=CREAM, font=label)

    margin_l, margin_r = 16, 16
    top, bottom = 64, 78
    cw = (width - margin_l - margin_r) // COLS
    ch = (height - top - bottom) // ROWS
    names = ("LEFT", "LEFT", "CENTER", "RIGHT", "RIGHT")
    for col, word in enumerate(names):
        if col in (0, 4):
            continue
        cx = margin_l + col * cw + cw / 2
        tw = draw.textlength(word, font=label)
        draw.text((cx - tw / 2, 40), word, fill=CREAM, font=_font(16, bold=True))

    blocked = choice["blocked"]
    disc = choice["disc"]
    gcol, grow = choice["goal_cell"]
    for row_near in range(ROWS):
        image_row = ROWS - 1 - row_near
        for col in range(COLS):
            x0 = margin_l + col * cw
            y0 = top + image_row * ch
            hot = (col, row_near) == disc
            wall = (col, row_near) in blocked
            if wall and not hot:
                fill = WALL_RGB
            elif hot:
                fill = (210, 40, 40) if choice["skill"] == "stand" else (232, 168, 48)
            else:
                fill = OPEN_RGB
            draw.rectangle([x0, y0, x0 + cw - 1, y0 + ch - 1], fill=fill, outline=INK, width=5)
            cx, cy = x0 + cw // 2, y0 + ch // 2
            if (col, row_near) == (gcol, grow) and not hot:
                rr = min(cw, ch) // 2 - 16
                draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=ORANGE, width=6)
            if hot:
                rr = min(cw, ch) // 2 - 14
                draw.ellipse(
                    [cx - rr, cy - rr, cx + rr, cy + rr],
                    fill=(250, 250, 250),
                    outline=INK,
                    width=6,
                )

    base_y = height - 24
    tip_y = height - bottom + 8
    mid_x = margin_l + CENTER * cw + cw // 2
    draw.polygon([(mid_x, tip_y), (mid_x - 20, base_y), (mid_x + 20, base_y)], fill=CREAM)
    return img


def make_duck() -> DuckRuntime:
    duck = DuckRuntime(goal_xy=GOAL, obstacles=OBSTACLES, path=())
    duck.chase_distance = 3.1
    duck.chase_elevation = -38.0
    return duck


def _contacts(duck: DuckRuntime) -> int:
    n = 0
    for i in range(duck.data.ncon):
        c = duck.data.contact[i]
        for gid in (int(c.geom1), int(c.geom2)):
            name = mujoco.mj_id2name(duck.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if name.startswith("obs_"):
                n += 1
                break
    return n


def sim(seconds: float) -> None:
    """Point robot at the tracked speed. No MuJoCo. Used to place the boxes."""
    x = y = yaw = 0.0
    prev = None
    t = 0.0
    dt = 0.5
    hits = 0
    trail = []
    while t < seconds:
        choice = plan(x, y, yaw, prev)
        skill = choice["skill"]
        prev = choice
        steps = int(dt / 0.02)
        for _ in range(steps):
            if skill == "walk_forward":
                dx, dy = choice["disc_xy"]
                bear = _wrap(math.atan2(dy - y, dx - x) - yaw)
                yaw = _wrap(yaw + max(-0.35, min(0.35, bear)) * 0.02)
                nx = x + 0.07 * math.cos(yaw) * 0.02
                ny = y + 0.07 * math.sin(yaw) * 0.02
                if _hits_box(nx, ny, BODY_R):
                    hits += 1
                else:
                    x, y = nx, ny
            elif skill == "turn_left":
                yaw = _wrap(yaw + 0.55 * 0.02)
            elif skill == "turn_right":
                yaw = _wrap(yaw - 0.55 * 0.02)
        trail.append((round(x, 2), round(y, 2), skill, round(choice["dist"], 2)))
        if t == 0.0 or abs((t + dt) - round(t + dt)) < 1e-9 and int(round(t + dt)) % 2 == 0:
            print(
                f"t={t:5.1f} xy=({x:+.2f},{y:+.2f}) yaw={yaw:+.2f} "
                f"aim={math.degrees(choice['aim']):+.0f} {skill:12} "
                f"post={choice['dist']:.2f} hits={hits}",
                flush=True,
            )
        if skill == "stand" and choice["dist"] <= STAND_M:
            print("stood", trail[-1], "hits", hits, flush=True)
            return
        t += dt
    closest = min(trail, key=lambda item: item[3])
    print("unfinished", trail[-1], "closest", closest, "hits", hits, flush=True)


def smoke(seconds: float) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    duck = make_duck()
    t = 0.0
    touches = 0
    min_z = 1.0
    prev = None
    next_cmd = 0.0
    next_print = 0.0
    info = duck.step()
    duck.set_skill("stand", "stop")
    while t < seconds:
        if t + 1e-9 >= next_cmd:
            choice = plan(info.x, info.y, info.yaw, prev)
            prev = choice
            if choice["skill"] == "walk_forward":
                dx, dy = choice["disc_xy"]
                bear = math.degrees(_wrap(math.atan2(dy - info.y, dx - info.x) - info.yaw))
                duck.set_walk_trim(bear)
            else:
                duck.set_walk_trim(None)
            duck.set_skill(choice["skill"], controller_speed(choice["skill"]))
            next_cmd += 0.5
        info = duck.step()
        touches += _contacts(duck)
        min_z = min(min_z, info.base_z)
        if t + 1e-9 >= next_print:
            dist = math.hypot(GOAL[0] - info.x, GOAL[1] - info.y)
            print(
                f"t={t:5.1f} xy=({info.x:+.2f},{info.y:+.2f}) z={info.base_z:.3f} "
                f"yaw={info.yaw:+.2f} post={dist:.2f} skill={info.skill} touch={touches}",
                flush=True,
            )
            next_print += 2.0
        if info.skill == "stand" and math.hypot(GOAL[0] - info.x, GOAL[1] - info.y) <= STAND_M and t > 2:
            break
        t += CONTROL_DT
    dist = math.hypot(GOAL[0] - info.x, GOAL[1] - info.y)
    print(
        f"done t={t:.1f} zmin={min_z:.3f} touches={touches} post={dist:.2f} skill={info.skill}",
        flush=True,
    )


async def run_clip(out_path: str, duration: float, caption: str) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    duck = make_duck()
    plate = Plate(duck)
    dep_big = mujoco.Renderer(duck.model, height=360, width=640)
    dep_big.enable_depth_rendering()
    dep_cam = mujoco.MjvCamera()
    silent = out_path + ".silent.mp4"
    log_path = os.path.splitext(out_path)[0] + ".jsonl"
    bed_path = os.path.splitext(out_path)[0] + ".wav"
    write_bed(bed_path, "drift", duration + 1.0)
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x1080", "-r", "30",
            "-i", "-",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            "-preset", "veryfast", "-movflags", "+faststart",
            silent,
        ],
        stdin=subprocess.PIPE,
    )
    opinion = {"skill": None, "conf": None, "probs": {}, "latency_ms": None}
    agrees = 0
    decisions = 0
    touches = 0
    prev = None
    t = 0.0
    acc = 0.0
    next_command = HOLD_S
    info = duck.step()
    duck.set_skill("stand", "stop")
    choice = plan(info.x, info.y, info.yaw, None)

    async with DecisionClient.from_env(
        timeout=60.0, max_image_side=512, image_quality=90, retries=1
    ) as client:
        while t < duration - 1e-9:
            if t + 1e-9 >= next_command:
                choice = plan(info.x, info.y, info.yaw, prev)
                grid = maze_grid(info.x, info.y, info.yaw, choice)
                depth = forward_depth_picture(duck, dep_big, dep_cam)
                state = {"view": "depth+grid"}
                try:
                    skill, conf, probs, lat = await _choose(client, state, MAZE_QUESTIONS, [depth, grid])
                except Exception as exc:
                    skill, conf, probs, lat = duck.skill, 0.0, {}, -1.0
                    print("decide failed", exc, flush=True)
                if skill not in ("turn_left", "turn_right", "walk_forward", "stand"):
                    skill = "stand"
                prev = choice
                if skill == "walk_forward":
                    dx, dy = choice["disc_xy"]
                    bear = math.degrees(_wrap(math.atan2(dy - info.y, dx - info.x) - info.yaw))
                    duck.set_walk_trim(bear)
                else:
                    duck.set_walk_trim(None)
                reason = duck.set_skill(skill, controller_speed(skill))
                hit = skill == choice["skill"]
                agrees += int(hit)
                decisions += 1
                opinion = {"skill": skill, "conf": conf, "probs": probs, "latency_ms": lat}
                log_f.write(json.dumps({
                    "t": round(t, 3),
                    "state": state,
                    "measured": {
                        "distance_m": round(choice["dist"], 2),
                        "bearing_deg": round(choice["bearing"], 1),
                    },
                    "rule_skill": choice["skill"],
                    "disc": list(choice["disc"]),
                    "model_skill": skill,
                    "agree": hit,
                    "confidence": round(conf, 3),
                    "latency_ms": round(lat, 1),
                    "applied": reason,
                    "executed_skill": duck.skill,
                    "walk_yaw_trim": round(duck.walk_yaw_trim, 3),
                    "contacts": touches,
                    "pose": {
                        "x": round(info.x, 3), "y": round(info.y, 3),
                        "z": round(info.base_z, 3), "yaw": round(info.yaw, 3),
                    },
                }) + "\n")
                log_f.flush()
                print(
                    f"t={t:5.1f} rule={choice['skill']:12} model={skill:12} "
                    f"d={choice['dist']:.2f} conf={conf:.2f} {lat:.0f}ms",
                    flush=True,
                )
                next_command += 0.5

            acc += FRAME_DT
            while acc >= CONTROL_DT - 1e-9:
                info = duck.step()
                acc -= CONTROL_DT
                touches += _contacts(duck)

            choice = plan(info.x, info.y, info.yaw, prev)
            dist = choice["dist"]
            bearing = choice["bearing"]
            if t < HOLD_S:
                mode_label = "hold"
                agree = "held"
            else:
                mode_label = "closed loop"
                agree = "match" if opinion["skill"] == choice["skill"] else f"rule {choice['skill'].replace('_', ' ')}"
            plate.mark(t, info.skill)
            chase, fwd, depth, _, _ = plate.grab()
            frame = plate.compose(
                t, chase, fwd, depth, info, caption, mode_label,
                opinion["skill"], opinion["probs"], opinion["conf"], opinion["latency_ms"],
                agree, dist, bearing,
                split_at=HOLD_S, split_left="hold", split_right="model",
                bar_duration=duration,
                goal_label="post",
                extra_rows=[
                    ("sent", "depth + grid"),
                    ("touch", str(touches)),
                ],
                subtitle="Violet cells are walls. The white disc is the next open step. Meters are not sent.",
                corner=maze_grid(info.x, info.y, info.yaw, choice),
                depth_view=depth_picture(depth),
            )
            proc.stdin.write(frame.tobytes())
            t += FRAME_DT

    summary = {
        "act": "summary",
        "bed": "drift",
        "decisions": decisions,
        "rule_matches": agrees,
        "agreement": None if not decisions else round(agrees / decisions, 3),
        "obstacle_contacts": touches,
        "final_m": round(math.hypot(GOAL[0] - info.x, GOAL[1] - info.y), 3),
        "final_pose": {
            "x": round(info.x, 3), "y": round(info.y, 3),
            "z": round(info.base_z, 3), "yaw": round(info.yaw, 3),
            "skill": info.skill,
        },
    }
    log_f.write(json.dumps(summary) + "\n")
    log_f.close()
    plate.close()
    dep_big.close()
    proc.stdin.close()
    code = proc.wait()
    if code != 0:
        raise SystemExit(f"ffmpeg exited {code}")
    mux = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", silent, "-i", bed_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
            "-shortest", "-movflags", "+faststart",
            out_path,
        ],
        check=False,
    )
    if mux.returncode != 0:
        raise SystemExit(f"mux exited {mux.returncode}")
    os.remove(silent)
    print("saved", out_path, summary, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="")
    parser.add_argument("--duration", type=float, default=72.0)
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument(
        "--caption",
        default="Six boxes. No dotted path. The disc steps into the next open cell.",
    )
    args = parser.parse_args()
    if args.sim:
        sim(args.seconds)
        return
    if args.smoke:
        smoke(args.seconds)
        return
    if not args.out:
        raise SystemExit("--out is required")
    asyncio.run(run_clip(args.out, args.duration, args.caption))


if __name__ == "__main__":
    main()
