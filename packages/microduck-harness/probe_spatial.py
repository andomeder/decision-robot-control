"""Static probe: can d1-3B read a color grid and a color depth camera?

No closed loop. Each case is one settled pose and one marker. The state sent
with the pictures has no distance and no bearing. The oracle is oracle_stand,
and skill_from_cell must match it before any call.

    MUJOCO_GL=egl .venv/bin/python packages/microduck-harness/probe_spatial.py
"""

from __future__ import annotations

import asyncio
import os
import sys

import mujoco
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "decision-client"))
sys.path.insert(0, os.path.dirname(__file__))

from decision_client import DecisionClient
from full_demo import DECISION_URL
from goal_clip import oracle_stand
from policy_runner import DuckRuntime, forward_camera, wider_forward_fov
from spatial import (
    BOTH_STAND,
    DEPTH_STAND,
    GRID_STAND,
    band_of,
    column_of,
    depth_picture,
    position_grid,
    skill_from_cell,
)

OUT = "/tmp/spatial_probe"

# World points. The settled duck is near the origin, facing +x.
CASES = (
    ("mid_center", (1.05, 0.08)),
    ("near_center", (0.55, -0.04)),
    ("close_center", (0.30, 0.02)),
    ("left", (0.75, 0.62)),
    ("right", (0.75, -0.62)),
    ("close_left", (0.28, 0.22)),
    ("far_center", (1.45, 0.10)),
    ("just_outside", (0.41, 0.00)),
    ("close_right", (0.25, -0.18)),
    ("wide_right", (0.40, -0.75)),
)


def _check_rules() -> None:
    for dist in (0.2, 0.4, 0.41, 0.55, 0.75, 0.76, 1.2, 1.5):
        for bearing in (-60, -18, -17, 0, 17, 18, 40):
            got = skill_from_cell(dist, bearing)
            want = oracle_stand(dist, bearing)
            if got != want:
                raise SystemExit(f"cell {got} != oracle {want} at {dist} m, {bearing} deg")


def _place(duck: DuckRuntime, body_id: int, xy: tuple[float, float]) -> None:
    duck.model.body_pos[body_id][0] = float(xy[0])
    duck.model.body_pos[body_id][1] = float(xy[1])
    duck.model.body_pos[body_id][2] = 0.0
    duck.set_goal(xy)
    duck.reset()


def _render(duck, rgb_r, dep_r, cam) -> tuple[Image.Image, Image.Image]:
    with wider_forward_fov(duck.model):
        forward_camera(duck, cam)
        rgb_r.update_scene(duck.data, camera=cam)
        dep_r.update_scene(duck.data, camera=cam)
        depth = dep_r.render().copy()
    dist, bearing = duck.goal_relative()
    return depth_picture(depth), position_grid(dist, bearing)


async def _ask(client, questions, images) -> dict:
    result = await client.decide({"view": "colors"}, questions, images=images)
    skill = result.choice("skill") or "stand"
    probs = (result.answers.get("skill") or {}).get("probabilities") or {}
    return {
        "skill": skill,
        "conf": round(float(result.confidence("skill") or 0.0), 3),
        "ms": round(result.latency_s * 1000.0, 1),
        "tokens": int(result.input_tokens),
        "probs": {k: round(float(v), 3) for k, v in probs.items()},
    }


async def main() -> None:
    _check_rules()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.makedirs(OUT, exist_ok=True)
    duck = DuckRuntime()
    body_id = mujoco.mj_name2id(duck.model, mujoco.mjtObj.mjOBJ_BODY, "goal_marker")
    if body_id < 0:
        raise SystemExit("goal_marker body missing")
    rgb_r = mujoco.Renderer(duck.model, height=360, width=640)
    dep_r = mujoco.Renderer(duck.model, height=360, width=640)
    dep_r.enable_depth_rendering()
    cam = mujoco.MjvCamera()
    rows = []
    async with DecisionClient.from_env(timeout=60.0, max_image_side=512, image_quality=90, retries=0) as client:
        for name, xy in CASES:
            _place(duck, body_id, xy)
            depth, grid = _render(duck, rgb_r, dep_r, cam)
            dist, bearing = duck.goal_relative()
            depth.save(os.path.join(OUT, f"{name}_depth.png"))
            grid.save(os.path.join(OUT, f"{name}_grid.png"))
            oracle = oracle_stand(dist, bearing)
            cell = f"{column_of(bearing)}/{band_of(dist)[0]}"
            grid_ans = await _ask(client, GRID_STAND, [grid])
            depth_ans = await _ask(client, DEPTH_STAND, [depth])
            both_ans = await _ask(client, BOTH_STAND, [depth, grid])
            row = {
                "name": name,
                "dist": round(dist, 3),
                "bearing": round(bearing, 1),
                "cell": cell,
                "oracle": oracle,
                "grid": grid_ans,
                "depth": depth_ans,
                "both": both_ans,
            }
            rows.append(row)
            print(
                f"{name:14} {dist:5.2f}m {bearing:+6.1f} {cell:14} {oracle:12} "
                f"grid {grid_ans['skill']:12} {grid_ans['conf']:.2f} {grid_ans['ms']:7.0f}ms "
                f"depth {depth_ans['skill']:12} {depth_ans['conf']:.2f} "
                f"both {both_ans['skill']:12} {both_ans['conf']:.2f} tok {both_ans['tokens']}",
                flush=True,
            )
    rgb_r.close()
    dep_r.close()
    for kind in ("grid", "depth", "both"):
        hit = sum(r[kind]["skill"] == r["oracle"] for r in rows)
        print(f"{kind} {hit}/{len(rows)}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
