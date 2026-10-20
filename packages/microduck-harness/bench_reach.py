"""Twenty reach-and-stand episodes, headless.

Success means the duck stayed upright (base z never under 0.08 m) and finished
within 0.45 m of the marker, skill stand, absolute bearing under 25 degrees.

Three conditions, same goals:
  vision   d1-3B sees the forward camera. Distance and bearing are not sent.
           A walk gets the controller heading trim.
  oracle   the numeric rule names the skill. Same trim.
  velstand always walk, with the trim, and stand when the rule would stand.

    MUJOCO_GL=egl .venv/bin/python packages/microduck-harness/bench_reach.py
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "decision-client"))
sys.path.insert(0, os.path.dirname(__file__))

from decision_client import DecisionClient
from full_demo import DECISION_URL, controller_speed
from goal_clip import STAND_QUESTIONS, oracle_stand
from policy_runner import CONTROL_DT, DuckRuntime, forward_camera, wider_forward_fov

TIMEOUT_S = 28.0
DECIDE_EVERY = 0.5

VISION_STAND = {
    "skill": {
        "type": "choice",
        "instructions": (
            "Look only at the forward camera. The orange marker is the goal. "
            "No distance or bearing is provided. stand means stop and hold."
        ),
        "criteria": {
            "turn_left": "the orange marker is clearly left of center",
            "turn_right": "the orange marker is clearly right of center",
            "walk_forward": "the orange marker is near the center and still far",
            "stand": "the orange marker is near the center and close enough to stop",
        },
    },
}


def goals() -> list[tuple[float, float]]:
    out = []
    for i in range(20):
        ang = math.radians(-75.0 + i * (150.0 / 19.0))
        radius = 0.82 + (i % 4) * 0.06
        out.append((round(radius * math.cos(ang), 3), round(radius * math.sin(ang), 3)))
    return out


def success(min_z: float, dist: float, bearing: float, skill: str) -> bool:
    return min_z >= 0.08 and dist <= 0.45 and abs(bearing) < 25.0 and skill == "stand"


async def one_episode(kind: str, goal, client) -> dict:
    import mujoco

    duck = DuckRuntime(goal_xy=goal)
    renderer = None
    cam = None
    if kind == "vision":
        renderer = mujoco.Renderer(duck.model, height=180, width=320)
        cam = mujoco.MjvCamera()
    t = 0.0
    next_cmd = 0.0
    min_z = 1.0
    lats = []
    agrees = 0
    decisions = 0
    info = duck.step()
    while t < TIMEOUT_S:
        dist, bearing = duck.goal_relative()
        if t + 1e-9 >= next_cmd:
            guess = oracle_stand(dist, bearing)
            if kind == "velstand":
                skill = "stand" if guess == "stand" else "walk_forward"
                conf, lat = None, None
            elif kind == "oracle":
                skill, conf, lat = guess, None, None
            else:
                forward_camera(duck, cam)
                with wider_forward_fov(duck.model):
                    renderer.update_scene(duck.data, camera=cam)
                    frame = renderer.render()
                result = await client.decide(
                    {"camera": "forward"}, VISION_STAND, images=[Image.fromarray(frame)]
                )
                skill = result.choice("skill") or "stand"
                if skill not in ("turn_left", "turn_right", "walk_forward", "stand"):
                    skill = "stand"
                conf = float(result.confidence("skill") or 0.0)
                lat = result.latency_s * 1000.0
                lats.append(lat)
            duck.set_walk_trim(bearing)
            duck.set_skill(skill, controller_speed(skill))
            agrees += int(skill == guess)
            decisions += 1
            next_cmd += DECIDE_EVERY
        info = duck.step()
        min_z = min(min_z, info.base_z)
        t += CONTROL_DT
        if info.skill == "stand" and dist <= 0.45 and t > 1.0:
            # Hold briefly so a drive-by stand is not counted early.
            if t >= next_cmd:
                break
    dist, bearing = duck.goal_relative()
    ok = success(min_z, dist, bearing, info.skill)
    return {
        "goal": list(goal),
        "success": ok,
        "dist": round(dist, 3),
        "bearing": round(bearing, 1),
        "zmin": round(min_z, 3),
        "skill": info.skill,
        "t": round(t, 2),
        "decisions": decisions,
        "agree": None if not decisions else round(agrees / decisions, 3),
        "latency_ms_mean": None if not lats else round(sum(lats) / len(lats), 1),
    }
    if renderer is not None:
        renderer.close()


async def main() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    rows = []
    async with DecisionClient.from_env(timeout=30.0, max_image_side=320, retries=1) as client:
        for kind in ("oracle", "velstand", "vision"):
            for i, goal in enumerate(goals()):
                row = await one_episode(kind, goal, client if kind == "vision" else None)
                row["kind"] = kind
                row["i"] = i
                rows.append(row)
                print(
                    f"{kind} {i:02d} ok={int(row['success'])} dist={row['dist']:.2f} "
                    f"zmin={row['zmin']:.3f} skill={row['skill']} t={row['t']:.1f}",
                    flush=True,
                )
    summary = {}
    for kind in ("oracle", "velstand", "vision"):
        subset = [r for r in rows if r["kind"] == kind]
        n = len(subset)
        k = sum(1 for r in subset if r["success"])
        lats = [r["latency_ms_mean"] for r in subset if r["latency_ms_mean"] is not None]
        summary[kind] = {
            "n": n,
            "success": k,
            "rate": round(k / n, 3),
            "mean_dist": round(sum(r["dist"] for r in subset) / n, 3),
            "falls": sum(1 for r in subset if r["zmin"] < 0.08),
            "latency_ms_mean": None if not lats else round(sum(lats) / len(lats), 1),
        }
    out = {
        "timeout_s": TIMEOUT_S,
        "success_rule": "zmin>=0.08 and dist<=0.45 and |bearing|<25 and skill==stand",
        "summary": summary,
        "episodes": rows,
    }
    path = "/tmp/bench_reach.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("summary", json.dumps(summary), flush=True)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
