"""Closed-loop agreement against the geometric oracle.

Runs the real policies. Each half-second d1-3B names a skill and the runtime
executes it at the controller speed. Prints the oracle match rate. This is
the same decision the video's second act records.

    MUJOCO_GL=egl .venv/bin/python packages/microduck-harness/chain.py --steps 16
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

import mujoco
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "decision-client"))
sys.path.insert(0, os.path.dirname(__file__))

from decision_client import DecisionClient
from full_demo import DECISION_URL, QUESTIONS, controller_speed, decision_state, oracle
from policy_runner import CONTROL_DT, DuckRuntime, forward_camera, wider_forward_fov


async def main(steps: int) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    duck = DuckRuntime()
    renderer = mujoco.Renderer(duck.model, height=180, width=320)
    cam = forward_camera(duck)
    hits = 0
    async with DecisionClient.from_env(timeout=30.0, max_image_side=320, retries=1) as client:
        pick_done = False
        was_pick = False
        for i in range(steps):
            for _ in range(int(0.5 / CONTROL_DT)):
                duck.step()
                if duck.mode == "pick":
                    was_pick = True
                elif was_pick:
                    pick_done = True
                    was_pick = False
            with wider_forward_fov(duck.model):
                forward_camera(duck, cam)
                renderer.update_scene(duck.data, camera=cam)
                rgb = renderer.render()
            dist, bearing = duck.goal_relative()
            state = decision_state(dist, bearing, pick_done)
            result = await client.decide(state, QUESTIONS, images=[Image.fromarray(rgb)])
            skill = result.choice("skill") or "stand"
            speed = controller_speed(skill)
            guess = oracle(dist, bearing, pick_done)
            reason = duck.set_skill(skill, speed)
            hit = skill == guess
            hits += int(hit)
            x, y, z, yaw = duck.pose()
            print(
                f"{i:02d}  model={skill:12s} oracle={guess:12s} {'ok' if hit else '  '}  "
                f"d={dist:.2f} b={bearing:+5.1f}  cmd={speed:6s} {reason:12s}  "
                f"x={x:+.2f} y={y:+.2f} z={z:.3f} yaw={yaw:+.2f}  {result.latency_s*1000:.0f}ms",
                flush=True,
            )
    renderer.close()
    print(f"agreement {hits}/{steps} = {hits / steps:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args()
    asyncio.run(main(args.steps))
