"""Save one chase frame and one forward frame from a standing MicroDuck.

    MUJOCO_GL=egl .venv/bin/python packages/microduck-harness/run.py
"""

from __future__ import annotations

import os
import sys

import mujoco
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from policy_runner import DuckRuntime, chase_camera, forward_camera


def main() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    duck = DuckRuntime()
    chase_r = mujoco.Renderer(duck.model, height=720, width=1280)
    fwd_r = mujoco.Renderer(duck.model, height=240, width=320)
    fwd_r.enable_depth_rendering()
    cam = chase_camera(duck)
    chase_r.update_scene(duck.data, camera=cam)
    rgb = chase_r.render()
    Image.fromarray(rgb).save("/tmp/duck_chase.png")
    fcam = forward_camera(duck)
    # Depth renderer can't also do color. A second color renderer for the aim.
    color = mujoco.Renderer(duck.model, height=240, width=320)
    color.update_scene(duck.data, camera=fcam)
    Image.fromarray(color.render()).save("/tmp/duck_forward.png")
    fwd_r.update_scene(duck.data, camera=fcam)
    depth = fwd_r.render()
    x, y, z, yaw = duck.pose()
    dist, bearing = duck.goal_relative()
    print(f"pose x={x:.3f} y={y:.3f} z={z:.3f} yaw={yaw:.3f}")
    print(f"goal dist={dist:.3f} bearing={bearing:.1f} deg")
    print(f"depth min={float(depth.min()):.3f} max={float(depth.max()):.3f}")
    print("wrote /tmp/duck_chase.png /tmp/duck_forward.png")
    chase_r.close()
    fwd_r.close()
    color.close()


if __name__ == "__main__":
    main()
