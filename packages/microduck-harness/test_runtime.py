"""Smoke test: the duck stands, walks forward, and the camera sees it.

Run from the harness directory with the project venv. Exits non-zero on failure.
"""

from __future__ import annotations

import os
import sys

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from policy_runner import CONTROL_DT, OBS_DIM, DuckRuntime


def main() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    duck = DuckRuntime()
    obs = duck._observe()
    assert obs.shape == (1, OBS_DIM), obs.shape
    assert abs(float(obs.sum()) ) < 1e6

    zs = []
    for _ in range(int(2.0 / CONTROL_DT)):
        info = duck.step()
        zs.append(info.base_z)
    zmin, zmax = min(zs), max(zs)
    print(f"stand 2s  z=[{zmin:.3f}, {zmax:.3f}]  end=({info.x:.3f}, {info.y:.3f}) upright={info.upright}")
    if zmin < 0.08 or zmax > 0.28 or not info.upright:
        raise SystemExit(f"stand failed: z=[{zmin:.3f}, {zmax:.3f}] upright={info.upright}")

    duck.reset()
    x0, y0, _, _ = duck.pose()
    duck.set_skill("walk_forward", "medium")
    for _ in range(int(3.0 / CONTROL_DT)):
        info = duck.step()
    dx, dy = info.x - x0, info.y - y0
    dist = (dx * dx + dy * dy) ** 0.5
    print(f"walk 3s  d=({dx:.3f}, {dy:.3f}) dist={dist:.3f} z={info.base_z:.3f} upright={info.upright}")
    if dist < 0.20 or not info.upright or info.base_z < 0.08:
        raise SystemExit(f"walk failed: dist={dist:.3f} z={info.base_z:.3f}")
    # +x is forward at yaw 0. The mission bearing math depends on this.
    if dx < 0.15:
        raise SystemExit(f"forward is not +x: dx={dx:.3f} dy={dy:.3f}")

    duck.reset()
    duck.set_skill("turn_left", "medium")
    _, _, _, yaw0 = duck.pose()
    for _ in range(int(2.0 / CONTROL_DT)):
        info = duck.step()
    dyaw = info.yaw - yaw0
    print(f"turn left 2s  dyaw={dyaw:.3f} rad  z={info.base_z:.3f}")
    if dyaw < 0.25 or not info.upright:
        raise SystemExit(f"turn failed: dyaw={dyaw:.3f}")

    renderer = mujoco.Renderer(duck.model, height=720, width=1280)
    from policy_runner import track_camera

    cam = track_camera(duck)
    renderer.update_scene(duck.data, camera=cam)
    rgb = renderer.render()
    mean = float(rgb.mean())
    frac_dark = float((rgb.max(axis=2) < 12).mean())
    out = "/tmp/duck_stand.png"
    Image.fromarray(rgb).save(out)
    print(f"render mean={mean:.1f} dark={frac_dark:.2f} saved {out} shape={rgb.shape}")
    if mean < 20 or frac_dark > 0.85:
        raise SystemExit(f"render looks empty: mean={mean:.1f} dark={frac_dark:.2f}")

    head = mujoco.Renderer(duck.model, height=240, width=320)
    head.update_scene(duck.data, camera="head_camera")
    head_rgb = head.render()
    Image.fromarray(head_rgb).save("/tmp/duck_head.png")
    print(f"head mean={float(head_rgb.mean()):.1f}")
    print("ok")


if __name__ == "__main__":
    main()
