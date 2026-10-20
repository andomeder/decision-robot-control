"""Slalom clip. Boxes block the straight line. Dots mark the path around them.

d1-3B still names one skill from distance and bearing. The bearing is to the
active corner, then to the orange post. The path is fixed. The model does not
plan it. Distances on the plate are for the viewer.

    MUJOCO_GL=egl .venv/bin/python packages/microduck-harness/course_clip.py \\
        --out /tmp/goal_course.mp4
    MUJOCO_GL=egl .venv/bin/python packages/microduck-harness/course_clip.py --smoke
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
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "decision-client"))
sys.path.insert(0, os.path.dirname(__file__))

from decision_client import DecisionClient
from full_demo import DECISION_URL, FRAME_DT, Plate, W, controller_speed
from goal_clip import HOLD_S, STAND_QUESTIONS, _choose, oracle_stand, write_bed
from policy_runner import CONTROL_DT, DuckRuntime

# Half-sizes are meters. The clay box sits on the straight line from the
# origin to the post. The slate wall makes the upper side a lane.
OBSTACLES = (
    {"pos": (0.85, -0.08, 0.11), "size": (0.30, 0.26, 0.11), "rgba": (0.45, 0.30, 0.24, 1.0)},
    {"pos": (0.95, 0.98, 0.11), "size": (0.72, 0.08, 0.11), "rgba": (0.26, 0.34, 0.44, 1.0)},
)
CORNERS = ((0.22, 0.64), (1.48, 0.58))
GOAL = (1.88, 0.00)
TARGETS = CORNERS + (GOAL,)
TRAIL = ((0.0, 0.0),) + CORNERS + (GOAL,)
ADVANCE_M = 0.30

# Stand is the stop only on the last leg. On a corner, stopping would park
# the duck in the lane, so that choice is not offered until the orange post.
MID_QUESTIONS = {
    "skill": {
        "type": "choice",
        "instructions": (
            "Choose the one skill whose criterion matches the state. "
            "bearing_deg is positive when the mark is to the robot's left. "
            "This mark is a corner. Walk once you are aligned."
        ),
        "criteria": {
            "turn_left": "bearing_deg >= 18",
            "turn_right": "bearing_deg <= -18",
            "walk_forward": "bearing_deg > -18 and bearing_deg < 18",
        },
    },
}


def _hypot(a, b) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def path_left(x: float, y: float, index: int) -> float:
    left = _hypot((x, y), TARGETS[index])
    for a, b in zip(TARGETS[index:], TARGETS[index + 1 :]):
        left += _hypot(a, b)
    return left


def make_duck() -> DuckRuntime:
    duck = DuckRuntime(
        goal_xy=TARGETS[0],
        marker_xy=GOAL,
        obstacles=OBSTACLES,
        path=TRAIL,
    )
    duck.chase_distance = 2.35
    duck.chase_elevation = -28.0
    duck.paint_waypoint(0)
    return duck


def oracle_mid(bearing: float) -> str:
    if bearing >= 18.0:
        return "turn_left"
    if bearing <= -18.0:
        return "turn_right"
    return "walk_forward"


VISION_MID = {
    "skill": {
        "type": "choice",
        "instructions": (
            "Look only at the forward camera. No distance or bearing is provided. "
            "Follow the nearest orange mark. The brown box and the slate wall are solid."
        ),
        "criteria": {
            "turn_left": "the nearest orange mark is clearly left of center",
            "turn_right": "the nearest orange mark is clearly right of center",
            "walk_forward": "the nearest orange mark is near the center",
        },
    },
}
VISION_LAST = {
    "skill": {
        "type": "choice",
        "instructions": (
            "Look only at the forward camera. No distance or bearing is provided. "
            "The orange post is the final mark. stand means stop and hold."
        ),
        "criteria": {
            "turn_left": "the orange post is clearly left of center",
            "turn_right": "the orange post is clearly right of center",
            "walk_forward": "the orange post is near the center and still far",
            "stand": "the orange post is near the center and close enough to stop",
        },
    },
}


def questions_for(index: int, vision: bool = False) -> dict:
    if vision:
        return VISION_LAST if index >= len(TARGETS) - 1 else VISION_MID
    if index >= len(TARGETS) - 1:
        return STAND_QUESTIONS
    return MID_QUESTIONS


def guess_for(index: int, dist: float, bearing: float) -> str:
    if index >= len(TARGETS) - 1:
        return oracle_stand(dist, bearing)
    return oracle_mid(bearing)


def _advance(duck: DuckRuntime, index: int) -> int:
    dist, _ = duck.goal_relative()
    if index < len(TARGETS) - 1 and dist <= ADVANCE_M:
        index += 1
        duck.set_goal(TARGETS[index])
        duck.paint_waypoint(index)
    return index


async def run_clip(out_path: str, duration: float, caption: str, bed: str, vision: bool = False) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    duck = make_duck()
    plate = Plate(duck)
    silent = out_path + ".silent.mp4"
    log_path = os.path.splitext(out_path)[0] + ".jsonl"
    bed_path = os.path.splitext(out_path)[0] + ".wav"
    write_bed(bed_path, bed, duration + 1.0)
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
    index = 0
    t = 0.0
    acc = 0.0
    next_command = HOLD_S
    info = duck.step()
    duck.set_skill("stand", "stop")

    async with DecisionClient.from_env(timeout=30.0, max_image_side=320, retries=1) as client:
        while t < duration - 1e-9:
            index = _advance(duck, index)
            if t + 1e-9 >= next_command:
                _, fwd, _, _, grid = plate.grab()
                dist, bearing = duck.goal_relative()
                measured = {"distance_m": round(dist, 2), "bearing_deg": round(bearing, 1)}
                state = {"camera": "forward"} if vision else measured
                if not vision:
                    duck.set_walk_trim(bearing)
                try:
                    skill, conf, probs, lat = await _choose(
                        client, state, questions_for(index, vision), Image.fromarray(fwd)
                    )
                except Exception as exc:
                    skill, conf, probs, lat = duck.skill, 0.0, {}, -1.0
                    print("decide failed", exc, flush=True)
                speed = controller_speed(skill)
                reason = duck.set_skill(skill, speed)
                guess = guess_for(index, dist, bearing)
                hit = skill == guess
                agrees += int(hit)
                decisions += 1
                opinion = {"skill": skill, "conf": conf, "probs": probs, "latency_ms": lat}
                log_f.write(json.dumps({
                    "t": round(t, 3),
                    "task": "course",
                    "leg": index + 1,
                    "state": state,
                    "measured": measured,
                    "vision": vision,
                    "path_left_m": round(path_left(info.x, info.y, index), 2),
                    "final_m": round(_hypot((info.x, info.y), GOAL), 2),
                    "model_skill": skill,
                    "applied_speed": speed,
                    "speed_source": "controller",
                    "applied": reason,
                    "executed_skill": duck.skill,
                    "oracle": guess,
                    "agree": hit,
                    "confidence": round(conf, 3),
                    "latency_ms": round(lat, 1),
                    "depth_grid_min": grid,
                    "pose": {
                        "x": round(info.x, 3), "y": round(info.y, 3),
                        "z": round(info.base_z, 3), "yaw": round(info.yaw, 3),
                    },
                }) + "\n")
                log_f.flush()
                next_command += 0.5

            acc += FRAME_DT
            while acc >= CONTROL_DT - 1e-9:
                info = duck.step()
                acc -= CONTROL_DT
                touches += _obstacle_contacts(duck)

            index = _advance(duck, index)
            dist, bearing = duck.goal_relative()
            if t < HOLD_S:
                mode_label = "hold"
                agree = "held"
            else:
                mode_label = "closed loop"
                guess = guess_for(index, dist, bearing)
                agree = "match" if opinion["skill"] == guess else f"oracle {guess.replace('_', ' ')}"
            extra = [
                ("leg", f"{index + 1}/{len(TARGETS)}"),
                ("path", f"{path_left(info.x, info.y, index):.2f} m"),
                ("final", f"{_hypot((info.x, info.y), GOAL):.2f} m"),
            ]
            plate.mark(t, info.skill)
            chase, fwd, depth, _, _ = plate.grab()
            frame = plate.compose(
                t, chase, fwd, depth, info, caption, mode_label,
                opinion["skill"], opinion["probs"], opinion["conf"], opinion["latency_ms"],
                agree, dist, bearing,
                split_at=HOLD_S, split_left="hold", split_right="model",
                bar_duration=duration,
                goal_label="next",
                extra_rows=extra,
            )
            proc.stdin.write(frame.tobytes())
            t += FRAME_DT

    summary = {
        "act": "summary",
        "task": "course",
        "bed": bed,
        "decisions": decisions,
        "oracle_matches": agrees,
        "agreement": None if not decisions else round(agrees / decisions, 3),
        "obstacle_contacts": touches,
        "final_pose": {
            "x": round(info.x, 3), "y": round(info.y, 3),
            "z": round(info.base_z, 3), "yaw": round(info.yaw, 3),
            "skill": info.skill,
        },
        "final_m": round(_hypot((info.x, info.y), GOAL), 3),
        "leg": index + 1,
    }
    log_f.write(json.dumps(summary) + "\n")
    log_f.close()
    plate.close()
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


def _obstacle_contacts(duck: DuckRuntime) -> int:
    n = 0
    for i in range(duck.data.ncon):
        c = duck.data.contact[i]
        names = []
        for gid in (int(c.geom1), int(c.geom2)):
            name = mujoco_name(duck, gid)
            names.append(name or "")
        if any(name.startswith("obs_") for name in names):
            n += 1
    return n


def mujoco_name(duck: DuckRuntime, gid: int) -> str | None:
    return mujoco.mj_id2name(duck.model, mujoco.mjtObj.mjOBJ_GEOM, gid)


def smoke(seconds: float) -> None:
    """Oracle drives the course. Prints pose, and whether a box was touched."""
    duck = make_duck()
    index = 0
    t = 0.0
    touches = 0
    duck.set_skill("stand", "stop")
    next_print = 0.0
    min_z = 1.0
    while t < seconds:
        index = _advance(duck, index)
        dist, bearing = duck.goal_relative()
        skill = guess_for(index, dist, bearing)
        duck.set_skill(skill, controller_speed(skill))
        info = duck.step()
        touches += _obstacle_contacts(duck)
        min_z = min(min_z, info.base_z)
        if t + 1e-9 >= next_print:
            print(
                f"t={t:5.1f} leg={index + 1} xy=({info.x:+.2f},{info.y:+.2f}) "
                f"z={info.base_z:.3f} yaw={info.yaw:+.2f} next={dist:.2f} "
                f"bear={bearing:+.0f} left={path_left(info.x, info.y, index):.2f} "
                f"final={_hypot((info.x, info.y), GOAL):.2f} skill={info.skill} "
                f"touch={touches}",
                flush=True,
            )
            next_print += 2.0
        t += CONTROL_DT
    print(
        f"done zmin={min_z:.3f} touches={touches} leg={index + 1}/{len(TARGETS)} "
        f"final={_hypot((info.x, info.y), GOAL):.2f} skill={info.skill}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="")
    parser.add_argument("--duration", type=float, default=64.0)
    parser.add_argument("--bed", choices=("funky", "chill", "bounce", "drift"), default="funky")
    parser.add_argument("--vision", action="store_true", help="hide distance and bearing; no heading trim")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-seconds", type=float, default=56.0)
    parser.add_argument(
        "--caption",
        default="Boxes block the straight line. The dots go around them. d1-3B names the skill to the next mark.",
    )
    args = parser.parse_args()
    if args.smoke:
        smoke(args.smoke_seconds)
        return
    if not args.out:
        raise SystemExit("--out is required")
    asyncio.run(run_clip(args.out, args.duration, args.caption, args.bed, vision=args.vision))


if __name__ == "__main__":
    main()
