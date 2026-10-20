"""Short closed-loop clips with a different marker, and an original music bed.

The catalog plate stays in full_demo.py. This one holds for two seconds, then
lets d1-3B drive. `pick` is turn, walk, ground pick, sit. `stand` is turn,
walk, and stand still on the marker. stand is the stop in that task. The
pick task has no stand option.

    MUJOCO_GL=egl .venv/bin/python packages/microduck-harness/goal_clip.py \\
        --task pick --goal 0.80 -0.62 --bed funky --out /tmp/goal_right.mp4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import subprocess
import sys
import wave

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "decision-client"))
sys.path.insert(0, os.path.dirname(__file__))

from decision_client import DecisionClient
from full_demo import DECISION_URL, FRAME_DT, Plate, QUESTIONS, W, controller_speed, oracle
from policy_runner import ALL_SKILLS, CONTROL_DT, DuckRuntime

HOLD_S = 2.0
SR = 44100

STAND_QUESTIONS = {
    "skill": {
        "type": "choice",
        "instructions": (
            "Choose the one skill whose criterion matches the state. "
            "bearing_deg is positive when the orange marker is to the robot's left. "
            "stand means stop and hold position."
        ),
        "criteria": {
            "turn_left": "bearing_deg >= 18",
            "turn_right": "bearing_deg <= -18",
            "walk_forward": "distance_m > 0.40 and bearing_deg > -18 and bearing_deg < 18",
            "stand": "distance_m <= 0.40 and bearing_deg > -18 and bearing_deg < 18",
        },
    },
}


def oracle_stand(distance: float, bearing: float) -> str:
    if bearing >= 18.0:
        return "turn_left"
    if bearing <= -18.0:
        return "turn_right"
    if distance <= 0.40:
        return "stand"
    return "walk_forward"


def _env(n: int, seconds: float) -> np.ndarray:
    t = np.arange(n) / SR
    return np.exp(-t / max(seconds, 1e-3))


def _kick(n: int) -> np.ndarray:
    t = np.arange(n) / SR
    return np.sin(2 * math.pi * (90 - 60 * np.clip(t / 0.08, 0, 1)) * t) * _env(n, 0.12)


def _hat(n: int) -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.uniform(-1, 1, n) * _env(n, 0.03)


def _clap(n: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    noise = rng.uniform(-1, 1, n)
    t = np.arange(n) / SR
    tone = np.sin(2 * math.pi * 180 * t)
    return (0.7 * noise + 0.3 * tone) * _env(n, 0.08)


def _tone(freq: float, n: int, decay: float, wave_kind: str) -> np.ndarray:
    t = np.arange(n) / SR
    phase = 2 * math.pi * freq * t
    if wave_kind == "square":
        body = np.sign(np.sin(phase))
    else:
        body = np.sin(phase)
    return body * _env(n, decay)


def _pluck(freq: float, n: int, decay: float) -> np.ndarray:
    """Soft mallet. Sine plus a short octave, no square wave."""
    t = np.arange(n) / SR
    body = np.sin(2 * math.pi * freq * t) * np.exp(-t / max(decay, 1e-3))
    over = 0.28 * np.sin(2 * math.pi * freq * 2.003 * t) * np.exp(-t / max(decay * 0.4, 1e-3))
    return body + over


def _wood(n: int) -> np.ndarray:
    """Hollow tick. Shorter and higher than the clap."""
    rng = np.random.default_rng(11)
    t = np.arange(n) / SR
    noise = rng.uniform(-1, 1, n)
    tone = np.sin(2 * math.pi * 880 * t)
    return (0.4 * noise + tone) * np.exp(-t / 0.018)


def _place(buf: np.ndarray, hit: np.ndarray, at: float, gain: float) -> None:
    i = int(at * SR)
    j = min(len(buf), i + len(hit))
    if i >= len(buf) or j <= i:
        return
    buf[i:j] += hit[: j - i] * gain


def write_bed(path: str, kind: str, seconds: float) -> None:
    n = int(seconds * SR)
    buf = np.zeros(n, dtype=np.float64)
    if kind == "funky":
        bpm = 98.0
        notes = [55.0, 65.41, 73.42, 82.41, 98.0]  # A2 C3 D3 E3 G3
        beat = 60.0 / bpm
        step = 0
        at = 0.0
        while at < seconds:
            _place(buf, _kick(int(0.2 * SR)), at, 0.55)
            if step % 2 == 1:
                _place(buf, _clap(int(0.15 * SR)), at, 0.22)
            if step % 2 == 0:
                _place(buf, _hat(int(0.05 * SR)), at + beat * 0.5, 0.12)
            note = notes[(step * 3) % len(notes)]
            if step % 8 in (4, 5):
                note *= 2
            _place(buf, _tone(note, int(0.22 * SR), 0.16, "square"), at, 0.16)
            at += beat
            step += 1
    elif kind == "chill":
        bpm = 72.0
        beat = 60.0 / bpm
        roots = [220.0, 261.63, 196.0, 246.94]
        at = 0.0
        bar = 0
        while at < seconds:
            root = roots[bar % len(roots)]
            dur = int(beat * 4 * SR)
            pad = (
                _tone(root, dur, 1.6, "sine")
                + 0.55 * _tone(root * 1.5, dur, 1.6, "sine")
                + 0.35 * _tone(root * 2, dur, 1.8, "sine")
            )
            _place(buf, pad, at, 0.12)
            for k in range(4):
                _place(buf, _kick(int(0.25 * SR)), at + k * beat, 0.18 if k % 2 == 0 else 0.08)
            at += beat * 4
            bar += 1
    elif kind == "bounce":
        bpm = 114.0
        beat = 60.0 / bpm
        notes = [261.63, 329.63, 392.0, 440.0, 523.25]
        at = 0.0
        step = 0
        while at < seconds:
            if step % 4 in (0, 2):
                _place(buf, _kick(int(0.12 * SR)), at, 0.4)
            else:
                _place(buf, _hat(int(0.04 * SR)), at, 0.16)
            note = notes[step % len(notes)]
            _place(buf, _tone(note, int(0.12 * SR), 0.08, "sine"), at, 0.2)
            at += beat * 0.5
            step += 1
    elif kind == "drift":
        # 6/8 mallet phrase. No kick, no clap, no square bass.
        bpm = 80.0
        eighth = 60.0 / bpm / 2.0
        phrase = (
            293.66, 0.0, 349.23, 392.0, 0.0, 440.0,
            392.0, 349.23, 293.66, 0.0, 261.63, 293.66,
            349.23, 0.0, 440.0, 523.25, 440.0, 349.23,
            329.63, 293.66, 0.0, 261.63, 0.0, 220.0,
        )
        i = 0
        at = 0.0
        while at < seconds:
            if i % 6 == 0:
                _place(buf, _pluck(55.0, int(0.7 * SR), 0.40), at, 0.62)
            elif i % 6 == 3:
                _place(buf, _pluck(82.41, int(0.45 * SR), 0.30), at, 0.26)
            if i % 2 == 1:
                _place(buf, _wood(int(0.04 * SR)), at, 0.16)
            note = phrase[i % len(phrase)]
            if note:
                _place(buf, _pluck(note, int(0.40 * SR), 0.18), at, 0.30)
            if i % 12 == 0:
                air_n = int(eighth * 12 * SR)
                rng = np.random.default_rng(1000 + i)
                env = np.sin(np.linspace(0, math.pi, air_n))
                _place(buf, rng.uniform(-1, 1, air_n) * env, at, 0.035)
            at += eighth
            i += 1
    else:
        raise SystemExit(f"unknown bed {kind}")
    peak = float(np.max(np.abs(buf))) or 1.0
    pcm = np.clip(buf / peak * 0.32, -1, 1)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((pcm * 32767).astype(np.int16).tobytes())


def task_spec(task: str):
    if task == "pick":
        return QUESTIONS, lambda d, b, done: oracle(d, b, done), True
    if task == "stand":
        return STAND_QUESTIONS, lambda d, b, done: oracle_stand(d, b), False
    raise SystemExit(f"unknown task {task}")


async def _choose(client, state, questions, images):
    if not isinstance(images, list):
        images = [images]
    result = await client.decide(state, questions, images=images)
    skill = result.choice("skill") or "stand"
    if skill not in ALL_SKILLS:
        skill = "stand"
    probs = (result.answers.get("skill") or {}).get("probabilities") or {}
    return skill, float(result.confidence("skill") or 0.0), probs, result.latency_s * 1000.0


async def run_clip(
    goal,
    task: str,
    out_path: str,
    bed: str,
    duration: float,
    caption: str,
    vision: bool = False,
    spatial: bool = False,
) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    questions, guess_of, use_pick = task_spec(task)
    if vision and spatial:
        raise SystemExit("vision and spatial are different inputs")
    if vision:
        if task != "stand":
            raise SystemExit("vision clips in this file are stand-only")
        from bench_reach import VISION_STAND
        questions = VISION_STAND
    if spatial:
        if task != "stand":
            raise SystemExit("spatial clips in this file are stand-only")
        import mujoco
        from spatial import BOTH_STAND
        questions = BOTH_STAND
    duck = DuckRuntime(goal_xy=goal)
    dep_big = None
    dep_cam = None
    if spatial:
        dep_big = mujoco.Renderer(duck.model, height=360, width=640)
        dep_big.enable_depth_rendering()
        dep_cam = mujoco.MjvCamera()
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
    pick_done = False
    was_pick = False
    agrees = 0
    decisions = 0
    t = 0.0
    acc = 0.0
    next_command = HOLD_S
    info = duck.step()
    duck.set_skill("stand", "stop")

    side = 512 if spatial else 320
    quality = 90 if spatial else 80
    async with DecisionClient.from_env(
        timeout=30.0, max_image_side=side, image_quality=quality, retries=1
    ) as client:
        while t < duration - 1e-9:
            if t + 1e-9 >= next_command:
                _, fwd, depth_small, _, grid = plate.grab()
                dist, bearing = duck.goal_relative()
                images = Image.fromarray(fwd)
                if spatial:
                    from spatial import forward_depth_picture, position_grid
                    images = [
                        forward_depth_picture(duck, dep_big, dep_cam),
                        position_grid(dist, bearing),
                    ]
                    state = {"view": "depth+grid"}
                elif vision:
                    state = {"camera": "forward"}
                elif use_pick:
                    state = {
                        "distance_m": round(dist, 2),
                        "bearing_deg": round(bearing, 1),
                        "pick_done": pick_done,
                    }
                else:
                    state = {
                        "distance_m": round(dist, 2),
                        "bearing_deg": round(bearing, 1),
                    }
                try:
                    skill, conf, probs, lat = await _choose(client, state, questions, images)
                except Exception as exc:
                    skill, conf, probs, lat = duck.skill, 0.0, {}, -1.0
                    print("decide failed", exc, flush=True)
                speed = controller_speed(skill)
                reason = duck.set_skill(skill, speed)
                guess = guess_of(dist, bearing, pick_done)
                hit = skill == guess
                agrees += int(hit)
                decisions += 1
                opinion = {"skill": skill, "conf": conf, "probs": probs, "latency_ms": lat}
                log_f.write(json.dumps({
                    "t": round(t, 3),
                    "task": task,
                    "goal": [round(goal[0], 3), round(goal[1], 3)],
                    "state": state,
                    "measured": {
                        "distance_m": round(dist, 2),
                        "bearing_deg": round(bearing, 1),
                    },
                    "spatial": spatial,
                    "walk_yaw_trim": round(duck.walk_yaw_trim, 3),
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
                if spatial:
                    _, bearing_now = duck.goal_relative()
                    duck.set_walk_trim(bearing_now)
                info = duck.step()
                acc -= CONTROL_DT
                if use_pick:
                    if duck.mode == "pick":
                        was_pick = True
                    elif was_pick:
                        pick_done = True
                        was_pick = False

            dist, bearing = duck.goal_relative()
            if t < HOLD_S:
                cap = caption
                mode_label = "hold"
                agree = "held"
            else:
                cap = caption
                mode_label = "closed loop"
                guess = guess_of(dist, bearing, pick_done)
                agree = "match" if opinion["skill"] == guess else f"oracle {guess.replace('_', ' ')}"
            plate.mark(t, info.skill)
            chase, fwd, depth, _, _ = plate.grab()
            corner = None
            depth_view = None
            if spatial:
                from spatial import depth_picture, position_grid
                corner = position_grid(dist, bearing)
                depth_view = depth_picture(depth)
            if spatial:
                sent_rows = [("sent", "depth + grid")]
                sub = "Colors for range. Meters and degrees are not sent. The grid is the marker pose."
            elif vision:
                sent_rows = [("sent", "camera")]
                sub = "Camera only. Distance and bearing are not sent."
            else:
                sent_rows = None
                sub = None
            frame = plate.compose(
                t, chase, fwd, depth, info, cap, mode_label,
                opinion["skill"], opinion["probs"], opinion["conf"], opinion["latency_ms"],
                agree, dist, bearing,
                split_at=HOLD_S, split_left="hold", split_right="model",
                bar_duration=duration,
                extra_rows=sent_rows,
                subtitle=sub,
                corner=corner,
                depth_view=depth_view,
            )
            proc.stdin.write(frame.tobytes())
            t += FRAME_DT

    summary = {
        "act": "summary",
        "task": task,
        "goal": [goal[0], goal[1]],
        "bed": bed,
        "decisions": decisions,
        "oracle_matches": agrees,
        "agreement": None if not decisions else round(agrees / decisions, 3),
        "final_m": round(dist, 3),
        "final_bearing": round(bearing, 1),
        "spatial": spatial,
        "final_pose": {
            "x": round(info.x, 3), "y": round(info.y, 3),
            "z": round(info.base_z, 3), "yaw": round(info.yaw, 3),
            "skill": info.skill,
        },
    }
    log_f.write(json.dumps(summary) + "\n")
    log_f.close()
    plate.close()
    if dep_big is not None:
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
    parser.add_argument("--task", choices=("pick", "stand"), required=True)
    parser.add_argument("--goal", type=float, nargs=2, metavar=("X", "Y"), required=True)
    parser.add_argument("--bed", choices=("funky", "chill", "bounce", "drift"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--duration", type=float, default=36.0)
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--spatial", action="store_true")
    parser.add_argument("--caption", default="")
    args = parser.parse_args()
    caption = args.caption or (
        "Color depth and a position grid. No meters or degrees are sent."
        if args.spatial
        else None
    ) or (
        "Marker on the right. Turn, walk, pick, sit."
        if args.goal[1] < -0.2 and args.task == "pick"
        else "Walk to the marker and stand. stand is the stop."
        if args.task == "stand"
        else "Marker ahead. Walk, pick, sit."
    )
    asyncio.run(run_clip(
        tuple(args.goal), args.task, args.out, args.bed, args.duration, caption,
        vision=args.vision, spatial=args.spatial,
    ))


if __name__ == "__main__":
    main()
