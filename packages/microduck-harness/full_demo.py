"""Research plate: real MicroDuck policies, then a closed loop on d1-3B.

The first act is a catalog. Each clip runs one trained network with the command
encoding that network was trained on. The second act resets the duck and lets
d1-3B name the next skill from the forward camera plus the marker bearing.
The policy, not the decision model, moves the joints. Walks and turns use a
fixed medium command so a stop answer cannot freeze the gait.

Frames are streamed to ffmpeg. Nothing is held in a frame list.

    MUJOCO_GL=egl .venv/bin/python packages/microduck-harness/full_demo.py
    ... full_demo.py --preview          # four stills, no encode
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "decision-client"))
sys.path.insert(0, os.path.dirname(__file__))

from decision_client import DecisionClient, decision_settings
from policy_runner import (
    ALL_SKILLS,
    CONTROL_DT,
    DuckRuntime,
    chase_camera,
    forward_camera,
    wider_forward_fov,
)

OUT_MP4 = "full_demo.mp4"
OUT_LOG = "demo_log.jsonl"
DECISION_URL = decision_settings()["base_url"]

FPS = 30
FRAME_DT = 1.0 / FPS
W, H = 1920, 1080

# (t0, t1, skill, speed, caption). Ends with a reset into the closed loop.
CATALOG = [
    (0.0, 5.0, "stand", "stop", "Stand. velstand, zero twist."),
    (5.0, 12.0, "walk_forward", "medium", "Walk. Same network, forward twist."),
    (12.0, 17.0, "turn_left", "medium", "Turn left in place."),
    (17.0, 22.0, "turn_right", "medium", "Turn right in place."),
    (22.0, 25.0, "stand", "stop", "Stop."),
    (25.0, 29.2, "ground_pick", "stop", "Ground pick. Separate network, phase in the twist slots."),
    (29.2, 33.4, "sit", "stop", "Sit. Posture flag on the sit-stand network."),
    (33.4, 37.0, "stand", "stop", "Stand up. The sit network rises, then velstand holds."),
]
RESET_AT = 37.0
CLOSED_FROM = 38.0
# Closed loop reaches the marker, picks, and sits in about 15s (measured).
# 64s leaves that ending on screen without a long freeze.
DURATION = 64.0

BG = (27, 34, 46)
PANEL = (33, 40, 54)
LINE = (86, 98, 118)
CREAM = (236, 230, 218)
MUTED = (168, 176, 188)
AMBER = (232, 168, 74)
INK = (18, 22, 30)

SKILL_COLOR = {
    "stand": (150, 168, 150),
    "walk_forward": (232, 168, 74),
    "turn_left": (120, 168, 196),
    "turn_right": (168, 146, 196),
    "ground_pick": (210, 122, 78),
    "sit": (196, 150, 150),
}

# Bands are exclusive. d1-3B followed this shape; the longer overlapping
# criteria (and a separate speed question) are what the 600M loop failed.
QUESTIONS = {
    "skill": {
        "type": "choice",
        "instructions": (
            "Choose the one skill whose criterion matches the state. "
            "bearing_deg is positive when the orange marker is to the robot's left."
        ),
        "criteria": {
            "turn_left": "pick_done is false and bearing_deg >= 18",
            "turn_right": "pick_done is false and bearing_deg <= -18",
            "walk_forward": "pick_done is false and distance_m > 0.34 and bearing_deg > -18 and bearing_deg < 18",
            "ground_pick": "pick_done is false and distance_m <= 0.34 and bearing_deg > -18 and bearing_deg < 18",
            "sit": "pick_done is true",
        },
    },
}

# Distance and bearing are not in the request. pick_done stays, because the
# forward camera cannot see the robot's own posture, so sit is not in the frame.
VISION_QUESTIONS = {
    "skill": {
        "type": "choice",
        "instructions": (
            "Look only at the forward camera. Do not expect distance or bearing numbers. "
            "The orange marker is the goal. Positive image-left is the robot's left."
        ),
        "criteria": {
            "turn_left": "pick_done is false and the orange marker sits clearly left of the image center",
            "turn_right": "pick_done is false and the orange marker sits clearly right of the image center",
            "walk_forward": "pick_done is false and the orange marker is near the center and still small, far from the feet",
            "ground_pick": "pick_done is false and the orange marker is near the center and large, close to the feet",
            "sit": "pick_done is true",
        },
    },
}


def _font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.ImageFont:
    if mono:
        path = "/usr/share/fonts/TTF/DejaVuSansMono.ttf"
    elif bold:
        path = "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"
    else:
        path = "/usr/share/fonts/TTF/DejaVuSans.ttf"
    return ImageFont.truetype(path, size)


FONT_KICKER = _font(15, bold=True)
FONT_TITLE = _font(34, bold=True)
FONT_BODY = _font(16)
FONT_SMALL = _font(14)
FONT_SKILL = _font(28, bold=True)
FONT_MONO = _font(15, mono=True)
FONT_CAPTION = _font(20, bold=True)


def fit(draw: ImageDraw.ImageDraw, text: str, font, max_w: float) -> str:
    if draw.textlength(text, font=font) <= max_w:
        return text
    ell = "…"
    while text and draw.textlength(text + ell, font=font) > max_w:
        text = text[:-1]
    return (text + ell) if text else ell


def wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: float) -> list[str]:
    lines: list[str] = []
    cur = ""
    for word in text.split():
        trial = word if not cur else f"{cur} {word}"
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def colorize_depth(depth: np.ndarray, near: float = 0.08, far: float = 2.4) -> np.ndarray:
    """Near is amber, far is the panel blue. One ramp, so the inset stays quiet."""
    d = np.nan_to_num(depth, nan=far, posinf=far, neginf=near)
    t = np.clip((d - near) / (far - near), 0.0, 1.0)[..., None]
    near_c = np.array([214.0, 148.0, 72.0], dtype=np.float32)
    far_c = np.array([36.0, 48.0, 64.0], dtype=np.float32)
    rgb = near_c * (1.0 - t) + far_c * t
    return np.clip(rgb, 0, 255).astype(np.uint8)


def depth_readout(depth: np.ndarray) -> tuple[float, list[list[float]]]:
    d = np.nan_to_num(depth, nan=3.0, posinf=3.0, neginf=0.05)
    h, w = d.shape
    y0, y1 = h // 2 - h // 8, h // 2 + h // 8
    x0, x1 = w // 2 - w // 8, w // 2 + w // 8
    ahead = float(np.median(d[y0:y1, x0:x1]))
    gh, gw = h // 4 * 4, w // 4 * 4
    grid = d[:gh, :gw].reshape(4, gh // 4, 4, gw // 4).min(axis=(1, 3))
    return ahead, [[round(float(v), 3) for v in row] for row in grid]


def oracle(distance: float, bearing: float, pick_done: bool) -> str:
    """Same bands as the skill criteria, so agreement has one right answer."""
    if pick_done:
        return "sit"
    if bearing >= 18.0:
        return "turn_left"
    if bearing <= -18.0:
        return "turn_right"
    if distance <= 0.34:
        return "ground_pick"
    return "walk_forward"


def controller_speed(skill: str) -> str:
    """Walks and turns get a medium command. Pick and sit hold still.

    The 600M loop answered stop on a walk and the twist went to zero, so the
    duck never left the origin. Speed here is the controller, not the model.
    """
    if skill in ("walk_forward", "turn_left", "turn_right"):
        return "medium"
    return "stop"


def decision_state(distance: float, bearing: float, pick_done: bool) -> dict:
    """Only the fields the criteria name. Extra keys pulled the 600M off the band."""
    return {
        "distance_m": round(float(distance), 2),
        "bearing_deg": round(float(bearing), 1),
        "pick_done": bool(pick_done),
    }


def catalog_at(t: float):
    for row in CATALOG:
        if row[0] <= t < row[1]:
            return row
    return None


class Plate:
    def __init__(self, duck: DuckRuntime):
        self.duck = duck
        self.chase_r = mujoco.Renderer(duck.model, height=860, width=1280)
        self.fwd_r = mujoco.Renderer(duck.model, height=180, width=250)
        self.dep_r = mujoco.Renderer(duck.model, height=180, width=250)
        self.dep_r.enable_depth_rendering()
        self.chase_cam = mujoco.MjvCamera()
        self.fwd_cam = mujoco.MjvCamera()
        self.segments: list[list] = []

    def close(self) -> None:
        for r in (self.chase_r, self.fwd_r, self.dep_r):
            r.close()

    def mark(self, t: float, skill: str) -> None:
        if not self.segments or self.segments[-1][2] != skill:
            if self.segments:
                self.segments[-1][1] = t
            self.segments.append([t, t, skill])
        else:
            self.segments[-1][1] = t

    def grab(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, list]:
        chase_camera(self.duck, self.chase_cam)
        self.chase_r.update_scene(self.duck.data, camera=self.chase_cam)
        chase = self.chase_r.render().copy()
        with wider_forward_fov(self.duck.model):
            forward_camera(self.duck, self.fwd_cam)
            self.fwd_r.update_scene(self.duck.data, camera=self.fwd_cam)
            fwd = self.fwd_r.render().copy()
            self.dep_r.update_scene(self.duck.data, camera=self.fwd_cam)
            depth = self.dep_r.render().copy()
        ahead, grid = depth_readout(depth)
        return chase, fwd, depth, ahead, grid

    def compose(
        self,
        t: float,
        chase: np.ndarray,
        fwd: np.ndarray,
        depth: np.ndarray,
        info,
        caption: str,
        mode_label: str,
        model_skill: str | None,
        probs: dict | None,
        conf: float | None,
        latency_ms: float | None,
        agree: str,
        distance: float,
        bearing: float,
        split_at: float | None = None,
        split_left: str = "catalog",
        split_right: str = "closed loop",
        bar_duration: float | None = None,
        goal_label: str = "goal",
        extra_rows: list[tuple[str, str]] | None = None,
        subtitle: str | None = None,
        corner: Image.Image | None = None,
        depth_view: Image.Image | None = None,
    ) -> Image.Image:
        img = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(img)
        img.paste(Image.fromarray(chase), (28, 24))
        draw.rectangle([28, 24, 28 + 1280 - 1, 24 + 860 - 1], outline=LINE)
        if corner is not None:
            cw = 300
            ch = max(1, int(cw * corner.height / corner.width))
            thumb = corner.resize((cw, ch), Image.BILINEAR)
            img.paste(thumb, (44, 40))
            draw.rectangle([44, 40, 44 + cw - 1, 40 + ch - 1], outline=LINE)

        # Caption band on the chase view.
        band_top = 24 + 860 - 78
        draw.rectangle([28, band_top, 28 + 1280 - 1, 24 + 860 - 1], fill=(16, 20, 28))
        cap_lines = wrap(draw, caption, FONT_CAPTION, 1230)[:2]
        y = band_top + 12
        for line in cap_lines:
            draw.text((44, y), fit(draw, line, FONT_CAPTION, 1230), fill=CREAM, font=FONT_CAPTION)
            y += 28

        # Right column.
        rx, ry, rw = 1332, 24, 560
        draw.rounded_rectangle([rx, ry, rx + rw, 884], radius=12, fill=PANEL, outline=LINE)
        x = rx + 22
        inner = rw - 44
        draw.text((x, 40), "MICRODUCK", fill=AMBER, font=FONT_KICKER)
        draw.text((x, 62), "System One", fill=CREAM, font=FONT_TITLE)
        sub = subtitle or "d1-3B names the skill. A 50 Hz policy moves the joints."
        for i, line in enumerate(wrap(draw, sub, FONT_SMALL, inner)[:2]):
            draw.text((x, 108 + i * 18), line, fill=MUTED, font=FONT_SMALL)

        draw.rounded_rectangle([x, 156, x + 150, 180], radius=8, fill=INK)
        draw.text((x + 12, 159), fit(draw, mode_label, FONT_SMALL, 126), fill=AMBER, font=FONT_SMALL)
        clock = f"{t:05.2f}s"
        draw.text((rx + rw - 22 - draw.textlength(clock, font=FONT_MONO), 159), clock, fill=MUTED, font=FONT_MONO)

        y = 196
        draw.text((x, y), "EXECUTED", fill=MUTED, font=FONT_KICKER)
        y += 22
        draw.text((x, y), fit(draw, info.skill.replace("_", " "), FONT_SKILL, inner), fill=CREAM, font=FONT_SKILL)
        y += 38
        meta = f"command {info.speed}   mode {info.mode}"
        draw.text((x, y), fit(draw, meta, FONT_BODY, inner), fill=MUTED, font=FONT_BODY)
        y += 28

        draw.text((x, y), "MODEL", fill=MUTED, font=FONT_KICKER)
        y += 20
        shown = model_skill.replace("_", " ") if model_skill else "—"
        if conf is not None:
            shown = f"{shown}   {conf:.2f}"
        draw.text((x, y), fit(draw, shown, FONT_BODY, inner - 90), fill=CREAM, font=FONT_BODY)
        if latency_ms is not None:
            lat = f"{latency_ms:.0f} ms"
            draw.text((x + inner - draw.textlength(lat, font=FONT_MONO), y + 1), lat, fill=MUTED, font=FONT_MONO)
        y += 26
        probs = probs or {}
        bar_x = x + 132
        bar_w = inner - 132 - 52
        for skill in ALL_SKILLS:
            p = float(probs.get(skill, 0.0))
            label = skill.replace("_", " ")
            col = SKILL_COLOR[skill] if skill == info.skill else MUTED
            draw.text((x, y), fit(draw, label, FONT_SMALL, 124), fill=col, font=FONT_SMALL)
            draw.rectangle([bar_x, y + 4, bar_x + bar_w, y + 14], fill=INK)
            fill_w = max(0, int(bar_w * max(0.0, min(1.0, p))))
            if fill_w:
                draw.rectangle([bar_x, y + 4, bar_x + fill_w, y + 14], fill=SKILL_COLOR[skill])
            pct = f"{p * 100:4.0f}%"
            draw.text((bar_x + bar_w + 8, y - 1), pct, fill=MUTED, font=FONT_MONO)
            y += 22

        y += 8
        fwd_im = Image.fromarray(fwd).resize((250, 160), Image.BILINEAR)
        if depth_view is None:
            dep_im = Image.fromarray(colorize_depth(depth)).resize((250, 160), Image.BILINEAR)
        else:
            dep_im = depth_view.convert("RGB").resize((250, 160), Image.BILINEAR)
        img.paste(fwd_im, (x, y))
        img.paste(dep_im, (x + 262, y))
        draw.text((x, y + 164), "forward", fill=MUTED, font=FONT_SMALL)
        draw.text((x + 262, y + 164), "depth", fill=MUTED, font=FONT_SMALL)
        y += 190

        rows = [(goal_label, f"{distance:.2f} m   {bearing:+.0f}°")]
        if extra_rows:
            rows.extend(extra_rows)
        rows.extend(
            [
                ("twist", f"{info.twist[0]:+.2f}  {info.twist[1]:+.2f}  {info.twist[2]:+.2f}"),
                ("base z", f"{info.base_z:.3f} m"),
                ("|action|", f"{info.action_norm:.2f}"),
                ("oracle", agree),
            ]
        )
        for label, value in rows:
            draw.text((x, y), label, fill=MUTED, font=FONT_SMALL)
            draw.text((x + 110, y), fit(draw, value, FONT_MONO, inner - 110), fill=CREAM, font=FONT_MONO)
            y += 20

        # Timeline.
        draw.rounded_rectangle([28, 900, W - 28, H - 20], radius=12, fill=PANEL, outline=LINE)
        bar_x0, bar_y0, bar_w, bar_h = 48, 940, W - 48 - 48, 28
        span = DURATION if bar_duration is None else bar_duration
        split = RESET_AT if split_at is None else split_at
        div = bar_x0 + int(bar_w * split / span)
        draw.text((48, 912), "SKILL", fill=MUTED, font=FONT_KICKER)
        skill_end = 48 + draw.textlength("SKILL", font=FONT_KICKER)
        left_x = div - 12 - draw.textlength(split_left, font=FONT_SMALL)
        if left_x > skill_end + 14:
            draw.text((left_x, 912), split_left, fill=MUTED, font=FONT_SMALL)
        right_x = div + 10
        if right_x > skill_end + 14:
            draw.text((right_x, 912), split_right, fill=AMBER, font=FONT_SMALL)
        draw.rectangle([bar_x0, bar_y0, bar_x0 + bar_w, bar_y0 + bar_h], fill=INK)
        for seg in self.segments:
            x0 = bar_x0 + int(bar_w * seg[0] / span)
            x1 = bar_x0 + int(bar_w * max(seg[1], seg[0] + FRAME_DT) / span)
            draw.rectangle([x0, bar_y0, max(x0 + 1, x1), bar_y0 + bar_h], fill=SKILL_COLOR.get(seg[2], MUTED))
        play = bar_x0 + int(bar_w * min(t, span) / span)
        draw.line([(div, bar_y0 - 4), (div, bar_y0 + bar_h + 4)], fill=LINE, width=1)
        draw.rectangle([play - 1, bar_y0 - 4, play + 1, bar_y0 + bar_h + 4], fill=CREAM)

        legend_x = 48
        ly = 982
        for skill in ALL_SKILLS:
            name = skill.replace("_", " ")
            draw.rectangle([legend_x, ly + 3, legend_x + 12, ly + 15], fill=SKILL_COLOR[skill])
            draw.text((legend_x + 18, ly), name, fill=MUTED, font=FONT_SMALL)
            legend_x += int(draw.textlength(name, font=FONT_SMALL)) + 36
        return img


async def _decide(client, state, image: Image.Image, questions=None):
    result = await client.decide(state, QUESTIONS if questions is None else questions, images=[image])
    skill = result.choice("skill") or "stand"
    if skill not in ALL_SKILLS:
        skill = "stand"
    probs = (result.answers.get("skill") or {}).get("probabilities") or {}
    return skill, float(result.confidence("skill") or 0.0), probs, result.latency_s * 1000.0


async def run(preview: bool, out_path: str, vision: bool = False) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    duck = DuckRuntime()
    plate = Plate(duck)
    log_path = "/tmp/demo_preview.jsonl" if preview else os.path.splitext(out_path)[0] + ".jsonl"
    log_f = open(log_path, "w")
    proc = None
    if not preview:
        proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                "-i", "-",
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17",
                "-preset", "veryfast", "-movflags", "+faststart",
                out_path,
            ],
            stdin=subprocess.PIPE,
        )

    opinion = {"skill": None, "speed": None, "conf": None, "probs": {}, "latency_ms": None}
    pick_done = False
    was_pick = False
    applied_segment = None
    next_opinion = 0.0
    next_command = CLOSED_FROM
    agrees = 0
    decisions = 0
    t = 0.0
    acc = 0.0
    info = duck.step()  # pose after the settle already inside reset; step once more for a filled info
    # The constructor reset already settled. One extra step is harmless.
    preview_at = {6.0: "walk", 27.0: "pick", 31.0: "sit", 50.0: "loop"}
    saved = set()

    async with DecisionClient.from_env(timeout=30.0, max_image_side=320, retries=1) as client:
        while t < DURATION - 1e-9:
            if t >= RESET_AT and applied_segment != "armed" and applied_segment != "loop":
                duck.reset()
                pick_done = False
                was_pick = False
                info = duck.step()
                applied_segment = "armed"
                acc = 0.0

            row = catalog_at(t) if t < RESET_AT else None
            if row is not None and applied_segment != row[0]:
                duck.set_skill(row[2], row[3])
                applied_segment = row[0]

            if t >= CLOSED_FROM and t + 1e-9 >= next_command:
                chase, fwd, depth, ahead, grid = plate.grab()
                dist, bearing = duck.goal_relative()
                measured = decision_state(dist, bearing, pick_done)
                state = {"camera": "forward", "pick_done": bool(pick_done)} if vision else measured
                if vision:
                    duck.set_walk_trim(bearing)
                try:
                    skill, conf, probs, lat = await _decide(
                        client, state, Image.fromarray(fwd), VISION_QUESTIONS if vision else None
                    )
                except Exception as exc:
                    skill, conf, probs, lat = duck.skill, 0.0, {}, -1.0
                    print("decide failed", exc, flush=True)
                speed = controller_speed(skill)
                reason = duck.set_skill(skill, speed)
                guess = oracle(dist, bearing, pick_done)
                hit = skill == guess
                agrees += int(hit)
                decisions += 1
                opinion = {"skill": skill, "speed": speed, "conf": conf, "probs": probs, "latency_ms": lat}
                log_f.write(json.dumps({
                    "t": round(t, 3), "act": "closed", "state": state,
                    "measured": measured, "vision": vision,
                    "walk_yaw_trim": round(duck.walk_yaw_trim, 3),
                    "model_skill": skill, "applied_speed": speed, "speed_source": "controller",
                    "confidence": round(conf, 3),
                    "probabilities": {k: round(float(v), 4) for k, v in probs.items()},
                    "latency_ms": round(lat, 1), "applied": reason,
                    "executed_skill": duck.skill, "oracle": guess, "agree": hit,
                    "depth_ahead_m": round(ahead, 3),
                    "depth_grid_min": grid,
                    "pose": {"x": round(info.x, 3), "y": round(info.y, 3), "z": round(info.base_z, 3), "yaw": round(info.yaw, 3)},
                }) + "\n")
                log_f.flush()
                next_command += 0.5
            elif t < RESET_AT and t + 1e-9 >= next_opinion:
                _, fwd, _, ahead, grid = plate.grab()
                dist, bearing = duck.goal_relative()
                measured = decision_state(dist, bearing, False)
                state = {"camera": "forward", "pick_done": False} if vision else measured
                try:
                    skill, conf, probs, lat = await _decide(
                        client, state, Image.fromarray(fwd), VISION_QUESTIONS if vision else None
                    )
                except Exception as exc:
                    skill, conf, probs, lat = None, None, {}, -1.0
                    print("opinion failed", exc, flush=True)
                opinion = {"skill": skill, "speed": None, "conf": conf, "probs": probs, "latency_ms": lat}
                log_f.write(json.dumps({
                    "t": round(t, 3), "act": "catalog", "executed_skill": duck.skill,
                    "model_skill": skill, "state": state, "measured": measured, "vision": vision,
                    "confidence": None if conf is None else round(conf, 3),
                    "probabilities": {k: round(float(v), 4) for k, v in (probs or {}).items()},
                    "latency_ms": None if lat is None else round(lat, 1),
                    "note": "opinion only; the catalog skill is what runs",
                    "depth_ahead_m": round(ahead, 3),
                    "depth_grid_min": grid,
                }) + "\n")
                log_f.flush()
                next_opinion += 1.0

            acc += FRAME_DT
            while acc >= CONTROL_DT - 1e-9:
                if vision and t >= CLOSED_FROM:
                    _, bearing_now = duck.goal_relative()
                    duck.set_walk_trim(bearing_now)
                else:
                    duck.set_walk_trim(None)
                info = duck.step()
                acc -= CONTROL_DT
                if duck.mode == "pick":
                    was_pick = True
                elif was_pick:
                    pick_done = True
                    was_pick = False

            if t >= CLOSED_FROM and applied_segment == "armed":
                applied_segment = "loop"

            dist, bearing = duck.goal_relative()
            if t < RESET_AT:
                row = catalog_at(t)
                caption = row[4] if row else ""
                mode_label = "catalog"
                agree = "held"
            elif t < CLOSED_FROM:
                caption = "Reset to the start. Closed loop begins on the next beat."
                mode_label = "reset"
                agree = "—"
            else:
                caption = (
                    "Closed loop from the camera. Distance and bearing are not sent. A walk gets a small heading trim."
                    if vision
                    else "Closed loop. d1-3B names the skill. Walk and turn use a medium command."
                )
                mode_label = "closed loop"
                guess = oracle(dist, bearing, pick_done)
                agree = "match" if opinion["skill"] == guess else f"oracle {guess.replace('_', ' ')}"

            plate.mark(t, info.skill)
            chase, fwd, depth, _, _ = plate.grab()
            frame = plate.compose(
                t, chase, fwd, depth, info, caption, mode_label,
                opinion["skill"], opinion["probs"], opinion["conf"], opinion["latency_ms"],
                agree, dist, bearing,
                extra_rows=[("sent", "camera + pick_done")] if vision else None,
                subtitle=(
                    "d1-3B sees the camera. It is not given distance or bearing."
                    if vision
                    else None
                ),
            )
            if preview:
                for stamp, name in preview_at.items():
                    if name not in saved and t >= stamp:
                        path = f"/tmp/plate_{name}.png"
                        frame.save(path)
                        print("preview", path, flush=True)
                        saved.add(name)
                if saved == set(preview_at.values()):
                    break
            else:
                proc.stdin.write(frame.tobytes())
            t += FRAME_DT
            if int(t) != int(t - FRAME_DT) and int(t) % 5 == 0:
                print(f"t={t:.0f}s  skill={info.skill}  z={info.base_z:.3f}  decisions={decisions}", flush=True)

    summary = {
        "t": round(t, 3),
        "act": "summary",
        "decision_url": DECISION_URL,
        "model": "d1-3B",
        "vision": vision,
        "closed_loop_decisions": decisions,
        "oracle_matches": agrees,
        "agreement": None if not decisions else round(agrees / decisions, 3),
        "log": log_path,
    }
    log_f.write(json.dumps(summary) + "\n")
    log_f.close()
    plate.close()
    if proc is not None:
        proc.stdin.close()
        code = proc.wait()
        if code != 0:
            raise SystemExit(f"ffmpeg exited {code}")
        print("saved", out_path, "log", log_path, "agreement", summary["agreement"], flush=True)
    else:
        print("preview done", summary, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="write four stills and stop")
    parser.add_argument("--vision", action="store_true", help="hide distance and bearing from the model")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = args.out or ("full_demo_vision.mp4" if args.vision else OUT_MP4)
    asyncio.run(run(args.preview, out, vision=args.vision))


if __name__ == "__main__":
    main()
