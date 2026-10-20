"""Color range for the decision model, with no meters and no degrees in the request.

The forward RGB frame is a poor range cue: the same post looks "close" at half a
meter and at a meter. Two pictures replace that cue.

Depth is the forward camera's z-buffer, painted in flat bands. Red is close
enough to stop, orange is the band where the RGB loop stopped early, yellow
and green are farther, blue is sky. The grid is a top-down map drawn from the
marker pose. The duck is the triangle at the bottom. A white disc is the
marker. Columns are left, center, and right. The bright cell's color is the
same band as the depth picture.

Thresholds match oracle_stand in goal_clip.py (18 degrees, 0.40 m). Those
numbers stay in this file. They are not written on the picture and they are
not put in the decision state.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Exclusive bands, closest first. The last band is everything farther than 1.20 m
# and nearer than the sky cut. Sky is painted separately.
BANDS = (
    ("close", 0.40, (210, 40, 40)),
    ("near", 0.75, (230, 130, 20)),
    ("mid", 1.20, (230, 210, 30)),
    ("far", 6.00, (30, 160, 70)),
)
SKY_RGB = (30, 48, 110)
TURN_DEG = 18.0
STOP_M = 0.40

# Top of the picture is far. The triangle sits under the close row.
ROW_ORDER = ("far", "mid", "near", "close")
COL_ORDER = ("left", "center", "right")

_BG = (27, 34, 46)
_INK = (18, 22, 30)
_CREAM = (236, 230, 218)
_LINE = (18, 22, 30)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/TTF/DejaVuSans.ttf"
    return ImageFont.truetype(path, size)


def band_of(distance_m: float) -> tuple[str, tuple[int, int, int]]:
    d = float(distance_m)
    for name, limit, rgb in BANDS:
        if d <= limit:
            return name, rgb
    return "far", BANDS[-1][2]


def column_of(bearing_deg: float) -> str:
    b = float(bearing_deg)
    if b >= TURN_DEG:
        return "left"
    if b <= -TURN_DEG:
        return "right"
    return "center"


def skill_from_cell(distance_m: float, bearing_deg: float) -> str:
    """Same answer as oracle_stand. Bearing wins, then the close band stands."""
    col = column_of(bearing_deg)
    if col == "left":
        return "turn_left"
    if col == "right":
        return "turn_right"
    if band_of(distance_m)[0] == "close":
        return "stand"
    return "walk_forward"


def colorize_depth_bands(depth: np.ndarray) -> np.ndarray:
    """Flat bands. A smooth ramp is hard to name. A red floor is not."""
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    d = np.nan_to_num(d, nan=99.0, posinf=99.0, neginf=0.0)
    rgb = np.empty(d.shape + (3,), dtype=np.uint8)
    rgb[:] = SKY_RGB
    # Far first, then closer bands overwrite.
    for name, limit, color in reversed(BANDS):
        rgb[d <= limit] = color
    return rgb


def forward_depth_picture(runtime, renderer, cam) -> Image.Image:
    """Color bands from the same forward aim the RGB camera uses."""
    from policy_runner import forward_camera, wider_forward_fov

    with wider_forward_fov(runtime.model):
        forward_camera(runtime, cam)
        renderer.update_scene(runtime.data, camera=cam)
        depth = renderer.render().copy()
    return depth_picture(depth)


def depth_picture(depth: np.ndarray) -> Image.Image:
    """Depth camera plus a color key. The key uses words, not meters."""
    rgb = colorize_depth_bands(depth)
    src = Image.fromarray(rgb, mode="RGB")
    # The key has four words. 480px clips the last one, so the canvas grows.
    band_h = 64
    width = max(src.width, 640)
    img = Image.new("RGB", (width, src.height + band_h), _BG)
    img.paste(src, ((width - src.width) // 2, band_h))
    draw = ImageDraw.Draw(img)
    font = _font(22, bold=True)
    draw.text((12, 18), "DEPTH", fill=_CREAM, font=font)
    chips = (
        ("close", BANDS[0][2]),
        ("near", BANDS[1][2]),
        ("mid", BANDS[2][2]),
        ("far", BANDS[3][2]),
    )
    x = 128
    for word, color in chips:
        draw.rectangle([x, 16, x + 28, 48], fill=color)
        draw.text((x + 36, 18), word, fill=_CREAM, font=font)
        x += 36 + int(draw.textlength(word, font=font)) + 22
    return img


def _shade(rgb: tuple[int, int, int], hot: bool) -> tuple[int, int, int]:
    if hot:
        return rgb
    return tuple(int(c * 0.34) for c in rgb)


def position_grid(distance_m: float, bearing_deg: float) -> Image.Image:
    """Top-down map. Up is forward. The bright cell is the marker."""
    width, height = 520, 680
    img = Image.new("RGB", (width, height), _BG)
    draw = ImageDraw.Draw(img)
    title = _font(28, bold=True)
    label = _font(22, bold=True)
    draw.text((16, 14), "GRID", fill=_CREAM, font=title)
    draw.text((130, 18), "you face up", fill=(168, 176, 188), font=label)

    margin_l, margin_r = 118, 18
    top, bottom = 92, 96
    cols, rows = 3, 4
    grid_w = width - margin_l - margin_r
    grid_h = height - top - bottom
    cw, ch = grid_w // cols, grid_h // rows
    goal_col = COL_ORDER.index(column_of(bearing_deg))
    goal_row = ROW_ORDER.index(band_of(distance_m)[0])

    for c, word in enumerate(COL_ORDER):
        cx = margin_l + c * cw + cw / 2
        tw = draw.textlength(word.upper(), font=label)
        draw.text((cx - tw / 2, 58), word.upper(), fill=_CREAM, font=label)

    for r, name in enumerate(ROW_ORDER):
        color = next(rgb for n, _, rgb in BANDS if n == name)
        draw.text((16, top + r * ch + ch / 2 - 14), name.upper(), fill=color, font=label)
        for c in range(cols):
            hot = r == goal_row and c == goal_col
            x0 = margin_l + c * cw
            y0 = top + r * ch
            draw.rectangle([x0, y0, x0 + cw - 1, y0 + ch - 1], fill=_shade(color, hot), outline=_LINE, width=6)
            if hot:
                disc_r = min(cw, ch) // 2 - 18
                cx, cy = x0 + cw // 2, y0 + ch // 2
                draw.ellipse([cx - disc_r, cy - disc_r, cx + disc_r, cy + disc_r], fill=(250, 250, 250), outline=_INK, width=6)

    # Triangle under the center column. Tip points up, toward far.
    base_y = height - 28
    tip_y = height - bottom + 18
    mid_x = margin_l + cw + cw // 2
    draw.polygon([(mid_x, tip_y), (mid_x - 22, base_y), (mid_x + 22, base_y)], fill=_CREAM)
    you = "YOU"
    tw = draw.textlength(you, font=label)
    draw.text((mid_x - tw / 2, base_y - 26), you, fill=_INK, font=_font(16, bold=True))
    return img


# The grid is the picture these criteria name. Depth uses DEPTH_STAND.
GRID_STAND = {
    "skill": {
        "type": "choice",
        "instructions": (
            "Choose the one skill whose criterion matches the GRID picture. "
            "You are the triangle at the bottom, facing up. The white disc is the marker. "
            "Columns are LEFT, CENTER, and RIGHT. "
            "Red means close. Orange, yellow, and green are farther. "
            "No distances or angles are given."
        ),
        "criteria": {
            "turn_left": "the white disc is in the LEFT column",
            "turn_right": "the white disc is in the RIGHT column",
            "walk_forward": "the white disc is in the CENTER column and the cell is not red",
            "stand": "the white disc is in the CENTER column and the cell is red",
        },
    },
}

DEPTH_STAND = {
    "skill": {
        "type": "choice",
        "instructions": (
            "Choose the one skill whose criterion matches the DEPTH picture. "
            "Red is close, orange is nearer, yellow is mid, green is far, blue is sky. "
            "The marker is the upright post. No distances or angles are given."
        ),
        "criteria": {
            "turn_left": "the post is left of the picture center",
            "turn_right": "the post is right of the picture center",
            "walk_forward": "the post is near the center and its color is not red",
            "stand": "the post is near the center and its color is red",
        },
    },
}

BOTH_STAND = {
    "skill": {
        "type": "choice",
        "instructions": (
            "Choose the one skill whose criterion matches the GRID picture. "
            "DEPTH is a depth camera with the same colors: red close, orange nearer, "
            "yellow mid, green far, blue sky. "
            "On the GRID you are the triangle at the bottom, facing up, and the white disc is the marker. "
            "No distances or angles are given."
        ),
        "criteria": {
            "turn_left": "the white disc is in the LEFT column",
            "turn_right": "the white disc is in the RIGHT column",
            "walk_forward": "the white disc is in the CENTER column and the cell is not red",
            "stand": "the white disc is in the CENTER column and the cell is red",
        },
    },
}
