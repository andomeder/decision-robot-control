"""MicroDuck ONNX runtime.

The shipped policies are 50 Hz, 61-D, and were exported with this observation
order (see each file's ONNX metadata):

    base_ang_vel(3), projected_gravity(3), joint_pos(14),
    joint_vel(14), last_action(14), command(13)

command is twist(3) + head_pose(4) + body_pose(6). Position targets are
``default_joint_pos + action * action_scale`` on the MJCF position actuators.
BAM is not installed, so these are the scene's position actuators and the
tracked speed sits below the commanded twist. Physics state is kept across
skill changes; only the ONNX session swaps.

Sit/stand and ground-pick are not velocity commands. Sit/stand writes a
posture flag into twist.x and must stay on that network while it rises.
Ground-pick writes a phase phasor into twist.xy and runs until phase 0.7.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from dataclasses import dataclass

import mujoco
import numpy as np
import onnxruntime as ort

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load_dotenv() -> None:
    path = os.path.join(os.getcwd(), ".env")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()

POLICY_DIR = os.environ.get("MICRODUCK_POLICY_DIR", os.path.join(_PKG_ROOT, "policies"))
SCENE_XML = os.environ.get("MICRODUCK_SCENE", "")


def policy_providers() -> list:
    """CUDA if that wheel is installed, else OpenVINO GPU, else CPU.

    ONNX Runtime has no XPU provider. An Arc runs through OpenVINO's GPU device.
    """
    have = set(ort.get_available_providers())
    if "CUDAExecutionProvider" in have:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "OpenVINOExecutionProvider" in have:
        return [
            ("OpenVINOExecutionProvider", {"device_type": "GPU"}),
            "CPUExecutionProvider",
        ]
    return ["CPUExecutionProvider"]

CONTROL_DT = 0.02  # 50 Hz
SIM_DT = 0.005
DECIMATION = 4  # SIM_DT * DECIMATION == CONTROL_DT
OBS_DIM = 61
ACT_DIM = 14

# Orange marker the closed-loop mission walks to. Visual only, so the gait
# stays the flat-floor policy. Close enough that a slow tracked walk arrives
# inside a minute, far enough that the turn-then-walk choice is visible.
GOAL_XY = (0.90, 0.50)

SPEED_VX = {"stop": 0.0, "slow": 0.12, "medium": 0.22, "fast": 0.35}
SPEED_YAW = {"stop": 0.0, "slow": 0.35, "medium": 0.70, "fast": 1.00}

LOCO_SKILLS = ("stand", "walk_forward", "turn_left", "turn_right")
ALL_SKILLS = LOCO_SKILLS + ("ground_pick", "sit")

PICK_PERIOD_S = 4.0
PICK_END_PHASE = 0.7
RISE_S = 2.4


def _quat_rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vec by the inverse of a wxyz quaternion. Matches infer_policy.py."""
    w = float(quat[0])
    xyz = quat[1:4]
    t = np.cross(xyz, vec) * 2.0
    return vec - w * t + np.cross(xyz, t)


def _yaw_from_quat(quat: np.ndarray) -> float:
    w, x, y, z = (float(v) for v in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class StepInfo:
    skill: str
    speed: str
    mode: str
    action_norm: float
    base_z: float
    x: float
    y: float
    yaw: float
    upright: bool
    twist: tuple[float, float, float]


class _Net:
    def __init__(self, path: str):
        self.path = path
        self.session = ort.InferenceSession(path, providers=policy_providers())
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        meta = self.session.get_modelmeta().custom_metadata_map
        self.joint_names = meta["joint_names"].split(",")
        self.default_joint_pos = np.array(
            [float(v) for v in meta["default_joint_pos"].split(",")], dtype=np.float32
        )
        self.action_scale = float(meta["action_scale"])
        obs_names = meta["observation_names"].split(",")
        if obs_names[:5] != [
            "base_ang_vel",
            "projected_gravity",
            "joint_pos",
            "joint_vel",
            "actions",
        ]:
            raise RuntimeError(f"unexpected observation order in {path}: {obs_names}")
        shape = self.session.get_inputs()[0].shape
        if list(shape) != [1, OBS_DIM]:
            raise RuntimeError(f"{path} expects {shape}, not [1, {OBS_DIM}]")


class DuckRuntime:
    """One compiled scene, three hot-swapped policies."""

    def __init__(
        self,
        scene_xml: str = SCENE_XML,
        policy_dir: str = POLICY_DIR,
        goal_xy=GOAL_XY,
        marker_xy=None,
        obstacles=(),
        path=(),
    ):
        scene_xml = scene_xml or os.environ.get("MICRODUCK_SCENE", "")
        if not scene_xml or not os.path.isfile(scene_xml):
            raise SystemExit(
                "Set MICRODUCK_SCENE to scene.xml from pollen-robotics/microduck_rl"
            )
        self.goal_xy = (float(goal_xy[0]), float(goal_xy[1]))
        marker = self.goal_xy if marker_xy is None else (float(marker_xy[0]), float(marker_xy[1]))
        self.model = _load_arena(scene_xml, marker, obstacles=obstacles, path=path)
        self.wp_geom_ids = []
        i = 0
        while True:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"wp_disc_{i}")
            if gid < 0:
                break
            self.wp_geom_ids.append(gid)
            i += 1
        self.model.opt.timestep = SIM_DT
        self.data = mujoco.MjData(self.model)

        self.nets = {
            "loco": _Net(os.path.join(policy_dir, "velstand.onnx")),
            "sit": _Net(os.path.join(policy_dir, "alpha_sitstand.onnx")),
            "pick": _Net(os.path.join(policy_dir, "alpha_ground_pick.onnx")),
        }
        ref = self.nets["loco"]
        for name, net in self.nets.items():
            if net.joint_names != ref.joint_names:
                raise RuntimeError(f"{name} joint order does not match velstand")
        self.joint_names = ref.joint_names
        self.default_pose = ref.default_joint_pos.copy()
        self.action_scale = ref.action_scale

        if self.model.nu != ACT_DIM:
            raise RuntimeError(f"expected {ACT_DIM} actuators, found {self.model.nu}")
        act_names = [self.model.actuator(i).name for i in range(self.model.nu)]
        if act_names != self.joint_names:
            raise RuntimeError(f"actuator order {act_names} != policy joints {self.joint_names}")

        self.qpos_adr = np.array(
            [int(self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]]) for i in range(ACT_DIM)],
            dtype=np.int32,
        )
        self.qvel_adr = np.array(
            [int(self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]]) for i in range(ACT_DIM)],
            dtype=np.int32,
        )
        self.trunk_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
        self.stand_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
        if self.trunk_id < 0 or self.gyro_id < 0 or self.stand_key < 0:
            raise RuntimeError("scene is missing trunk_base, imu_ang_vel, or the STAND keyframe")
        self.gyro_adr = int(self.model.sensor_adr[self.gyro_id])

        self.last_action = np.zeros(ACT_DIM, dtype=np.float32)
        self.command = np.zeros(13, dtype=np.float32)
        self.skill = "stand"
        self.speed = "stop"
        self.mode = "loco"
        self._net = self.nets["loco"]
        self._pick_phase = 0.0
        self._rise_left = 0.0
        self._pending_skill = "stand"
        self._pending_speed = "stop"
        self.walk_yaw_trim = 0.0
        self.sim_time = 0.0
        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.stand_key)
        self.data.ctrl[:] = self.default_pose
        self.last_action[:] = 0
        self.command[:] = 0
        self.skill = "stand"
        self.speed = "stop"
        self.mode = "loco"
        self._net = self.nets["loco"]
        self._pick_phase = 0.0
        self._rise_left = 0.0
        self._pending_skill = "stand"
        self._pending_speed = "stop"
        self.walk_yaw_trim = 0.0
        self.sim_time = 0.0
        mujoco.mj_forward(self.model, self.data)
        for _ in range(15):
            self.step()
        self.sim_time = 0.0

    def set_walk_trim(self, bearing_deg: float | None) -> None:
        """Yaw added only while walking. None clears it. The model is not asked for this."""
        if bearing_deg is None:
            self.walk_yaw_trim = 0.0
            return
        yaw = math.radians(float(bearing_deg))
        self.walk_yaw_trim = max(-0.35, min(0.35, yaw))

    def set_skill(self, skill: str, speed: str = "medium") -> str:
        """Request a skill. Returns a short reason if the runtime holds the current one."""
        if skill not in ALL_SKILLS:
            return "unknown"
        if speed not in SPEED_VX:
            speed = "medium"
        if self.mode == "pick":
            return "locked:pick"
        if self.mode == "rise":
            self._pending_skill = skill if skill in LOCO_SKILLS else "stand"
            self._pending_speed = speed if skill in LOCO_SKILLS else "stop"
            return "locked:rise"
        if skill == "ground_pick":
            if self.mode == "sit":
                return "locked:sit"
            self.mode = "pick"
            self._net = self.nets["pick"]
            self._pick_phase = 0.0
            self.skill = "ground_pick"
            self.speed = "stop"
            self._write_pick_command()
            return "ok"
        if skill == "sit":
            self.mode = "sit"
            self._net = self.nets["sit"]
            self.skill = "sit"
            self.speed = "stop"
            self.command[:] = 0
            self.command[0] = 1.0
            return "ok"
        # Locomotion. A seated robot has to rise on the sit/stand network first.
        if self.mode == "sit":
            self.mode = "rise"
            self._net = self.nets["sit"]
            self._rise_left = RISE_S
            self._pending_skill = skill
            self._pending_speed = speed
            self.skill = "stand"
            self.speed = "stop"
            self.command[:] = 0
            return "rising"
        self.mode = "loco"
        self._net = self.nets["loco"]
        self.skill = skill
        self.speed = "stop" if skill == "stand" else speed
        self._write_loco_command()
        return "ok"

    def _write_loco_command(self) -> None:
        vx = SPEED_VX[self.speed] if self.skill == "walk_forward" else 0.0
        yaw = 0.0
        if self.skill == "turn_left":
            yaw = SPEED_YAW[self.speed]
        elif self.skill == "turn_right":
            yaw = -SPEED_YAW[self.speed]
        elif self.skill == "walk_forward":
            yaw = self.walk_yaw_trim
        self.command[:] = 0
        self.command[0] = vx
        self.command[2] = yaw

    def _write_pick_command(self) -> None:
        phase = self._pick_phase
        self.command[:] = 0
        self.command[0] = math.cos(2.0 * math.pi * phase)
        self.command[1] = math.sin(2.0 * math.pi * phase)

    def step(self) -> StepInfo:
        if self.mode == "pick":
            self._pick_phase += CONTROL_DT / PICK_PERIOD_S
            if self._pick_phase >= PICK_END_PHASE:
                self.mode = "loco"
                self._net = self.nets["loco"]
                self.skill = "stand"
                self.speed = "stop"
                self._write_loco_command()
            else:
                self._write_pick_command()
        elif self.mode == "rise":
            self.command[:] = 0
            self._rise_left -= CONTROL_DT
            if self._rise_left <= 0.0:
                self.mode = "loco"
                self._net = self.nets["loco"]
                self.skill = self._pending_skill if self._pending_skill in LOCO_SKILLS else "stand"
                self.speed = self._pending_speed if self.skill != "stand" else "stop"
                self._write_loco_command()
        elif self.mode == "sit":
            self.command[:] = 0
            self.command[0] = 1.0
        else:
            self._write_loco_command()

        obs = self._observe()
        action = self._net.session.run(
            [self._net.output_name], {self._net.input_name: obs}
        )[0].reshape(-1).astype(np.float32)
        self.last_action[:] = action
        target = self.default_pose + action * self._net.action_scale
        self.data.ctrl[:] = target
        for _ in range(DECIMATION):
            mujoco.mj_step(self.model, self.data)
        self.sim_time += CONTROL_DT

        pos = self.data.xpos[self.trunk_id]
        quat = self.data.xquat[self.trunk_id]
        grav = _quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0], dtype=np.float64))
        upright = bool(pos[2] > 0.06 and grav[2] < -0.45)
        return StepInfo(
            skill=self.skill,
            speed=self.speed,
            mode=self.mode,
            action_norm=float(np.linalg.norm(action)),
            base_z=float(pos[2]),
            x=float(pos[0]),
            y=float(pos[1]),
            yaw=_yaw_from_quat(quat),
            upright=upright,
            twist=(float(self.command[0]), float(self.command[1]), float(self.command[2])),
        )

    def _observe(self) -> np.ndarray:
        gyro = self.data.sensordata[self.gyro_adr : self.gyro_adr + 3]
        quat = self.data.xquat[self.trunk_id]
        grav = _quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0], dtype=np.float64))
        q = self.data.qpos[self.qpos_adr]
        qd = self.data.qvel[self.qvel_adr]
        obs = np.concatenate(
            [
                np.asarray(gyro, dtype=np.float32),
                grav.astype(np.float32),
                (q - self.default_pose).astype(np.float32),
                np.asarray(qd, dtype=np.float32),
                self.last_action,
                self.command,
            ]
        ).reshape(1, OBS_DIM)
        return obs

    def set_goal(self, xy) -> None:
        self.goal_xy = (float(xy[0]), float(xy[1]))

    def paint_waypoint(self, index: int) -> None:
        """Brighten the active path disc. The orange post is the final marker."""
        for i, gid in enumerate(self.wp_geom_ids):
            hot = i == index
            self.model.geom_rgba[gid] = (
                [0.98, 0.62, 0.12, 1.0] if hot else [0.55, 0.42, 0.24, 0.9]
            )

    def goal_relative(self) -> tuple[float, float]:
        """Distance (m) and bearing (deg). Positive bearing means the marker is to the left."""
        pos = self.data.xpos[self.trunk_id]
        dx = self.goal_xy[0] - float(pos[0])
        dy = self.goal_xy[1] - float(pos[1])
        dist = math.hypot(dx, dy)
        bearing = wrap_pi(math.atan2(dy, dx) - _yaw_from_quat(self.data.xquat[self.trunk_id]))
        return dist, math.degrees(bearing)

    def pose(self) -> tuple[float, float, float, float]:
        pos = self.data.xpos[self.trunk_id]
        return float(pos[0]), float(pos[1]), float(pos[2]), _yaw_from_quat(self.data.xquat[self.trunk_id])


def _add_path(spec: mujoco.MjSpec, path) -> None:
    """Dotted line plus a disc at each intermediate waypoint. No collision."""
    if len(path) < 2:
        return
    dots = spec.worldbody.add_body(pos=[0.0, 0.0, 0.0], name="path_dots")
    dot_i = 0
    for a, b in zip(path, path[1:]):
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        length = math.hypot(bx - ax, by - ay)
        n = max(1, int(length / 0.10))
        for k in range(1, n):
            u = k / n
            dots.add_geom(
                name=f"path_dot_{dot_i}",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[0.012, 0.002, 0.0],
                pos=[ax + (bx - ax) * u, ay + (by - ay) * u, 0.004],
                rgba=[0.93, 0.55, 0.16, 0.9],
                contype=0,
                conaffinity=0,
            )
            dot_i += 1
    # The last point is the orange post. Discs mark the corners before it.
    for i, pt in enumerate(path[:-1]):
        if i == 0:
            continue
        disc = spec.worldbody.add_body(pos=[float(pt[0]), float(pt[1]), 0.0], name=f"wp_{i}")
        disc.add_geom(
            name=f"wp_disc_{i - 1}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.05, 0.004, 0.0],
            pos=[0.0, 0.0, 0.008],
            rgba=[0.98, 0.62, 0.12, 1.0],
            contype=0,
            conaffinity=0,
        )


def _load_arena(scene_xml: str, goal_xy: tuple[float, float], obstacles=(), path=()) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(scene_xml)
    # Offscreen renders for the 1080p plate are larger than the MJCF default.
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    _add_path(spec, path)
    for i, obs in enumerate(obstacles):
        hx, hy, hz = (float(v) for v in obs["size"])
        px, py, pz = (float(v) for v in obs["pos"])
        box = spec.worldbody.add_body(pos=[px, py, pz], name=f"obs_{i}")
        box.add_geom(
            name=f"obs_box_{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[hx, hy, hz],
            rgba=list(obs.get("rgba", (0.36, 0.28, 0.24, 1.0))),
        )
    goal = spec.worldbody.add_body(pos=[goal_xy[0], goal_xy[1], 0.0], name="goal_marker")
    # Flat ring plus a short post. No collision, so the gait stays the trained one.
    goal.add_geom(
        name="goal_ring",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.14, 0.004, 0.0],
        pos=[0.0, 0.0, 0.005],
        rgba=[0.93, 0.55, 0.16, 1.0],
        contype=0,
        conaffinity=0,
        group=0,
    )
    goal.add_geom(
        name="goal_post",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.028, 0.09, 0.0],
        pos=[0.0, 0.0, 0.09],
        rgba=[0.86, 0.32, 0.10, 1.0],
        contype=0,
        conaffinity=0,
        group=0,
    )
    return spec.compile()


def chase_camera(runtime: DuckRuntime, cam: mujoco.MjvCamera | None = None) -> mujoco.MjvCamera:
    """Rear-quarter view while walking, closer front view for sit and pick.

    Azimuth 42 keeps the orange marker in front of the duck. Azimuth 205 looks
    at the face, which is what makes a crouch or a sit readable.
    """
    if cam is None:
        cam = mujoco.MjvCamera()
    x, y, _, yaw = runtime.pose()
    gx, gy = runtime.goal_xy
    close = runtime.mode in ("pick", "sit", "rise")
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    if close:
        cam.lookat[:] = [x, y, 0.08]
        cam.distance = 0.95
        cam.elevation = -12
        cam.azimuth = math.degrees(yaw) + 205
    else:
        cam.lookat[:] = [x * 0.70 + gx * 0.30, y * 0.70 + gy * 0.30, 0.08]
        cam.distance = float(getattr(runtime, "chase_distance", 1.70))
        cam.elevation = float(getattr(runtime, "chase_elevation", -16.0))
        cam.azimuth = math.degrees(yaw) + 42
    return cam


def forward_camera(runtime: DuckRuntime, cam: mujoco.MjvCamera | None = None) -> mujoco.MjvCamera:
    """View along the duck's forward axis, a little above the floor.

    The MJCF head_camera looks at the sky (its quaternion is a 180 degree yaw),
    so the decision image uses this aim instead.
    """
    if cam is None:
        cam = mujoco.MjvCamera()
    x, y, _, yaw = runtime.pose()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [x + math.cos(yaw) * 0.62, y + math.sin(yaw) * 0.62, 0.04]
    cam.distance = 0.52
    cam.elevation = -16
    cam.azimuth = math.degrees(yaw)
    return cam


# Default free-camera fovy is 45 degrees. On the forward inset that leaves a
# marker at about +29 degrees on the edge of the frame. Chase views keep 45.
FORWARD_FOVY = 68.0


@contextmanager
def wider_forward_fov(model: mujoco.MjModel):
    saved = float(model.vis.global_.fovy)
    model.vis.global_.fovy = FORWARD_FOVY
    try:
        yield
    finally:
        model.vis.global_.fovy = saved


# Older name kept so a caller that tracked the body still resolves.
track_camera = chase_camera
