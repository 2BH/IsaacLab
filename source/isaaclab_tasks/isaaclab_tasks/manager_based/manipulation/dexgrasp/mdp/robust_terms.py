# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_inv, quat_mul

from .utils import sample_object_point_cloud

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _resolve_env_ids(env: ManagerBasedRLEnv, env_ids: Sequence[int] | slice | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def _sensor_force_w(sensor: ContactSensor, filtered_only: bool = False) -> torch.Tensor:
    if sensor.data.force_matrix_w is not None:
        fm = sensor.data.force_matrix_w
        if fm.ndim == 4:
            # Avoid vector cancellation across many matched colliders:
            # pick the strongest contact vector per environment.
            n = fm.shape[0]
            fm_flat = fm.reshape(n, -1, 3)
            mags = fm_flat.norm(dim=-1)
            idx = mags.argmax(dim=1)
            return fm_flat[torch.arange(n, device=fm.device), idx]
        if fm.ndim == 3:
            n = fm.shape[0]
            mags = fm.norm(dim=-1)
            idx = mags.argmax(dim=1)
            return fm[torch.arange(n, device=fm.device), idx]
        return fm
    if filtered_only:
        # If filtered force matrix is unavailable, return zero so this path does not
        # accidentally use unfiltered net contact force (all contacts).
        nf = sensor.data.net_forces_w
        if nf is None or nf.ndim < 2:
            raise RuntimeError("Filtered-only contact requested, but sensor net force buffer is unavailable.")
        return torch.zeros((nf.shape[0], 3), device=nf.device, dtype=nf.dtype)
    nf = sensor.data.net_forces_w
    if nf.ndim == 3:
        n = nf.shape[0]
        mags = nf.norm(dim=-1)
        idx = mags.argmax(dim=1)
        return nf[torch.arange(n, device=nf.device), idx]
    return nf


def _forces_from_sensors(
    env: ManagerBasedRLEnv, sensor_names: list[str], filtered_only: bool = False
) -> torch.Tensor:
    return torch.stack(
        [_sensor_force_w(env.scene.sensors[name], filtered_only=filtered_only) for name in sensor_names], dim=1
    )


def _body_ids(asset_cfg: SceneEntityCfg, asset: Articulation) -> list[int] | slice:
    return asset_cfg.body_ids if asset_cfg.body_ids is not None else slice(None)


def _joint_ids(asset_cfg: SceneEntityCfg, asset: Articulation) -> list[int] | slice:
    return asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)


def _num_ids(ids: list[int] | slice, total: int) -> int:
    if isinstance(ids, slice):
        return len(range(*ids.indices(total)))
    return len(ids)


def _rigid_pose_w(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    object_body_cfg: SceneEntityCfg | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    obj = env.scene[object_cfg.name]
    if isinstance(obj, Articulation):
        if object_body_cfg is None or object_body_cfg.body_ids is None:
            return obj.data.root_pos_w, obj.data.root_quat_w
        body_id = object_body_cfg.body_ids[0]
        return obj.data.body_pos_w[:, body_id], obj.data.body_quat_w[:, body_id]
    if isinstance(obj, RigidObject):
        return obj.data.root_pos_w, obj.data.root_quat_w
    raise TypeError(f"Unsupported object type for {object_cfg.name}: {type(obj)}")


def _python_finger_weights(num_contacts: int, device: str) -> torch.Tensor:
    w = torch.ones(num_contacts, device=device)
    for i in range(4):
        idx = 4 * i + 4
        if idx < num_contacts:
            w[idx] *= 4.0
    if 16 < num_contacts:
        w[16] *= 2.0
    w = w / (w.sum() + 1.0e-8)
    if num_contacts > 0:
        w[0] = 0.0
    return w * 16.0


def _contact_weights_cpp(num_contacts: int, device: str) -> torch.Tensor:
    w = torch.ones(num_contacts, device=device)
    for i in range(1, 5):
        idx = 3 * i
        if idx < num_contacts:
            w[idx] *= 3.0
    if num_contacts > 0:
        w[0] = 0.0
        w[-1] *= 2.0
    for idx in range(10, min(13, num_contacts)):
        w[idx] *= 2.0
    w = w / (w.sum() + 1.0e-8)
    return w * float(num_contacts)


def _impulse_clip_high(num_contacts: int, device: str) -> torch.Tensor:
    high = torch.full((num_contacts,), 0.3, device=device)
    if num_contacts >= 3:
        high[-3:] = 0.4
    return high


def _nearest_affordance(
    joint_pos_w: torch.Tensor,
    affordance_points_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dists = torch.cdist(joint_pos_w, affordance_points_w)
    min_dist, min_idx = dists.min(dim=2)
    nearest = torch.gather(affordance_points_w, 1, min_idx.unsqueeze(-1).expand(-1, -1, 3))
    return min_dist, nearest - joint_pos_w


def joint_target_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = _joint_ids(asset_cfg, asset)
    return asset.data.joint_pos_target[:, joint_ids] - asset.data.joint_pos[:, joint_ids]


def body_height_to_table(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    table_height: float,
) -> torch.Tensor:
    asset: Articulation = env.scene[body_asset_cfg.name]
    body_ids = _body_ids(body_asset_cfg, asset)
    return asset.data.body_pos_w[:, body_ids, 2] - table_height


def hand_center_world(
    env: ManagerBasedRLEnv,
    hand_body_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Articulation = env.scene[hand_body_cfg.name]
    body_id = hand_body_cfg.body_ids[0]
    return asset.data.body_pos_w[:, body_id]


def wrist_euler_xyz(
    env: ManagerBasedRLEnv,
    wrist_body_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Articulation = env.scene[wrist_body_cfg.name]
    body_id = wrist_body_cfg.body_ids[0]
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(asset.data.body_quat_w[:, body_id])
    return torch.stack((roll, pitch, yaw), dim=-1)


def contact_binary_from_sensors(
    env: ManagerBasedRLEnv,
    sensor_names: list[str],
    threshold: float = 0.01,
) -> torch.Tensor:
    return (_forces_from_sensors(env, sensor_names).norm(dim=-1) > threshold).float()


def force_norm_from_sensors(
    env: ManagerBasedRLEnv,
    sensor_names: list[str],
) -> torch.Tensor:
    return _forces_from_sensors(env, sensor_names).norm(dim=-1)


class WristEulerDeltaObservation(ManagerTermBase):
    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.wrist_body_cfg: SceneEntityCfg = cfg.params["wrist_body_cfg"]
        self.asset: Articulation = env.scene[self.wrist_body_cfg.name]
        self.initial_quat = torch.zeros((env.num_envs, 4), device=env.device)
        self.initial_quat[:, 0] = 1.0
        self.reset()

    def reset(self, env_ids: Sequence[int] | slice | None = None):
        ids = _resolve_env_ids(self._env, env_ids)
        body_id = self.wrist_body_cfg.body_ids[0]
        self.initial_quat[ids] = self.asset.data.body_quat_w[ids, body_id]

    def __call__(self, env: ManagerBasedRLEnv, wrist_body_cfg: SceneEntityCfg) -> torch.Tensor:
        body_id = self.wrist_body_cfg.body_ids[0]
        current_quat = self.asset.data.body_quat_w[:, body_id]
        rel_quat = quat_mul(quat_inv(self.initial_quat), current_quat)
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(rel_quat)
        return torch.stack((roll, pitch, yaw), dim=-1)


class AffordanceVectorObservation(ManagerTermBase):
    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_body_cfg: SceneEntityCfg = cfg.params["contact_body_cfg"]
        self.object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        self.object_body_cfg: SceneEntityCfg | None = cfg.params.get("object_body_cfg", None)
        self.num_points: int = cfg.params.get("num_points", 200)
        self.object_top_prim_path: str = cfg.params.get("object_top_prim_path", "{ENV_REGEX_NS}/Object/top")
        self.rotate_with_object: bool = cfg.params.get("rotate_with_object", True)
        self.robot: Articulation = env.scene[self.contact_body_cfg.name]
        self.points_local = sample_object_point_cloud(
            env.num_envs,
            self.num_points,
            self.object_top_prim_path,
            device=env.device,
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        contact_body_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        object_body_cfg: SceneEntityCfg | None = None,
        num_points: int = 200,
        object_top_prim_path: str = "{ENV_REGEX_NS}/Object/top",
        rotate_with_object: bool = True,
    ) -> torch.Tensor:
        body_ids = _body_ids(self.contact_body_cfg, self.robot)
        joint_pos_w = self.robot.data.body_pos_w[:, body_ids]
        obj_pos_w, obj_quat_w = _rigid_pose_w(env, self.object_cfg, self.object_body_cfg)
        points_w = quat_apply(obj_quat_w.unsqueeze(1).repeat(1, self.num_points, 1), self.points_local)
        points_w = points_w + obj_pos_w.unsqueeze(1)
        _, vectors = _nearest_affordance(joint_pos_w, points_w)
        if self.rotate_with_object:
            vectors = quat_apply(obj_quat_w.unsqueeze(1).repeat(1, vectors.shape[1], 1), vectors)
        return vectors.reshape(env.num_envs, -1)


class AffordanceDistanceReward(ManagerTermBase):
    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_body_cfg: SceneEntityCfg = cfg.params["contact_body_cfg"]
        self.object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        self.object_body_cfg: SceneEntityCfg | None = cfg.params.get("object_body_cfg", None)
        self.num_points: int = cfg.params.get("num_points", 200)
        self.object_top_prim_path: str = cfg.params.get("object_top_prim_path", "{ENV_REGEX_NS}/Object/top")
        self.robot: Articulation = env.scene[self.contact_body_cfg.name]
        body_ids = _body_ids(self.contact_body_cfg, self.robot)
        self.weights = _python_finger_weights(_num_ids(body_ids, self.robot.num_bodies), env.device)
        self.points_local = sample_object_point_cloud(
            env.num_envs,
            self.num_points,
            self.object_top_prim_path,
            device=env.device,
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        contact_body_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        object_body_cfg: SceneEntityCfg | None = None,
        num_points: int = 200,
        object_top_prim_path: str = "{ENV_REGEX_NS}/Object/top",
    ) -> torch.Tensor:
        body_ids = _body_ids(self.contact_body_cfg, self.robot)
        joint_pos_w = self.robot.data.body_pos_w[:, body_ids]
        obj_pos_w, obj_quat_w = _rigid_pose_w(env, self.object_cfg, self.object_body_cfg)
        points_w = quat_apply(obj_quat_w.unsqueeze(1).repeat(1, self.num_points, 1), self.points_local)
        points_w = points_w + obj_pos_w.unsqueeze(1)
        min_dist, _ = _nearest_affordance(joint_pos_w, points_w)
        return -(min_dist * self.weights).sum(dim=1)

class AffordanceDistanceTanhReward(ManagerTermBase):
    """Dense bounded reward from contact-body distance to affordance surface points."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_body_cfg: SceneEntityCfg = cfg.params["contact_body_cfg"]
        self.object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        self.object_body_cfg: SceneEntityCfg | None = cfg.params.get("object_body_cfg", None)
        self.num_points: int = cfg.params.get("num_points", 200)
        self.object_top_prim_path: str = cfg.params.get("object_top_prim_path", "{ENV_REGEX_NS}/Object/top")
        self.std: float = cfg.params.get("std", 0.25)
        self.robot: Articulation = env.scene[self.contact_body_cfg.name]
        body_ids = _body_ids(self.contact_body_cfg, self.robot)
        self.weights = _python_finger_weights(_num_ids(body_ids, self.robot.num_bodies), env.device)
        self.points_local = sample_object_point_cloud(
            env.num_envs,
            self.num_points,
            self.object_top_prim_path,
            device=env.device,
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        contact_body_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        object_body_cfg: SceneEntityCfg | None = None,
        num_points: int = 200,
        object_top_prim_path: str = "{ENV_REGEX_NS}/Object/top",
        std: float = 0.25,
    ) -> torch.Tensor:
        body_ids = _body_ids(self.contact_body_cfg, self.robot)
        joint_pos_w = self.robot.data.body_pos_w[:, body_ids]
        obj_pos_w, obj_quat_w = _rigid_pose_w(env, self.object_cfg, self.object_body_cfg)
        points_w = quat_apply(obj_quat_w.unsqueeze(1).repeat(1, self.num_points, 1), self.points_local)
        points_w = points_w + obj_pos_w.unsqueeze(1)
        min_dist, _ = _nearest_affordance(joint_pos_w, points_w)
        weighted_dist = (min_dist * self.weights).sum(dim=1) / (self.weights.sum() + 1.0e-8)
        return 1.0 - torch.tanh(weighted_dist / self.std)


def table_log_penalty(
    env: ManagerBasedRLEnv,
    contact_body_cfg: SceneEntityCfg,
    table_height: float,
    max_height: float = 0.03,
) -> torch.Tensor:
    """Log-barrier penalty for hand links near the table.

    Active range is ``max_height`` (default 3cm). Above this height, penalty is zero.
    Below, it ramps logarithmically toward the table surface.
    """
    heights = body_height_to_table(env, contact_body_cfg, table_height)
    weights = _python_finger_weights(heights.shape[1], env.device).unsqueeze(0)
    clipped = torch.clamp(heights, min=0.002, max=max_height)
    return -(torch.log(clipped / max_height) * weights).sum(dim=1)


def arm_height_log_penalty(
    env: ManagerBasedRLEnv,
    arm_body_cfg: SceneEntityCfg,
    table_height: float,
) -> torch.Tensor:
    heights = body_height_to_table(env, arm_body_cfg, table_height)
    clipped = torch.clamp(heights, min=0.002, max=0.02)
    return -torch.log(50.0 * clipped).sum(dim=1)


def arm_collision_count(
    env: ManagerBasedRLEnv,
    arm_contact_sensor_names: list[str],
    threshold: float = 0.01,
) -> torch.Tensor:
    forces = _forces_from_sensors(env, arm_contact_sensor_names)
    return (forces.norm(dim=-1) > threshold).float().sum(dim=1)


def affordance_contact_reward(
    env: ManagerBasedRLEnv,
    object_contact_sensor_names: list[str],
    threshold: float = 0.01,
) -> torch.Tensor:
    forces = _forces_from_sensors(env, object_contact_sensor_names)
    contacts = (forces.norm(dim=-1) > threshold).float()
    weights = _contact_weights_cpp(contacts.shape[1], env.device).unsqueeze(0)
    return (contacts * weights).sum(dim=1) / max(contacts.shape[1], 1)


def affordance_impulse_reward(
    env: ManagerBasedRLEnv,
    hand_object_sensor_names: list[str],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    table_height: float = 0.255,
    on_table_scale: float = 0.2,
    lift_bonus_height: float = 0.02,
) -> torch.Tensor:
    """Reward for hand-object contact force in the horizontal (XY) plane.

    Measures XY impulse magnitude across ALL hand-object contact sensors and rewards
    the agent for grasping the object with its fingers. The palm sensor weight is
    zeroed by ``_contact_weights_cpp`` to discourage palm-only contact.

    Uses a soft height-based scale: the agent earns ``on_table_scale`` (default 20%)
    of the reward while the object sits on the table (enough gradient to learn
    grasping), and the full 100% once the object is lifted by ``lift_bonus_height``
    above the table. This prevents the local optimum of pressing/squeezing without
    lifting while still providing grasping signal.

    Args:
        hand_object_sensor_names: Sensor names for every hand link's contact with the object.
        object_cfg: Scene entity for the object.
        table_height: Height of the table surface.
        on_table_scale: Fraction of reward given while object is on the table (0-1).
        lift_bonus_height: Height above table at which full reward is given.
    """
    forces_w = _forces_from_sensors(env, hand_object_sensor_names)
    # XY-plane force magnitude (horizontal squeeze / grasp force)
    impulses_xy = forces_w[..., :2].norm(dim=-1)
    high = _impulse_clip_high(impulses_xy.shape[1], env.device).unsqueeze(0)
    clipped = impulses_xy.clamp(min=0.0).minimum(high)
    weights = _contact_weights_cpp(clipped.shape[1], env.device).unsqueeze(0)
    raw = (clipped * weights).sum(dim=1)

    # Soft height scale: on_table_scale on table, ramps to 1.0 at lift_bonus_height.
    obj_pos_w, _ = _rigid_pose_w(env, object_cfg)
    height_above_table = (obj_pos_w[:, 2] - table_height).clamp(min=0.0)
    lift_frac = (height_above_table / max(lift_bonus_height, 1e-6)).clamp(max=1.0)
    scale = on_table_scale + (1.0 - on_table_scale) * lift_frac
    return raw * scale


def _good_contact_mask(
    env: ManagerBasedRLEnv,
    good_contact_tip_sensor_names: list[str],
    good_contact_threshold: float,
) -> torch.Tensor:
    """Compute dexgrasp-style good-contact mask: thumb tip + any opposing fingertip."""
    contact_norms = _forces_from_sensors(env, good_contact_tip_sensor_names).norm(dim=-1)
    if contact_norms.shape[1] >= 2:
        thumb_active = contact_norms[:, 0] > good_contact_threshold
        opposing_active = (contact_norms[:, 1:] > good_contact_threshold).any(dim=1)
        return thumb_active & opposing_active
    return (contact_norms > good_contact_threshold).any(dim=1)


def _soft_contact_gate(
    env: ManagerBasedRLEnv,
    good_contact_tip_sensor_names: list[str],
    good_contact_threshold: float,
    temperature: float = 0.5,

) -> torch.Tensor:
    """Soft contact gate: smooth sigmoid instead of hard binary mask.

    Returns a value in (0, 1) that provides gradient even when contact is weak,
    unlike the binary ``_good_contact_mask``. Requires thumb + any opposing finger.
    """
    contact_norms = _forces_from_sensors(env, good_contact_tip_sensor_names).norm(dim=-1)
    if contact_norms.shape[1] >= 2:
        thumb_sig = torch.sigmoid((contact_norms[:, 0] - good_contact_threshold) / max(temperature, 1e-6))
        opposing_sig = torch.sigmoid(
            (contact_norms[:, 1:].max(dim=1).values - good_contact_threshold) / max(temperature, 1e-6)
        )
        return thumb_sig * opposing_sig
    return torch.sigmoid(
        (contact_norms.max(dim=1).values - good_contact_threshold) / max(temperature, 1e-6)
    )


def finger_diversity_reward(
    env: ManagerBasedRLEnv,
    good_contact_tip_sensor_names: list[str],
    good_contact_threshold: float = 0.5,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Additive reward for engaging more fingertips with the object.

    Returns the mean sigmoid activation across ALL fingertip sensors (including thumb).
    Two-finger grips get ~0.5, three-finger ~0.75, four-finger ~1.0, providing a
    smooth incentive to use more fingers without penalizing partial grasps.
    """
    contact_norms = _forces_from_sensors(env, good_contact_tip_sensor_names).norm(dim=-1)
    finger_sigs = torch.sigmoid(
        (contact_norms - good_contact_threshold) / max(temperature, 1e-6)
    )
    return finger_sigs.mean(dim=1)


def lift_above_table_reward(
    env: ManagerBasedRLEnv,
    table_height: float,
    target_height: float = 0.20,
    std: float = 0.03,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object_body_cfg: SceneEntityCfg | None = None,
    good_contact_tip_sensor_names: list[str] | None = None,
    good_contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Continuous tanh lift reward toward target height above table, optionally gated by good contact."""
    obj_pos_w, _ = _rigid_pose_w(env, object_cfg, object_body_cfg)
    # Use a smooth tanh kernel on remaining lift error to reduce sensitivity to small height noise.
    height_over_table = obj_pos_w[:, 2] - table_height
    height_error = torch.clamp(target_height - height_over_table, min=0.0)
    lifted = 1.0 - torch.tanh(height_error / max(float(std), 1.0e-6))
    if good_contact_tip_sensor_names:
        lifted = lifted * _good_contact_mask(env, good_contact_tip_sensor_names, good_contact_threshold).float()
    return lifted


def fixed_hold_position_reward(
    env: ManagerBasedRLEnv,
    table_height: float,
    target_height: float = 0.20,
    target_x: float = -0.40,
    target_y: float = 0.0,
    std: float = 0.08,
    min_hold_height: float = 0.03,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object_body_cfg: SceneEntityCfg | None = None,
    good_contact_tip_sensor_names: list[str] | None = None,
    good_contact_threshold: float = 1.0,
    contact_temperature: float = 0.5,
) -> torch.Tensor:
    """Reward holding object near a fixed world-frame target position.

    Target is fixed in each environment's local frame, i.e.
    ``(target_x, target_y, table_height + target_height)`` relative to ``env_origins``.
    """
    obj_pos_w, _ = _rigid_pose_w(env, object_cfg, object_body_cfg)
    target_local = obj_pos_w.new_tensor((target_x, target_y, table_height + target_height)).unsqueeze(0)
    target_pos_w = env.scene.env_origins + target_local
    pos_error = torch.norm(obj_pos_w - target_pos_w, dim=1)
    reward = 1.0 - torch.tanh(pos_error / max(float(std), 1.0e-6))

    if good_contact_tip_sensor_names:
        reward = reward * _soft_contact_gate(
            env, good_contact_tip_sensor_names, good_contact_threshold, temperature=contact_temperature,
        )

    height_over_table = obj_pos_w[:, 2] - table_height
    reward = reward * (height_over_table > min_hold_height).float()
    return reward


class PositionTrackingReward(ManagerTermBase):
    """Dense 3D proximity reward toward a fixed hold target.

    Replaces the separate lift, hold-position, and XY-proximity rewards with a
    single term.  Reward ∈ [0, 1] based on tanh distance to the 3D target
    ``(target_x, target_y, table_height + target_height)``, gated by soft
    fingertip contact so the agent cannot score by tossing.

    The 3D distance is computed in the env-local frame so targets are
    consistent across parallel environments.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        self.object_body_cfg: SceneEntityCfg | None = cfg.params.get("object_body_cfg", None)
        self.table_height: float = cfg.params["table_height"]
        self.target_height: float = cfg.params.get("target_height", 0.20)
        self.target_x: float = cfg.params.get("target_x", -0.40)
        self.target_y: float = cfg.params.get("target_y", 0.0)
        self.std: float = cfg.params.get("std", 0.10)
        self.good_contact_tip_sensor_names: list[str] | None = cfg.params.get(
            "good_contact_tip_sensor_names", None
        )
        self.good_contact_threshold: float = cfg.params.get("good_contact_threshold", 1.0)
        self.contact_temperature: float = cfg.params.get("contact_temperature", 0.5)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        table_height: float = 0.255,
        target_height: float = 0.20,
        target_x: float = -0.40,
        target_y: float = 0.0,
        std: float = 0.10,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        object_body_cfg: SceneEntityCfg | None = None,
        good_contact_tip_sensor_names: list[str] | None = None,
        good_contact_threshold: float = 1.0,
        contact_temperature: float = 0.5,
    ) -> torch.Tensor:
        obj_pos_w, _ = _rigid_pose_w(env, self.object_cfg, self.object_body_cfg)

        # 3D target in world frame (per-env)
        target_local = obj_pos_w.new_tensor(
            (self.target_x, self.target_y, self.table_height + self.target_height)
        ).unsqueeze(0)
        target_w = env.scene.env_origins + target_local

        # 3D distance
        dist = torch.norm(obj_pos_w - target_w, dim=1)
        reward = 1.0 - torch.tanh(dist / max(self.std, 1e-6))

        # Gate by fingertip contact
        if self.good_contact_tip_sensor_names:
            contact_ok = _soft_contact_gate(
                env, self.good_contact_tip_sensor_names, self.good_contact_threshold,
                temperature=self.contact_temperature,
            )
            reward = reward * contact_ok

        return reward


class GraspSuccessReward(ManagerTermBase):
    """Dense tanh-squared success reward for holding an object near the 3-D target.

    Following the DexGrasp convention, the reward is::

        (1 - tanh(dist / std)) ** 2

    where ``dist`` is the 3-D distance to the hold target
    ``(target_x, target_y, table_height + target_height)``.  The squared
    tanh concentrates reward sharply near the target (complementing the
    broader ``PositionTrackingReward``).

    A per-env ``success`` flag latches True after ``required_hold_steps``
    consecutive steps within ``success_radius``.  The flag is read by
    ``LiftDifficultyScheduler`` for ADR and cannot be re-gained after
    release.

    When ``table_asset_name`` is provided, the real table is hidden and
    overlaid with colored visualization markers (red = not yet succeeded,
    green = success latched) following the DexGrasp convention.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        self.object_body_cfg: SceneEntityCfg | None = cfg.params.get("object_body_cfg", None)
        self.table_height: float = cfg.params["table_height"]
        self.target_height: float = cfg.params.get("target_height", 0.25)
        self.target_x: float = cfg.params.get("target_x", -0.75)
        self.target_y: float = cfg.params.get("target_y", 0.0)
        self.std: float = cfg.params.get("std", 0.1)
        self.success_radius: float = cfg.params.get("success_radius", 0.05)
        self.required_hold_steps: int = cfg.params.get("required_hold_steps", 30)

        # Per-env state
        self._hold_counter = torch.zeros(env.num_envs, device=env.device)
        self.success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        # Table color visualizer (optional)
        table_asset_name: str | None = cfg.params.get("table_asset_name", None)
        self._success_viz = None
        self._table_asset = None
        if table_asset_name is not None:
            table_asset: RigidObject = env.scene[table_asset_name]
            self._table_asset = table_asset
            # Get table spawn config from scene config
            table_spawn = getattr(env.scene.cfg, table_asset_name).spawn
            viz_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/SuccessTableMarkers",
                markers={
                    "failure": table_spawn.replace(
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.25, 0.15, 0.15), roughness=0.25,
                        ),
                        visible=True,
                    ),
                    "success": table_spawn.replace(
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.15, 0.25, 0.15), roughness=0.25,
                        ),
                        visible=True,
                    ),
                },
            )
            self._success_viz = VisualizationMarkers(viz_cfg)
            self._success_viz.set_visibility(True)

    def reset(self, env_ids: Sequence[int] | slice | None = None):
        ids = _resolve_env_ids(self._env, env_ids)
        self._hold_counter[ids] = 0.0
        self.success[ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        table_height: float = 0.255,
        target_height: float = 0.25,
        target_x: float = -0.75,
        target_y: float = 0.0,
        std: float = 0.1,
        success_radius: float = 0.05,
        required_hold_steps: int = 30,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        object_body_cfg: SceneEntityCfg | None = None,
        table_asset_name: str | None = None,
    ) -> torch.Tensor:
        obj_pos_w, _ = _rigid_pose_w(env, self.object_cfg, self.object_body_cfg)

        # 3D target in world frame (per-env)
        target_local = obj_pos_w.new_tensor(
            (self.target_x, self.target_y, self.table_height + self.target_height)
        ).unsqueeze(0)
        target_w = env.scene.env_origins + target_local

        # 3D distance
        dist = torch.norm(obj_pos_w - target_w, dim=1)

        # Dense tanh-squared reward (DexGrasp convention)
        reward = (1.0 - torch.tanh(dist / max(self.std, 1e-6))) ** 2

        # Success flag: latch after required_hold_steps within success_radius
        at_goal = dist < self.success_radius
        self._hold_counter = (self._hold_counter + 1.0) * at_goal.float()
        newly_succeeded = (~self.success) & (self._hold_counter >= self.required_hold_steps)
        self.success = self.success | newly_succeeded

        # Update table color visualization
        if self._success_viz is not None and self._table_asset is not None:
            self._success_viz.visualize(
                self._table_asset.data.root_pos_w,
                marker_indices=self.success.int(),
            )

        return reward


def table_contact_reward(
    env: ManagerBasedRLEnv,
    hand_table_sensor_names: list[str],
    threshold: float = 0.01,
) -> torch.Tensor:
    """Penalize hand-table contact count.

    Returns the weighted sum of binary contacts (NOT normalized by link count).
    Previous normalization by ``contacts.shape[1]`` capped the output at ~1.0,
    making it negligible against +20/step lift rewards.
    """
    forces = _forces_from_sensors(env, hand_table_sensor_names, filtered_only=True)
    contacts = (forces.norm(dim=-1) > threshold).float()
    weights = _contact_weights_cpp(contacts.shape[1], env.device).unsqueeze(0)
    return (contacts * weights).sum(dim=1)


def table_impulse_reward(
    env: ManagerBasedRLEnv,
    hand_table_sensor_names: list[str],
    clip_max: float = 20.0,
) -> torch.Tensor:
    """Penalize hand-table contact impulse magnitude.

    Previous version clipped per-contact forces at 0.3-0.4 N, making a 100 N smash
    indistinguishable from a gentle brush. Now uses a much higher clip (default 20 N)
    so the penalty scales with actual impact force.
    """
    norms = _forces_from_sensors(env, hand_table_sensor_names, filtered_only=True).norm(dim=-1)
    clipped = norms.clamp(min=0.0, max=clip_max)
    weights = _contact_weights_cpp(clipped.shape[1], env.device).unsqueeze(0)
    return (clipped * weights).sum(dim=1)


def push_reward(
    env: ManagerBasedRLEnv,
    object_contact_sensor_names: list[str],
    strict_threshold_sensor_names: list[str] | None = None,
    strict_threshold: float = 0.5,
    default_threshold: float = 1.0,
    max_penalty: float = 20.0,
) -> torch.Tensor:
    """Penalize downward pressing of the object into the table (Z-axis force only).

    Intentionally Z-only: horizontal (XY) forces are what a proper grasp needs
    (squeezing the object between fingers). Only the downward push component
    should be penalized to discourage slamming/pressing strategies.

    Thresholds lowered from 2.0/1.0 to 1.0/0.5 so that distributed smash forces
    (each link individually below 2N) are still caught.
    max_penalty raised from 10 to 20 to not cap out during violent impacts.
    """
    forces_w = _forces_from_sensors(env, object_contact_sensor_names)
    # Z-only: penalize downward push, not horizontal grasp force.
    z_impulse = forces_w[..., 2]
    thresholds = torch.full(
        (z_impulse.shape[1],), float(default_threshold), device=env.device, dtype=z_impulse.dtype,
    )
    if strict_threshold_sensor_names:
        low_set = set(strict_threshold_sensor_names)
        for i, sensor_name in enumerate(object_contact_sensor_names):
            if sensor_name in low_set:
                thresholds[i] = float(strict_threshold)
    out = torch.clamp(z_impulse - thresholds.unsqueeze(0), min=0.0).sum(dim=1)
    return out.clamp(max=float(max_penalty))


def arm_contact_reward(
    env: ManagerBasedRLEnv,
    arm_sensor_names: list[str],
    threshold: float = 0.01,
) -> torch.Tensor:
    contacts = (_forces_from_sensors(env, arm_sensor_names).norm(dim=-1) > threshold).float()
    return contacts.norm(dim=1)


def arm_impulse_reward(
    env: ManagerBasedRLEnv,
    arm_sensor_names: list[str],
) -> torch.Tensor:
    return _forces_from_sensors(env, arm_sensor_names).norm(dim=-1).norm(dim=1)


def hand_table_penalty(
    env: ManagerBasedRLEnv,
    contact_body_cfg: SceneEntityCfg,
    hand_table_sensor_names: list[str],
    table_height: float,
    proximity_max_height: float = 0.03,
    impulse_clip_max: float = 20.0,
    impulse_scale: float = 1.0,
) -> torch.Tensor:
    """Unified hand-table penalty: proximity + contact force.

    Combines the old ``table_log_penalty``, ``table_contact_reward``, and
    ``table_impulse_reward`` into a single term.

    * **Proximity**: log-barrier when hand links are within ``proximity_max_height``
      of the table surface (approach deterrent).
    * **Impulse**: contact force magnitude, scaled by ``impulse_scale``
      (can be ramped by curriculum for gentleness shaping).
    """
    # Proximity component (log barrier)
    heights = body_height_to_table(env, contact_body_cfg, table_height)
    clipped_h = torch.clamp(heights, min=0.002, max=proximity_max_height)
    in_range = heights < proximity_max_height
    log_pen = -torch.log(clipped_h / proximity_max_height) * in_range.float()
    proximity = log_pen.sum(dim=1)

    # Impulse component
    norms = _forces_from_sensors(env, hand_table_sensor_names, filtered_only=True).norm(dim=-1)
    impulse = norms.clamp(min=0.0, max=impulse_clip_max).sum(dim=1) * impulse_scale

    return proximity + impulse


def arm_collision_penalty(
    env: ManagerBasedRLEnv,
    arm_body_cfg: SceneEntityCfg,
    arm_collision_sensor_names: list[str],
    table_height: float,
    impulse_scale: float = 1.0,
) -> torch.Tensor:
    """Unified arm collision penalty: proximity + contact force (incl. self-collision).

    Combines the old ``arm_height_log_penalty``, ``arm_contact_reward``,
    ``arm_impulse_reward``, and ``arm_collision_count`` into a single term.

    Uses unfiltered sensors (``_all_s``) to catch self-collisions
    (arm-to-arm, arm-to-hand) in addition to arm-table and arm-object.

    * **Proximity**: log-barrier for arm links near the table.
    * **Collision force**: total contact force norm from all contacts,
      scaled by ``impulse_scale`` (can be ramped by curriculum).
    """
    # Proximity component (log barrier near table)
    heights = body_height_to_table(env, arm_body_cfg, table_height)
    clipped_h = torch.clamp(heights, min=0.002, max=0.02)
    proximity = -torch.log(50.0 * clipped_h).sum(dim=1)

    # Collision force (unfiltered — includes self-collision)
    forces = _forces_from_sensors(env, arm_collision_sensor_names)
    impulse = forces.norm(dim=-1).sum(dim=1) * impulse_scale

    return proximity + impulse


def wrist_vel_reward(
    env: ManagerBasedRLEnv,
    wrist_body_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Articulation = env.scene[wrist_body_cfg.name]
    body_id = wrist_body_cfg.body_ids[0]
    wrist_quat = asset.data.body_quat_w[:, body_id]
    wrist_lin_w = asset.data.body_lin_vel_w[:, body_id]
    wrist_lin_local = quat_apply_inverse(wrist_quat, wrist_lin_w)
    reward = wrist_lin_local.square().sum(dim=1)
    return torch.where(wrist_lin_local.norm(dim=1) > 0.25, reward * 10.0, reward)


def wrist_qvel_reward(
    env: ManagerBasedRLEnv,
    wrist_body_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Articulation = env.scene[wrist_body_cfg.name]
    body_id = wrist_body_cfg.body_ids[0]
    wrist_quat = asset.data.body_quat_w[:, body_id]
    wrist_ang_w = asset.data.body_ang_vel_w[:, body_id]
    wrist_ang_local = quat_apply_inverse(wrist_quat, wrist_ang_w)
    return wrist_ang_local.square().sum(dim=1)


def obj_vel_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    obj = env.scene[object_cfg.name]
    if isinstance(obj, Articulation):
        body_id = object_body_cfg.body_ids[0] if object_body_cfg and object_body_cfg.body_ids else 0
        vel = obj.data.body_lin_vel_w[:, body_id]
    else:
        vel = obj.data.root_lin_vel_w
    return vel.square().sum(dim=1)


def obj_qvel_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    obj = env.scene[object_cfg.name]
    if isinstance(obj, Articulation):
        body_id = object_body_cfg.body_ids[0] if object_body_cfg and object_body_cfg.body_ids else 0
        vel = obj.data.body_ang_vel_w[:, body_id]
    else:
        vel = obj.data.root_ang_vel_w
    return vel.square().sum(dim=1)


class ObjectDisplacementReward(ManagerTermBase):
    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.object_cfg: SceneEntityCfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        self.object_body_cfg: SceneEntityCfg | None = cfg.params.get("object_body_cfg", None)
        self.initial_pos = torch.zeros((env.num_envs, 3), device=env.device)
        self.reset()

    def reset(self, env_ids: Sequence[int] | slice | None = None):
        ids = _resolve_env_ids(self._env, env_ids)
        pos_w, _ = _rigid_pose_w(self._env, self.object_cfg, self.object_body_cfg)
        self.initial_pos[ids] = pos_w[ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        object_body_cfg: SceneEntityCfg | None = None,
    ) -> torch.Tensor:
        pos_w, _ = _rigid_pose_w(env, self.object_cfg, self.object_body_cfg)
        return torch.norm(pos_w - self.initial_pos, dim=1)


def arm_joint_vel_reward(
    env: ManagerBasedRLEnv,
    arm_joint_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot: Articulation = env.scene[arm_joint_cfg.name]
    joint_ids = _joint_ids(arm_joint_cfg, robot)
    vel = robot.data.joint_vel[:, joint_ids]
    scaled = torch.where(vel.abs() > 0.5, vel * 4.0, vel)
    return scaled.square().sum(dim=1)


def xy_proximity_reward(
    env: ManagerBasedRLEnv,
    target_x: float = -0.40,
    target_y: float = 0.0,
    std: float = 0.15,
    min_hold_height: float = 0.03,
    table_height: float = 0.255,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object_body_cfg: SceneEntityCfg | None = None,
    good_contact_tip_sensor_names: list[str] | None = None,
    good_contact_threshold: float = 0.5,
    contact_temperature: float = 0.5,
) -> torch.Tensor:
    """Additive reward for keeping the object near the hold target in XY.

    Decoupled from the lift reward so it provides independent gradient for
    horizontal positioning without killing the vertical lift signal.
    Gated by soft contact and minimum height so it only activates once the
    object is being held above the table.
    """
    obj_pos_w, _ = _rigid_pose_w(env, object_cfg, object_body_cfg)
    target_xy_w = env.scene.env_origins.clone()
    target_xy_w[:, 0] += target_x
    target_xy_w[:, 1] += target_y
    xy_error = torch.norm(obj_pos_w[:, :2] - target_xy_w[:, :2], dim=1)
    reward = 1.0 - torch.tanh(xy_error / max(float(std), 1e-6))

    # Gate: only reward XY positioning when object is lifted and grasped.
    height_over_table = obj_pos_w[:, 2] - table_height
    reward = reward * (height_over_table > min_hold_height).float()
    if good_contact_tip_sensor_names:
        reward = reward * _soft_contact_gate(
            env, good_contact_tip_sensor_names, good_contact_threshold, temperature=contact_temperature,
        )
    return reward


def any_body_below_table(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    table_height: float,
    margin: float = 0.0,
) -> torch.Tensor:
    return (body_height_to_table(env, body_asset_cfg, table_height) < margin).any(dim=1)
