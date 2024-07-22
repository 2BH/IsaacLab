# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg, patterns
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import sample_uniform
##
# Pre-defined configs
##
# from omni.isaac.lab_assets.anymal import ANYMAL_C_CFG  # isort: skip
from omni.isaac.lab_assets.unitree import UNITREE_A1_CFG  # isort: skip
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG, FLAT_ROUGH_TERRAINS_CFG  # isort: skip
from omni.isaac.lab_tasks.direct.unitree_a1.unitree_a1_env import UnitreeA1Env, UnitreeA1FlatEnvCfg, UnitreeA1FlatRoughEnvCfg, UnitreeA1RoughEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg
from omni.isaac.lab.managers import SceneEntityCfg


@configclass
class UnitreeA1NoisyCfg(UnitreeA1FlatRoughEnvCfg):
    # Friction randomization
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.0, 2.0),
            "dynamic_friction_range": (0.0, 2.),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # Bass mass randomization
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # Base mass center randomization
    modify_base_mass_center = EventTerm(
        func=mdp.randomize_body_coms,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "max_displacement_x": (-0.05, 0.15),
            "max_displacement_y": (-0.1, 0.1),
            "max_displacement_z": (-0.05, 0.05),
            "distribution": "uniform",
            "operation": "add",
        },
    )
    
    # external force and torque disturbance
    # currently not used
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.2, 0.6), "y": (-0.25, 0.25), "yaw": (-3.14, 3.14)},
            # "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )
     
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (45, 55),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    action_delay_range = (0.0, 0.0)
    propriocetive_latency_range = (0.04-0.0025, 0.04+0.0075)
    proprioceptive_latency_resample_time = 2.0

class UnitreeA1NoisyEnv(UnitreeA1Env):
    cfg: UnitreeA1NoisyCfg

    def __init__(self, cfg: UnitreeA1NoisyCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        height_data = None
        if isinstance(self.cfg, UnitreeA1RoughEnvCfg):
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            ).clip(-1.0, 1.0)
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self._commands,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    height_data,
                    self._actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )
        # undersired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._underisred_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undersired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        # Sample new commands
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

    def build_action_delay_buffer(self):
        """ Used in pre-physics step """
        action_history_buffer_length = int((self.cfg.action_delay_range[1] + self.cfg.sim.dt) / self.cfg.sim.dt)
        self.actions_history_buffer = torch.zeros(
                (
                    action_history_buffer_length,
                    self.num_envs,
                    self.num_actions,
                ),
                dtype= torch.float32,
                device= self.device,
            )

        self.action_delay_buffer = sample_uniform(
            self.cfg.action_delay_range[0],
            self.cfg.action_delay_range[1],
            (self.num_envs, 1),
            device= self.device,
        ).to(torch.float32).flatten()

        self.action_delayed_frames = ((self.action_delay_buffer / self.cfg.sim.dt) + 1).to(int)

    def build_obs_buffers_for_component(self, component=None, sensor_name=None):
        buffer_length = int(self.cfg.propriocetive_latency_range[1] / self.cfg.sim.dt) + 1
        # use super().get_obs_segment_from_components() to get the obs shape to prevent post processing
        # overrides the buffer shape
        self.obs_buffer = torch.zeros(
            (
                buffer_length,
                self.num_envs,
                self.cfg.num_observations, # tuple(obs_shape)
            ),
            dtype= torch.float32,
            device= self.device,
        )
        self.obs_latency_buffer = sample_uniform(
            self.cfg.propriocetive_latency_range[0],
            self.cfg.propriocetive_latency_range[1],
            (self.num_envs, 1),
            device= self.device,
        ).to(torch.float32).flatten()

    def _resample_sensor_latency(self):
        resampling_time = self.cfg.proprioceptive_latency_resample_time
        resample_env_ids = (self.episode_length_buf % int(resampling_time / self.cfg.sim.dt) == 0).nonzero(as_tuple= False).flatten()
        if len(resample_env_ids) > 0:
            sample_uniform(
                self.cfg.propriocetive_latency_range[0],
                self.cfg.propriocetive_latency_range[1],
                (len(resample_env_ids), 1),
                device= self.device,
            ).flatten()

            self.obs_latency_buffer[resample_env_ids] = sample_uniform(
                self.cfg.propriocetive_latency_range[0],
                self.cfg.propriocetive_latency_range[1],
                (len(resample_env_ids), 1),
                device= self.device,
            ).flatten()
        
    def _resample_action_delay(self, env_ids):
        self.action_delay_buffer[env_ids] = sample_uniform(
            self.cfg.action_delay_range[0],
            self.cfg.action_delay_range[1],
            (len(env_ids), 1),
            device= self.device,
        ).flatten()

    # def _reset_buffers(self, env_ids):
    #     return_ = super()._reset_buffers(env_ids)
    #     if hasattr(self, "actions_history_buffer"):
    #         self.actions_history_buffer[:, env_ids] = 0.
    #         self.action_delayed_frames[env_ids] = self.cfg.control.action_history_buffer_length
    #     for sensor_name in self.available_sensors:
    #         if not hasattr(self.cfg.sensor, sensor_name):
    #             continue
    #         for component in getattr(self.cfg.sensor, sensor_name).obs_components:
    #             if hasattr(self, component + "_obs_buffer"):
    #                 getattr(self, component + "_obs_buffer")[:, env_ids] = 0.
    #                 setattr(self, component + "_obs_refreshed", False)
    #         if "camera" in sensor_name:
    #             getattr(self, sensor_name + "_delayed_frames")[env_ids] = 0
    #     return return_

    