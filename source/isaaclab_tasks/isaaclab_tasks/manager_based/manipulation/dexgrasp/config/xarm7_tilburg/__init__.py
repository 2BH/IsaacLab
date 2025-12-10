# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Dextra Kuka Allegro environments.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# State Observation
gym.register(
    id="Isaac-Dexgrasp-XArm7-Tilburg-Reorient-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexgrasp_xarm7_tilburg_env_cfg:DexgraspXArm7TilburgReorientEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexgraspXArm7TilburgPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-dexgrasp-Kuka-Allegro-Reorient-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexgrasp_xarm7_tilburg_env_cfg:DexgraspXArm7TilburgReorientEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexgraspXArm7TilburgPPORunnerCfg",
    },
)

# dexgrasp Lift Environments
gym.register(
    id="Isaac-Dexgrasp-XArm7-Tilburg-Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexgrasp_xarm7_tilburg_env_cfg:DexgraspXArm7TilburgLiftEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexgraspXArm7TilburgPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Dexgrasp-XArm7-Tilburg-Lift-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexgrasp_xarm7_tilburg_env_cfg:DexgraspXArm7TilburgLiftEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexgraspXArm7TilburgPPORunnerCfg",
    },
)
