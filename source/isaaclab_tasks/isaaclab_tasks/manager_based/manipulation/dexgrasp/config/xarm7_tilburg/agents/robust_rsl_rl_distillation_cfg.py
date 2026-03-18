# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL distillation runner config for teacher→student transfer with depth camera.

Uses a custom ``VisionDistillationRunner`` that builds a ``StudentTeacherVision``
module (CNN depth encoder + LSTM + MLP student, frozen MLP teacher).

Architecture aligned with DEXTRAH (NVlabs):
  - CNN: 4-layer (1→16→32→64→128) → AdaptiveAvgPool → Linear → 32D embedding
  - Student: LSTM(512, 1 layer) → MLP [512, 512, 256] → actions
  - Teacher: MLP [128, 128] (frozen, loaded from PPO checkpoint)
  - Depth preprocessing: zero out-of-range [0.50, 1.20] m, scale valid to [0, 1]
  - Aux head: predict object position (3D) from depth embedding
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlDistillationAlgorithmCfg, RslRlDistillationRunnerCfg
from isaaclab_rl.rsl_rl.distillation_cfg import RslRlDistillationStudentTeacherRecurrentCfg


@configclass
class RslRlVisionStudentTeacherCfg(RslRlDistillationStudentTeacherRecurrentCfg):
    """Extends recurrent config with depth-encoder specific fields."""

    class_name: str = "StudentTeacherVision"

    depth_obs_group: str = "student_depth"
    """Name of the observation group containing the depth image."""

    depth_embedding_dim: int = 32
    """Output dimension of the CNN depth encoder (DEXTRAH uses 32)."""

    depth_norm_min: float = 0.40
    """Near end of depth normalization range (meters). Closer pixels zeroed."""

    depth_norm_max: float = 1.20
    """Far end of depth normalization range (meters). Farther pixels zeroed."""

    aux_output_dim: int = 3
    """Auxiliary head output dimension (object position, 3D)."""


@configclass
class RslRlVisionDistillationAlgorithmCfg(RslRlDistillationAlgorithmCfg):
    """Extends distillation algorithm config with auxiliary loss fields."""

    class_name: str = "DistillationVision"

    aux_coeff: float = 1.0
    """Weight for the auxiliary object-position prediction loss."""

    backbone_freeze_iters: int = 5000
    """Number of iterations to freeze CNN backbone before finetuning."""


@configclass
class DexgraspXArm7TilburgDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Distillation runner config for robust dexgrasp with depth vision."""

    # Use our custom runner that handles CNN-encoded depth images.
    class_name: str = "VisionDistillationRunner"

    num_steps_per_env = 32
    max_iterations = 50000
    save_interval = 1000
    experiment_name = "robust_dexgrasp_distillation"

    # Observation groups:
    #   "policy"  → fed to the student (proprio + depth image)
    #   "teacher" → fed to the teacher (full privileged obs matching the PPO actor)
    obs_groups = {
        "policy": ["student", "student_depth"],
        "teacher": ["teacher"],
    }

    policy = RslRlVisionStudentTeacherCfg(
        init_noise_std=0.1,
        noise_std_type="scalar",
        student_obs_normalization=True,
        teacher_obs_normalization=True,
        # Student: LSTM(512) → MLP [512, 512, 256] (DEXTRAH-aligned)
        student_hidden_dims=[512, 512, 256],
        # Teacher: MLP [128, 128] (must match trained PPO actor)
        teacher_hidden_dims=[128, 128],
        activation="elu",
        # LSTM for the student (DEXTRAH uses 512 units)
        rnn_type="lstm",
        rnn_hidden_dim=512,
        rnn_num_layers=1,
        # Teacher is a plain MLP (from PPO), not recurrent
        teacher_recurrent=False,
        # Depth encoder (32D embedding, DEXTRAH-aligned)
        depth_obs_group="student_depth",
        depth_embedding_dim=32,
        depth_norm_min=0.40,
        depth_norm_max=1.20,
        aux_output_dim=3,
    )

    algorithm = RslRlVisionDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1e-4,
        gradient_length=20,
        max_grad_norm=1.0,
        loss_type="mse",
        aux_coeff=1.0,
        backbone_freeze_iters=0,
    )
