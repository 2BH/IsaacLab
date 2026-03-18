"""Verify depth camera setup for the distillation environment.

Spawns the distill env, runs a few steps, and saves diagnostic depth images:
  1. raw_depth.png       — raw depth values (auto-scaled for visualization)
  2. normalized_depth.png — after [NORM_MIN, NORM_MAX] clipping + [0,1] scaling
  3. depth_histogram.png  — histogram of raw depth values (to tune norm range)

Also prints min/max/mean depth statistics for the workspace.

Usage:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/verify_depth.py \
        --task Isaac-RobustDexgrasp-XArm7-Tilburg-Distill-v0 \
        --num_envs 4 --headless --enable_cameras
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify depth camera setup.")
parser.add_argument("--task", type=str, default="Isaac-RobustDexgrasp-XArm7-Tilburg-Distill-v0")
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

sys.argv = [sys.argv[0]]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # Find the depth camera sensor
    depth_sensor = None
    for name, sensor in env.unwrapped.scene.sensors.items():
        if "depth" in name.lower() or "camera" in name.lower():
            depth_sensor = sensor
            sensor_name = name
            break

    if depth_sensor is None:
        print("[ERROR] No depth camera found in scene sensors!")
        env.close()
        return

    print(f"[INFO] Found depth sensor: '{sensor_name}'")
    print(f"[INFO] Data types: {depth_sensor.cfg.data_types}")
    print(f"[INFO] Resolution: {depth_sensor.cfg.width}x{depth_sensor.cfg.height}")
    print(f"[INFO] Clipping range: {depth_sensor.cfg.spawn.clipping_range}")
    print(f"[INFO] Camera position: {depth_sensor.cfg.offset.pos}")
    print(f"[INFO] Camera rotation: {depth_sensor.cfg.offset.rot}")
    print(f"[INFO] Convention: {depth_sensor.cfg.offset.convention}")

    # Reset and run a few steps to get stable depth data
    obs, _ = env.reset()
    for i in range(10):
        action = torch.zeros(
            env.unwrapped.num_envs,
            env.unwrapped.action_manager.total_action_dim,
            device=env.unwrapped.device,
        )
        obs, _, _, _, _ = env.step(action)

    # Get raw depth data
    depth_data = depth_sensor.data.output["distance_to_camera"]
    print(f"\n[INFO] Depth tensor shape: {depth_data.shape}")
    print(f"[INFO] Depth tensor dtype: {depth_data.dtype}")
    print(f"[INFO] Depth tensor device: {depth_data.device}")

    # Output directory
    out_dir = os.path.join(os.path.dirname(__file__), "depth_verification")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[INFO] Saving outputs to: {out_dir}")

    # Analyze each env
    for env_idx in range(min(depth_data.shape[0], 4)):
        depth_img = depth_data[env_idx].cpu().numpy()
        if depth_img.ndim == 3 and depth_img.shape[-1] == 1:
            depth_img = depth_img[..., 0]

        # Replace inf with nan for stats
        depth_clean = depth_img.copy()
        depth_clean[~np.isfinite(depth_clean)] = np.nan

        print(f"\n--- Env {env_idx} ---")
        if np.all(np.isnan(depth_clean)):
            print("  [WARN] ALL pixels are inf/nan — camera sees nothing!")
            continue

        valid = np.isfinite(depth_clean)
        print(f"  Valid pixels: {valid.sum()} / {depth_img.size} ({100 * valid.sum() / depth_img.size:.1f}%)")
        print(f"  Depth range:  [{np.nanmin(depth_clean):.4f}, {np.nanmax(depth_clean):.4f}] m")
        print(f"  Depth mean:   {np.nanmean(depth_clean):.4f} m")
        print(f"  Depth std:    {np.nanstd(depth_clean):.4f} m")

        # Percentiles
        vals = depth_clean[valid]
        for p in [1, 5, 25, 50, 75, 95, 99]:
            print(f"  P{p:02d}: {np.percentile(vals, p):.4f} m")

        # --- Visualizations ---
        try:
            from PIL import Image
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # 1. Raw depth (auto-scaled)
            d_min, d_max = np.nanmin(depth_clean), np.nanmax(depth_clean)
            raw_norm = np.clip((depth_clean - d_min) / max(d_max - d_min, 1e-6), 0, 1)
            raw_norm = np.nan_to_num(raw_norm, nan=0.0)
            raw_vis = (raw_norm * 255).astype(np.uint8)
            Image.fromarray(raw_vis, mode="L").save(os.path.join(out_dir, f"env{env_idx}_raw_depth.png"))
            print(f"  Saved: env{env_idx}_raw_depth.png")

            # 2. Normalized depth (simulating what the CNN sees)
            NORM_MIN = 0.40
            NORM_MAX = 1.20
            norm_depth = depth_clean.copy()
            invalid_mask = (~np.isfinite(norm_depth)) | (norm_depth < NORM_MIN) | (norm_depth > NORM_MAX)
            norm_depth = (norm_depth - NORM_MIN) / (NORM_MAX - NORM_MIN)
            norm_depth = np.clip(norm_depth, 0, 1)
            norm_depth[invalid_mask] = 0.0
            norm_vis = (norm_depth * 255).astype(np.uint8)
            Image.fromarray(norm_vis, mode="L").save(os.path.join(out_dir, f"env{env_idx}_normalized_depth.png"))
            print(f"  Saved: env{env_idx}_normalized_depth.png")

            # Stats after normalization
            valid_norm = norm_depth > 0
            if valid_norm.any():
                print(f"  After normalization [{NORM_MIN}, {NORM_MAX}]m:")
                print(f"    Valid pixels: {valid_norm.sum()} / {norm_depth.size} "
                      f"({100 * valid_norm.sum() / norm_depth.size:.1f}%)")
                print(f"    Range: [{norm_depth[valid_norm].min():.4f}, {norm_depth[valid_norm].max():.4f}]")
                print(f"    Mean:  {norm_depth[valid_norm].mean():.4f}")
            else:
                print(f"  [WARN] After normalization [{NORM_MIN}, {NORM_MAX}]m: ALL pixels zeroed!")

            # 3. Histogram of raw depth values
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            axes[0].hist(vals, bins=100, color="steelblue", edgecolor="black", linewidth=0.3)
            axes[0].axvline(NORM_MIN, color="red", linestyle="--", label=f"norm_min={NORM_MIN}")
            axes[0].axvline(NORM_MAX, color="orange", linestyle="--", label=f"norm_max={NORM_MAX}")
            axes[0].set_xlabel("Depth (m)")
            axes[0].set_ylabel("Pixel count")
            axes[0].set_title(f"Env {env_idx}: Raw depth histogram")
            axes[0].legend()

            in_range = vals[(vals >= NORM_MIN - 0.1) & (vals <= NORM_MAX + 0.1)]
            if len(in_range) > 0:
                axes[1].hist(in_range, bins=100, color="steelblue", edgecolor="black", linewidth=0.3)
                axes[1].axvline(NORM_MIN, color="red", linestyle="--", label=f"norm_min={NORM_MIN}")
                axes[1].axvline(NORM_MAX, color="orange", linestyle="--", label=f"norm_max={NORM_MAX}")
                axes[1].set_xlabel("Depth (m)")
                axes[1].set_ylabel("Pixel count")
                axes[1].set_title(f"Env {env_idx}: Zoomed to norm range")
                axes[1].legend()

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"env{env_idx}_depth_histogram.png"), dpi=150)
            plt.close()
            print(f"  Saved: env{env_idx}_depth_histogram.png")

        except ImportError as e:
            print(f"  [WARN] Could not generate images: {e}")
            print("  Install PIL and matplotlib: pip install Pillow matplotlib")

    print(f"\n[INFO] Done. Check outputs in: {out_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
