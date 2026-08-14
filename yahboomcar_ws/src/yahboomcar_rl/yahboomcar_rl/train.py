#!/usr/bin/env python3
# encoding: utf-8
"""PPO で Goal-Reaching タスクを学習するスクリプト.

前提: 別ターミナルで Gazebo と robot が起動していること
    ros2 launch yahboomcar_description display_gazebo.launch.py

使用例:
    ros2 run yahboomcar_rl train --timesteps 50000 --save-dir ~/rl_models
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

from .gazebo_env import YahboomGoalEnv, EnvConfig


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO training for Yahboom goal-reaching")
    p.add_argument("--timesteps", type=int, default=50_000, help="学習ステップ総数")
    p.add_argument("--save-dir", type=str, default=os.path.expanduser("~/rl_models"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    p.add_argument("--n-steps", type=int, default=1024, help="PPO ロールアウト長")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--log-name", type=str, default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.callbacks import CheckpointCallback
    except ImportError:
        print(
            "[ERROR] stable-baselines3 が見つかりません。\n"
            "  pip install stable-baselines3[extra] torch gymnasium",
            file=sys.stderr,
        )
        return 1

    os.makedirs(args.save_dir, exist_ok=True)
    run_name = args.log_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)

    env = YahboomGoalEnv(cfg=EnvConfig(), seed=args.seed)
    env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))

    if args.algo == "ppo":
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            verbose=1,
            seed=args.seed,
            tensorboard_log=log_dir,
        )
    else:
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=args.learning_rate,
            buffer_size=100_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            verbose=1,
            seed=args.seed,
            tensorboard_log=log_dir,
        )

    ckpt_cb = CheckpointCallback(
        save_freq=max(args.timesteps // 5, 1000),
        save_path=log_dir,
        name_prefix="ckpt",
    )

    try:
        model.learn(total_timesteps=args.timesteps, callback=ckpt_cb, progress_bar=False)
    except KeyboardInterrupt:
        print("[INFO] 学習を中断しました。中間モデルを保存します。")

    final_path = os.path.join(log_dir, f"{args.algo}_final.zip")
    model.save(final_path)
    print(f"[OK] モデルを保存: {final_path}")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
