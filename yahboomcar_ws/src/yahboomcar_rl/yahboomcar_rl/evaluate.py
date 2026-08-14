#!/usr/bin/env python3
# encoding: utf-8
"""学習済みモデルで Goal-Reaching タスクを評価するスクリプト.

使用例:
    ros2 run yahboomcar_rl evaluate --model ~/rl_models/20260814_xxx/ppo_final.zip \
        --episodes 10
"""
from __future__ import annotations

import argparse
import sys
from typing import List

import numpy as np

from .gazebo_env import YahboomGoalEnv, EnvConfig


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained RL model")
    p.add_argument("--model", type=str, required=True, help=".zip モデルへのパス")
    p.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        from stable_baselines3 import PPO, SAC
    except ImportError:
        print("[ERROR] stable-baselines3 が必要です。", file=sys.stderr)
        return 1

    env = YahboomGoalEnv(cfg=EnvConfig(), seed=args.seed)
    Cls = PPO if args.algo == "ppo" else SAC
    model = Cls.load(args.model, env=env)

    ep_returns: List[float] = []
    ep_lengths: List[int] = []
    successes = 0

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        total_r = 0.0
        length = 0
        while True:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, r, terminated, truncated, info = env.step(action)
            total_r += r
            length += 1
            if terminated or truncated:
                break
        ep_returns.append(total_r)
        ep_lengths.append(length)
        succ = int(bool(info.get("is_success", False)))
        successes += succ
        print(
            f"[ep {ep+1:02d}] return={total_r:8.2f}  len={length:3d}  "
            f"success={bool(succ)}  info={info}"
        )

    ep_returns_np = np.array(ep_returns, dtype=np.float32)
    print("=" * 60)
    print(f"平均リターン : {ep_returns_np.mean():.2f} ± {ep_returns_np.std():.2f}")
    print(f"平均エピソード長: {np.mean(ep_lengths):.1f}")
    print(f"成功率       : {successes}/{args.episodes} = {successes / args.episodes:.0%}")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
