#!/usr/bin/env python3
# encoding: utf-8
"""環境が正しく動くかを確認するランダムエージェント (sanity check).

学習を始める前に、Gazebo との通信・報酬・境界判定などを目視確認するために使う。

使用例:
    ros2 run yahboomcar_rl random_agent --episodes 3
"""
from __future__ import annotations

import argparse
import sys

from .gazebo_env import YahboomGoalEnv, EnvConfig


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random agent sanity check")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    env = YahboomGoalEnv(cfg=EnvConfig(), seed=args.seed)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        total_r = 0.0
        length = 0
        print(f"[ep {ep+1}] start  goal={info.get('goal')}  obs={obs.round(2)}")
        while True:
            action = env.action_space.sample()
            obs, r, terminated, truncated, info = env.step(action)
            total_r += r
            length += 1
            if terminated or truncated:
                break
        print(
            f"[ep {ep+1}] end    return={total_r:.2f}  len={length}  info={info}"
        )

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
