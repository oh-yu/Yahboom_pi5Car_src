#!/usr/bin/env python3
# encoding: utf-8
"""特定ゴールへ「必ず到達」させる手続き的デモ (P制御).

強化学習を使わず、単純な比例制御で目標へ直進する。
環境の end-to-end 検証（到達判定・報酬・ゴール表示・reset）を
学習抜きで素早く確認するために使う。

制御則:
    1. まず目標方向へ向き（方位誤差 heading_err を角速度で詰める）
    2. 概ね正面を向いたら前進（cos(heading_err) で前進量をゲーティング）

使用例:
    ros2 run yahboomcar_rl demo_reach --goal-x 1.5 --goal-y 0.5
    ros2 run yahboomcar_rl demo_reach --goal-x 1.0 --goal-y -1.0 --episodes 3
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from .gazebo_env import YahboomGoalEnv, EnvConfig


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scripted goal-reaching demo (P-control)")
    p.add_argument("--goal-x", type=float, default=1.5, help="目標 x 座標 [m]")
    p.add_argument("--goal-y", type=float, default=0.0, help="目標 y 座標 [m]")
    p.add_argument("--episodes", type=int, default=1, help="繰り返し回数")
    p.add_argument("--k-ang", type=float, default=1.5, help="角速度の比例ゲイン")
    p.add_argument("--k-lin", type=float, default=1.2, help="前進速度の比例ゲイン")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def p_control(obs: np.ndarray, k_ang: float, k_lin: float) -> np.ndarray:
    """観測から [linear, angular]（正規化済み [-1,1]）を返す P 制御."""
    dist = float(obs[2])
    # heading_err は cos/sin で観測されている → 角度に復元
    heading_err = math.atan2(float(obs[4]), float(obs[3]))

    # 目標方向へ旋回
    ang = float(np.clip(k_ang * heading_err, -1.0, 1.0))
    # 正面を向いているときだけ前進（後ろ向きに突っ込まない）
    forward_gate = max(0.0, math.cos(heading_err))
    lin = float(np.clip(k_lin * dist, 0.0, 1.0)) * forward_gate

    return np.array([lin, ang], dtype=np.float32)


def main(argv=None) -> int:
    args = parse_args(argv)
    env = YahboomGoalEnv(cfg=EnvConfig(), seed=args.seed)
    goal = (args.goal_x, args.goal_y)

    successes = 0
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep, options={"goal": goal})
        print(
            f"[ep {ep+1}] start  goal={info.get('goal')}  "
            f"dist0={float(obs[2]):.2f}"
        )
        total_r = 0.0
        length = 0
        while True:
            action = p_control(obs, args.k_ang, args.k_lin)
            obs, r, terminated, truncated, info = env.step(action)
            total_r += r
            length += 1
            if terminated or truncated:
                break
        succ = bool(info.get("is_success", False))
        successes += int(succ)
        print(
            f"[ep {ep+1}] end    return={total_r:8.2f}  len={length:3d}  "
            f"success={succ}  final_dist={float(obs[2]):.3f}  info={info}"
        )

    print("=" * 60)
    print(f"到達成功率: {successes}/{args.episodes} = {successes / args.episodes:.0%}")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
