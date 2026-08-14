# yahboomcar_rl

Yahboom pi5 Car 用の **強化学習** サンプルパッケージ。
初めての実装として **Goal-Reaching（目標到達）** タスクを Gazebo Classic 上で解く。

## タスク仕様

| 項目 | 内容 |
|------|------|
| 世界 | Gazebo (`display_gazebo.launch.py`) の空ワールド |
| ロボット | `MicroROS_robot1.urdf`（差動駆動プラグイン内蔵） |
| 状態 (7-dim) | `[dx, dy, dist, cos(heading_err), sin(heading_err), v, ω]` |
| 行動 (2-dim, 連続) | `[linear.x, angular.z]` を最大 ±0.3 m/s, ±1.0 rad/s にスケール |
| 報酬 | `+30 × Δdist` (シェイピング) − 0.01 (生存) + 100 (到達) − 50 (境界外) |
| 終了条件 | 目標との距離 < 0.2 m ／ ±3 m の境界を越える ／ 400 step 到達 |
| 目標 | 各エピソード開始時に半径 2 m 内でランダム生成 (`/reset_simulation`) |

## セットアップ

```bash
pip install "gymnasium>=0.29" "stable-baselines3[extra]>=2.3" torch numpy
```

ROS2 依存 (Humble 想定):

```bash
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-visualization-msgs
```

ビルド:

```bash
cd ~/cloudfiles/code/Users/H.OSHIMA/Yahboom_pi5Car_src/yahboomcar_ws
colcon build --packages-select yahboomcar_rl
source install/setup.bash
```

## 使い方

### 1. Gazebo 起動 (別ターミナル)

```bash
ros2 launch yahboomcar_description display_gazebo.launch.py
```

`/odom`, `/cmd_vel`, `/reset_simulation` が publish/subscribe できていることを確認：

```bash
ros2 topic list | grep -E "odom|cmd_vel"
ros2 service list | grep reset_simulation
```

### 2. Sanity check（ランダム行動で動作確認）

```bash
ros2 run yahboomcar_rl random_agent --episodes 3
```

`rviz2` で `rl_goal_marker` を表示すると、目標位置が赤球で可視化される。
Gazebo GUI 上でも各エピソード開始時に**赤い円柱** (`rl_goal` エンティティ) が
目標位置に自動配置される（`/spawn_entity` サービスを利用）。
色や大きさは `EnvConfig.goal_color_rgb` / `goal_radius` / `goal_height` で変更可能。

### 3. 学習

```bash
ros2 run yahboomcar_rl train --timesteps 50000 --algo ppo --save-dir ~/rl_models
```

TensorBoard:

```bash
tensorboard --logdir ~/rl_models
```

### 4. 評価

```bash
ros2 run yahboomcar_rl evaluate \
    --model ~/rl_models/<run_name>/ppo_final.zip \
    --episodes 10
```

## 次のステップ（デジタルツイン化への発展）

現状は Gazebo 完結。以下の順で実機との**デジタルツイン化**を進められる：

1. 実機 odom / IMU トピックをそのまま `YahboomGoalEnv` に流し、実機評価を可能にする。
2. `/reset_simulation` を「実機を手動で原点に戻す」に置き換え、goal のみ RViz でクリック入力に。
3. Sim-to-Real ギャップを埋めるため、Gazebo 側で摩擦・慣性・遅延をドメインランダマイズ。
4. 障害物センサ (2D LiDAR 追加, `libgazebo_ros_ray_sensor.so`) を URDF に足して回避タスクへ拡張。

## ファイル構成

```
yahboomcar_rl/
├── package.xml
├── setup.py / setup.cfg
├── README.md
├── launch/
│   └── goal_reach_launch.py         # Gazebo + random_agent 同時起動
├── resource/yahboomcar_rl
└── yahboomcar_rl/
    ├── __init__.py
    ├── gazebo_env.py                # Gymnasium 環境 (ROS2 ラッパー)
    ├── random_agent.py              # sanity check
    ├── train.py                     # PPO / SAC 学習
    └── evaluate.py                  # 学習済み方策の評価
```
