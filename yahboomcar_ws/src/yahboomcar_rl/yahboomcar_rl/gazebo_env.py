#!/usr/bin/env python3
# encoding: utf-8
"""Goal-reaching 用 Gazebo 環境ラッパー (Gymnasium 互換).

ROS2 ノードとして cmd_vel を publish し、odom を subscribe する。
エピソード開始時に /reset_simulation を呼び、目標座標をランダム化する。

依存:
    pip install gymnasium numpy
    apt: ros-humble-gazebo-ros-pkgs (reset_simulation service)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "gymnasium が見つかりません。`pip install gymnasium` を実行してください。"
    ) from e

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
try:  # SetEntityState は world SDF に libgazebo_ros_state.so が必要
    from gazebo_msgs.srv import SetEntityState
    _HAS_SET_STATE = True
except ImportError:  # pragma: no cover
    SetEntityState = None  # type: ignore
    _HAS_SET_STATE = False


# Gazebo 上に配置する赤いシリンダ（衝突無し・静的）
GOAL_SDF_TEMPLATE = """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>{radius}</radius>
            <length>{length}</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
          <specular>0.3 0.3 0.3 1</specular>
          <emissive>{er} {eg} {eb} 1</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""


@dataclass
class EnvConfig:
    """環境ハイパーパラメータ."""
    # 行動空間
    max_linear: float = 0.3       # [m/s]
    max_angular: float = 1.0      # [rad/s]

    # 目標
    goal_range: float = 2.0       # 目標が生成される矩形の半辺 [m]
    goal_reach_dist: float = 0.20 # 到達判定 [m]

    # 目標の見た目 (Gazebo 上の色付きシリンダ)
    goal_entity_name: str = "rl_goal"
    goal_radius: float = 0.10
    goal_height: float = 0.30
    goal_color_rgb: Tuple[float, float, float] = (1.0, 0.1, 0.1)
    goal_emissive: float = 0.4  # 発光成分 (影でも見えるようにする)

    # 境界（この矩形の外に出たら失敗）
    bounds: float = 3.0

    # エピソード
    max_steps: int = 400
    step_dt: float = 0.1          # 制御周期 [s]

    # 報酬係数
    reward_progress: float = 30.0  # 距離短縮 1m あたり +30
    reward_step: float = -0.01     # 生存ペナルティ
    reward_goal: float = 100.0     # 到達ボーナス
    reward_fail: float = -50.0     # 境界外 or タイムアウト
    reward_action: float = -0.001  # 行動大きさペナルティ


class YahboomGoalEnv(gym.Env):
    """差動駆動ロボットの Goal-Reaching 環境."""

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        cfg: Optional[EnvConfig] = None,
        node_name: str = "yahboom_rl_env",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or EnvConfig()

        # 状態: [dx, dy, dist, cos(heading_err), sin(heading_err), lin_v, ang_v]
        high = np.array(
            [self.cfg.bounds * 2] * 2
            + [self.cfg.bounds * 4]
            + [1.0, 1.0]
            + [self.cfg.max_linear * 2, self.cfg.max_angular * 2],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # rclpy 初期化 (既に初期化済みでも安全に済ませる)
        if not rclpy.ok():
            rclpy.init()

        self._node = Node(node_name)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._pub_cmd = self._node.create_publisher(Twist, "cmd_vel", qos)
        self._pub_marker = self._node.create_publisher(Marker, "rl_goal_marker", 1)
        self._sub_odom = self._node.create_subscription(
            Odometry, "odom", self._on_odom, qos
        )
        self._reset_client = self._node.create_client(Empty, "/reset_simulation")
        self._spawn_client = self._node.create_client(SpawnEntity, "/spawn_entity")
        self._delete_client = self._node.create_client(DeleteEntity, "/delete_entity")
        if _HAS_SET_STATE:
            self._set_state_client = self._node.create_client(
                SetEntityState, "/set_entity_state"
            )
        else:
            self._set_state_client = None
        self._goal_spawned: bool = False

        # ロボット状態
        self._x: float = 0.0
        self._y: float = 0.0
        self._yaw: float = 0.0
        self._v: float = 0.0
        self._w: float = 0.0
        self._odom_ready = False

        # ゴール
        self._gx: float = 0.0
        self._gy: float = 0.0
        self._prev_dist: float = 0.0
        self._steps: int = 0

        self._rng = np.random.default_rng(seed)

        # サービス待機 (最大 10 秒)
        if not self._reset_client.wait_for_service(timeout_sec=10.0):
            self._node.get_logger().warn(
                "/reset_simulation サービスが見つかりません。"
                "Gazebo が起動しているか確認してください。"
            )
        self._spawn_client.wait_for_service(timeout_sec=5.0)
        self._delete_client.wait_for_service(timeout_sec=5.0)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _on_odom(self, msg: Odometry) -> None:
        self._x = msg.pose.pose.position.x
        self._y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        # quaternion → yaw
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._yaw = math.atan2(siny_cosp, cosy_cosp)
        self._v = msg.twist.twist.linear.x
        self._w = msg.twist.twist.angular.z
        self._odom_ready = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _spin_until(self, predicate, timeout: float = 5.0) -> bool:
        """predicate() が True になるまで rclpy をスピン."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self._node, timeout_sec=0.05)
            if predicate():
                return True
        return False

    def _publish_goal_marker(self) -> None:
        m = Marker()
        m.header.frame_id = "odom"
        m.header.stamp = self._node.get_clock().now().to_msg()
        m.ns = "rl"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(self._gx)
        m.pose.position.y = float(self._gy)
        m.pose.position.z = 0.1
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.2
        r, g, b = self.cfg.goal_color_rgb
        m.color.r = float(r)
        m.color.g = float(g)
        m.color.b = float(b)
        m.color.a = 0.9
        self._pub_marker.publish(m)

    def _goal_sdf(self) -> str:
        r, g, b = self.cfg.goal_color_rgb
        e = self.cfg.goal_emissive
        return GOAL_SDF_TEMPLATE.format(
            name=self.cfg.goal_entity_name,
            radius=self.cfg.goal_radius,
            length=self.cfg.goal_height,
            r=r, g=g, b=b,
            er=r * e, eg=g * e, eb=b * e,
        )

    def _spawn_goal_entity(self, x: float, y: float) -> bool:
        """色付きシリンダを Gazebo に spawn."""
        if not self._spawn_client.service_is_ready():
            return False
        req = SpawnEntity.Request()
        req.name = self.cfg.goal_entity_name
        req.xml = self._goal_sdf()
        req.robot_namespace = ""
        req.initial_pose.position.x = float(x)
        req.initial_pose.position.y = float(y)
        req.initial_pose.position.z = self.cfg.goal_height / 2.0
        req.initial_pose.orientation.w = 1.0
        req.reference_frame = "world"
        future = self._spawn_client.call_async(req)
        self._spin_until(lambda: future.done(), timeout=3.0)
        try:
            res = future.result()
            if res is None or not res.success:
                # 同名モデルが既に存在する場合は成功扱いにする
                self._node.get_logger().debug(
                    f"spawn_entity failed: {getattr(res, 'status_message', 'no result')}"
                )
                return False
            return True
        except Exception as e:
            self._node.get_logger().warn(f"spawn_entity exception: {e}")
            return False

    def _delete_goal_entity(self) -> None:
        if not self._delete_client.service_is_ready():
            return
        req = DeleteEntity.Request()
        req.name = self.cfg.goal_entity_name
        future = self._delete_client.call_async(req)
        self._spin_until(lambda: future.done(), timeout=2.0)

    def _move_goal_entity(self, x: float, y: float) -> bool:
        """SetEntityState でゴールを移動 (失敗時 False)."""
        if self._set_state_client is None or not self._set_state_client.service_is_ready():
            return False
        req = SetEntityState.Request()
        req.state.name = self.cfg.goal_entity_name
        req.state.pose.position.x = float(x)
        req.state.pose.position.y = float(y)
        req.state.pose.position.z = self.cfg.goal_height / 2.0
        req.state.pose.orientation.w = 1.0
        req.state.reference_frame = "world"
        future = self._set_state_client.call_async(req)
        self._spin_until(lambda: future.done(), timeout=1.0)
        try:
            res = future.result()
            return bool(res is not None and res.success)
        except Exception:
            return False

    def _place_goal(self, x: float, y: float) -> None:
        """Gazebo 上のゴール標識を (x, y) に配置する.

        まだ spawn 前 → spawn。既にある → set_entity_state で移動、
        失敗すれば delete+spawn でリカバリする。
        """
        if not self._goal_spawned:
            if self._spawn_goal_entity(x, y):
                self._goal_spawned = True
            return
        if self._move_goal_entity(x, y):
            return
        # フォールバック: delete して再 spawn
        self._delete_goal_entity()
        if self._spawn_goal_entity(x, y):
            self._goal_spawned = True
        else:
            self._goal_spawned = False

    def _obs(self) -> np.ndarray:
        dx = self._gx - self._x
        dy = self._gy - self._y
        dist = math.hypot(dx, dy)
        # ロボット座標系での目標方位誤差
        goal_yaw = math.atan2(dy, dx)
        heading_err = goal_yaw - self._yaw
        # 正規化
        heading_err = math.atan2(math.sin(heading_err), math.cos(heading_err))
        return np.array(
            [dx, dy, dist, math.cos(heading_err), math.sin(heading_err), self._v, self._w],
            dtype=np.float32,
        )

    def _sample_goal(self) -> Tuple[float, float]:
        r = self.cfg.goal_range
        # 原点付近を避けて 0.5m 以上離す
        for _ in range(20):
            gx = float(self._rng.uniform(-r, r))
            gy = float(self._rng.uniform(-r, r))
            if math.hypot(gx, gy) > 0.5:
                return gx, gy
        return r * 0.8, 0.0

    def _stop_robot(self) -> None:
        self._pub_cmd.publish(Twist())

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options=None):  # type: ignore[override]
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # ロボット停止
        self._stop_robot()

        # Gazebo をリセット
        if self._reset_client.service_is_ready():
            future = self._reset_client.call_async(Empty.Request())
            self._spin_until(lambda: future.done(), timeout=3.0)
        else:
            self._node.get_logger().warn("reset_simulation 未接続。物理リセットをスキップ。")

        # odom が原点付近に戻るまで少し待つ
        self._odom_ready = False
        self._spin_until(lambda: self._odom_ready, timeout=2.0)
        # 数フレーム余分に取り込み最新値へ
        for _ in range(3):
            rclpy.spin_once(self._node, timeout_sec=0.05)

        self._gx, self._gy = self._sample_goal()
        self._steps = 0
        self._prev_dist = math.hypot(self._gx - self._x, self._gy - self._y)
        self._place_goal(self._gx, self._gy)
        self._publish_goal_marker()

        return self._obs(), {"goal": (self._gx, self._gy)}

    def step(self, action: np.ndarray):  # type: ignore[override]
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        a = np.clip(a, -1.0, 1.0)
        cmd = Twist()
        cmd.linear.x = float(a[0]) * self.cfg.max_linear
        cmd.angular.z = float(a[1]) * self.cfg.max_angular
        self._pub_cmd.publish(cmd)

        # 制御周期ぶん spin して odom を反映
        t0 = time.time()
        while time.time() - t0 < self.cfg.step_dt:
            rclpy.spin_once(self._node, timeout_sec=0.01)

        self._steps += 1
        obs = self._obs()
        dist = float(obs[2])

        # 報酬
        progress = self._prev_dist - dist
        reward = (
            self.cfg.reward_progress * progress
            + self.cfg.reward_step
            + self.cfg.reward_action * float(np.sum(a**2))
        )
        self._prev_dist = dist

        terminated = False
        truncated = False
        info: dict = {}

        if dist < self.cfg.goal_reach_dist:
            reward += self.cfg.reward_goal
            terminated = True
            info["is_success"] = True
        elif abs(self._x) > self.cfg.bounds or abs(self._y) > self.cfg.bounds:
            reward += self.cfg.reward_fail
            terminated = True
            info["is_success"] = False
            info["reason"] = "out_of_bounds"
        elif self._steps >= self.cfg.max_steps:
            truncated = True
            info["is_success"] = False
            info["reason"] = "timeout"

        if terminated or truncated:
            self._stop_robot()

        return obs, float(reward), terminated, truncated, info

    def render(self):  # noqa: D401
        """Marker を再送するのみ (Gazebo GUI で観察)."""
        self._publish_goal_marker()

    def close(self) -> None:
        try:
            self._stop_robot()
            if self._goal_spawned:
                self._delete_goal_entity()
            self._node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()
