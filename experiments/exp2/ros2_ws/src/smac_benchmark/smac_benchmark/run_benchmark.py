#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose
from smac_benchmark.plot_path import plot_path_on_map


def _wrap_to_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _yaw_to_quat(yaw: float) -> Tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _path_length_xy(points: List[Tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    s = 0.0
    for i in range(len(points) - 1):
        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]
        s += math.hypot(dx, dy)
    return s


def _curvature_stats(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Discrete local curvature approximation for each 3-point window:
      v1 = P1 - P0, v2 = P2 - P1
      angle = wrap(atan2(v2.y, v2.x) - atan2(v1.y, v1.x))
      seg_len = 0.5 * (|v1| + |v2|)
      kappa_i = |angle| / seg_len
    Returns (kappa_max, kappa_mean).
    """
    if len(points) < 3:
        return 0.0, 0.0

    kappas: List[float] = []
    for i in range(len(points) - 2):
        p0 = points[i]
        p1 = points[i + 1]
        p2 = points[i + 2]

        v1x = p1[0] - p0[0]
        v1y = p1[1] - p0[1]
        v2x = p2[0] - p1[0]
        v2y = p2[1] - p1[1]

        seg1 = math.hypot(v1x, v1y)
        seg2 = math.hypot(v2x, v2y)
        seg_len = 0.5 * (seg1 + seg2)
        if seg_len <= 1e-12:
            continue

        theta1 = math.atan2(v1y, v1x)
        theta2 = math.atan2(v2y, v2x)
        angle = _wrap_to_pi(theta2 - theta1)
        kappas.append(abs(angle) / seg_len)

    if not kappas:
        return 0.0, 0.0
    return max(kappas), sum(kappas) / len(kappas)


def _read_proc_status_value_kb(pid: int, key: str) -> Optional[int]:
    """
    Read a memory field in kB from /proc/<pid>/status (e.g., VmRSS, VmHWM).
    Returns None if not available.
    """
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(f"{key}:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])
    except Exception:
        return None
    return None


def _find_pid_by_cmd_substring(substr: str) -> Optional[int]:
    """
    Find a process whose cmdline contains `substr`.
    This is a heuristic; good enough for a controlled benchmark run.
    """
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().decode(errors="ignore").replace("\x00", " ")
            if substr in cmd:
                return pid
        except Exception:
            continue
    return None


@dataclass
class ProcessMemoryResult:
    baseline_rss_kb: Optional[int]
    peak_rss_kb: Optional[int]


@dataclass
class PlanningAttempt:
    start_yaw: float
    goal_yaw: float
    success: bool
    status_text: str
    wall_sec: float
    poses: Optional[List[PoseStamped]]


@dataclass
class BenchmarkRunConfig:
    start_xy: Tuple[float, float]
    goal_xy: Tuple[float, float]
    frame_id: str = "map"
    timeout_sec: float = 20.0
    start_yaw: float = 0.0
    goal_yaw: float = 0.0
    auto_start_yaw: bool = False
    auto_goal_yaw: bool = False
    yaw_samples: int = 16
    planner_cmd_substr: str = "planner_server"
    max_planning_attempts: Optional[int] = None


@dataclass
class BenchmarkExecutionResult:
    payload: Dict[str, object]
    waypoints_xy: List[Tuple[float, float]]
    csv_points: List[Tuple[float, float]]


def _linspace_yaws(n: int) -> List[float]:
    if n <= 1:
        return [0.0]
    step = (2.0 * math.pi) / float(n)
    return [_wrap_to_pi(-math.pi + i * step) for i in range(n)]


def _run_planning_attempts(
    node: "BenchmarkNode",
    frame_id: str,
    timeout_sec: float,
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    start_yaws: List[float],
    goal_yaws: List[float],
    max_planning_attempts: Optional[int] = None,
) -> Tuple[PlanningAttempt, List[PlanningAttempt]]:
    attempts: List[PlanningAttempt] = []
    max_attempts = None if max_planning_attempts is None or max_planning_attempts < 1 else int(max_planning_attempts)

    for syaw in start_yaws:
        for gyaw in goal_yaws:
            ok, status_text, poses, wall_sec = node.compute_path(
                start_xy=start_xy,
                goal_xy=goal_xy,
                frame_id=frame_id,
                start_yaw=syaw,
                goal_yaw=gyaw,
                timeout_sec=timeout_sec,
            )
            attempts.append(
                PlanningAttempt(
                    start_yaw=syaw,
                    goal_yaw=gyaw,
                    success=bool(ok),
                    status_text=status_text,
                    wall_sec=float(wall_sec),
                    poses=poses,
                )
            )
            if max_attempts is not None and len(attempts) >= max_attempts:
                break
        if max_attempts is not None and len(attempts) >= max_attempts:
            break

    successes = [a for a in attempts if a.success and a.poses is not None]
    if not successes:
        # Return shortest-time failure as representative.
        best_fail = min(attempts, key=lambda a: a.wall_sec)
        return best_fail, attempts

    # Select the shortest path. Use wall time as tie-breaker.
    def _rank(a: PlanningAttempt) -> Tuple[float, float]:
        pts = [(float(p.pose.position.x), float(p.pose.position.y)) for p in a.poses or []]
        return (_path_length_xy(pts), a.wall_sec)

    best = min(successes, key=_rank)
    return best, attempts



class BenchmarkNode(Node):
    def __init__(self):
        super().__init__("smac_benchmark_runner")
        self._client = ActionClient(self, ComputePathToPose, "/compute_path_to_pose")

    def compute_path(
        self,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        frame_id: str = "map",
        start_yaw: float = 0.0,
        goal_yaw: float = 0.0,
        timeout_sec: float = 20.0,
    ) -> Tuple[bool, str, Optional[List[PoseStamped]], float]:
        """
        Returns: (success, status_text, path_poses, wall_time_sec)
        """
        if not self._client.wait_for_server(timeout_sec=timeout_sec):
            return False, "ACTION_SERVER_NOT_AVAILABLE", None, 0.0

        start = PoseStamped()
        start.header.frame_id = frame_id
        start.header.stamp = self.get_clock().now().to_msg()
        start.pose.position.x = float(start_xy[0])
        start.pose.position.y = float(start_xy[1])
        qx, qy, qz, qw = _yaw_to_quat(start_yaw)
        start.pose.orientation.x = qx
        start.pose.orientation.y = qy
        start.pose.orientation.z = qz
        start.pose.orientation.w = qw

        goal = PoseStamped()
        goal.header.frame_id = frame_id
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = float(goal_xy[0])
        goal.pose.position.y = float(goal_xy[1])
        qx, qy, qz, qw = _yaw_to_quat(goal_yaw)
        goal.pose.orientation.x = qx
        goal.pose.orientation.y = qy
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw

        req = ComputePathToPose.Goal()

        if hasattr(req, "start"):
            req.start = start
        if hasattr(req, "goal"):
            req.goal = goal
        if hasattr(req, "use_start"):
            req.use_start = True

        t0 = time.perf_counter()
        goal_future = self._client.send_goal_async(req)
        rclpy.spin_until_future_complete(self, goal_future, timeout_sec=timeout_sec)
        if not goal_future.done() or goal_future.result() is None:
            return False, "GOAL_SEND_FAILED", None, time.perf_counter() - t0

        goal_handle = goal_future.result()
        if not goal_handle.accepted:
            return False, "GOAL_REJECTED", None, time.perf_counter() - t0

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout_sec)
        wall = time.perf_counter() - t0

        if not result_future.done() or result_future.result() is None:
            return False, "RESULT_TIMEOUT", None, wall

        res = result_future.result()
        status = getattr(res, "status", None)

        # status code mapping (best-effort)
        status_text = f"STATUS_{status}"
        success = False
        if status == 4:
            success = True
            status_text = "SUCCEEDED"
        elif status == 6:
            status_text = "ABORTED"
        elif status == 5:
            status_text = "CANCELED"

        path_msg = getattr(res.result, "path", None)
        if not success or path_msg is None:
            return False, status_text, None, wall

        poses = list(path_msg.poses)
        return True, status_text, poses, wall


def run_benchmark_once(node: BenchmarkNode, config: BenchmarkRunConfig) -> BenchmarkExecutionResult:
    pid = _find_pid_by_cmd_substring(config.planner_cmd_substr)
    mem_result = ProcessMemoryResult(
        baseline_rss_kb=_read_proc_status_value_kb(pid, "VmRSS") if pid is not None else None,
        peak_rss_kb=None,
    )

    if config.yaw_samples < 1:
        raise ValueError("yaw_samples must be >= 1")

    start_yaws = _linspace_yaws(config.yaw_samples) if config.auto_start_yaw else [config.start_yaw]
    goal_yaws = _linspace_yaws(config.yaw_samples) if config.auto_goal_yaw else [config.goal_yaw]

    best_attempt, attempts = _run_planning_attempts(
        node=node,
        frame_id=config.frame_id,
        timeout_sec=config.timeout_sec,
        start_xy=config.start_xy,
        goal_xy=config.goal_xy,
        start_yaws=start_yaws,
        goal_yaws=goal_yaws,
        max_planning_attempts=config.max_planning_attempts,
    )

    if pid is not None:
        mem_result.peak_rss_kb = _read_proc_status_value_kb(pid, "VmHWM")

    waypoints_xy: List[Tuple[float, float]] = []
    waypoints_out: List[Dict[str, float]] = []
    if best_attempt.poses is not None:
        for ps in best_attempt.poses:
            x = float(ps.pose.position.x)
            y = float(ps.pose.position.y)
            waypoints_xy.append((x, y))
            waypoints_out.append({"x": x, "y": y})

    length_m = _path_length_xy(waypoints_xy)
    k_max, k_mean = _curvature_stats(waypoints_xy)
    density = (len(waypoints_xy) / length_m) if length_m > 1e-12 else 0.0

    payload: Dict[str, object] = {
        "timestamp_unix": time.time(),
        "success": bool(best_attempt.success),
        "status": best_attempt.status_text,
        "planning_time_sec": float(best_attempt.wall_sec),
        "memory_rss_kb_baseline": mem_result.baseline_rss_kb,
        "memory_peak_rss_kb": mem_result.peak_rss_kb,
        "memory_rss_kb_max": mem_result.peak_rss_kb,
        "start_xy": [config.start_xy[0], config.start_xy[1]],
        "goal_xy": [config.goal_xy[0], config.goal_xy[1]],
        "frame_id": config.frame_id,
        "selected_start_yaw_rad": float(best_attempt.start_yaw),
        "selected_goal_yaw_rad": float(best_attempt.goal_yaw),
        "path_waypoints": waypoints_out,
        "path_waypoint_count": len(waypoints_out),
        "path_length_m": float(length_m),
        "curvature_max_1_per_m": float(k_max),
        "curvature_mean_1_per_m": float(k_mean),
        "waypoint_density_1_per_m": float(density),
        "notes": {
            "yaw_free_expected": True,
            "auto_start_yaw": bool(config.auto_start_yaw),
            "auto_goal_yaw": bool(config.auto_goal_yaw),
            "yaw_samples": int(config.yaw_samples),
            "max_planning_attempts": config.max_planning_attempts,
            "planning_attempt_count": len(attempts),
            "planning_attempts": [
                {
                    "start_yaw_rad": float(a.start_yaw),
                    "goal_yaw_rad": float(a.goal_yaw),
                    "success": bool(a.success),
                    "status": a.status_text,
                    "planning_time_sec": float(a.wall_sec),
                }
                for a in attempts
            ],
            "planner_pid_found": pid is not None,
            "planner_pid": pid,
            "memory_peak_source": "VmHWM",
        },
    }

    csv_points: List[Tuple[float, float]] = [
        (config.start_xy[0], config.start_xy[1]),
        *waypoints_xy,
        (config.goal_xy[0], config.goal_xy[1]),
    ]
    return BenchmarkExecutionResult(payload=payload, waypoints_xy=waypoints_xy, csv_points=csv_points)


def write_benchmark_outputs(
    node: BenchmarkNode,
    result: BenchmarkExecutionResult,
    out_base: str,
    map_yaml_path: str,
    save_plot: bool = True,
) -> None:
    out_base = os.path.expanduser(out_base)
    out_dir = os.path.dirname(out_base)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    txt_out = f"{out_base}.txt"
    csv_out = f"{out_base}.csv"
    plot_out = f"{out_base}.png"

    with open(txt_out, "w", encoding="utf-8") as f:
        json.dump(result.payload, f, indent=2, ensure_ascii=False)
    node.get_logger().info(f"Wrote results to: {txt_out}")

    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["count", "x", "y", "halt"])
        for i, (x, y) in enumerate(result.csv_points):
            writer.writerow([i, x, y, 0])
    node.get_logger().info(f"Wrote waypoints CSV to: {csv_out}")

    if not save_plot:
        return

    start_xy = tuple(result.payload["start_xy"])
    goal_xy = tuple(result.payload["goal_xy"])
    try:
        plot_path_on_map(
            map_yaml_path=map_yaml_path,
            waypoints_xy=result.waypoints_xy,
            start_xy=(float(start_xy[0]), float(start_xy[1])),
            goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
            output_path=plot_out,
        )
        node.get_logger().info(f"Wrote plot image to: {plot_out}")
    except Exception as e:
        node.get_logger().warning(f"Failed to create plot image: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", nargs=2, type=float, required=True, metavar=("X", "Y"))
    ap.add_argument("--goal", nargs=2, type=float, required=True, metavar=("X", "Y"))
    ap.add_argument("--frame", type=str, default="map")
    ap.add_argument("--timeout", type=float, default=20.0)

    ap.add_argument("--start-yaw", type=float, default=0.0)
    ap.add_argument("--goal-yaw", type=float, default=0.0)
    ap.add_argument(
        "--auto_start-yaw",
        "--auto-start-yaw",
        dest="auto_start_yaw",
        action="store_true",
        help="Search multiple start yaw candidates and pick the best successful plan.",
    )
    ap.add_argument(
        "--auto_goal-yaw",
        "--auto-goal-yaw",
        dest="auto_goal_yaw",
        action="store_true",
        help="Search multiple goal yaw candidates and pick the best successful plan.",
    )
    ap.add_argument(
        "--yaw-samples",
        type=int,
        default=16,
        help="Number of yaw samples over [-pi, pi) used by auto yaw search.",
    )
    ap.add_argument(
        "--max-planning-attempts",
        type=int,
        default=0,
        help="Maximum number of planning requests per start/goal pair. 0 means no limit.",
    )
    
    ap.add_argument("--map_yaml", type=str, required=True,
                help="Path to map yaml used for plotting")

    ap.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output base path (without extension), e.g. ~/result",
    )

    ap.add_argument("--planner-cmd-substr", type=str, default="planner_server")

    args = ap.parse_args()

    rclpy.init()
    node = BenchmarkNode()

    run_config = BenchmarkRunConfig(
        start_xy=(args.start[0], args.start[1]),
        goal_xy=(args.goal[0], args.goal[1]),
        frame_id=args.frame,
        timeout_sec=args.timeout,
        start_yaw=args.start_yaw,
        goal_yaw=args.goal_yaw,
        auto_start_yaw=args.auto_start_yaw,
        auto_goal_yaw=args.auto_goal_yaw,
        yaw_samples=args.yaw_samples,
        planner_cmd_substr=args.planner_cmd_substr,
        max_planning_attempts=(args.max_planning_attempts if args.max_planning_attempts > 0 else None),
    )
    result = run_benchmark_once(node=node, config=run_config)
    write_benchmark_outputs(
        node=node,
        result=result,
        out_base=args.out,
        map_yaml_path=args.map_yaml,
        save_plot=True,
    )

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()


