"""ROS subscriber layer feeding the pure assembler in ``observation.py``.

Subscribes ``/scan``, ``/odom`` and ``/clock``, and blocks until both sensor
topics carry a stamp at or beyond a requested sim time. All arithmetic lives in
``observation.py``; this file only moves data.

The one rule that matters
-------------------------
``get_obs`` **raises** on timeout. It does not return the previous
observation, it does not log a warning and carry on, and it does not
extrapolate. A staleness fallback is the most tempting change in this codebase
and the most damaging: it converts a loud, immediate failure into a quiet
distribution shift that survives training and surfaces only as an unexplained
success-rate drop after deployment.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np
from nav_msgs.msg import Odometry
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan

from robot_rl_env import contract
from robot_rl_env.observation import assemble_observation, pool_ranges, yaw_from_quaternion

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)
"""Reliable, deliberately.

Best-effort would match any publisher and so could never produce a QoS
mismatch -- but a single dropped scan means waiting for a stamp that will never
arrive, i.e. a hard failure one step later anyway, with a less informative
message. Reliable either matches (and never drops) or fails immediately at
startup, which is the diagnosable end of the trade.
"""


class ObservationTimeout(RuntimeError):
    """Sensor data did not reach the requested sim time. See the module docstring."""


def _stamp_ns(header) -> int:
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


@dataclass(frozen=True)
class Sample:
    """One consistent snapshot of ``/scan`` + ``/odom``.

    Handing the caller a snapshot rather than assembling immediately is what
    lets ``env.reset()`` compute the goal *from the robot pose it just read* and
    then assemble an observation against that exact same pose. Calling
    ``get_obs`` twice to do the same job would let a scan that lands between the
    two calls pair a goal derived from pose A with an observation built from
    pose B -- a 50 ms frame offset that no assertion would ever catch.
    """

    pooled: np.ndarray
    x: float
    y: float
    yaw: float
    stamp_ns: int

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)


class ObservationAssembler(Node):
    """Latest-sample cache over ``/scan`` + ``/odom``, with a sim-time barrier."""

    def __init__(self, node_name: str = "observation_assembler", *, context=None):
        super().__init__(node_name, context=context)
        self._group = ReentrantCallbackGroup()
        self._cv = threading.Condition()

        self._scan_stamp_ns: int | None = None
        self._scan_pooled: np.ndarray | None = None
        self._odom_stamp_ns: int | None = None
        self._odom_pose: tuple[float, float, float] | None = None
        self._clock_ns: int = 0

        self.create_subscription(
            LaserScan, "/scan", self._on_scan, SENSOR_QOS, callback_group=self._group
        )
        self.create_subscription(
            Odometry, "/odom", self._on_odom, SENSOR_QOS, callback_group=self._group
        )
        self.create_subscription(
            Clock, "/clock", self._on_clock, SENSOR_QOS, callback_group=self._group
        )

    # --- callbacks ------------------------------------------------------------

    def _on_scan(self, msg: LaserScan) -> None:
        # Pool inside the callback so the expensive part happens off the step
        # loop's critical path, and so the lock is held only for the assignment.
        pooled = pool_ranges(msg.ranges)
        stamp = _stamp_ns(msg.header)
        with self._cv:
            self._scan_pooled = pooled
            self._scan_stamp_ns = stamp
            self._cv.notify_all()

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        pose = (p.x, p.y, yaw_from_quaternion(q.x, q.y, q.z, q.w))
        stamp = _stamp_ns(msg.header)
        with self._cv:
            self._odom_pose = pose
            self._odom_stamp_ns = stamp
            self._cv.notify_all()

    def _on_clock(self, msg: Clock) -> None:
        with self._cv:
            self._clock_ns = msg.clock.sec * 1_000_000_000 + msg.clock.nanosec
            self._cv.notify_all()

    # --- accessors ------------------------------------------------------------

    def sim_time_ns(self) -> int:
        with self._cv:
            return self._clock_ns

    def wait_for_first_message(self, timeout: float = 30.0) -> None:
        """Block until every topic has produced at least one message, or raise."""
        def ready() -> bool:
            return self._scan_stamp_ns is not None and self._odom_stamp_ns is not None

        with self._cv:
            if not self._cv.wait_for(ready, timeout=timeout):
                raise ObservationTimeout(self._diagnose(timeout))

    # --- the barrier ----------------------------------------------------------

    def wait_for_sample(
        self,
        min_stamp_ns: int,
        *,
        timeout: float = contract.OBS_TIMEOUT,
    ) -> Sample:
        """Block until both sensors reach ``min_stamp_ns``, then snapshot them.

        The stamp is the barrier, and nothing else may stand in for it. "Wait
        for a message newer than the one I last saw" is the tempting substitute
        and it is unsound: advancing the world by many iterations at once (the
        reset brake is 1000) queues a burst of samples, and the first to arrive
        afterwards is the *oldest* of that burst -- a scan from the middle of
        the advance, presented as the one from the end. Sim time here only ever
        moves forward, so the stamp alone is sufficient as well as necessary.

        Raises ``ObservationTimeout`` if the data does not arrive. It never
        returns stale data. See the module docstring.
        """

        def fresh() -> bool:
            return (
                self._scan_stamp_ns is not None
                and self._odom_stamp_ns is not None
                and self._scan_stamp_ns >= min_stamp_ns
                and self._odom_stamp_ns >= min_stamp_ns
            )

        with self._cv:
            if not self._cv.wait_for(fresh, timeout=timeout):
                raise ObservationTimeout(self._diagnose(timeout, min_stamp_ns))
            rx, ry, ryaw = self._odom_pose
            return Sample(
                pooled=self._scan_pooled,
                x=rx,
                y=ry,
                yaw=ryaw,
                # The older of the two: the sample as a whole is only as fresh
                # as its stalest component.
                stamp_ns=min(self._scan_stamp_ns, self._odom_stamp_ns),
            )

    @staticmethod
    def assemble(
        sample: Sample,
        goal_xy: tuple[float, float],
        prev_action,
        step_count: int,
    ) -> tuple[np.ndarray, dict]:
        """Turn a snapshot into ``(obs, info)``. Pure; no waiting, no ROS."""
        obs = assemble_observation(
            None,
            sample.xy,
            sample.yaw,
            goal_xy,
            prev_action,
            step_count,
            pooled=sample.pooled,
        )
        info = {
            "distance_to_goal": math.dist(sample.xy, goal_xy),
            "min_lidar": float(sample.pooled.min()),
            "sim_time": sample.stamp_ns / 1e9,
            "robot_pose": (sample.x, sample.y, sample.yaw),
        }
        return obs, info

    def get_obs(
        self,
        min_stamp_ns: int,
        goal_xy: tuple[float, float],
        prev_action,
        step_count: int,
        timeout: float = contract.OBS_TIMEOUT,
    ) -> tuple[np.ndarray, dict]:
        """:meth:`wait_for_sample` followed by :meth:`assemble`."""
        sample = self.wait_for_sample(min_stamp_ns, timeout=timeout)
        return self.assemble(sample, goal_xy, prev_action, step_count)

    def _diagnose(self, timeout: float, min_stamp_ns: int | None = None) -> str:
        """Turn a timeout into a message that names the likely cause.

        Silence is the default failure mode of a misconfigured bridge, so the
        exception has to do the work a log line would not.
        """
        lines = [f"no observation within {timeout}s of wall clock."]
        if min_stamp_ns is not None:
            lines.append(
                f"waiting for sim stamp >= {min_stamp_ns / 1e9:.3f}s; "
                f"scan at {self._fmt(self._scan_stamp_ns)}, "
                f"odom at {self._fmt(self._odom_stamp_ns)}, "
                f"clock at {self._clock_ns / 1e9:.3f}s."
            )
        for topic in ("/scan", "/odom", "/clock"):
            n = self.count_publishers(topic)
            if n == 0:
                lines.append(f"{topic}: NO PUBLISHER -- ros_gz_bridge is not bridging it.")
            else:
                lines.append(f"{topic}: {n} publisher(s) present.")
        lines.append(
            "If publishers exist but nothing arrives, suspect a QoS mismatch "
            "(this node subscribes RELIABLE) or a sensor update_rate that is "
            "not an exact multiple of the step boundary -- see the header "
            "comment in models/diffbot/model.sdf."
        )
        return " ".join(lines)

    @staticmethod
    def _fmt(stamp_ns: int | None) -> str:
        return "never" if stamp_ns is None else f"{stamp_ns / 1e9:.3f}s"
