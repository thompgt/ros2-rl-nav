#!/usr/bin/env bash
#
# Phase 1 verification. Every check prints PASS or FAIL with a reason.
#
# This script exists because a ros_gz_bridge misconfiguration produces SILENCE,
# not an error. Gazebo starts, the bridge starts, nothing complains, and no data
# ever flows. That failure is invisible in logs and is where this kind of
# project loses three days. Each check below converts a specific flavour of
# silence into a non-zero exit code.
#
# Usage:  scripts/verify_phase1.sh      (from /ws inside the container)

set -uo pipefail

# ROS's setup scripts read unset variables, so -u has to come off around them
# or sourcing dies on AMENT_TRACE_SETUP_FILES before anything runs.
set +u
source /opt/ros/jazzy/setup.bash
[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash
set -u

PASS=0
FAIL=0
GZ_PID=""

pass() { echo "  PASS  $1"; PASS=$((PASS + 1)); }
fail() { echo "  FAIL  $1"; FAIL=$((FAIL + 1)); }

cleanup() {
  echo
  echo "--- cleanup ---"
  [ -n "$GZ_PID" ] && kill "$GZ_PID" 2>/dev/null
  pkill -f "gz sim"          2>/dev/null
  pkill -f parameter_bridge  2>/dev/null
  sleep 1
  pkill -9 -f "gz sim"       2>/dev/null
  return 0
}
trap cleanup EXIT

echo "=== Phase 1 verification ==="
echo

# --- build -------------------------------------------------------------------
echo "--- building workspace ---"
cd /ws || exit 1
if ! colcon build --symlink-install --event-handlers console_cohesion- > /tmp/colcon.log 2>&1; then
  echo "colcon build FAILED:"
  tail -40 /tmp/colcon.log
  exit 1
fi
set +u
source /ws/install/setup.bash
set -u
echo "build ok"
echo

# --- launch ------------------------------------------------------------------
echo "--- launching world (headless, paused) ---"
ros2 launch robot_rl_env world.launch.py headless:=true > /tmp/launch.log 2>&1 &
GZ_PID=$!

# Wait for the bridge to advertise rather than sleeping a fixed amount.
for _ in $(seq 1 30); do
  if ros2 topic list 2>/dev/null | grep -q "^/scan$"; then break; fi
  sleep 1
done
echo

# =============================================================================
# CHECK 1 -- topics exist at all
# =============================================================================
echo "--- check 1: bridged topics exist ---"
TOPICS=$(ros2 topic list 2>/dev/null)
for t in /scan /odom /cmd_vel /clock; do
  if echo "$TOPICS" | grep -q "^${t}$"; then
    pass "$t is advertised"
  else
    fail "$t is MISSING -- check config/bridge.yaml against the <topic> names in model.sdf"
  fi
done
echo

# =============================================================================
# CHECK 2 -- services exist
# Without these, step-synchronization is impossible.
# =============================================================================
echo "--- check 2: world control services exist ---"
SERVICES=$(ros2 service list 2>/dev/null)
for s in /world/arena/control /world/arena/set_pose; do
  if echo "$SERVICES" | grep -q "^${s}$"; then
    pass "$s is advertised"
  else
    fail "$s is MISSING -- is gz-sim-user-commands-system loaded in arena.sdf?"
  fi
done
echo

# =============================================================================
# CHECK 3 -- /clock is FROZEN while paused
# The single most important check in this file. If the clock advances on its
# own, the world is not paused and the entire step-synchronized design is
# already broken.
# =============================================================================
echo "--- check 3: /clock frozen while paused ---"
read_clock() {
  timeout 10 ros2 topic echo --once /clock 2>/dev/null \
    | grep -A2 "^clock:" | tr -d ' \n'
}
C1=$(read_clock)
sleep 3
C2=$(read_clock)

if [ -z "$C1" ] || [ -z "$C2" ]; then
  fail "could not read /clock at all -- the clock bridge is not working"
elif [ "$C1" = "$C2" ]; then
  pass "/clock did not advance over 3 s of wall time (world is paused)"
else
  fail "/clock ADVANCED while paused ($C1 -> $C2) -- the world is free-running,
        which breaks step-synchronization. Check that the launch file omits -r."
fi
echo

# =============================================================================
# CHECK 4 -- stepping the world advances /clock by exactly 50 ms
# This is the empirical proof that the core architectural decision works,
# established before any Gym code exists.
# =============================================================================
echo "--- check 4: 50 sim steps advance /clock by exactly 50 ms ---"
python3 - <<'PY'
import subprocess
import sys

import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from ros_gz_interfaces.srv import ControlWorld

EXPECTED_NS = 50_000_000  # 50 steps x 1 ms


class ClockProbe(Node):
    def __init__(self):
        super().__init__("clock_probe")
        self.stamp_ns = None
        self.create_subscription(Clock, "/clock", self._cb, 10)
        self.cli = self.create_client(ControlWorld, "/world/arena/control")

    def _cb(self, msg):
        self.stamp_ns = msg.clock.sec * 1_000_000_000 + msg.clock.nanosec

    def wait_for_clock(self, timeout=10.0):
        self.stamp_ns = None
        end = self.get_clock().now().nanoseconds + int(timeout * 1e9)
        while self.stamp_ns is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.get_clock().now().nanoseconds > end:
                raise TimeoutError("no /clock message")
        return self.stamp_ns


rclpy.init()
node = ClockProbe()
try:
    if not node.cli.wait_for_service(timeout_sec=15.0):
        print("  FAIL  /world/arena/control never became available")
        sys.exit(1)

    before = node.wait_for_clock()

    req = ControlWorld.Request()
    req.world_control.pause = True        # remain paused afterwards
    req.world_control.multi_step = 50     # advance exactly 50 iterations
    future = node.cli.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=15.0)
    if future.result() is None:
        print("  FAIL  ControlWorld service call timed out")
        sys.exit(1)

    after = node.wait_for_clock()
    delta = after - before

    # One physics tick of slack: the clock publisher may land on either side of
    # the boundary. Anything beyond that is a real desynchronization.
    if abs(delta - EXPECTED_NS) <= 1_000_000:
        print(f"  PASS  /clock advanced {delta/1e6:.1f} ms (expected 50.0 ms)")
        sys.exit(0)
    print(f"  FAIL  /clock advanced {delta/1e6:.1f} ms, expected 50.0 ms.")
    print("        Check <max_step_size> in arena.sdf against")
    print("        contract.PHYSICS_STEP_SIZE.")
    sys.exit(1)
finally:
    node.destroy_node()
    rclpy.shutdown()
PY
if [ $? -eq 0 ]; then PASS=$((PASS + 1)); else FAIL=$((FAIL + 1)); fi
echo

# =============================================================================
# CHECK 5 -- /scan rate matches the SDF update_rate while free-running
# =============================================================================
echo "--- check 5: /scan publishes at 20 Hz when unpaused ---"
ros2 service call /world/arena/control ros_gz_interfaces/srv/ControlWorld \
  "{world_control: {pause: false}}" > /dev/null 2>&1
sleep 1
HZ_OUT=$(timeout 12 ros2 topic hz /scan --window 20 2>&1 | grep -m1 "average rate" || true)
if [ -z "$HZ_OUT" ]; then
  fail "/scan published nothing -- the lidar sensor or the scan bridge is broken"
else
  RATE=$(echo "$HZ_OUT" | sed -n 's/.*average rate: \([0-9.]*\).*/\1/p')
  echo "        $HZ_OUT"
  # Generous band: this runs unpaused under an unbounded RTF, so the rate is
  # only loosely tied to wall-clock. We are checking it is alive and in the
  # right order of magnitude, not measuring it precisely.
  if awk "BEGIN{exit !($RATE > 5)}"; then
    pass "/scan is publishing (${RATE} Hz wall-clock)"
  else
    fail "/scan rate ${RATE} Hz is implausibly low"
  fi
fi
echo

# =============================================================================
# CHECK 6 -- /cmd_vel actually moves the robot and /odom reflects it
# Catches a bridge that is wired but pointed at the wrong gz topic: the command
# is published, nothing errors, and the robot never moves.
# =============================================================================
echo "--- check 6: /cmd_vel produces sane odometry ---"
python3 - <<'PY'
import sys

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


class DriveProbe(Node):
    def __init__(self):
        super().__init__("drive_probe")
        self.odom = None
        self.create_subscription(Odometry, "/odom", self._cb, 10)
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)

    def _cb(self, msg):
        self.odom = msg

    def wait_odom(self, timeout=10.0):
        self.odom = None
        end = self.get_clock().now().nanoseconds + int(timeout * 1e9)
        while self.odom is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.get_clock().now().nanoseconds > end:
                raise TimeoutError("no /odom message")
        return self.odom


rclpy.init()
node = DriveProbe()
try:
    start = node.wait_odom()
    x0, y0 = start.pose.pose.position.x, start.pose.pose.position.y

    cmd = Twist()
    cmd.linear.x = 0.2
    end = node.get_clock().now().nanoseconds + int(3.0 * 1e9)
    while node.get_clock().now().nanoseconds < end:
        node.pub.publish(cmd)
        rclpy.spin_once(node, timeout_sec=0.05)

    node.pub.publish(Twist())  # stop
    rclpy.spin_once(node, timeout_sec=0.5)

    final = node.wait_odom()
    x1, y1 = final.pose.pose.position.x, final.pose.pose.position.y
    moved = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5

    if moved > 0.05:
        print(f"  PASS  robot moved {moved:.3f} m under a 0.2 m/s command")
        sys.exit(0)
    print(f"  FAIL  robot moved only {moved:.3f} m under a 0.2 m/s command.")
    print("        /cmd_vel is bridged but not reaching the DiffDrive plugin --")
    print("        compare bridge.yaml gz_topic_name with <topic> in model.sdf.")
    sys.exit(1)
finally:
    node.destroy_node()
    rclpy.shutdown()
PY
if [ $? -eq 0 ]; then PASS=$((PASS + 1)); else FAIL=$((FAIL + 1)); fi
echo

# =============================================================================
# CHECK 7 -- headless really is headless
# =============================================================================
echo "--- check 7: no GUI process in headless mode ---"
if pgrep -af "gz sim" | grep -q -- "-g"; then
  fail "a 'gz sim -g' GUI process is running despite headless mode"
else
  pass "no GUI process running"
fi
echo

# --- summary -----------------------------------------------------------------
echo "============================"
echo " PASS: $PASS    FAIL: $FAIL"
echo "============================"
[ "$FAIL" -eq 0 ] || exit 1
