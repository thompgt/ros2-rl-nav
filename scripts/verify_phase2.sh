#!/usr/bin/env bash
#
# Phase 2 verification: launch the world, then run the exit-criteria checks in
# scripts/phase2_checks.py. Prints PASS/FAIL per check and exits non-zero on any
# failure.
#
# Usage:  scripts/verify_phase2.sh [n_episodes]      (from /ws in the container)

set -uo pipefail

source /opt/ros/jazzy/setup.bash

N_EPISODES="${1:-100}"
GZ_PID=""

cleanup() {
  echo
  echo "--- cleanup ---"
  [ -n "$GZ_PID" ] && kill "$GZ_PID" 2>/dev/null
  pkill -f "gz sim"         2>/dev/null
  pkill -f parameter_bridge 2>/dev/null
  sleep 1
  pkill -9 -f "gz sim"      2>/dev/null
  return 0
}
trap cleanup EXIT

cd /ws || exit 1
echo "--- building workspace ---"
if ! colcon build --symlink-install --event-handlers console_cohesion- > /tmp/colcon.log 2>&1; then
  echo "colcon build FAILED:"
  tail -40 /tmp/colcon.log
  exit 1
fi
source /ws/install/setup.bash
echo "build ok"
echo

echo "--- launching world (headless, paused) ---"
ros2 launch robot_rl_env world.launch.py headless:=true > /tmp/launch.log 2>&1 &
GZ_PID=$!

# Wait for the service bridge rather than sleeping a fixed amount: without
# /world/arena/control the env cannot advance sim time, and its constructor
# would simply block for its discovery timeout and raise.
for _ in $(seq 1 60); do
  if ros2 service list 2>/dev/null | grep -q "^/world/arena/control$"; then break; fi
  sleep 1
done
echo

python3 scripts/phase2_checks.py "$N_EPISODES"
