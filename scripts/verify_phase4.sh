#!/usr/bin/env bash
# Phase 4 smoke test: the whole deployment path, end to end, against a live
# Gazebo. Trains a throwaway policy for a few hundred steps, exports it, and
# runs the free-running evaluation on a couple of episodes.
#
# This proves nothing about the *policy* -- a few hundred steps of SAC is noise
# and the success rate here is meaningless. It proves the path: that the export
# agrees with the trained model, that the node loads a .pt and drives
# /cmd_vel, that goals survive the world->odom bridge, that episodes terminate,
# and that the gap report renders. Every one of those fails silently otherwise:
# a deployment node that never receives an observation simply publishes zeros
# forever, and looks exactly like a policy that learned to stand still.
#
#   scripts/verify_phase4.sh [TIMESTEPS] [EPISODES]
set -euo pipefail

TIMESTEPS="${1:-300}"
EPISODES="${2:-2}"
RUN_DIR="runs/smoke-sac-seed0"

# ROS setup scripts read unset variables; set -u has to be lifted around them.
set +u
source /opt/ros/jazzy/setup.bash
source install/setup.bash
set -u

echo "=== Phase 4 smoke: ${TIMESTEPS} training steps, ${EPISODES} episodes ==="

echo
echo "--- 1/3 train a throwaway policy ---------------------------------------"
python3 -m robot_rl_env.train \
  --algo sac --seed 0 --n-envs 1 \
  --timesteps "${TIMESTEPS}" \
  --run-dir "${RUN_DIR}"

echo
echo "--- 2/3 export it to TorchScript ---------------------------------------"
# The export verifies itself against model.predict and exits non-zero on a
# mismatch, so this step is its own assertion.
python3 -m robot_rl_env.export_policy --model "${RUN_DIR}/final.zip" --algo sac

test -f "${RUN_DIR}/policy.pt" || { echo "FAIL: no policy.pt written"; exit 1; }

echo
echo "--- 3/4 free-running evaluation ----------------------------------------"
python3 -m robot_rl_env.deploy_eval \
  --policy "${RUN_DIR}/policy.pt" \
  --episodes "${EPISODES}" \
  --json "${RUN_DIR}/gap.json"

echo
echo "--- 4/4 render one episode ---------------------------------------------"
# The drawing itself is unit-tested; what this covers is the wiring -- that the
# trace comes back populated from a real env, that the odom->world transform
# has the fields it expects, and that matplotlib is actually in the image.
python3 -m robot_rl_env.record \
  --model "${RUN_DIR}/final.zip" --algo sac \
  --episode 0 --out "${RUN_DIR}/nav.gif"

test -s "${RUN_DIR}/nav.gif" || { echo "FAIL: no GIF written"; exit 1; }

echo
echo "=== PASS: the deployment path runs end to end ==="
echo "The numbers above are meaningless -- ${TIMESTEPS} steps of SAC is noise."
echo "What passed is the path: export, load, goal, episode, termination, report."
