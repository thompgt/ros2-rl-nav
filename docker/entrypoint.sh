#!/usr/bin/env bash
# Sources the ROS 2 underlay and, if it exists, the workspace overlay, then
# execs whatever was asked for. `set -e` is deliberately NOT set before the
# sourcing: ROS setup scripts reference unbound variables and will abort.
set -o pipefail

source /opt/ros/jazzy/setup.bash

if [ -f /ws/install/setup.bash ]; then
  source /ws/install/setup.bash
fi

exec "$@"
