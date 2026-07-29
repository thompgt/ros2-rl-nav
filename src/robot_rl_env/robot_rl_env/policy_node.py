"""Phase 4 -- standalone ROS 2 inference node.

NOT IMPLEMENTED. This module is scaffolding.

This is the part that separates the project from a Gym tutorial, and the part
interviewers ask about. It runs against an **unpaused, real-time** world.

Contract for the implementer
----------------------------
- Load the policy exported to TorchScript or ONNX. **No stable-baselines3
  import at runtime** -- the deployment artifact should not depend on the
  training framework.
- Subscribe ``/scan`` and ``/odom``; publish ``/cmd_vel`` on a timer at
  ``contract.CONTROL_HZ`` (20 Hz).
- Accept a goal via a ``/set_goal`` service or a ``geometry_msgs/PoseStamped``
  topic.
- Safety layer, independent of the policy:
  - hard stop if min pooled LiDAR < ``contract.SAFETY_STOP_THRESHOLD``
  - watchdog zeroing ``/cmd_vel`` if no observation arrives within
    ``contract.WATCHDOG_TIMEOUT``

Non-negotiable
--------------
Import ``assemble_observation`` from ``robot_rl_env.observation``. Do not
reimplement the preprocessing here, do not "adapt" it for real-time use, and do
not re-declare the constants. Divergence between training-time and
inference-time preprocessing is the classic silent failure of this project
type: it produces no error, no log line, and no metric -- just a policy that is
quietly worse than the one you trained.

The interesting measurement
---------------------------
Measure success rate in step-synchronized mode and in free-running real-time
mode, on the same held-out start/goal pairs. There *will* be a gap, driven by
the sensor latency and timing jitter that the training loop eliminated by
construction. Quantifying it and explaining its mechanism is the most
interesting paragraph in the README -- do not paper over it.
"""


def main(args=None):
    raise NotImplementedError("Phase 4. See the module docstring for the contract.")
