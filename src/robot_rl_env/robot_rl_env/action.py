"""The action mapping. PURE, and SHARED between training and deployment.

The mirror image of ``observation.py``, and it exists for the same reason. That
module is the single implementation of ``sensors -> 26-vector`` because a
training/inference divergence there is silent; this one is the single
implementation of ``action -> (v, omega)`` because a divergence *here* is
silent in exactly the same way. A deployment node that scaled ``a[0]`` to
``[-0.4, 0.4]`` instead of ``[0, 0.4]`` would drive a policy that never learned
to reverse, at speeds it never saw, and report no error at any point.

It lived on ``RobotNavEnv`` until Phase 4, which is where the asymmetry showed:
``policy_node.py`` cannot import ``env.py`` without pulling in a Gymnasium
environment, a simulator control client, and the assumption of a world it can
pause. It is arithmetic over two floats and it belongs where the rest of the
contract arithmetic lives -- next to ``observation.py``, importable on a robot
with no Gazebo, and tested on the host in milliseconds rather than only behind
``pytest.importorskip("rclpy")``.
"""

from __future__ import annotations

import numpy as np

from robot_rl_env import contract

STOP = np.array([-1.0, 0.0], dtype=np.float32)
"""The action that means "stop", expressed in the policy's own space.

``a[0] = -1`` is a full stop rather than reverse -- see ``scale_action`` -- so
the safety layer, the watchdog and the idle state all have a representation
inside ``[-1, 1]^2`` and never need a second code path that bypasses the
scaling. Anything that can zero the wheels can do it by proposing this action.
"""


def clip_action(action) -> np.ndarray:
    """Coerce to a float32 ``(2,)`` in ``[-1, 1]``, or raise on the wrong shape.

    The clip is cheap insurance against an exploding policy; the shape check is
    not insurance but a real guard, because a ``(1, 2)`` batch dimension left on
    by a caller broadcasts silently through the scaling arithmetic below and
    yields a Twist built from a numpy array rather than a float.
    """
    a = np.asarray(action, dtype=np.float32).reshape(contract.ACT_DIM)
    return np.clip(a, -1.0, 1.0)


def scale_action(action) -> tuple[float, float]:
    """``[-1, 1]^2`` -> ``(v, omega)`` in SI units. See CONTRACTS.md ("Action space").

    ``a[0] = -1`` is a full stop, not reverse: a policy that can back out of a
    concave obstacle never has to learn to avoid entering it, and the behaviour
    does not transfer to a robot whose LiDAR only covers the front.
    """
    a = clip_action(action)
    v = contract.MAX_LINEAR_VEL * (float(a[0]) + 1.0) / 2.0
    omega = contract.MAX_ANGULAR_VEL * float(a[1])
    return v, omega
