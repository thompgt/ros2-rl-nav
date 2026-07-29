"""Phase 2b -- observation assembly. SHARED between training and deployment.

NOT IMPLEMENTED. This module is scaffolding; the next session builds it.

This is the most important module in the repo to get right, because it is the
one whose bugs are invisible. Training and inference must transform sensor data
into a 26-vector *identically*. If they diverge, nothing errors and no metric
moves -- the deployed policy is simply worse, forever, for no visible reason.

Contract for the implementer
----------------------------
Two clearly separated layers.

**1. A pure function.** No ROS, no threads, no state::

    assemble_observation(
        ranges: np.ndarray,        # raw LaserScan.ranges, shape (360,)
        robot_xy: tuple[float, float],
        robot_yaw: float,
        goal_xy: tuple[float, float],
        prev_action: np.ndarray,   # shape (2,), already in [-1, 1]
        step_count: int,
    ) -> np.ndarray                # shape (26,), float32, all values in [-1, 1]

Exact arithmetic is specified in ``CONTRACTS.md`` -- min-pooling (NOT mean),
nan/inf handling, and the sin/cos bearing convention. Read it before writing a
line. Constants come from ``robot_rl_env.contract``.

Being pure, this layer is unit-testable with no simulator running, which is
what makes it cheap to pin down. Test it hard: nan-filled scans, a goal exactly
behind the robot, the +/-pi bearing wrap.

**2. A ROS subscriber wrapper**, ``ObservationAssembler``::

    get_obs(min_stamp: Time) -> tuple[np.ndarray, dict]

Subscribes ``/scan`` and ``/odom`` on a ``ReentrantCallbackGroup``, spun by a
``MultiThreadedExecutor``. Blocks until *both* topics have a message stamped
``>= min_stamp``, then calls the pure function.

The one rule that matters
-------------------------
On timeout, ``get_obs`` **raises**. It does not return ``self._last_obs``. It
does not log a warning and return the previous value. It does not extrapolate.

A staleness fallback is the single most tempting change in this codebase and
the single most damaging: it converts a loud, immediate failure into a quiet
distribution shift that survives training and only manifests as an unexplained
success-rate drop after deployment. If a timeout fires, the simulation is
misconfigured and you want to know now.

Reuse in Phase 4
----------------
``policy_node.py`` imports ``assemble_observation`` from here. It does not
reimplement any part of it, and it does not copy the constants.
"""

raise NotImplementedError("Phase 2b. See the module docstring for the contract.")
