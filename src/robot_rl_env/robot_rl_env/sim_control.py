"""Phase 2a -- client for the Gazebo Harmonic world control services.

NOT IMPLEMENTED. This module is scaffolding; the next session builds it.

Contract for the implementer
----------------------------
A single class ``SimControl`` wrapping the bridged Gazebo services, testable
standalone with no Gymnasium involvement whatsoever::

    pause()                        -> None
    unpause()                      -> None
    step(n: int)                   -> None    # advance exactly n iterations
    reset_world()                  -> None
    set_entity_pose(name, pose)    -> None

Use ``ros_gz_interfaces/srv/ControlWorld`` on ``contract.WORLD_CONTROL_SERVICE``
and ``ros_gz_interfaces/srv/SetEntityPose`` on ``contract.SET_POSE_SERVICE``.

``step(n)`` sets ``WorldControl.multi_step = n`` with ``pause = True``, which
advances exactly n iterations and returns to the paused state. It does not
unpause-then-sleep-then-pause; that reintroduces wall-clock coupling.

Hard requirements
-----------------
- Service calls go on their own ``MutuallyExclusiveCallbackGroup``, distinct
  from the subscription group, so a service future can complete while a
  subscription callback is executing.
- Never ``rclpy.spin_once()``. The node is spun by a ``MultiThreadedExecutor``
  on a background thread owned by the caller.
- A service call that times out raises. It does not warn and continue -- a
  silently dropped step desynchronizes sim time from the env's step counter and
  every subsequent observation is wrong in a way nothing will report.

First test to write (before the implementation)
-----------------------------------------------
``test_step_advances_clock_exactly``: read ``/clock``, call ``step(50)``, read
``/clock`` again, assert the delta is exactly ``contract.STEP_DURATION`` within
one physics tick. If this test cannot be made to pass, nothing downstream is
worth writing.
"""

raise NotImplementedError("Phase 2a. See the module docstring for the contract.")
