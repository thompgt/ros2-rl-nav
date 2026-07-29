"""Phase 2c -- ``RobotNavEnv(gymnasium.Env)``.

NOT IMPLEMENTED. This module is scaffolding; the next session builds it.

Wires ``sim_control`` (2a) and ``observation`` (2b) to the spec in
``CONTRACTS.md``. Spaces, reward, termination, truncation, reset randomization,
and seeding all come from that document; do not invent values here.

The step loop, in full
----------------------
::

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self._publish_cmd_vel(action)                       # 1. act
        target = self._sim_time + contract.STEP_DURATION
        self._sim.step(contract.SIM_STEPS_PER_ACTION)       # 2. advance exactly
        obs, info = self._obs.get_obs(min_stamp=target)     # 3. block, or raise
        reward, terminated = self._reward_and_termination(info)
        truncated = self._step_count >= contract.MAX_EPISODE_STEPS
        return obs, reward, terminated, truncated, info

No ``time.sleep``. No polling with a wall-clock deadline that returns something
anyway. Step 3 either produces data at or after the target sim time, or it
raises.

Tests to write FIRST
--------------------
Write each of these as a failing test before implementing the behaviour it
covers. In this module that ordering measurably changes the result -- it is
what forces the async handshake to be correct rather than approximately
correct.

- ``check_env(RobotNavEnv())`` from ``gymnasium.utils.env_checker`` passes.
- Same seed + same action sequence -> bit-identical observation sequence.
  This is the canary for async leakage; if it flakes, the step loop is wrong.
- Every ``step()`` advances ``/clock`` by exactly ``contract.STEP_DURATION``.
- Collision termination fires when the robot is teleported into a wall.
- Reward is positive when the robot is teleported closer to the goal.
- ``reset(options={"goal_radius": 2.0})`` never samples a goal beyond 2 m.

Exit criterion for Phase 2
--------------------------
A random-action agent runs 100 episodes with no hang, crash, or memory growth.
Log ``sim_time / wall_time``; you need >= 5x for Phase 3 to be practical. Below
1x, profile before going further -- do not try to fix it with more parallel
envs.
"""

raise NotImplementedError("Phase 2c. See the module docstring for the contract.")
