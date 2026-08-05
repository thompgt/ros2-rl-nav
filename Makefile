COMPOSE := docker compose -f docker/docker-compose.yml

.PHONY: help build shell test verify verify2 lint verify4 train evaluate export-policy board deploy monitor gap report gif world clean

help:
	@echo "build    - build the ROS 2 Jazzy + Gazebo Harmonic image"
	@echo "shell    - interactive shell in the container (repo mounted at /ws)"
	@echo "test     - colcon build + pytest, headless"
	@echo "verify   - Phase 1 bridge verification, PASS/FAIL per check"
	@echo "verify2  - Phase 2 exit criteria: 100 random episodes, throughput, memory"
	@echo "lint     - ruff check"
	@echo "verify4  - Phase 4 smoke: train tiny, export, free-running eval"
	@echo "world    - launch the arena headless (paused), for manual poking"
	@echo "train    - Phase 3 training: make train ALGO=sac SEED=0 ENVS=4"
	@echo "evaluate - Phase 3 evaluation: make evaluate MODEL=<path to .zip>"
	@echo "export-policy - TorchScript export: make export-policy MODEL=<path to .zip>"
	@echo "board    - TensorBoard over runs/ on port 6006"
	@echo "deploy   - Phase 4: run the policy against a free-running world"
	@echo "monitor  - Phase 4 deployment plus a live browser view on :8080"
	@echo "gap      - Phase 4 measurement: free-running vs step-synchronized"
	@echo "report   - aggregate runs/*/eval.json over seeds into the README tables"
	@echo "gif      - render one held-out episode as a top-down GIF"
	@echo "clean    - remove colcon build artifacts"

build:
	$(COMPOSE) build

shell:
	$(COMPOSE) run --rm dev

test:
	$(COMPOSE) run --rm test

verify:
	$(COMPOSE) run --rm verify

# make verify2 EPISODES=25 to shorten the run. The episode count cannot be
# appended to `compose run <service>` -- that replaces the service command
# rather than adding to it.
EPISODES ?= 100
verify2:
	$(COMPOSE) run --rm verify2 bash -lc "scripts/verify_phase2.sh $(EPISODES)"

# Phase 4 smoke: trains a throwaway policy, exports it, and runs the
# free-running evaluation. Proves the deployment path, not the policy.
# make verify4 TIMESTEPS=300 EPISODES=2
TIMESTEPS ?= 300
verify4:
	$(COMPOSE) run --rm train bash -lc "colcon build --symlink-install \
	  && scripts/verify_phase4.sh $(TIMESTEPS) $(EPISODES4)"

EPISODES4 ?= 2

lint:
	$(COMPOSE) run --rm dev bash -lc "ruff check src scripts"

world:
	$(COMPOSE) run --rm dev bash -lc "colcon build --symlink-install \
	  && source install/setup.bash \
	  && ros2 launch robot_rl_env world.launch.py headless:=true"

# make train ALGO=ppo SEED=1 ENVS=4 -- same reason as EPISODES above: arguments
# cannot be appended to `compose run <service>`.
#
# Training runs for hours. Launch it yourself and read the TensorBoard scalars
# or runs/<algo>-seed<N>/config.txt afterwards; do not run it inside an agent
# loop, which will sit on the output for the whole run and learn nothing.
ALGO ?= sac
SEED ?= 0
ENVS ?= 4
train:
	$(COMPOSE) run --rm train bash -lc "colcon build --symlink-install \
	  && source install/setup.bash \
	  && python3 -m robot_rl_env.train --algo $(ALGO) --seed $(SEED) --n-envs $(ENVS)"

# make evaluate MODEL=runs/sac-seed0/best/best_model.zip
MODEL ?= runs/$(ALGO)-seed$(SEED)/best/best_model.zip
evaluate:
	$(COMPOSE) run --rm train bash -lc "colcon build --symlink-install \
	  && source install/setup.bash \
	  && python3 -m robot_rl_env.evaluate --model $(MODEL) \
	     --json runs/$(ALGO)-seed$(SEED)/eval.json"

# make export-policy MODEL=runs/sac-seed0/best/best_model.zip
#
# Named `export-policy` rather than `export` because `export` is a GNU make
# directive and a target sharing its name is a trap for whoever adds the next
# one. Writes <run dir>/policy.pt, and refuses to write anything if the traced
# graph disagrees with the trained policy.
export-policy:
	$(COMPOSE) run --rm train bash -lc "colcon build --symlink-install \
	  && source install/setup.bash \
	  && python3 -m robot_rl_env.export_policy --model $(MODEL)"

# TensorBoard on all runs at once, so seeds and algorithms are compared on one
# axis rather than by flipping between screenshots.
board:
	$(COMPOSE) run --rm --service-ports train bash -lc \
	  "tensorboard --logdir runs --host 0.0.0.0 --port 6006"

# make deploy POLICY=runs/sac-seed0/policy.pt
#
# Launches the arena unpaused and the inference node against it, then waits for
# a goal. From another shell (`make shell`):
#
#   ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
#     '{header: {frame_id: odom}, pose: {position: {x: 2.0, y: 1.0}}}'
POLICY ?= runs/$(ALGO)-seed$(SEED)/policy.pt
deploy:
	POLICY=$(POLICY) $(COMPOSE) run --rm deploy

# make monitor POLICY=runs/sac-seed0/policy.pt
#
# The same unpaused deployment as `make deploy`, plus a browser view of it on
# http://localhost:8080 -- the arena, the 20 pooled LiDAR sectors the policy
# receives, the commanded velocities, and the observation age against the
# fixed 50 ms training saw. Click the arena to send a goal.
#
# Not the target to measure with: `make gap` uses deploy.launch.py, without a
# second node and a browser sharing a container that only sustains RTF 0.3-0.5.
MONITOR_PORT ?= 8080
monitor:
	POLICY=$(POLICY) MONITOR_PORT=$(MONITOR_PORT) $(COMPOSE) run --rm \
	  --service-ports monitor

# The Phase 4 measurement. Scores the exported policy on the same held-out
# episodes as `make evaluate`, but free-running, and prints the two side by
# side. Run `make evaluate` first so there is a baseline to subtract.
gap:
	$(COMPOSE) run --rm train bash -lc "colcon build --symlink-install \
	  && source install/setup.bash \
	  && python3 -m robot_rl_env.deploy_eval --policy $(POLICY) \
	     --baseline runs/$(ALGO)-seed$(SEED)/eval.json \
	     --json runs/$(ALGO)-seed$(SEED)/gap.json"

# make gif MODEL=runs/sac-seed0/best/best_model.zip EPISODE=0
#
# Renders one *scored* held-out episode as a top-down GIF: the arena, the path,
# and the 20 pooled LiDAR sectors the policy actually sees. Needs a simulator
# (it replays the episode), not a GPU.
EPISODE ?= 0
GIF ?= docs/nav.gif
gif:
	$(COMPOSE) run --rm train bash -lc "colcon build --symlink-install \
	  && source install/setup.bash \
	  && python3 -m robot_rl_env.record --model $(MODEL) \
	     --episode $(EPISODE) --out $(GIF)"

# Aggregate every runs/<algo>-seed<N>/{eval,gap}.json into the README's two
# tables, mean +/- std over seeds, and splice them in between the markers.
# Pure arithmetic -- no simulator, no torch -- so it runs on the host too:
#
#   python3 -m robot_rl_env.report            # print without writing
report:
	$(COMPOSE) run --rm test bash -lc "python3 -m robot_rl_env.report --write README.md"

clean:
	rm -rf build install log
