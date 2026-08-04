COMPOSE := docker compose -f docker/docker-compose.yml

.PHONY: help build shell test verify verify2 lint train deploy world clean

help:
	@echo "build    - build the ROS 2 Jazzy + Gazebo Harmonic image"
	@echo "shell    - interactive shell in the container (repo mounted at /ws)"
	@echo "test     - colcon build + pytest, headless"
	@echo "verify   - Phase 1 bridge verification, PASS/FAIL per check"
	@echo "verify2  - Phase 2 exit criteria: 100 random episodes, throughput, memory"
	@echo "lint     - ruff check"
	@echo "world    - launch the arena headless (paused), for manual poking"
	@echo "train    - Phase 3"
	@echo "deploy   - Phase 4"
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

lint:
	$(COMPOSE) run --rm dev bash -lc "ruff check src scripts"

world:
	$(COMPOSE) run --rm dev bash -lc "colcon build --symlink-install \
	  && source install/setup.bash \
	  && ros2 launch robot_rl_env world.launch.py headless:=true"

train:
	$(COMPOSE) run --rm train

deploy:
	$(COMPOSE) run --rm deploy

clean:
	rm -rf build install log
