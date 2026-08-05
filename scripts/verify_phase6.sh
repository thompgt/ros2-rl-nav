#!/usr/bin/env bash
# Phase 6 smoke test: the live monitor against a real deployment.
#
# Launches monitor.launch.py -- unpaused world, policy node, monitor node --
# and then talks to the HTTP surface the way a browser does: fetches the page
# and the scene, reads frames off the event stream, posts a goal, and checks it
# arrived at the controller.
#
# It does not grade the policy -- the smoke policy is noise. It grades wiring.
#
# Everything below fails *silently* if it is wrong, which is why it is a script
# and not a paragraph in the README. A monitor whose stream carries no robot
# looks like a Gazebo that has not started; a goal that never reaches the
# policy node looks like a policy ignoring it; a page served from the wrong
# directory is a 404 nobody sees until they open a browser on a machine with no
# browser.
#
#   scripts/verify_phase6.sh [POLICY] [PORT]
set -euo pipefail

POLICY="${1:-runs/smoke-sac-seed0/policy.pt}"
PORT="${2:-8080}"
BASE="http://127.0.0.1:${PORT}"
LOG="$(mktemp)"

test -f "${POLICY}" || {
  echo "FAIL: no policy at ${POLICY}"
  echo "Run scripts/verify_phase4.sh first -- it trains and exports a throwaway one."
  exit 1
}

# ROS setup scripts read unset variables; set -u has to be lifted around them.
set +u
source /opt/ros/jazzy/setup.bash
source install/setup.bash
set -u

echo "=== Phase 6 smoke: the monitor, on ${BASE} ==="

ros2 launch robot_rl_env monitor.launch.py \
  policy:="${POLICY}" host:=127.0.0.1 port:="${PORT}" > "${LOG}" 2>&1 &
LAUNCH_PID=$!

cleanup() {
  kill "${LAUNCH_PID}" 2>/dev/null || true
  wait "${LAUNCH_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# Gazebo takes a while to come up on a container with no GPU, and the monitor
# serves before the simulator publishes -- deliberately, so the page can say
# "no sensor data" rather than refuse to load. So: wait for the port, then wait
# separately for the stream to carry a robot.
echo
echo "--- 1/5 the server comes up --------------------------------------------"
for _ in $(seq 60); do
  if curl -fsS "${BASE}/arena.json" > /dev/null 2>&1; then break; fi
  sleep 1
done
curl -fsS "${BASE}/arena.json" > /tmp/arena.json || {
  echo "FAIL: no /arena.json after 60 s"; tail -40 "${LOG}"; exit 1;
}
python3 -c "
import json, sys
scene = json.load(open('/tmp/arena.json'))
assert len(scene['obstacles']) == 7, scene['obstacles']
assert scene['size'] == 10.0, scene
print(f\"  arena {scene['size']:g} m, {len(scene['obstacles'])} obstacles\")
"

echo
echo "--- 2/5 the page is served ---------------------------------------------"
curl -fsS "${BASE}/" | grep -q "<canvas" || {
  echo "FAIL: / did not serve index.html -- wrong web root?"; exit 1;
}
curl -fsS "${BASE}/app.js" > /dev/null || { echo "FAIL: app.js not served"; exit 1; }
echo "  index.html and app.js served"

echo
echo "--- 3/5 the stream carries the observation -----------------------------"
# Read a bounded slice of the stream. curl exits non-zero when we cut it off,
# hence the `|| true`; the assertion is on what arrived, not on the exit code.
MONITOR_BASE="${BASE}" python3 - <<'PY'
import json, os, subprocess, sys, time

BASE = os.environ["MONITOR_BASE"]
deadline = time.time() + 90
proc = subprocess.Popen(
    ["curl", "-fsSN", f"{BASE}/stream"], stdout=subprocess.PIPE
)
scene = robot = None
try:
    for raw in proc.stdout:
        line = raw.decode().strip()
        if line.startswith("data:"):
            payload = json.loads(line[5:])
            if "obstacles" in payload:
                scene = payload
            elif payload.get("connected"):
                robot = payload
                break
        if time.time() > deadline:
            break
finally:
    proc.kill()

if scene is None:
    sys.exit("FAIL: the stream never sent the arena event")
if robot is None:
    sys.exit("FAIL: 90 s of stream and no frame with sensor data -- is Gazebo up?")

beams = robot["beams"]
assert len(beams) == 20, f"expected 20 pooled sectors, got {len(beams)}"
print(f"  robot at ({robot['robot']['x']:.2f}, {robot['robot']['y']:.2f}) world, "
      f"{len(beams)} sectors, obs age {robot['age'] * 1000:.0f} ms")
# The whole point of the view: this is not 50 ms, and training's was.
if robot["age"] < 0.05:
    print("  note: observation fresher than training's fixed 50 ms, which is "
          "possible on an idle machine but unusual here")
PY

echo
echo "--- 4/5 a goal outside the arena is refused ----------------------------"
CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${BASE}/goal" \
  -H 'Content-Type: application/json' -d '{"x": 40.0, "y": 0.0}')
test "${CODE}" = "400" || { echo "FAIL: expected 400 for a goal outside the arena, got ${CODE}"; exit 1; }
echo "  400, with a reason the page can show"

echo
echo "--- 5/5 a goal reaches the policy node ---------------------------------"
curl -fsS -X POST "${BASE}/goal" \
  -H 'Content-Type: application/json' -d '{"x": 0.0, "y": 0.0}' > /dev/null
sleep 8

grep -q "goal (" "${LOG}" || {
  echo "FAIL: the policy node never logged a goal"; tail -40 "${LOG}"; exit 1;
}
MONITOR_BASE="${BASE}" python3 - <<'PY'
import json, os, subprocess, sys, time

BASE = os.environ["MONITOR_BASE"]
# The status only exists because monitor.launch.py turned it on; its absence
# here means the launch argument stopped being threaded through.
proc = subprocess.Popen(
    ["curl", "-fsSN", f"{BASE}/stream"], stdout=subprocess.PIPE
)
deadline, status = time.time() + 30, None
try:
    for raw in proc.stdout:
        line = raw.decode().strip()
        if line.startswith("data:"):
            payload = json.loads(line[5:])
            if payload.get("status"):
                status = payload["status"]
                break
        if time.time() > deadline:
            break
finally:
    proc.kill()

if status is None:
    sys.exit("FAIL: no status in the stream -- status_topic not threaded through?")
print(f"  status: reason={status['reason']} outcome={status['outcome']} "
      f"step={status['step']} goal={status['goal']}")
PY

echo
echo "=== PASS: the monitor serves, streams, and steers ==="
echo "What passed is the wiring: the scene, the page, pooled sectors on the"
echo "stream, a refused goal refused, and a goal that reached the controller."
