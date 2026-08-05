// The renderer. It maps world metres to pixels and draws; nothing else.
//
// Deliberately absent: LiDAR pooling, beam angles, quaternion -> yaw, the
// odom -> world transform, the sensor's forward offset. All of it arrives
// pre-computed from robot_rl_env/monitor.py, because that module shares the
// code path the policy's own observation comes from. A second implementation
// here would drift from it silently and this page would stop being evidence
// of what the robot saw -- see monitor.py's docstring.

"use strict";

const canvas = document.getElementById("arena");
const ctx = canvas.getContext("2d");
const $ = (id) => document.getElementById(id);

let scene = null;   // the arena, from the stream's opening `arena` event
let frame = null;   // the newest telemetry frame
let trail = [];     // recent robot positions, world metres

// About 15 s at 20 Hz. Long enough to show the shape of an approach, short
// enough that a robot circling does not paint the arena solid.
const TRAIL_MAX = 300;

// --- world <-> canvas --------------------------------------------------------
// The canvas is square and the arena is square, so one scale serves both axes.
// y is flipped: the world's +y is north, the canvas's +y is down.

const PAD = 18;

function scale() {
  return (canvas.width - 2 * PAD) / (scene.size + 0.4);
}

function toPixels(x, y) {
  const s = scale();
  return [canvas.width / 2 + x * s, canvas.height / 2 - y * s];
}

function toWorld(px, py) {
  const s = scale();
  return [(px - canvas.width / 2) / s, (canvas.height / 2 - py) / s];
}

// --- drawing -----------------------------------------------------------------

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!scene) return;

  drawArena();
  if (frame && frame.goal) drawGoal(frame.goal);
  drawTrail();
  if (frame && frame.beams) drawBeams(frame.beams, frame.robot);
  if (frame && frame.robot) drawRobot(frame.robot);
}

function drawArena() {
  const s = scale();
  const [x0, y0] = toPixels(-scene.half, scene.half);

  ctx.lineWidth = 2;
  ctx.strokeStyle = "#3a4048";
  ctx.strokeRect(x0, y0, scene.size * s, scene.size * s);

  ctx.fillStyle = "#3a4048";
  for (const o of scene.obstacles) {
    if (o.kind === "circle") {
      const [cx, cy] = toPixels(o.cx, o.cy);
      ctx.beginPath();
      ctx.arc(cx, cy, o.radius * s, 0, 2 * Math.PI);
      ctx.fill();
    } else {
      // Rotate about the box's own centre, then place it -- the order the SDF
      // pose applies, and the reason this translates before it rotates.
      const [cx, cy] = toPixels(o.cx, o.cy);
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(-o.yaw);  // canvas angles run clockwise; world angles do not
      ctx.fillRect((-o.sx / 2) * s, (-o.sy / 2) * s, o.sx * s, o.sy * s);
      ctx.restore();
    }
  }
}

function drawGoal(goal) {
  const s = scale();
  const [gx, gy] = toPixels(goal[0], goal[1]);

  ctx.fillStyle = "rgba(63, 176, 122, 0.22)";
  ctx.beginPath();
  ctx.arc(gx, gy, scene.goal_tolerance * s, 0, 2 * Math.PI);
  ctx.fill();

  ctx.strokeStyle = "#3fb07a";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(gx - 7, gy);
  ctx.lineTo(gx + 7, gy);
  ctx.moveTo(gx, gy - 7);
  ctx.lineTo(gx, gy + 7);
  ctx.stroke();
}

function drawTrail() {
  if (trail.length < 2) return;
  ctx.strokeStyle = "rgba(76, 155, 232, 0.55)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  trail.forEach(([x, y], i) => {
    const [px, py] = toPixels(x, y);
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  });
  ctx.stroke();
}

function drawBeams(beams, robot) {
  if (!robot) return;
  const [rx, ry] = toPixels(robot.x, robot.y);

  ctx.lineWidth = 1;
  for (const [wx, wy] of beams) {
    const [bx, by] = toPixels(wx, wy);
    // Beams that stop short of the sensor's range have hit something. Colour
    // by whether that something is inside the distance the policy was trained
    // to treat as a collision.
    const reach = Math.hypot(wx - robot.x, wy - robot.y);
    ctx.strokeStyle =
      reach < scene.collision_threshold + 0.2
        ? "rgba(232, 97, 92, 0.95)"
        : "rgba(232, 97, 92, 0.32)";
    ctx.beginPath();
    ctx.moveTo(rx, ry);
    ctx.lineTo(bx, by);
    ctx.stroke();
  }
}

function drawRobot(robot) {
  const s = scale();
  const [rx, ry] = toPixels(robot.x, robot.y);

  ctx.fillStyle = "#4c9be8";
  ctx.beginPath();
  ctx.arc(rx, ry, 0.15 * s, 0, 2 * Math.PI);
  ctx.fill();

  ctx.strokeStyle = "#e6e8ec";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(rx, ry);
  ctx.lineTo(rx + Math.cos(robot.yaw) * 0.35 * s, ry - Math.sin(robot.yaw) * 0.35 * s);
  ctx.stroke();
}

// --- the panel ---------------------------------------------------------------

const fmt = (value, digits = 2) =>
  value === undefined || value === null ? "—" : value.toFixed(digits);

function meter(element, fraction) {
  element.style.width = `${Math.min(Math.abs(fraction), 1) * 100}%`;
}

function updatePanel() {
  if (!frame) return;

  $("linear").textContent = fmt(frame.linear);
  $("angular").textContent = fmt(frame.angular);
  $("distance").textContent = fmt(frame.distance_to_goal);
  $("clearance").textContent = fmt(frame.min_range);

  meter($("linear-bar"), (frame.linear || 0) / scene.max_linear);

  // Signed: the bar grows left or right of centre, because which way it is
  // turning is the thing you are watching for when a policy starts spinning.
  const turn = (frame.angular || 0) / scene.max_angular;
  const bar = $("angular-bar");
  bar.style.width = `${Math.min(Math.abs(turn), 1) * 50}%`;
  bar.style.marginLeft = turn < 0 ? `${50 - Math.min(Math.abs(turn), 1) * 50}%` : "50%";

  const age = frame.age === undefined ? null : frame.age * 1000;
  $("age").textContent = age === null ? "—" : age.toFixed(0);
  meter($("age-bar"), age === null ? 0 : age / (scene.watchdog_timeout * 1000));
  $("age-bar").style.background = frame.stale ? "#e8615c" : age > 100 ? "#e0a03c" : "#4c9be8";

  const status = frame.status || {};
  $("reason").textContent = status.reason || (frame.connected ? "—" : "no data");
  $("outcome").textContent = status.outcome || "—";
  $("step").textContent = status.step === undefined ? "—" : status.step;

  const link = $("link");
  if (!frame.connected) {
    link.textContent = "no sensor data";
    link.className = "pill down";
  } else if (frame.stale) {
    link.textContent = "watchdog";
    link.className = "pill stale";
  } else {
    link.textContent = "live";
    link.className = "pill live";
  }
}

// --- the stream --------------------------------------------------------------

function connect() {
  const stream = new EventSource("/stream");

  stream.addEventListener("arena", (event) => {
    scene = JSON.parse(event.data);
    $("max-steps").textContent = `/ ${scene.max_steps}`;
    $("watchdog").textContent = (scene.watchdog_timeout * 1000).toFixed(0);
    draw();
  });

  stream.onmessage = (event) => {
    frame = JSON.parse(event.data);
    if (frame.robot) {
      trail.push([frame.robot.x, frame.robot.y]);
      if (trail.length > TRAIL_MAX) trail.shift();
    }
    // A new goal starts a new run, so the previous approach stops being part
    // of the picture.
    if (frame.status && frame.status.episode !== undefined) {
      if (frame.status.episode !== lastEpisode) {
        trail = [];
        lastEpisode = frame.status.episode;
      }
    }
    draw();
    updatePanel();
  };

  // EventSource reconnects on its own; this only reports the gap, so a page
  // left open overnight against a stopped container says so.
  stream.onerror = () => {
    const link = $("link");
    link.textContent = "reconnecting…";
    link.className = "pill down";
  };
}

let lastEpisode = null;

// --- goals out ---------------------------------------------------------------

canvas.addEventListener("click", async (event) => {
  if (!scene) return;
  const box = canvas.getBoundingClientRect();
  // The canvas is laid out responsively, so its CSS size is not its pixel
  // size; without this ratio every click lands somewhere else.
  const px = ((event.clientX - box.left) / box.width) * canvas.width;
  const py = ((event.clientY - box.top) / box.height) * canvas.height;
  const [x, y] = toWorld(px, py);

  const error = $("error");
  try {
    const response = await fetch("/goal", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x, y }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || response.statusText);
    error.hidden = true;
    trail = [];
  } catch (failure) {
    // Shown rather than swallowed: a goal that vanished looks exactly like a
    // policy that ignored it.
    error.textContent = `goal refused: ${failure.message}`;
    error.hidden = false;
  }
});

connect();
