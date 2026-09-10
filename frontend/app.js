import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9/vision_bundle.mjs";

// ---------------------------------------------------------------------------
// Config (mirrors config/config.yaml thresholds of the desktop app)
// ---------------------------------------------------------------------------
const CFG = {
  EAR_THRESHOLD: 0.21,
  EAR_CONSECUTIVE_FRAMES: 45,   // ~1.5s of closure -> "EYES CLOSED"
  BLINK_MIN_FRAMES: 2,
  BLINK_MAX_FRAMES: 6,
  MAR_THRESHOLD: 0.50,
  YAWN_CONSECUTIVE_FRAMES: 30,
  PITCH_THRESHOLD: 15,
  YAW_THRESHOLD: 20,
  AWAY_CONSECUTIVE_FRAMES: 20,
  FACE_LOST_FRAMES: 45,         // face missing this long -> EYES OFF ROAD
  WEIGHTS: { eye_closure: 35, yawn: 20, head_away: 25, hand_down: 10, phone: 30 },
  THRESHOLDS: { safe_min: 80, caution_min: 60, attention_min: 40, drowsy_min: 20 },
};

const EYE_LEFT = [33, 160, 158, 133, 153, 144];
const EYE_RIGHT = [362, 385, 387, 263, 373, 380];
const STATES = ["SAFE", "CAUTION", "ATTENTION REQUIRED", "DROWSY / DISTRACTED", "DANGER"];

const API_BASE = new URLSearchParams(location.search).get("api") ||
  "https://smart-driver-monitor-api.onrender.com";

// ---------------------------------------------------------------------------
// Elements
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const video = $("webcam"), overlay = $("overlay"), ctx = overlay.getContext("2d");
const noFaceEl = $("noFace");
const apiStatusEl = $("apiStatus"), apiTextEl = $("apiText");
const stateEl = $("state"), reasonEl = $("reason"), gaugeFill = $("gaugeFill"), scoreEl = $("score");
const fpsBadge = $("fpsBadge");

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let faceLandmarker = null;
let running = false;
let demoMode = false;
let rafId = 0;
let lastVideoTime = -1;
let sessionId = "web-" + Math.random().toString(36).slice(2, 9);
let sessionStart = null;

// per-frame analysis
let prevTime = performance.now();
let fpsEma = 0;

let ear = 0, mar = 0, pitch = 0, yaw = 0, roll = 0, headDir = "FORWARD";
let closedFrames = 0, yawnFrames = 0, awayFrames = 0, faceLostFrames = 0, awayActive = false, yawnActive = false;
let sim = null; // demo-mode signal generator

const stats = { blinks: 0, yawns: 0, closures: 0, away: 0, phone: 0 };

const risk = {
  risk_score: 0,
  safety_score: 100,
  state: "SAFE",
  state_counter: 0,
  total_risk_events: 0,
  update(inputs) {
    let points = 0;
    if (inputs.eye_closure) points += CFG.WEIGHTS.eye_closure * Math.min(inputs.eye_closure_frames / 60, 1);
    if (inputs.yawn) points += CFG.WEIGHTS.yawn * Math.min(inputs.yawn_frames / 45, 1);
    if (inputs.looking_away) points += CFG.WEIGHTS.head_away * Math.min(inputs.away_frames / 40, 1);
    if (inputs.hand_down) points += CFG.WEIGHTS.hand_down * 0.8;
    if (inputs.phone) points += CFG.WEIGHTS.phone * 0.9;
    const maxPossible = Object.values(CFG.WEIGHTS).reduce((a, b) => a + b, 0);
    const norm = Math.min((points / maxPossible) * 100, 100);
    this.risk_score = norm;
    this.safety_score = Math.max(0, 100 - norm);
    const newState = computeState(this.safety_score);
    if (newState !== this.state) {
      this.state_counter++;
      if (this.state_counter >= 5) {
        const old = this.state;
        this.state = newState;
        this.state_counter = 0;
        if (STATES.indexOf(newState) > STATES.indexOf(old)) this.total_risk_events++;
      }
    } else this.state_counter = 0;
  },
  reset() { this.risk_score = 0; this.safety_score = 100; this.state = "SAFE"; this.state_counter = 0; this.total_risk_events = 0; },
};
const computeState = (s) => s >= 80 ? "SAFE" : s >= 60 ? "CAUTION" : s >= 40 ? "ATTENTION REQUIRED" : s >= 20 ? "DROWSY / DISTRACTED" : "DANGER";
const stateIndex = (s) => STATES.indexOf(s);

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const earOf = (lm, idx) => {
  const a = dist(lm[idx[1]], lm[idx[5]]);
  const b = dist(lm[idx[2]], lm[idx[4]]);
  const c = dist(lm[idx[0]], lm[idx[3]]);
  return c > 0 ? (a + b) / (2 * c) : 0;
};
const smooth = (ema, next, k = 0.3) => ema === 0 ? next : ema + k * (next - ema);

function headPose(lm) {
  const eyeL = lm[33], eyeR = lm[263];
  const earPtL = lm[234], earPtR = lm[454];
  const nose = lm[1];
  const faceW = Math.max(dist(earPtL, earPtR), 1e-4);
  const r = (Math.atan2(eyeR.y - eyeL.y, eyeR.x - eyeL.x) * 180) / Math.PI;
  const midX = (earPtL.x + earPtR.x) / 2;
  const y = (Math.atan2(nose.y - (earPtL.y + earPtR.y) / 2, nose.x - midX) * 180) / Math.PI;
  const p = (Math.atan2(nose.y - (lm[10].y + lm[152].y) / 2, Math.abs(nose.z + 0.1)) * 180) / Math.PI * 0.5;
  return { pitch: p, yaw: y, roll: r };
}

// ---------------------------------------------------------------------------
// MediaPipe setup
// ---------------------------------------------------------------------------
async function loadModel() {
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9/wasm"
    );
    try {
      faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numFaces: 1,
        outputFaceBlendshapes: false,
      });
    } catch {
      faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task", delegate: "CPU" },
        runningMode: "VIDEO",
        numFaces: 1,
      });
    }
    return true;
  } catch (err) {
    console.error("Model load failed:", err);
    return false;
  }
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" } });
  video.srcObject = stream;
  await video.play();
  overlay.width = video.videoWidth || 640;
  overlay.height = video.videoHeight || 480;
}

function drawFace(lm) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  // pupil lines
  for (const idx of [...EYE_LEFT.slice(0, 4), ...EYE_RIGHT.slice(0, 4), 1, 13, 14, 78, 308, 10, 152]) {
    const p = lm[idx];
    ctx.beginPath();
    ctx.arc(p.x * overlay.width, p.y * overlay.height, 2.4, 0, Math.PI * 2);
    ctx.fillStyle = idx === 1 ? "#ffd166" : "#4d7cff";
    ctx.fill();
  }
  // eye ellipses
  for (const idx of [EYE_LEFT, EYE_RIGHT]) {
    ctx.beginPath();
    ctx.moveTo(lm[idx[0]].x * overlay.width, lm[idx[0]].y * overlay.height);
    for (let i = 1; i < 6; i++) ctx.lineTo(lm[idx[i]].x * overlay.width, lm[idx[i]].y * overlay.height);
    ctx.closePath();
    ctx.strokeStyle = ear < CFG.EAR_THRESHOLD ? "#ff4d5e" : "#2dd06e";
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

// ---------------------------------------------------------------------------
// Analysis pipeline (real webcam frames)
// ---------------------------------------------------------------------------
function analyzeFrame(timestamp) {
  if (!faceLandmarker) return;
  const now = performance.now();
  const dt = now - prevTime;
  prevTime = now;
  if (dt > 0) fpsEma = 0.9 * fpsEma + 0.1 * (1000 / dt);
  fpsBadge.textContent = Math.round(fpsEma) + " FPS";

  if (video.currentTime !== lastVideoTime && video.readyState >= 2) {
    lastVideoTime = video.currentTime;
    faceLandmarker.detectForVideo(video, timestamp).then((res) => {
      let lm = null;
      if (res.faceLandmarks && res.faceLandmarks.length > 0) lm = res.faceLandmarks[0];
      processLandmarks(lm);
    });
  } else {
    processLandmarks(null);
  }
  rafId = requestAnimationFrame(analyzeFrame);
}

function processLandmarks(lm) {
  let eyeClosure = false, yawn = false, lookingAway = false;

  if (lm) {
    faceLostFrames = 0;
    drawFace(lm);
    const rawEar = (earOf(lm, EYE_LEFT) + earOf(lm, EYE_RIGHT)) / 2;
    ear = smooth(ear, rawEar, 0.45);
    const mouthW = dist(lm[78], lm[308]);
    mar = mouthW > 0 ? dist(lm[13], lm[14]) / mouthW : 0;
    const hp = headPose(lm);
    pitch = smooth(pitch, hp.pitch, 0.25);
    yaw = smooth(yaw, hp.yaw, 0.25);
    roll = smooth(roll, hp.roll, 0.25);
  } else {
    faceLostFrames++;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    noFaceEl.classList.toggle("hidden", faceLostFrames < 3);
  }

  if (!lm) {
    ear = smooth(ear, 0, 0.2); mar = smooth(mar, 0, 0.2);
  }

  // --- blink / eye closure ---
  if (ear < CFG.EAR_THRESHOLD) {
    closedFrames++;
    if (closedFrames >= CFG.EAR_CONSECUTIVE_FRAMES) eyeClosure = true;
  } else {
    if (closedFrames >= CFG.BLINK_MIN_FRAMES && closedFrames <= CFG.BLINK_MAX_FRAMES) stats.blinks++;
    if (closedFrames >= CFG.EAR_CONSECUTIVE_FRAMES) stats.closures++;
    closedFrames = 0;
  }

  // --- yawn ---
  if (mar > CFG.MAR_THRESHOLD) {
    if (!yawnActive) yawnFrames++;
    if (yawnFrames >= CFG.YAWN_CONSECUTIVE_FRAMES && !yawnActive) { yawn = true; yawnActive = true; stats.yawns++; }
  } else {
    if (yawnActive && yawnFrames >= CFG.YAWN_CONSECUTIVE_FRAMES) yawnFrames = 0;
    yawnFrames = 0;
    yawnActive = false;
  }

  // --- head direction + away ---
  headDir = Math.abs(pitch) > CFG.PITCH_THRESHOLD
    ? (pitch > 0 ? "DOWN" : "UP")
    : Math.abs(yaw) > CFG.YAW_THRESHOLD
      ? (yaw > 0 ? "LEFT" : "RIGHT")
      : "FORWARD";

  if (headDir !== "FORWARD") awayFrames++; else awayFrames = 0;
  if (awayFrames >= CFG.AWAY_CONSECUTIVE_FRAMES) {
    if (!awayActive) { awayActive = true; stats.away++; }
    lookingAway = true;
  } else if (awayFrames === 0) awayActive = false;

  const eyesOffRoad = faceLostFrames >= CFG.FACE_LOST_FRAMES;

  risk.update({
    eye_closure: eyeClosure, eye_closure_frames: closedFrames,
    yawn, yawn_frames: yawnFrames,
    looking_away: lookingAway || eyesOffRoad, away_frames: Math.max(awayFrames, faceLostFrames),
    hand_down: sim ? sim.handDown : false,
    phone: sim ? sim.phone : false,
  });
  updateUI({ eyeClosure, yawn, lookingAway, eyesOffRoad });
  reportToBackend({ eyeClosure, yawn, lookingAway, eyesOffRoad });
}

function updateUI(flags) {
  const s = risk.safety_score;
  scoreEl.textContent = Math.round(s);
  gaugeFill.style.width = s + "%";
  gaugeFill.style.background = s >= 80 ? "var(--green)" : s >= 60 ? "var(--yellow)" : s >= 40 ? "var(--orange)" : "var(--red)";
  stateEl.textContent = risk.state;
  stateEl.className = "state-pill " + (s >= 80 ? "" : s >= 60 ? "warn" : s >= 40 ? "alert" : s >= 20 ? "danger" : "critical");

  const reasons = [];
  if (flags.eyeClosure) reasons.push("EYES CLOSED");
  if (flags.yawn) reasons.push("YAWNING");
  if (flags.lookingAway) reasons.push("LOOKING AWAY");
  if (flags.eyesOffRoad) reasons.push("EYES OFF ROAD");
  if (sim && sim.phone) reasons.push("PHONE");
  if (sim && sim.handDown) reasons.push("HANDS OFF WHEEL");
  reasonEl.textContent = reasons.length ? "ALERT: " + reasons.join(", ") : "All clear \u2014 keep driving.";
  reasonEl.classList.toggle("bad", reasons.length > 0);

  $("mEar").textContent = ear.toFixed(2);
  $("barEar").style.width = Math.min(ear * 100, 100) + "%";
  $("mMar").textContent = mar.toFixed(2);
  $("barMar").style.width = Math.min(mar * 100, 100) + "%";
  $("mPitch").textContent = Math.round(pitch) + "\u00b0";
  $("mYaw").textContent = Math.round(yaw) + "\u00b0";
  $("mRoll").textContent = Math.round(roll) + "\u00b0";
  $("mDir").textContent = headDir;

  $("sBlinks").textContent = stats.blinks;
  $("sYawns").textContent = stats.yawns;
  $("sClosures").textContent = stats.closures;
  $("sAway").textContent = stats.away;
  $("sPhone").textContent = stats.phone;
  if (sessionStart) {
    const t = Math.floor((Date.now() - sessionStart) / 1000);
    $("sTime").textContent = Math.floor(t / 60) + ":" + String(t % 60).padStart(2, "0");
  }
}

// ---------------------------------------------------------------------------
// Backend (Render) reporting
// ---------------------------------------------------------------------------
let apiOnline = null;
let lastReport = 0;
async function reportToBackend(flags) {
  const nowT = performance.now();
  if (nowT - lastReport < 500) return;
  lastReport = nowT;
  try {
    const res = await fetch(API_BASE + "/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId, ear, mar, pitch, yaw, roll,
        eye_closed: flags.eyeClosure, eye_closure_frames: closedFrames,
        yawn: flags.yawn, yawn_frames: yawnFrames,
        looking_away: flags.lookingAway || flags.eyesOffRoad, away_frames: Math.max(awayFrames, faceLostFrames),
        hand_down: sim ? sim.handDown : false,
        phone: sim ? sim.phone : false,
        face_present: faceLostFrames < CFG.FACE_LOST_FRAMES,
      }),
    });
    if (!res.ok) throw new Error("bad status " + res.status);
    setApiStatus(true);
  } catch (e) {
    setApiStatus(false);
  }
}
function setApiStatus(online) {
  if (apiOnline === online) return;
  apiOnline = online;
  apiStatusEl.classList.toggle("online", !!online);
  apiStatusEl.classList.toggle("offline", !online);
  apiTextEl.textContent = online ? "Risk API online" : "Local mode (API offline)";
}

// ---------------------------------------------------------------------------
// Demo mode - synthetic signals (proves the system with no webcam)
// ---------------------------------------------------------------------------
function startDemo() {
  demoMode = true;
  running = true;
  sessionStart = Date.now();
  $("btnStart").disabled = true;
  $("btnDemo").disabled = true;
  $("btnReset").disabled = false;
  sim = createDemoSimulator();
  const loop = () => {
    if (!running) return;
    const frame = sim();
    ear = smooth(ear, frame.ear, 0.4);
    mar = smooth(mar, frame.mar, 0.4);
    pitch = smooth(pitch, frame.pitch, 0.25);
    yaw = smooth(yaw, frame.yaw, 0.25);
    roll = smooth(roll, 4, 0.25);
    headDir = frame.dir;

    closedFrames = frame.eyeClosed ? closedFrames + 1 : 0;
    if (frame.eyeClosed && closedFrames >= CFG.EAR_CONSECUTIVE_FRAMES) frame.eyeClosure = true;
    mar < CFG.MAR_THRESHOLD ? (yawnFrames = 0) : null;
    yawnFrames = frame.yawn ? Math.max(yawnFrames, CFG.YAWN_CONSECUTIVE_FRAMES) : Math.max(0, yawnFrames - 1);
    const yawnNow = frame.yawn && !yawnActive;
    if (yawnNow) { stats.yawns++; yawnActive = true; }
    if (!frame.yawn) yawnActive = false;

    awayFrames = frame.away ? Math.max(awayFrames + 1, CFG.AWAY_CONSECUTIVE_FRAMES) : 0;
    if (frame.away && !awayActive) { awayActive = true; stats.away++; }
    if (!frame.away) awayActive = false;

    risk.update({
      eye_closure: !!frame.eyeClosure, eye_closure_frames: Math.max(closedFrames, CFG.EAR_CONSECUTIVE_FRAMES),
      yawn: frame.yawn, yawn_frames: Math.max(yawnFrames, CFG.YAWN_CONSECUTIVE_FRAMES),
      looking_away: frame.away, away_frames: Math.max(awayFrames, CFG.AWAY_CONSECUTIVE_FRAMES),
      hand_down: frame.handDown, phone: frame.phone,
    });
    updateUI({ eyeClosure: !!frame.eyeClosure, yawn: frame.yawn, lookingAway: frame.away, eyesOffRoad: false });
    reportToBackend({ eyeClosure: !!frame.eyeClosure, yawn: frame.yawn, lookingAway: frame.away, eyesOffRoad: false });
    setTimeout(loop, 100);
  };
  loop();
  noFaceEl.classList.add("hidden");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  ctx.fillStyle = "#2dd06e";
  ctx.font = "bold 26px Segoe UI, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("DEMO MODE", overlay.width / 2, overlay.height / 2);
}

function createDemoSimulator() {
  let t = 0, blinkPhase = 0, blinkOn = false;
  return function step() {
    t += 100;
    // blink every ~4s
    blinkPhase += 100;
    if (blinkPhase > 4000) { blinkOn = true; blinkPhase = 0; blinkDur = 0; }
    let ear = 0.34;
    let eyeClosed = false;
    if (blinkOn) { blinkDur++; if (blinkDur < 4) { ear = 0.15; if (blinkDur === 3 && ear < CFG.EAR_THRESHOLD) {} } else blinkOn = false; }
    if (t > 9000 && t < 17500) { ear = 0.12; eyeClosed = true; }       // sustained closure
    if (t > 21000 && t < 26000) { ear = 0.12; eyeClosed = true; }       // second closure -> DANGER
    if (t > 28000) { ear = 0.12; eyeClosed = true; }

    let mar = 0.18;
    let yawn = false;
    if ((t > 8000 && t < 9500) || (t > 20500 && t < 22000)) { mar = 0.62; yawn = true; }

    let pitch = 0, yaw = 0, dir = "FORWARD";
    if (t > 12000 && t < 13500) { yaw = 26; dir = "LEFT"; }            // looking away
    if (t > 13500 && t < 15000) { yaw = -26; dir = "RIGHT"; }

    return {
      ear, eyeClosed, mar, yawn, pitch, yaw, dir,
      away: dir !== "FORWARD",
      phone: t > 19000 && t < 20500,
      handDown: false,
    };
  };
}
let blinkDur = 0;

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------
async function startMonitoring() {
  try {
    await startCamera();
  } catch (e) {
    alert("Camera blocked or unavailable. Use Demo Mode instead.");
    return;
  }
  demoMode = false;
  running = true;
  sessionStart = Date.now();
  $("btnStart").disabled = true;
  $("btnDemo").disabled = false;
  $("btnReset").disabled = false;
  await loadModel();
  lastVideoTime = -1;
  rafId = requestAnimationFrame(analyzeFrame);
}

function stopAll() {
  running = false;
  cancelAnimationFrame(rafId);
  sim = null;
  if (video.srcObject) video.srcObject.getTracks().forEach((tr) => tr.stop());
}

$("btnStart").addEventListener("click", startMonitoring);
$("btnDemo").addEventListener("click", () => { stopAll(); startDemo(); });
$("btnReset").addEventListener("click", () => {
  Object.assign(stats, { blinks: 0, yawns: 0, closures: 0, away: 0, phone: 0 });
  risk.reset();
  sessionStart = Date.now();
  ear = 0; mar = 0; pitch = 0; yaw = 0; roll = 0; closedFrames = 0; yawnFrames = 0; awayFrames = 0;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  updateUI({ eyeClosure: false, yawn: false, lookingAway: false, eyesOffRoad: false });
  fetch(API_BASE + "/api/reset?session_id=" + encodeURIComponent(sessionId)).catch(() => {});
});

setApiStatus(false);
apiTextEl.textContent = "API: " + API_BASE.replace(/^https?:\/\//, "");