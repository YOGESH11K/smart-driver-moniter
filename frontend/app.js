import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9/vision_bundle.mjs";

// ---------------------------------------------------------------------------
// Thresholds (time-based; mirror config/config.yaml where applicable)
// ---------------------------------------------------------------------------
const CFG = {
  EAR_THRESHOLD: 0.25,          // EAR below this = eye closed (sensitive: works from normal webcam distance)
  EYE_MIN_FRAMES: 2,            // require 2 bad frames before starting the eye timer (no single-frame alerts)
  EYE_ALERT_S: 2.0,             // 2s continuous eye closure -> ALERT
  MAR_THRESHOLD: 0.45,          // MAR above this = mouth open (yawning)
  YAWN_MIN_FRAMES: 2,
  YAWN_ALERT_S: 2.0,            // 2s continuous yawning -> ALERT
  PITCH_THRESHOLD: 15,
  YAW_THRESHOLD: 20,
  AWAY_ALERT_S: 3.0,            // head away longer than this -> warning
  FACE_LOST_ALERT_S: 2.0,       // face missing -> "EYES OFF ROAD" alert
  PHONE_CONFIDENCE: 0.32,
  PHONE_CHECK_MS: 1000,         // how often the AI phone detector runs (~1/sec)
  PHONE_STRIKES: 2,             // 2 consecutive positives before phone is "confirmed"
  PHONE_ALERT_S: 1.2,           // 1.2s sustained phone -> ALERT (debounced)
  ALARM_COOLDOWN_MS: 4000,      // min gap between alarm restarts
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
const noFaceEl = $("noFace"), camErrorEl = $("camError"), camNoteEl = $("camNote"), alertOverlay = $("alertOverlay");
const alertReasonEl = $("alertReason");
const apiStatusEl = $("apiStatus"), apiTextEl = $("apiText");
const statusEl = $("statusIndicator"), statusTextEl = $("statusText");
const gaugeFill = $("gaugeFill"), scoreEl = $("score"), reasonEl = $("reason");
const fpsBadge = $("fpsBadge");
const btnStart = $("btnStart"), btnDemo = $("btnDemo"), btnReset = $("btnReset"), btnMute = $("btnMute"), btnTest = $("btnTest");
const btnPhoneAI = $("btnPhoneAI"), btnPhoneSim = $("btnPhoneSim"), btnHandSim = $("btnHandSim");
const dots = { eye: $("dotEye"), yawn: $("dotYawn"), phone: $("dotPhone") };

const browserOk = {
  camera: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
  audio: !!(window.AudioContext || window.webkitAudioContext),
};

// ---------------------------------------------------------------------------
// Session / runtime state
// ---------------------------------------------------------------------------
let faceLandmarker = null;
let running = false;
let demoMode = false;
let rafId = 0;
let lastVideoTime = -1;
let sessionId = "web-" + Math.random().toString(36).slice(2, 9);
let sessionStart = null;
let sim = null;

let prevTime = performance.now();
let fpsEma = 0;

let ear = 0, mar = 0, pitch = 0, yaw = 0, roll = 0, headDir = "FORWARD";
let faceLostFrames = 0;
let earStrikes = 0, marStrikes = 0, aweStrike = 0, belowStreak = 0;
const holds = { eye: 0, yawn: 0, phone: 0, face: 0 };
let awayActive = false;

let phoneSimFlag = false, handSimFlag = false;
let phoneAIFlag = false, phoneStrikes = 0, phoneConf = 0, phoneAIReady = false;

const stats = { blinks: 0, yawns: 0, closures: 0, away: 0, phone: 0 };

// ---------------------------------------------------------------------------
// Time-based condition timers (reset when the condition disappears)
// ---------------------------------------------------------------------------
function makeCond(alertS, onAlert, onClear) {
  return {
    since: null, elapsed: 0, alert: false, alertS, onAlert, onClear,
    update(active, now) {
      if (active) {
        if (this.since === null) this.since = now;
        this.elapsed = (now - this.since) / 1000;
        if (this.elapsed >= this.alertS && !this.alert) {
          this.alert = true;
          if (this.onAlert) this.onAlert();
        }
      } else {
        if (this.since !== null && this.alert && this.onClear) this.onClear();
        this.since = null; this.elapsed = 0; this.alert = false;
      }
    },
    reset() { this.since = null; this.elapsed = 0; this.alert = false; },
  };
}
const eyeCond = makeCond(CFG.EYE_ALERT_S, () => stats.closures++);
const yawnCond = makeCond(CFG.YAWN_ALERT_S, () => stats.yawns++);
const phoneCond = makeCond(CFG.PHONE_ALERT_S, () => stats.phone++);
const faceCond = makeCond(CFG.FACE_LOST_ALERT_S);

// ---------------------------------------------------------------------------
// Risk engine mirror (safety gauge + backend parity)
// ---------------------------------------------------------------------------
const risk = {
  risk_score: 0, safety_score: 100, state: "SAFE", state_counter: 0, total_risk_events: 0,
  update(inputs) {
    let points = 0;
    if (inputs.eye_closure) points += CFG.WEIGHTS.eye_closure * Math.min(inputs.eye_closure_seconds / 2, 1);
    if (inputs.yawn) points += CFG.WEIGHTS.yawn * Math.min(inputs.yawn_seconds / 2, 1);
    if (inputs.looking_away) points += CFG.WEIGHTS.head_away * Math.min(inputs.away_seconds / 3, 1);
    if (inputs.hand_down) points += CFG.WEIGHTS.hand_down * 0.8;
    if (inputs.phone) points += CFG.WEIGHTS.phone * 0.9;
    const maxPossible = Object.values(CFG.WEIGHTS).reduce((a, b) => a + b, 0);
    const norm = Math.min((points / maxPossible) * 100, 100);
    this.risk_score = norm;
    this.safety_score = Math.max(0, 100 - norm);
    const newState = computeState(this.safety_score);
    const oldState = this.state;
    if (newState !== this.state) {
      this.state_counter++;
      if (this.state_counter >= 5) {
        this.state = newState;
        this.state_counter = 0;
        if (STATES.indexOf(newState) > STATES.indexOf(oldState)) this.total_risk_events++;
      }
    } else this.state_counter = 0;
  },
  reset() { this.risk_score = 0; this.safety_score = 100; this.state = "SAFE"; this.state_counter = 0; this.total_risk_events = 0; },
};
const computeState = (s) => s >= 80 ? "SAFE" : s >= 60 ? "CAUTION" : s >= 40 ? "ATTENTION REQUIRED" : s >= 20 ? "DROWSY / DISTRACTED" : "DANGER";

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
  const r = (Math.atan2(eyeR.y - eyeL.y, eyeR.x - eyeL.x) * 180) / Math.PI;
  const midX = (earPtL.x + earPtR.x) / 2;
  const y = (Math.atan2(nose.y - (earPtL.y + earPtR.y) / 2, nose.x - midX) * 180) / Math.PI;
  const p = (Math.atan2(nose.y - (lm[10].y + lm[152].y) / 2, Math.abs(nose.z + 0.1)) * 180) / Math.PI * 0.5;
  return { pitch: p, yaw: y, roll: r };
}

// ---------------------------------------------------------------------------
// MediaPipe face landmarks (client-side, no frames leave the browser)
// ---------------------------------------------------------------------------
async function loadFaceModel() {
  if (faceLandmarker) return true;
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9/wasm"
    );
    try {
      faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task", delegate: "GPU" },
        runningMode: "VIDEO", numFaces: 1,
      });
    } catch {
      faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task", delegate: "CPU" },
        runningMode: "VIDEO", numFaces: 1,
      });
    }
    return !!faceLandmarker;
  } catch (err) {
    console.error("Face model load failed:", err);
    return false;
  }
}

// ---------------------------------------------------------------------------
// Enable / disable UI around camera + privacy
// ---------------------------------------------------------------------------
function showCamError(msg) {
  camErrorEl.textContent = msg;
  camErrorEl.classList.remove("hidden");
}
function clearCamError() {
  camErrorEl.classList.add("hidden");
}

async function startCamera() {
  if (!browserOk.camera) {
    showCamError("This browser does not support camera access. Use Demo Mode instead.");
    return false;
  }
  clearCamError();
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" }, audio: false });
    video.srcObject = stream;
    await video.play();
    overlay.width = video.videoWidth || 640;
    overlay.height = video.videoHeight || 480;
    return true;
  } catch (err) {
    if (err && (err.name === "NotAllowedError" || err.name === "PermissionDeniedError")) {
      showCamError("Camera permission was denied. Allow access in your browser, or use Demo Mode.");
    } else {
      showCamError("Camera unavailable (" + (err && err.name ? err.name : "error") + "). Use Demo Mode.");
    }
    throw err;
  }
}

// ---------------------------------------------------------------------------
// In-browser phone detection (TensorFlow.js COCO-SSD) - debounced
// ---------------------------------------------------------------------------
let cocoModel = null, cocoLoading = false, lastPhoneCheck = 0;

async function loadPhoneModel() {
  if (cocoModel || cocoLoading) return;
  cocoLoading = true;
  btnPhoneAI.textContent = "Phone AI: loading…";
  try {
    if (!window.cocoSsd || !window.tf) throw new Error("tf/coco-ssd missing");
    cocoModel = await cocoSsd.load({ base: "mobilenet_v2" });
    phoneAIReady = true;
    btnPhoneAI.classList.add("on");
    btnPhoneAI.textContent = "Phone AI: ON";
  } catch (e) {
    console.error("Phone model failed:", e);
    btnPhoneAI.textContent = "Phone AI: unavail.";
    btnPhoneAI.disabled = true;
  } finally {
    cocoLoading = false;
  }
}

async function checkPhoneAI(now) {
  if (!cocoModel || !running || demoMode) return;
  if (now - lastPhoneCheck < CFG.PHONE_CHECK_MS) return;
  lastPhoneCheck = now;
  try {
    const preds = await cocoModel.detect(video);
    let seen = false, best = 0;
    for (const p of preds) {
      if (p.class === "cell phone" && p.score >= CFG.PHONE_CONFIDENCE) {
        seen = true; best = Math.max(best, p.score);
      }
    }
    phoneConf = seen ? best : 0;
    if (seen) {
      phoneStrikes++;
      if (phoneStrikes >= CFG.PHONE_STRIKES) phoneAIFlag = true;
    } else {
      phoneStrikes = 0;
      phoneAIFlag = false;
    }
    if (seen) {
      ctx.strokeStyle = "#ff4d5e";
      ctx.lineWidth = 3;
      for (const p of preds) {
        if (p.class !== "cell phone") continue;
        const bw = overlay.width, bh = overlay.height, vw = video.videoWidth || 1, vh = video.videoHeight || 1;
        ctx.strokeRect(p.bbox[0] / vw * bw, p.bbox[1] / vh * bh, p.bbox[2] / vw * bw, p.bbox[3] / vh * bh);
        ctx.fillStyle = "rgba(255,77,94,.9)";
        ctx.font = "bold 14px Segoe UI, sans-serif";
        ctx.fillText("PHONE " + Math.round(p.score * 100) + "%", p.bbox[0] / vw * bw, (p.bbox[1] / vh * bh) - 6);
        break;
      }
    }
  } catch (e) { /* skip frame */ }
}

// ---------------------------------------------------------------------------
// Alarm (Web Audio; unlocked by the user clicking Start / Demo)
// ---------------------------------------------------------------------------
let audioCtx = null, masterGain = null, alarmTimer = null, alarmStoppedAt = 0, muted = false, alarmPhase = false;

function initAudio() {
  if (!browserOk.audio) return;
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    masterGain = audioCtx.createGain();
    masterGain.gain.value = 1.0;
    const comp = audioCtx.createDynamicsCompressor();
    comp.threshold.value = -8;
    comp.knee.value = 14;
    comp.ratio.value = 16;
    masterGain.connect(comp).connect(audioCtx.destination);
  }
  if (audioCtx.state === "suspended") audioCtx.resume().catch(() => {});
}
function emergencyTone(freq, dur, vol, sub) {
  if (!audioCtx || audioCtx.state !== "running") return;
  const t0 = audioCtx.currentTime;
  const env = audioCtx.createGain();
  env.gain.setValueAtTime(0.0001, t0);
  env.gain.exponentialRampToValueAtTime(vol, t0 + 0.03);
  env.gain.setValueAtTime(vol, t0 + dur - 0.05);
  env.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
  const osc = audioCtx.createOscillator();
  osc.type = "sawtooth";
  osc.frequency.setValueAtTime(freq * 0.82, t0);
  osc.frequency.exponentialRampToValueAtTime(freq, t0 + 0.06);
  osc.connect(env).connect(masterGain);
  osc.start(t0);
  osc.stop(t0 + dur + 0.05);
  const osc2 = audioCtx.createOscillator();
  const g2 = audioCtx.createGain();
  osc2.type = "square";
  osc2.frequency.setValueAtTime(freq * 1.5, t0);
  g2.gain.setValueAtTime(0.0001, t0);
  g2.gain.exponentialRampToValueAtTime(vol * 0.42, t0 + 0.03);
  g2.gain.setValueAtTime(vol * 0.42, t0 + dur - 0.05);
  g2.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
  osc2.connect(g2).connect(masterGain);
  osc2.start(t0);
  osc2.stop(t0 + dur + 0.05);
  if (sub) {
    const osc3 = audioCtx.createOscillator();
    const g3 = audioCtx.createGain();
    osc3.type = "sine";
    osc3.frequency.setValueAtTime(sub, t0);
    g3.gain.setValueAtTime(0.0001, t0);
    g3.gain.exponentialRampToValueAtTime(vol * 0.5, t0 + 0.05);
    g3.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
    osc3.connect(g3).connect(masterGain);
    osc3.start(t0);
    osc3.stop(t0 + dur + 0.05);
  }
}
function emergencySiren() {
  if (muted) return;
  initAudio();
  // company-grade "NEE-NAA" emergency horn on a low warning tone
  alarmPhase = !alarmPhase;
  emergencyTone(alarmPhase ? 660 : 1040, 0.36, 0.98, 196);
  if (navigator.vibrate) navigator.vibrate([220, 100, 220]);
}
function startAlarm() {
  if (muted || alarmTimer) return;
  initAudio();
  emergencySiren();
  alarmTimer = setInterval(emergencySiren, 390);
}
function testAlarm() {
  muted = false;
  applyMuteUI();
  let n = 0;
  initAudio();
  const ring = () => {
    if (!audioCtx || audioCtx.state !== "running") { initAudio(); return; }
    if (n < 8) { emergencySiren(); n++; setTimeout(ring, 400); }
    else if (alarmTimer) { clearInterval(alarmTimer); alarmTimer = null; }
  };
  ring();
}
function stopAlarm(recordCooldown) {
  if (alarmTimer) { clearInterval(alarmTimer); alarmTimer = null; }
  if (recordCooldown) alarmStoppedAt = performance.now();
}
function updateAlarm(now, alertActive) {
  if (!alertActive) { stopAlarm(true); return; }
  if (muted) { stopAlarm(false); return; }
  if (!alarmTimer && now - alarmStoppedAt >= CFG.ALARM_COOLDOWN_MS) startAlarm();
}
function applyMuteUI() {
  btnMute.textContent = muted ? "🔇 Unmute Alarm" : "Mute Alarm";
}

// ---------------------------------------------------------------------------
// Core evaluation - shared by live camera and demo mode
// ---------------------------------------------------------------------------
function evaluate(now, sig) {
  // flicker tolerance: a signal stays "on" for a few frames after it drops,
  // so a camera/laptop hiccup can't reset a real yawning / eye-closure timer.
  const holdCond = (active, key) => {
    if (active) { holds[key] = 0; return true; }
    if (holds[key] < 4) { holds[key]++; return true; }
    return false;
  };
  const eyeOn = holdCond(sig.eyeBelow && sig.facePresent, "eye");
  const yawnOn = holdCond(sig.marAbove && sig.facePresent, "yawn");
  const phoneOn = holdCond(sig.phone, "phone");
  const faceOn = holdCond(!sig.facePresent, "face");

  // eye closure (debounced, then timed)
  if (eyeOn) earStrikes++; else earStrikes = 0;
  const eyeEl = earStrikes >= CFG.EYE_MIN_FRAMES;
  eyeCond.update(eyeEl, now);

  // yawning
  if (yawnOn) marStrikes++; else marStrikes = 0;
  const yawn = marStrikes >= CFG.YAWN_MIN_FRAMES;
  yawnCond.update(yawn, now);

  // phone (debounced AI strikes / sim flags)
  phoneCond.update(phoneOn, now);

  // face present
  if (sig.facePresent) faceLostFrames = 0; else faceLostFrames++;
  faceCond.update(faceOn, now);

  // head away (warning + risk only, no alarm)
  if (sig.away) aweStrike++; else { aweStrike = 0; awayActive = false; }
  if (aweStrike >= 3 && sig.away && !awayActive) { awayActive = true; stats.away++; }

  const alertActive = eyeCond.alert || yawnCond.alert || phoneCond.alert || faceCond.alert;
  const anyActive = eyeCond.since !== null || yawnCond.since !== null || phoneCond.since !== null || faceCond.since !== null || awayActive;

  risk.update({
    eye_closure: eyeCond.alert, eye_closure_seconds: eyeCond.elapsed,
    yawn: yawnCond.alert, yawn_seconds: yawnCond.elapsed,
    looking_away: awayActive, away_seconds: aweStrike * 0.05,
    hand_down: sig.handDown,
    phone: phoneCond.alert,
  });

  updateUI({ now, eyeEl, yawn, phone: sig.phone, awayActive, faceLost: !sig.facePresent, alertActive, anyActive });
  updateAlarm(now, alertActive);
  reportToBackend(now, sig);
}

// ---------------------------------------------------------------------------
// UI updates
// ---------------------------------------------------------------------------
function updateUI(flags) {
  // --- big status indicator: NORMAL / WARNING / ALERT ---
  const s = risk.safety_score;
  scoreEl.textContent = Math.round(s);
  gaugeFill.style.width = s + "%";
  gaugeFill.style.background = s >= 80 ? "var(--green)" : s >= 60 ? "var(--yellow)" : s >= 40 ? "var(--orange)" : "var(--red)";

  statusEl.classList.toggle("normal", !flags.alertActive && !flags.anyActive);
  statusEl.classList.toggle("warning", !flags.alertActive && flags.anyActive);
  statusEl.classList.toggle("alert", flags.alertActive);
  statusTextEl.textContent = flags.alertActive ? "ALERT" : flags.anyActive ? "WARNING" : "NORMAL";

  // --- alert overlay + reason ---
  alertOverlay.classList.toggle("hidden", !flags.alertActive);
  const reasons = [];
  if (eyeCond.alert) reasons.push("Possible Drowsiness Detected");
  if (yawnCond.alert) reasons.push("Yawning Detected");
  if (phoneCond.alert) reasons.push("Phone Usage Detected");
  if (faceCond.alert) reasons.push("Eyes Off Road");
  alertReasonEl.textContent = reasons.length ? reasons.join(" & ") : "";

  if (flags.alertActive && reasons.length) {
    reasonEl.textContent = "ALERT: " + reasons.join(" & ");
    reasonEl.className = "reason bad";
  } else if (flags.anyActive) {
    const warns = [];
    if (eyeCond.since !== null) warns.push("eyes closed " + eyeCond.elapsed.toFixed(1) + "s");
    if (yawnCond.since !== null) warns.push("yawning " + yawnCond.elapsed.toFixed(1) + "s");
    if (phoneCond.since !== null) warns.push("phone " + phoneCond.elapsed.toFixed(1) + "s");
    if (faceCond.since !== null) warns.push("face lost " + faceCond.elapsed.toFixed(1) + "s");
    if (flags.awayActive) warns.push("looking away");
    reasonEl.textContent = "WARNING: " + warns.join(" · ");
    reasonEl.className = "reason warn";
  } else {
    reasonEl.textContent = "All clear — keep driving.";
    reasonEl.className = "reason";
  }

  // --- condition timers ---
  $("mEar").textContent = ear.toFixed(2);
  $("mMar").textContent = mar.toFixed(2);
  setTimer("eye", eyeCond, flags.eyeEl, "Eyes open", (c) => "Closed " + c.elapsed.toFixed(1) + "s");
  setTimer("yawn", yawnCond, flags.yawn, "Normal", (c) => "Yawning " + c.elapsed.toFixed(1) + "s");
  setTimer("phone", phoneCond, flags.phone, "Not detected", (c) => "Phone " + c.elapsed.toFixed(1) + "s", CFG.PHONE_ALERT_S);
  $("phoneConf").textContent = phoneAIReady ? (Math.round(phoneConf * 100) + "%") : "AI off";
  $("phoneMode").textContent = phoneAIReady ? (phoneAIFlag ? "detected" : "none") : (phoneSimFlag ? "simulated" : "disabled");

  // --- live signals ---
  $("mPitch").textContent = Math.round(pitch) + "°";
  $("mYaw").textContent = Math.round(yaw) + "°";
  $("mRoll").textContent = Math.round(roll) + "°";
  $("mDir").textContent = headDir;

  // --- session stats ---
  $("sBlinks").textContent = stats.blinks;
  $("sYawns").textContent = stats.yawns;
  $("sClosures").textContent = stats.closures;
  $("sPhone").textContent = stats.phone;
  $("sAway").textContent = stats.away;
  if (sessionStart) {
    const t = Math.floor((Date.now() - sessionStart) / 1000);
    $("sTime").textContent = Math.floor(t / 60) + ":" + String(t % 60).padStart(2, "0");
  }
}

function setTimer(key, cond, active, idleText, activeText, threshold = cond.alertS) {
  dots[key].className = "dot" + (cond.alert ? " alert" : active ? " active" : "");
  $("t" + key[0].toUpperCase() + key.slice(1)).textContent = active ? activeText(cond) : idleText;
  const bar = $("t" + key[0].toUpperCase() + key.slice(1) + "Bar");
  const pct = Math.min((cond.elapsed / threshold) * 100, 100);
  bar.style.width = (active ? pct : 0) + "%";
  bar.classList.toggle("alert", cond.alert);
}

// ---------------------------------------------------------------------------
// Backend reporting (metrics only - never frames)
// ---------------------------------------------------------------------------
let apiOnline = null, lastReport = 0;
async function reportToBackend(now, sig) {
  if (now - lastReport < 500) return;
  lastReport = now;
  try {
    const res = await fetch(API_BASE + "/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId, ear: +ear.toFixed(3), mar: +mar.toFixed(3),
        pitch: +pitch.toFixed(1), yaw: +yaw.toFixed(1), roll: +roll.toFixed(1),
        eye_closed: eyeCond.alert, eye_closure_frames: Math.round(eyeCond.elapsed * 30),
        yawn: yawnCond.alert, yawn_frames: Math.round(yawnCond.elapsed * 30),
        looking_away: awayActive || faceCond.alert, away_frames: 0,
        hand_down: sig.handDown,
        phone: phoneCond.alert,
        face_present: sig.facePresent,
      }),
    });
    if (!res.ok) throw new Error("bad status");
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
// Live camera pipeline
// ---------------------------------------------------------------------------
async function startMonitoring() {
  initAudio();
  btnStart.disabled = true;
  const camOk = await startCamera();
  if (!camOk) { btnStart.disabled = browserOk.camera; return; }
  btnStart.disabled = false;
  if (!running) {
    running = true; demoMode = false;
    sessionStart = Date.now();
    btnDemo.disabled = false; btnReset.disabled = false;
    btnPhoneSim.disabled = false; btnHandSim.disabled = false; btnPhoneAI.disabled = false;
    camNoteEl.classList.add("hidden");
    const ok = await loadFaceModel();
    if (!ok) {
      showCamError("Face AI failed to load (internet needed for the model). Use Demo Mode.");
    }
    if (!cocoModel) loadPhoneModel(); // phone AI is ON by default while monitoring
    lastVideoTime = -1;
    rafId = requestAnimationFrame(analyzeFrame);
  }
}

function analyzeFrame(timestamp) {
  if (!running || demoMode) return;
  const now = performance.now();
  const dt = now - prevTime;
  prevTime = now;
  if (dt > 0) fpsEma = 0.9 * fpsEma + 0.1 * (1000 / dt);
  fpsBadge.textContent = Math.round(fpsEma) + " FPS";

  let lm = null;
  if (video.currentTime !== lastVideoTime && video.readyState >= 2) {
    lastVideoTime = video.currentTime;
    if (faceLandmarker) {
      try {
        const res = faceLandmarker.detectForVideo(video, timestamp);
        if (res && res.faceLandmarks && res.faceLandmarks.length > 0) lm = res.faceLandmarks[0];
      } catch (e) { /* single-frame error */ }
    }
  }

  if (lm) {
    faceLostFrames = 0;
    drawFace(lm);
    ear = smooth(ear, (earOf(lm, EYE_LEFT) + earOf(lm, EYE_RIGHT)) / 2, 0.45);
    const mouthW = dist(lm[78], lm[308]);
    mar = mouthW > 0 ? dist(lm[13], lm[14]) / mouthW : 0;
    const hp = headPose(lm);
    pitch = smooth(pitch, hp.pitch, 0.25);
    yaw = smooth(yaw, hp.yaw, 0.25);
    roll = smooth(roll, hp.roll, 0.25);
    headDir = Math.abs(pitch) > CFG.PITCH_THRESHOLD ? (pitch > 0 ? "DOWN" : "UP")
      : Math.abs(yaw) > CFG.YAW_THRESHOLD ? (yaw > 0 ? "LEFT" : "RIGHT") : "FORWARD";
    noFaceEl.classList.add("hidden");
  } else {
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    noFaceEl.classList.toggle("hidden", faceLostFrames < 3);
    if (faceLostFrames >= 3) { ear = smooth(ear, ear, 0); mar = smooth(mar, mar, 0); }
  }

  // blink counter: a dip that lasted a bit but clearly was not a 2s drowsy closure
  if (ear < CFG.EAR_THRESHOLD) belowStreak++;
  else {
    if (belowStreak >= CFG.EYE_MIN_FRAMES && belowStreak < 18) stats.blinks++;
    belowStreak = 0;
  }

  evaluate(performance.now(), {
    facePresent: !!lm,
    eyeBelow: ear < CFG.EAR_THRESHOLD,
    marAbove: mar > CFG.MAR_THRESHOLD,
    phone: phoneAIFlag || phoneSimFlag,
    handDown: handSimFlag,
    away: headDir !== "FORWARD",
  });
  checkPhoneAI(performance.now()).then(() => {});
  rafId = requestAnimationFrame(analyzeFrame);
}

function drawFace(lm) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  for (const idx of [...EYE_LEFT.slice(0, 4), ...EYE_RIGHT.slice(0, 4), 1, 13, 14, 78, 308, 10, 152]) {
    const p = lm[idx];
    ctx.beginPath();
    ctx.arc(p.x * overlay.width, p.y * overlay.height, idx === 1 ? 4 : 2.4, 0, Math.PI * 2);
    ctx.fillStyle = idx === 1 ? "#ffd166" : "#4d7cff";
    ctx.fill();
  }
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
// Demo mode (proves the whole pipeline without a camera/phone)
// ---------------------------------------------------------------------------
function startDemo() {
  initAudio();
  if (running) stopAll();
  demoMode = true; running = true;
  sessionStart = Date.now();
  btnStart.textContent = "Stop Monitoring";
  btnStart.classList.add("danger");
  btnDemo.disabled = true; btnReset.disabled = false;
  btnPhoneSim.disabled = true; btnHandSim.disabled = true; btnPhoneAI.disabled = true;
  camNoteEl.classList.add("hidden");
  clearCamError();
  noFaceEl.classList.add("hidden");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  ctx.fillStyle = "#2dd06e";
  ctx.font = "bold 26px Segoe UI, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("DEMO MODE", overlay.width / 2, overlay.height / 2 - 16);
  sim = createDemoSimulator();
  const loop = () => {
    if (!running || !demoMode) return;
    const f = sim();
    ear = smooth(ear, Math.max(f.ear, 0.02), 0.4);
    mar = smooth(mar, Math.max(f.mar, 0.05), 0.4);
    pitch = f.pitch; yaw = f.yaw; roll = 4; headDir = f.dir;
    evaluate(performance.now(), {
      facePresent: true,
      eyeBelow: f.eyeBelow,
      marAbove: f.marAbove,
      phone: f.phone,
      handDown: f.handDown,
      away: f.away,
    });
    setTimeout(loop, 100);
  };
  loop();
}

function createDemoSimulator() {
  let t = 0;
  return function step() {
    t += 100; // 10 Hz
    const s = t / 1000; // seconds into the demo

    let eyeBelow = false;
    // brief blinks every ~4s (should NOT alert)
    if (t % 4000 >= 0 && t % 4000 < 400) eyeBelow = true;
    // sustained eye closures
    if ((s > 10 && s < 16) || (s > 24 && s < 30)) eyeBelow = true;

    let marAbove = false;
    if ((s > 6 && s < 9) || (s > 20 && s < 23) || (s > 34 && s < 37)) marAbove = true;

    let dir = "FORWARD", away = false;
    if ((s > 17 && s < 19.5) || (s > 31 && s < 33.5)) { away = true; dir = "LEFT"; }

    const phone = s > 31 && s < 33.5;

    return { ear: eyeBelow ? 0.12 : 0.34, eyeBelow, marAbove, phone, away, handDown: false, dir };
  };
}

// ---------------------------------------------------------------------------
// Stop / cleanup
// ---------------------------------------------------------------------------
function stopAll() {
  running = false; demoMode = false;
  cancelAnimationFrame(rafId);
  stopAlarm(false);
  sim = null;
  earStrikes = 0; marStrikes = 0; aweStrike = 0; awayActive = false;
  [eyeCond, yawnCond, phoneCond, faceCond].forEach((c) => c.reset());
  if (video.srcObject) {
    video.srcObject.getTracks().forEach((tr) => tr.stop());
    video.srcObject = null;
  }
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  btnStart.textContent = "Start Monitoring";
  btnStart.classList.remove("danger");
  btnReset.disabled = true;
  btnPhoneSim.disabled = btnHandSim.disabled = btnPhoneAI.disabled = true;
}

function refreshButtonState() {
  btnStart.textContent = running ? "Stop Monitoring" : "Start Monitoring";
  btnStart.classList.toggle("danger", running);
}

// ---------------------------------------------------------------------------
// Event wiring
// ---------------------------------------------------------------------------
async function toggleStart() {
  if (running) {
    stopAll();
    camNoteEl.classList.remove("hidden");
    statusEl.className = "status-indicator normal";
    statusTextEl.textContent = "NORMAL";
    return;
  }
  refreshButtonState();
  try {
    await startMonitoring();
  } catch (e) {
    btnStart.disabled = browserOk.camera ? false : true;
  }
  refreshButtonState();
}
btnStart.addEventListener("click", toggleStart);

btnDemo.addEventListener("click", () => {
  if (running) stopAll();
  startDemo();
  refreshButtonState();
});
btnReset.addEventListener("click", () => {
  const wasDemo = demoMode;
  Object.assign(stats, { blinks: 0, yawns: 0, closures: 0, away: 0, phone: 0 });
  risk.reset();
  sessionStart = Date.now();
  ear = 0; mar = 0; pitch = 0; yaw = 0; roll = 0;
  phoneStrikes = 0; phoneAIFlag = false; phoneConf = 0; phoneSimFlag = false; handSimFlag = false;
  btnPhoneSim.classList.remove("on"); btnHandSim.classList.remove("on");
  stopAll();
  if (wasDemo) startDemo(); else startMonitoring();
  refreshButtonState();
  fetch(API_BASE + "/api/reset?session_id=" + encodeURIComponent(sessionId)).catch(() => {});
});

btnMute.addEventListener("click", () => {
  initAudio();
  muted = !muted;
  if (muted) stopAlarm(false);
  else updateAlarm(performance.now(), eyeCond.alert || yawnCond.alert || phoneCond.alert || faceCond.alert);
  applyMuteUI();
});

btnTest.addEventListener("click", () => {
  initAudio();
  testAlarm();
});

btnPhoneAI.addEventListener("click", () => {
  if (cocoModel) {
    cocoModel = null; phoneAIReady = false; phoneAIFlag = false; phoneStrikes = 0;
    btnPhoneAI.classList.remove("on");
    btnPhoneAI.textContent = "Phone AI: off";
  } else {
    loadPhoneModel();
  }
});
btnPhoneSim.addEventListener("click", () => {
  phoneSimFlag = !phoneSimFlag;
  btnPhoneSim.classList.toggle("on", phoneSimFlag);
});
btnHandSim.addEventListener("click", () => {
  handSimFlag = !handSimFlag;
  btnHandSim.classList.toggle("on", handSimFlag);
});

// ---------------------------------------------------------------------------
// Initial state / graceful feature detection
// ---------------------------------------------------------------------------
applyMuteUI();
camNoteEl.classList.remove("hidden");
if (!browserOk.camera) {
  btnStart.disabled = true;
  showCamError("Camera API not supported in this browser. Use Demo Mode (AI simulation).");
}
if (!browserOk.audio) {
  btnMute.disabled = true;
  btnMute.textContent = "Alarm unavailable";
}
setApiStatus(false);
apiTextEl.textContent = "API: " + API_BASE.replace(/^https?:\/\//, "");
loadPhoneModel(); // warm up the phone AI in the background