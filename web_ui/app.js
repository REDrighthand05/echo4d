"use strict";
/* ECHO web shell.
   Layers, back to front: the living wall (random graffiti of tiny
   matte beads + the scene echo while a job runs) -- one whole
   sheet of frosted glass -- borderless transparent components.

   The graffiti is redrawn at random on every launch: one to three
   strokes, never more. Since v3 the hand is GIANT -- each stroke
   is a thick sweep wider than the window, cropped by its edges --
   and the idle wall carries ambient motion: the strokes breathe
   and wander very slowly, the mist drifts, a pale sheen passes.

   If a pywebview bridge is present it drives everything for real;
   in a plain browser a demo state machine mirrors the app's own
   action logic so the shell can be judged on its own. */

/* The bridge is bound lazily, NEVER snapshotted at script load:
   pywebview injects window.pywebview.api on its own schedule, and
   on some machines that lands after this script runs. Snapshotting
   it here once stranded the whole app in demo mode for the entire
   session (2026-11-02, the user's machine): simulated jobs, dead
   buttons, and not a word about it. bindBridge() below retries
   until the api appears, and demo mode must announce itself. */
let bridge = null;
let bridgeMode = "waiting";   // "waiting" -> "real" | "demo"

/* ------------------------------------------------------------------ */
/* appearance （设置 → 显示）: every tunable number of the look.     */
/* The defaults ARE v0.6.4's shipped values, so an untouched       */
/* install looks untouched; the bridge overwrites them from the    */
/* saved settings at boot, the settings page rewrites them live,   */
/* and each change is saved back through the bridge.               */
/* ------------------------------------------------------------------ */
const APPEARANCE_DEFAULTS = {
  frostBlur: 2.5, frostVeil: 0.26, frostSaturate: 155,
  workHaze: 0,
  setBlur: 24, setVeil: 0.72, setSaturate: 200,
  grafCount: 0, grafWidth: 1, grafOpacity: 1, grafSaturate: 115,
  mistDensity: 100, mistSize: 1,
  motionOn: true, motionSpeed: 1, motionAmp: 1, sheenOn: true,
};
let appearance = { ...APPEARANCE_DEFAULTS };
let ambT = 0;         // the ambient clock, scaled by the speed setting
let motionLevel = 1;  // eased 0..1 gate so toggling motion never snaps

function applyAppearance() {
  const a = appearance, root = document.documentElement.style;
  root.setProperty("--frost-blur", a.frostBlur + "px");
  root.setProperty("--frost-veil", a.frostVeil);
  root.setProperty("--frost-sat", a.frostSaturate + "%");
  /* the running pane: the haze slider interpolates the cleared
     switchable glass back towards the idle frost -- at 0 it is
     exactly v0.6.4's nearly transparent pane */
  const t = a.workHaze / 100;
  root.setProperty("--frost-blur-w", (0.5 + t * (a.frostBlur - 0.5)) + "px");
  root.setProperty("--frost-veil-w", 0.05 + t * (a.frostVeil - 0.05));
  root.setProperty("--frost-sat-w", (108 + t * (a.frostSaturate - 108)) + "%");
  root.setProperty("--set-blur", a.setBlur + "px");
  root.setProperty("--set-veil", a.setVeil);
  root.setProperty("--set-sat", a.setSaturate + "%");
  root.setProperty("--graf-opacity", a.grafOpacity);
  root.setProperty("--graf-sat", a.grafSaturate + "%");
  root.setProperty("--sheen-dur", (34 / Math.max(0.2, a.motionSpeed)) + "s");
  document.body.classList.toggle("no-sheen", !a.sheenOn || !a.motionOn);
}

let appearanceSaveTimer = null;
function saveAppearance() {
  if (!bridge) return;
  clearTimeout(appearanceSaveTimer);
  appearanceSaveTimer = setTimeout(() => {
    bridge.save_settings({ appearance: { ...appearance } });
  }, 250);
}

/* ------------------------------------------------------------------ */
/* random graffiti: 1-3 strokes, a fresh hand every launch            */
/* ------------------------------------------------------------------ */
const INK_BLUE = [0, 87, 217], INK_SOFT = [110, 134, 175],
      INK_PALE = [150, 168, 198], PALE_FAR = [198, 214, 238];

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const launchRng = mulberry32((Date.now() ^ (Math.random() * 1e9)) >>> 0);

function smoothLine(pts, sub = 7) {
  // Catmull-Rom through the control points -> a hand-drawn polyline
  if (pts.length < 3) return pts;
  const out = [pts[0]];
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[Math.max(0, i - 1)], p1 = pts[i],
          p2 = pts[i + 1], p3 = pts[Math.min(pts.length - 1, i + 2)];
    for (let j = 1; j <= sub; j++) {
      const t = j / sub, t2 = t * t, t3 = t2 * t;
      out.push([
        0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t +
               (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
               (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3),
        0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t +
               (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
               (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3),
      ]);
    }
  }
  return out;
}

function randomStrokes(w, h, rng) {
  const R = (a, b) => a + rng() * (b - a);
  const pick = arr => arr[Math.floor(rng() * arr.length)];
  /* GIANT hand (v3, user 2026-11-04): every stroke is a window-
     filling sweep with a thick brush, cropped by the window's
     edges -- only part of each gesture is ever on stage. Still
     one to three strokes per launch, never more. The small
     doodle kinds (spiral, loop) retire: at giant scale only the
     sweep family reads as a hand passing over the whole wall. */
  const kinds = {
    // a freehand zigzag wave clear across the window
    wave(cx, cy, span) {
      const n = 5 + Math.floor(rng() * 3), pts = [];
      let up = rng() < 0.5;
      for (let i = 0; i < n; i++) {
        pts.push([cx - span / 2 + (span * i) / (n - 1),
                  cy + (up ? -1 : 1) * span * R(0.07, 0.13)]);
        up = !up;
      }
      return { pts: smoothLine(pts, 6) };
    },
    // one huge swoosh / underline curve, edge to edge and beyond
    swoosh(cx, cy, span) {
      const pts = [[cx - span / 2, cy + R(-0.04, 0.04) * h],
                   [cx - span * 0.15, cy + R(0.10, 0.20) * h],
                   [cx + span * 0.25, cy + R(-0.16, -0.05) * h],
                   [cx + span / 2, cy + R(-0.05, 0.06) * h]];
      return { pts: smoothLine(pts, 9) };
    },
    // a sharp pulse / mountain polyline across the whole width
    pulse(cx, cy, span) {
      const n = 5 + Math.floor(rng() * 3), pts = [];
      for (let i = 0; i < n; i++)
        pts.push([cx - span / 2 + (span * i) / (n - 1),
                  cy + (i % 2 ? -1 : 1) * span * R(0.05, 0.11)]);
      return { pts: smoothLine(pts, 6) };
    },
    // a vast gentle arc -- part of a circle far bigger than
    // the window, its centre thrown well off-stage
    arc(cx, cy) {
      const pts = [], rad = R(0.55, 1.05) * w, dir = rng() < 0.5 ? 1 : -1;
      const ccx = cx + R(-0.15, 0.15) * w,
            ccy = cy + R(0.30, 0.80) * h * (rng() < 0.5 ? 1 : -1);
      const a0 = Math.atan2(cy - ccy, cx - ccx), sweep = R(0.5, 0.95);
      for (let i = 0; i <= 36; i++) {
        const th = a0 + dir * (i / 36) * sweep;
        pts.push([ccx + rad * Math.cos(th), ccy + rad * Math.sin(th)]);
      }
      return { pts };
    },
  };
  /* 笔数： 0 in the settings means the shipped behaviour --
     a random one to three; a fixed number deals exactly that */
  const count = appearance.grafCount > 0 ? appearance.grafCount
              : rng() < 0.4 ? 1 : rng() < 0.62 ? 2 : 3;
  const strokes = [];
  for (let i = 0; i < count; i++) {
    const kind = i === 0 ? pick(["wave", "wave", "swoosh", "arc"])
                         : pick(["wave", "swoosh", "pulse", "arc"]);
    // every span overflows the window: the brush enters one
    // edge and leaves by another (or the same one, for the arc)
    const span = (i === 0 ? R(1.18, 1.50) : R(0.92, 1.30)) * w;
    const cx = w / 2 + R(-0.10, 0.10) * w, cy = R(0.08, 0.92) * h;
    const st = kinds[kind](cx, cy, span);
    st.width = (i === 0 ? R(0.11, 0.18) : R(0.06, 0.115)) * w
             * appearance.grafWidth;
    st.color = i === 0 ? INK_BLUE : pick([INK_SOFT, INK_PALE, INK_SOFT]);
    st.alpha = i === 0 ? 238 : Math.round(R(130, 165));
    strokes.push(st);
  }
  return strokes;
}

/* ------------------------------------------------------------------ */
/* the bead field: tiny matte dots, no gloss, no balloons             */
/* ------------------------------------------------------------------ */
const canvas = document.getElementById("beads");
const ctx = canvas.getContext("2d");
const echoCanvas = document.getElementById("echo");
const echoCtx = echoCanvas.getContext("2d");
let W = 0, H = 0, beads = [], mouse = null, lastT = 0;
let wallStrokes = [];
/* ambient motion (v3): each giant stroke gets its own slow
   clock -- a two-to-three-minute wander and a ~48 s breath.
   Reduced-motion users get a still wall. */
let strokeAnim = [];
const REDUCED_MOTION = !!(window.matchMedia &&
  matchMedia("(prefers-reduced-motion: reduce)").matches);
let echoSrc = null;  // image/canvas/video shown behind the glass

function gauss() {
  return Math.max(-1.15, Math.min(1.15, (Math.random() + Math.random() + Math.random()) / 1.5 - 1));
}

function pushOutOfZones(x, y, zones) {
  for (const [zx0, zy0, zx1, zy1] of zones) {
    if (x > zx0 && x < zx1 && y > zy0 && y < zy1) {
      const pens = [x - zx0, zx1 - x, y - zy0, zy1 - y];
      const side = pens.indexOf(Math.min(...pens));
      if (side === 0) x = zx0 - 5; else if (side === 1) x = zx1 + 5;
      else if (side === 2) y = zy0 - 5; else y = zy1 + 5;
    }
  }
  return [x, y];
}

function buildField() {
  W = window.innerWidth; H = window.innerHeight;
  canvas.width = W; canvas.height = H;
  echoCanvas.width = W; echoCanvas.height = H;
  beads = [];
  const zones = calmZones();
  wallStrokes = randomStrokes(W, H, launchRng);
  strokeAnim = wallStrokes.map(() => ({
    phX: Math.random() * Math.PI * 2,
    phY: Math.random() * Math.PI * 2,
    phB: Math.random() * Math.PI * 2,
  }));
  wallStrokes.forEach((st, si) => {
    // The stroke is not painted -- it is CONSTITUTED by micro
    // particles: a dense jittered lattice fills the whole band,
    // each dot a tiny fraction of the whole, overlapping into
    // one rich solid ribbon whose rim feathers out particle by
    // particle. Same bead objects, same physics as always.
    // The lattice step grows with the brush: a giant ribbon
    // keeps the same ~30 rows of mist across its band instead
    // of exploding the bead count (and the frame budget).
    const step = Math.max(2.4, Math.min(6.5, st.width / 30));
    let sAcc = 0, carry = 0;
    const pushBead = (hx, hy, hz) => {
      const [px, py] = pushOutOfZones(hx, hy, zones);
      const ang = Math.random() * Math.PI * 2;
      beads.push({
        si,
        rx: hx, ry: hy, hx: px, hy: py, hz, x: px, y: py, z: hz,
        vx: 0, vy: 0, vz: 0,
        rj: 0.85 + Math.random() * 0.3,
        color: st.color, alpha: st.alpha / 255,
        sdirx: Math.cos(ang), sdiry: Math.sin(ang),
        zdir: Math.random() < 0.5 ? -1 : 1,
        phase: Math.random() * Math.PI * 2,
      });
    };
    for (let i = 0; i < st.pts.length - 1; i++) {
      const [x0, y0] = st.pts[i], [x1, y1] = st.pts[i + 1];
      const segLen = Math.hypot(x1 - x0, y1 - y0);
      if (!segLen) continue;
      const nx = -(y1 - y0) / segLen, ny = (x1 - x0) / segLen;
      let d = carry;
      while (d < segLen) {
        const t = d / segLen, sHere = sAcc + d;
        const cxp = x0 + (x1 - x0) * t, cyp = y0 + (y1 - y0) * t;
        for (let off = -st.width / 2; off <= st.width / 2; off += step) {
          const u = off / (st.width / 2);
          /* 雾点密度： the acceptance probability scales with
             the setting -- at 0 no dot is ever dealt and the
             wall is blank, as the slider promises */
          if (Math.random() > (0.10 + 0.90 * Math.cos(u * Math.PI / 2))
                            * (appearance.mistDensity / 100)) continue;
          const hx = cxp + nx * off + (Math.random() * 2 - 1) * 1.15;
          const hy = cyp + ny * off + (Math.random() * 2 - 1) * 1.15;
          let hz = 0.45 * Math.cos(u * Math.PI / 2)
                 + 0.24 * Math.sin(sHere * 0.010 + si * 1.7)
                 + (Math.random() * 2 - 1) * 0.30;
          pushBead(hx, hy, Math.max(-1, Math.min(1, hz)));
        }
        d += step;
      }
      carry = d - segLen;
      sAcc += segLen;
    }
  });
}

/* Window geometry changes NEVER re-roll the graffiti: the launch
   strokes and their beads are the launch's hand, for the whole
   session. A resize only scales homes and positions along, and
   the canvases follow immediately so nothing stretches, jumps or
   re-deals while the window is dragged. */
function applyCalmZones() {
  const zones = calmZones();
  for (const b of beads) {
    const [px, py] = pushOutOfZones(b.rx, b.ry, zones);
    b.hx = px; b.hy = py;
  }
}

function rescaleField() {
  const nW = window.innerWidth, nH = window.innerHeight;
  if (!nW || !nH) return;
  if (nW === W && nH === H) { applyCalmZones(); return; }
  const sx = W ? nW / W : 1, sy = H ? nH / H : 1;
  W = nW; H = nH;
  canvas.width = W; canvas.height = H;
  echoCanvas.width = W; echoCanvas.height = H;
  for (const b of beads) {
    b.rx *= sx; b.ry *= sy; b.x *= sx; b.y *= sy;
  }
  applyCalmZones();
}

function calmZones() {
  const zones = [];
  const range = document.createRange();
  document.querySelectorAll(".calm, .btn, .tile h2, #header h1").forEach(el => {
    range.selectNodeContents(el);
    const r = range.getBoundingClientRect();
    if (r.width >= 4 && r.height >= 4)
      zones.push([r.left - 12, r.top - 10, r.right + 12, r.bottom + 10]);
  });
  return zones;
}

/* matte dot sprites: flat colour, only the alpha falls off at the
   rim so a dot never shines -- no highlight, no belly, no balloon */
const sprites = new Map();
function spriteFor(color, zn) {
  const step = Math.round(zn * 7);
  const key = color.join(",") + "|" + step;
  if (sprites.has(key)) return sprites.get(key);
  const S = 32, c = document.createElement("canvas");
  c.width = c.height = S;
  const g = c.getContext("2d"), R = S / 2;
  const mix = [150, 185, 255].map((f, i) => Math.round(color[i] + (f - color[i]) * 0.55 * (1 - step / 7)));
  const grad = g.createRadialGradient(R, R, 0, R, R, R);
  grad.addColorStop(0, `rgba(${mix.join(",")},1)`);
  grad.addColorStop(0.8, `rgba(${mix.join(",")},1)`);
  grad.addColorStop(1, `rgba(${mix.join(",")},0)`);
  g.fillStyle = grad;
  g.fillRect(0, 0, S, S);
  sprites.set(key, c);
  return c;
}

function drawEcho(time) {
  echoCtx.clearRect(0, 0, W, H);
  if (!echoSrc) return;
  const iw = echoSrc.videoWidth || echoSrc.width;
  const ih = echoSrc.videoHeight || echoSrc.height;
  if (!iw || !ih) return;
  const drift = 1.07 + 0.03 * Math.sin(time / 14000);
  const scale = Math.max(W / iw, H / ih) * drift;
  const w = iw * scale, h = ih * scale;
  const px = (W - w) / 2 + Math.sin(time / 19000) * 14;
  const py = (H - h) / 2 + Math.cos(time / 23000) * 10;
  echoCtx.drawImage(echoSrc, px, py, w, h);
}

function tick(t) {
  requestAnimationFrame(tick);
  const dt = Math.min(0.05, Math.max(0.008, lastT ? (t - lastT) / 1000 : 0.016));
  lastT = t;
  const time = t / 1000;
  drawEcho(t);
  ctx.clearRect(0, 0, W, H);
  /* ambient motion: the giant strokes themselves are alive --
     each wanders a few centimetres over minutes and breathes
     in and out over ~48 s, on top of the mist's own shimmer.
     Everything stays slow enough to read as weather, not UI.
     The settings own its clock now （设置 → 显示 → 常态动效）:
     ambT advances at the user's speed, motionLevel eases the
     whole dance in and out so the switch never snaps, and the
     amplitude scales wander + breath + shimmer together.
     Reduced-motion users keep their still wall regardless. */
  const motionTarget = (appearance.motionOn && !REDUCED_MOTION) ? 1 : 0;
  motionLevel += (motionTarget - motionLevel) * Math.min(1, dt * 3);
  ambT += dt * appearance.motionSpeed * motionLevel;
  const ampL = appearance.motionAmp * motionLevel;
  const driftX = [], driftY = [], breathA = [];
  for (let s = 0; s < strokeAnim.length; s++) {
    const a = strokeAnim[s];
    driftX[s] = Math.sin(ambT * 0.045 + a.phX) * 15 * ampL;
    driftY[s] = Math.cos(ambT * 0.037 + a.phY) * 10 * ampL;
    breathA[s] = 1 - ampL * 0.15 + ampL * 0.15 * Math.sin(ambT * 0.13 + a.phB);
  }
  const order = beads.map((b, i) => [b.z, i]).sort((a, b) => a[0] - b[0]);
  for (const [, i] of order) {
    const b = beads[i];
    const swayx = b.hx + (driftX[b.si] || 0) + Math.sin(ambT * 0.9 + b.phase) * 3.0 * ampL;
    const swayy = b.hy + (driftY[b.si] || 0) + Math.cos(ambT * 0.7 + b.phase * 1.3) * 2.1 * ampL;
    const swayz = b.hz + Math.sin(ambT * 0.5 + b.phase) * 0.12 * ampL;
    let ax = 30 * (swayx - b.x) - 9.3 * b.vx;
    let ay = 30 * (swayy - b.y) - 9.3 * b.vy;
    b.vz += (14 * (swayz - b.z) - 6 * b.vz) * dt;
    if (mouse) {
      const dx = b.x - mouse.x, dy = b.y - mouse.y;
      const d = Math.hypot(dx, dy);
      if (d < 140 && d > 0.01) {
        const zn0 = (b.z + 1.15) / 2.3;
        const f = 1 - d / 140;
        ax += (dx / d) * 4200 * (0.5 + 0.8 * zn0) * f + b.sdirx * 950 * f;
        ay += (dy / d) * 4200 * (0.5 + 0.8 * zn0) * f + b.sdiry * 950 * f;
        b.vz += f * b.zdir * 62 * dt * 3;
      }
    }
    b.vx += ax * dt; b.vy += ay * dt;
    const sp = Math.hypot(b.vx, b.vy);
    if (sp > 950) { b.vx *= 950 / sp; b.vy *= 950 / sp; }
    b.x += b.vx * dt; b.y += b.vy * dt;
    b.z = Math.max(-1.3, Math.min(1.3, b.z + b.vz * dt));
    const zn = Math.max(0, Math.min(1, (b.z + 1.15) / 2.3));
    const r = (1.7 + 1.15 * zn) * b.rj * appearance.mistSize;
    ctx.globalAlpha = b.alpha * (breathA[b.si] || 1) * (0.6 + 0.4 * zn);
    ctx.drawImage(spriteFor(b.color, zn), b.x - r, b.y - r, r * 2, r * 2);
  }
  ctx.globalAlpha = 1;
}

window.addEventListener("mousemove", e => { mouse = { x: e.clientX, y: e.clientY }; });
window.addEventListener("mouseout", () => { mouse = null; });
window.addEventListener("resize", rescaleField);

/* working state: the graffiti recedes, the scene takes the wall */
let holdTimer = null;
function setWorking(on) {
  clearTimeout(holdTimer);
  if (on) document.body.classList.add("working");
  else holdTimer = setTimeout(() => document.body.classList.remove("working"), 2600);
}

/* ------------------------------------------------------------------ */
/* app state (mirrors the desktop build's action logic)               */
/* ------------------------------------------------------------------ */
const $ = id => document.getElementById(id);
const state = {
  video: null, videoPath: null, busy: false,
  jobDir: null, depthReady: false,
  jobOutputs: { any: false, scene: false, preview: false, replay: false },
  lastHistory: [], lastInbox: [],
};
const logBody = $("logbody");
function log(line) {
  const ts = new Date().toTimeString().slice(0, 8);
  logBody.textContent += `[${ts}] ${line}\n`;
  logBody.scrollTop = logBody.scrollHeight;
}
function setStatus(text, frac) {
  $("status").textContent = text;
  if (frac !== undefined) {
    $("track").classList.toggle("active", frac > 0);
    $("fill").style.width = (frac * 100).toFixed(1) + "%";
    $("filldot").style.left = (frac * 100).toFixed(1) + "%";
  }
}
function refresh() {
  const en = (id, on) => { $(id).disabled = !on; };
  en("btn-start", state.video && !state.busy);
  en("btn-herodepth", state.video && !state.busy);
  en("btn-pick", !state.busy);
  en("btn-open", state.jobOutputs.scene && !state.busy);
  en("btn-folder", state.jobOutputs.any && !state.busy);
  en("btn-replay", state.jobOutputs.preview && !state.busy);
  en("btn-route", state.jobOutputs.replay && !state.busy);
  en("btn-depth", !state.busy);
  en("btn-playdepth", state.depthReady && !state.busy);
  en("btn-markers", !state.busy);
  $("breathdot").classList.toggle("busy", state.busy);
}
function selectVideo(name, path) {
  state.video = name;
  state.videoPath = path || null;
  setOmnibox(path || name);
  $("hero-desc").innerHTML =
    `已选：${name}<br>点「开始生成」建模，或者先做一段深度视频。`;
  setStatus("已选好视频，点「开始生成」就行。", 0);
  if (!bridge) log(`已选择视频：${name}`);
  refresh();
}

/* file picking + drag and drop */
/* frameless window controls (bridge only; inert in demo mode) */
$("win-min").addEventListener("click", () => { if (bridge) bridge.win_minimize(); });
$("win-max").addEventListener("click", () => { if (bridge) bridge.win_toggle_max(); });
$("win-close").addEventListener("click", () => { if (bridge) bridge.win_close(); });
$("titlebar").addEventListener("dblclick", (e) => {
  if (e.target.closest(".tb-btns, #tabstrip")) return;
  if (bridge) bridge.win_toggle_max();
});

$("btn-pick").addEventListener("click", async () => {
  if (bridge) {
    const r = await bridge.pick_video();
    if (r && r.ok) selectVideo(r.name, r.name);
    return;
  }
  $("fileinput").click();
});
$("fileinput").addEventListener("change", e => {
  if (e.target.files[0]) selectVideo(e.target.files[0].name);
});
const hero = $("hero");
["dragenter", "dragover"].forEach(ev => hero.addEventListener(ev, e => {
  e.preventDefault(); if (!state.busy) hero.classList.add("drag");
}));
["dragleave", "drop"].forEach(ev => hero.addEventListener(ev, e => {
  e.preventDefault(); hero.classList.remove("drag");
}));
hero.addEventListener("drop", async e => {
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (!f || state.busy) return;
  if (bridge) {
    if (f.path) {
      const r = await bridge.select_video_path(f.path);
      if (r && r.ok) selectVideo(r.name, f.path);
      else if (r) setStatus(r.message);
    } else {
      setStatus("拖放取不到文件路径，点「选择视频」挑一下。");
    }
    return;
  }
  selectVideo(f.name);
});

/* demo visuals: stand-in frames so the sync experience can be
   judged in a plain browser (the bridge feeds real frames) */
const demoScene = document.createElement("canvas");
demoScene.width = 880; demoScene.height = 520;
const demoPts = [];
{
  const r = mulberry32(20261102);
  for (let i = 0; i < 2400; i++) {
    const a = r(), b = r();
    demoPts.push([90 + a * 700, 90 + b * 340, r()]);
  }
}
function demoSceneFrame(k) {
  const g = demoScene.getContext("2d");
  g.fillStyle = "#05070e"; g.fillRect(0, 0, 880, 520);
  g.strokeStyle = "rgba(70,130,255,0.16)"; g.lineWidth = 1;
  for (let i = 0; i < 9; i++) {
    g.beginPath(); g.moveTo(60 + i * 95, 440); g.lineTo(300 + i * 35, 90); g.stroke();
  }
  for (let i = 0; i < 6; i++) {
    g.beginPath(); g.moveTo(40, 430 - i * 58); g.lineTo(840, 430 - i * 58); g.stroke();
  }
  const n = Math.min(demoPts.length, 120 + k * 160);
  for (let i = 0; i < n; i++) {
    const [x, y, z] = demoPts[i];
    g.fillStyle = z > 0.7 ? "rgba(210,230,255,0.95)" : "rgba(120,170,255,0.75)";
    g.fillRect(x, y, 2.2, 2.2);
  }
  g.strokeStyle = "rgba(140,190,255,0.8)"; g.lineWidth = 1.6;
  g.beginPath();
  const routeN = Math.min(40, 4 + k * 4);
  for (let i = 0; i < routeN; i++) {
    const t = i / 40, x = 140 + t * 560, y = 400 - Math.sin(t * 5.2) * 46 - t * 120;
    i ? g.lineTo(x, y) : g.moveTo(x, y);
  }
  g.stroke();
  return demoScene;
}
const demoDepth = document.createElement("canvas");
demoDepth.width = 880; demoDepth.height = 520;
function demoDepthFrame(k) {
  const g = demoDepth.getContext("2d");
  g.fillStyle = "#000"; g.fillRect(0, 0, 880, 520);
  const blobs = [[300 + Math.sin(k * 0.22) * 60, 260, 240],
                 [600 + Math.cos(k * 0.17) * 50, 210, 180],
                 [450, 430 + Math.sin(k * 0.11) * 22, 300]];
  for (const [x, y, r] of blobs) {
    const grad = g.createRadialGradient(x, y, 8, x, y, r);
    grad.addColorStop(0, "rgba(255,255,255,0.95)");
    grad.addColorStop(0.55, "rgba(150,150,150,0.55)");
    grad.addColorStop(1, "rgba(0,0,0,0)");
    g.fillStyle = grad;
    g.fillRect(x - r, y - r, r * 2, r * 2);
  }
  return demoDepth;
}

/* demo job run (plain browser only) */
function demoRun(kind) {
  if (bridgeMode !== "demo") {
    setStatus("正在连上软件后台，稍等几秒再点。");
    return;
  }
  state.busy = true; refresh(); setWorking(true);
  const stages = kind === "depth"
    ? ["正在读视频", "正在估深度", "正在渲染深度画面"]
    : ["正在抽帧", "正在估深度", "正在融合点云", "正在生成场景"];
  let i = 0, frame = 0;
  log(kind === "depth" ? "开始做深度视频。" : "开始生成场景。");
  const frameTimer = setInterval(() => {
    const src = kind === "depth" ? demoDepthFrame(frame) : demoSceneFrame(frame);
    frame++;
    drawPreviewSource(src);
    echoSrc = src;
  }, kind === "depth" ? 340 : 820);
  const step = () => {
    if (i < stages.length) {
      setStatus(stages[i] + "…", (i + 0.4) / stages.length);
      log(stages[i]);
      i++; setTimeout(step, 1250);
    } else {
      clearInterval(frameTimer);
      state.busy = false;
      if (kind === "depth") { state.depthReady = true; setStatus("深度视频做好了。", 1); log("深度视频完成。"); }
      else {
        state.jobOutputs = { any: true, scene: true, preview: true, replay: true };
        setStatus("场景做好了。点「打开场景」就能看。", 1); log("场景生成完成。");
      }
      setWorking(false);
      refresh();
    }
  };
  step();
}
$("btn-start").addEventListener("click", async () => {
  if (bridge) { await bridge.start_scene(); return; }
  demoRun("scene");
});
$("btn-herodepth").addEventListener("click", async () => {
  if (bridge) { await bridge.start_depth(); return; }
  demoRun("depth");
});
$("btn-depth").addEventListener("click", async () => {
  if (!state.video) {
    if (bridge) { const r = await bridge.pick_video(); if (r && r.ok) selectVideo(r.name, r.name); }
    else $("fileinput").click();
    return;
  }
  if (bridge) { await bridge.start_depth(); return; }
  demoRun("depth");
});
$("btn-playdepth").addEventListener("click", async () => {
  if (bridge) { await bridge.play_depth(); return; }
  log("播放深度视频（演示）。");
});
$("btn-open").addEventListener("click", async () => {
  if (bridge) { await bridge.open_scene(state.jobDir); return; }
  log("打开场景（演示）。");
});
$("btn-folder").addEventListener("click", async () => {
  if (bridge) { await bridge.open_folder(state.jobDir); return; }
  log("打开文件夹（演示）。");
});
$("btn-replay").addEventListener("click", async () => {
  if (bridge) { await bridge.open_preview_replay(state.jobDir); return; }
  log("过程回放（演示）。");
});
$("btn-route").addEventListener("click", async () => {
  if (bridge) { await bridge.open_route_replay(state.jobDir); return; }
  log("路线回放（演示）。");
});
$("btn-markers").addEventListener("click", async () => {
  if (bridge) { await bridge.print_markers(); return; }
  log("打印标记纸（演示）。");
});

/* history */
function renderHistory(items) {
  state.lastHistory = items;
  const box = $("histlist");
  box.innerHTML = "";
  for (const it of items) {
    const div = document.createElement("div");
    div.className = "hist-item" + (state.jobDir === it.dir ? " sel" : "");
    const label = it.finished ? it.name : it.name + "（未完成）";
    div.innerHTML = `<span>${label}</span>` +
      (it.scale ? `<span class="when">${it.scale}</span>` : "") +
      `<span class="when">${it.when}</span>`;
    div.addEventListener("click", async () => {
      state.jobDir = it.dir;
      setOmnibox(it.name);
      state.jobOutputs = { any: true, scene: it.finished,
                           preview: it.preview, replay: it.replay };
      if (bridge) await bridge.select_job(it.dir);
      renderHistory(items);
      refresh();
    });
    box.appendChild(div);
  }
}

/* phone-capture inbox: captures that synced themselves home */
function renderInbox(items) {
  state.lastInbox = items;
  const tile = $("inbox");
  const box = $("inboxlist");
  tile.hidden = !items || items.length === 0;
  box.innerHTML = "";
  for (const it of items || []) {
    const div = document.createElement("div");
    div.className = "hist-item";
    div.innerHTML = `<span>${it.name}</span>` +
      `<span class="when">${it.mb} MB</span>` +
      `<span class="when">${it.when}</span>`;
    div.addEventListener("click", async () => {
      if (state.busy || !bridge) return;
      const r = await bridge.select_inbox_capture(it.dir);
      if (r && r.ok) selectVideo(r.name, it.video);
      else if (r) setStatus(r.message);
    });
    box.appendChild(div);
  }
}

/* log panel + settings page + fullscreen */
$("logtab").addEventListener("click", () => {
  const p = $("logpanel");
  p.classList.toggle("open");
  $("logtab").innerHTML = p.classList.contains("open") ? "日<br>志<br>‹" : "日<br>志<br>›";
});
function openSettings() {
  const page = $("settings-page");
  page.classList.remove("leave");
  page.hidden = false;
  page.classList.add("enter");
  settingsOpen = true;
  syncTabs();
  syncNav();
}
$("btn-settings").addEventListener("click", openSettings);
$("btn-settings-top").addEventListener("click", openSettings);
$("btn-set-back").addEventListener("click", () => {
  settingsOpen = false;
  syncTabs();
  syncNav();
  const page = $("settings-page");
  page.classList.remove("enter");
  page.classList.add("leave");
  setTimeout(() => { page.hidden = true; page.classList.remove("leave"); }, 230);
});
$("btn-full").addEventListener("click", () => {
  const pv = $("preview");
  if (document.fullscreenElement) document.exitFullscreen();
  else pv.requestFullscreen && pv.requestFullscreen();
});
$("btn-update").addEventListener("click", async () => {
  if (bridge) await bridge.download_update();
});
$("btn-set-update").addEventListener("click", async () => {
  const note = $("set-update-note");
  if (!bridge) {
    note.textContent = bridgeMode === "demo"
      ? "演示模式，没有真的去检查。"
      : "正在连上软件后台，稍等几秒再试。";
    return;
  }
  note.textContent = "正在检查…";
  try {
    const r = await bridge.manual_check_update();
    if (r && r.status === "update") {
      note.textContent = `有新版本 v${r.version} —— 回到主界面，顶部点「下载并安装」。`;
    } else if (r && r.status === "latest") {
      note.textContent = "已是最新版。";
    } else {
      note.textContent = "检查失败：直连和本机常见代理都试过，还是连不上 GitHub。可以到发布页手动更新。";
    }
  } catch (e) {
    note.textContent = "检查失败：连不上 GitHub。";
  }
});
$("set-proxy").addEventListener("change", async e => {
  if (bridge) await bridge.save_settings({ updateProxy: e.target.value.trim() });
});
$("btn-set-folder").addEventListener("click", async () => {
  if (!bridge) return;
  const p = await bridge.pick_jobs_root();
  if (p) {
    await bridge.save_settings({ jobsRoot: p });
    $("set-jobs").textContent = p;
    $("settings-desc").textContent = `任务文件夹：${p} · 当前版本 v${window.__echoVersion || ""}`;
  }
});
$("set-autostart").addEventListener("change", async e => {
  if (bridge) await bridge.save_settings({ autostart: e.target.checked });
});
$("set-tray").addEventListener("change", async e => {
  if (bridge) await bridge.save_settings({ closeToTray: e.target.checked });
});

/* settings page → 显示： every appearance slider and switch.
   "input" applies the number live (the settings veil itself is
   right there changing), "change" persists it -- and for the
   structural numbers (stroke count / brush width / mist
   density, which live inside the bead field) re-deals the wall
   once the drag lets go. */
const AP_RANGES = [
  // [element id, appearance key, value format, structural?]
  ["ap-frost-blur", "frostBlur", v => v.toFixed(1) + " px"],
  ["ap-frost-veil", "frostVeil", v => v.toFixed(2)],
  ["ap-frost-saturate", "frostSaturate", v => Math.round(v) + "%"],
  ["ap-work-haze", "workHaze", v => String(Math.round(v))],
  ["ap-set-blur", "setBlur", v => Math.round(v) + " px"],
  ["ap-set-veil", "setVeil", v => v.toFixed(2)],
  ["ap-set-saturate", "setSaturate", v => Math.round(v) + "%"],
  ["ap-graf-count", "grafCount", v => v === 0 ? "随机" : v + " 笔", true],
  ["ap-graf-width", "grafWidth", v => "×" + v.toFixed(2), true],
  ["ap-graf-opacity", "grafOpacity", v => Math.round(v * 100) + "%"],
  ["ap-graf-saturate", "grafSaturate", v => Math.round(v) + "%"],
  ["ap-mist-density", "mistDensity", v => Math.round(v) + "%", true],
  ["ap-mist-size", "mistSize", v => "×" + v.toFixed(2)],
  ["ap-motion-speed", "motionSpeed", v => "×" + v.toFixed(1)],
  ["ap-motion-amp", "motionAmp", v => "×" + v.toFixed(2)],
];
function syncAppearanceControls() {
  for (const [id, key, fmt] of AP_RANGES) {
    $(id).value = appearance[key];
    $(id + "-val").textContent = fmt(appearance[key]);
  }
  $("ap-motion-on").checked = !!appearance.motionOn;
  $("ap-sheen-on").checked = !!appearance.sheenOn;
}
function rebuildWall() {
  buildField();
  requestAnimationFrame(applyCalmZones);
}
for (const [id, key, fmt, structural] of AP_RANGES) {
  const el = $(id);
  el.addEventListener("input", () => {
    appearance[key] = parseFloat(el.value);
    $(id + "-val").textContent = fmt(appearance[key]);
    document.body.classList.add("tuning");
    applyAppearance();
  });
  el.addEventListener("change", () => {
    document.body.classList.remove("tuning");
    if (structural) rebuildWall();
    saveAppearance();
  });
}
for (const [id, key] of [["ap-motion-on", "motionOn"],
                          ["ap-sheen-on", "sheenOn"]]) {
  $(id).addEventListener("change", e => {
    appearance[key] = e.target.checked;
    applyAppearance();
    saveAppearance();
  });
}
$("ap-reroll").addEventListener("click", () => {
  rebuildWall();
  log("换了一手涂鸦。");
});
$("ap-reset").addEventListener("click", () => {
  appearance = { ...APPEARANCE_DEFAULTS };
  syncAppearanceControls();
  applyAppearance();
  rebuildWall();
  saveAppearance();
  log("显示参数已恢复默认。");
});

/* ------------------------------------------------------------------ */
/* chrome frame: tabs, nav cluster, omnibox, overflow menu. Every    */
/* control only re-aims an action the shell already has -- no new    */
/* behaviour, just Chrome's skeleton over the same logic.            */
/* ------------------------------------------------------------------ */
let settingsOpen = false;
function setOmnibox(text) {
  if (text) $("omnibox-text").textContent = text;
}
function syncNav() {
  $("nav-back").classList.toggle("is-disabled", !settingsOpen);
}
function syncTabs() {
  $("tab-main").classList.toggle("active", !settingsOpen);
  $("tab-settings").classList.toggle("active", settingsOpen);
}
$("tab-main").addEventListener("click", () => {
  if (settingsOpen) $("btn-set-back").click();
});
$("tab-settings").addEventListener("click", () => {
  if (!settingsOpen) openSettings();
});
$("nav-back").addEventListener("click", () => {
  if (settingsOpen) $("btn-set-back").click();
});
$("nav-fwd").addEventListener("click", () => { /* nowhere forward to go */ });
$("nav-reload").addEventListener("click", () => {
  renderHistory(state.lastHistory || []);
  renderInbox(state.lastInbox || []);
  const b = $("nav-reload");
  b.classList.remove("spinning");
  void b.offsetWidth;
  b.classList.add("spinning");
  if (!bridge) log("已刷新列表。");
});
$("omnibox").addEventListener("click", () => { $("btn-pick").click(); });

const menuEl = $("menu");
function closeMenu() { menuEl.hidden = true; }
$("btn-menu").addEventListener("click", e => {
  e.stopPropagation();
  menuEl.hidden = !menuEl.hidden;
});
document.addEventListener("click", e => {
  if (!menuEl.hidden && !e.target.closest(".tb-right")) closeMenu();
});
document.addEventListener("keydown", e => {
  if (e.key === "Escape") closeMenu();
});
$("menu-settings").addEventListener("click", () => { closeMenu(); openSettings(); });
$("menu-update").addEventListener("click", () => {
  closeMenu(); openSettings();
  $("btn-set-update").click();
});
$("menu-log").addEventListener("click", () => { closeMenu(); $("logtab").click(); });
$("menu-about").addEventListener("click", () => {
  closeMenu(); openSettings(); setSetFilter("update");
});

/* settings page: category nav + live search -- both are pure
   filters over the rows that are already on the page */
function applySetFilter() {
  const input = $("set-search-input");
  const q = (input ? input.value : "").trim().toLowerCase();
  const active = document.querySelector(".set-nav-item.active");
  const cat = active ? active.dataset.cat : "all";
  let any = false;
  document.querySelectorAll(".set-group").forEach(g => {
    let groupVisible = false;
    g.querySelectorAll(".set-row").forEach(r => {
      const vis = q ? r.textContent.toLowerCase().includes(q)
                    : (cat === "all" || g.dataset.cat === cat);
      r.style.display = vis ? "" : "none";
      if (vis) groupVisible = true;
    });
    g.style.display = groupVisible ? "" : "none";
    if (groupVisible) any = true;
  });
  $("set-noresult").hidden = any;
}
function setSetFilter(cat) {
  document.querySelectorAll(".set-nav-item").forEach(b =>
    b.classList.toggle("active", b.dataset.cat === cat));
  applySetFilter();
}
document.querySelectorAll(".set-nav-item").forEach(b =>
  b.addEventListener("click", () => {
    $("set-search-input").value = "";
    setSetFilter(b.dataset.cat);
  }));
$("set-search-input").addEventListener("input", applySetFilter);

/* preview screen: frames, depth video, orbit drag */
const pvc = $("pvcanvas").getContext("2d");
let pvSeq = -1, pvVideoUrl = null;
function drawPreviewSource(src) {
  const c = $("pvcanvas");
  const iw = src.videoWidth || src.width, ih = src.videoHeight || src.height;
  if (!iw || !ih) return;
  $("pvvideo").hidden = true;
  c.width = c.clientWidth * 2 || 880; c.height = c.clientHeight * 2 || 520;
  const scale = Math.max(c.width / iw, c.height / ih);
  const w = iw * scale, h = ih * scale;
  pvc.drawImage(src, (c.width - w) / 2, (c.height - h) / 2, w, h);
  document.querySelector(".pv-hint").style.display = "none";
}
function showDepthVideo(url) {
  if (pvVideoUrl === url) return;
  pvVideoUrl = url;
  const v = $("pvvideo");
  v.src = url;
  v.hidden = false;
  v.play().catch(() => {});
  echoSrc = v;
  document.querySelector(".pv-hint").style.display = "none";
}
function drawPreview(dataUrl) {
  const img = new Image();
  img.onload = () => { drawPreviewSource(img); echoSrc = img; };
  img.src = dataUrl;
}
let pvDrag = null;
$("pvcanvas").addEventListener("mousedown", e => {
  pvDrag = { x: e.clientX, y: e.clientY, pan: e.button === 2 };
});
window.addEventListener("mousemove", e => {
  if (pvDrag && bridge) {
    bridge.preview_drag(e.clientX - pvDrag.x, e.clientY - pvDrag.y, pvDrag.pan);
    pvDrag.x = e.clientX; pvDrag.y = e.clientY;
  }
});
window.addEventListener("mouseup", () => { pvDrag = null; });
$("pvcanvas").addEventListener("wheel", e => {
  if (bridge) { e.preventDefault(); bridge.preview_zoom(e.deltaY < 0 ? 1 : -1); }
}, { passive: false });
$("pvcanvas").addEventListener("contextmenu", e => e.preventDefault());
$("pvcanvas").addEventListener("dblclick", () => $("btn-full").click());

/* bridge boot + polling */
async function initBridge() {
  const snap = await bridge.get_snapshot();
  window.__echoVersion = snap.version;
  $("settings-desc").textContent =
    `任务文件夹：${snap.jobsRoot} · 当前版本 v${snap.version}`;
  $("set-jobs").textContent = snap.jobsRoot;
  $("set-version").textContent = `当前版本 v${snap.version}`;
  $("set-autostart").checked = !!snap.settings.autostart;
  $("set-tray").checked = !!snap.settings.closeToTray;
  $("set-proxy").value = snap.settings.updateProxy || "";
  if (snap.settings.appearance) {
    appearance = { ...APPEARANCE_DEFAULTS, ...snap.settings.appearance };
    syncAppearanceControls();
    applyAppearance();
    /* a saved hand shapes the wall only when its structural
       numbers differ from the shipped ones -- otherwise the
       launch's own random hand stands, as it always has */
    const structural = ["grafCount", "grafWidth", "mistDensity"];
    if (structural.some(k => appearance[k] !== APPEARANCE_DEFAULTS[k]))
      rebuildWall();
  }
  (snap.log || []).forEach(l => log(l));
  renderHistory(snap.history || []);
  renderInbox(snap.inbox || []);
  setStatus(snap.status, snap.frac);
  bridge.check_update();
  let wasBusy = false;
  setInterval(async () => {
    let out;
    try { out = await bridge.poll(); } catch (e) { return; }
    state.busy = !!out.busy;
    state.depthReady = !!out.depthReady;
    if (out.busy && !wasBusy) setWorking(true);
    if (!out.busy && wasBusy) setWorking(false);
    wasBusy = !!out.busy;
    (out.logs || []).forEach(l => log(l));
    if (out.history) renderHistory(out.history);
    if (out.inbox) renderInbox(out.inbox);
    if (!out.busy && out.hasJob && !state.jobDir && out.jobOutputs) {
      // Only outputs the job really holds may light their buttons;
      // a failed job used to light them all and every click then
      // silently did nothing.
      state.jobOutputs = out.jobOutputs;
    }
    setStatus(out.status, out.frac);
    refresh();
    if (out.update) {
      $("updatebox").hidden = false;
      $("updatelabel").textContent = out.update.hot
        ? `有新版本 v${out.update.version}（热更新 · 约 ${out.update.mb} MB）`
        : `有新版本 v${out.update.version}`;
    }
    if (out.depthUrl) showDepthVideo(out.depthUrl);
    if (out.previewSeq !== pvSeq) {
      pvSeq = out.previewSeq;
      const fr = await bridge.get_preview_frame();
      if (fr && fr.data) drawPreview(fr.data);
    }
  }, 450);
}

/* boot */
buildField();
applyAppearance();
syncAppearanceControls();
requestAnimationFrame(tick);
setTimeout(applyCalmZones, 350);  // re-measure calm zones once text layout settles
refresh();
log("ECHO 已就绪。");

/* bridge binding: try at once, on pywebview's ready event, and on
   a retry timer. Six seconds without an api and the shell admits
   it is in demo mode (visible badge + status), but the retry never
   stops: if the api shows up late, the shell upgrades to real. */
function bindBridge() {
  if (bridge) return true;
  const api = window.pywebview && window.pywebview.api;
  if (!api) return false;
  bridge = api;
  bridgeMode = "real";
  const badge = $("demobadge");
  if (badge) badge.hidden = true;
  log("已连上软件后台。");
  initBridge();
  return true;
}
function declareDemo() {
  if (bridge || bridgeMode === "demo") return;
  bridgeMode = "demo";
  const badge = $("demobadge");
  if (badge) badge.hidden = false;
  setStatus("演示模式：没有连上软件后台，生成和按钮都是演示。", 0);
  log("演示模式：没有连上软件后台。");
}
window.addEventListener("pywebviewready", bindBridge);
bindBridge();
let bridgeTries = 0;
const bridgeTimer = setInterval(() => {
  bridgeTries++;
  if (bindBridge()) { clearInterval(bridgeTimer); return; }
  if (bridgeTries === 24) declareDemo();          // ~6 s in
  if (bridgeTries % 8 === 0) bindBridge();        // keep retrying
}, 250);
