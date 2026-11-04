"use strict";
/* ECHO web shell.
   Layers, back to front: the baked ambient picture (seamless
   loops on two stacked video layers, crossfaded) -- the GPU
   particle layer (the one live layer, hard-capped, see below)
   -- the scene echo while a job runs -- one static translucent
   veil -- borderless transparent components.

   Since v0.6.12 the ambience is a finished file, not a live
   render: the old bead wall and its full-window backdrop blur
   kept a renderer at a full CPU core while the app idled, so
   the whole light field was baked offline (tools/
   bake_ambient.py) and the page now only plays it back.

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
/* appearance （设置 → 显示）: playback of the baked picture.       */
/* The ambience is a finished seamless loop (tools/bake_ambient.py) */
/* that the page only plays; what the settings tune is which loop,  */
/* whether it moves, how bright it is, and the static veils above   */
/* it. The veil defaults ARE v0.6.4's shipped numbers, so an         */
/* untouched install looks untouched; the bridge overwrites them    */
/* from the saved settings at boot, the settings page rewrites      */
/* them live, and each change is saved back through the bridge.     */
/* ------------------------------------------------------------------ */
const AMBIENT_THEMES = {
  graffiti: { mp4: "assets/ambient/graffiti.mp4",
              poster: "assets/ambient/graffiti.jpg" },
  rings: { mp4: "assets/ambient/rings.mp4",
           poster: "assets/ambient/rings.jpg" },
};
const APPEARANCE_DEFAULTS = {
  motionOn: true, theme: "graffiti", brightness: 100, particleDensity: 85,
  veil: 0.26, workVeil: 0.05, setVeil: 0.72,
};
let appearance = { ...APPEARANCE_DEFAULTS };

/* two stacked video layers; a theme switch loads the incoming
   loop on the idle layer and crossfades opacity (~0.65 s, the
   .amb-layer transition), then parks the old layer paused. The
   switch runs only when the theme NAME changes, so dragging
   any slider never reloads a video. */
const ambStack = { active: null, idle: null, theme: null };

function applyAppearance() {
  const a = appearance, root = document.documentElement.style;
  root.setProperty("--veil", a.veil);
  root.setProperty("--veil-w", a.workVeil);
  root.setProperty("--set-veil", a.setVeil);
  root.setProperty("--amb-bright", a.brightness + "%");
  if (!ambStack.active) {
    ambStack.active = document.getElementById("amb-a");
    ambStack.idle = document.getElementById("amb-b");
    ambStack.theme = ambStack.active ? ambStack.active.dataset.theme : null;
  }
  if (ambStack.active) {
    const want = AMBIENT_THEMES[a.theme] ? a.theme : "graffiti";
    if (ambStack.theme !== want) {
      const from = ambStack.active, to = ambStack.idle;
      const f = AMBIENT_THEMES[want];
      if (to.dataset.theme !== want) {
        to.poster = f.poster;
        to.src = f.mp4;
        to.dataset.theme = want;
        to.load();
      }
      if (a.motionOn) { const p = to.play(); if (p) p.catch(() => {}); }
      to.classList.add("on");
      from.classList.remove("on");
      ambStack.active = to;
      ambStack.idle = from;
      ambStack.theme = want;
      setTimeout(() => { if (ambStack.idle === from) from.pause(); }, 750);
    }
    if (a.motionOn) {
      const p = ambStack.active.play();
      if (p) p.catch(() => {});
    } else {
      ambStack.active.pause();   // hold the current frame: a still
    }
  }
  ptsApply();
}

let appearanceSaveTimer = null;
function saveAppearance() {
  if (!bridge) return;
  clearTimeout(appearanceSaveTimer);
  appearanceSaveTimer = setTimeout(() => {
    bridge.save_settings({ appearance: { ...appearance } });
  }, 250);
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ------------------------------------------------------------------ */
/* the particle layer: the 3D mist dots + pointer dodge the user    */
/* ordered kept -- but NOT on the old DOM/2D road. One WebGL canvas, */
/* one draw call per frame, positions in a single dynamic buffer.   */
/* Hard caps: PTS_MAX particles whatever the setting says, DPR      */
/* capped at 1.5. The loop runs only while the window is visible    */
/* AND focused AND motion is on; otherwise it draws one frozen      */
/* frame and stops -- an idle or hidden app pays nothing for it.    */
/* The dodge is one distance check per particle per frame; nothing  */
/* here reads layout, moves DOM, or rebuilds on resize (positions   */
/* live in a virtual stage and map to NDC arithmetically, so a      */
/* window drag costs this layer exactly nothing extra).             */
/* ------------------------------------------------------------------ */
const PTS_MAX = 3600;
const PTS_DPR_CAP = 1.5;
const PTS_VW = 1600, PTS_VH = 900;
const pts = {
  gl: null, failed: false, posBuf: null, uPx: null,
  posArr: new Float32Array(PTS_MAX * 2),
  home: new Float32Array(PTS_MAX * 2),
  off: new Float32Array(PTS_MAX * 2),
  vel: new Float32Array(PTS_MAX * 2),
  z: new Float32Array(PTS_MAX),
  phase: new Float32Array(PTS_MAX),
  count: 0, raf: 0, focused: true,
  mx: -9999, my: -9999,
};

function ptsInit() {
  const cv = document.getElementById("pts");
  if (!cv) { pts.failed = true; return; }
  let gl = null;
  try {
    gl = cv.getContext("webgl", { alpha: true, antialias: false,
                                  depthWrite: false, depthTest: false });
  } catch (e) { gl = null; }
  if (!gl) { pts.failed = true; cv.style.display = "none"; return; }
  const sh = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error("shader");
    return s;
  };
  try {
    const prog = gl.createProgram();
    gl.attachShader(prog, sh(gl.VERTEX_SHADER,
      "attribute vec2 aPos; attribute float aZ; attribute float aSize;" +
      "varying float vZ; uniform float uPx;" +
      "void main(){ vZ = aZ; gl_Position = vec4(aPos, 0.0, 1.0);" +
      " gl_PointSize = aSize * (0.45 + 0.95 * aZ) * uPx; }"));
    gl.attachShader(prog, sh(gl.FRAGMENT_SHADER,
      "precision mediump float; varying float vZ;" +
      "void main(){ float d = length(gl_PointCoord - 0.5);" +
      " float a = smoothstep(0.5, 0.08, d) * (0.05 + 0.15 * vZ);" +
      " gl_FragColor = vec4(0.34, 0.44, 0.62, a); }"));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error("link");
    gl.useProgram(prog);
    pts.uPx = gl.getUniformLocation(prog, "uPx");

    const rnd = mulberry32(20261104);
    const misc = new Float32Array(PTS_MAX * 2);   // aZ, aSize pairs
    for (let i = 0; i < PTS_MAX; i++) {
      pts.home[i * 2] = rnd() * PTS_VW;
      pts.home[i * 2 + 1] = rnd() * PTS_VH;
      pts.z[i] = Math.pow(rnd(), 1.6);
      pts.phase[i] = rnd() * Math.PI * 2;
      misc[i * 2] = pts.z[i];
      misc[i * 2 + 1] = 2.0 + pts.z[i] * 3.4 + rnd() * 1.2;
    }
    const miscBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, miscBuf);
    gl.bufferData(gl.ARRAY_BUFFER, misc, gl.STATIC_DRAW);
    const aZ = gl.getAttribLocation(prog, "aZ");
    const aSize = gl.getAttribLocation(prog, "aSize");
    gl.enableVertexAttribArray(aZ);
    gl.vertexAttribPointer(aZ, 1, gl.FLOAT, false, 8, 0);
    gl.enableVertexAttribArray(aSize);
    gl.vertexAttribPointer(aSize, 1, gl.FLOAT, false, 8, 4);
    pts.posBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, pts.posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, pts.posArr, gl.DYNAMIC_DRAW);
    const aPos = gl.getAttribLocation(prog, "aPos");
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    pts.gl = gl;
    ptsResize();
  } catch (e) {
    pts.failed = true;
    cv.style.display = "none";
  }
}

function ptsResize() {
  const cv = document.getElementById("pts");
  if (!cv || !pts.gl) return;
  const dpr = Math.min(window.devicePixelRatio || 1, PTS_DPR_CAP);
  cv.width = Math.max(1, Math.round(window.innerWidth * dpr));
  cv.height = Math.max(1, Math.round(window.innerHeight * dpr));
  pts.gl.viewport(0, 0, cv.width, cv.height);
  pts.gl.uniform1f(pts.uPx, cv.height / Math.max(1, window.innerHeight));
}

function ptsWanted() {
  return !!pts.gl && appearance.motionOn && !document.hidden
    && pts.focused && pts.count > 0;
}

function ptsDraw(now) {
  const gl = pts.gl;
  if (!gl) return;
  const w = window.innerWidth, h = window.innerHeight;
  const s = Math.max(w / PTS_VW, h / PTS_VH);
  const ox = (w - PTS_VW * s) / 2, oy = (h - PTS_VH * s) / 2;
  const mx = (pts.mx - ox) / s, my = (pts.my - oy) / s;
  const tt = now / 1000, R = 130;
  const pos = pts.posArr, home = pts.home, off = pts.off, vel = pts.vel;
  for (let i = 0; i < pts.count; i++) {
    const j = i * 2, zz = pts.z[i], ph = pts.phase[i];
    const sway = 0.4 + zz;
    let px = home[j] + off[j] + Math.sin(tt * 0.13 * sway + ph) * 11 * sway;
    let py = home[j + 1] + off[j + 1]
      + Math.cos(tt * 0.10 * sway + ph * 1.3) * 7 * sway;
    const dx = px - mx, dy = py - my, d2 = dx * dx + dy * dy;
    if (d2 < R * R && d2 > 0.01) {
      const d = Math.sqrt(d2), f = (1 - d / R) * 3.0;
      vel[j] += (dx / d) * f;
      vel[j + 1] += (dy / d) * f;
    }
    vel[j] += -off[j] * 0.016;
    vel[j + 1] += -off[j + 1] * 0.016;
    vel[j] *= 0.90;
    vel[j + 1] *= 0.90;
    off[j] += vel[j];
    off[j + 1] += vel[j + 1];
    px = home[j] + off[j] + Math.sin(tt * 0.13 * sway + ph) * 11 * sway;
    py = home[j + 1] + off[j + 1]
      + Math.cos(tt * 0.10 * sway + ph * 1.3) * 7 * sway;
    pos[j] = ((px * s + ox) / w) * 2 - 1;
    pos[j + 1] = 1 - ((py * s + oy) / h) * 2;
  }
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  if (pts.count > 0) {
    gl.bindBuffer(gl.ARRAY_BUFFER, pts.posBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, pos.subarray(0, pts.count * 2));
    gl.drawArrays(gl.POINTS, 0, pts.count);
  }
}

function ptsTick(now) {
  if (!ptsWanted()) {
    pts.raf = 0;
    ptsDraw(now);          // one frozen frame, then fully stopped
    return;
  }
  ptsDraw(now);
  pts.raf = requestAnimationFrame(ptsTick);
}

function ptsApply() {
  if (!pts.gl && !pts.failed) ptsInit();
  if (!pts.gl) return;
  const d = Math.max(0, Math.min(100, appearance.particleDensity || 0));
  pts.count = Math.round(PTS_MAX * d / 100);
  if (ptsWanted()) {
    if (!pts.raf) pts.raf = requestAnimationFrame(ptsTick);
  } else if (!pts.raf) {
    ptsDraw(performance.now());
  }
}

window.addEventListener("pointermove", e => {
  pts.mx = e.clientX;
  pts.my = e.clientY;
}, { passive: true });
window.addEventListener("blur", () => { pts.focused = false; ptsApply(); });
window.addEventListener("focus", () => { pts.focused = true; ptsApply(); });
document.addEventListener("visibilitychange", () => ptsApply());
window.addEventListener("resize", () => {
  ptsResize();
  if (!pts.raf) ptsDraw(performance.now());
});

/* ------------------------------------------------------------------ */
/* the scene echo: while a job runs, its frames ghost through the   */
/* thinned veil on this canvas. It is content, not decoration --    */
/* the painter below runs only in the working state (plus one       */
/* still paint per fresh frame) and never at idle. The ambient      */
/* picture needs no painter at all: the video element plays it.     */
/* ------------------------------------------------------------------ */
const echoCanvas = document.getElementById("echo");
const echoCtx = echoCanvas.getContext("2d");
let echoSrc = null;  // image/canvas/video shown behind the veil
let echoRaf = 0;

function sizeEchoCanvas() {
  echoCanvas.width = window.innerWidth;
  echoCanvas.height = window.innerHeight;
  paintEcho();
}

function paintEcho() {
  const W = echoCanvas.width, H = echoCanvas.height;
  if (!W || !H) return;
  echoCtx.clearRect(0, 0, W, H);
  if (!echoSrc) return;
  const iw = echoSrc.videoWidth || echoSrc.width;
  const ih = echoSrc.videoHeight || echoSrc.height;
  if (!iw || !ih) return;
  const scale = Math.max(W / iw, H / ih);
  const w = iw * scale, h = ih * scale;
  echoCtx.drawImage(echoSrc, (W - w) / 2, (H - h) / 2, w, h);
}

function startEchoPainter() {
  if (echoRaf) return;
  const step = () => {
    if (!document.body.classList.contains("working")) {
      echoRaf = 0;
      return;
    }
    paintEcho();
    echoRaf = requestAnimationFrame(step);
  };
  echoRaf = requestAnimationFrame(step);
}
window.addEventListener("resize", sizeEchoCanvas);

/* working state: the veil thins, the scene takes the wall */
let holdTimer = null;
function setWorking(on) {
  clearTimeout(holdTimer);
  if (on) {
    document.body.classList.add("working");
    startEchoPainter();
  }
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
/* The titlebar is the window's drag region, and it carries working
   buttons (the view tabs, the caption trio). pywebview starts a
   window move whenever a mousedown bubbles up from inside a drag
   region, so a press that begins on one of those buttons would
   drag the window instead of pressing it. Swallow the bubble for
   interactive targets only: the press still lands and clicks, the
   window stays put. Presses on the strip's empty glass fall
   through and drag as they should. */
$("titlebar").addEventListener("mousedown", (e) => {
  if (e.target.closest("button, a, input, select, textarea, [contenteditable]")) {
    e.stopPropagation();
  }
});
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

/* settings page → 显示： the baked picture's playback controls.
   Sliders apply live ("input") and persist on release ("change");
   the switch and the theme select apply + persist at once. There
   is nothing structural to rebuild any more -- the picture is a
   file and the veils are CSS variables. */
const AP_RANGES = [
  // [element id, appearance key, value format]
  ["ap-brightness", "brightness", v => Math.round(v) + "%"],
  ["ap-particle-density", "particleDensity", v => Math.round(v) + "%"],
  ["ap-veil", "veil", v => v.toFixed(2)],
  ["ap-work-veil", "workVeil", v => v.toFixed(2)],
  ["ap-set-veil", "setVeil", v => v.toFixed(2)],
];
function syncAppearanceControls() {
  for (const [id, key, fmt] of AP_RANGES) {
    $(id).value = appearance[key];
    $(id + "-val").textContent = fmt(appearance[key]);
  }
  $("ap-motion-on").checked = !!appearance.motionOn;
  $("ap-theme").value = appearance.theme;
}
for (const [id, key, fmt] of AP_RANGES) {
  const el = $(id);
  el.addEventListener("input", () => {
    appearance[key] = parseFloat(el.value);
    $(id + "-val").textContent = fmt(appearance[key]);
    applyAppearance();
  });
  el.addEventListener("change", saveAppearance);
}
$("ap-motion-on").addEventListener("change", e => {
  appearance.motionOn = e.target.checked;
  applyAppearance();
  saveAppearance();
});
$("ap-theme").addEventListener("change", e => {
  appearance.theme = e.target.value;
  applyAppearance();
  saveAppearance();
});
$("ap-reset").addEventListener("click", () => {
  appearance = { ...APPEARANCE_DEFAULTS };
  syncAppearanceControls();
  applyAppearance();
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
  paintEcho();
  document.querySelector(".pv-hint").style.display = "none";
}
function drawPreview(dataUrl) {
  const img = new Image();
  img.onload = () => { drawPreviewSource(img); echoSrc = img; paintEcho(); };
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

/* CLI activity toasts: one silent line per finished CLI command,
   down the right side of the sheet. The bridge drains the CLI's
   event file (snapshot catch-up + poll), this side only displays:
   at most three live, the rest queue in order, each lives ~6 s.
   No focus, no sound, no clicks -- an eyeball check, nothing more. */
const cliToastQueue = [];
let cliToastLive = 0;
function queueCliToasts(events) {
  (events || []).forEach(ev => {
    if (ev && ev.summary) cliToastQueue.push(ev);
  });
  pumpCliToasts();
}
function pumpCliToasts() {
  const host = $("cli-toasts");
  if (!host) return;
  while (cliToastLive < 3 && cliToastQueue.length) {
    const ev = cliToastQueue.shift();
    cliToastLive++;
    const el = document.createElement("div");
    el.className = "cli-toast" + (ev.ok ? "" : " bad");
    const dot = document.createElement("span");
    dot.className = "cli-dot";
    const tag = document.createElement("span");
    tag.className = "cli-tag";
    tag.textContent = "CLI";
    const text = document.createElement("span");
    text.className = "cli-text";
    text.textContent = ev.summary;
    el.append(dot, tag, text);
    host.appendChild(el);
    requestAnimationFrame(() => el.classList.add("show"));
    setTimeout(() => {
      el.classList.remove("show");
      setTimeout(() => {
        el.remove();
        cliToastLive--;
        pumpCliToasts();
      }, 380);
    }, 6000);
  }
}

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
    /* whitelist merge: a settings file from the live-rendered
       era carries dead keys (frostBlur, grafCount, ...) -- they
       are dropped here and never written back */
    for (const k of Object.keys(APPEARANCE_DEFAULTS))
      if (k in snap.settings.appearance)
        appearance[k] = snap.settings.appearance[k];
    syncAppearanceControls();
    applyAppearance();
  }
  (snap.log || []).forEach(l => log(l));
  queueCliToasts(snap.cliEvents);
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
    if (out.cliEvents) queueCliToasts(out.cliEvents);
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
sizeEchoCanvas();
applyAppearance();
syncAppearanceControls();
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
