import {
  validateRecording,
  validateScene,
  unpackScene,
  sampleEntity,
  toPLY,
  clamp,
} from "./core.mjs";
import { createDemo, BOXES } from "./demo.mjs";
import { Viewport } from "./renderer.mjs";
import { SoftwareViewport } from "./software-renderer.mjs";

const $ = (id) => document.getElementById(id),
  text = (id, v) => {
    $(id).textContent = v;
  };
const fmt = (t) =>
  `${String(Math.floor(t / 60)).padStart(2, "0")}:${(t % 60).toFixed(2).padStart(5, "0")}`;
const number = (n) => (n == null ? "—" : n.toLocaleString("en-US"));
let recording,
  payload,
  mesh,
  metrics,
  benchmark,
  view,
  worker,
  playing = false,
  time = 0,
  speed = 1;
let demoMode = true,
  nativeScene = false,
  revision = 0,
  buildId = 0,
  nextJob = 0,
  lastFrame = -1,
  lastTick = 0,
  lastUi = 0,
  lastRender = 0,
  debounce,
  toastTimer,
  cumulativeCount = -1;
const jobs = new Map();

function toast(message, error = false) {
  text("toast", message);
  $("toast").className = `toast visible${error ? " error" : ""}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(
    () => $("toast").classList.remove("visible"),
    error ? 8000 : 4000,
  );
}
function kernel(data) {
  if (!worker) {
    worker = new Worker(new URL("./worker.mjs", import.meta.url), {
      type: "module",
    });
    worker.onmessage = ({ data: r }) => {
      const j = jobs.get(r.id);
      if (j) {
        clearTimeout(j.timeout);
        jobs.delete(r.id);
        r.error ? j.reject(new Error(r.error)) : j.resolve(r);
      }
    };
    worker.onerror = () => cancelJobs("重建任务失败，请重新导入。");
  }
  const id = ++nextJob;
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(
      () => cancelJobs("重建超过 30 秒，请减少输入规模。"),
      30000,
    );
    jobs.set(id, { resolve, reject, timeout });
    worker.postMessage({ ...data, id });
  });
}
function cancelJobs(message) {
  worker?.terminate();
  worker = null;
  for (const j of jobs.values()) {
    clearTimeout(j.timeout);
    j.reject(new Error(message));
  }
  jobs.clear();
}
function options() {
  return {
    stride: +$("stride").value,
    noise: +$("noise").value / 1000,
    drift: +$("drift").value / 100,
    voxel: +$("voxel").value / 100,
    until: $("cumulative").checked ? time : Infinity,
  };
}
function labels() {
  text("strideOutput", `1 / ${$("stride").value}`);
  text("noiseOutput", `±${$("noise").value} mm`);
  text("driftOutput", `${$("drift").value} cm`);
  text("voxelOutput", `${(+$("voxel").value).toFixed(1)} cm`);
}
async function rebuild() {
  if (!recording || nativeScene) return;
  const id = ++buildId,
    rev = revision;
  $("busy").hidden = false;
  try {
    const r = await kernel({
      recording,
      options: options(),
      evaluate: demoMode,
    });
    if (id !== buildId || rev !== revision) return;
    mesh = r.mesh;
    metrics = r.metrics;
    view?.setMesh(mesh);
    text(
      "meshCount",
      `${number(mesh.vertices.length)} 顶点 / ${number(mesh.faces.length)} 三角形`,
    );
    text("usedFrames", `${mesh.usedFrames} / ${recording.frames.length}`);
    text("rawPoints", number(mesh.rawPoints));
    text("elapsed", `${Math.round(r.elapsed)} ms`);
    text(
      "rmseValue",
      metrics?.rmse == null ? "—" : (metrics.rmse * 100).toFixed(2),
    );
    text("coverageValue", metrics ? (metrics.coverage * 100).toFixed(1) : "—");
    $("coverageBar").style.width = `${(metrics?.coverage || 0) * 100}%`;
    text(
      "metricNote",
      demoMode
        ? "合成几何检验，不代表真实设备精度。"
        : "导入数据没有独立真值，不显示几何精度评分。",
    );
    $("pointCard").hidden = true;
  } catch (e) {
    if (rev === revision) toast(e.message, true);
  } finally {
    if (id === buildId) $("busy").hidden = true;
  }
}
function schedule() {
  labels();
  clearTimeout(debounce);
  debounce = setTimeout(rebuild, 170);
}
function setPlaying(value) {
  playing = value;
  $("play").setAttribute("aria-label", value ? "暂停" : "播放");
  $("play")
    .querySelector("use")
    .setAttribute("href", value ? "#i-pause" : "#i-play");
  lastTick = performance.now();
}
function drawDepth(index) {
  const f = recording.frames[index],
    k = recording.intrinsics,
    c = $("depth");
  c.width = k.width;
  c.height = k.height;
  const ctx = c.getContext("2d"),
    im = ctx.createImageData(k.width, k.height);
  let valid = 0;
  f.depth.forEach((z, i) => {
    const at = i * 4;
    if (z > 0 && !f.excluded?.[i]) {
      valid++;
      const t = clamp(z / 6, 0, 1);
      im.data[at] = Math.round(170 - 135 * t);
      im.data[at + 1] = Math.round(205 - 150 * t);
      im.data[at + 2] = Math.round(189 - 118 * t);
    } else {
      im.data[at] = 8;
      im.data[at + 1] = 17;
      im.data[at + 2] = 23;
    }
    im.data[at + 3] = 255;
  });
  ctx.putImageData(im, 0, 0);
  text("frameLabel", `FRAME ${String(index + 1).padStart(3, "0")}`);
  text("depthTime", fmt(f.t));
  text("readingTime", `${f.t.toFixed(3)} s`);
  text("readingPixels", `${number(valid)} / ${number(f.depth.length)}`);
  text(
    "readingPosition",
    f.cameraToWorld
      .slice(12, 15)
      .map((v) => v.toFixed(2))
      .join(" / "),
  );
  if (view?.cameraOverride) {
    view.cameraOverride = f.cameraToWorld;
    view.dirty = true;
  }
}
function setTime(value) {
  time = clamp(value, 0, recording.duration);
  $("timeSlider").value = time;
  text("currentTime", fmt(time));
  $("playhead").style.left = `${(time / recording.duration) * 100}%`;
  let index = 0;
  for (let i = 0; i < recording.frames.length; i++) {
    if (recording.frames[i].t <= time) index = i;
    else break;
  }
  if (index !== lastFrame) {
    lastFrame = index;
    drawDepth(index);
  }
  if ($("cumulative").checked && !nativeScene) {
    const count = recording.frames.filter(
      (f, i) => i % options().stride === 0 && f.t <= time,
    ).length;
    if (count !== cumulativeCount) {
      cumulativeCount = count;
      rebuild();
    }
  }
}
function posePanel(poses) {
  const first = poses.find((p) => p.pose),
    label = $("entityLabel");
  $("gapBanner").hidden = true;
  if (!first) {
    label.hidden = true;
    text("poseTag", "无观测");
    text(
      "poseState",
      recording.entities?.length
        ? "此刻没有可显示的动作证据"
        : "这份记录没有人物骨架",
    );
    text("poseDescription", "静态空间可以独立重建。人物识别与跟踪是额外输入。");
    return;
  }
  const { entity, pose } = first,
    p = view?.project(pose.joints[3] || pose.joints[0]);
  label.hidden = !p || !view.layers.skeleton;
  if (p) {
    label.style.left = `${p[0] + 12}px`;
    label.style.top = `${p[1] - 26}px`;
  }
  label.querySelector("span:nth-child(2)").textContent = entity.id;
  label.classList.toggle("gap", pose.gap);
  text(
    "entityStatus",
    pose.gap ? "假设插值" : demoMode ? "合成骨架" : "输入轨迹",
  );
  $("gapBanner").hidden = !pose.gap;
  if (pose.gap)
    text(
      "gapMessage",
      `${fmt(pose.from)}–${fmt(pose.to)} 缺失；当前为直线插值。`,
    );
  text(
    "poseTag",
    pose.gap ? "假设插值" : pose.type === "sample" ? "关键帧" : "帧间插值",
  );
  text(
    "poseState",
    `${entity.id} · ${pose.gap ? "动作证据中断" : "世界坐标中的关节"}`,
  );
  text(
    "poseDescription",
    pose.gap
      ? "两端位置已知，中间动作未知。琥珀色插值不能当作历史事实。"
      : `${recording.source === "synthetic" ? "程序生成" : "文件提供"}的骨架；${pose.type === "sample" ? "当前时刻有采样。" : "正在相邻采样之间插值。"}`,
  );
}
function timeline() {
  $("timeSlider").max = recording.duration;
  text("totalTime", `/ ${fmt(recording.duration)}`);
  $("timeTicks").replaceChildren(
    ...Array.from({ length: 9 }, (_, i) => {
      const n = document.createElement("span");
      n.textContent = fmt((recording.duration * i) / 8).slice(0, -3);
      return n;
    }),
  );
  const track = $("dataTrack");
  track.replaceChildren();
  const entity = recording.entities?.[0],
    samples = entity?.samples || recording.frames;
  samples.forEach((s, i) => {
    const n = document.createElement("i");
    n.className = "observed";
    n.style.left = `${(s.t / recording.duration) * 100}%`;
    track.append(n);
    if (entity && i && s.t - samples[i - 1].t > 0.65) {
      const g = document.createElement("i");
      g.className = "missing";
      g.style.left = `${(samples[i - 1].t / recording.duration) * 100}%`;
      g.style.width = `${((s.t - samples[i - 1].t) / recording.duration) * 100}%`;
      track.append(g);
    }
  });
  $("events").hidden = !demoMode;
}
function fit() {
  if (!view || !mesh?.vertices.length) return;
  const lo = [Infinity, Infinity, Infinity],
    hi = [-Infinity, -Infinity, -Infinity];
  for (const { p } of mesh.vertices)
    for (let i = 0; i < 3; i++) {
      lo[i] = Math.min(lo[i], p[i]);
      hi[i] = Math.max(hi[i], p[i]);
    }
  view.target = lo.map((v, i) => (v + hi[i]) / 2);
  view.distance = clamp(
    Math.hypot(...hi.map((v, i) => v - lo[i])) * 1.35,
    2,
    28,
  );
  view.yaw = 0.68;
  view.pitch = 0.5;
  view.cameraOverride = null;
  view.dirty = true;
}
async function load(data, isDemo = false) {
  const scene = data.version === "echo-scene-1" ? validateScene(data) : null,
    next = scene ? scene.recording : validateRecording(data);
  ++revision;
  ++buildId;
  clearTimeout(debounce);
  cancelJobs("记录已切换");
  setPlaying(false);
  recording = next;
  payload = data;
  demoMode = isDemo;
  nativeScene = !!scene;
  lastFrame = -1;
  mesh = metrics = benchmark = null;
  cumulativeCount = -1;
  $("benchmarkSummary").hidden = true;
  $("pointCard").hidden = true;
  $("cumulative").checked = false;
  $("cumulative").disabled = nativeScene;
  for (const id of ["stride", "noise", "drift", "voxel", "resetExperiment"])
    $(id).disabled = nativeScene;
  $("runBenchmark").disabled = !demoMode;
  $("referenceLayer").hidden = !demoMode;
  document.querySelector('[data-layer="reference"]').checked = false;
  if (view) {
    view.layers.reference = false;
    view.setRecording(recording);
    view.reset();
  }
  text(
    "sourceBadge",
    isDemo
      ? "◇ 合成演示数据"
      : `${recording.source === "synthetic" ? "◇ 导入合成记录" : "◇ 导入实拍记录"}${nativeScene ? " · TSDF" : ""}`,
  );
  text("sessionName", recording.name || "导入记录");
  text(
    "sessionMeta",
    `${recording.duration.toFixed(1)} 秒 / ${recording.frames.length} 帧${nativeScene ? "预览" : "深度"}`,
  );
  text(
    "viewName",
    isDemo
      ? "LIVING ROOM"
      : nativeScene
        ? "TSDF RECONSTRUCTION"
        : "IMPORTED RECORDING",
  );
  text("meshScope", nativeScene ? "原生重建结果" : "完整记录融合");
  text(
    "boundaryText",
    isDemo
      ? "房间与骨架均为合成数据。"
      : nativeScene
        ? "本地 TSDF 网格；深度帧是降采样预览。真实精度需要独立参考。"
        : "来源由输入文件声明，不构成设备或身份验证。",
  );
  text(
    "readingSource",
    recording.source === "synthetic" ? "合成传感器" : "导入深度文件",
  );
  text(
    "stageNote",
    nativeScene
      ? "原生 TSDF 表面 · 点选查看近邻体素观测数。"
      : "深度观测形成表面，未观测区域保留空洞。",
  );
  text(
    "footerStatus",
    nativeScene
      ? "原生 CPU TSDF · 观测支持过滤"
      : "深度反投影 · 局部三角化 · 空间合并",
  );
  text(
    "metricNote",
    nativeScene ? "已载入原生 TSDF 网格；参数需在命令行修改。" : "",
  );
  timeline();
  setTime(isDemo ? 4.8 : 0);
  if (scene) {
    mesh = unpackScene(scene);
    view?.setMesh(mesh);
    fit();
    text(
      "meshCount",
      `${number(mesh.vertices.length)} 顶点 / ${number(mesh.faces.length)} 三角形`,
    );
    for (const id of [
      "rmseValue",
      "coverageValue",
      "usedFrames",
      "rawPoints",
      "elapsed",
    ])
      text(id, "—");
    $("coverageBar").style.width = "0%";
    $("busy").hidden = true;
  } else {
    await rebuild();
    if (!isDemo) fit();
  }
}
function mode(next) {
  for (const id of ["replayControls", "evidencePanel"])
    $(id).hidden = next !== "replay";
  for (const id of ["experimentControls", "metricsPanel"])
    $(id).hidden = next !== "experiment";
  for (const [id, name] of [
    ["replayTab", "replay"],
    ["experimentTab", "experiment"],
  ]) {
    $(id).classList.toggle("active", next === name);
    $(id).setAttribute("aria-selected", String(next === name));
  }
  text(
    "workspaceTitle",
    next === "replay" ? "让过去，重新成为空间。" : "少一点信息，会失去什么？",
  );
  text(
    "workspaceSubtitle",
    next === "replay"
      ? "拖动时间，走近现场。让每一片几何都有迹可循。"
      : "改变输入，重新计算。用几何误差检查重建。",
  );
  view?.resize();
}
function inspect(v) {
  $("pointCard").hidden = !v;
  if (!v) return;
  mode("replay");
  text("pointPosition", v.p.map((n) => n.toFixed(3)).join(" / ") + " m");
  text(
    "pointEvidence",
    nativeScene
      ? `近邻体素被观测 ${v.supportCount} 次；支持时间 ${v.first.toFixed(2)}–${v.last.toFixed(2)} 秒。计数不等于置信概率。`
      : `${v.frames.length} 帧、${v.n} 个深度采样合并；支持时间 ${v.first.toFixed(2)}–${v.last.toFixed(2)} 秒。`,
  );
}
function download(content, name, type = "application/json") {
  const url = URL.createObjectURL(new Blob([content], { type })),
    a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
  document.querySelector(".export-menu").open = false;
}
function report() {
  return {
    version: "echo-browser-report-1",
    source: recording.source,
    name: recording.name,
    algorithm: nativeScene
      ? "projective-tsdf-v1"
      : "depth-triangulation-voxel-average-v1",
    parameters: nativeScene
      ? payload.provenance?.config
      : {
          ...options(),
          until: Number.isFinite(options().until) ? options().until : null,
        },
    mesh: mesh && {
      vertices: mesh.vertices.length,
      triangles: mesh.faces.length,
    },
    synthetic_metrics: metrics,
    benchmark,
    limitations:
      "Known poses; skeletons are input. Imported data has no independent accuracy metric.",
  };
}
async function runBenchmark() {
  if (!demoMode) return;
  $("runBenchmark").disabled = true;
  const rev = revision,
    rows = [];
  try {
    for (const c of [
      { name: "理想输入", options: {} },
      { name: "每 12 帧留 1 帧", options: { stride: 12 } },
      { name: "测距噪声 ±40 mm", options: { noise: 0.04 } },
      { name: "末帧位置偏移 20 cm", options: { drift: 0.2 } },
    ]) {
      text("runBenchmark", `计算对照 ${rows.length + 1} / 4…`);
      const r = await kernel({
        recording,
        options: { stride: 1, noise: 0, drift: 0, voxel: 0.045, ...c.options },
        evaluate: true,
        metricsOnly: true,
      });
      rows.push({
        condition: c.name,
        options: c.options,
        frames: r.counts.usedFrames,
        ...r.metrics,
      });
    }
    if (rev !== revision) return;
    benchmark = rows;
    $("benchmarkRows").replaceChildren(
      ...rows.map((r) => {
        const tr = document.createElement("tr");
        for (const v of [
          r.condition,
          r.frames,
          `${(r.rmse * 100).toFixed(2)} cm`,
          `${(r.coverage * 100).toFixed(1)}%`,
        ]) {
          const td = document.createElement("td");
          td.textContent = v;
          tr.append(td);
        }
        return tr;
      }),
    );
    text(
      "benchmarkConclusion",
      "数值来自这次实际重算。低表面误差与高覆盖率须同时检查。",
    );
    $("benchmarkSummary").hidden = false;
    text("benchmarkMessage", "四组条件已完成，可导出。");
    $("benchmarkDialog").showModal();
  } catch (e) {
    if (rev === revision) toast(e.message, true);
  } finally {
    $("runBenchmark").disabled = !demoMode;
    text("runBenchmark", "运行四组对照实验 →");
  }
}
async function importFile(file) {
  if (!file) return;
  try {
    if (file.size > 40 * 1024 * 1024)
      throw new Error("文件超过 40 MiB，请先降采样。");
    await load(JSON.parse(await file.text()));
    $("importDialog").close();
    mode("replay");
    toast("已载入，文件仅在当前浏览器处理。");
  } catch (e) {
    toast(e instanceof SyntaxError ? "文件不是有效 JSON。" : e.message, true);
  } finally {
    $("fileInput").value = "";
  }
}

$("replayTab").onclick = () => mode("replay");
$("experimentTab").onclick = () => mode("experiment");
for (const id of ["guideButton", "readMethod", "limitationsButton"])
  $(id).onclick = () => $("guideDialog").showModal();
for (const id of ["importButton", "inspectHint", "sessionInfo"])
  $(id).onclick = () => $("importDialog").showModal();
$("guideImport").onclick = () => {
  $("guideDialog").close();
  $("importDialog").showModal();
};
for (const b of document.querySelectorAll("[data-close]"))
  b.onclick = () => $(b.dataset.close).close();
for (const d of document.querySelectorAll("dialog"))
  d.addEventListener("click", (e) => {
    const r = d.getBoundingClientRect();
    if (
      e.target === d &&
      (e.clientX < r.left ||
        e.clientX > r.right ||
        e.clientY < r.top ||
        e.clientY > r.bottom)
    )
      d.close();
  });
for (const el of document.querySelectorAll("[data-layer]"))
  el.onchange = () => {
    if (view) {
      view.layers[el.dataset.layer] = el.checked;
      view.dirty = true;
    }
  };
for (const b of document.querySelectorAll("[data-mode]"))
  b.onclick = () => {
    if (view) {
      view.mode = b.dataset.mode;
      view.dirty = true;
    }
    for (const a of document.querySelectorAll("[data-mode]"))
      a.classList.toggle("active", a === b);
  };
for (const id of ["stride", "noise", "drift", "voxel"])
  $(id).oninput = schedule;
$("resetExperiment").onclick = () => {
  $("stride").value = 2;
  $("noise").value = 4;
  $("drift").value = 0;
  $("voxel").value = 4.5;
  schedule();
};
$("runBenchmark").onclick = runBenchmark;
$("openBenchmark").onclick = () => $("benchmarkDialog").showModal();
$("benchmarkExport").onclick = $("exportReport").onclick = () =>
  download(JSON.stringify(report(), null, 2), "echo-report.json");
$("exportMesh").onclick = () => {
  if (mesh?.vertices.length)
    download(
      toPLY(mesh),
      "echo-reconstruction.ply",
      "application/octet-stream",
    );
  else toast("没有可导出的网格。", true);
};
$("exportRecord").onclick = () =>
  download(
    JSON.stringify(payload),
    nativeScene ? "scene.echo.json" : "recording.echo.json",
  );
$("exportSource").onclick = () => {
  const a = document.createElement("a");
  a.href = "./echo4d-source.zip";
  a.download = "echo4d-source.zip";
  a.click();
  document.querySelector(".export-menu").open = false;
};
$("restart").onclick = () => {
  setPlaying(false);
  setTime(0);
};
$("play").onclick = () => {
  if (time >= recording.duration) setTime(0);
  setPlaying(!playing);
};
$("speed").onclick = () => {
  speed = speed === 1 ? 0.5 : speed === 0.5 ? 2 : 1;
  text("speed", `${speed}×`);
};
$("timeSlider").oninput = (e) => {
  setPlaying(false);
  setTime(+e.target.value);
};
for (const b of document.querySelectorAll("[data-time]"))
  b.onclick = () => {
    setPlaying(false);
    setTime(+b.dataset.time);
  };
$("allowGaps").onchange = () => {
  if (view) view.dirty = true;
};
$("cumulative").onchange = () => {
  cumulativeCount = -1;
  text(
    "meshScope",
    $("cumulative").checked ? "仅此刻之前的帧" : "完整记录融合",
  );
  rebuild();
};
function preset(id, fn) {
  $(id).onclick = () => {
    if (!view) return;
    fn();
    view.dirty = true;
    for (const b of document.querySelectorAll(".view-presets button"))
      b.classList.toggle("active", b.id === id);
  };
}
preset("orbitView", () => {
  view.cameraOverride = null;
  view.pitch = 0.5;
});
preset("topView", () => {
  view.cameraOverride = null;
  view.pitch = 1.55;
});
preset("sensorView", () => {
  view.cameraOverride = recording.frames[Math.max(lastFrame, 0)].cameraToWorld;
});
preset("resetView", () => {
  demoMode ? view.reset() : fit();
});
$("clearPoint").onclick = () => {
  if (view) {
    view.selected = null;
    view.dirty = true;
  }
  inspect(null);
};
$("chooseFile").onclick = () => $("fileInput").click();
$("fileInput").onchange = (e) => importFile(e.target.files[0]);
const zone = $("dropZone");
zone.ondragover = (e) => {
  e.preventDefault();
  zone.classList.add("dragover");
};
zone.ondragleave = () => zone.classList.remove("dragover");
zone.ondrop = (e) => {
  e.preventDefault();
  zone.classList.remove("dragover");
  importFile(e.dataTransfer.files[0]);
};
$("restoreDemo").onclick = async () => {
  await load(createDemo(), true);
  $("importDialog").close();
  toast("已恢复合成演示。");
};
$("downloadTemplate").onclick = () =>
  download(
    JSON.stringify(
      {
        version: "echo-1",
        name: "最小平面样本",
        source: "synthetic",
        units: "meters",
        duration: 1,
        intrinsics: { width: 2, height: 2, fx: 1, fy: 1, cx: 0.5, cy: 0.5 },
        frames: [
          {
            t: 0,
            cameraToWorld: [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 1.5, 0, 1],
            depth: [2, 2, 2, 2],
          },
        ],
      },
      null,
      2,
    ),
    "echo-format-sample.json",
  );
if ($("loadRealPreview"))
  $("loadRealPreview").onclick = async () => {
    const b = $("loadRealPreview");
    b.disabled = true;
    try {
      const r = await fetch("./samples/tum-xyz.echo.json");
      if (!r.ok) throw new Error("样例下载失败。");
      await load(await r.json());
      $("importDialog").close();
      mode("replay");
      toast("TUM 实拍重建已载入。");
    } catch (e) {
      toast(e.message, true);
    } finally {
      b.disabled = false;
    }
  };
document.addEventListener("keydown", (e) => {
  if (
    e.code === "Space" &&
    !["INPUT", "BUTTON", "SUMMARY", "TEXTAREA"].includes(
      document.activeElement.tagName,
    ) &&
    !document.querySelector("dialog[open]")
  ) {
    e.preventDefault();
    $("play").click();
  }
});
try {
  view = new Viewport($("scene"), inspect);
} catch (e) {
  try {
    view = new SoftwareViewport($("scene"), inspect);
  } catch (error) {
    toast(error.message, true);
  }
}
view?.setReference(BOXES);
labels();
await load(createDemo(), true);
function animate(now) {
  const dt = Math.min((now - lastTick) / 1000, 0.1);
  lastTick = now;
  if (playing && !document.hidden) {
    time = Math.min(recording.duration, time + dt * speed);
    if (now - lastUi > 70) {
      setTime(time);
      lastUi = now;
    }
    if (time >= recording.duration) {
      setPlaying(false);
      setTime(time);
    }
  }
  if (
    view &&
    !document.hidden &&
    (view.dirty || playing || now - lastRender > 250)
  ) {
    const poses = (recording.entities || []).map((entity) => ({
      entity,
      pose: sampleEntity(entity, time, $("allowGaps").checked),
    }));
    view.render(poses);
    posePanel(poses);
    lastRender = now;
  }
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);
