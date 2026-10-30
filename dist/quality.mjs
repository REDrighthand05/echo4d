// ECHO quality evidence intake: local image preview and hash inventory, never automated P0 approval.
import { parseIntake, matchesExpected } from "./quality-intake.mjs";
const slots = [];
let importGeneration = 0;
const host = (id) => document.getElementById(id);
function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}
function addSlot(container, id, title, subtitle) {
  const section = element("div", "slot");
  const top = element("div", "slot-top");
  top.append(element("strong", "", title), element("span", "", subtitle));
  const picture = element("div", "image-box");
  const placeholder = element("div", "empty", "尚未载入真实证据");
  picture.append(placeholder);
  const label = element("label", "file-button", "选择本地图片");
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "image/jpeg,image/png,image/webp";
  input.setAttribute("aria-label", title + "文件");
  label.append(input);
  const meta = element("div", "file-meta", "文件仅在浏览器内预览，不会上传。");
  section.append(top, picture, label, meta);
  container.append(section);
  const slot = { id, title, subtitle, input, picture, meta, file: null, url: null, sha256: null, size: null, expectedSha: null, expectedName: null, generation: 0 };
  slots.push(slot);
  input.addEventListener("change", () => readImage(slot));
  return slot;
}
for (let i = 1; i <= 3; i++) {
  const pair = element("article", "pair");
  const header = element("div", "pair-header");
  header.append(element("strong", "", "固定视角 0" + i), element("span", "", "两图必须来自相同相机位姿与 Mesh"));
  const images = element("div", "pair-images");
  addSlot(images, "rgb-" + i, "高清源 RGB 0" + i, "原始分辨率，不是预览");
  addSlot(images, "projection-" + i, "对应 Mesh 投影 0" + i, "同位姿 · 同一网格");
  pair.append(header, images);
  host("pairs").append(pair);
}
for (let i = 1; i <= 2; i++) {
  addSlot(host("orbits"), "orbit-" + i, "第三人称视角 0" + i, "自由观察 · 同一网格");
}
function hex(bytes) {
  return [...bytes].map((n) => n.toString(16).padStart(2, "0")).join("");
}
function checkDimensions(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve({ width: image.naturalWidth, height: image.naturalHeight });
    image.onerror = () => reject(new Error("无法解码图片"));
    image.src = url;
  });
}
function clearSlot(slot, resetInput = false) {
  slot.generation++;
  if (slot.url) URL.revokeObjectURL(slot.url);
  slot.url = null;
  slot.file = null;
  slot.size = null;
  slot.sha256 = null;
  if (resetInput) slot.input.value = "";
  slot.picture.replaceChildren(element("div", "empty", "尚未载入真实证据"));
  updateSummary();
}
async function readImage(slot) {
  const file = slot.input.files && slot.input.files[0];
  if (!file) return;
  clearSlot(slot);
  const generation = slot.generation;
  if (!["image/png", "image/jpeg", "image/webp"].includes(file.type) || file.size > 50 * 1024 * 1024) {
    slot.meta.textContent = "只接受 PNG/JPEG/WebP，单张上限 50 MB。";
    slot.input.value = "";
    return;
  }
  const nextUrl = URL.createObjectURL(file);
  slot.meta.textContent = "正在检查图像尺寸与 SHA-256…";
  try {
    const [size, digest] = await Promise.all([
      checkDimensions(nextUrl),
      crypto.subtle.digest("SHA-256", await file.arrayBuffer())
    ]);
    if (slot.generation !== generation || slot.input.files[0] !== file) {
      URL.revokeObjectURL(nextUrl);
      return;
    }
    const sha = hex(new Uint8Array(digest));
    if (!matchesExpected(sha, slot.expectedSha)) {
      throw new Error("SHA-256 与已导入清单不符，拒绝将此图计入证据");
    }
    slot.url = nextUrl;
    slot.file = file;
    slot.size = size;
    slot.sha256 = sha;
    const img = document.createElement("img");
    img.src = nextUrl;
    img.alt = slot.title + "：本地导入的证据图片";
    slot.picture.replaceChildren(img);
    slot.meta.textContent = file.name + " · " + size.width + "×" + size.height + " px · SHA-256 " + slot.sha256.slice(0, 16) + "…" + (slot.expectedSha ? " · 与清单匹配" : "");
    updateSummary();
  } catch (error) {
    URL.revokeObjectURL(nextUrl);
    if (slot.generation !== generation || slot.input.files[0] !== file) return;
    slot.meta.textContent = "图片无法验证：" + error.message;
    slot.input.value = "";
  }
}
function updateSummary() {
  const n = slots.filter((s) => s.sha256).length;
  const pairs = [1, 2, 3].filter((i) => slots.find((s) => s.id === "rgb-" + i).sha256 && slots.find((s) => s.id === "projection-" + i).sha256).length;
  const orbit = slots.filter((s) => s.id.startsWith("orbit-") && s.sha256).length;
  host("pairsCount").textContent = pairs + " / 3 组齐全";
  host("orbitCount").textContent = orbit + " / 2 张";
  host("progressText").textContent = n + " / 8 图像材料";
  host("progressFill").style.width = (n / 8 * 100) + "%";
  const checks = [
    [pairs === 3, "3 对原图与同视角投影文件齐全"],
    [orbit === 2, "2 个自由第三人称文件齐全"],
    [/^[0-9a-f]{64}$/i.test(host("meshSha").value.trim()), "填写真实 Mesh 输出 SHA-256"],
    [false, "原始像素、逐帧位姿、12帧评分及人工签署：需线下独立核实"]
  ];
  host("gateList").replaceChildren(...checks.map(([ok, label]) => {
    const li = element("li", ok ? "good" : "", (ok ? "✓ " : "○ ") + label);
    return li;
  }));
}
host("importManifest").addEventListener("change", async (event) => {
  const generation = ++importGeneration;
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    if (file.size > 1024 * 1024) throw new Error("清单不能超过 1 MiB");
    const intake = parseIntake(JSON.parse(await file.text()));
    if (generation !== importGeneration) return;
    for (const slot of slots) {
      clearSlot(slot, true);
      slot.expectedSha = intake.expected[slot.id].sha256;
      slot.expectedName = intake.expected[slot.id].file_name;
      slot.meta.textContent = slot.expectedSha
        ? "待重新选择原图核对 · 清单记录：" + (slot.expectedName || "未命名") + " · SHA-256 " + slot.expectedSha.slice(0, 16) + "…"
        : "清单未提供此图哈希；须重新选择并人工核实来源。";
    }
    host("sceneId").value = intake.scene_id;
    host("meshSha").value = intake.mesh_sha256;
    host("reviewNotes").value = intake.notes;
    host("importStatus").textContent = "清单已读取；图片没有从清单恢复。请逐张重新选择文件并核对 SHA。";
    updateSummary();
  } catch (error) {
    if (generation !== importGeneration) return;
    host("importStatus").textContent = "导入失败（原有内容未更改）：" + error.message;
  } finally {
    if (generation === importGeneration) event.target.value = "";
  }
});
host("meshSha").addEventListener("input", updateSummary);
updateSummary();
host("exportManifest").addEventListener("click", () => {
  const files = slots.map((s) => ({
    slot: s.id,
    role: s.title,
    file_name: s.file ? s.file.name : null,
    mime: s.file ? s.file.type : null,
    bytes: s.file ? s.file.size : null,
    pixels: s.size,
    sha256: s.sha256,
    expected_sha256: s.expectedSha,
    expected_file_name: s.expectedName,
    source_verified: false,
    actual_camera_pose_verified: false
  }));
  const completeImages = files.every((f) => f.sha256);
  const meshSha = host("meshSha").value.trim().toLowerCase();
  const manifest = {
    format: "echo-visual-review-intake-1",
    scene_id: host("sceneId").value.trim(),
    mesh_sha256: /^[0-9a-f]{64}$/.test(meshSha) ? meshSha : null,
    notes: host("reviewNotes").value,
    images_complete: completeImages,
    visual_recognizable: "not_reviewed",
    depth_screening: "not_assessed_for_this_submission",
    overall_accepted: false,
    requirements: {
      source_licence_and_original_resolution: "not_verified",
      pose_and_same_mesh_alignment: "not_verified",
      frozen_heldout_all_frames: "not_verified",
      independent_human_review: "not_verified"
    },
    files
  };
  const blob = new Blob([JSON.stringify(manifest, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "echo-visual-review-intake.json";
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
window.addEventListener("pagehide", () => slots.forEach((s) => { if (s.url) URL.revokeObjectURL(s.url); }));