// ECHO reconstruction kernel. No renderer or access to scene ground truth here.
export const add = (a, b) => a.map((v, i) => v + b[i]);
export const sub = (a, b) => a.map((v, i) => v - b[i]);
export const mul = (a, s) => a.map((v) => v * s);
export const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
export const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const norm = (a) => mul(a, 1 / (Math.hypot(...a) || 1));
export const lerp = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);
export const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
export function transform(m, p) {
  return [
    m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
    m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
    m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
  ];
}
export function unproject(u, v, z, k, m) {
  return transform(m, [((u - k.cx) * z) / k.fx, ((v - k.cy) * z) / k.fy, z]);
}
export function cameraMatrix(position, target) {
  const f = norm(sub(target, position)),
    r = norm(cross(f, [0, 1, 0])),
    d = cross(f, r);
  return [...r, 0, ...d, 0, ...f, 0, ...position, 1];
}
export function random(seed = 917) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}
const finite = (x) => typeof x === "number" && Number.isFinite(x);
const vec = (x, n) => Array.isArray(x) && x.length === n && x.every(finite);
export function validateRecording(r) {
  const fail = (m) => {
    throw new Error(m);
  };
  if (!r || r.version !== "echo-1") fail("记录版本须为 echo-1。");
  if (r.units !== "meters") fail("长度单位须为 meters（米）。");
  if (!["synthetic", "recorded"].includes(r.source))
    fail("source 须标明 synthetic 或 recorded。");
  const k = r.intrinsics;
  if (
    !k ||
    !Number.isInteger(k.width) ||
    !Number.isInteger(k.height) ||
    k.width < 2 ||
    k.height < 2 ||
    k.width > 640 ||
    k.height > 480
  )
    fail("深度图尺寸应为 2–640 × 2–480。");
  if (
    ![k.fx, k.fy, k.cx, k.cy].every(finite) ||
    k.fx < 1e-3 ||
    k.fy < 1e-3 ||
    k.fx > 1e7 ||
    k.fy > 1e7 ||
    Math.abs(k.cx) > 1e7 ||
    Math.abs(k.cy) > 1e7
  )
    fail("相机内参无效。");
  if (
    !Array.isArray(r.frames) ||
    !r.frames.length ||
    r.frames.length > 300 ||
    r.frames.length * k.width * k.height > 1500000
  )
    fail("最多 300 帧、合计 150 万深度像素。请先降采样。");
  let prev = -Infinity;
  r.frames.forEach((f, i) => {
    if (!f || typeof f !== "object") fail("深度帧须为对象。");
    if (!finite(f.t) || f.t < 0 || f.t <= prev)
      fail(`第 ${i + 1} 帧时间须非负且严格递增。`);
    prev = f.t;
    if (!vec(f.cameraToWorld, 16))
      fail(`第 ${i + 1} 帧须有 16 个有限数构成的 cameraToWorld。`);
    const m = f.cameraToWorld,
      a = m.slice(0, 3),
      b = m.slice(4, 7),
      c = m.slice(8, 11);
    if (
      Math.abs(m[3]) + Math.abs(m[7]) + Math.abs(m[11]) + Math.abs(m[15] - 1) >
        1e-4 ||
      [a, b, c].some((v) => Math.abs(dot(v, v) - 1) > 0.03) ||
      Math.abs(dot(a, b)) + Math.abs(dot(b, c)) + Math.abs(dot(c, a)) > 0.03 ||
      dot(cross(a, b), c) < 0.98
    )
      fail(`第 ${i + 1} 帧变换须为右手刚体变换，按列存储。`);
    if (m.slice(12, 15).some((v) => Math.abs(v) > 10000))
      fail("本原型支持局部坐标，位置绝对值须小于 10 km。");
    if (
      !Array.isArray(f.depth) ||
      f.depth.length !== k.width * k.height ||
      f.depth.some((z) => !finite(z) || z < 0 || z > 100)
    )
      fail(
        `第 ${i + 1} 帧深度须是按行排列的米制数组；0 表示缺测，最大 100 米。`,
      );
    if (
      f.colors !== undefined &&
      (!Array.isArray(f.colors) ||
        f.colors.length !== k.width * k.height * 3 ||
        f.colors.some((c) => !Number.isInteger(c) || c < 0 || c > 255))
    )
      fail(`第 ${i + 1} 帧颜色须为 RGB 0–255 数组。`);
    if (
      f.excluded !== undefined &&
      (!Array.isArray(f.excluded) ||
        f.excluded.length !== k.width * k.height ||
        f.excluded.some((v) => typeof v !== "boolean"))
    )
      fail("排除掩码必须为等长布尔数组。");
  });
  if (
    !finite(r.duration) ||
    r.duration <= 0 ||
    r.duration < prev ||
    r.duration > 86400
  )
    fail("duration 须覆盖所有帧，且不超过 24 小时。");
  if (r.entities !== undefined) {
    if (!Array.isArray(r.entities) || r.entities.length > 8)
      fail("最多支持 8 条人物轨迹。");
    for (const e of r.entities) {
      if (
        !e ||
        typeof e.id !== "string" ||
        e.id.length > 80 ||
        !Array.isArray(e.jointNames) ||
        e.jointNames.length < 2 ||
        e.jointNames.length > 40 ||
        e.jointNames.some((n) => typeof n !== "string") ||
        new Set(e.jointNames).size !== e.jointNames.length
      )
        fail("骨架关节定义无效。");
      if (
        !Array.isArray(e.bones) ||
        e.bones.length > 80 ||
        e.bones.some(
          (b) =>
            !Array.isArray(b) ||
            b.length !== 2 ||
            b.some(
              (j) => !Number.isInteger(j) || j < 0 || j >= e.jointNames.length,
            ),
        )
      )
        fail("骨架连线索引无效。");
      if (!Array.isArray(e.samples) || e.samples.length > 10000)
        fail("每条轨迹最多支持 10000 个采样。");
      let t = -Infinity;
      for (const s of e.samples) {
        if (
          !s ||
          !finite(s.t) ||
          s.t < 0 ||
          s.t <= t ||
          s.t > r.duration ||
          !Array.isArray(s.joints) ||
          s.joints.length !== e.jointNames.length ||
          s.joints.some((p) => !vec(p, 3) || p.some((v) => Math.abs(v) > 10000))
        )
          fail(
            "骨架采样须严格递增、位于记录时间内，并提供每个关节的世界坐标。",
          );
        t = s.t;
      }
    }
  }
  return r;
}
export function reconstruct(recording, options = {}) {
  const o = {
    stride: 1,
    noise: 0,
    drift: 0,
    voxel: 0.045,
    until: Infinity,
    ...options,
  };
  if (
    !Number.isInteger(o.stride) ||
    o.stride < 1 ||
    o.stride > 100 ||
    ![o.noise, o.drift, o.voxel].every(finite) ||
    o.noise < 0 ||
    o.drift < 0 ||
    o.voxel <= 0
  )
    throw new Error("重建参数无效。");
  const k = recording.intrinsics,
    voxels = new Map(),
    vertices = [],
    faces = [],
    dedup = new Set(),
    rng = random(8675309);
  let rawPoints = 0,
    usedFrames = 0,
    rejectedTriangles = 0;
  const frames = recording.frames.filter(
    (f, i) => i % o.stride === 0 && f.t <= o.until,
  );
  for (const f of frames) {
    usedFrames++;
    const map = new Int32Array(k.width * k.height).fill(-1),
      measured = new Float32Array(map.length);
    const m = f.cameraToWorld.slice(),
      progress = f.t / recording.duration;
    m[12] += o.drift * progress;
    m[14] += o.drift * 0.42 * Math.sin(progress * Math.PI);
    for (let v = 0; v < k.height; v++)
      for (let u = 0; u < k.width; u++) {
        const pixel = v * k.width + u,
          depth = f.depth[pixel];
        if (!depth || f.excluded?.[pixel]) continue;
        // Deterministic zero-mean uniform noise; UI specifies its half-width.
        const z = depth + (rng() * 2 - 1) * o.noise;
        if (z <= 0) continue;
        measured[pixel] = z;
        const p = unproject(u, v, z, k, m),
          key = p.map((n) => Math.floor(n / o.voxel)).join(",");
        let ix = voxels.get(key);
        const c = f.colors
          ? f.colors.slice(pixel * 3, pixel * 3 + 3)
          : [93, 149, 151];
        if (ix === undefined) {
          ix = vertices.length;
          voxels.set(key, ix);
          vertices.push({
            p: [0, 0, 0],
            c: [0, 0, 0],
            n: 0,
            frames: [],
            first: f.t,
            last: f.t,
          });
        }
        const a = vertices[ix];
        for (let j = 0; j < 3; j++) {
          a.p[j] += p[j];
          a.c[j] += c[j];
        }
        a.n++;
        a.last = f.t;
        if (a.frames.at(-1) !== usedFrames - 1) a.frames.push(usedFrames - 1);
        map[pixel] = ix;
        rawPoints++;
      }
    function tri(a, b, c) {
      const ia = map[a],
        ib = map[b],
        ic = map[c];
      if (ia < 0 || ib < 0 || ic < 0 || ia === ib || ib === ic || ia === ic)
        return;
      // Never bridge a depth discontinuity; this is local triangulation, not TSDF.
      const z = [measured[a], measured[b], measured[c]],
        min = Math.min(...z);
      if (Math.max(...z) - min > Math.max(0.085, min * 0.055) + o.noise * 2.1) {
        rejectedTriangles++;
        return;
      }
      const key = [ia, ib, ic].sort((x, y) => x - y).join(",");
      if (dedup.has(key)) return;
      dedup.add(key);
      faces.push([ia, ib, ic]);
    }
    for (let y = 0; y < k.height - 1; y++)
      for (let x = 0; x < k.width - 1; x++) {
        const i = y * k.width + x;
        tri(i, i + k.width, i + 1);
        tri(i + 1, i + k.width, i + k.width + 1);
      }
  }
  for (const v of vertices) {
    v.p = v.p.map((p) => p / v.n);
    v.c = v.c.map((c) => Math.round(c / v.n));
  }
  return {
    vertices,
    faces,
    rawPoints,
    usedFrames,
    rejectedTriangles,
    options: o,
  };
}
export function sampleEntity(entity, time, allowGaps = true) {
  const a = entity.samples;
  if (!a.length || time < a[0].t - 1e-6 || time > a.at(-1).t + 1e-6)
    return null;
  let hi = a.findIndex((s) => s.t >= time - 1e-6);
  if (hi < 0) return null;
  if (Math.abs(a[hi].t - time) < 1e-5)
    return { ...a[hi], type: "sample", gap: false, from: a[hi].t, to: a[hi].t };
  if (!hi) return null;
  const l = a[hi - 1],
    r = a[hi],
    gap = r.t - l.t > 0.65;
  if (gap && (!allowGaps || r.t - l.t > 4.1)) return null;
  const t = (time - l.t) / (r.t - l.t);
  return {
    t: time,
    joints: l.joints.map((p, i) => lerp(p, r.joints[i], t)),
    type: "interpolated",
    gap,
    from: l.t,
    to: r.t,
  };
}
export function toPLY(mesh, comment = "ECHO depth reconstruction") {
  const h = [
    "ply",
    "format ascii 1.0",
    `comment ${comment.replace(/[\r\n]/g, " ")}`,
    `element vertex ${mesh.vertices.length}`,
    "property float x",
    "property float y",
    "property float z",
    "property uchar red",
    "property uchar green",
    "property uchar blue",
    `element face ${mesh.faces.length}`,
    "property list uchar int vertex_indices",
    "end_header",
  ];
  return (
    h.join("\n") +
    "\n" +
    mesh.vertices
      .map((v) => [...v.p.map((n) => n.toFixed(5)), ...v.c].join(" "))
      .join("\n") +
    "\n" +
    mesh.faces.map((f) => "3 " + f.join(" ")).join("\n") +
    "\n"
  );
}

export function validateScene(scene) {
  if (
    !scene ||
    scene.version !== "echo-scene-1" ||
    scene.algorithm !== "projective-tsdf-v1"
  )
    throw new Error("场景版本或算法标记无效。");
  validateRecording(scene.recording);
  const m = scene.mesh;
  if (
    !m ||
    !Array.isArray(m.vertices) ||
    !m.vertices.length ||
    m.vertices.length > 180000 ||
    !Array.isArray(m.faces) ||
    !m.faces.length ||
    m.faces.length > 300000
  )
    throw new Error("网格最多 18 万顶点、30 万三角形。");
  const n = m.vertices.length;
  if (
    m.vertices.some((p) => !vec(p, 3) || p.some((x) => Math.abs(x) > 10000)) ||
    m.faces.some(
      (f) =>
        !Array.isArray(f) ||
        f.length !== 3 ||
        f.some((i) => !Number.isInteger(i) || i < 0 || i >= n),
    )
  )
    throw new Error("网格坐标或面索引无效。");
  for (const key of ["colors", "observations", "first", "last"])
    if (!Array.isArray(m[key]) || m[key].length !== n)
      throw new Error(`网格 ${key} 长度不符。`);
  if (
    m.colors.some(
      (c) =>
        !vec(c, 3) || c.some((x) => !Number.isInteger(x) || x < 0 || x > 255),
    ) ||
    m.observations.some((x) => !Number.isInteger(x) || x < 1 || x > 100000) ||
    m.first.some(
      (t, i) =>
        !finite(t) ||
        t < 0 ||
        !finite(m.last[i]) ||
        m.last[i] < t ||
        m.last[i] > scene.recording.duration,
    )
  )
    throw new Error("网格颜色或观测证据无效。");
  return scene;
}

export function unpackScene(scene) {
  validateScene(scene);
  const m = scene.mesh;
  return {
    vertices: m.vertices.map((p, i) => ({
      p,
      c: m.colors[i],
      n: m.observations[i],
      frames: [],
      supportCount: m.observations[i],
      first: m.first[i],
      last: m.last[i],
    })),
    faces: m.faces,
    rawPoints: null,
    usedFrames: null,
    native: true,
    options: scene.provenance?.config || {},
  };
}
