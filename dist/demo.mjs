// The synthetic world is ONLY used by the sensor simulator and evaluator.
import {
  add,
  mul,
  sub,
  dot,
  cross,
  norm,
  cameraMatrix,
  unproject,
  lerp,
  clamp,
} from "./core.mjs";
export const BOXES = [
  {
    name: "地面",
    min: [-3.2, -0.12, -2.6],
    max: [3.2, 0, 2.6],
    color: [78, 104, 112],
  },
  {
    name: "后墙",
    min: [-3.2, 0, -2.72],
    max: [3.2, 2.8, -2.6],
    color: [116, 145, 151],
  },
  {
    name: "侧墙",
    min: [-3.32, 0, -2.6],
    max: [-3.2, 2.8, 2.6],
    color: [102, 130, 139],
  },
  {
    name: "沙发",
    min: [-2.75, 0.2, -2.22],
    max: [-0.55, 0.52, -1.25],
    color: [76, 133, 135],
  },
  {
    name: "沙发靠背",
    min: [-2.75, 0.45, -2.24],
    max: [-0.55, 1.05, -2.02],
    color: [87, 143, 146],
  },
  {
    name: "左扶手",
    min: [-2.85, 0.28, -2.22],
    max: [-2.62, 0.77, -1.25],
    color: [97, 151, 153],
  },
  {
    name: "右扶手",
    min: [-0.67, 0.28, -2.22],
    max: [-0.44, 0.77, -1.25],
    color: [97, 151, 153],
  },
  {
    name: "茶几",
    min: [-0.7, 0.48, -0.47],
    max: [0.85, 0.58, 0.45],
    color: [147, 164, 158],
  },
  ...[
    [-0.57, -0.34],
    [0.7, -0.34],
    [-0.57, 0.32],
    [0.7, 0.32],
  ].map(([x, z]) => ({
    name: "桌腿",
    min: [x - 0.045, 0, z - 0.045],
    max: [x + 0.045, 0.48, z + 0.045],
    color: [100, 124, 126],
  })),
  {
    name: "储物架背板",
    min: [1.85, 0, -2.56],
    max: [2.88, 2, -2.45],
    color: [85, 125, 133],
  },
  ...[0, 0.63, 1.25, 1.94].map((y) => ({
    name: "层板",
    min: [1.85, y, -2.48],
    max: [2.88, y + 0.07, -1.95],
    color: [119, 156, 159],
  })),
  ...[1.85, 2.81].map((x) => ({
    name: "架侧板",
    min: [x, 0, -2.48],
    max: [x + 0.07, 2, -1.95],
    color: [108, 143, 150],
  })),
  {
    name: "箱体 A",
    min: [2.02, 0.7, -2.39],
    max: [2.42, 1.08, -2.04],
    color: [145, 161, 151],
  },
  {
    name: "箱体 B",
    min: [2.47, 1.32, -2.4],
    max: [2.75, 1.73, -2.07],
    color: [153, 146, 124],
  },
  {
    name: "展示台",
    min: [1.85, 0, 0.2],
    max: [2.43, 0.8, 0.78],
    color: [96, 137, 143],
  },
  {
    name: "陈列物",
    min: [2.04, 0.8, 0.38],
    max: [2.23, 1.1, 0.59],
    color: [152, 178, 169],
  },
];
export function rayBox(origin, dir, b) {
  let near = -Infinity,
    far = Infinity;
  for (let a = 0; a < 3; a++) {
    if (Math.abs(dir[a]) < 1e-9) {
      if (origin[a] < b.min[a] || origin[a] > b.max[a]) return Infinity;
      continue;
    }
    let t1 = (b.min[a] - origin[a]) / dir[a],
      t2 = (b.max[a] - origin[a]) / dir[a];
    if (t1 > t2) [t1, t2] = [t2, t1];
    near = Math.max(near, t1);
    far = Math.min(far, t2);
    if (far < near) return Infinity;
  }
  return far < 0.05 ? Infinity : near > 0.05 ? near : far;
}
export function renderDepth(k, m, t) {
  const depth = [],
    colors = [],
    p = m.slice(12, 15);
  for (let v = 0; v < k.height; v++)
    for (let u = 0; u < k.width; u++) {
      const ray = sub(unproject(u, v, 1, k, m), p);
      let z = 12,
        obj = null;
      for (const b of BOXES) {
        const d = rayBox(p, ray, b);
        if (d < z) {
          z = d;
          obj = b;
        }
      }
      depth.push(obj ? Math.round(z * 100000) / 100000 : 0);
      colors.push(...(obj ? obj.color : [0, 0, 0]));
    }
  return { t, cameraToWorld: m, depth, colors };
}
const names = [
  "pelvis",
  "chest",
  "neck",
  "head",
  "left_shoulder",
  "left_elbow",
  "left_wrist",
  "right_shoulder",
  "right_elbow",
  "right_wrist",
  "left_hip",
  "left_knee",
  "left_ankle",
  "right_hip",
  "right_knee",
  "right_ankle",
];
export const BONES = [
  [0, 1],
  [1, 2],
  [2, 3],
  [2, 4],
  [4, 5],
  [5, 6],
  [2, 7],
  [7, 8],
  [8, 9],
  [0, 10],
  [10, 11],
  [11, 12],
  [0, 13],
  [13, 14],
  [14, 15],
  [4, 7],
  [10, 13],
];
function root(t) {
  const keys = [
    [0, [-2.3, 0, 1.7]],
    [4, [-1.4, 0, 0.95]],
    [10, [1.45, 0, 1.15]],
    [14, [1.55, 0, -0.95]],
    [16, [1.55, 0, -0.95]],
  ];
  let n = keys.findIndex((k) => k[0] >= t);
  if (n <= 0) return keys[0][1];
  const a = keys[n - 1],
    b = keys[n];
  return lerp(a[1], b[1], (t - a[0]) / (b[0] - a[0]));
}
function joints(t) {
  const p = root(t),
    moving = t < 14,
    phase = t * 5.4,
    s = moving ? Math.sin(phase) : 0,
    bob = moving ? Math.abs(Math.cos(phase)) * 0.025 : 0,
    reach = clamp((t - 13.5) / 1.5, 0, 1);
  let j = [
    [0, 0.94, 0],
    [0, 1.3, 0],
    [0, 1.51, 0],
    [0, 1.73, 0],
    [-0.21, 1.47, 0],
    [-0.28, 1.19, s * 0.16],
    [-0.3, 0.99, s * 0.28],
    [0.21, 1.47, 0],
    [0.28, 1.19, -s * 0.16],
    [0.3, 0.99, -s * 0.28],
    [-0.13, 0.92, 0],
    [-0.14, 0.49, s * 0.15],
    [-0.14, 0.06, s * 0.28],
    [0.13, 0.92, 0],
    [0.14, 0.49, -s * 0.15],
    [0.14, 0.06, -s * 0.28],
  ];
  j[8] = lerp(j[8], [0.35, 1.33, -0.18], reach);
  j[9] = lerp(j[9], [0.57, 1.44, -0.43], reach);
  return j.map((v) => add(p, [v[0], v[1] + bob, v[2]]));
}
export function createDemo() {
  const k = { width: 64, height: 48, fx: 45, fy: 45, cx: 31.5, cy: 23.5 },
    frames = [];
  for (let i = 0; i < 24; i++) {
    const a = -1.35 + (i / 23) * 2.7;
    const p = [
      Math.sin(a) * 2.85,
      1.65 + 0.18 * Math.sin(i * 0.5),
      Math.cos(a) * 2.32,
    ];
    frames.push(
      renderDepth(k, cameraMatrix(p, [0, 0.78, -0.95]), (i / 23) * 16),
    );
  }
  const samples = [];
  for (let i = 0; i <= 80; i++) {
    const t = Math.round(i * 0.2 * 100) / 100;
    if (t > 5.6 && t < 8.8) continue;
    samples.push({ t, joints: joints(t) });
  }
  return {
    version: "echo-1",
    name: "客厅 · 一次经过",
    source: "synthetic",
    units: "meters",
    duration: 16,
    intrinsics: k,
    frames,
    entities: [{ id: "P-01", jointNames: names, bones: BONES, samples }],
    notes:
      "合成深度与程序生成的骨架；5.6–8.8 秒骨架关键帧被主动移除。相机位姿为已知输入。静态深度不含移动人物。",
  };
}
export function surfaceDistance(p) {
  let best = Infinity;
  for (const b of BOXES) {
    const outside = p.map((v, i) => Math.max(b.min[i] - v, 0, v - b.max[i]));
    let d = Math.hypot(...outside);
    if (d === 0)
      d = Math.min(...p.flatMap((v, i) => [v - b.min[i], b.max[i] - v]));
    best = Math.min(best, d);
  }
  return best;
}
let heldout;
export function heldoutPoints() {
  if (heldout) return heldout;
  heldout = [];
  const k = { width: 32, height: 24, fx: 22.5, fy: 22.5, cx: 15.5, cy: 11.5 };
  for (const [i, p] of [
    [-2.55, 1.85, 1.22],
    [-1.1, 1.5, 2.42],
    [0.65, 1.88, 2.38],
    [2.55, 1.54, 1.22],
    [0, 2.25, 1.6],
  ].entries()) {
    const f = renderDepth(k, cameraMatrix(p, [0, 0.65, -1]), i);
    for (let v = 0; v < k.height; v++)
      for (let u = 0; u < k.width; u++) {
        const z = f.depth[v * k.width + u];
        if (z) heldout.push(unproject(u, v, z, k, f.cameraToWorld));
      }
  }
  return heldout;
}
export function evaluate(mesh) {
  if (!mesh.vertices.length)
    return {
      rmse: null,
      coverage: 0,
      heldoutCount: heldoutPoints().length,
      threshold: 0.12,
    };
  const sq = mesh.vertices.reduce((s, v) => s + surfaceDistance(v.p) ** 2, 0);
  const cell = 0.12,
    grid = new Map();
  for (const v of mesh.vertices) {
    const key = v.p.map((x) => Math.floor(x / cell)).join(",");
    if (!grid.has(key)) grid.set(key, []);
    grid.get(key).push(v.p);
  }
  let covered = 0;
  const points = heldoutPoints();
  for (const p of points) {
    const cellp = p.map((x) => Math.floor(x / cell));
    let found = false;
    search: for (let x = -1; x <= 1; x++)
      for (let y = -1; y <= 1; y++)
        for (let z = -1; z <= 1; z++) {
          const list = grid.get(
            [cellp[0] + x, cellp[1] + y, cellp[2] + z].join(","),
          );
          if (list)
            for (const q of list) {
              if (dot(sub(p, q), sub(p, q)) < cell * cell) {
                found = true;
                break search;
              }
            }
        }
    if (found) covered++;
  }
  return {
    rmse: Math.sqrt(sq / mesh.vertices.length),
    coverage: covered / points.length,
    heldoutCount: points.length,
    threshold: cell,
  };
}
