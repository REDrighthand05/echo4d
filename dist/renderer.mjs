import { add, sub, mul, dot, cross, norm, clamp } from "./core.mjs";
export function multiply(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++)
      for (let k = 0; k < 4; k++) o[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
  return o;
}
export function lookAt(e, t) {
  const z = norm(sub(e, t)),
    x = norm(cross([0, 1, 0], z)),
    y = cross(z, x);
  return new Float32Array([
    x[0],
    y[0],
    z[0],
    0,
    x[1],
    y[1],
    z[1],
    0,
    x[2],
    y[2],
    z[2],
    0,
    -dot(x, e),
    -dot(y, e),
    -dot(z, e),
    1,
  ]);
}
export function perspective(fov, aspect, near, far) {
  const f = 1 / Math.tan(fov / 2),
    nf = 1 / (near - far);
  return new Float32Array([
    f / aspect,
    0,
    0,
    0,
    0,
    f,
    0,
    0,
    0,
    0,
    (far + near) * nf,
    -1,
    0,
    0,
    2 * far * near * nf,
    0,
  ]);
}
const shaders = {
  vertex: `attribute vec3 aPosition;attribute vec3 aNormal;attribute vec3 aColor;uniform mat4 uMatrix;uniform float uPointSize;varying vec3 vColor;varying vec3 vNormal;varying float vDepth;void main(){gl_Position=uMatrix*vec4(aPosition,1.);gl_PointSize=uPointSize;vColor=aColor;vNormal=aNormal;vDepth=gl_Position.w;}`,
  fragment: `precision mediump float;varying vec3 vColor;varying vec3 vNormal;varying float vDepth;uniform float uLit;uniform float uPoint;uniform float uAlpha;void main(){if(uPoint>0.5 && distance(gl_PointCoord,vec2(0.5))>0.5)discard;float shade=.50+.50*abs(dot(normalize(vNormal+vec3(.0001)),normalize(vec3(.3,1.,.6))));vec3 c=vColor*mix(1.,shade,uLit);float fog=clamp((vDepth-8.)/28.,0.,.55);gl_FragColor=vec4(mix(c,vec3(.035,.065,.082),fog),uAlpha);}`,
};
function append(out, p, n, c) {
  out.push(...p, ...n, ...c);
}
function triangle(out, a, b, c, color) {
  const n = norm(cross(sub(b, a), sub(c, a)));
  append(out, a, n, color);
  append(out, b, n, color);
  append(out, c, n, color);
}
export function line(out, a, b, c) {
  append(out, a, [0, 1, 0], c);
  append(out, b, [0, 1, 0], c);
}
function cylinder(out, a, b, r, c) {
  const f = norm(sub(b, a)),
    u = norm(cross(f, Math.abs(f[1]) > 0.9 ? [1, 0, 0] : [0, 1, 0])),
    v = cross(f, u);
  for (let i = 0; i < 7; i++) {
    const at = (t) => add(mul(u, Math.cos(t) * r), mul(v, Math.sin(t) * r)),
      p = at((i / 7) * Math.PI * 2),
      q = at(((i + 1) / 7) * Math.PI * 2);
    triangle(out, add(a, p), add(b, p), add(a, q), c);
    triangle(out, add(a, q), add(b, p), add(b, q), c);
  }
}
function sphere(out, p, r, c) {
  const at = (u, v) =>
    add(p, [
      r * Math.sin(v) * Math.cos(u),
      r * Math.cos(v),
      r * Math.sin(v) * Math.sin(u),
    ]);
  for (let i = 0; i < 8; i++)
    for (let j = 0; j < 5; j++) {
      const a = (i / 8) * Math.PI * 2,
        b = ((i + 1) / 8) * Math.PI * 2,
        v = (j / 5) * Math.PI,
        w = ((j + 1) / 5) * Math.PI;
      triangle(out, at(a, v), at(a, w), at(b, w), c);
      triangle(out, at(a, v), at(b, w), at(b, v), c);
    }
}
export class Viewport {
  constructor(canvas, onPick) {
    this.canvas = canvas;
    this.onPick = onPick;
    this.gl = canvas.getContext("webgl", { antialias: true, alpha: false });
    if (!this.gl)
      throw new Error(
        "浏览器未启用 WebGL，无法显示三维场景。请开启硬件加速后重试。",
      );
    const g = this.gl,
      compile = (type, source) => {
        const s = g.createShader(type);
        g.shaderSource(s, source);
        g.compileShader(s);
        if (!g.getShaderParameter(s, g.COMPILE_STATUS))
          throw new Error(g.getShaderInfoLog(s));
        return s;
      };
    this.program = g.createProgram();
    g.attachShader(this.program, compile(g.VERTEX_SHADER, shaders.vertex));
    g.attachShader(this.program, compile(g.FRAGMENT_SHADER, shaders.fragment));
    g.linkProgram(this.program);
    if (!g.getProgramParameter(this.program, g.LINK_STATUS))
      throw new Error(g.getProgramInfoLog(this.program));
    g.useProgram(this.program);
    this.locations = Object.fromEntries(
      ["uMatrix", "uPointSize", "uLit", "uPoint", "uAlpha"].map((n) => [
        n,
        g.getUniformLocation(this.program, n),
      ]),
    );
    this.attributes = ["aPosition", "aNormal", "aColor"].map((n) =>
      g.getAttribLocation(this.program, n),
    );
    this.dynamic = g.createBuffer();
    this.target = [0, 0.65, -0.35];
    this.yaw = 0.68;
    this.pitch = 0.5;
    this.distance = 9.5;
    this.mode = "surface";
    this.mesh = null;
    this.selected = null;
    this.cameraOverride = null;
    this.layers = {
      mesh: true,
      skeleton: true,
      trail: true,
      cameras: false,
      reference: false,
    };
    const grid = [];
    for (let i = -6; i <= 6; i++) {
      line(
        grid,
        [i, -0.015, -6],
        [i, -0.015, 6],
        i === 0 ? [0.17, 0.29, 0.32] : [0.1, 0.16, 0.19],
      );
      line(
        grid,
        [-6, -0.015, i],
        [6, -0.015, i],
        i === 0 ? [0.17, 0.29, 0.32] : [0.1, 0.16, 0.19],
      );
    }
    this.grid = this.upload(grid);
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
    this.resize();
    let start = null,
      last = null,
      moved = false;
    canvas.addEventListener("pointerdown", (e) => {
      start = [e.clientX, e.clientY];
      last = start;
      moved = false;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", (e) => {
      if (!last) return;
      const dx = e.clientX - last[0],
        dy = e.clientY - last[1];
      if (Math.hypot(e.clientX - start[0], e.clientY - start[1]) > 4)
        moved = true;
      this.cameraOverride = null;
      if (e.shiftKey || e.button === 2) {
        const s = this.distance * 0.0015;
        this.target[0] -= dx * s * Math.cos(this.yaw);
        this.target[2] += dx * s * Math.sin(this.yaw);
        this.target[1] += dy * s;
      } else {
        this.yaw -= dx * 0.007;
        this.pitch = clamp(this.pitch + dy * 0.006, 0.06, 1.56);
      }
      last = [e.clientX, e.clientY];
      this.dirty = true;
    });
    canvas.addEventListener("pointerup", (e) => {
      if (start && !moved) this.pick(e);
      last = null;
      start = null;
    });
    canvas.addEventListener("pointercancel", () => {
      last = null;
      start = null;
    });
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    canvas.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        this.distance = clamp(
          this.distance * Math.exp(e.deltaY * 0.001),
          2,
          28,
        );
        this.cameraOverride = null;
        this.dirty = true;
      },
      { passive: false },
    );
    canvas.addEventListener("keydown", (e) => {
      if (
        [
          "ArrowLeft",
          "ArrowRight",
          "ArrowUp",
          "ArrowDown",
          "+",
          "-",
          "=",
          "Home",
        ].includes(e.key)
      ) {
        e.preventDefault();
        this.cameraOverride = null;
        if (e.key === "ArrowLeft") this.yaw -= 0.15;
        if (e.key === "ArrowRight") this.yaw += 0.15;
        if (e.key === "ArrowUp")
          this.pitch = clamp(this.pitch + 0.1, 0.06, 1.56);
        if (e.key === "ArrowDown")
          this.pitch = clamp(this.pitch - 0.1, 0.06, 1.56);
        if (["+", "="].includes(e.key))
          this.distance = Math.max(2, this.distance * 0.9);
        if (e.key === "-") this.distance = Math.min(28, this.distance * 1.1);
        if (e.key === "Home") this.reset();
        this.dirty = true;
      }
    });
  }
  reset() {
    this.target = [0, 0.65, -0.35];
    this.yaw = 0.68;
    this.pitch = 0.5;
    this.distance = 9.5;
    this.cameraOverride = null;
    this.dirty = true;
  }
  resize() {
    const r = this.canvas.getBoundingClientRect(),
      d = Math.min(devicePixelRatio || 1, 2);
    if (r.width && r.height) {
      this.canvas.width = Math.round(r.width * d);
      this.canvas.height = Math.round(r.height * d);
      this.dirty = true;
    }
  }
  upload(data) {
    const g = this.gl,
      b = g.createBuffer();
    g.bindBuffer(g.ARRAY_BUFFER, b);
    g.bufferData(g.ARRAY_BUFFER, new Float32Array(data), g.STATIC_DRAW);
    return { b, count: data.length / 9 };
  }
  dispose(b) {
    if (b) this.gl.deleteBuffer(b.b);
  }
  setMesh(mesh) {
    this.mesh = mesh;
    this.selected = null;
    this.dispose(this.surface);
    this.dispose(this.points);
    this.dispose(this.edges);
    const tris = [],
      pts = [],
      edges = [],
      seen = new Set();
    for (const v of mesh.vertices)
      append(
        pts,
        v.p,
        [0, 1, 0],
        v.c.map((c) => c / 255),
      );
    for (const f of mesh.faces) {
      const vv = f.map((i) => mesh.vertices[i]);
      if (vv.some((v) => !v)) continue;
      const c = [0, 1, 2].map(
        (i) => vv.reduce((s, v) => s + v.c[i], 0) / 3 / 255,
      );
      triangle(tris, ...vv.map((v) => v.p), c);
      for (let i = 0; i < 3; i++) {
        const a = f[i],
          b = f[(i + 1) % 3],
          key = a < b ? `${a},${b}` : `${b},${a}`;
        if (!seen.has(key)) {
          seen.add(key);
          line(
            edges,
            mesh.vertices[a].p,
            mesh.vertices[b].p,
            [0.19, 0.45, 0.48],
          );
        }
      }
    }
    this.surface = this.upload(tris);
    this.points = this.upload(pts);
    this.edges = this.upload(edges);
    this.dirty = true;
  }
  setReference(boxes) {
    this.dispose(this.reference);
    const lines = [];
    for (const b of boxes) {
      const corners = [];
      for (let i = 0; i < 8; i++)
        corners.push([
          i & 1 ? b.max[0] : b.min[0],
          i & 2 ? b.max[1] : b.min[1],
          i & 4 ? b.max[2] : b.min[2],
        ]);
      for (let i = 0; i < 8; i++)
        for (let j = 0; j < 3; j++) {
          const k = i ^ (1 << j);
          if (i < k) line(lines, corners[i], corners[k], [0.56, 0.43, 0.26]);
        }
    }
    this.reference = this.upload(lines);
  }
  setRecording(recording) {
    this.recording = recording;
    this.dispose(this.trails);
    this.dispose(this.cameras);
    const trails = [],
      cameras = [];
    for (const e of recording.entities || []) {
      for (let i = 1; i < e.samples.length; i++) {
        const a = e.samples[i - 1],
          b = e.samples[i];
        if (b.t - a.t > 0.65) continue;
        const p = [a.joints[0][0], 0.025, a.joints[0][2]],
          q = [b.joints[0][0], 0.025, b.joints[0][2]];
        line(trails, p, q, [0.16, 0.57, 0.51]);
      }
    }
    for (let i = 1; i < recording.frames.length; i++)
      line(
        cameras,
        recording.frames[i - 1].cameraToWorld.slice(12, 15),
        recording.frames[i].cameraToWorld.slice(12, 15),
        [0.34, 0.37, 0.63],
      );
    this.trails = this.upload(trails);
    this.cameras = this.upload(cameras);
    this.dirty = true;
  }
  matrix() {
    let eye,
      target = this.target;
    if (this.cameraOverride) {
      eye = this.cameraOverride.slice(12, 15);
      target = add(eye, this.cameraOverride.slice(8, 11));
    } else
      eye = add(target, [
        Math.sin(this.yaw) * Math.cos(this.pitch) * this.distance,
        Math.sin(this.pitch) * this.distance,
        Math.cos(this.yaw) * Math.cos(this.pitch) * this.distance,
      ]);
    this.eye = eye;
    this.mvp = multiply(
      perspective(0.76, this.canvas.width / this.canvas.height, 0.05, 80),
      lookAt(eye, target),
    );
  }
  project(p) {
    const m = this.mvp;
    if (!m) return null;
    const a = [p[0], p[1], p[2], 1],
      v = [0, 0, 0, 0];
    for (let r = 0; r < 4; r++)
      for (let c = 0; c < 4; c++) v[r] += m[c * 4 + r] * a[c];
    if (v[3] <= 0) return null;
    const rect = this.canvas.getBoundingClientRect();
    return [
      ((v[0] / v[3]) * 0.5 + 0.5) * rect.width,
      ((-v[1] / v[3]) * 0.5 + 0.5) * rect.height,
      v[2] / v[3],
    ];
  }
  pick(e) {
    if (!this.mesh) return;
    const rect = this.canvas.getBoundingClientRect(),
      x = e.clientX - rect.left,
      y = e.clientY - rect.top;
    let best = null,
      dist = 13;
    for (let i = 0; i < this.mesh.vertices.length; i++) {
      const p = this.project(this.mesh.vertices[i].p);
      if (!p || p[2] > 1) continue;
      const d = Math.hypot(x - p[0], y - p[1]);
      if (d < dist) {
        dist = d;
        best = i;
      }
    }
    this.selected = best;
    this.onPick?.(best === null ? null : this.mesh.vertices[best]);
    this.dirty = true;
  }
  drawBuffer(buffer, type, lit = 0, alpha = 1, pointSize = 3) {
    if (!buffer || !buffer.count) return;
    const g = this.gl;
    g.bindBuffer(g.ARRAY_BUFFER, buffer.b);
    for (let i = 0; i < 3; i++) {
      g.enableVertexAttribArray(this.attributes[i]);
      g.vertexAttribPointer(this.attributes[i], 3, g.FLOAT, false, 36, i * 12);
    }
    g.uniform1f(this.locations.uLit, lit);
    g.uniform1f(this.locations.uPoint, type === g.POINTS ? 1 : 0);
    g.uniform1f(this.locations.uPointSize, pointSize * (devicePixelRatio || 1));
    g.uniform1f(this.locations.uAlpha, alpha);
    g.drawArrays(type, 0, buffer.count);
  }
  drawDynamic(data, type, lit = 0, alpha = 1, pointSize = 3) {
    const g = this.gl;
    g.bindBuffer(g.ARRAY_BUFFER, this.dynamic);
    g.bufferData(g.ARRAY_BUFFER, new Float32Array(data), g.DYNAMIC_DRAW);
    this.drawBuffer(
      { b: this.dynamic, count: data.length / 9 },
      type,
      lit,
      alpha,
      pointSize,
    );
  }
  render(poses = []) {
    const g = this.gl;
    g.viewport(0, 0, this.canvas.width, this.canvas.height);
    g.clearColor(0.035, 0.065, 0.082, 1);
    g.clear(g.COLOR_BUFFER_BIT | g.DEPTH_BUFFER_BIT);
    g.enable(g.DEPTH_TEST);
    g.enable(g.BLEND);
    g.blendFunc(g.SRC_ALPHA, g.ONE_MINUS_SRC_ALPHA);
    g.useProgram(this.program);
    this.matrix();
    g.uniformMatrix4fv(this.locations.uMatrix, false, this.mvp);
    this.drawBuffer(this.grid, g.LINES);
    if (this.layers.mesh) {
      if (this.mode === "points")
        this.drawBuffer(this.points, g.POINTS, 0, 1, 2);
      else if (this.mode === "wire")
        this.drawBuffer(this.edges, g.LINES, 0, 0.78);
      else {
        g.enable(g.POLYGON_OFFSET_FILL);
        g.polygonOffset(1, 1);
        this.drawBuffer(this.surface, g.TRIANGLES, 1);
        g.disable(g.POLYGON_OFFSET_FILL);
      }
    }
    if (this.layers.reference)
      this.drawBuffer(this.reference, g.LINES, 0, 0.65);
    if (this.layers.trail) this.drawBuffer(this.trails, g.LINES);
    if (this.layers.cameras) this.drawBuffer(this.cameras, g.LINES);
    if (this.layers.skeleton) {
      const body = [],
        joints = [];
      for (const { entity, pose } of poses) {
        if (!pose) continue;
        const c = pose.gap ? [1, 0.62, 0.26] : [0.36, 0.97, 0.83];
        for (const [a, b] of entity.bones)
          cylinder(body, pose.joints[a], pose.joints[b], 0.021, c);
        pose.joints.forEach((p, i) =>
          sphere(joints, p, i === 3 ? 0.105 : 0.035, c),
        );
      }
      this.drawDynamic(body, g.TRIANGLES, 0.25);
      this.drawDynamic(joints, g.TRIANGLES, 0.3);
    }
    if (this.selected !== null && this.mesh?.vertices[this.selected]) {
      const data = [];
      append(
        data,
        this.mesh.vertices[this.selected].p,
        [0, 1, 0],
        [1, 0.77, 0.39],
      );
      g.disable(g.DEPTH_TEST);
      this.drawDynamic(data, g.POINTS, 0, 1, 7);
      g.enable(g.DEPTH_TEST);
    }
    this.dirty = false;
  }
}
