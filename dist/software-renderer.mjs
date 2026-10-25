// Canvas fallback: complete geometry, painter's depth sorting, cached static layer.
// Skeletons are diagnostic overlays; exact occlusion requires the WebGL renderer.
import { Viewport } from "./renderer.mjs";
import { clamp, cross, sub, norm, dot } from "./core.mjs";

export class SoftwareViewport {
  constructor(canvas, onPick) {
    this.canvas = canvas;
    this.onPick = onPick;
    this.ctx = canvas.getContext("2d");
    if (!this.ctx) throw new Error("当前浏览器无法创建图形上下文。");
    this.cache = document.createElement("canvas");
    this.cacheCtx = this.cache.getContext("2d");
    this.layers = {
      mesh: true,
      skeleton: true,
      trail: true,
      cameras: false,
      reference: false,
    };
    this.mode = "surface";
    this.mesh = null;
    this.selected = null;
    this.reset();
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(canvas);
    this.resize();
    let start = null,
      last = null,
      moved = false;
    canvas.addEventListener("pointerdown", (e) => {
      start = last = [e.clientX, e.clientY];
      moved = false;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", (e) => {
      if (!last) return;
      const dx = e.clientX - last[0],
        dy = e.clientY - last[1];
      moved ||= Math.hypot(e.clientX - start[0], e.clientY - start[1]) > 4;
      this.cameraOverride = null;
      if (e.shiftKey || e.buttons & 2) {
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
      start = last = null;
    });
    canvas.addEventListener("pointercancel", () => {
      start = last = null;
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
        ![
          "ArrowLeft",
          "ArrowRight",
          "ArrowUp",
          "ArrowDown",
          "+",
          "-",
          "=",
          "Home",
        ].includes(e.key)
      )
        return;
      e.preventDefault();
      this.cameraOverride = null;
      if (e.key === "ArrowLeft") this.yaw -= 0.15;
      if (e.key === "ArrowRight") this.yaw += 0.15;
      if (e.key === "ArrowUp") this.pitch = clamp(this.pitch + 0.1, 0.06, 1.56);
      if (e.key === "ArrowDown")
        this.pitch = clamp(this.pitch - 0.1, 0.06, 1.56);
      if (["+", "="].includes(e.key))
        this.distance = Math.max(2, this.distance * 0.9);
      if (e.key === "-") this.distance = Math.min(28, this.distance * 1.1);
      if (e.key === "Home") this.reset();
      this.dirty = true;
    });
  }
  reset() {
    Viewport.prototype.reset.call(this);
  }
  resize() {
    Viewport.prototype.resize.call(this);
    if (this.cache) {
      this.cache.width = this.canvas.width;
      this.cache.height = this.canvas.height;
    }
  }
  matrix() {
    Viewport.prototype.matrix.call(this);
  }
  project(p) {
    return Viewport.prototype.project.call(this, p);
  }
  pick(e) {
    Viewport.prototype.pick.call(this, e);
  }
  setMesh(mesh) {
    this.mesh = mesh;
    this.selected = null;
    this.dirty = true;
  }
  setReference(boxes) {
    this.boxes = boxes;
    this.dirty = true;
  }
  setRecording(recording) {
    this.recording = recording;
    this.dirty = true;
  }
  line(ctx, a, b, color, width = 1) {
    const p = this.project(a),
      q = this.project(b);
    if (!p || !q || p[2] > 1 || q[2] > 1) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(p[0], p[1]);
    ctx.lineTo(q[0], q[1]);
    ctx.stroke();
  }
  staticScene(ctx, w, h) {
    ctx.fillStyle = "#091119";
    ctx.fillRect(0, 0, w, h);
    for (let i = -6; i <= 6; i++) {
      this.line(ctx, [i, -0.015, -6], [i, -0.015, 6], "#172a32");
      this.line(ctx, [-6, -0.015, i], [6, -0.015, i], "#172a32");
    }
    if (this.layers.mesh && this.mesh) {
      const points = this.mesh.vertices.map((v) => this.project(v.p));
      if (this.mode === "points") {
        this.mesh.vertices.forEach((v, i) => {
          const p = points[i];
          if (!p || p[2] > 1) return;
          ctx.fillStyle = `rgb(${v.c.join(",")})`;
          ctx.fillRect(p[0], p[1], 2, 2);
        });
      } else {
        const triangles = [];
        for (const f of this.mesh.faces) {
          const p = f.map((i) => points[i]);
          if (p.some((v) => !v || v[2] > 1)) continue;
          triangles.push({ f, p, z: p.reduce((s, v) => s + v[2], 0) / 3 });
        }
        triangles.sort((a, b) => b.z - a.z);
        ctx.lineWidth = 0.55;
        for (const { f, p } of triangles) {
          ctx.beginPath();
          ctx.moveTo(p[0][0], p[0][1]);
          ctx.lineTo(p[1][0], p[1][1]);
          ctx.lineTo(p[2][0], p[2][1]);
          ctx.closePath();
          if (this.mode === "wire") {
            ctx.strokeStyle = "#39777e";
            ctx.stroke();
          } else {
            const vs = f.map((i) => this.mesh.vertices[i]),
              n = norm(cross(sub(vs[1].p, vs[0].p), sub(vs[2].p, vs[0].p))),
              shade = 0.5 + 0.5 * Math.abs(dot(n, norm([0.3, 1, 0.6]))),
              rgb = [0, 1, 2].map((i) =>
                Math.round((vs.reduce((s, v) => s + v.c[i], 0) / 3) * shade),
              );
            ctx.fillStyle = `rgb(${rgb.join(",")})`;
            ctx.fill();
          }
        }
      }
    }
    if (this.layers.reference)
      for (const b of this.boxes || []) {
        const c = Array.from({ length: 8 }, (_, i) => [
          i & 1 ? b.max[0] : b.min[0],
          i & 2 ? b.max[1] : b.min[1],
          i & 4 ? b.max[2] : b.min[2],
        ]);
        for (let i = 0; i < 8; i++)
          for (let j = 0; j < 3; j++) {
            const k = i ^ (1 << j);
            if (i < k) this.line(ctx, c[i], c[k], "#9b794b");
          }
      }
    if (this.layers.cameras) {
      const frames = this.recording?.frames || [];
      for (let i = 1; i < frames.length; i++)
        this.line(
          ctx,
          frames[i - 1].cameraToWorld.slice(12, 15),
          frames[i].cameraToWorld.slice(12, 15),
          "#7276aa",
        );
    }
    if (this.layers.trail)
      for (const e of this.recording?.entities || [])
        for (let i = 1; i < e.samples.length; i++) {
          const a = e.samples[i - 1],
            b = e.samples[i];
          if (b.t - a.t <= 0.65)
            this.line(
              ctx,
              [a.joints[0][0], 0.025, a.joints[0][2]],
              [b.joints[0][0], 0.025, b.joints[0][2]],
              "#36998b",
              1.5,
            );
        }
  }
  render(poses = []) {
    this.matrix();
    const rect = this.canvas.getBoundingClientRect(),
      d = this.canvas.width / rect.width;
    if (!rect.width || !rect.height) return;
    if (this.dirty) {
      this.cacheCtx.setTransform(d, 0, 0, d, 0, 0);
      this.staticScene(this.cacheCtx, rect.width, rect.height);
    }
    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.ctx.drawImage(this.cache, 0, 0);
    this.ctx.setTransform(d, 0, 0, d, 0, 0);
    if (this.layers.skeleton)
      for (const { entity, pose } of poses) {
        if (!pose) continue;
        const color = pose.gap ? "#ffb257" : "#8aefd3";
        for (const [a, b] of entity.bones)
          this.line(this.ctx, pose.joints[a], pose.joints[b], color, 2.2);
        for (const joint of pose.joints) {
          const p = this.project(joint);
          if (!p) continue;
          this.ctx.fillStyle = color;
          this.ctx.beginPath();
          this.ctx.arc(p[0], p[1], 3, 0, Math.PI * 2);
          this.ctx.fill();
        }
      }
    if (this.selected !== null && this.mesh?.vertices[this.selected]) {
      const p = this.project(this.mesh.vertices[this.selected].p);
      if (p) {
        this.ctx.strokeStyle = "#ffc773";
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.arc(p[0], p[1], 6, 0, Math.PI * 2);
        this.ctx.stroke();
      }
    }
    this.dirty = false;
  }
}
