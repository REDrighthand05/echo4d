import { reconstruct } from "./core.mjs";
import { evaluate } from "./demo.mjs";
self.onmessage = ({ data }) => {
  try {
    const start = performance.now(),
      mesh = reconstruct(data.recording, data.options);
    self.postMessage({
      id: data.id,
      mesh: data.metricsOnly ? null : mesh,
      counts: {
        usedFrames: mesh.usedFrames,
        vertices: mesh.vertices.length,
        faces: mesh.faces.length,
      },
      metrics: data.evaluate ? evaluate(mesh) : null,
      elapsed: performance.now() - start,
    });
  } catch (e) {
    self.postMessage({ id: data.id, error: e.message });
  }
};
