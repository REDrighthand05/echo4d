# EchoScene reconstruction performance

This change follows #74 and reuses the desktop window and stream-export work in #194. The baseline is [the v0.6.81 payload in #195](https://github.com/REDrighthand05/echo4d/pull/195), commit `1ba6bae3ac54918fe0e3b03e0d1a5c8687cac098`. To run that snapshot from source, copy its four partial hot-update files under `app/` over the matching files at repository root. The candidate already has those changes in the source layout.

## Same-video comparison

Input is the RGB stream of [TUM Freiburg1 desk](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download), originally 640×480. From the sorted RGB filenames, take Python slice `[45:139:3]`, copy those 32 images as numbered PNGs starting at 000 and encode with `ffmpeg -framerate 10 -i %03d.png -c:v libx264 -threads 2 -crf 18 -pix_fmt yuv420p tum-desk-32.mp4`. MP4 SHA-256 from this run is `6eb4aa7b106b83bb0872287ef393eaa96d2719536f12bc0bbf5de40ed1e481be`. The archive SHA-256 is `e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c`. Neither the depth stream nor the TUM trajectory was used.

Linux, AMD EPYC 9V74 with 9 visible logical CPUs, Python 3.12.14, ONNX Runtime 1.32.0 and PyCOLMAP 4.2.1 CPU. Both runs used the shipped Base ONNX model (SHA-256 `dab40f67d5ead5e0ed52acfae41a717a708b02bc57f6cb761c5de828f26a7289`), `OPENBLAS_NUM_THREADS=2`, `OMP_NUM_THREADS=4`, `ECHO_DEPTH_PROVIDER=CPUExecutionProvider` and default scheduling. Each timed run starts in a fresh process and job directory. The desktop calls the same `app.run_pipeline` function. This test omits the live window and its preview thread.

| Stage | v0.6.81 (s) | Candidate (s) | Device in this run |
| --- | ---: | ---: | --- |
| Decode and extract | 0.73 | 0.73 | CPU (two FFmpeg passes, JPEG output) |
| Sharpness selection | 0.13 | 0.15 | CPU |
| Camera trajectory | 12.56 | 12.74 | CPU (SIFT, matching, mapping and bundle adjustment) |
| Depth stage | 82.55 | 44.00 | CPU (Pillow/NumPy and ONNX Runtime) |
| Fusion | 1.11 | 1.20 | CPU (NumPy and disk spill) |
| Cleanup | 2.19 | 2.29 | CPU |
| Scale | 1.61 | 1.87 | CPU |
| Mesh | 6.47 | 8.51 | CPU (Open3D Poisson) |
| Voxel | 0.11 | 0.12 | CPU |
| Standalone scene export | 0.02 | 0.03 | CPU and disk |
| Preview and replay export | 0.28 | 0.28 | CPU and disk |
| End to end | **108.13** | **72.20** | **33.2% less time on this input** |

The baseline used two single-thread CPU sessions for Base when at least eight frames were registered. A separate baseline run with `ECHO_DEPTH_WORKERS=0` took 70.36 s (38.73 s in depth), identifying scheduling as the main measured gain. The candidate defaults Base on CPU to one ORT session; Small keeps its existing scheduling, and an explicit worker override still applies. CPU input preparation took 0.39 s cumulatively in the baseline and 0.17 s in the candidate. The candidate spent 41.83 s inside ORT calls and 0.67 s writing depths and previews. The ORT call includes any input transfer, synchronization and output transfer on a GPU runtime. CPU preparation can overlap the current GPU group and file writes with one group of lookahead. A fixed single-frame graph is never fed a larger batch. Dynamic batches can drop to single frames after rejection without preprocessing those frames again.

Both runs selected, registered, inferred and fused all 32 frames, and all 32 depth arrays were bit-for-bit equal. Independently estimated trajectories differed by 0.032% of the baseline path's bounding-box diagonal in aligned camera-center RMS, with 0.042° median orientation difference. Each exported HTML embedded the original point and mesh buffers byte-for-byte. The two runs exported 79,207 / 80,990 points and 249,999 triangles each. The point count varies with the independent COLMAP solves.

## Windows and NVIDIA reproduction

Keep the same MP4, model SHA-256 and Python dependencies for both worktrees. Put the Base model at `models/depth_anything_v2_base.onnx`, with FFmpeg and FFprobe on PATH. Open PowerShell at each worktree root:

```powershell
$env:PYTHONUTF8 = '1'
$env:ECHO_DEPTH_PROVIDER = 'DmlExecutionProvider'
python -c "import sys; from pathlib import Path; import app; app.run_pipeline(Path(sys.argv[1]),Path(sys.argv[2]),print)" C:/data/tum-desk-32.mp4 C:/echo-ab/run-a
```

Use distinct output directories and repeat at least three times after a warm-up. To record baseline per-stage timings, use the same command with the timing wrapper below saved outside both checkouts. For the candidate, `reconstruction-timing.json` also records total time, time up to scene write, per-stage execution details and valid frame counts.

```python
import functools
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path.cwd()))
import app
from echo_scene import fuse, pipeline

def timed(fn):
    @functools.wraps(fn)
    def run(*args, **kwargs):
        start = perf_counter()
        result = fn(*args, **kwargs)
        print(result["stage"], round(perf_counter() - start, 3), flush=True)
        return result
    return run

for module in (pipeline, fuse):
    for name, fn in vars(module).copy().items():
        if name.startswith("stage_"):
            setattr(module, name, timed(fn))
start = perf_counter()
job = app.run_pipeline(Path(sys.argv[1]), Path(sys.argv[2]), print)
print("total", round(perf_counter() - start, 3), job)
```

The released installer uses DirectML. For a CUDA run, use a separate Python environment with `onnxruntime-gpu` and its compatible NVIDIA CUDA/cuDNN libraries in place of `onnxruntime-directml`, then set `$env:ECHO_DEPTH_PROVIDER = 'CUDAExecutionProvider'`. `nvidia-smi` device discovery, runtime provider availability and actual operator execution are reported separately. Default reporting samples initial inference when a GPU provider is requested. Set `$env:ECHO_DEPTH_PROFILE = '1'` for the full operator trace, in a separate diagnostic run because profiling changes timing. A CUDA model-node event confirms CUDA ran. The trace excludes memory-copy events from this conclusion. Use `ECHO_DEPTH_BATCH_SIZE=1` on low-VRAM systems and `ECHO_DEPTH_PREFETCH=0` to compare preparation overlap. Hardware decode in the legacy extraction route is logged as requested, while the normal dense decode is software. COLMAP may select a device when its wheel supports CUDA, and its SIFT and bundle-adjustment logs are authoritative for that stage.

For the window check, leave a scene reconstructing in the desktop app, orbit the preview, maximize, minimize and restore. Check document identity, task progress, view and preview sequence. The Windows regression exercises the native WebView2 window and the Edge WebGL resize path. The Linux CPU run cannot measure Windows/CUDA throughput or interactive frame latency while a consumer GPU is busy.
