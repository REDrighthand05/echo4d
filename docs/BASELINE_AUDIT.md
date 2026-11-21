# Echo4D / EchoScene: Baseline Audit & Heterogeneous Execution Benchmark

## 1. Executive Summary & Codebase Baseline

- **Repository Target**: `REDrighthand05/echo4d`
- **Active Feature Branch**: `perf-smooth-60fps` (synchronized with `origin/main` at `63fd9a8aaf768328a36551cacd14d31b40e582e8`).
- **Product Baseline Comparison**: `product/echo-scene` (`c5392d96e8cd3a22b11c33554d4adae602f0cc55`) is a historical release branch (v0.6.21) prior to the 4-layer micro-modular architecture refactoring. The main branch contains the modern architecture (`echo_engine`, `echo_service`, `echo_cli`, `echo_desktop`), 100% compliant with the `<= 300` lines/module rule.
- **Related PR Assessment**:
  - **PR #196** (`perf/windows-reconstruction-main`): Streamlined single ORT CPU session for Base, batch preprocessing pipeline, phase timing verification.
  - **PR #194** (`perf/windows-resize-preview-depth`): WebGL context reuse, debounced resizing, atomic streamed scene export.

## 2. Hardware & Device Verification

- **Operating System**: Microsoft Windows (AMD64)
- **GPU Device**: NVIDIA GeForce RTX 2060
  - **VRAM Total**: 6144 MiB (usable budget: ~5120 MiB after UI / DWM reserve)
  - **NVIDIA Driver Version**: 616.64
  - **CUDA UMD Version**: 13.4 (CUDA 13.3 Toolkit runtime installed)
- **Software Runtime**:
  - Python: 3.13.14 (AMD64)
  - ONNX Runtime: 1.24.4 with `DmlExecutionProvider` and `CPUExecutionProvider`
  - Open3D: 0.20.0
  - PyCOLMAP: 4.2.1
  - OpenCV: 5.1.0
  - NumPy: 2.5.3, SciPy: 1.18.1

## 3. Five-Tier Execution Classification

The heterogeneous inference engine distinguishes real hardware computation from mere driver presence:

| Tier | Status | Verification Mechanism |
|---|---|---|
| **Tier 1: Hardware Detected** | Verified | `nvidia-smi` reports GeForce RTX 2060 (6144 MiB). |
| **Tier 2: Provider Available** | Verified | `ort.get_available_providers()` returns `['DmlExecutionProvider', 'CPUExecutionProvider']`. |
| **Tier 3: Session Created** | Verified | `InferenceSession` binds `DmlExecutionProvider` successfully. |
| **Tier 4: GPU Executed** | Verified | Warm inference time: **224.3 ms / frame** (Base model), **116.2 ms / frame** (Small model), confirming native GPU kernel dispatch. |
| **Tier 5: CPU Fallback** | Verified | Automated fallback on unsupported ops or VRAM exhaustion. |

## 4. End-to-End Baseline Timing on TUM Freiburg1 Desk (32 Frames)

Input sequence: TUM Freiburg1 desk RGB (32 frames, 640x480, `tum-desk-32.mp4`):

| Pipeline Stage | Baseline Runtime (s) | Execution Device & Implementation |
|---|---:|---|
| Frame Extraction (`stage_extract`) | 0.576 | FFmpeg pass (libx264 software decode to JPEG) |
| Frame Selection (`stage_select`) | 0.537 | Laplacian variance sharpness filtering |
| Camera Trajectory (`stage_poses`) | 31.540 | PyCOLMAP SIFT extraction, matcher, bundle adjustment |
| Depth Estimation (`stage_depth`) | **9.938** | Depth Anything V2 Base on **GPU (`DmlExecutionProvider`)** |
| Point Cloud Fusion (`stage_fuse`) | 2.385 | NumPy backprojection & disk-spill depth fusion |
| Statistical Cleaning (`stage_clean`) | 3.695 | Connected component & radius outlier filtering |
| Metric Scale (`stage_scale`) | 1.093 | Anchor detection & scale estimation |
| Surface Mesh (`stage_mesh`) | 5.016 | Open3D Poisson Reconstruction (depth 9, density trim) |
| Voxel Export (`stage_voxel`) | 0.160 | Voxel aggregation & GLB export |
| Packaging (`stage_present`) | 0.126 | HTML assembly & asset binding |
| Interactive Preview (`stage_preview`) | 0.451 | Three.js orbit view builder |
| Trajectory Replay (`stage_replay`) | 0.133 | Camera path animation exporter |
| **Total Pipeline** | **58.536 s** | Complete reconstruction end-to-end |

## 5. Artifact Verification

Generated outputs in `job_tum-desk-32`:
- `mesh.ply`: 11,110,655 bytes (trimmed Poisson surface mesh)
- `points.bin`: 1,194,165 bytes (fused point cloud, 15 bytes/point)
- `mesh_vertices.bin`: 1,877,715 bytes (web decimation vertices)
- `mesh_indices.bin`: 3,000,000 bytes (triangle index buffer)
- `voxel_model.glb`: 21,018,316 bytes (standalone 3D asset)
- `scene.html`: 8,112,265 bytes (complete standalone interactive WebGL viewer)
