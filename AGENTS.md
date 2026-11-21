# ECHO 4D Architecture & AI Agent Maintenance Guide (AGENTS.md)

## 1. Architectural Philosophy & Token Dilution Defense
In complex projects, LLM agents begin experiencing context dilution and attention drift when maintaining monolithic source files exceeding 1,000 lines (often surpassing 260k tokens across multi-turn reasoning).

To guarantee lifelong maintainability, ECHO is structured around a **4-Layer Micro-Modular Architecture** governed by one strict invariant:
> **The Micro-Modular Invariant**: Every single Python module must strictly be **<= 300 lines of code**. Large monolithic classes must be decoupled into cohesive domain modules, facades, and helpers.

```
+-------------------------------------------------------------------------+
|                              Entrypoints                                |
|   EchoScene.exe / web_app.py (<80 lines)  |  app.py (<120 lines facade)  |
+-------------------------------------------------------------------------+
                                    |
      +-----------------------------+-----------------------------+
      |                             |                             |
      v                             v                             v
+------------------+      +-------------------+      +-------------------+
|  echo_desktop    |      |    echo_cli       |      |   echo_service    |
| (PyWebview, Win32|      | (Headless CLI:    |      | (REST API Server, |
|  Bridge & WebUI) |      |  run, depth, ...) |      |  Queue & Export)  |
+------------------+      +-------------------+      +-------------------+
      |                             |                             |
      +-----------------------------+-----------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                               echo_engine                               |
|        (Pure 12-Stage 3D Reconstruction Pipeline, Depth, & Jobs)        |
+-------------------------------------------------------------------------+
```

---

## 2. Directory Layout & Module Responsibilities

### Layer 1: Pure Algorithm Engine (`echo_engine/`)
Completely detached from GUI frameworks, windowing systems, and HTTP protocols.
* **`echo_engine/types.py`** (21 lines): Type annotations (`JobOutput`, callback definitions).
* **`echo_engine/jobs.py`** (133 lines): Job queries, directory listings, status detection, scale lines.
* **`echo_engine/depth.py`** (26 lines): Depth estimation kernel runner.
* **`echo_engine/pipeline.py`** (139 lines): 12-stage 3D reconstruction pipeline orchestrator with atomic crash-recovery markers and inter-stage memory management (`gc.collect()`).
* **`echo_engine/__init__.py`** (39 lines): Clean public API re-exports.

### Layer 2: Service & Task Management (`echo_service/`)
Asynchronous execution, disk cache throttling, validators, and REST endpoints.
* **`echo_service/types.py`** (29 lines): Serializable data schemas (`JobInfo`, `ServiceStatus`).
* **`echo_service/cache.py`** (34 lines): `ThrottledCache` (2.5s / 3.0s TTL) eliminating filesystem polling churn.
* **`echo_service/validator.py`** (48 lines): Pre-flight validation (video format, file size, disk space >= 500MB, duration > 0.2s).
* **`echo_service/exporter.py`** (80 lines): Universal voxel solidifier (GLB/PLY), camera trajectory export, batch export.
* **`echo_service/task_queue.py`** (238 lines): `JobManager` thread management, busy locks, progress callbacks, and post-task garbage collection.
* **`echo_service/server.py`** (169 lines): Lightweight local REST API server exposing `/api/status`, `/api/jobs`, `/api/jobs/select`, `/api/jobs/scene`, `/api/jobs/depth`, `/api/export`.
* **`echo_service/__init__.py`** (35 lines): Clean service facade.

### Layer 3: Headless Command-Line Interface (`echo_cli/`)
Standardized CLI interface for automated scripting and server environments.
* **`echo_cli/parser.py`** (107 lines): `argparse` configuration for subcommands: `run`, `depth`, `serve`, `export`, `status`.
* **`echo_cli/runner.py`** (173 lines): Headless execution handlers with terminal progress formatting.
* **`echo_cli/main.py`** (42 lines): CLI entrypoint.
* **`echo_cli/__main__.py`** (7 lines): `python -m echo_cli` execution entry.
* **`echo_cli/__init__.py`** (6 lines): Public exports.

### Layer 4: Desktop Shell & WebUI (`echo_desktop/`)
Native Windows frameless window integration, PyWebview JavaScript bridge, and WebUI runtime.
* **`echo_desktop/win32_types.py`** (154 lines): Ctypes structures, hit-testing geometry (`hit_test_resize`), monitor work-area queries.
* **`echo_desktop/win32_wndproc.py`** (217 lines): Subclassed window procedure, border snapping, maximize handling.
* **`echo_desktop/win32_child.py`** (96 lines): WebView2 browser-child window resizing with message loop deferral (`WM_APP`).
* **`echo_desktop/win32_drop.py`** (79 lines): Native Win32 drag-and-drop file ingestion (`DragAcceptFiles`, `WM_DROPFILES`).
* **`echo_desktop/win32_hook.py`** (222 lines): DWM cosmetic polish (rounded corners, caption tint) and hook verification.
* **`echo_desktop/bridge.py`** (298 lines): PyWebview JS bridge class (`Bridge`) connecting WebUI to `echo_service`.
* **`echo_desktop/bridge_helpers.py`** (101 lines): Dialog prompts and target file opening routines.
* **`echo_desktop/launcher.py`** (88 lines): PyWebview window creation, DPI awareness, and event hooks.
* **`echo_desktop/tk_gui/`** (Subpackage): Fallback Tkinter interface broken into micro-modules (`theme.py`, `components.py`, `handlers.py`, `runner.py`), all <= 165 lines.
* **`echo_desktop/__init__.py`** (21 lines): Desktop module entrypoints.

### Root Facades (Binary Compatibility)
* **`app.py`** (112 lines): Re-exports legacy functions (`run_pipeline`, `list_previous_jobs`, `JOBS_ROOT`). Dispatches CLI arguments if provided, or launches desktop.
* **`web_app.py`** (77 lines): PyInstaller `EchoScene.exe` entrypoint facade delegating directly to `echo_desktop.launch_desktop()`.

---

## 3. WebUI & Presentation Layer Invariants
The WebUI (`app/web_ui/`) is pre-tuned for high performance and visual clarity:
* **Option 1 Ultra-Clear Frosted Glass**: Configured with `--frost-veil: 0.08` and `--frost-blur: 1.0px` to maximize background particle visibility while maintaining text contrast.
* **GPU 60 FPS Hardware Acceleration**: Hardware compositing enabled via `translateZ(0)` and `will-change: transform`.
* **Particle Physics**: 180px mouse ripple interaction and dynamic particle animation loop with frame capping.
* Agents modifying the WebUI must **never** degrade these visual parameters.

---

## 4. Verification & Testing
Before committing any changes, agents must execute the automated test suite:
```bash
python tests/test_modern_architecture.py
```
This verifies:
1. Micro-modular line count constraints (every file <= 300 lines).
2. All 4 layers import with zero warnings or missing dependencies.
3. CLI commands and REST endpoints respond correctly.
4. Backward compatibility facades retain all legacy symbols.
