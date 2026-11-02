# PyInstaller spec for the ECHO scene generator (Windows, onedir).
#
# STANDARD FRONTEND (user decision, 2026-11-02): the web shell
# is the face carried forward into coming versions; the Tk shell is
# the fallback inside the same payload. tests/test_web_bridge.py
# guards the shell against drifting from the app's feature set.
#
# LOOSE-PAYLOAD LAYOUT (v0.4, the hot-update spine): the frozen exe
# is only launcher.py + the third-party stack (frozen_imports is
# the dependency magnet). ALL product code ships as loose files
# under app/ and is hot-updatable through the batch-5 kernel
# pipeline (manifest -> plan -> pack -> echo_app.hotapply). The
# Analysis excludes below are load-bearing: if a product module is
# ever frozen into the archive it SHADOWS its loose copy and hot
# updates silently stop applying to it.
from PyInstaller.utils.hooks import collect_all

datas = [("viewer.html", "."),
         # Ship the MEDIUM (Base) depth model (user decision 2026-10-31):
         # stage_depth prefers Base when present and only falls back to
         # Small when it is missing. The workflow downloads this exact
         # file (hash-verified) into models/ before PyInstaller runs.
         ("models/depth_anything_v2_base.onnx", "models"),
         ("assets/markers_sheet.pdf", "assets"),
         # Base resources the frozen side still resolves through
         # _MEIPASS: the tray icon.
         ("installer/echo.ico", "installer"),
         # stage_extract shells out to ffmpeg/ffprobe; ship static Windows
         # builds in the bundle. pipeline._tool_path resolves them at
         # <bundle>/assets/ffmpeg/ when frozen, PATH otherwise.
         ("assets/ffmpeg/ffmpeg.exe", "assets/ffmpeg"),
         ("assets/ffmpeg/ffprobe.exe", "assets/ffmpeg"),
         # ---- the loose, hot-updatable payload (app/**) ----
         ("echo_scene", "app/echo_scene"),
         ("echo_app", "app/echo_app"),
         ("echo_super", "app/echo_super"),
         ("web_ui", "app/web_ui"),
         ("app.py", "app"),
         ("web_app.py", "app"),
         # The version stamp lives in the payload so a hot update
         # flips it (last, per the apply-order kernel) and
         # echo_app.version picks the new value up on next launch.
         ("installer/version.inc", "app/installer")]
binaries = []
hiddenimports = []
# open3d is a HARD dependency of the mesh stage (echo_scene/mesh.py
# Poisson reconstruction); it imports lazily behind a guard, so a
# missing bundle does not fail the build -- v0.1 shipped without it
# and died on the user's machine at the mesh step. It must be
# collected here, and requirements-app.txt must install it, or the
# freeze is broken by construction. scipy joined the list in v0.2:
# the continuity-cleanup stage (echo_scene/clean.py) runs its
# connected components, also behind a lazy import.
for pkg in ("pycolmap", "onnxruntime", "open3d", "scipy"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# tkinterdnd2 ships the tkdnd native libraries as package data; the app
# degrades to click-to-pick without it, so a missing package must not
# break the build.
try:
    d, b, h = collect_all("tkinterdnd2")
    datas += d
    binaries += b
    hiddenimports += h
except Exception:
    pass

# customtkinter (the fallback shell) loads its theme JSON and
# image assets from package data at runtime, and pulls in darkdetect;
# both must ride along or the frozen window will not build.
for pkg in ("customtkinter", "darkdetect"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

# v0.4 tray + toast. pystray is pure ctypes on Windows; collect it so
# the frozen app keeps its tray (guarded like tkinterdnd2: a missing
# package must degrade the tray, never break the build). winotify is
# pure Python too but imports winreg at module level, so it can only
# ever be collected on the Windows build itself; the guarded collect
# keeps this spec buildable on any platform.
for pkg in ("pystray", "winotify"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

# pywebview (the web shell's window) ships its JS bridge assets as
# package data; collect it guarded, like the tray packages: a
# missing collect must fail loudly at window creation, not silently.
try:
    d, b, h = collect_all("webview")
    datas += d
    binaries += b
    hiddenimports += h
except Exception:
    pass

a = Analysis(
    ["launcher.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # Load-bearing: the product's own modules must stay OUT of the
    # frozen archive so the loose payload under app/ is what runs
    # (and what hot updates replace). See the header.
    excludes=["web_app", "app", "echo_app", "echo_scene", "echo_super"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="EchoScene",
    console=False,
    # Contents beside the exe, NOT under _internal/: the loose app/
    # payload must sit at <install>/app so the hot-update wrench can
    # address it as "app/..." and users' manifests stay layout-stable.
    contents_directory=".",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="EchoScene",
)
