# PyInstaller spec for the ECHO scene generator (Windows, onedir).
from PyInstaller.utils.hooks import collect_all

# echo_scene.replay builds its route with the registered echo_super
# kernel build_replay_path; both are reached through lazy imports, so
# name them explicitly -- a missing bundle here would silently cost
# every job its replay.html (the stage is non-fatal by design).
# The echo_app.* edge modules (v0.4: tray, toast, updater, settings,
# autostart, app state) are imported lazily from app.main() for the
# same reason, so they are named explicitly too -- a missing bundle
# there would silently cost the tray/update behaviour the same way.
hiddenimports_extra = ["echo_scene.replay", "echo_scene.scale", "echo_super.ds_batch1",
                       "echo_super.ds_batch2",
                       "echo_app.version", "echo_app.settings",
                       "echo_app.autostart", "echo_app.updater",
                       "echo_app.notify", "echo_app.tray",
                       "echo_app.appstate"]

datas = [("viewer.html", "."),
         # Ship the MEDIUM (Base) depth model (user decision 2026-10-31):
         # stage_depth prefers Base when present and only falls back to
         # Small when it is missing. The workflow downloads this exact
         # file (hash-verified) into models/ before PyInstaller runs.
         ("models/depth_anything_v2_base.onnx", "models"),
         ("assets/markers_sheet.pdf", "assets"),
         # v0.4 shell resources: the app reads its version from the
         # same version.inc the installer uses, and the tray icon is
         # the installer icon. app._resource_path resolves both under
         # installer/ in the bundle, mirroring the source tree.
         ("installer/version.inc", "installer"),
         ("installer/echo.ico", "installer"),
         # stage_extract shells out to ffmpeg/ffprobe; ship static Windows
         # builds in the bundle. pipeline._tool_path resolves them at
         # <bundle>/assets/ffmpeg/ when frozen, PATH otherwise.
         ("assets/ffmpeg/ffmpeg.exe", "assets/ffmpeg"),
         ("assets/ffmpeg/ffprobe.exe", "assets/ffmpeg")]
binaries = []
hiddenimports = list(hiddenimports_extra)
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

# customtkinter (the rounded Meta-style shell) loads its theme JSON and
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

a = Analysis(
    ["app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
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
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="EchoScene",
)
