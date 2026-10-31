# PyInstaller spec for the ECHO scene generator (Windows, onedir).
from PyInstaller.utils.hooks import collect_all

datas = [("viewer.html", "."),
         # Ship the MEDIUM (Base) depth model (user decision 2026-10-31):
         # stage_depth prefers Base when present and only falls back to
         # Small when it is missing. The workflow downloads this exact
         # file (hash-verified) into models/ before PyInstaller runs.
         ("models/depth_anything_v2_base.onnx", "models"),
         ("assets/markers_sheet.pdf", "assets"),
         # stage_extract shells out to ffmpeg/ffprobe; ship static Windows
         # builds in the bundle. pipeline._tool_path resolves them at
         # <bundle>/assets/ffmpeg/ when frozen, PATH otherwise.
         ("assets/ffmpeg/ffmpeg.exe", "assets/ffmpeg"),
         ("assets/ffmpeg/ffprobe.exe", "assets/ffmpeg")]
binaries = []
hiddenimports = []
for pkg in ("pycolmap", "onnxruntime"):
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
