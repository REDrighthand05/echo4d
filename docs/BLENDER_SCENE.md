# Open in Blender

After the official complete reconstruction run finishes, download `room-full.glb` or `room-preview.glb` from its artifact and use Blender **File → Import → glTF 2.0**. The full file preserves native mesh geometry; the preview is separately decimated.

For a native Blender project, run:
```sh
blender --background --python-exit-code 1 --python tools/export_blender_scene.py -- \
  --input room-full.glb --output room.blend --provenance "Official complete RGB-D reconstruction"
```

The exporter actually imports GLB, saves .blend and reopens it. It refuses missing/empty geometry and checks mesh/triangle/material/color-layer counts after reopening. It records source and native-file hashes. It does not estimate scene quality or change geometry.

The independent compatibility workflow uses a frozen **old #112 model** as a large real fixture. Its successful result must never be described as the new #114 reconstruction. It does not rerun the reconstruction or alter its running job.

Current reconstruction input: real RGB-D (color plus measured depth). Ordinary uncalibrated RGB-only phone video is not yet a reliable product entry point.
