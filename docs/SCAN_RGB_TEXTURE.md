# Measured room scan + real RGB texture: mature Open3D provider

This delivers the user's requested high-information path: keep an existing measured-depth room surface, project calibrated real RGB onto it, and export an embedded-texture GLB. It does not re-infer geometry from RGB or add more plane-sweep filters.

The implementation calls Open3D 0.19.0 `compute_uvatlas` and `project_images_to_albedo` (CPU), then uses trimesh 4.11.1 to export GLB. MIT Open3D, MIT trimesh; no third-party source is copied. Provider reference:
https://www.open3d.org/docs/release/python_api/open3d.t.geometry.TriangleMesh.html#open3d.t.geometry.TriangleMesh.project_images_to_albedo

## Actual first execution

- Real TUM fr1_room measured TSDF surface, source Actions 40750777392; its exact retained PLY is available in Actions 40760283937 artifact 12848191246 under replay-inputs/baseline/mesh.ply.
- Geometry input SHA 65b8b25e5f8d46f494acbd1e4ab905fd7b5415e885705c104c00cee75704c741; 231610 vertices / 422714 faces. This is a previously reconstructed measured-depth scan, not a new independently surveyed laser scan.
- 24 evenly distributed original 640x480 RGB images selected from the existing 96 training identities. No held-out review image is used for texturing. Known metric camera poses and intrinsics retained.
- Local actual CPU run: UV atlas + image projection + GLB export in 58.28 seconds. Output GLB SHA 70ad7edbafeead40ab7cc37f189d1774b94108d9a677bcb13673d6a99f4dfe78, 2048x2048 albedo. UV atlas is nondeterministic, so output hashes may change on a fresh run.
- The GLB was reloaded and every exported triangle's coordinates compared against the input scan within 1e-6m. UV seams duplicate vertex indices, not physical geometry. There is no hole fill, mesh optimization or estimated camera motion.
- Five actual CPU raycast views (three frozen original camera poses and two shifted free cameras) were generated. Desk and monitor are recognizable; dark texture patches, seams, scan noise and holes remain. This is a working texture integration, not a claim that scan geometry is perfect or that old depth errors disappeared.

## Use

    pip install open3d==0.19.0 trimesh==4.11.1 pillow numpy
    python tools/texture_scan_open3d.py --mesh scan.ply --manifest calibrated-rgb.json --output textured-room

Manifest: `intrinsics` has width/height/fx/fy/cx/cy; `frames` has relative RGB `color`, `rgb_sha256` and column-major 16-value `cameraToWorld`. RGB files remain at their real resolution. Texture tool only reads these RGB files and the supplied mesh; sensor depth maps are not required at this stage.

Run `tools/prepare_room_texture_manifest.py --replay-root <downloaded artifact>/replay-inputs` to build the 24-training-image manifest from the retained public room input. Then pass `<replay-root>/baseline/mesh.ply` and `<replay-root>/source/texture-24.json` to the command above. Output includes GLB, albedo, per-corner UV archive and a small run report.

The next product deliverable is a usable scan+video import flow around this provider, alongside a full-scene mature reconstruction baseline. Existing RGB-only experiments remain historical optional research; they are no longer the default next step for this branch.
