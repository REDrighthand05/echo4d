# ECHO Scene — Windows installer (Inno Setup)

The primary Windows distribution is a single `EchoScene-Setup.exe`,
built by Inno Setup 6 from the PyInstaller one-folder output. The old
`EchoScene-windows.zip` stays as a portable alternative. This route
exists so the app behaves like proper Windows software end to end —
install, upgrade, uninstall — with no admin rights and no litter.

## How a release produces the setup.exe

1. The release snapshot (Git Data API upload to branch
   `product/echo-scene`) must include, alongside everything it already
   carries:
   - `installer/EchoScene.iss`
   - `installer/version.inc`
   - `installer/echo.ico`
   - `.github/workflows/build-windows.yml` (already in the list)
2. `build-windows.yml` runs as before (deps → ffmpeg → hash-verified
   Base model → PyInstaller → open3d/scipy bundle guards → zip +
   artifact), then additionally:
   - **Resolve installer version** — from the release tag `vX.Y.Z` if
     the build runs from one, otherwise from `installer/version.inc`
     (the single source of the version; bump it once per release).
   - **Install Inno Setup** via Chocolatey.
   - **Build installer** — `ISCC.exe /DAppVersion=<ver>
     installer\EchoScene.iss`, compiling `dist\EchoScene\*` into
     `EchoScene-Setup.exe` (LZMA2, solid).
   - Uploads it as a second artifact, `EchoScene-setup`.
3. The per-release attach step (the `release-attach.yml` regenerated
   in `~/workspace/echo4d_supervisor/` for each release) must, in
   addition to the zip, download the `EchoScene-setup` artifact and
   `gh release upload EchoScene-Setup.exe` so the setup sits on the
   Release page alongside the zip.

## The closed loop, as the user experiences it

- **Install** — double-click `EchoScene-Setup.exe`. Per-user install
  into `%LOCALAPPDATA%\Programs\ECHO Scene`; no UAC/admin prompt.
  Start-menu entry always; desktop shortcut only if ticked. Optional
  "launch after install".
- **Upgrade** — run a newer `EchoScene-Setup.exe`. The permanent AppId
  GUID in the script makes Windows treat it as the same app: files are
  replaced in place (a running app is closed via Restart Manager),
  there is one Settings entry, and job data is untouched.
- **Uninstall** — from Windows Settings → Apps, with the ECHO icon.
  Removes exactly what Setup installed under the install folder.
  **Job data is never deleted**: the app stores all jobs in
  `%USERPROFILE%\ECHO Jobs` (`app.py` `JOBS_ROOT`), outside the
  install tree, and the installer script deliberately contains no
  reference to that folder — no files entry, no uninstall-delete, no
  code hook. Keeping the records on uninstall is a deliberate product
  decision: the jobs are the user's own life records.

Verified while preparing this: the app never writes into its install
(bundle) directory at runtime — every pipeline write lands inside the
job folder (frames, depth, points, scene/replay HTML, logs); the model
and viewer template inside the bundle are read-only. So the per-user,
no-admin install is clean as-is; no app change was needed.

## The one remaining gap: code signing

`EchoScene-Setup.exe` (and the zip's exe) ship **unsigned**. Windows
SmartScreen will warn on first run ("Windows protected your PC" →
"More info" → "Run anyway"), and browsers flag the download. Nothing
in this folder fixes that: signing requires a code-signing identity —
an OV/EV certificate or Azure Trusted Signing — which is a
purchase/identity-verification step only the user can do. Once a
certificate exists, signing is a small addition: a `SignTool`
definition in `EchoScene.iss` plus signing the PyInstaller exe before
packaging, and this README should be updated. Until then, expect the
SmartScreen prompt and say so on the release page.

## Notes / not yet verified

- There is no Inno Setup compiler on the Linux dev machine: the
  script was written and reviewed against Inno Setup 6 conventions
  but has not been compiled. The first Windows CI run is the real
  syntax check; if `chinesesimplified` is unavailable on the runner's
  Inno install, drop that one `[Languages]` line (English then
  remains) — it is the only line with an external file dependency.
- `installer/echo.ico` is the app icon used by Setup, the shortcuts
  and the Settings entry. The exe itself still carries PyInstaller's
  default icon (the spec sets none); giving the exe the same icon is
  a one-line `icon=` addition to `echo-scene.spec`, deliberately left
  out of this change set.
