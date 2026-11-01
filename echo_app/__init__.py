"""Desktop-shell support for the ECHO scene generator (v0.4).

Everything in this package is EDGE code: it makes the app behave like
a mature Windows citizen (system tray, completion toasts, update
checks, settings, autostart) and holds the UI's action/state
discipline. None of it touches the reconstruction pipeline
(echo_scene.*) -- the pipeline's business code stays exactly as it
was, and every subsystem here is optional and failure-contained: if
the tray, the toast or the update check breaks, the app and any
running job carry on without it.

Modules:
  version   -- the app version, read from installer/version.inc
               (the single source of truth the installer also uses)
  settings  -- persisted user settings (JSON under the user's app
               data, NEVER the install dir)
  autostart -- per-user 开机自启 (HKCU Run key / XDG autostart)
  updater   -- 检查更新 against GitHub Releases; downloads the new
               EchoScene-Setup.exe, which upgrades in place
  notify    -- Windows completion toast (winotify), silent no-op
               everywhere it cannot work
  tray      -- system-tray residence via pystray, fully guarded
  appstate  -- the one explicit job/app state every action's
               enabled-state is derived from (state x action matrix)
"""
