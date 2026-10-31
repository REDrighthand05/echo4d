; ECHO Scene -- THE single source of the product version for the
; installer. Bump this one file per release (or let the Windows build
; workflow override it from the release tag vX.Y.Z via /DAppVersion).
; EchoScene.iss includes this file only when AppVersion is not already
; defined on the ISCC command line.
#define AppVersion "0.4.0"
#define AppVersionInfo "0.4.0.0"
