; EchoScene.iss -- Inno Setup 6 script for the ECHO Scene Windows app.
;
; Design contract (user directive 2026-10-31: a proper Windows
; experience loop -- clean install, clean upgrade, clean uninstall):
;   * Per-user install, NO admin prompt: PrivilegesRequired=lowest and
;     the default directory is {localappdata}\Programs\ECHO Scene
;     (the same pattern VS Code and other per-user apps use).
;   * Upgrade in place: the AppId GUID below is PERMANENT. Never change
;     it -- Windows/Inno identify "the same app" by this GUID, so a new
;     version installed over an old one replaces it instead of leaving
;     two entries in Settings.
;   * Uninstall removes only what Setup installed under {app}.
;     USER JOB DATA IS NEVER TOUCHED: the app keeps every job in
;     %USERPROFILE%\ECHO Jobs (app.py: JOBS_ROOT = Path.home() /
;     "ECHO Jobs"), which is outside {app} and is deliberately not
;     referenced anywhere in this script -- no [Files], no
;     [UninstallDelete], no [Code] deletion. Uninstalling or upgrading
;     can therefore never destroy a user's recorded scenes. This is a
;     deliberate decision, not an omission: those jobs are the user's
;     own life records, and no uninstall checkbox gets to gamble them.
;   * Payload = the PyInstaller one-folder output (dist\EchoScene),
;     produced by .github\workflows\build-windows.yml before this
;     script is compiled. Paths here are relative to this script's
;     directory (installer\).
;
; Version: single source is installer\version.inc, unless the build
; passes /DAppVersion=x.y.z (the workflow does, from the release tag).

#ifndef AppVersion
#include "version.inc"
#endif

#define AppName "ECHO Scene"
#define AppExeName "EchoScene.exe"
#define AppPublisher "ECHO"
#define AppURL "https://github.com/REDrighthand05/echo4d/releases"

[Setup]
; The doubled opening brace is Inno syntax for a literal '{'.
AppId={{0F1EF389-1B7B-44A8-97B7-63BD1AECCC24}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}
VersionInfoVersion={#AppVersionInfo}
VersionInfoCompany={#AppPublisher}
VersionInfoDescription={#AppName} Setup
VersionInfoProductName={#AppName}
DefaultDirName={localappdata}\Programs\ECHO Scene
DefaultGroupName={#AppName}
PrivilegesRequired=lowest
MinVersion=10.0
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\
OutputBaseFilename=EchoScene-Setup
SetupIconFile=echo.ico
UninstallDisplayName={#AppName}
UninstallDisplayIcon={app}\echo.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
UsePreviousAppDir=yes
SetupLogging=yes
ChangesEnvironment=no
ChangesAssociations=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; The whole PyInstaller folder, recreated under {app}. ignoreversion:
; bundle files carry no reliable version info, and an upgrade must be
; able to overwrite every one of them.
Source: "..\dist\EchoScene\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; App icon, installed so shortcuts and the Settings uninstall entry
; can point at it (the exe itself still carries the PyInstaller icon).
Source: "echo.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\echo.ico"
Name: "{userdesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\echo.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

; Deliberately NO [UninstallDelete] and NO [Code] section: Setup's own
; uninstall log already removes exactly what was installed under
; {app}, and nothing outside {app} -- in particular nothing in
; %USERPROFILE%\ECHO Jobs -- may ever be deleted by install/uninstall.
