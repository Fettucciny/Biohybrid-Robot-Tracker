; Inno Setup script -- wraps dist\robotrack into a single distributable .exe.
;
; Build with:  ISCC.exe launcher\installer.iss     (build_exe.ps1 does this)
; Output:      dist\robotrack-setup.exe
;
; This is the right way to get "one file to hand around" for a bundle this
; large. PyInstaller's onefile mode would also produce a single .exe, but it
; re-extracts several gigabytes to a temp folder on every launch; an installer
; unpacks once and then starts instantly forever after.

; build_exe.ps1 passes /DAppVersion=<version>, read from robotrack/__init__.py, so
; the installer, the running application and the update manifest cannot disagree.
; The fallback below applies only when ISCC is invoked by hand.
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

#define AppName      "robotrack"
#define AppPublisher "BioHybrid Lab"
#define AppExe       "robotrack.exe"

[Setup]
AppId={{7C4B2E10-9D3A-4F51-9C2E-6A1B0D5E8F42}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
OutputDir=..\dist
OutputBaseFilename=robotrack-setup
SetupIconFile=robotrack.ico
UninstallDisplayIcon={app}\{#AppExe}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; Per-user install by default: no admin prompt, which matters on managed
; university machines where lab members are not local administrators.
PrivilegesRequiredOverridesAllowed=dialog
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes
VersionInfoVersion={#AppVersion}
UninstallDisplayName={#AppName} {#AppVersion}
; The in-app updater runs this installer with /VERYSILENT over an existing
; install while the old copy is still shutting down. Letting Inno close and
; count the running instance avoids "file in use" failures mid-upgrade.
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "..\dist\robotrack\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}";        Filename: "{app}\{#AppExe}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}";  Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
