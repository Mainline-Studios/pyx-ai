; Pyx 1.5 — Inno Setup script
; Produces Pyx-<version>-setup.exe
;
; Build:  iscc /DPyxVersion=1.5.0 packaging\windows\pyx.iss
; Inno Setup: https://jrsoftware.org/isinfo.php

#ifndef PyxVersion
  #define PyxVersion "1.5.0"
#endif

[Setup]
AppId={{7B4BB3C9-96F1-4A59-9B4B-B0F5D1A2E6C8}}
AppName=PYX.
AppVersion={#PyxVersion}
AppVerName=PYX. {#PyxVersion}
AppPublisher=Mainline Studios
AppPublisherURL=https://github.com/Mainline-Studios/pyx-ai
DefaultDirName={autopf}\Pyx
DefaultGroupName=Pyx
UninstallDisplayIcon={app}\Pyx.exe
OutputDir=..\..\dist
OutputBaseFilename=Pyx-{#PyxVersion}-setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64
MinVersion=10.0.17763
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
DisableProgramGroupPage=auto

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
; Expect pyinstaller output at dist\Pyx\ before running iscc.
Source: "..\..\dist\Pyx\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\Pyx";          Filename: "{app}\Pyx.exe"
Name: "{autodesktop}\Pyx";    Filename: "{app}\Pyx.exe"; Tasks: desktopicon
Name: "{group}\Uninstall Pyx"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\Pyx.exe"; Description: "Launch Pyx now"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
