; Script generated for Use As Server
#define MyAppName "Use As Server"
#define MyAppVersion "1.4"
#define MyAppPublisher "Jeet Studio"
#define MyAppExeName "UseAsServer.exe"

[Setup]
AppId={{A1B2C3D4-E5F6-7890-1234-56789ABCDEF0}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DisableProgramGroupPage=yes
WizardStyle=modern
OutputBaseFilename=UseAs_Setup_v1.4
Compression=lzma
SolidCompression=yes
; Admin rights required for ViGEmBus driver
PrivilegesRequired=admin
ArchitecturesInstallIn64BitMode=x64

; Icons
SetupIconFile=C:\Users\Manjeet\Downloads\test\icon.ico
WizardImageFile=compiler:WizModernImage.bmp
WizardSmallImageFile=compiler:WizModernSmallImage.bmp

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; 1. The Main App
Source: "Final_Build\UseAsServer.exe"; DestDir: "{app}"; Flags: ignoreversion

; 2. ViGEmBus Driver (Gamepad) - We KEEP this one because it is Open Source and allows bundling.
Source: "Final_Build\drivers\ViGEmBus_Setup_x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

; NOTE: We REMOVED the VB-CABLE files line because we are downloading it via browser now.

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon; IconFilename: "{app}\{#MyAppExeName}"

[Run]
; --- Step 1: Install Gamepad Driver (Silent) ---
Filename: "{tmp}\ViGEmBus_Setup_x64.exe"; Parameters: "/exenoui /qn /norestart"; StatusMsg: "Installing Gamepad Driver..."; Flags: runascurrentuser waituntilterminated

; --- Step 2: Redirect to Audio Driver Website ---
; This opens the browser. We cannot wait for it to finish because it's a browser window.
; The user must download and install it manually.
Filename: "https://vb-audio.com/Cable/"; Description: "Open VB-CABLE Website (Required for Audio)"; StatusMsg: "Opening Audio Driver Website..."; Flags: shellexec runasoriginaluser

; --- Step 3: Run the App ---
Filename: "{app}\{#MyAppExeName}"; Description: "Launch Use As Server"; Flags: nowait postinstall skipifsilent