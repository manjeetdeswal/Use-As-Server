; Script generated for Use As Server
#define MyAppName "Use As Server"
; The build workflow passes the version (ISCC /DMyAppVersion=2.1); this is the local default.
#ifndef MyAppVersion
  #define MyAppVersion "2.0"
#endif
#define MyAppPublisher "Jeet Studio"
#define MyAppExeName "UseAsServer.exe"

[Setup]
AppId={{A1B2C3D4-E5F6-7890-1234-56789ABCDEF0}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DisableProgramGroupPage=yes
WizardStyle=modern
OutputBaseFilename=UseAs_Setup_v{#MyAppVersion}
Compression=lzma
SolidCompression=yes
; Admin rights required for ViGEmBus driver
PrivilegesRequired=admin
ArchitecturesInstallIn64BitMode=x64

; Icons
; Relative to this file, so it also builds on other PCs and on GitHub
SetupIconFile=icon.ico
WizardImageFile=compiler:WizModernImage.bmp
WizardSmallImageFile=compiler:WizModernSmallImage.bmp

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
; Free virtual display driver by Amyuni Technologies (https://www.amyuni.com), used with permission
; under its license (Final_Build\usbmmidd_v2\License.txt). Lets the phone be a real extra monitor.
Name: "virtualdisplay"; Description: "Virtual monitor driver (use the phone as an extra screen)"; GroupDescription: "Drivers:"; Check: IsWin64

[Files]
; 1. The Main App
Source: "dist\UseAsServer.exe"; DestDir: "{app}"; Flags: ignoreversion

; 2. ViGEmBus Driver (Gamepad) - We KEEP this one because it is Open Source and allows bundling.
Source: "Final_Build\drivers\ViGEmBus_Setup_x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

; NOTE: We REMOVED the VB-CABLE files line because we are downloading it via browser now.

; 3. Virtual display driver (unzipped usbmmidd_v2.zip). The app adds/removes monitors with it.
Source: "Final_Build\usbmmidd_v2\*"; DestDir: "{app}\drivers\usbmmidd"; Flags: ignoreversion recursesubdirs createallsubdirs; Tasks: virtualdisplay

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon; IconFilename: "{app}\{#MyAppExeName}"

[Run]
; --- Step 0: Firewall rules (the installer runs as admin; the app usually doesn't, so it can't add them) ---
; Delete first so reinstalling/upgrading never stacks duplicate rules.
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall delete rule name=""UseAsServer_TCP"""; Flags: runhidden waituntilterminated; StatusMsg: "Configuring Windows Firewall..."
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall delete rule name=""UseAsServer_UDP"""; Flags: runhidden waituntilterminated
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall add rule name=""UseAsServer_TCP"" dir=in action=allow program=""{app}\{#MyAppExeName}"" enable=yes profile=any protocol=TCP"; Flags: runhidden waituntilterminated
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall add rule name=""UseAsServer_UDP"" dir=in action=allow program=""{app}\{#MyAppExeName}"" enable=yes profile=any protocol=UDP"; Flags: runhidden waituntilterminated

; --- Step 1: Install Gamepad Driver (Silent) ---
Filename: "{tmp}\ViGEmBus_Setup_x64.exe"; Parameters: "/exenoui /qn /norestart"; StatusMsg: "Installing Gamepad Driver..."; Flags: runascurrentuser waituntilterminated

; --- Step 1b: Install the virtual display driver (no monitor is added until the phone asks) ---
Filename: "{app}\drivers\usbmmidd\deviceinstaller64.exe"; Parameters: "install usbmmidd.inf usbmmidd"; WorkingDir: "{app}\drivers\usbmmidd"; StatusMsg: "Installing virtual monitor driver..."; Flags: runhidden waituntilterminated; Tasks: virtualdisplay

; --- Step 2: Redirect to Audio Driver Website ---
; This opens the browser. We cannot wait for it to finish because it's a browser window.
; The user must download and install it manually.
Filename: "https://vb-audio.com/Cable/"; Description: "Open VB-CABLE Website (Required for Audio)"; StatusMsg: "Opening Audio Driver Website..."; Flags: shellexec runasoriginaluser

; --- Step 3: Run the App ---
Filename: "{app}\{#MyAppExeName}"; Description: "Launch Use As Server"; Flags: nowait postinstall skipifsilent runasoriginaluser

[UninstallRun]
; Virtual monitor: switch any leftover one off, then remove the driver
Filename: "{app}\drivers\usbmmidd\deviceinstaller64.exe"; Parameters: "enableidd 0"; WorkingDir: "{app}\drivers\usbmmidd"; Flags: runhidden waituntilterminated; RunOnceId: "IddOff"; Check: IddToolExists
Filename: "{app}\drivers\usbmmidd\deviceinstaller64.exe"; Parameters: "stop usbmmidd"; WorkingDir: "{app}\drivers\usbmmidd"; Flags: runhidden waituntilterminated; RunOnceId: "IddStop"; Check: IddToolExists
Filename: "{app}\drivers\usbmmidd\deviceinstaller64.exe"; Parameters: "remove usbmmidd"; WorkingDir: "{app}\drivers\usbmmidd"; Flags: runhidden waituntilterminated; RunOnceId: "IddRemove"; Check: IddToolExists
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall delete rule name=""UseAsServer_TCP"""; Flags: runhidden; RunOnceId: "DelFwTCP"
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall delete rule name=""UseAsServer_UDP"""; Flags: runhidden; RunOnceId: "DelFwUDP"

[UninstallDelete]
; "Send to → Use As (phone)" shortcut the app creates in Explorer's right-click menu
Type: files; Name: "{userappdata}\Microsoft\Windows\SendTo\Use As (phone).lnk"

[Code]
// The uninstaller only runs the virtual-display steps when the driver was installed.
function IddToolExists: Boolean;
begin
  Result := FileExists(ExpandConstant('{app}\drivers\usbmmidd\deviceinstaller64.exe'));
end;
