"""Text of the "How to use" tab: every feature, what it needs and how to use it.

Each section is (title, [paragraphs]); a paragraph starting with "• " is a bullet.
"""

SECTIONS = [
    ("Getting connected", [
        "Install the Use As app on your phone and keep this server running (START on the Dashboard).",
        "• Same network: the phone and this PC must be on the same Wi-Fi or hotspot. Office and "
        "college networks often block devices from seeing each other; a phone hotspot always works. "
        "For the best speed and lowest lag, use a USB cable instead (USB tethering, below).",
        "• Automatic: the PC appears under \"Nearby Computers\" in the app. Tap it.",
        "• QR code: Dashboard › Pair with QR, then \"Scan QR code from PC\" in the app. This also "
        "works when a password is set.",
        "• By address: type the address shown on the Dashboard (ws://…:port) in the app.",
        "• Android 17: allow \"Nearby devices / local network\" when the app asks, or no PCs are found.",
        "• Password: Settings › Security. Phones then need the password (or the QR code) once.",
        "Not working? Open the Setup tab: it checks the firewall, network and drivers.",
    ]),
    ("My computers, auto-connect and Wake-on-LAN", [
        "The app remembers every PC you connect to, each with its own password, under \"My computers\" "
        "on its connection screen. Tap one to connect; a green dot means it's online right now.",
        "• Auto-connect (switch next to \"My computers\"): opening the app reconnects to the PC used last, "
        "even if its address changed.",
        "• Wake-on-LAN: the power button next to an offline PC turns it on. It must be enabled in the PC's "
        "BIOS/UEFI and network adapter settings (\"Wake on Magic Packet\"), and works best over Ethernet.",
    ]),
    ("Best speed: USB tethering", [
        "For the lowest lag and the fastest transfers, connect the phone with a USB cable and turn on "
        "USB tethering. Everything (second screen, gamepad, sound, files) then runs over the cable "
        "instead of Wi-Fi: no dropouts, much less delay, and it works where Wi-Fi blocks devices.",
        "• 1. Plug the phone into the PC with a USB cable that carries data (not a charge-only cable).",
        "• 2. On the phone: Settings › Network & internet › Hotspot & tethering › USB tethering: on.",
        "• 3. In the app the PC shows up again labelled \"USB Tethering\" (an address like 192.168.42.x). "
        "Tap that one. You can also type the address from the Dashboard.",
        "• 4. If Windows asks what kind of network it is, choose Private. If the phone can't connect, "
        "open the Setup tab and use the Firewall fix.",
        "• If the phone is on mobile data, the PC may also use it for its internet. Keep the phone on "
        "Wi-Fi (or turn mobile data off) to avoid using your data allowance.",
        "• Bonus: the phone charges while you use it.",
    ]),
    ("Mouse & touchpad", [
        "Phone tab: Mouse.",
        "• Move: slide one finger. Tap: left click. Tap the bottom-right corner: right click.",
        "• Drag: hold one finger still for half a second (the phone vibrates), then move.",
        "• Two fingers: slide to scroll, pinch to zoom, tap to right-click.",
        "• Three fingers: tap for a middle click; swipe up for Task view, down for the desktop, "
        "left or right to switch apps.",
        "• Edge scroll: slide along the right edge (Touchpad settings).",
        "• Gyro mouse: point the phone like a laser pointer.",
        "• Macros: auto-clicker, keep-awake jiggle and key repeats, with start/stop.",
    ]),
    ("Keyboard", [
        "Phone tab: Keyboard. Type in the box and the text appears on the PC, in any language. "
        "Your phone keyboard's microphone button gives you voice typing on the PC.",
        "• Shortcut keys: media keys, Win+D, Alt+Tab, Print Screen… plus your own layouts of keys "
        "and combos (e.g. Ctrl+Shift+S).",
    ]),
    ("Gamepad", [
        "Phone tab: Gamepad. Needs the ViGEmBus driver on Windows (Setup tab); games then see an "
        "Xbox controller. Without it, buttons are sent as keyboard keys.",
        "• Each phone is its own controller (P1–P4 badge). Rumble, tilt steering and custom "
        "layouts are in the gamepad settings.",
    ]),
    ("Webcam", [
        "Phone tab: Camera. Needs OBS Studio installed (Setup tab). In Zoom, Teams, Discord or OBS "
        "pick \"OBS Virtual Camera\".",
        "• Camera Studio on this PC: background blur or a picture behind you, mirror, flip, brightness.",
    ]),
    ("Microphone", [
        "Phone tab: Microphone. Needs VB-CABLE (Setup tab). In your apps choose \"CABLE Output\" "
        "as the microphone. Noise suppression and auto gain are in the app.",
    ]),
    ("PC sound on the phone", [
        "Phone tab: Speaker, turn the switch on.",
        "• Hearing it twice? The sound is copied, not moved: mute the PC (taskbar speaker icon or "
        "the mute key) and the phone keeps playing.",
        "• Lag: set the buffer to Low. Crackling on weak Wi-Fi: Stable.",
    ]),
    ("Now playing & media controls", [
        "The phone shows the song or video playing on the PC (title, artist, cover) with "
        "previous / play-pause / next, in the Speaker tab and as a notification, so you can "
        "control the PC without opening the app. Works with Spotify, YouTube in the browser, "
        "the Media Player app and more.",
        "• Song details need the extra listed in the Setup tab; the buttons work either way.",
    ]),
    ("Second screen", [
        "Phone tab: Display. Pick a monitor and press start.",
        "• Extra monitor: pick a monitor this PC doesn't have (Monitor 2 on a one-screen PC) and a "
        "virtual monitor is added. Drag windows onto it like a real screen. It's removed about 20 "
        "seconds after the phone stops viewing. Turn this off in Settings › Second screen.",
        "• The virtual monitor gets the phone's own shape (e.g. 2400x1080), so Fit fills the whole phone. "
        "Windows asks once for permission to add that resolution.",
        "• Prefer another size? Display tab › Virtual monitor size: 16:9, 16:10, 4:3, 21:9 presets or "
        "your own width × height (Windows asks for permission again when it changes).",
        "• Fit shows the whole screen, Fill uses the whole phone, Stretch fills it unevenly.",
        "• 720p/1080p, 30/60 fps and quality are in the app. With the codecs from the Setup tab "
        "the picture is smoother and uses far less Wi-Fi.",
        "• For 1080p at 60 fps, or if the picture stutters, use USB tethering (see \"Best speed\" above).",
    ]),
    ("Presentations", [
        "Phone tab: Present. Big Next / Previous buttons and the phone's volume keys change slides "
        "(PowerPoint, Google Slides, PDF viewers). Start the show, black out the screen, use the "
        "phone as a laser pointer, and keep time with a timer that vibrates near the end.",
    ]),
    ("PC tab: power, volume, apps, files", [
        "Phone tab: PC.",
        "• Quick controls: PC volume (the slider icon opens a mixer with each app's own volume), screen "
        "brightness (laptop screens and monitors with DDC/CI), and buttons for Files, Windows and Open app. "
        "Apps you pin show up there for one-tap launching.",
        "• Power: Lock, Screen off, Sleep, Hibernate, Restart, Shut down, Sign out. Everything except Lock and "
        "Screen off asks first; Restart and Shut down give you 15 seconds to cancel. Timer: \"sleep in 1 hour\" "
        "and the like (cancel any time from the PC tab). Hibernate shows up only when it's switched on in "
        "Windows (powercfg /hibernate on). Settings › Phone access can turn power controls off.",
        "• Windows: live pictures of the open windows; tap one to bring it to the front, or minimise, "
        "maximise or close it.",
        "• Open app: everything in the Start menu, Microsoft Store apps included. Tap the star to pin one.",
        "• Files: browse the PC, search and sort a folder, pin folders you use often, and see Recent files. "
        "Long-press to select several files or folders and download them together (folders arrive as "
        ".zip). ⋮ › Upload files here sends phone files straight into the folder you're looking at.",
        "• Tap a file (or ⋮ on any row) for more: download, open it on the PC, show it in its folder, "
        "copy its path, rename, or move it to the Recycle Bin. The path at the top is tappable — jump "
        "back to any folder in it; the house button goes to the start. Settings › Phone access › Rename "
        "and delete files turns the changing part off.",
        "• Downloads show their progress in the app and in the phone's notifications, with Cancel.",
        "• Settings › Phone access › Folders the phone can browse: add folders to share only those "
        "instead of the whole PC.",
    ]),
    ("Custom buttons (Deck)", [
        "Phone tab: Deck. Make your own buttons, like a Stream Deck: tap + and pick what it does.",
        "• Open an app (from the Start menu), open a file or folder, open a website, press a key "
        "combination (e.g. Ctrl+Shift+Esc), type saved text, or run a command.",
        "• Give each button a name, an icon and a colour. Long-press a button to edit, move or delete it.",
        "• Opening files and running commands can be turned off in Settings › Phone access › Custom "
        "buttons can run programs.",
    ]),
    ("PC status, photos and videos", [
        "• The phone's PC tab shows this PC's CPU, memory, graphics card (NVIDIA), disks, battery and "
        "network speed, live.",
        "• In the PC tab's file browser, tap a photo to see it full screen (swipe for the next one, pinch "
        "to zoom), or a video / song to play it on the phone straight from the PC — nothing is downloaded. "
        "This uses a second port (normally 8082); the installer's firewall rule already allows it.",
    ]),
    ("Phone calls on the PC", [
        "With phone notifications on (phone's PC tab), an incoming call pops up at the top right of the "
        "PC: who's calling, with Decline and Answer (answering happens on the phone — talk on the phone "
        "or its speaker). During a call there's a Hang up button.",
        "• Settings › Phone access › Pop up phone calls turns the pop-up off.",
    ]),
    ("Phone notifications on the PC", [
        "Turn on \"Phone notifications on the PC\" in the phone's PC tab and allow notification access when "
        "Android asks. Notifications then pop up in the corner of this screen and are listed in the "
        "Notifications tab. Messaging apps (WhatsApp, Messages, Telegram…) can be answered from the PC.",
        "• Ongoing ones (music, downloads, navigation) and notifications hidden on the lock screen aren't sent.",
        "• Settings › Phone access › Pop up phone notifications turns the pop-ups off (the list stays).",
        "• In the phone's PC tab, Choose apps keeps chosen apps' notifications (banking, OTP codes…) on the "
        "phone only.",
    ]),
    ("Shortcuts: Quick Settings tiles and widget", [
        "On the phone, edit the Quick Settings panel (the pencil icon) and add any of: Lock PC, PC play/pause, "
        "Mute PC, Touchpad, Send clipboard. On the home screen, add the \"Use As PC controls\" widget "
        "(previous / play-pause / next, mute, lock, touchpad). They work while the app is connected.",
    ]),
    ("Files", [
        "• PC to phone: Sharing tab › send a file, drag files onto this window, or right-click a "
        "file in Explorer › Send to › Use As (phone).",
        "• Phone to PC: share any file to \"Send to PC\" from another app, or use the Files tab. "
        "Received files go to Downloads\\UseAs_Received.",
    ]),
    ("Clipboard", [
        "Sharing tab › Auto-sync: text, screenshots and copied files go from the PC to the phone "
        "automatically.",
        "• Phone to PC: Android only lets the app on screen read the clipboard, so a copy is sent "
        "when Use As is open (with Auto-sync on in the phone's Files tab). From any other app, tap "
        "\"Send clipboard to PC\" on the Use As notification or the Quick Settings tile.",
        "• Passwords from password managers are never synced automatically.",
        "• Clipboard history: the last 20 texts copied between the PC and the phone are listed in the "
        "Sharing tab (Copy / Send again) and in the phone's Files tab (History).",
    ]),
    ("Troubleshooting", [
        "• Phone can't find the PC: same Wi-Fi? Setup tab › Firewall. Try the QR code or type the address.",
        "• Lag, stutter or disconnects on Wi-Fi: use USB tethering (\"Best speed\" above).",
        "• Disconnects on screen-off: Android battery saver — set Use As to \"Unrestricted\" battery use.",
        "• Nothing happens in a game or Task Manager: Settings › Request admin rights.",
        "• Still stuck: the Activity log on the Dashboard says what went wrong.",
    ]),
]
