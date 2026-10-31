"""Desktop window (CustomTkinter) shared by Windows, Linux and macOS."""

import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import pyperclip
import pystray
from PIL import Image, ImageDraw
from pystray import MenuItem as item

from . import pairing
from .common import APP_VERSION, GITHUB_REPO, SAVE_DIR, load_prefs, resource_path, save_prefs
from .core import UnifiedRemoteServer
from .ipc import IpcServer
from .platforms import get_platform

# Drag-and-drop needs the optional tkinterdnd2 package (pip install tkinterdnd2).
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    class _DnDRoot(ctk.CTk, TkinterDnD.DnDWrapper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.TkdndVersion = TkinterDnD._require(self)
except Exception:
    DND_FILES = None
    _DnDRoot = None


class QrPairDialog(ctk.CTkToplevel):
    """Shows the pairing QR code. Scanning it in the Use As app (or with the phone's
    camera) connects and pairs without typing an address or password."""

    def __init__(self, gui):
        super().__init__(gui.root)
        self.gui = gui
        self.title("Pair a phone")
        self.geometry("420x600")
        self.resizable(False, False)
        self.configure(fg_color="#141414")
        self.transient(gui.root)

        ctk.CTkLabel(self, text="SCAN TO CONNECT", font=("Courier New", 16, "bold"),
                     text_color="#00e676").pack(pady=(20, 4))
        ctk.CTkLabel(self, text="Use As app → Scan QR code, or your phone's camera",
                     font=("Segoe UI", 12), text_color="#999").pack()
        self.lbl_qr = ctk.CTkLabel(self, text="", width=320, height=320)
        self.lbl_qr.pack(pady=16)
        self.lbl_info = ctk.CTkLabel(self, text="", font=("Courier New", 11), text_color="#bbb",
                                     justify="left", wraplength=380)
        self.lbl_info.pack(padx=20)
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=20, pady=16, side="bottom")
        gui._pill_btn(row, "New code", self.new_code).pack(side="left", expand=True, fill="x", padx=(0, 6))
        gui._pill_btn(row, "Close", self.destroy, accent=True).pack(side="right", expand=True, fill="x")
        self.refresh()

    def new_code(self):
        if messagebox.askyesno("New pairing code",
                               "Phones that paired with the old QR code will need to scan the new one "
                               "(if a password is required). Continue?", parent=self):
            self.gui.regenerate_pairing_token()
            self.refresh()

    def refresh(self):
        g = self.gui
        ips = pairing.ordered_ips(g._endpoints)
        port = g.server.port if g.server else g.prefs.get("port", 8080)
        if not g.is_running or not ips:
            self.lbl_qr.configure(image=None, text="Start the server first\n(and connect to a network).")
            self.lbl_info.configure(text="")
            return
        data = pairing.qr_payload(ips, port, g.prefs["pairing_token"])
        img = pairing.qr_image(data)
        if img is None:
            self.lbl_qr.configure(image=None, text="QR codes need the 'qrcode' package:\npip install qrcode")
        else:
            self._img = ctk.CTkImage(light_image=img, dark_image=img, size=(320, 320))
            self.lbl_qr.configure(image=self._img, text="")
        lines = [f"Port {port}"] + [f"• {ip}" for ip in ips]
        self.lbl_info.configure(text="\n".join(lines))


class SettingsDialog(ctk.CTkToplevel):
    def __init__(self, parent, prefs, callback_save):
        super().__init__(parent)
        self.title("Settings")
        self.geometry("460x640")
        self.resizable(False, False)
        self.prefs = prefs
        self.callback_save = callback_save
        self.configure(fg_color="#141414")
        self.transient(parent)
        self.grab_set()
 
        # Header
        header = ctk.CTkFrame(self, fg_color="#1a1a1a", corner_radius=0, height=64)
        header.pack(fill="x")
        header.pack_propagate(False)
        ctk.CTkLabel(header, text="SETTINGS", font=("Courier New", 18, "bold"),
                      text_color="#00e676").pack(side="left", padx=24, pady=20)
        ctk.CTkLabel(header, text=f"v{APP_VERSION}", font=("Courier New", 11),
                      text_color="#555").pack(side="right", padx=24)
 
        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent", scrollbar_button_color="#2a2a2a",
                                         scrollbar_button_hover_color="#00e676")
        scroll.pack(fill="both", expand=True, padx=16, pady=(12, 0))
 
        def section(parent, title):
            f = ctk.CTkFrame(parent, fg_color="#1c1c1c", corner_radius=10,
                              border_width=1, border_color="#2a2a2a")
            f.pack(fill="x", pady=(0, 12))
            ctk.CTkLabel(f, text=title, font=("Courier New", 11, "bold"),
                          text_color="#00e676").pack(anchor="w", padx=16, pady=(12, 6))
            sep = ctk.CTkFrame(f, height=1, fg_color="#2a2a2a")
            sep.pack(fill="x", padx=16, pady=(0, 10))
            return f
 
        def toggle_row(parent, text, hint, var):
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", padx=16, pady=4)
            left = ctk.CTkFrame(row, fg_color="transparent")
            left.pack(side="left", fill="x", expand=True)
            ctk.CTkLabel(left, text=text, font=("Segoe UI", 13), text_color="#e0e0e0",
                          anchor="w").pack(anchor="w")
            if hint:
                ctk.CTkLabel(left, text=hint, font=("Segoe UI", 10), text_color="#555",
                              anchor="w").pack(anchor="w")
            ctk.CTkSwitch(row, text="", variable=var, width=48,
                           button_color="#00e676", button_hover_color="#00c853",
                           progress_color="#00e676").pack(side="right")
 
        # General
        sec1 = section(scroll, "  GENERAL")
        self.platform = get_platform()
        self.var_autostart_pc = ctk.BooleanVar(value=prefs.get("autostart_pc", False))
        toggle_row(sec1, f"Auto-start with {self.platform.display_name}", "Launch on login", self.var_autostart_pc)
        self.var_autostart_server = ctk.BooleanVar(value=prefs.get("autostart_server", True))
        toggle_row(sec1, "Auto-start server", "Start server immediately on launch", self.var_autostart_server)
        self.var_admin = ctk.BooleanVar(value=prefs.get("run_as_admin", False))
        if self.platform.supports_admin():
            toggle_row(sec1, "Request admin rights", "Required for some input features", self.var_admin)

        scale_row = ctk.CTkFrame(sec1, fg_color="transparent")
        scale_row.pack(fill="x", padx=16, pady=4)
        ctk.CTkLabel(scale_row, text="Interface scale", font=("Segoe UI", 13),
                     text_color="#e0e0e0").pack(side="left")
        self.opt_scale = ctk.CTkOptionMenu(scale_row, width=90,
                                           values=["1.0", "1.2", "1.4", "1.5", "1.6", "1.8", "2.0"])
        self.opt_scale.set(str(prefs.get("ui_scale", self.platform.default_ui_scale)))
        self.opt_scale.pack(side="right")
        ctk.CTkFrame(sec1, height=8, fg_color="transparent").pack()
 
        # Gaming
        sec2 = section(scroll, "  INPUT & GAMING")
        self.var_send_to = ctk.BooleanVar(value=prefs.get("send_to_menu", True))
        self.var_gaming_mode = ctk.BooleanVar(value=prefs.get("gaming_mode", True))
        toggle_row(sec2, "Gaming mode", "Micro-delays so games register input", self.var_gaming_mode)
        if self.platform.send_to_label:
            toggle_row(sec2, "Send to phone in right-click menu", self.platform.send_to_label, self.var_send_to)
        ctk.CTkFrame(sec2, height=8, fg_color="transparent").pack()

        # Phone access
        sec_pa = section(scroll, "  PHONE ACCESS")
        self.var_file_browsing = ctk.BooleanVar(value=prefs.get("allow_file_browsing", True))
        toggle_row(sec_pa, "Browse this PC's files", "The phone's PC tab can list and download files",
                   self.var_file_browsing)
        self.var_file_changes = ctk.BooleanVar(value=prefs.get("allow_file_changes", True))
        toggle_row(sec_pa, "Rename and delete files", "Deleted files go to the Recycle Bin, never gone for good",
                   self.var_file_changes)

        # Which folders the phone may browse (empty = the whole PC)
        self._browse_folders = [str(f) for f in prefs.get("browse_folders", []) if f]
        ctk.CTkLabel(sec_pa, text="Folders the phone can browse", font=("Segoe UI", 13),
                     text_color="#e0e0e0", anchor="w").pack(anchor="w", padx=16, pady=(8, 0))
        ctk.CTkLabel(sec_pa, text="None added = the whole PC. Add folders to share only those.",
                     font=("Segoe UI", 10), text_color="#555", anchor="w").pack(anchor="w", padx=16)
        self._folders_box = ctk.CTkFrame(sec_pa, fg_color="#151515", corner_radius=8)
        self._folders_box.pack(fill="x", padx=16, pady=(6, 4))
        ctk.CTkButton(sec_pa, text="+  Add folder", height=30, font=("Segoe UI", 12),
                      fg_color="#1e1e1e", hover_color="#2a2a2a", text_color="#00e676",
                      command=self._add_browse_folder).pack(anchor="w", padx=16, pady=(2, 6))
        self._render_browse_folders()
        self.var_power_control = ctk.BooleanVar(value=prefs.get("allow_power_control", True))
        toggle_row(sec_pa, "Lock, sleep and shut down", "From the phone's PC tab (always asks first)",
                   self.var_power_control)
        self.var_notif_popups = ctk.BooleanVar(value=prefs.get("notification_popups", True))
        toggle_row(sec_pa, "Pop up phone notifications", "Small window in the corner, with quick reply",
                   self.var_notif_popups)
        self.var_call_popups = ctk.BooleanVar(value=prefs.get("call_popups", True))
        toggle_row(sec_pa, "Pop up phone calls", "Who's calling, with Decline and Answer",
                   self.var_call_popups)
        self.var_deck_commands = ctk.BooleanVar(value=prefs.get("allow_deck_commands", True))
        toggle_row(sec_pa, "Custom buttons can run programs", "The phone's Deck: open files, run commands",
                   self.var_deck_commands)
        ctk.CTkFrame(sec_pa, height=8, fg_color="transparent").pack()

        # Second screen
        self.var_virtual_display = ctk.BooleanVar(value=prefs.get("virtual_display", True))
        if "virtual_display" in self.platform.features:
            sec_vd = section(scroll, "  SECOND SCREEN")
            toggle_row(sec_vd, "Virtual monitor for the phone",
                       "Picking a monitor this PC doesn't have adds a real extra screen",
                       self.var_virtual_display)
            ctk.CTkFrame(sec_vd, height=8, fg_color="transparent").pack()
 
        # Network
        sec3 = section(scroll, "  NETWORK")
        net_row = ctk.CTkFrame(sec3, fg_color="transparent")
        net_row.pack(fill="x", padx=16, pady=(0, 12))
        ctk.CTkLabel(net_row, text="WebSocket port", font=("Segoe UI", 13), text_color="#e0e0e0").pack(side="left")
        self.ent_port = ctk.CTkEntry(net_row, width=90, font=("Courier New", 13),
                                      fg_color="#111", border_color="#2a2a2a",
                                      text_color="#00e676", justify="center")
        self.ent_port.insert(0, str(prefs.get("port", 8080)))
        self.ent_port.pack(side="right")
        
       # Security
        sec4 = section(scroll, "  SECURITY")
        self.var_require_password = ctk.BooleanVar(value=prefs.get("password_enabled", False))
        toggle_row(sec4, "Require Password", "Phone must enter this password to connect", self.var_require_password)

        ctk.CTkLabel(sec4, text="Connection Password", font=("Segoe UI", 13),
                      text_color="#e0e0e0", anchor="w").pack(anchor="w", padx=16, pady=(8, 0))
        ctk.CTkLabel(sec4, text="Changing this will require reconnecting on all devices",
                      font=("Segoe UI", 10), text_color="#555", anchor="w").pack(anchor="w", padx=16, pady=(0, 8))

        pw_row = ctk.CTkFrame(sec4, fg_color="transparent")
        pw_row.pack(fill="x", padx=16, pady=(0, 12))
        self.ent_password = ctk.CTkEntry(pw_row, font=("Courier New", 13),
                                          fg_color="#111", border_color="#2a2a2a",
                                          text_color="#00e676", show="•")
        self.ent_password.insert(0, prefs.get("password", "useas"))
        self.ent_password.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self._pw_visible = False
        def toggle_pw_visibility():
            self._pw_visible = not self._pw_visible
            self.ent_password.configure(show="" if self._pw_visible else "•")
            btn_show.configure(text="🙈" if self._pw_visible else "👁")

        btn_show = ctk.CTkButton(pw_row, text="👁", width=36, height=28,
                                  fg_color="#111", hover_color="#1e1e1e",
                                  text_color="#e0e0e0",
                                  command=toggle_pw_visibility)
        btn_show.pack(side="right")

        # Save
        ctk.CTkButton(self, text="SAVE CHANGES", font=("Courier New", 13, "bold"),
                        fg_color="#00e676", text_color="#000", hover_color="#00c853",
                        height=48, corner_radius=0, command=self.save_and_close).pack(
            fill="x", padx=0, pady=0, side="bottom")
 
    def _render_browse_folders(self):
        for w in self._folders_box.winfo_children():
            w.destroy()
        if not self._browse_folders:
            ctk.CTkLabel(self._folders_box, text="Whole PC (all drives)", font=("Segoe UI", 12),
                         text_color="#888", anchor="w").pack(anchor="w", padx=10, pady=8)
            return
        for folder in self._browse_folders:
            row = ctk.CTkFrame(self._folders_box, fg_color="transparent")
            row.pack(fill="x", padx=6, pady=2)
            ctk.CTkLabel(row, text="📁 " + folder, font=("Segoe UI", 11), text_color="#e0e0e0",
                         anchor="w").pack(side="left", fill="x", expand=True, padx=4)
            ctk.CTkButton(row, text="✕", width=28, height=24, fg_color="#1e1e1e", hover_color="#3a1a1a",
                          text_color="#ff6b6b",
                          command=lambda f=folder: self._remove_browse_folder(f)).pack(side="right")

    def _add_browse_folder(self):
        folder = filedialog.askdirectory(parent=self, title="Share a folder with the phone")
        if folder:
            folder = os.path.normpath(folder)
            if folder not in self._browse_folders:
                self._browse_folders.append(folder)
            self._render_browse_folders()

    def _remove_browse_folder(self, folder):
        self._browse_folders = [f for f in self._browse_folders if f != folder]
        self._render_browse_folders()

    def save_and_close(self):
        try:
            port = int(self.ent_port.get())
            if not (1024 <= port <= 65535): raise ValueError
        except:
            messagebox.showerror("Invalid Port", "Port must be between 1024 and 65535")
            return
        self.callback_save({
            "autostart_pc": self.var_autostart_pc.get(),
            "autostart_server": self.var_autostart_server.get(),
            "run_as_admin": self.var_admin.get(),
            "gaming_mode": self.var_gaming_mode.get(),
            "send_to_menu": self.var_send_to.get(),
            "virtual_display": self.var_virtual_display.get(),
            "allow_file_browsing": self.var_file_browsing.get(),
            "browse_folders": list(self._browse_folders),
            "allow_file_changes": self.var_file_changes.get(),
            "allow_power_control": self.var_power_control.get(),
            "notification_popups": self.var_notif_popups.get(),
            "call_popups": self.var_call_popups.get(),
            "allow_deck_commands": self.var_deck_commands.get(),
            "port": port,
            "ui_scale": float(self.opt_scale.get()),
            "password": self.ent_password.get().strip(),
            "password_enabled": self.var_require_password.get()
        })
        self.destroy()
 
 
# ============================================================
# MAIN GUI — Fully redesigned
# ============================================================
 
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")
 
 
class ServerGUI:
    # ── colour palette ──────────────────────────────────────
    C_BG       = "#0d0d0d"
    C_SURFACE  = "#141414"
    C_CARD     = "#1a1a1a"
    C_BORDER   = "#242424"
    C_ACCENT   = "#00e676"
    C_ACCENT2  = "#00c853"
    C_ERR      = "#ff5252"
    C_WARN     = "#ffd740"
    C_DIM      = "#FFFFFF"
    C_TEXT     = "#e8e8e8"
    C_TEXT2    = "#999999"
 
    def __init__(self):
        self.platform = get_platform()
        self.prefs = load_prefs(self.platform)
        try:
            scale = float(self.prefs.get("ui_scale", self.platform.default_ui_scale))
            ctk.set_widget_scaling(scale)
            ctk.set_window_scaling(scale)
        except Exception:
            pass

        self.root = None
        if _DnDRoot is not None:
            try:
                self.root = _DnDRoot()
            except Exception as e:   # tkdnd library missing for this Tcl/Tk
                print(f"Drag-and-drop unavailable: {e}")
        if self.root is None:
            self.root = ctk.CTk()
        self.root.title(f"Use As Server  ·  v{APP_VERSION}  ·  {self.platform.display_name}")
        self.root.geometry("980x700")
        self.root.minsize(860, 620)
        self.root.configure(fg_color=self.C_BG)
 
        self.tray_icon = None
        self.vcam_options = self.platform.vcam_options()   # [(label, device)]
        self.vcam_chips = {}
        self.vcam_studio_btns = {}
        self._active_vcam_label = None

        try:
            if self.platform.name == "windows":
                icon_path = resource_path("icon.ico")
                if os.path.exists(icon_path):
                    self.root.iconbitmap(icon_path)
            else:
                png = resource_path("icon.png")
                if os.path.exists(png):
                    self._icon_img = tk.PhotoImage(file=png)
                    self.root.iconphoto(True, self._icon_img)
        except Exception:
            pass
 
        self.update_queue = queue.Queue()
        self.server = None
        self.is_running = False
        self._client_count = 0
        self._endpoints = []
        self._pending_sends = []        # files waiting for a phone ("Send to phone", drops, picks)
        self._pending_notice = 0
        self._sending = False
        self._transfer_hide_job = None

        if not self.prefs.get("pairing_token"):
            self.prefs["pairing_token"] = pairing.new_token()
            save_prefs(self.platform, self.prefs)

        self._build_root_layout()

        # Second launches ("Send to phone", or opening the app again) talk to this window.
        self.ipc = IpcServer(lambda msg: self.update_queue.put(("ipc", msg)))
        if self.ipc.bind():
            self.ipc.start()
        if self.prefs.get("send_to_menu", True) and self.platform.send_to_label:
            threading.Thread(target=self.platform.install_send_to, args=(True,), daemon=True).start()
        self._setup_drop_target()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.platform.style_window(self.root)
        self.check_for_updates(manual=False)
 
        if self.prefs["autostart_server"]:
            self.root.after(1000, self.toggle_server)
 
        self.root.after(100, self.process_queue)
        self.root.after(50, self.update_preview)





    def check_for_updates(self, manual=False):
        """Spawns a background thread to check GitHub for updates."""
        import threading
        threading.Thread(target=self._fetch_latest_version, args=(manual,), daemon=True).start()

    def _fetch_latest_version(self, manual):
        import urllib.request
        import json
        import webbrowser
        import re
        from tkinter import messagebox

        url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
        
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                
                remote_tag = data.get("tag_name", "0.0").strip().lower().lstrip('v')
                current_tag = APP_VERSION.strip().lower().lstrip('v')

                # Helper to convert "1.10.1" -> (1, 10, 1)
                def parse_version(v_str):
                    parts = re.findall(r'\d+', v_str)
                    return tuple(map(int, parts)) if parts else (0,)

                try:
                    remote_tuple = parse_version(remote_tag)
                    current_tuple = parse_version(current_tag)
                    is_newer = remote_tuple > current_tuple
                except ValueError:
                    # Fallback if tags have weird characters
                    is_newer = remote_tag != current_tag

                if is_newer:
                    html_url = data.get("html_url", f"https://github.com/{GITHUB_REPO}/releases")
                    self.root.after(0, lambda: self._show_update_dialog(remote_tag, html_url))
                elif manual:
                    self.root.after(0, lambda: messagebox.showinfo("Up to Date", f"You are running the latest version (v{APP_VERSION})."))
        except Exception as e:
            if manual:
                # Bind the message now: Python deletes `e` when the except block ends, so the
                # lambda used to raise NameError and the error dialog never appeared.
                err = str(e)
                self.root.after(0, lambda: messagebox.showerror("Update Error", f"Could not check for updates:\n{err}"))

    def _show_update_dialog(self, new_version, download_url):
        import webbrowser
        from tkinter import messagebox
        
        ans = messagebox.askyesno(
            "Update Available", 
            f"A new version (v{new_version}) is available!\n\nWould you like to open GitHub to download it?"
        )
        if ans:
            webbrowser.open(download_url)
 
    # ── preferences ────────────────────────────────────────
    def _write_prefs_file(self):
        save_prefs(self.platform, self.prefs)

    def save_preferences(self, new_prefs=None):
        # Merge, don't replace: the Settings dialog doesn't know about every key
        # (e.g. clipboard_autosync), and replacing the dict silently dropped them.
        old_scale = self.prefs.get("ui_scale")
        if new_prefs: self.prefs.update(new_prefs)
        self._write_prefs_file()
        self.platform.set_autostart(bool(self.prefs.get("autostart_pc")))
        if self.platform.send_to_label:
            threading.Thread(target=self.platform.install_send_to,
                             args=(bool(self.prefs.get("send_to_menu", True)),), daemon=True).start()
        if new_prefs and new_prefs.get("ui_scale") not in (None, old_scale):
            self.log("ℹ️ Interface scale changes apply after restarting the app.")

        if self.is_running:
            self.log("⚙️ Settings changed — restarting server to apply changes…")
            self.stop_server()
            self.root.after(400, self.toggle_server)
 
    def open_settings(self):
        SettingsDialog(self.root, self.prefs, self.save_preferences)
 
    # ── tray ───────────────────────────────────────────────
    def create_default_icon(self):
        img = Image.new('RGB', (64, 64), "#00e676")
        dc = ImageDraw.Draw(img)
        dc.rectangle((16, 16, 48, 48), fill="#141414")
        return img
 
    def minimize_to_tray(self):
        self.root.withdraw()
        
        # --- NEW CODE ---
        image = None
        for name in ("icon.ico", "icon.png"):
            icon_path = resource_path(name)
            if os.path.exists(icon_path):
                try:
                    image = Image.open(icon_path)
                    break
                except Exception:
                    pass
        if image is None:
            image = self.create_default_icon()
            
        menu = (item('Restore', self.restore_from_tray, default=True),
                item('Stop & Quit', self.quit_app))
        self.tray_icon = pystray.Icon("UseAsServer", image, "Use As Server", menu)
        self.tray_icon.run_detached()
 
    def restore_from_tray(self, icon=None, item=None):
        if self.tray_icon: self.tray_icon.stop(); self.tray_icon = None
        self.root.after(0, self.root.deiconify)
 
    def quit_app(self, icon=None, item=None):
        if self.tray_icon: self.tray_icon.stop()
        self.root.after(0, self.on_closing)
 
    # ── root layout ────────────────────────────────────────
    def _build_root_layout(self):
        self.root.grid_columnconfigure(0, weight=0)   # sidebar (fixed)
        self.root.grid_columnconfigure(1, weight=1)   # content
        self.root.grid_rowconfigure(0, weight=1)
 
        self._build_sidebar()
        self._build_content_area()
        self._nav_go("dashboard")
 
    # ── sidebar ────────────────────────────────────────────
    def _build_sidebar(self):
        sb = ctk.CTkFrame(self.root, width=200, fg_color=self.C_SURFACE,
                           corner_radius=0, border_width=0)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_propagate(False)
        sb.grid_rowconfigure(10, weight=1)
 
        # Logo block
        logo_f = ctk.CTkFrame(sb, fg_color="#0d0d0d", corner_radius=0, height=80)
        logo_f.pack(fill="x")
        logo_f.pack_propagate(False)
        ctk.CTkLabel(logo_f, text="USE AS", font=("Courier New", 15, "bold"),
                      text_color=self.C_DIM, anchor="w").place(x=20, y=12)
                      
        
        ctk.CTkLabel(logo_f, text="SERVER", font=("Courier New", 22, "bold"),
                      text_color=self.C_ACCENT, anchor="w").place(x=16, y=38)
 
        # Thin accent line
        ctk.CTkFrame(sb, height=2, fg_color=self.C_ACCENT, corner_radius=0).pack(fill="x")
 
        # Nav buttons
        self._active_nav = "dashboard"
        self._nav_buttons = {}
 
        nav_items = [
            ("dashboard", "⬛  Dashboard"),
            ("camera",    "◉  Camera Studio"),
            ("sharing",   "⇆  Sharing"),
            ("notifications", "🔔  Notifications"),
            ("setup",     "✔  Setup check"),
            ("help",      "?  How to use"),
        ]
 
        nav_container = ctk.CTkFrame(sb, fg_color="transparent")
        nav_container.pack(expand=True, fill="x")

        # 👇 2. UPDATE THE LOOP TO PACK INTO 'nav_container' 👇
        for key, label in nav_items:
            btn = ctk.CTkButton(
                nav_container, text=label, anchor="w", 
                font=("Segoe UI", 15),
                fg_color="transparent",
                text_color=self.C_TEXT2,
                hover_color="#1e1e1e",
                corner_radius=6, height=40,
                command=lambda k=key: self._nav_go(k)
            )
            btn.pack(fill="x", padx=10, pady=2)
            self._nav_buttons[key] = btn
 
          # set initial active
 
        # Bottom area
        bottom = ctk.CTkFrame(sb, fg_color="transparent")
        bottom.pack(side="bottom", fill="x", padx=10, pady=16)
 
        # Connection badge
        self.lbl_clients = ctk.CTkLabel(
            bottom, text="● 0 connected",
            font=("Courier New", 15), text_color=self.C_DIM)
        self.lbl_clients.pack(anchor="w", pady=(0, 8))
 
        # Settings Button
        ctk.CTkButton(bottom, text="⚙  Settings", height=42,
                       font=("Segoe UI", 14, "bold"), 
                       fg_color="#000000", hover_color="#0a2914", text_color=self.C_ACCENT,
                       border_width=1.5, border_color=self.C_ACCENT,
                       corner_radius=8, command=self.open_settings).pack(fill="x", pady=(0, 8))
        
        ctk.CTkButton(bottom, text="🔄  Check for Update", height=42,
                       font=("Segoe UI", 14, "bold"), 
                       fg_color="#000000", hover_color="#0a2914", text_color=self.C_ACCENT,
                       border_width=1.5, border_color=self.C_ACCENT,
                       corner_radius=8, command=lambda: self.check_for_updates(manual=True)).pack(fill="x", pady=(0, 8))

        # Tray Button
        ctk.CTkButton(bottom, text="⊡  Tray", height=42,
                       font=("Segoe UI", 14, "bold"), 
                       fg_color="#000000", hover_color="#0a2914", text_color=self.C_ACCENT,
                       border_width=1.5, border_color=self.C_ACCENT,
                       corner_radius=8, command=self.minimize_to_tray).pack(fill="x")
 
    def _nav_go(self, key):
        self._active_nav = key
        for k, btn in self._nav_buttons.items():
            if k == key:
                btn.configure(fg_color="#1f2e24", text_color=self.C_ACCENT)
            else:
                btn.configure(fg_color="transparent", text_color=self.C_TEXT2)
        if key == "setup" and not getattr(self, "_health_ran", False):
            self.run_health_checks()        # first visit: check right away
        # Show matching frame
        for k, frame in self._pages.items():
            if k == key:
                frame.grid(row=0, column=0, sticky="nsew")
            else:
                frame.grid_forget()
 
    # ── content area ───────────────────────────────────────
    def _build_content_area(self):
        self._content = ctk.CTkFrame(self.root, fg_color=self.C_BG, corner_radius=0)
        self._content.grid(row=0, column=1, sticky="nsew")
        self._content.grid_columnconfigure(0, weight=1)
        self._content.grid_rowconfigure(0, weight=1)
 
        self._pages = {}
        for key, builder in [
            ("dashboard", self._build_dashboard),
            ("camera",    self._build_camera),
            ("sharing",   self._build_sharing),
            ("notifications", self._build_notifications),
            ("setup",     self._build_setup),
            ("help",      self._build_help),
        ]:
            page = ctk.CTkFrame(self._content, fg_color=self.C_BG, corner_radius=0)
            page.grid_columnconfigure(0, weight=1)
            page.grid_rowconfigure(0, weight=1)
            builder(page)
            self._pages[key] = page
 
    # ── helpers ────────────────────────────────────────────
    def _card(self, parent, **kw):
        return ctk.CTkFrame(parent, fg_color=self.C_CARD, corner_radius=10,
                             border_width=1, border_color=self.C_BORDER, **kw)
 
    def _section_label(self, parent, text):
        ctk.CTkLabel(parent, text=text, font=("Courier New", 10, "bold"),
                      text_color=self.C_DIM).pack(anchor="w", padx=4, pady=(0, 4))
 
    def _pill_btn(self, parent, text, command, accent=False, danger=False, **kw):
        # Default style matches the new Feature Chips (Black with Green Border)
        fg = "#000000"
        hov = "#0a2914"
        tc = self.C_ACCENT
        bw = 1.5
        bc = self.C_ACCENT

        if danger:
            fg, hov, tc = self.C_ERR, "#cc3333", "#ffffff"
            bw = 0
        elif accent:
            # Accent buttons become solid green with black text
            fg, hov, tc = self.C_ACCENT, self.C_ACCENT2, "#000000"
            bw = 0

        return ctk.CTkButton(parent, text=text, command=command,
                              fg_color=fg, hover_color=hov, text_color=tc,
                              font=("Segoe UI", 14, "bold"), corner_radius=8, height=42,
                              border_width=bw, border_color=bc, **kw)
 
    # ── Setup check ────────────────────────────────────────
    def _build_setup(self, parent):
        scroll = ctk.CTkScrollableFrame(parent, fg_color=self.C_BG,
                                          scrollbar_button_color="#1e1e1e",
                                          scrollbar_button_hover_color=self.C_ACCENT)
        scroll.grid(row=0, column=0, sticky="nsew", padx=24, pady=20)
        scroll.grid_columnconfigure(0, weight=1)

        head = ctk.CTkFrame(scroll, fg_color="transparent")
        head.pack(fill="x")
        ctk.CTkLabel(head, text="SETUP CHECK", font=("Courier New", 13, "bold"),
                     text_color=self.C_DIM).pack(side="left")
        self.btn_recheck = self._pill_btn(head, "↻  Check again", self.run_health_checks, width=150)
        self.btn_recheck.pack(side="right")
        ctk.CTkFrame(scroll, height=1, fg_color=self.C_BORDER).pack(fill="x", pady=(8, 8))
        self.lbl_health_summary = ctk.CTkLabel(
            scroll, text="Everything each feature needs, and a button to fix what's missing.",
            font=("Segoe UI", 13), text_color=self.C_TEXT2, anchor="w", justify="left")
        self.lbl_health_summary.pack(anchor="w", pady=(0, 12))
        self.health_list = ctk.CTkFrame(scroll, fg_color="transparent")
        self.health_list.pack(fill="x")

    def run_health_checks(self):
        """Runs the checks on a worker thread (some take a second); results come back
        through the update queue."""
        from . import health
        self._health_ran = True
        try:
            self.btn_recheck.configure(state="disabled", text="Checking…")
        except Exception:
            pass

        def work():
            try:
                rows = health.run_checks(self.platform, self.server if self.is_running else None)
            except Exception as e:
                rows = [health.Check("Setup check", "", health.INFO, f"Failed: {e}")]
            self.update_queue.put(("health", rows))
        threading.Thread(target=work, daemon=True, name="health").start()

    def _render_health(self, rows):
        from . import health
        try:
            self.btn_recheck.configure(state="normal", text="↻  Check again")
        except Exception:
            pass
        for child in self.health_list.winfo_children():
            child.destroy()
        icons = {health.OK: ("✅", self.C_ACCENT), health.WARN: ("⚠️", self.C_WARN),
                 health.MISSING: ("❌", self.C_ERR), health.INFO: ("ℹ️", self.C_TEXT2)}
        problems = sum(1 for r in rows if r.status == health.MISSING)
        optional = sum(1 for r in rows if r.status == health.WARN)
        if problems:
            summary = f"{problems} thing(s) need attention" + (f", {optional} optional" if optional else "")
        elif optional:
            summary = f"All set. {optional} optional extra(s) would add more features."
        else:
            summary = "All set — every feature is ready."
        self.lbl_health_summary.configure(text=summary)

        for row in rows:
            card = self._card(self.health_list)
            card.pack(fill="x", pady=(0, 8))
            card.grid_columnconfigure(1, weight=1)
            icon, colour = icons.get(row.status, ("•", self.C_TEXT2))
            ctk.CTkLabel(card, text=icon, font=("Segoe UI Emoji", 18), width=36).grid(
                row=0, column=0, rowspan=3, padx=(12, 4), pady=12, sticky="n")
            ctk.CTkLabel(card, text=row.title, font=("Segoe UI", 14, "bold"), text_color=colour,
                         anchor="w").grid(row=0, column=1, sticky="w", padx=4, pady=(10, 0))
            if row.feature:
                ctk.CTkLabel(card, text=row.feature, font=("Segoe UI", 11), text_color=self.C_TEXT2,
                             anchor="w").grid(row=1, column=1, sticky="w", padx=4)
            if row.detail:
                ctk.CTkLabel(card, text=row.detail, font=("Segoe UI", 12), text_color=self.C_TEXT,
                             anchor="w", justify="left", wraplength=520).grid(
                    row=2, column=1, sticky="w", padx=4, pady=(2, 10))
            if row.fix is not None and row.fix_label and row.status != health.OK:
                btn = self._pill_btn(card, row.fix_label, None, accent=row.status == health.MISSING, width=110)
                btn.configure(command=lambda r=row, b=btn: self._run_fix(r, b))
                btn.grid(row=0, column=2, rowspan=3, padx=12, pady=12)

    def _run_fix(self, row, button):
        button.configure(state="disabled", text="Working…")

        def work():
            try:
                row.fix(lambda m: self.update_queue.put(("log", m)))
            except Exception as e:
                self.update_queue.put(("log", f"❌ {row.title}: {e}"))
            self.run_health_checks()       # the queue re-renders the list
        threading.Thread(target=work, daemon=True, name="health-fix").start()

    # ── Phone notifications ────────────────────────────────
    def _build_notifications(self, parent):
        scroll = ctk.CTkScrollableFrame(parent, fg_color=self.C_BG,
                                          scrollbar_button_color="#1e1e1e",
                                          scrollbar_button_hover_color=self.C_ACCENT)
        scroll.grid(row=0, column=0, sticky="nsew", padx=24, pady=20)
        head = ctk.CTkFrame(scroll, fg_color="transparent")
        head.pack(fill="x")
        ctk.CTkLabel(head, text="PHONE NOTIFICATIONS", font=("Courier New", 13, "bold"),
                     text_color=self.C_DIM).pack(side="left")
        self._small_btn(head, "Clear", self._clear_notifications).pack(side="right")
        ctk.CTkLabel(scroll, text="Notifications from your phone, with quick reply for messaging apps. "
                                  "Turn it on in the phone app: PC tab › Show phone notifications on PC.",
                     font=("Segoe UI", 13), text_color=self.C_TEXT2, anchor="w", justify="left",
                     wraplength=700).pack(anchor="w", pady=(2, 0))
        ctk.CTkFrame(scroll, height=1, fg_color=self.C_BORDER).pack(fill="x", pady=(8, 16))
        self.notif_list = ctk.CTkFrame(scroll, fg_color="transparent")
        self.notif_list.pack(fill="x")
        self._notifications = []        # newest first: dicts from the phone
        self._notif_popups = {}
        self._render_notifications()

    def _clear_notifications(self):
        self._notifications.clear()
        self._render_notifications()

    def _on_phone_notification(self, n):
        key = str(n.get("key", ""))
        self._notifications = [x for x in self._notifications if x.get("key") != key]
        n = dict(n, received=time.strftime("%H:%M"))
        self._notifications.insert(0, n)
        del self._notifications[50:]
        self._render_notifications()
        if self.prefs.get("notification_popups", True) and not n.get("silent"):
            self._show_notification_popup(n)

    def _remove_notification(self, key):
        self._notifications = [x for x in self._notifications if x.get("key") != key]
        self._render_notifications()
        popup = self._notif_popups.pop(key, None)
        if popup is not None:
            try:
                popup.destroy()
            except Exception:
                pass

    def _notif_icon(self, n, size=28):
        data = n.get("icon")
        if not data:
            return None
        try:
            import base64
            import io
            from PIL import Image
            img = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGBA")
            return ctk.CTkImage(light_image=img, dark_image=img, size=(size, size))
        except Exception:
            return None

    def _render_notifications(self):
        if not hasattr(self, "notif_list"):
            return
        for child in self.notif_list.winfo_children():
            child.destroy()
        if not self._notifications:
            ctk.CTkLabel(self.notif_list, text="No notifications yet.", font=("Segoe UI", 13),
                         text_color="#555").pack(pady=24)
            return
        for n in self._notifications:
            self._notification_card(self.notif_list, n).pack(fill="x", pady=(0, 8))

    def _notification_card(self, parent, n, popup=None):
        key = str(n.get("key", ""))
        card = self._card(parent)
        top = ctk.CTkFrame(card, fg_color="transparent")
        top.pack(fill="x", padx=14, pady=(12, 0))
        icon = self._notif_icon(n)
        ctk.CTkLabel(top, text="" if icon else "🔔", image=icon, width=28).pack(side="left")
        ctk.CTkLabel(top, text=str(n.get("app", "Phone")), font=("Segoe UI", 12, "bold"),
                     text_color=self.C_ACCENT).pack(side="left", padx=(8, 0))
        ctk.CTkLabel(top, text=n.get("received", ""), font=("Segoe UI", 11),
                     text_color=self.C_TEXT2).pack(side="right")
        if n.get("title"):
            ctk.CTkLabel(card, text=str(n["title"])[:120], font=("Segoe UI", 14, "bold"), text_color=self.C_TEXT,
                         anchor="w", justify="left", wraplength=560 if popup is None else 300).pack(
                anchor="w", padx=14, pady=(6, 0))
        if n.get("text"):
            ctk.CTkLabel(card, text=str(n["text"])[:500], font=("Segoe UI", 12), text_color=self.C_TEXT,
                         anchor="w", justify="left", wraplength=560 if popup is None else 300).pack(
                anchor="w", padx=14, pady=(2, 0))
        row = ctk.CTkFrame(card, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=(8, 12))

        def dismiss():
            if self.server:
                self.server.dismiss_notification(key)
            self._remove_notification(key)

        if n.get("can_reply"):
            entry = ctk.CTkEntry(row, placeholder_text="Reply…", height=32, font=("Segoe UI", 12),
                                 fg_color="#0d0d0d", border_color=self.C_BORDER)
            entry.pack(side="left", fill="x", expand=True)

            def send(_event=None):
                text = entry.get().strip()
                if not text or not self.server:
                    return
                if self.server.reply_to_notification(key, text):
                    self.log(f"💬 Replied to {n.get('title') or n.get('app')}")
                    entry.delete(0, "end")
                    if popup is not None:
                        popup.destroy()
                        self._notif_popups.pop(key, None)
            entry.bind("<Return>", send)
            if popup is not None:
                entry.bind("<FocusIn>", lambda e: setattr(popup, "_keep", True))
            self._small_btn(row, "Send", send, accent=True).pack(side="left", padx=(8, 0))
        ctk.CTkButton(row, text="Dismiss", width=80, height=32, corner_radius=8, font=("Segoe UI", 12),
                      fg_color="transparent", hover_color="#1e1e1e", text_color=self.C_TEXT2,
                      border_width=1, border_color=self.C_BORDER, command=dismiss).pack(side="right")
        return card

    def _show_notification_popup(self, n):
        """Small always-on-top card in the bottom-right corner for ~8 s (longer while replying)."""
        key = str(n.get("key", ""))
        old = self._notif_popups.pop(key, None)
        if old is not None:
            try:
                old.destroy()
            except Exception:
                pass
        # Keep at most three on screen.
        while len(self._notif_popups) >= 3:
            _k, p = next(iter(self._notif_popups.items()))
            self._notif_popups.pop(_k, None)
            try:
                p.destroy()
            except Exception:
                pass
        popup = ctk.CTkToplevel(self.root)
        popup.overrideredirect(True)
        popup.attributes("-topmost", True)
        popup.configure(fg_color=self.C_BG)
        popup._keep = False
        self._notification_card(popup, n, popup=popup).pack(fill="both", expand=True, padx=1, pady=1)
        popup.update_idletasks()
        w, h = 380, popup.winfo_reqheight()
        idx = len(self._notif_popups)
        x = popup.winfo_screenwidth() - w - 16
        y = popup.winfo_screenheight() - (h + 12) * (idx + 1) - 56
        popup.geometry(f"{w}x{h}+{x}+{max(0, y)}")
        self._notif_popups[key] = popup

        def expire():
            if not popup.winfo_exists():
                return
            if getattr(popup, "_keep", False):
                popup.after(4000, expire)      # still typing a reply
                return
            self._notif_popups.pop(key, None)
            popup.destroy()
        popup.after(8000, expire)

    # ── How to use ─────────────────────────────────────────
    def _build_help(self, parent):
        from .help_text import SECTIONS
        scroll = ctk.CTkScrollableFrame(parent, fg_color=self.C_BG,
                                          scrollbar_button_color="#1e1e1e",
                                          scrollbar_button_hover_color=self.C_ACCENT)
        scroll.grid(row=0, column=0, sticky="nsew", padx=24, pady=20)
        scroll.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(scroll, text="HOW TO USE", font=("Courier New", 13, "bold"),
                     text_color=self.C_DIM).pack(anchor="w")
        ctk.CTkFrame(scroll, height=1, fg_color=self.C_BORDER).pack(fill="x", pady=(4, 16))

        for title, paragraphs in SECTIONS:
            card = self._card(scroll)
            card.pack(fill="x", pady=(0, 12))
            ctk.CTkLabel(card, text=title, font=("Segoe UI", 15, "bold"), text_color=self.C_ACCENT,
                         anchor="w").pack(anchor="w", padx=16, pady=(12, 4))
            for para in paragraphs:
                bullet = para.startswith("• ")
                ctk.CTkLabel(card, text=para, font=("Segoe UI", 12),
                             text_color=self.C_TEXT if bullet else self.C_TEXT2,
                             anchor="w", justify="left", wraplength=600).pack(
                    anchor="w", padx=(28 if bullet else 16, 16), pady=2)
            ctk.CTkFrame(card, height=8, fg_color="transparent").pack()

    # ── Dashboard ──────────────────────────────────────────
    def _build_dashboard(self, parent):
        scroll = ctk.CTkScrollableFrame(parent, fg_color=self.C_BG,
                                          scrollbar_button_color="#1e1e1e",
                                          scrollbar_button_hover_color=self.C_ACCENT)
        scroll.grid(row=0, column=0, sticky="nsew", padx=24, pady=20)
        scroll.grid_columnconfigure(0, weight=1)
 
        # Page title
        ctk.CTkLabel(scroll, text="DASHBOARD",
                      font=("Courier New", 13, "bold"), text_color=self.C_DIM).pack(anchor="w")
        ctk.CTkFrame(scroll, height=1, fg_color=self.C_BORDER).pack(fill="x", pady=(4, 16))
 
        # ─ Server control card ─────────────────────────────
        ctrl = self._card(scroll)
        ctrl.pack(fill="x", pady=(0, 16))
 
        top_row = ctk.CTkFrame(ctrl, fg_color="transparent")
        top_row.pack(fill="x", padx=20, pady=(20, 12))
 
        # Status indicator
        ind_f = ctk.CTkFrame(top_row, fg_color="transparent")
        ind_f.pack(side="left")
        self._status_dot = ctk.CTkLabel(ind_f, text="●",
                                          font=("Segoe UI", 28), text_color=self.C_ERR)
        self._status_dot.pack(side="left")
        status_labels = ctk.CTkFrame(ind_f, fg_color="transparent")
        status_labels.pack(side="left", padx=12)
        self.lbl_status = ctk.CTkLabel(status_labels, text="OFFLINE",
                                         font=("Courier New", 18, "bold"), text_color=self.C_ERR)
        self.lbl_status.pack(anchor="w")
        self.lbl_status_sub = ctk.CTkLabel(status_labels, text="Server not running",
                                             font=("Segoe UI", 14), text_color=self.C_DIM)
        self.lbl_status_sub.pack(anchor="w")
 
        self.btn_start = ctk.CTkButton(
            top_row, text="START", width=110, height=44,
            font=("Courier New", 16, "bold"),
            fg_color=self.C_ACCENT, text_color="#000", hover_color=self.C_ACCENT2,
            corner_radius=6, command=self.toggle_server)
        self.btn_start.pack(side="right")
 
        # IP display
        ip_f = ctk.CTkFrame(ctrl, fg_color="#111111", corner_radius=6)
        ip_f.pack(fill="x", padx=20, pady=(0, 20))
        ctk.CTkLabel(ip_f, text="ENDPOINT", font=("Courier New", 9, "bold"),
                      text_color=self.C_DIM).pack(anchor="w", padx=12, pady=(8, 2))
        self.ent_ip = ctk.CTkEntry(ip_f, font=("Courier New", 14),
                                    fg_color="transparent", border_width=0,
                                    text_color=self.C_ACCENT, state="readonly",
                                    placeholder_text="ws://—.—.—.—:——", height=34)
        self.ent_ip.pack(fill="x", padx=8, pady=(0, 10))
        ctk.CTkLabel(ip_f, text="ALL NETWORKS", font=("Courier New", 9, "bold"),
              text_color=self.C_DIM).pack(anchor="w", padx=12, pady=(8, 2))
        self.txt_endpoints = ctk.CTkTextbox(ip_f, font=("Courier New", 12),
                                     fg_color="transparent", text_color=self.C_TEXT2, height=70)
        self.txt_endpoints.pack(fill="x", padx=8, pady=(0, 10))
        self.txt_endpoints.configure(state="disabled")
        
        
        
 
        # ─ Quick features row ──────────────────────────────
        self._section_label(scroll, "FEATURES")
        feat = self._card(scroll)
        feat.pack(fill="x", pady=(0, 16))
 
        feat_inner = ctk.CTkFrame(feat, fg_color="transparent")
        feat_inner.pack(fill="x", padx=16, pady=14)
 
        # One chip per virtual-camera option this OS has (Windows: OBS + Unity).
        for label, _device in self.vcam_options:
            chip = self._feature_chip(feat_inner, label, "📹", lambda l=label: self.toggle_camera(l))
            chip.pack(side="left", padx=(0, 8))
            chip.configure(state="disabled")
            self.vcam_chips[label] = chip
 
        self.btn_audio = self._feature_chip(feat_inner, "Audio Stream", "🔊",
                                             self.toggle_audio)
        self.btn_audio.pack(side="left", padx=(0, 8))
        self.btn_audio.configure(state="disabled")

        self.btn_qr = self._feature_chip(feat_inner, "Pair with QR", "📱", self.open_qr_dialog)
        self.btn_qr.pack(side="left")
 
        # ─ Activity log ────────────────────────────────────
        log_header = ctk.CTkFrame(scroll, fg_color="transparent")
        log_header.pack(fill="x", pady=(10, 4))
        
        # 2. Pack the section label to the left
        ctk.CTkLabel(log_header, text="ACTIVITY LOG", 
                     font=("Courier New", 10, "bold"), 
                     text_color=self.C_DIM).pack(side="left", padx=4)

        # 3. Add a slider to dynamically adjust the log box height
        self.log_height_slider = ctk.CTkSlider(
            log_header, from_=100, to=800, width=120, height=14,
            button_color=self.C_ACCENT, button_hover_color=self.C_ACCENT2,
            progress_color=self.C_ACCENT,
            command=lambda val: self.log_box.configure(height=int(val))
        )
        self.log_height_slider.set(220) # Set default height
        self.log_height_slider.pack(side="right", padx=10)
        
        ctk.CTkLabel(log_header, text="Window Height:", 
                     font=("Segoe UI", 10), text_color=self.C_DIM).pack(side="right")

        # 4. Create the card and textbox (Now with green text!)
        log_card = self._card(scroll)
        log_card.pack(fill="both", expand=True, pady=(0, 4))

        self.log_box = ctk.CTkTextbox(
            log_card, font=("Courier New", 14),
            fg_color="#0d0d0d", 
            text_color=self.C_ACCENT, # 🟢 THIS MAKES THE TEXT GREEN
            scrollbar_button_color="#1e1e1e",
            scrollbar_button_hover_color=self.C_ACCENT,
            height=220, corner_radius=8)
        self.log_box.pack(fill="both", expand=True, padx=4, pady=4)
 
    def _feature_chip(self, parent, label, icon, command):
        """Updated toggle chip for feature shortcuts."""
        return ctk.CTkButton(
            parent, text=f"{icon}  {label}", height=42,
            font=("Segoe UI", 14, "bold"), corner_radius=8,
            fg_color="#000000", hover_color="#0a2914",
            text_color=self.C_ACCENT, border_width=1.5,
            border_color=self.C_ACCENT, command=command)
            
            
            
            
     
    def _update_endpoints(self, endpoints):
        self._endpoints = list(endpoints or [])
        dlg = getattr(self, "_qr_dialog", None)
        if dlg is not None and dlg.winfo_exists():
            dlg.refresh()
        self.txt_endpoints.configure(state="normal")
        self.txt_endpoints.delete("1.0", "end")
        if not endpoints:
            self.txt_endpoints.insert("end", "⚠️ No active network interfaces detected")
        else:
            for ep in endpoints:
                self.txt_endpoints.insert("end", f"● {ep['label']}: ws://{ep['ip']}:{ep['port']}\n")
            priority = ["USB Tethering", "Mobile Hotspot", "Wi-Fi", "Ethernet / LAN"]
            best = min(endpoints, key=lambda e: priority.index(e['label']) if e['label'] in priority else 99)
            self._set_ip(f"{best['ip']}:{best['port']}")
        self.txt_endpoints.configure(state="disabled")
 
    # ── Camera Studio ──────────────────────────────────────
    def _build_camera(self, parent):
        parent.grid_columnconfigure(0, weight=0)
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_rowconfigure(0, weight=1)
 
        # Left panel
        left = ctk.CTkScrollableFrame(parent, width=220, fg_color=self.C_SURFACE,
                                       corner_radius=0, border_width=0,
                                       scrollbar_button_color="#1e1e1e",
                                       scrollbar_button_hover_color=self.C_ACCENT)
        left.grid(row=0, column=0, sticky="nsew")
 
        # Page title in panel
        ctk.CTkFrame(left, height=2, fg_color=self.C_BORDER).pack(fill="x")
        ctk.CTkLabel(left, text="CAMERA STUDIO",
                      font=("Courier New", 10, "bold"), text_color=self.C_DIM).pack(
            anchor="w", padx=16, pady=(14, 10))
 
        # Background section
        bg_card = self._card(left)
        bg_card.pack(fill="x", padx=10, pady=(0, 10))
        ctk.CTkLabel(bg_card, text="BACKGROUND", font=("Courier New", 9, "bold"),
                      text_color=self.C_DIM).pack(anchor="w", padx=14, pady=(10, 6))
 
        self.var_bg = ctk.StringVar(value="none")
        for val, lbl in [("none", "None"), ("blur", "Blur"), ("image", "Custom Image")]:
            ctk.CTkRadioButton(
                bg_card, text=lbl, variable=self.var_bg, value=val,
                font=("Segoe UI", 12), text_color=self.C_TEXT,
                fg_color=self.C_ACCENT, hover_color=self.C_ACCENT2,
                command=self.update_cam_settings).pack(anchor="w", padx=16, pady=3)
 
        ctk.CTkButton(bg_card, text="Choose image…", height=30,
                       font=("Segoe UI", 11), fg_color="#111",
                       hover_color="#1e1e1e", text_color=self.C_TEXT2,
                       border_width=1, border_color=self.C_BORDER,
                       command=self.select_bg_image).pack(
            fill="x", padx=14, pady=(6, 12))
 
        # Adjustments section
        adj_card = self._card(left)
        adj_card.pack(fill="x", padx=10, pady=(0, 10))
        ctk.CTkLabel(adj_card, text="ADJUSTMENTS", font=("Courier New", 9, "bold"),
                      text_color=self.C_DIM).pack(anchor="w", padx=14, pady=(10, 6))
 
        self.var_mirror = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(adj_card, text="Mirror", variable=self.var_mirror,
                       font=("Segoe UI", 12), text_color=self.C_TEXT,
                       button_color=self.C_ACCENT, button_hover_color=self.C_ACCENT2,
                       progress_color=self.C_ACCENT,
                       command=self.update_cam_settings).pack(anchor="w", padx=14, pady=4)
        self.var_flip = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(adj_card, text="Flip Vertical", variable=self.var_flip,
                       font=("Segoe UI", 12), text_color=self.C_TEXT,
                       button_color=self.C_ACCENT, button_hover_color=self.C_ACCENT2,
                       progress_color=self.C_ACCENT,
                       command=self.update_cam_settings).pack(anchor="w", padx=14, pady=4)
 
        ctk.CTkLabel(adj_card, text="Brightness", font=("Segoe UI", 11),
                      text_color=self.C_TEXT2).pack(anchor="w", padx=14, pady=(6, 2))
        self.scale_bright = ctk.CTkSlider(adj_card, from_=-100, to=100,
                                           button_color=self.C_ACCENT,
                                           button_hover_color=self.C_ACCENT2,
                                           progress_color=self.C_ACCENT,
                                           command=lambda x: self.update_cam_settings())
        self.scale_bright.set(0)
        self.scale_bright.pack(fill="x", padx=14, pady=(0, 12))
 
        # Resolution
        res_card = self._card(left)
        res_card.pack(fill="x", padx=10, pady=(0, 10))
        ctk.CTkLabel(res_card, text="RESOLUTION", font=("Courier New", 9, "bold"),
                      text_color=self.C_DIM).pack(anchor="w", padx=14, pady=(10, 6))
        self.combo_res = ctk.CTkOptionMenu(
            res_card,
            values=["1280x720 (16:9)", "1920x1080 (16:9)", "800x600 (4:3)", "720x720 (1:1)"],
            font=("Segoe UI", 12), fg_color="#111",
            button_color="#1e1e1e", button_hover_color="#252525",
            dropdown_fg_color="#141414", text_color=self.C_TEXT,
            command=self.change_aspect_ratio)
        self.combo_res.pack(fill="x", padx=14, pady=(0, 12))
 
        # Output control
        out_card = self._card(left)
        out_card.pack(fill="x", padx=10, pady=(0, 12))
        ctk.CTkLabel(out_card, text="OUTPUT", font=("Courier New", 9, "bold"),
                      text_color=self.C_DIM).pack(anchor="w", padx=14, pady=(10, 6))
 
        for label, _device in self.vcam_options:
            btn = ctk.CTkButton(
                out_card, text=f"▶  {label}", height=36,
                font=("Segoe UI", 12), fg_color="#1a2a1f",
                hover_color="#1f3325", text_color=self.C_ACCENT,
                border_width=1, border_color="#2a3d2d",
                command=lambda l=label: self.toggle_camera(l))
            btn.pack(fill="x", padx=14, pady=(0, 6))
            self.vcam_studio_btns[label] = btn
 
        # Right — preview
        right = ctk.CTkFrame(parent, fg_color="#080808", corner_radius=0)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_propagate(False)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=1)
 
        self.lbl_preview = ctk.CTkLabel(
            right, text="No signal\nWaiting for connection…",
            font=("Courier New", 13), text_color="#333",
            justify="center")
        # Pack it so it physically expands to fill the entire black frame
        self.lbl_preview.pack(fill="both", expand=True)
 
       
        badge = ctk.CTkFrame(right, fg_color="#0d0d0d", corner_radius=6, width=120, height=28)
        badge.place(x=12, y=12)
        self._preview_badge = ctk.CTkLabel(badge, text="● LIVE", font=("Courier New", 10, "bold"),
                                            text_color="#555")
        self._preview_badge.pack(padx=10, pady=4)
 
    # ── Sharing ────────────────────────────────────────────
    def _build_sharing(self, parent):
        scroll = ctk.CTkScrollableFrame(parent, fg_color=self.C_BG,
                                          scrollbar_button_color="#1e1e1e",
                                          scrollbar_button_hover_color=self.C_ACCENT)
        scroll.grid(row=0, column=0, sticky="nsew", padx=24, pady=20)
        scroll.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(scroll, text="SHARING",
                      font=("Courier New", 13, "bold"), text_color=self.C_DIM).pack(anchor="w")
        ctk.CTkLabel(scroll, text="Send text, images and files between this PC and your phone.",
                     font=("Segoe UI", 13), text_color=self.C_TEXT2).pack(anchor="w", pady=(2, 0))
        ctk.CTkFrame(scroll, height=1, fg_color=self.C_BORDER).pack(fill="x", pady=(8, 16))

        # Two cards: side by side on a wide window, stacked on a narrow one.
        cards = ctk.CTkFrame(scroll, fg_color="transparent")
        cards.pack(fill="x")
        clip_card = self._card(cards)
        file_card = self._card(cards)
        self._sharing_cards = (cards, clip_card, file_card)
        self._sharing_wide = None
        cards.bind("<Configure>", lambda e: self._layout_sharing(e.width))
        self._layout_sharing(900)

        # ─ Clipboard ───────────────────────────────────────
        head = ctk.CTkFrame(clip_card, fg_color="transparent")
        head.pack(fill="x", padx=18, pady=(16, 2))
        ctk.CTkLabel(head, text="📋  Clipboard", font=("Segoe UI", 16, "bold"),
                     text_color=self.C_TEXT).pack(side="left")
        self.var_clip_autosync = ctk.BooleanVar(value=self.prefs.get("clipboard_autosync", False))
        ctk.CTkSwitch(head, text="Auto-sync", variable=self.var_clip_autosync,
                       font=("Segoe UI", 12), text_color=self.C_TEXT,
                       button_color=self.C_ACCENT, button_hover_color=self.C_ACCENT2,
                       progress_color=self.C_ACCENT,
                       command=self.toggle_clipboard_autosync).pack(side="right")
        ctk.CTkLabel(clip_card, text="With Auto-sync on, whatever you copy on this PC (text, screenshots, "
                                     "files) goes to the phone. Text from the phone shows up here.",
                     font=("Segoe UI", 12), text_color=self.C_TEXT2, anchor="w", justify="left",
                     wraplength=460).pack(anchor="w", padx=18, pady=(0, 10))

        box = ctk.CTkFrame(clip_card, fg_color="transparent")
        box.pack(fill="x", padx=18)
        self.txt_clip = ctk.CTkTextbox(
            box, height=110, font=("Segoe UI", 13),
            fg_color="#0d0d0d", text_color=self.C_TEXT, border_width=1, border_color=self.C_BORDER,
            scrollbar_button_color="#1e1e1e", corner_radius=8, wrap="word")
        self.txt_clip.pack(fill="x")
        # CTkTextbox has no placeholder: a hint label sits on top while it's empty.
        self._clip_hint = ctk.CTkLabel(box, text="Type or paste text to send to the phone…",
                                       font=("Segoe UI", 13), text_color="#555", fg_color="#0d0d0d")
        self._clip_hint.place(x=14, y=10)
        self._clip_hint.bind("<Button-1>", lambda e: self.txt_clip.focus_set())
        for ev in ("<KeyRelease>", "<FocusIn>", "<FocusOut>", "<<Paste>>"):
            self.txt_clip.bind(ev, lambda e: self.root.after(1, self._update_clip_hint), add="+")

        actions = ctk.CTkFrame(clip_card, fg_color="transparent")
        actions.pack(fill="x", padx=18, pady=(12, 18))
        self._small_btn(actions, "Send to phone  →", self.send_text_to_phone, accent=True).pack(side="left")
        self._small_btn(actions, "Copy to PC", self.copy_to_pc).pack(side="left", padx=(8, 0))
        self._small_btn(actions, "🖼  Send copied image / files", self.send_rich_clipboard).pack(
            side="left", padx=(8, 0))

        hist_head = ctk.CTkFrame(clip_card, fg_color="transparent")
        hist_head.pack(fill="x", padx=18, pady=(0, 6))
        ctk.CTkLabel(hist_head, text="HISTORY", font=("Courier New", 10, "bold"),
                     text_color=self.C_DIM).pack(side="left")
        self._small_btn(hist_head, "Clear", self._clear_clip_history).pack(side="right")
        self.clip_hist_list = ctk.CTkScrollableFrame(clip_card, fg_color="#0d0d0d", corner_radius=8, height=200,
                                                     scrollbar_button_color="#1e1e1e")
        self.clip_hist_list.pack(fill="x", padx=18, pady=(0, 18))
        self._render_clip_history()

        # ─ Files ───────────────────────────────────────────
        ctk.CTkLabel(file_card, text="📁  Files", font=("Segoe UI", 16, "bold"),
                     text_color=self.C_TEXT).pack(anchor="w", padx=18, pady=(16, 2))
        ctk.CTkLabel(file_card, text="Files from the phone are saved in " + str(SAVE_DIR),
                     font=("Segoe UI", 12), text_color=self.C_TEXT2, anchor="w", justify="left",
                     wraplength=460).pack(anchor="w", padx=18, pady=(0, 10))

        drop = ctk.CTkFrame(file_card, fg_color="#0f1a13", corner_radius=12,
                            border_width=2, border_color="#1f4d2e")
        drop.pack(fill="x", padx=18)
        ctk.CTkLabel(drop, text="⇪", font=("Segoe UI", 30), text_color=self.C_ACCENT).pack(pady=(14, 0))
        ctk.CTkLabel(drop, text=("Drop files on this window to send them to the phone"
                                 if DND_FILES is not None else "Send files to the phone"),
                     font=("Segoe UI", 13), text_color=self.C_TEXT).pack()
        ctk.CTkLabel(drop, text="or right-click a file in Explorer › Send to › Use As (phone)"
                     if self.platform.name == "windows" else "",
                     font=("Segoe UI", 11), text_color=self.C_TEXT2).pack()
        self._small_btn(drop, "Choose files…", self.send_file_pick, accent=True).pack(pady=(8, 14))

        # Progress of the current transfer (hidden when idle)
        self.transfer_row = ctk.CTkFrame(file_card, fg_color="transparent")
        self.lbl_transfer = ctk.CTkLabel(self.transfer_row, text="", font=("Segoe UI", 12),
                                         text_color=self.C_TEXT, anchor="w")
        self.lbl_transfer.pack(fill="x")
        self.pb_transfer = ctk.CTkProgressBar(self.transfer_row, progress_color=self.C_ACCENT,
                                              fg_color="#222", height=8)
        self.pb_transfer.set(0)
        self.pb_transfer.pack(fill="x", pady=(4, 0))
        self._transfer_anchor = drop

        recent_head = ctk.CTkFrame(file_card, fg_color="transparent")
        recent_head.pack(fill="x", padx=18, pady=(14, 6))
        ctk.CTkLabel(recent_head, text="RECENT", font=("Courier New", 10, "bold"),
                     text_color=self.C_DIM).pack(side="left")
        self._small_btn(recent_head, "Open folder", self.open_folder).pack(side="right")
        self.recent_list = ctk.CTkFrame(file_card, fg_color="#0d0d0d", corner_radius=8)
        self.recent_list.pack(fill="x", padx=18, pady=(0, 18))
        self._recent_items = []
        self._render_recent()

    # Sharing-tab helpers ------------------------------------------------------------
    def _small_btn(self, parent, text, command, accent=False):
        """Compact button (the pill buttons stretched across the whole window)."""
        return ctk.CTkButton(parent, text=text, command=command, height=36, corner_radius=8,
                             font=("Segoe UI", 13, "bold"),
                             fg_color=self.C_ACCENT if accent else "transparent",
                             hover_color=self.C_ACCENT2 if accent else "#0a2914",
                             text_color="#000" if accent else self.C_ACCENT,
                             border_width=0 if accent else 1.5, border_color=self.C_ACCENT)

    def _layout_sharing(self, width):
        cards, clip_card, file_card = self._sharing_cards
        wide = width >= 1000
        if wide == self._sharing_wide:
            return
        self._sharing_wide = wide
        clip_card.grid_forget()
        file_card.grid_forget()
        if wide:
            cards.grid_columnconfigure((0, 1), weight=1, uniform="share")
            clip_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
            file_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        else:
            cards.grid_columnconfigure(0, weight=1, uniform="")
            cards.grid_columnconfigure(1, weight=0, uniform="")
            clip_card.grid(row=0, column=0, sticky="nsew", pady=(0, 16))
            file_card.grid(row=1, column=0, sticky="nsew")

    def _update_clip_hint(self):
        try:
            empty = not self.txt_clip.get("1.0", "end").strip()
            focused = self.root.focus_get() == self.txt_clip._textbox
        except Exception:
            return
        if empty and not focused:
            self._clip_hint.place(x=14, y=10)
        else:
            self._clip_hint.place_forget()

    def _clear_clip_history(self):
        if self.server:
            self.server.clip_history.clear()
        self._render_clip_history()

    def _render_clip_history(self):
        if not hasattr(self, "clip_hist_list"):
            return
        for child in self.clip_hist_list.winfo_children():
            child.destroy()
        items = list(self.server.clip_history) if self.server else []
        if not items:
            ctk.CTkLabel(self.clip_hist_list, text="Text you copy between the PC and the phone shows up here.",
                         font=("Segoe UI", 12), text_color="#555").pack(padx=14, pady=16)
            return
        for item in items:
            text = item["text"]
            one_line = " ".join(text.split())
            row = ctk.CTkFrame(self.clip_hist_list, fg_color="transparent")
            row.pack(fill="x", padx=6, pady=3)
            from_phone = item["source"] == "phone"
            ctk.CTkLabel(row, text="⬇" if from_phone else "⬆", font=("Segoe UI", 13, "bold"),
                         text_color=self.C_ACCENT if from_phone else "#64b5f6", width=18).pack(side="left")
            ctk.CTkLabel(row, text=one_line if len(one_line) <= 52 else one_line[:50] + "…",
                         font=("Segoe UI", 12), text_color=self.C_TEXT, anchor="w").pack(
                side="left", padx=(6, 0), fill="x", expand=True)
            ctk.CTkButton(row, text="Send", width=50, height=24, corner_radius=6, font=("Segoe UI", 11),
                          fg_color="transparent", hover_color="#1e1e1e", text_color="#64b5f6",
                          border_width=1, border_color="#1e3a5f",
                          command=lambda t=text: self._clip_history_send(t)).pack(side="right", padx=(4, 0))
            ctk.CTkButton(row, text="Copy", width=50, height=24, corner_radius=6, font=("Segoe UI", 11),
                          fg_color="transparent", hover_color="#1e1e1e", text_color=self.C_ACCENT,
                          border_width=1, border_color="#1f4d2e",
                          command=lambda t=text: self._clip_history_copy(t)).pack(side="right")
            ctk.CTkLabel(row, text=time.strftime("%H:%M", time.localtime(item["time"])), font=("Segoe UI", 10),
                         text_color=self.C_TEXT2).pack(side="right", padx=6)

    def _clip_history_copy(self, text):
        try:
            if self.server:
                self.server._last_clip = text      # don't auto-sync it straight back
            pyperclip.copy(text)
            self.log("📋 Copied from history")
        except Exception as e:
            self.log(f"⚠️ Clipboard error: {e}")

    def _clip_history_send(self, text):
        if self.server:
            self.server.send_to_android("clipboard_text", text)
            self.log("📋 Sent to the phone from history")

    # ── phone calls ───────────────────────────────────────
    def _on_phone_call(self, c):
        key = str(c.get("key", ""))
        who = c.get("name") or c.get("text") or "Unknown"
        state = c.get("state", "ringing")
        if state == "ringing" and key not in getattr(self, "_call_popups", {}):
            self.log(f"📞 Incoming call: {who}")
        if not self.prefs.get("call_popups", True):
            return
        self._show_call_popup(c)

    def _end_phone_call(self, key):
        popup = getattr(self, "_call_popups", {}).pop(key, None)
        if popup is not None:
            try:
                popup.destroy()
            except Exception:
                pass

    def _show_call_popup(self, c):
        """Top-right card while the phone rings (Decline / Answer) or a call is going on (Hang up)."""
        if not hasattr(self, "_call_popups"):
            self._call_popups = {}
        key = str(c.get("key", ""))
        self._end_phone_call(key)
        ringing = c.get("state", "ringing") == "ringing"
        popup = ctk.CTkToplevel(self.root)
        popup.overrideredirect(True)
        popup.attributes("-topmost", True)
        popup.configure(fg_color="#0f1a13" if ringing else self.C_BG)
        card = ctk.CTkFrame(popup, fg_color="#111", corner_radius=14, border_width=2,
                            border_color=self.C_ACCENT if ringing else self.C_BORDER)
        card.pack(fill="both", expand=True, padx=1, pady=1)
        top = ctk.CTkFrame(card, fg_color="transparent")
        top.pack(fill="x", padx=16, pady=(14, 4))
        icon = self._notif_icon(c, size=48)
        ctk.CTkLabel(top, text="" if icon else "📞", image=icon, font=("Segoe UI", 30), width=52).pack(side="left")
        info = ctk.CTkFrame(top, fg_color="transparent")
        info.pack(side="left", fill="x", expand=True, padx=(12, 0))
        ctk.CTkLabel(info, text=("Incoming call" if ringing else "On a call") + f"  ·  {c.get('app') or 'Phone'}",
                     font=("Segoe UI", 11), text_color=self.C_ACCENT if ringing else self.C_TEXT2,
                     anchor="w").pack(anchor="w")
        ctk.CTkLabel(info, text=str(c.get("name") or "Unknown caller")[:40], font=("Segoe UI", 17, "bold"),
                     text_color=self.C_TEXT, anchor="w").pack(anchor="w")
        if c.get("text") and c.get("text") != c.get("name"):
            ctk.CTkLabel(info, text=str(c["text"])[:60], font=("Segoe UI", 12), text_color=self.C_TEXT2,
                         anchor="w").pack(anchor="w")
        row = ctk.CTkFrame(card, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=(8, 14))

        def act(action):
            if self.server and self.server.call_action(key, action):
                self.log({"decline": "📵 Call declined", "answer": "📞 Answered on the phone",
                          "hangup": "📵 Call ended"}.get(action, action))
            self._end_phone_call(key)

        if ringing:
            if c.get("can_decline", True):
                ctk.CTkButton(row, text="Decline", height=38, corner_radius=19, font=("Segoe UI", 13, "bold"),
                              fg_color="#d32f2f", hover_color="#b71c1c", text_color="#fff",
                              command=lambda: act("decline")).pack(side="left", expand=True, fill="x", padx=(0, 6))
            if c.get("can_answer", True):
                ctk.CTkButton(row, text="Answer", height=38, corner_radius=19, font=("Segoe UI", 13, "bold"),
                              fg_color="#2e7d32", hover_color="#1b5e20", text_color="#fff",
                              command=lambda: act("answer")).pack(side="left", expand=True, fill="x", padx=(6, 0))
        else:
            if c.get("can_hangup", True):
                ctk.CTkButton(row, text="Hang up", height=38, corner_radius=19, font=("Segoe UI", 13, "bold"),
                              fg_color="#d32f2f", hover_color="#b71c1c", text_color="#fff",
                              command=lambda: act("hangup")).pack(side="left", expand=True, fill="x")
            ctk.CTkButton(row, text="Hide", width=70, height=38, corner_radius=19, font=("Segoe UI", 12),
                          fg_color="transparent", hover_color="#1e1e1e", text_color=self.C_TEXT2,
                          border_width=1, border_color=self.C_BORDER,
                          command=lambda: self._end_phone_call(key)).pack(side="right", padx=(8, 0))
        popup.update_idletasks()
        w, h = 400, popup.winfo_reqheight()
        x = popup.winfo_screenwidth() - w - 16
        popup.geometry(f"{w}x{h}+{x}+16")
        self._call_popups[key] = popup
        if ringing:
            # The phone says when the call ends; this is only a safety net.
            popup.after(90_000, lambda: self._end_phone_call(key))

    def _add_recent(self, direction, name, path=None):
        """direction: "in" (from the phone) or "out" (to the phone)."""
        self._recent_items.insert(0, (direction, name, path, time.strftime("%H:%M")))
        del self._recent_items[8:]
        self._render_recent()

    def _render_recent(self):
        for child in self.recent_list.winfo_children():
            child.destroy()
        if not self._recent_items:
            ctk.CTkLabel(self.recent_list, text="Nothing yet. Files you send or receive show up here.",
                         font=("Segoe UI", 12), text_color="#555").pack(padx=14, pady=16)
            return
        for direction, name, path, when in self._recent_items:
            row = ctk.CTkFrame(self.recent_list, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=3)
            arrow, colour = ("⬇", self.C_ACCENT) if direction == "in" else ("⬆", "#64b5f6")
            ctk.CTkLabel(row, text=arrow, font=("Segoe UI", 13, "bold"), text_color=colour,
                         width=20).pack(side="left")
            ctk.CTkLabel(row, text=name if len(name) <= 48 else name[:45] + "…", font=("Segoe UI", 12),
                         text_color=self.C_TEXT, anchor="w").pack(side="left", padx=(6, 0))
            ctk.CTkLabel(row, text=("from phone" if direction == "in" else "to phone") + f" · {when}",
                         font=("Segoe UI", 11), text_color=self.C_TEXT2).pack(side="right")
            if path:
                ctk.CTkButton(row, text="Open", width=56, height=24, corner_radius=6,
                              font=("Segoe UI", 11), fg_color="transparent", hover_color="#1e1e1e",
                              text_color=self.C_ACCENT, border_width=1, border_color="#1f4d2e",
                              command=lambda p=path: self._open_received(p)).pack(side="right", padx=8)

    def _open_received(self, path):
        try:
            self.platform.open_path(path)
        except Exception as e:
            self.log(f"⚠️ Couldn't open {os.path.basename(str(path))}: {e}")

    # ── server logic ───────────────────────────────────────
    def toggle_server(self):
        if getattr(self, "_starting", False):
            return  # a start is in flight; clicking again used to launch a second server
        if not self.is_running:
            self._starting = True
            self.btn_start.configure(text="STOP", fg_color=self.C_ERR, hover_color="#cc3333",
                                      text_color="#fff", state="disabled")
            self.lbl_status.configure(text="STARTING…", text_color=self.C_WARN)
            self._status_dot.configure(text_color=self.C_WARN)
            self.lbl_status_sub.configure(text="Initialising server…")
            threading.Thread(target=self.start_sequence, daemon=True).start()
        else:
            self.stop_server()
 
    def stop_server(self):
        if self.server: self.server.stop()
        self.is_running = False
        self._starting = False
        self.btn_start.configure(text="START", fg_color=self.C_ACCENT,
                                  hover_color=self.C_ACCENT2, text_color="#000", state="normal")
        self.lbl_status.configure(text="OFFLINE", text_color=self.C_ERR)
        self._status_dot.configure(text_color=self.C_ERR)
        self.lbl_status_sub.configure(text="Server not running")
        self._set_ip("")
        for btn in [*self.vcam_chips.values(), self.btn_audio]:
            btn.configure(state="disabled", fg_color=self.C_CARD, text_color=self.C_TEXT2)
        self._update_client_badge(0)
 
    def _set_ip(self, val):
        self.ent_ip.configure(state="normal")
        self.ent_ip.delete(0, "end")
        if val:
            self.ent_ip.insert(0, f"ws://{val}")
        self.ent_ip.configure(state="readonly")
 
    def start_sequence(self):
        try:
            if self.server: self.server.stop()
            port = self.prefs.get("port", 8080)
            gaming_mode = self.prefs.get("gaming_mode", True)
            password = self.prefs.get("password", "useas")
            require_password = self.prefs.get("password_enabled", False)
            
            self.server = UnifiedRemoteServer(port=port, gaming_mode=gaming_mode, password=password,
                                              require_password=require_password, update_queue=self.update_queue,
                                              pairing_token=self.prefs.get("pairing_token", ""))
            self.server.virtual_display_enabled = bool(self.prefs.get("virtual_display", True))
            self.server.allow_file_browsing = bool(self.prefs.get("allow_file_browsing", True))
            self.server.allow_power_control = bool(self.prefs.get("allow_power_control", True))
            self.server.browse_folders = [str(f) for f in self.prefs.get("browse_folders", []) if f]
            self.server.allow_file_changes = bool(self.prefs.get("allow_file_changes", True))
            self.server.allow_deck_commands = bool(self.prefs.get("allow_deck_commands", True))
            self.server.start()
            for _ in range(20):
                if self.server._loop and self.server._loop.is_running():
                    ip = self.server.get_local_ip()
                    self.update_queue.put(("server_ready", f"{ip}:{port}"))
                    return
                time.sleep(0.1)
            raise Exception("Timeout waiting for server loop")
        except Exception as e:
            # Runs on a worker thread: Tk widgets must only be touched from the GUI thread,
            # so just stop the server object here; process_queue("error") resets the UI.
            try:
                if self.server: self.server.stop()
            except Exception:
                pass
            self.update_queue.put(("error", str(e)))
 
    def _vcam_device(self, label):
        return dict(self.vcam_options).get(label)

    def _vcam_label_for(self, device_name):
        """The phone asks for a device name; show it on the matching button."""
        for label, device in self.vcam_options:
            if device == device_name:
                return label
        return self.vcam_options[0][0] if self.vcam_options else None

    def toggle_camera(self, label):
        if not self.server: return
        if self.server._vcam_running:
            self.server.stop_virtual_camera()
            self._update_cam_buttons(None)
        else:
            self.server.start_virtual_camera(self._vcam_device(label))
            self.root.after(150, lambda: self._update_cam_buttons(label))

    def _update_cam_buttons(self, active_label):
        is_running = self.server and self.server._vcam_running
        if is_running and active_label in self.vcam_chips:
            self._active_vcam_label = active_label
            self._preview_badge.configure(text_color=self.C_ACCENT)
            for label in self.vcam_chips:
                if label == active_label:
                    self.vcam_studio_btns[label].configure(text=f"⏹  {label}", fg_color="#2a1a1a",
                                                           border_color="#5a2a2a", text_color=self.C_ERR)
                    self.vcam_chips[label].configure(text=f"⏹ {label}", fg_color="#2a1a1a", text_color=self.C_ERR)
                else:
                    self.vcam_studio_btns[label].configure(state="disabled", text_color=self.C_DIM)
                    self.vcam_chips[label].configure(state="disabled")
        else:
            self._active_vcam_label = None
            self._preview_badge.configure(text_color="#333")
            for label in self.vcam_chips:
                self.vcam_studio_btns[label].configure(state="normal", text=f"▶  {label}", fg_color="#1a2a1f",
                                                       border_color="#2a3d2d", text_color=self.C_ACCENT)
                self.vcam_chips[label].configure(text=f"📹  {label}", fg_color=self.C_CARD,
                                                 text_color=self.C_TEXT2,
                                                 state="normal" if self.is_running else "disabled")
 
    def toggle_audio(self):
        if not self.server: return
        if not self.server._streaming_audio:
            self.server.start_audio_streaming()
            self.btn_audio.configure(text="⏹ Audio Stream", fg_color="#2a1a1a", text_color=self.C_ERR)
        else:
            self.server.stop_audio_streaming()
            self.btn_audio.configure(text="🔊  Audio Stream", fg_color=self.C_CARD, text_color=self.C_TEXT2)
 
    def toggle_clipboard_autosync(self):
        enabled = bool(self.var_clip_autosync.get())
        self.prefs["clipboard_autosync"] = enabled
        self._write_prefs_file()          # no server restart needed
        if self.server:
            self.server.set_clipboard_autosync(enabled)
        self.log("📋 Clipboard auto-sync " + ("ON — text, images and copied files (small ones) "
                                              "go to the phone" if enabled else "OFF"))

    def send_rich_clipboard(self):
        if not self.server:
            self.log("⚠️ Start the server first")
            return
        threading.Thread(target=self.server.send_rich_clipboard, kwargs={"manual": True},
                         daemon=True).start()

    def send_text_to_phone(self):
        text = self.txt_clip.get("1.0", "end").strip()
        if text and self.server: self.server.send_to_android("clipboard_text", text)
 
    def copy_to_pc(self):
        pyperclip.copy(self.txt_clip.get("1.0", "end").strip())
 
    def send_file_pick(self):
        paths = filedialog.askopenfilenames()
        if paths:
            self.queue_files(paths)

    def queue_files(self, paths):
        """Send files to the phone now, or as soon as one is connected and paired."""
        files, skipped = [], 0
        for p in paths:
            p = os.path.abspath(str(p))
            if os.path.isfile(p):
                files.append(p)
            else:
                skipped += 1
        if skipped:
            self.log(f"⚠️ Skipped {skipped} item(s) that aren't files (folders can't be sent yet)")
        if not files:
            return
        self._pending_sends.extend(files)
        self._nav_go("sharing")
        self._flush_pending_sends()

    def _flush_pending_sends(self):
        if not self._pending_sends or self._sending:
            return
        if not (self.server and self.is_running and self.server._auth_targets()):
            n = len(self._pending_sends)
            if n != self._pending_notice:
                self._pending_notice = n
                self.log(f"📤 {n} file(s) will be sent as soon as a phone connects")
                self._show_transfer(f"Waiting for a phone… ({n} file(s) queued)", None)
            return
        files, self._pending_sends = self._pending_sends, []
        self._pending_notice = 0
        self._sending = True
        threading.Thread(target=self.process_multiple_files, args=(files,), daemon=True).start()

    def process_multiple_files(self, paths):
        try:
            for p in paths:
                self.server.send_file_to_phone_thread(p)
                time.sleep(0.3)
        finally:
            self.update_queue.put(("sending_done", None))

    # ── transfers / drag-and-drop / QR / IPC ───────────────
    def _setup_drop_target(self):
        if DND_FILES is None or not hasattr(self.root, "drop_target_register"):
            return
        try:
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind("<<Drop>>", self._on_drop)
        except Exception as e:
            print(f"Drag-and-drop unavailable: {e}")

    def _on_drop(self, event):
        try:
            paths = self.root.tk.splitlist(event.data)
        except Exception:
            paths = [event.data]
        self.queue_files(paths)
        return getattr(event, "action", None)

    @staticmethod
    def _fmt_size(n):
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024 or unit == "GB":
                return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
            n /= 1024

    def _show_transfer(self, text, fraction):
        if not self.transfer_row.winfo_ismapped():
            self.transfer_row.pack(fill="x", padx=18, pady=(10, 0), after=self._transfer_anchor)
        self.lbl_transfer.configure(text=text)
        if fraction is None:
            self.pb_transfer.configure(mode="indeterminate")
            self.pb_transfer.start()
        else:
            self.pb_transfer.stop()
            self.pb_transfer.configure(mode="determinate")
            self.pb_transfer.set(max(0.0, min(1.0, fraction)))
        if self._transfer_hide_job is not None:
            self.root.after_cancel(self._transfer_hide_job)
            self._transfer_hide_job = None

    def _on_transfer(self, t):
        arrow = "⬆ To phone" if t["dir"] == "out" else "⬇ From phone"
        total, done_bytes = t.get("total") or 0, t.get("bytes") or 0
        if t.get("done"):
            if t["dir"] == "out" and not t.get("failed"):
                self._add_recent("out", t["name"])
            text = f"{'❌ Failed' if t.get('failed') else '✅ Done'}: {t['name']}"
            self._show_transfer(text, 0 if t.get("failed") else 1.0)
            self._transfer_hide_job = self.root.after(4000, self._hide_transfer)
            return
        if total > 0:
            text = f"{arrow}: {t['name']} — {self._fmt_size(done_bytes)} / {self._fmt_size(total)}"
            self._show_transfer(text, done_bytes / total)
        else:   # size unknown (older phone app)
            self._show_transfer(f"{arrow}: {t['name']} — {self._fmt_size(done_bytes)}", None)

    def _hide_transfer(self):
        self._transfer_hide_job = None
        self.pb_transfer.stop()
        self.transfer_row.pack_forget()

    def open_qr_dialog(self):
        dlg = getattr(self, "_qr_dialog", None)
        if dlg is not None and dlg.winfo_exists():
            dlg.refresh()
            dlg.lift()
            return
        self._qr_dialog = QrPairDialog(self)

    def regenerate_pairing_token(self):
        self.prefs["pairing_token"] = pairing.new_token()
        self._write_prefs_file()
        if self.server:
            self.server.pairing_token = self.prefs["pairing_token"]
        self.log("📱 New pairing QR code created — the old one no longer works")

    def _on_ipc(self, msg):
        cmd = msg.get("cmd")
        if cmd == "send":
            files = msg.get("files") or []
            self.log(f"📤 Send to phone: {len(files)} file(s)")
            self.restore_window()
            self.queue_files(files)
        elif cmd == "show":
            self.restore_window()

    def restore_window(self):
        if self.tray_icon:
            self.restore_from_tray()
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
        except Exception:
            pass
 
    def open_folder(self):
        try:
            self.platform.open_path(SAVE_DIR)
        except Exception as e:
            self.log(f"⚠️ Couldn't open folder: {e}")
 
    def update_cam_settings(self):
        if self.server:
            self.server.bg_mode = self.var_bg.get()
            self.server.is_mirrored = self.var_mirror.get()
            self.server.is_flipped = self.var_flip.get()
            self.server.brightness_boost = int(self.scale_bright.get())
 
    def select_bg_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if path and self.server:
            self.server.bg_image_path = path
            self.server.bg_image_cache = None
            self.var_bg.set("image")
            self.update_cam_settings()
 
    def change_aspect_ratio(self, selection):
        if not self.server: return
        try:
            w, h = map(int, selection.split(" ")[0].split("x"))
            self.server.target_w = w
            self.server.target_h = h
            if self.server._vcam_running:
                label = self._active_vcam_label
                self.server.stop_virtual_camera()
                self.root.after(500, lambda: (self.server.start_virtual_camera(self._vcam_device(label)),
                                              self._update_cam_buttons(label)))
        except: pass
 
    def update_preview(self):
        if self.server and self.server.latest_preview_frame is not None:
            try:
                frame = self.server.latest_preview_frame
                
                # Ensure the window is actually visible to prevent math errors
                pw = self.lbl_preview.winfo_width()
                ph = self.lbl_preview.winfo_height()
                
                if pw > 10 and ph > 10:
                    ih, iw = frame.shape[:2]
                    scale = min(pw / iw, ph / ih)
                    nw, nh = int(iw * scale), int(ih * scale)
                    
                    if nw > 0 and nh > 0:
                        img = Image.fromarray(frame)
                        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
                        
                        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(nw, nh))
                        self.lbl_preview.configure(image=ctk_img, text="")
                        
                        # 🟢 CRITICAL: Prevent Python's garbage collector from deleting the frame
                        self.lbl_preview.image = ctk_img 
            except Exception as e:
                print(f"Preview Drawing Error: {e}")
                
        self.root.after(33, self.update_preview)
 
    def _update_client_badge(self, count):
        self._client_count = count
        if count == 0:
            self.lbl_clients.configure(text="● 0 connected", text_color=self.C_DIM)
        else:
            self.lbl_clients.configure(text=f"● {count} connected", text_color=self.C_ACCENT)
 
    def process_queue(self):
        try:
            while True:
                kind, data = self.update_queue.get_nowait()
 
                if kind == "log":
                    self.log(data)

                elif kind == "health":
                    self._render_health(data)

                elif kind == "phone_notification":
                    self._on_phone_notification(data)

                elif kind == "phone_call":
                    self._on_phone_call(data)

                elif kind == "phone_call_end":
                    self._end_phone_call(data)

                elif kind == "clip_history":
                    self._render_clip_history()

                elif kind == "phone_notification_removed":
                    self._remove_notification(data)
 
                elif kind == "server_ready":
                    self.is_running = True
                    self._starting = False
                    self.btn_start.configure(state="normal")
                    # A restart creates a new server object: re-apply the Camera Studio settings.
                    self.update_cam_settings()
                    self.server.set_clipboard_autosync(self.prefs.get("clipboard_autosync", False))
                    try:
                        w, h = map(int, self.combo_res.get().split(" ")[0].split("x"))
                        self.server.target_w, self.server.target_h = w, h
                    except Exception:
                        pass
                    self.lbl_status.configure(text="ONLINE", text_color=self.C_ACCENT)
                    self._status_dot.configure(text_color=self.C_ACCENT)
                    self.lbl_status_sub.configure(text="Accepting connections")
                    self._set_ip(data)
                    for btn in [*self.vcam_chips.values(), self.btn_audio]:
                        btn.configure(state="normal")
                    dlg = getattr(self, "_qr_dialog", None)
                    if dlg is not None and dlg.winfo_exists():
                        dlg.refresh()
                        
                # --- ADD THIS NEW BLOCK ---
                elif kind == "update_ip":
                    self._set_ip(data)
                # --------------------------
 
                elif kind == "client_count":
                    self._update_client_badge(data)
 
                elif kind == "error":
                    messagebox.showerror("Server Error", data)
                    self.stop_server()
 
                elif kind == "clipboard":
                    self.txt_clip.delete("1.0", "end")
                    self.txt_clip.insert("end", data)
                    self._update_clip_hint()
                    self._nav_go("sharing")
 
                elif kind == "file_received":
                    self._add_recent("in", os.path.basename(data), data)
                    
                elif kind == "update_endpoints":
                    self._update_endpoints(data)
 
                elif kind == "audio_status":
                    if data:
                        self.btn_audio.configure(text="⏹ Audio Stream",
                                                  fg_color="#2a1a1a", text_color=self.C_ERR)
                    else:
                        self.btn_audio.configure(text="🔊  Audio Stream",
                                                  fg_color=self.C_CARD, text_color=self.C_TEXT2)
 
                elif kind == "camera_status":
                    self._update_cam_buttons(self._vcam_label_for(data) if data else None)

                elif kind == "client_authenticated":
                    self._flush_pending_sends()

                elif kind == "sending_done":
                    self._sending = False
                    self._flush_pending_sends()

                elif kind == "transfer":
                    self._on_transfer(data)

                elif kind == "ipc":
                    self._on_ipc(data)
 
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Queue Error: {e}")
 
        self.root.after(150, self.process_queue)
 
    def log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_box.insert("end", f"[{ts}]  {msg}\n")
        self.log_box.see("end")
 
    def on_closing(self):
        if self.is_running: self.stop_server()
        self.root.destroy()
        
        
    
    

