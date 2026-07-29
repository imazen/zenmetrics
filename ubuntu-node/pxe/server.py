#!/usr/bin/env python3
"""zen-pxe control service — the brain of the network-install fleet.

nginx front-ends this: it serves the static netboot assets (kernel/initrd/ISO, with
native HTTP Range) and reverse-proxies /api/ here. This service owns only the dynamic
endpoints that encode the four safety layers so we NEVER wipe the wrong disk:

  GET  /api/boot/<mac>              -> the iPXE script for this box. DEFAULT is
                                       local-disk boot. Installs only when the box
                                       is explicitly opted in (a flag file) AND its
                                       target disk serial is registered.
  GET  /api/seed/<mac>/<kind>/user-data|meta-data
                                     -> cloud-init/autoinstall seed. kind=inventory
                                        (read-only, powers off before touching disk)
                                        or install (the real, serial-matched install).
  POST /api/inventory/<mac>         -> a box reports its disks/NICs; we save it and
                                        clear the inventory flag (one-shot).
  POST /api/done/<mac>              -> the installer calls this before first reboot to
                                        clear its own install flag (install-once, no
                                        re-wipe loop).
  GET  /api/choose/<mac>/<serial>   -> console menu: arm an install for a disk the operator
                                        picked and confirmed AT THE CONSOLE, then boot it.
                                        Rejects any serial not in that box's inventory.
  GET  /api/scan/<mac>              -> console menu: read-only hardware probe that reports
                                        its disks and reboots back to the menu.

Boot decision (safe by construction):
  inventory flag present            -> inventory boot (never touches storage)
  install flag present AND registered serial present -> install (that serial only)
  worker flag present               -> boot the installed Ubuntu node
  unknown box (or `menu` flag)      -> INTERACTIVE menu, default local boot
  otherwise (a box we manage)       -> local-disk boot

Nothing installs without either a flag set from the dev box or two deliberate console
picks, and no path can wipe a disk whose serial isn't the registered one.

State lives under $PXE_DATA/state/{flags,registry,inventory,seen}; static assets
under $PXE_DATA/http. stdlib only — nothing to pip-install.
"""
import os, re, sys, time, json, shlex, subprocess, http.server, socketserver, urllib.parse

DATA = os.environ.get("PXE_DATA", "/data")
HTTP = os.path.join(DATA, "http")
STATE = os.path.join(DATA, "state")
BASE = os.environ.get("PXE_BASE", "http://192.168.50.170:3080")  # how boxes reach us
ISO = os.environ.get("PXE_ISO", "iso/ubuntu-26.04-live-server-amd64.iso")
KVER = os.environ.get("PXE_UBU", "ubuntu-26.04")
PORT = int(os.environ.get("PXE_PORT", "3081"))   # nginx (3080) reverse-proxies /api/ here
MENU_TIMEOUT = int(os.environ.get("PXE_MENU_TIMEOUT", "30"))   # seconds before the SAFE default wins
RENDER = os.environ.get("PXE_RENDER", os.path.join(DATA, "render_install.py"))
DEFAULTS = os.environ.get("PXE_DEFAULTS", os.path.join(DATA, "install-defaults.env"))

MAC_RE = re.compile(r"^[0-9a-f]{2}(-[0-9a-f]{2}){5}$")   # iPXE ${mac:hexhyp} form
for d in ("flags", "registry", "inventory", "seen", "installed"):
    os.makedirs(os.path.join(STATE, d), exist_ok=True)


def _mac(s):
    s = s.lower().replace(":", "-")     # accept iPXE hexhyp (aa-bb-..) AND GRUB's colon form
    return s if MAC_RE.match(s) else None

HOSTPORT = BASE.split("://", 1)[-1]     # e.g. 192.168.50.170:3080 (for GRUB's (http,host:port) device)


def _has(kind, mac):     return os.path.exists(os.path.join(STATE, "flags", f"{mac}.{kind}"))
def _clear(kind, mac):
    try: os.remove(os.path.join(STATE, "flags", f"{mac}.{kind}"))
    except FileNotFoundError: pass
def _registered(mac):    return os.path.exists(os.path.join(STATE, "registry", f"{mac}.install-user-data"))
def _installed(mac):     return os.path.exists(os.path.join(STATE, "installed", mac))  # loop guard


def _seen(mac):
    try:
        with open(os.path.join(STATE, "seen", mac), "w") as f:
            f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    except OSError:
        pass


# ---- iPXE scripts --------------------------------------------------------------
def ipxe_localboot(mac):
    return f"""#!ipxe
echo zen-pxe: {mac} -> local disk (no install flag)
sanboot --no-describe --drive 0x80 || exit
"""

def _kernel_line(mac, kind, autoinstall):
    ai = "autoinstall " if autoinstall else ""
    return (f"kernel {BASE}/{KVER}/vmlinuz initrd=initrd ip=dhcp "
            f"url={BASE}/{ISO} {ai}cloud-config-url=/dev/null "
            f"ds=nocloud-net;s={BASE}/api/seed/{mac}/{kind}/ ---")

def ipxe_via_grub(mac, kind):
    # Hand off to GRUB for the kernel-boot cases: many UEFI firmwares don't honor iPXE's
    # EFI initrd (LoadFile2) handoff, so the kernel boots with no initrd and panics. GRUB
    # loads the initrd into memory itself (classic boot_params method) which works on all
    # firmware. GRUB re-asks /api/grub/<mac>, so the flag logic stays server-side.
    return f"""#!ipxe
echo zen-pxe: {mac} -> {kind} via GRUB (firmware-safe initrd handoff)
chain --replace {BASE}/grubnet.efi || shell
"""


def boot_script(mac):
    _seen(mac)
    if _has("inventory", mac):
        return ipxe_via_grub(mac, "INVENTORY")
    if _has("install", mac) and _registered(mac):
        return ipxe_via_grub(mac, "INSTALL")
    return ipxe_localboot(mac)


# ---- GRUB configs (the actual kernel/initrd boot; robust initrd handoff) --------
def _grub_kernel(mac, kind, autoinstall):
    ai = "autoinstall " if autoinstall else ""
    # quote the ds= value so GRUB doesn't split on its ';' (GRUB strips the quotes for the kernel)
    return (f"linux (http,{HOSTPORT})/{KVER}/vmlinuz ip=dhcp url={BASE}/{ISO} "
            f'{ai}cloud-config-url=/dev/null "ds=nocloud-net;s={BASE}/api/seed/{mac}/{kind}/" ---\n'
            f"initrd (http,{HOSTPORT})/{KVER}/initrd\n"
            f"boot")

def _grub_chain(*paths):
    # `exit` doesn't reliably hand off to the local disk on all firmware, so chainload the
    # OS bootloader directly, trying each path in order across all local disks.
    out = []
    for p in paths:
        out.append(f"if search --no-floppy --file --set=root {p}\nthen chainloader {p}\nboot\nfi")
    out.append("echo zen-grub: no matching local bootloader found")
    return "\n".join(out) + "\n"

_UBUNTU = ("/EFI/ubuntu/shimx64.efi", "/EFI/ubuntu/grubx64.efi", "/EFI/BOOT/BOOTX64.EFI")
_WINDOWS = ("/EFI/Microsoft/Boot/bootmgfw.efi",)


# ---- interactive console menu (unknown boxes only) -----------------------------
# Lets someone AT THE CONSOLE provision a brand-new box — pick its OS disk and confirm —
# without the dev box. Safety, in order of importance:
#   1. Only UNKNOWN boxes get it (not registered, not installed) or an explicit `menu` flag.
#      Managed boxes (jason/ian/i265) keep the old instant chainload — they PXE on EVERY
#      boot, so a menu there would delay every kid's boot AND park a wipe option in reach.
#   2. The DEFAULT entry is local boot and the timeout picks it, so an unattended or
#      walked-away-from PXE boot can never install.
#   3. Choosing a disk only opens a submenu; the submenu's default is "back", and the
#      confirm entry spells out device/serial/size/model.
#   4. /api/choose re-validates the serial against THIS mac's own inventory, so a typo or
#      hand-crafted URL can't target an unseen disk; the autoinstall's serial match then
#      makes wiping the wrong disk impossible (it errors out instead of guessing).
_SIZE_RE = re.compile(r"^\d+(\.\d+)?[KMGTPE]?$")
_TRAN = {"nvme", "sata", "usb", "ata", "scsi", "sas", "virtio", "mmc"}
_SKIP_DEV = re.compile(r"^/dev/(loop|sr|ram|fd|zram|dm-)")


def _parse_disks_text(text):
    """Parse `lsblk -dpno NAME,SERIAL,WWN,SIZE,MODEL,TRAN` output (the pre-2026-07-29 format).

    Columns are space-padded and MODEL contains spaces, so anchor on the first size-looking
    token instead of splitting positionally. Disks with no serial are DROPPED: the install
    matches on serial, so one we can't name is one we must not offer.
    """
    out = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln.startswith("/dev/") or _SKIP_DEV.match(ln):
            continue
        t = ln.split()
        si = next((i for i in range(1, len(t)) if _SIZE_RE.match(t[i])), None)
        if si is None or si < 2:      # si<2 => no serial column at all
            continue
        tran = t[-1].lower() if t[-1].lower() in _TRAN else ""
        model = " ".join(t[si + 1:len(t) - 1 if tran else len(t)]).strip()
        out.append({"dev": t[0], "serial": t[1], "size": t[si], "model": model or "?", "tran": tran})
    return out


def _disks(mac):
    """Disks for this box, newest format first.

    Scans since 2026-07-29 append a ---DISKS-JSON--- section (`lsblk -dJ`), which is
    unambiguous; older reports (ian/jason/i265) only have the padded text table, so fall
    back to parsing that.
    """
    try:
        with open(os.path.join(STATE, "inventory", f"{mac}.txt"), "r", errors="replace") as f:
            text = f.read()
    except OSError:
        return []
    if "---DISKS-JSON---" in text:
        blob = text.split("---DISKS-JSON---", 1)[1].split("---", 1)[0].strip()
        try:
            out = []
            for d in json.loads(blob).get("blockdevices", []):
                dev = d.get("name", "")
                dev = dev if dev.startswith("/dev/") else f"/dev/{dev}"
                if _SKIP_DEV.match(dev) or not (d.get("serial") or "").strip():
                    continue
                out.append({"dev": dev, "serial": d["serial"].strip(),
                            "size": d.get("size") or "?", "model": (d.get("model") or "?").strip(),
                            "tran": (d.get("tran") or "").strip()})
            if out:
                return out
        except (ValueError, KeyError, TypeError):
            pass          # malformed JSON -> fall through to the text parser
    return _parse_disks_text(text)


def _gsafe(s, n=60):
    """GRUB menu titles are double-quoted; drop what would break or expand inside them."""
    s = "".join(c for c in str(s) if c.isprintable() and c not in '"\\$`{}')
    return s[:n].strip()


def grub_menu(mac, disks):
    """Interactive console menu: default = local boot, install needs two deliberate picks."""
    url = f"(http,{HOSTPORT})/api"
    t = [f"echo zen-grub: {mac} INTERACTIVE (no flag set; default is local boot)",
         "set menu_color_highlight=black/light-gray",
         f"set timeout={MENU_TIMEOUT}", "set default=0",
         'menuentry "Boot from local disk  [DEFAULT]" {',
         _grub_chain(*_WINDOWS, *_UBUNTU).rstrip("\n"), "}"]
    for d in disks:
        # Keep each line inside an 80-col console: the operator reads these to decide what
        # gets erased, so a title truncated mid-word is a safety problem, not a cosmetic one.
        label = _gsafe(f"{d['dev']}  {d['size']}  {_gsafe(d['model'], 24)}"
                       + (f" ({d['tran']})" if d["tran"] else ""), 52)
        ser = _gsafe(d["serial"], 24)
        t += [f'submenu "Install Ubuntu to {label}" {{',
              '  menuentry "NO - go back (nothing is changed)" {',
              f"    configfile {url}/grub/{mac}", "  }",
              f'  menuentry "YES - ERASE serial {ser} - ALL DATA ON IT IS LOST" {{',
              f"    echo Arming install for {ser} ...",
              f"    configfile {url}/choose/{mac}/{urllib.parse.quote(d['serial'], safe='')}", "  }",
              f'  menuentry "   (target: {label})" {{',
              f"    configfile {url}/grub/{mac}", "  }", "}"]
    t += ['menuentry "Scan this hardware (read-only; reboots back to this menu)" {',
          f"  configfile {url}/scan/{mac}", "}"]
    if not disks:
        t += ['menuentry "(no disks known yet - run the hardware scan above first)" { true }']
    return "\n".join(t) + "\n"


def _defaults():
    """Read install-defaults.env (shell-style KEY='value'), shared with fleet-pxe register."""
    out = {}
    try:
        with open(DEFAULTS, "r") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#") or "=" not in ln:
                    continue
                k, v = ln.split("=", 1)
                try:
                    parts = shlex.split(v)
                except ValueError:
                    continue
                out[k.strip()] = parts[0] if parts else ""
    except OSError:
        pass
    return out


def _auto_host(mac):
    return "zen-" + mac.replace("-", "")[-6:]      # 60-cf-84-76-20-d2 -> zen-7620d2

def grub_config(mac):
    _seen(mac)   # grub is now the direct entry point; keep 'seen' tracking here
    if _has("inventory", mac):
        return f"echo zen-grub: {mac} INVENTORY (read-only)\n{_grub_kernel(mac, 'inventory', False)}\n"
    if _has("install", mac) and _registered(mac) and not _installed(mac):
        # guard 2: even if the flag lingers, once the box has POSTed /api/done we refuse to
        # re-install (no wipe loop). Re-arming via `fleet-pxe install` clears the marker.
        return f"echo zen-grub: {mac} INSTALL (serial-matched)\n{_grub_kernel(mac, 'install', True)}\n"
    if _has("worker", mac):
        # remotely-toggled worker mode: boot the installed Ubuntu zen node
        return "echo zen-grub: WORKER mode -- booting Ubuntu zen node\n" + _grub_chain(*_UBUNTU, *_WINDOWS)
    if _has("menu", mac) or not (_registered(mac) or _installed(mac)):
        # UNKNOWN box (or explicitly opted in): offer the interactive console installer.
        # Managed boxes fall through to the instant chainload below — they PXE on every
        # boot, so a menu there would delay each one and expose a wipe option.
        return grub_menu(mac, _disks(mac))
    # default: boot Windows (user choice: Windows stays the default OS)
    return "echo zen-grub: booting Windows (default)\n" + _grub_chain(*_WINDOWS, *_UBUNTU)


# ---- interactive actions (GRUB fetches these; a GET here has side effects) ------
def choose(mac, serial):
    """Arm an install for a console-chosen disk, then boot straight into it.

    Refuses any serial that isn't in THIS box's own inventory — that check is what keeps a
    mistyped or hand-made URL from targeting a disk we've never seen.
    """
    disks = _disks(mac)
    hit = next((d for d in disks if d["serial"] == serial), None)
    if not hit:
        return (f"echo zen-grub: REFUSED - serial '{_gsafe(serial, 40)}' is not in {mac}'s inventory\n"
                f"echo Nothing was changed. Re-run the hardware scan, then pick a listed disk.\n"
                f"sleep 10\nconfigfile (http,{HOSTPORT})/api/grub/{mac}\n")
    d = _defaults()
    host = _auto_host(mac)
    env = dict(os.environ)
    env.update({"SERIAL": hit["serial"], "HOST": host, "MAC": mac,
                "CRED": "",                      # tower can't mint: no-cred install (see render_install.py)
                "R2EP": d.get("R2_ENDPOINT", ""), "KEY": d.get("SSH_PUBKEY", ""),
                "HASH": d.get("PW_HASH", ""), "IMG": d.get("WORKER_IMAGE", ""),
                "TOWER_IP": d.get("TOWER_IP", BASE.split("//", 1)[-1].split(":")[0])})
    try:
        r = subprocess.run([sys.executable, RENDER], env=env, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        r = None
        err = str(e)
    if r is None or r.returncode != 0 or not r.stdout.strip():
        err = err if r is None else (r.stderr.strip() or "renderer produced no output")
        sys.stderr.write(f"choose {mac}: render failed: {err}\n")
        return (f"echo zen-grub: install NOT armed - the tower could not render a config\n"
                f"echo {_gsafe(err, 120)}\nsleep 15\nconfigfile (http,{HOSTPORT})/api/grub/{mac}\n")
    with open(os.path.join(STATE, "registry", f"{mac}.install-user-data"), "w") as f:
        f.write(r.stdout)
    with open(os.path.join(STATE, "registry", f"{mac}.env"), "w") as f:
        f.write(f"HOST={host}\nSERIAL={hit['serial']}\nSOURCE=console-menu\n")
    try:                                  # a deliberate re-pick clears the install-once guard
        os.remove(os.path.join(STATE, "installed", mac))
    except FileNotFoundError:
        pass
    with open(os.path.join(STATE, "flags", f"{mac}.install"), "w") as f:
        f.write(f"console-menu {hit['serial']} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    sys.stderr.write(f"choose {mac}: armed install host={host} serial={hit['serial']} (console menu)\n")
    return (f"echo zen-grub: {mac} INSTALL armed for {_gsafe(hit['dev'])} [{_gsafe(hit['serial'], 40)}] as {host}\n"
            f"echo Wiping ONLY that serial. Starting the installer ...\nsleep 3\n"
            f"{_grub_kernel(mac, 'install', True)}\n")


def scan(mac):
    """Boot the read-only hardware probe, which reports its disks and reboots to the menu."""
    return (f"echo zen-grub: {mac} hardware scan (read-only; storage is never touched)\n"
            f"{_grub_kernel(mac, 'rescan', False)}\n")


# ---- seeds ---------------------------------------------------------------------
def seed_inventory_userdata(mac, after="poweroff"):
    # Plain cloud-init (NOT autoinstall): report hardware, then power off (fleet-pxe's flow)
    # or reboot (the console menu's "scan", so the box returns to the menu with its disks
    # listed). The live installer sits idle on tty1 while cloud-init runs this; storage is
    # never touched. The ---DISKS-JSON--- section is the machine-readable disk list the
    # menu parses; the padded text table above it stays for humans and older reports.
    post = f"{BASE}/api/inventory/{mac}"
    probe = ("echo MAC=%s; echo; lsblk -dpno NAME,SERIAL,WWN,SIZE,MODEL,TRAN 2>/dev/null; "
             "echo ---DISKS-JSON---; lsblk -dJ -o NAME,SERIAL,SIZE,MODEL,TRAN 2>/dev/null; "
             "echo ---NICS---; ip -o link 2>/dev/null") % mac
    return f"""#cloud-config
runcmd:
  - [ sh, -c, "curl -sS -X POST --data-binary \\"$({probe})\\" {post} || true" ]
power_state:
  mode: {after}
  timeout: 30
  condition: true
"""


def seed_install_userdata(mac):
    p = os.path.join(STATE, "registry", f"{mac}.install-user-data")
    with open(p, "r") as f:
        return f.read()


def seed(mac, kind, name):
    if name == "meta-data":
        return f"instance-id: zen-{mac}\n"
    if name != "user-data":
        return None
    if kind == "inventory":
        return seed_inventory_userdata(mac)
    if kind == "rescan":               # console-menu scan: come back to the menu, don't power off
        return seed_inventory_userdata(mac, after="reboot")
    if kind == "install" and _registered(mac):
        return seed_install_userdata(mac)
    return None


# ---- HTTP ----------------------------------------------------------------------
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _text(self, body, code=200, ctype="text/plain; charset=utf-8"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(b)

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        m = re.match(r"^/api/(inventory|done)/([0-9a-f-]+)$", path)
        if not m:
            return self._text("not found\n", 404)
        action, mac = m.group(1), _mac(m.group(2))
        if not mac:
            return self._text("bad mac\n", 400)
        n = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(n) if n else b""
        if action == "inventory":
            with open(os.path.join(STATE, "inventory", f"{mac}.txt"), "wb") as f:
                f.write(body)
            _clear("inventory", mac)
            return self._text("inventory saved\n")
        else:  # done — installer finished: clear the flag AND drop a loop-guard marker
            _clear("install", mac)
            with open(os.path.join(STATE, "installed", mac), "w") as f:
                f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
            return self._text("install flag cleared\n")

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        m = re.match(r"^/api/boot/([0-9a-f-]+)$", path)
        if m:
            mac = _mac(m.group(1))
            return self._text(boot_script(mac) if mac else "#!ipxe\nexit\n")
        m = re.match(r"^/api/grub/([0-9a-f:\-]+)$", path)   # GRUB asks in colon form
        if m:
            mac = _mac(m.group(1))
            return self._text(grub_config(mac) if mac else "echo bad mac; sleep 3; reboot\n")
        m = re.match(r"^/api/(choose)/([0-9a-f:\-]+)/(.+)$", path)   # console menu: arm + boot install
        if m:
            mac = _mac(m.group(2))
            if not mac:
                return self._text("echo bad mac; sleep 3; reboot\n")
            return self._text(choose(mac, urllib.parse.unquote(m.group(3))))
        m = re.match(r"^/api/scan/([0-9a-f:\-]+)$", path)            # console menu: read-only probe
        if m:
            mac = _mac(m.group(1))
            return self._text(scan(mac) if mac else "echo bad mac; sleep 3; reboot\n")
        m = re.match(r"^/api/seed/([0-9a-f-]+)/(inventory|rescan|install)/(user-data|meta-data)$", path)
        if m:
            mac = _mac(m.group(1))
            s = seed(mac, m.group(2), m.group(3)) if mac else None
            return self._text(s, 200) if s is not None else self._text("no seed\n", 404)
        if path == "/api/status":
            def ls(d): return sorted(os.listdir(os.path.join(STATE, d)))
            import json
            return self._text(json.dumps({k: ls(k) for k in ("seen", "flags", "registry", "inventory")}, indent=2) + "\n",
                              ctype="application/json")
        # static assets are served by nginx, not here
        return self._text("not found\n", 404)


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    print(f"zen-pxe on :{PORT}  data={DATA}  base={BASE}", flush=True)
    Server(("0.0.0.0", PORT), H).serve_forever()
