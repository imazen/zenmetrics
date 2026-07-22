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

Boot decision (safe by construction):
  inventory flag present            -> inventory boot (never touches storage)
  install flag present AND registered serial present -> install (that serial only)
  otherwise                         -> local-disk boot

State lives under $PXE_DATA/state/{flags,registry,inventory,seen}; static assets
under $PXE_DATA/http. stdlib only — nothing to pip-install.
"""
import os, re, sys, time, http.server, socketserver, urllib.parse

DATA = os.environ.get("PXE_DATA", "/data")
HTTP = os.path.join(DATA, "http")
STATE = os.path.join(DATA, "state")
BASE = os.environ.get("PXE_BASE", "http://192.168.50.170:3080")  # how boxes reach us
ISO = os.environ.get("PXE_ISO", "iso/ubuntu-26.04-live-server-amd64.iso")
KVER = os.environ.get("PXE_UBU", "ubuntu-26.04")
PORT = int(os.environ.get("PXE_PORT", "3081"))   # nginx (3080) reverse-proxies /api/ here

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
    # default: boot Windows (user choice: Windows stays the default OS)
    return "echo zen-grub: booting Windows (default)\n" + _grub_chain(*_WINDOWS, *_UBUNTU)


# ---- seeds ---------------------------------------------------------------------
def seed_inventory_userdata(mac):
    # Plain cloud-init (NOT autoinstall): report hardware, then power off. The live
    # installer sits idle on tty1 while cloud-init runs this; storage is never touched.
    post = f"{BASE}/api/inventory/{mac}"
    return f"""#cloud-config
runcmd:
  - [ sh, -c, "curl -sS -X POST --data-binary \\"$(echo MAC={mac}; echo; lsblk -dpno NAME,SERIAL,WWN,SIZE,MODEL,TRAN 2>/dev/null; echo ---NICS---; ip -o link 2>/dev/null)\\" {post} || true" ]
power_state:
  mode: poweroff
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
        m = re.match(r"^/api/seed/([0-9a-f-]+)/(inventory|install)/(user-data|meta-data)$", path)
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
