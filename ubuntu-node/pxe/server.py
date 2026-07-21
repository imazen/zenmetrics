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
for d in ("flags", "registry", "inventory", "seen"):
    os.makedirs(os.path.join(STATE, d), exist_ok=True)


def _mac(s):
    s = s.lower()
    return s if MAC_RE.match(s) else None


def _has(kind, mac):     return os.path.exists(os.path.join(STATE, "flags", f"{mac}.{kind}"))
def _clear(kind, mac):
    try: os.remove(os.path.join(STATE, "flags", f"{mac}.{kind}"))
    except FileNotFoundError: pass
def _registered(mac):    return os.path.exists(os.path.join(STATE, "registry", f"{mac}.install-user-data"))


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

def ipxe_inventory(mac):
    # NOTE: no 'autoinstall' keyword -> the installer never runs; our seed's runcmd
    # dumps hardware and powers the box off. Read-only by construction.
    return f"""#!ipxe
echo zen-pxe: {mac} -> INVENTORY (read-only, powers off; touches no disk)
{_kernel_line(mac, 'inventory', autoinstall=False)}
initrd {BASE}/{KVER}/initrd
boot
"""

def ipxe_install(mac):
    return f"""#!ipxe
echo zen-pxe: {mac} -> INSTALL to registered disk serial (serial-matched)
{_kernel_line(mac, 'install', autoinstall=True)}
initrd {BASE}/{KVER}/initrd
boot
"""


def boot_script(mac):
    _seen(mac)
    if _has("inventory", mac):
        return ipxe_inventory(mac)
    if _has("install", mac) and _registered(mac):
        return ipxe_install(mac)
    return ipxe_localboot(mac)


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
        else:  # done
            _clear("install", mac)
            return self._text("install flag cleared\n")

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        m = re.match(r"^/api/boot/([0-9a-f-]+)$", path)
        if m:
            mac = _mac(m.group(1))
            return self._text(boot_script(mac) if mac else "#!ipxe\nexit\n")
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
