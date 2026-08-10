"""zen_s3env — THE resolver for the object-store endpoint + credentials (Python side).

Mirror of scripts/lib/s3env.sh. Use instead of re-deriving endpoints:

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "lib"))
    from zen_s3env import resolve
    ep, ak, sk = resolve()

Contract — `ZEN_STORE` selects the store; the LAN store is the DEFAULT.

    ZEN_STORE unset/"tower"/"lan" -> the LAN store (DEFAULT as of 2026-08-10, user
                             directive). Endpoint + creds from ZEN_S3_ENDPOINT /
                             ZEN_S3_ACCESS_KEY_ID / ZEN_S3_SECRET_ACCESS_KEY, else
                             from $ZEN_S3_ENV (default ~/.config/zen/lanstore.env).
    ZEN_STORE="r2"        -> Cloudflare R2; endpoint derived from R2_ACCOUNT_ID in
                             ~/.config/cloudflare/r2-credentials. EXPLICIT opt-out,
                             REQUIRED for anything that launches/feeds cloud boxes.
    ZEN_S3_ENDPOINT set   -> that endpoint verbatim, whatever ZEN_STORE says.

BEFORE 2026-08-10 the default was R2. The flip means a caller that selects nothing
now writes to the LAN store. `resolve_full()` returns the resolved store kind and
whether cloud boxes can route to it — branch on THAT, never on "is ZEN_S3_ENDPOINT
set" (that test silently inverted when the default flipped).

Fail-loud on missing creds for the selected store — never a silent fall-through.
"""

from __future__ import annotations

import os


def _load_env_file(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip()
    except OSError:
        pass
    return out


def resolve_full() -> tuple[str, str, str, str, bool]:
    """Return (endpoint, access_key_id, secret_access_key, store_kind, cloud_reachable).

    store_kind is "lan" | "r2" | "explicit". cloud_reachable says whether a cloud box
    can route to the endpoint — gate cloud launches on it, not on env-var presence.
    """
    forced = os.environ.get("ZEN_S3_CLOUD_REACHABLE")
    # Precedence: explicit ZEN_STORE=r2 wins over everything (reliable escape hatch even when
    # ZEN_S3_ENDPOINT is exported ambiently); then explicit ZEN_S3_ENDPOINT; then the default.
    sel = os.environ.get("ZEN_STORE") or "tower"
    if sel.lower() != "r2" and os.environ.get("ZEN_S3_ENDPOINT"):
        sel = "explicit"

    if sel.lower() == "r2":
        env = dict(os.environ)
        if "R2_ACCOUNT_ID" not in env:
            env.update(_load_env_file(os.path.expanduser("~/.config/cloudflare/r2-credentials")))
        try:
            acct = env["R2_ACCOUNT_ID"]
            ak = env["R2_ACCESS_KEY_ID"]
            sk = env["R2_SECRET_ACCESS_KEY"]
        except KeyError as e:
            raise SystemExit(
                f"ZEN_STORE=r2 but {e.args[0]} missing (need ~/.config/cloudflare/r2-credentials)"
            ) from None
        reach = True if forced is None else forced == "1"
        return f"https://{acct}.r2.cloudflarestorage.com", ak, sk, "r2", reach

    # LAN store (the default) or an explicit endpoint.
    ep = os.environ.get("ZEN_S3_ENDPOINT")
    ak = os.environ.get("ZEN_S3_ACCESS_KEY_ID")
    sk = os.environ.get("ZEN_S3_SECRET_ACCESS_KEY")
    if not (ep and ak and sk):
        envfile = os.environ.get("ZEN_S3_ENV", os.path.expanduser("~/.config/zen/lanstore.env"))
        filed = _load_env_file(envfile)
        ep = ep or filed.get("ZEN_S3_ENDPOINT")
        ak = ak or filed.get("ZEN_S3_ACCESS_KEY_ID")
        sk = sk or filed.get("ZEN_S3_SECRET_ACCESS_KEY")
    if not ep:
        raise SystemExit(
            f"LAN store selected (ZEN_STORE={os.environ.get('ZEN_STORE', 'tower')}) but no "
            "ZEN_S3_ENDPOINT and no lanstore env file — set ZEN_STORE=r2 to use Cloudflare R2"
        )
    if not (ak and sk):
        raise SystemExit(
            "LAN store selected but ZEN_S3_ACCESS_KEY_ID/ZEN_S3_SECRET_ACCESS_KEY are not set "
            "(export them or provide the ZEN_S3_ENV file)"
        )
    reach = False if forced is None else forced == "1"
    return ep, ak, sk, sel if sel == "explicit" else "lan", reach


def resolve() -> tuple[str, str, str]:
    """Return (endpoint, access_key_id, secret_access_key) for the selected store."""
    ep, ak, sk, _kind, _reach = resolve_full()
    return ep, ak, sk


def export_env() -> str:
    """Resolve and inject AWS_* + EP into os.environ; return the endpoint."""
    ep, ak, sk, kind, reach = resolve_full()
    os.environ["AWS_ACCESS_KEY_ID"] = ak
    os.environ["AWS_SECRET_ACCESS_KEY"] = sk
    os.environ.setdefault("AWS_REGION", "auto")
    os.environ["EP"] = ep
    os.environ["ZEN_S3_STORE"] = kind
    os.environ["ZEN_S3_CLOUD_REACHABLE"] = "1" if reach else "0"
    return ep
