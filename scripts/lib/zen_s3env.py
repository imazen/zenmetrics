"""zen_s3env — THE resolver for the object-store endpoint + credentials (Python side).

Mirror of scripts/lib/s3env.sh. Use instead of re-deriving endpoints:

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "lib"))
    from zen_s3env import resolve
    ep, ak, sk = resolve()

Contract (ONE env var switches the store):

    ZEN_S3_ENDPOINT unset -> Cloudflare R2; endpoint derived from R2_ACCOUNT_ID in
                             ~/.config/cloudflare/r2-credentials (legacy behavior,
                             unchanged).
    ZEN_S3_ENDPOINT set   -> that endpoint verbatim; creds from
                             ZEN_S3_ACCESS_KEY_ID / ZEN_S3_SECRET_ACCESS_KEY, else
                             from $ZEN_S3_ENV (default ~/.config/zen/lanstore.env).

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


def resolve() -> tuple[str, str, str]:
    """Return (endpoint, access_key_id, secret_access_key) for the selected store."""
    ep = os.environ.get("ZEN_S3_ENDPOINT")
    if ep:
        ak = os.environ.get("ZEN_S3_ACCESS_KEY_ID")
        sk = os.environ.get("ZEN_S3_SECRET_ACCESS_KEY")
        if not (ak and sk):
            envfile = os.environ.get(
                "ZEN_S3_ENV", os.path.expanduser("~/.config/zen/lanstore.env")
            )
            filed = _load_env_file(envfile)
            ak = ak or filed.get("ZEN_S3_ACCESS_KEY_ID")
            sk = sk or filed.get("ZEN_S3_SECRET_ACCESS_KEY")
        if not (ak and sk):
            raise SystemExit(
                "ZEN_S3_ENDPOINT is set but ZEN_S3_ACCESS_KEY_ID/ZEN_S3_SECRET_ACCESS_KEY "
                "are not (export them or provide the ZEN_S3_ENV file)"
            )
        return ep, ak, sk

    env = dict(os.environ)
    if "R2_ACCOUNT_ID" not in env:
        env.update(_load_env_file(os.path.expanduser("~/.config/cloudflare/r2-credentials")))
    try:
        acct = env["R2_ACCOUNT_ID"]
        ak = env["R2_ACCESS_KEY_ID"]
        sk = env["R2_SECRET_ACCESS_KEY"]
    except KeyError as e:
        raise SystemExit(
            f"no ZEN_S3_ENDPOINT and {e.args[0]} missing (need ~/.config/cloudflare/r2-credentials)"
        ) from None
    return f"https://{acct}.r2.cloudflarestorage.com", ak, sk


def export_env() -> str:
    """Resolve and inject AWS_* + EP into os.environ; return the endpoint."""
    ep, ak, sk = resolve()
    os.environ["AWS_ACCESS_KEY_ID"] = ak
    os.environ["AWS_SECRET_ACCESS_KEY"] = sk
    os.environ.setdefault("AWS_REGION", "auto")
    os.environ["EP"] = ep
    return ep
