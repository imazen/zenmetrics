#!/usr/bin/env python3
"""LANSTORE conditional-create gate: the zenfleet claim-lease primitive.

Gate (from homefleet zenmetrics/tower/LANSTORE.md): a two-writer `PUT If-None-Match: *`
race on ONE key must yield exactly one 200 and one PreconditionFailed(412).
"""
import sys, boto3, botocore, concurrent.futures as cf
from botocore.config import Config

ep, ak, sk, label = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
BUCKET = "zengate"
s3 = boto3.client("s3", endpoint_url=ep, aws_access_key_id=ak, aws_secret_access_key=sk,
                  region_name="us-east-1",
                  config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                                retries={"max_attempts": 0}))
try: s3.create_bucket(Bucket=BUCKET)
except Exception: pass

def cput(key, body):
    try:
        s3.put_object(Bucket=BUCKET, Key=key, Body=body, IfNoneMatch="*")
        return "200"
    except botocore.exceptions.ClientError as e:
        return e.response["Error"].get("Code", "ERR")
    except Exception as e:
        return type(e).__name__

print(f"=== {label} ({ep}) ===")

# 1. sequential: create then re-create
k1 = "seq/key1"
try: s3.delete_object(Bucket=BUCKET, Key=k1)
except Exception: pass
a = cput(k1, b"first"); b = cput(k1, b"second")
seq_ok = (a == "200" and b in ("PreconditionFailed", "412"))
print(f"  sequential : first={a} second={b}   -> {'PASS' if seq_ok else 'FAIL'}")

# 2. the real gate: N concurrent creators of ONE key -> exactly 1 winner
N = 8
race_ok = True
for trial in range(5):
    k = f"race/key{trial}"
    try: s3.delete_object(Bucket=BUCKET, Key=k)
    except Exception: pass
    with cf.ThreadPoolExecutor(max_workers=N) as ex:
        res = list(ex.map(lambda i: cput(k, b"w%d" % i), range(N)))
    wins = res.count("200")
    pf = sum(1 for r in res if r in ("PreconditionFailed", "412"))
    ok = (wins == 1 and pf == N - 1)
    race_ok &= ok
    print(f"  race t{trial} : {N} writers -> {wins} win / {pf} precondition-failed "
          f"/ other={[r for r in res if r not in ('200','PreconditionFailed','412')]}  -> {'PASS' if ok else 'FAIL'}")

verdict = "PASS" if (seq_ok and race_ok) else "FAIL"
print(f"  VERDICT: {verdict}  (exactly-once claim lease {'usable' if verdict=='PASS' else 'NOT usable'})")
sys.exit(0 if verdict == "PASS" else 1)
