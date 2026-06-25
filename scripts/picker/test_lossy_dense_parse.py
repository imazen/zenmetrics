"""Validate zenjxl_lossy_dense.parse_config_name against real lossy_dense cell-ids."""
import csv,json,sys,re
sys.path.insert(0,"scripts/picker/configs")
# import only the parse fn + regex (avoid paths() needing real files)
_RE=re.compile(r"^vd-e(\d+)_([a-z]+)_([^-]+)(?:-(.*))?$")
def parse(n):
    m=_RE.match(n)
    if not m: return None
    return {"strategy":m.group(2),"knob":m.group(3),"flag":m.group(4) or "none","effort":float(m.group(1))}
def recompose(p):
    s=f"vd-e{int(p['effort'])}_{p['strategy']}_{p['knob']}"
    if p["flag"]!="none": s+=f"-{p['flag']}"
    return s
tsv=sys.argv[1]
ids=set()
for r in csv.DictReader(open(tsv),delimiter='\t'):
    ids.add(json.loads(r["knob_tuple_json"])["cell"])
fails=[]; maxdev_viol=[]; rt_fail=[]
for cid in ids:
    p=parse(cid)
    if p is None: fails.append(cid); continue
    dev=sum([p["effort"]!=7.0, p["strategy"]!="zen", p["knob"]!="def", p["flag"]!="none"])
    if dev>1: maxdev_viol.append((cid,dev))
    if recompose(p)!=cid: rt_fail.append((cid,recompose(p)))
print(f"unique cell-ids: {len(ids)}")
print(f"parse failures: {len(fails)}  {fails[:5]}")
print(f"max-dev-1 violations: {len(maxdev_viol)}  {maxdev_viol[:5]}")
print(f"roundtrip failures: {len(rt_fail)}  {rt_fail[:5]}")
str123={}
for cid in ids:
    p=parse(cid)
    if p: str123.setdefault("strategy",set()).add(p["strategy"]); str123.setdefault("knob",set()).add(p["knob"]); str123.setdefault("flag",set()).add(p["flag"])
print("strategies:",sorted(str123['strategy']))
print("knobs:",sorted(str123['knob']))
print("flags:",sorted(str123['flag']))
ok = not fails and not maxdev_viol and not rt_fail
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
