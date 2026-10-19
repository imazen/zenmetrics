import csv, sys, math
def load(p):
    delim = '\t' if p.endswith('.tab') else ','
    rows = list(csv.DictReader(open(p, newline=''), delimiter=delim))
    key = lambda d: d.strip('"').rsplit('.', 1)[0].replace('PTC_', '')
    return {key(r['distorted']): r for r in rows}, list(rows[0].keys())
paths = sys.argv[1:]
data = {p: load(p) for p in paths}
for p in paths:
    d, cols = data[p]
    print(p.split('/tmp/')[-1], 'rows', len(d), 'cols', len(cols), 'HDR' if 'HDR_VDP_2' in cols else 'noHDR')
base = paths[0]
bd, _ = data[base]
for p in paths[1:]:
    d, cols = data[p]
    for col in ('HDR_VDP_2', 'HDR_VDP_3'):
        if col not in cols: continue
        common = [k for k in d if k in bd]
        diff = 0; strdiff = 0; maxd = 0.0; ex = None
        for k in common:
            a, b = bd[k][col].strip('"'), d[k][col].strip('"')
            if a != b:
                strdiff += 1
                try:
                    dd = abs(float(a) - float(b))
                except ValueError:
                    dd = float('nan')
                if not (dd == 0): diff += 1
                if dd > maxd or ex is None: maxd = max(maxd, dd); ex = (k, a, b)
        print(f'  {col}: {p.split("/tmp/")[-1]} vs base: common {len(common)} string-diff {strdiff} value-diff {diff} max|d| {maxd} ex {ex}')
