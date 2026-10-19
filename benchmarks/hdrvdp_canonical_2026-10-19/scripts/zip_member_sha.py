import zipfile, hashlib, sys
names = set(l.strip() for l in open(sys.argv[2]) if l.strip())
z = zipfile.ZipFile(sys.argv[1])
found = 0
for info in z.infolist():
    base = info.filename.rsplit('/', 1)[-1]
    if base in names:
        h = hashlib.sha256()
        with z.open(info) as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        print(h.hexdigest() + '  ' + base + '\t' + info.filename + '\t' + '%04d-%02d-%02d %02d:%02d:%02d' % info.date_time)
        found += 1
print('# found', found, 'of', len(names), file=sys.stderr)
