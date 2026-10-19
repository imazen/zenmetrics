import struct, sys, collections
summary = collections.Counter()
for p in sys.argv[1:]:
    b = open(p, 'rb').read()
    assert b[:8] == b'\x89PNG\r\n\x1a\n'
    i = 8; chunks = []
    ihdr = None
    while i < len(b):
        n, = struct.unpack('>I', b[i:i+4]); t = b[i+4:i+8].decode('latin1')
        if t == 'IHDR':
            w, h, bd, ct, cm, fm, il = struct.unpack('>IIBBBBB', b[i+8:i+8+13]); ihdr = (w, h, bd, ct, il)
        if t not in ('IDAT',): chunks.append(t)
        i += 12 + n
    summary[(ihdr[2:], tuple(c for c in chunks if c not in ('IHDR', 'IEND')))] += 1
    if '-v' in sys.argv[0]: print(p, ihdr, chunks)
for k, v in summary.items(): print(v, 'bitdepth,colortype,interlace=', k[0], 'chunks=', k[1])
