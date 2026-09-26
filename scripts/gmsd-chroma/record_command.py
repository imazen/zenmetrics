import datetime, hashlib, json, os, pathlib, subprocess, sys
root = pathlib.Path('/var/tmp/gmsd-chroma')
start = datetime.datetime.now(datetime.timezone.utc).isoformat()
name = sys.argv[1]
cmd = sys.argv[2:]
log = root / 'logs' / (name + '.log')
if log.exists():
    raise SystemExit('refusing to overwrite log: ' + str(log))
env = os.environ.copy()
env.update(TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'), CARGO_TARGET_DIR=str(root / 'target'), CARGO_HOME=str(root / 'cache' / 'cargo'), UV_CACHE_DIR=str(root / 'cache' / 'uv'), PYTHONDONTWRITEBYTECODE='1')
with log.open('wb') as out:
    result = subprocess.run(cmd, env=env, stdout=out, stderr=subprocess.STDOUT)
end = datetime.datetime.now(datetime.timezone.utc).isoformat()
digest = hashlib.sha256(log.read_bytes()).hexdigest()
record = dict(start=start, end=end, cwd=os.getcwd(), argv=cmd, exit_code=result.returncode, output=str(log), sha256=digest)
with (root / 'commands.jsonl').open('a') as f:
    f.write(json.dumps(record) + '\n')
manifest = pathlib.Path('/home/lilith/tmp/devin/rev4_gmsd-chroma_manifest.tsv')
with manifest.open('a') as f:
    for path in (log, root / 'commands.jsonl'):
        f.write(end + '\t' + str(path) + '\tcreate-or-modify\n')
print(json.dumps(record))
print(log.read_text(errors='replace'))
sys.exit(result.returncode)
