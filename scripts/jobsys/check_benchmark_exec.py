#!/usr/bin/env python3
"""Real runner/store/adapter controls. Supply a built zenbench and new output dir."""
import importlib.util
from importlib.machinery import SourceFileLoader
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

adapter = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else Path(__file__).with_name('benchmark_exec.py').resolve()
spec = importlib.util.spec_from_loader('benchmark_exec', SourceFileLoader('benchmark_exec', str(adapter)))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
runner, root = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
root.mkdir(parents=True, exist_ok=False)
os.environ.update(ZENBENCH_BIN=str(runner), ZENBENCH_WORK_DIR=str(root),
                  ZEN_BLOBS_DIR=str(root/'blobs'), ZENBENCH_MACHINE='control-cpu',
                  ZENBENCH_DOMAIN='unrestricted', ZENBENCH_CPU_POLICY='unrestricted',
                  ZENBENCH_COORDINATION_DIR=str(root/'coordination'))
os.environ.pop('ZENBENCH_BLOB_URI', None)
# These synthetic adapter fixtures select unrestricted placement explicitly;
# an inherited checked-policy CPU request must not contradict that selection.
os.environ.pop('ZENBENCH_CPU_SET', None)
(root/'coordination').mkdir()
caps, _ = module.capabilities(str(runner))
driver = root/'driver.py'
driver.write_text('''import json, os, time
from pathlib import Path
if os.environ.get('CONTROL_SLEEP') == '1':
    time.sleep(20)
data = bytearray(1024 * 1024)
assert data[1024] == 0
Path(os.environ['ZENBENCH_RESULT_FILE']).write_text(json.dumps({
    'schema':1,'operations':1,'correctness':'ok','memory':[]}))
''')
records = []
# Shared CI runners can stay busy briefly after compilation. Admission has its
# own budget; keep the two-second child timeout and every outcome assertion.
admission_seconds = 30
program = 'python.exe' if os.name == 'nt' else 'python'
target_os = 'darwin' if caps['runner']['os'] == 'macos' else caps['runner']['os']
for index, (backend, expected, sleep, status) in enumerate([
    ('wall', 'ok', False, 'complete'),
    ('os_peak', 'ok', False, 'complete'),
    ('wall', 'wrong', False, 'wrong_output'),
    ('wall', 'ok', True, 'timeout'),
    ('wall', 'ok', True, 'cancelled'),
]):
    case = root/f'case-{index}'
    case.mkdir()
    bundle = case/'bundle'
    plan = {'schema':1,'program':program,'args':['driver.py'],
            'env':{'CONTROL_SLEEP':'1' if sleep else '0', 'PYTHONHOME':sys.base_prefix},'backend':backend,
            'operations':1,'timeout_seconds':2,'expected_correctness':expected,
            'workload':{'fixture':'adapter-control','input':'synthetic-1MiB'}}
    bundle_spec = {'schema':1,'target':f"{caps['runner']['process_arch']}-test-{target_os}",
                   'plan':plan,'files':{
                       program:{'source':sys.executable,'executable':True},
                       'driver.py':{'source':str(driver),'executable':False}}}
    source = case/'spec.json'; source.write_text(json.dumps(bundle_spec))
    subprocess.run([str(runner),'bundle',str(source),str(bundle)],check=True,stdout=subprocess.DEVNULL)
    hashes = module.stage(bundle)
    kind = dict(kind='benchmark', **hashes, experiment='adapter-control', machine='control-cpu',
                domain='unrestricted', replicate=index, backend=backend,
                timeout_seconds=2, admission_seconds=admission_seconds)
    job = {'kind':kind,'inputs':sorted(hashes.values())}
    if index == 0:
        first_job = json.loads(json.dumps(job))
    receipt_file = case/'receipt.json'
    with receipt_file.open('wb') as out, (case/'adapter.stderr').open('wb') as err:
        process = subprocess.Popen([sys.executable,str(adapter)],stdin=subprocess.PIPE,stdout=out,stderr=err)
        process.stdin.write(json.dumps(job).encode()); process.stdin.close()
        if status == 'cancelled':
            deadline = time.monotonic()+15
            while True:
                started = list((root/'attempts').glob('*/started.json'))
                if any(json.loads(p.read_bytes())['replicate'] == index for p in started):
                    break
                assert process.poll() is None, (case/'adapter.stderr').read_text()
                assert time.monotonic() < deadline, 'runner did not start'
                time.sleep(.02)
            if os.name == 'nt':
                attempt = next(json.loads(p.read_bytes())['attempt'] for p in started
                               if json.loads(p.read_bytes())['replicate'] == index)
                subprocess.run([sys.executable,str(adapter),'cancel',attempt],check=True)
            else:
                process.send_signal(signal.SIGTERM)
        wait_seconds = kind['admission_seconds'] + kind['timeout_seconds'] + 30
        assert process.wait(timeout=wait_seconds) == 0, (case/'adapter.stderr').read_text()
    receipt = json.loads(receipt_file.read_bytes())
    assert receipt['status'] == status, receipt
    assert receipt['job_id'] == module.canonical_job_id(kind,job['inputs'])
    recovered = case/'recovered'
    module.collect(str(receipt_file), recovered)
    packet = json.loads((recovered/'packet.json').read_bytes())
    assert packet['status'] == status
    assert (recovered/'runner.stderr').is_file()
    assert not list((root/'coordination').glob('*.json')), 'reservation leaked'
    if backend == 'os_peak':
        assert packet['memory'][0]['bytes'] >= 1024*1024, packet
    records.append({'status':status,'backend':backend,'job_id':receipt['job_id'],
                    'packet':receipt['packet']})
# Routing mistakes must refuse before creating another attempt.
kind['machine'] = 'different-cpu'
try:
    module.execute({'kind':kind,'inputs':sorted(hashes.values())})
except ValueError as error:
    assert 'worker does not satisfy' in str(error), error
else:
    raise AssertionError('wrong machine accepted')
(root/'controls.json').write_text(json.dumps(records,indent=2)+'\n')
if len(sys.argv) > 4:
    first_job['kind']['replicate'] = 100
    first_job['kind']['admission_seconds'] = 30
    first_job['cell'] = {'image_path':'control','codec':'benchmark','q':0,'knob_tuple_json':'{}'}
    first_job['hint'] = {'peak_mem_bytes':256*1024*1024,'threads':1,'vram_bytes':None}
    _, capabilities = module.capabilities(str(runner))
    first_job['requires'] = sorted(capabilities)
    manifest = root/'manifest.json'; manifest.write_text(json.dumps([first_job]))
    ledger = root/'ledger'; ledger.mkdir()
    command = [sys.argv[4],'--manifest',str(manifest),'--ledger-out',str(ledger/'result.parquet'),
               '--blobs',str(root/'blobs'),'--exec',str(adapter.with_name('benchmark-exec.cmd') if os.name == 'nt' else adapter)]
    env = dict(os.environ,ZEN_PERSISTENT_EXEC='1',ZEN_WARM_KINDS='all',ZEN_CHUNK_WALL_SEC='300')
    result = subprocess.run(command,env=env,capture_output=True,timeout=60)
    (root/'worker.stdout').write_bytes(result.stdout); (root/'worker.stderr').write_bytes(result.stderr)
    assert result.returncode == 0, result.stderr.decode()
    if len(sys.argv) > 5:
        decoded = subprocess.run([sys.argv[5],str(ledger)],capture_output=True,check=True,timeout=30)
        rows = json.loads(decoded.stdout)
    else:
        import pyarrow.parquet as pq
        rows = [r for p in ledger.glob('*.parquet') for r in pq.read_table(p).to_pylist()]
    completed = [r for r in rows if r['status']=='done']
    assert len(completed)==1, rows
    receipt = json.loads(module.Store(root).get(completed[0]['output_sha']).read_bytes())
    assert receipt['status']=='complete', receipt
    assert receipt['job_id']==module.canonical_job_id(first_job['kind'],first_job['inputs'])
    module.collect(completed[0]['output_sha'],root/'worker-recovered')
    print(json.dumps({'worker_ledger':'done','observation_status':receipt['status'],
                      'persistent_all':'fresh benchmark succeeded'}))
print(json.dumps({'controls':records,'wrong_machine':'refused'}))
