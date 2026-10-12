#!/usr/bin/env python3
"""Real runner/store/adapter controls. Supply a built zenbench and new output dir."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

adapter = Path(__file__).with_name('benchmark_exec.py').resolve()
spec = importlib.util.spec_from_file_location('benchmark_exec', adapter)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
runner, root = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
root.mkdir(parents=True, exist_ok=False)
os.environ.update(ZENBENCH_BIN=str(runner), ZENBENCH_WORK_DIR=str(root),
                  ZEN_BLOBS_DIR=str(root/'blobs'), ZENBENCH_MACHINE='control-cpu',
                  ZENBENCH_DOMAIN='unrestricted', ZENBENCH_CPU_POLICY='unrestricted',
                  ZENBENCH_COORDINATION_DIR=str(root/'coordination'))
os.environ.pop('ZENBENCH_BLOB_URI', None)
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
    plan = {'schema':1,'program':'python','args':['driver.py'],
            'env':{'CONTROL_SLEEP':'1' if sleep else '0'},'backend':backend,
            'operations':1,'timeout_seconds':2,'expected_correctness':expected,
            'workload':{'fixture':'adapter-control','input':'synthetic-1MiB'}}
    bundle_spec = {'schema':1,'target':f"{caps['runner']['process_arch']}-test-{caps['runner']['os']}",
                   'plan':plan,'files':{
                       'python':{'source':sys.executable,'executable':True},
                       'driver.py':{'source':str(driver),'executable':False}}}
    source = case/'spec.json'; source.write_text(json.dumps(bundle_spec))
    subprocess.run([str(runner),'bundle',str(source),str(bundle)],check=True,stdout=subprocess.DEVNULL)
    hashes = module.stage(bundle)
    kind = dict(kind='benchmark', **hashes, experiment='adapter-control', machine='control-cpu',
                domain='unrestricted', replicate=index, backend=backend,
                timeout_seconds=2, admission_seconds=2)
    job = {'kind':kind,'inputs':sorted(hashes.values())}
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
            process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=30) == 0, (case/'adapter.stderr').read_text()
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
print(json.dumps({'controls':records,'wrong_machine':'refused'}))
