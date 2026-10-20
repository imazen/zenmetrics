#!/usr/bin/env python3
"""Enable perf after the frozen worker's existing completed-warmup event.

No timers or counters are added to a metric. The recording includes native
benchmark calibration/rounds, and excludes construction and initial warmup.
Run this process under the same resource admission as the comparison.
"""
import argparse
import json
import os
from pathlib import Path
import select
import subprocess

from compare_canonical import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('binary',type=Path);p.add_argument('config',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--frequency',type=int,default=199)
    a=p.parse_args();a.output.mkdir();root=a.output.resolve()
    control,ack=[root/name for name in ['control.fifo','ack.fifo']]
    for path in [control,ack]:os.mkfifo(path)
    ctl_fd=os.open(control,os.O_RDWR|os.O_NONBLOCK)
    ack_fd=os.open(ack,os.O_RDWR|os.O_NONBLOCK)
    env=dict(os.environ,ZENBENCH_NO_SAVE='1',ZENBENCH_CPU_POLICY='unrestricted')
    command=['perf','record','-F',str(a.frequency),'--call-graph','dwarf','--delay=-1',
             '--control=fifo:'+str(control)+','+str(ack),'-o',str(root/'perf.data'),'--',
             str(a.binary.resolve()),str(a.config.resolve())]
    enabled=disabled=False
    def send(message):
        os.write(ctl_fd,(message+'\n').encode())
        ready,_,_=select.select([ack_fd],[],[],5)
        if not ready or b'ack' not in os.read(ack_fd,1024):
            raise ValueError('perf did not acknowledge '+message)
    try:
        with (root/'stdout.log').open('x') as out,(root/'stderr.log').open('x') as err:
            with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=err,text=True,env=env) as child:
                for line in child.stdout:
                    out.write(line);out.flush()
                    if line.startswith('IQA_WARMUP '):
                        if enabled or not json.loads(line.split(' ',1)[1])['completed']:
                            raise ValueError('unexpected native warmup event')
                        send('enable');enabled=True
                    elif line.startswith('IQA_ROW ') and enabled:
                        send('disable');disabled=True
                rc=child.wait()
        if rc or not enabled or not disabled:
            raise ValueError(f'incomplete warm profile: rc={rc}, enabled={enabled}, disabled={disabled}')
        with (root/'report.txt').open('x') as stream:
            subprocess.run(['perf','report','--stdio','--no-children','-i',str(root/'perf.data')],stdout=stream,check=True)
        (root/'receipt.json').write_text(json.dumps(dict(command=command,binary_sha256=sha(a.binary),
            config_sha256=sha(a.config),profile_sha256=sha(root/'perf.data'),report_sha256=sha(root/'report.txt'),
            recording='Enabled after existing IQA_WARMUP completed event; disabled after IQA_ROW; includes benchmark calibration and measured rounds'),indent=2)+'\n')
        print(str(root/'receipt.json'))
    finally:
        os.close(ctl_fd);os.close(ack_fd)
        control.unlink();ack.unlink()


if __name__=='__main__':main()
