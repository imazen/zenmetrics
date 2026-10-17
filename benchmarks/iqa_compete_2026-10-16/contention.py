"""Linux process accounting inside the shared benchmark lock (not a profiler)."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path


def snapshot():
    processes={}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            raw=(entry/'stat').read_text();end=raw.rindex(')');fields=raw[end+2:].split()
            pid=int(entry.name)
            processes[(pid,int(fields[19]))]=dict(pid=pid,ppid=int(fields[1]),pgid=int(fields[2]),
                name=raw[raw.index('(')+1:end],ticks=int(fields[11])+int(fields[12]))
        except (OSError,ValueError,IndexError):continue
    return time.monotonic(),processes


def process_view():
    raw=subprocess.check_output(['ps','-eo','pid=,ppid=,pgid=,pcpu=,comm='],text=True)
    rows=[]
    for line in raw.splitlines():
        fields=line.strip().split(None,4)
        if len(fields)==5:
            if int(fields[1])==os.getpid():continue
            rows.append(dict(pid=int(fields[0]),ppid=int(fields[1]),pgid=int(fields[2]),lifetime_cpu_percent=float(fields[3]),name=fields[4]))
    return rows

def interval(first,last,hz):
    elapsed=last[0]-first[0]
    return [dict(value,cores=max(0,value['ticks']-first[1][key]['ticks'])/hz/elapsed)
            for key,value in last[1].items() if key in first[1] and elapsed>0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt',type=Path);parser.add_argument('log',type=Path)
    parser.add_argument('command',nargs=argparse.REMAINDER);args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    hz=os.sysconf('SC_CLK_TCK');view_before=process_view();before=snapshot();load_before=list(os.getloadavg())
    owned={os.getpid()};samples=[];durations=[];previous=before
    inherited=(int(os.environ['IQA_HEAVY_LOCK_FD']),) if os.environ.get('IQA_HEAVY_LOCK_FD') else ()
    process=subprocess.Popen(command,pass_fds=inherited)
    while True:
        try:code=process.wait(timeout=1)
        except subprocess.TimeoutExpired:code=None
        current=snapshot()
        changed=True
        while changed:
            additions={v['pid'] for v in current[1].values() if v['ppid'] in owned}-owned
            changed=bool(additions);owned.update(additions)
        samples.append(interval(previous,current,hz));durations.append(current[0]-previous[0]);previous=current
        if code is not None:break
    load_after=list(os.getloadavg());view_after=process_view()
    # systemd scope children may be reparented; the driver publishes its identity.
    groups=set()
    if args.log.exists():
        for line in args.log.read_text().splitlines():
            if line.startswith('IQA_PROCESS '):
                proof=json.loads(line[len('IQA_PROCESS '):]);owned.add(proof['pid']);groups.add(proof['pgid'])
    def competing(values):
        return sorted((v for v in values if v['pid'] not in owned and v['pgid'] not in groups),key=lambda v:v['cores'],reverse=True)
    peaks={}
    for values in samples:
        for value in competing(values):
            key=(value['pid'],value['name']);peaks[key]=max(peaks.get(key,0),value['cores'])
    accumulated={}
    for values,duration in zip(samples,durations):
        for value in competing(values):
            key=(value['pid'],value['name'])
            old=accumulated.get(key,dict(value,cores=0))
            old['cores']+=value['cores']*duration/(previous[0]-before[0]);accumulated[key]=old
    total=sorted(accumulated.values(),key=lambda v:v['cores'],reverse=True)
    # Retry if a competitor exceeds one core in any sampled interval.
    worst=max((v['cores'] for v in total),default=0)
    result=dict(protocol='locked-process-accounting-v1',supervisor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),load_before=load_before,load_after=load_after,
        elapsed_seconds=previous[0]-before[0],cpu_tick_hz=hz,top_competing_processes=total[:20],
        top_competing_processes_before=sorted((v for v in view_before if v['pid'] not in owned and v['pgid'] not in groups),key=lambda v:v['lifetime_cpu_percent'],reverse=True)[:20],
        top_competing_processes_after=sorted((v for v in view_after if v['pid'] not in owned and v['pgid'] not in groups),key=lambda v:v['lifetime_cpu_percent'],reverse=True)[:20],
        top_competing_processes_first_interval=competing(samples[0])[:20],top_competing_processes_last_interval=competing(samples[-1])[:20],
        interval_peak_cores=[dict(pid=k[0],name=k[1],cores=v) for k,v in sorted(peaks.items(),key=lambda item:item[1],reverse=True)[:20]],
        max_competing_mean_cores=worst,max_competing_interval_cores=max(peaks.values(),default=0),contaminated=max(peaks.values(),default=0)>1,exit_code=code)
    args.receipt.write_text(json.dumps(result,sort_keys=True)+'\n')
    raise SystemExit(code)

if __name__=='__main__':main()
