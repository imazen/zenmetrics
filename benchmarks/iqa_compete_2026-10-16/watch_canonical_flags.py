#!/usr/bin/env python3
"""Audit new completed cases while an existing comparison queue runs."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path);p.add_argument('output',type=Path);p.add_argument('regression_log',type=Path)
    p.add_argument('progress_log',type=Path);p.add_argument('coordinator_pid',type=int)
    a=p.parse_args();here=Path(__file__).resolve().parent
    command=[sys.executable,'-B',str(here/'audit_canonical_flags.py'),str(a.manifest),str(a.output),str(a.regression_log)]
    subprocess.run(command,check=True)
    with a.progress_log.open() as log:
        log.seek(0,os.SEEK_END)
        while True:
            lines=log.readlines()
            if any(line.startswith('CASE_DONE ') for line in lines):subprocess.run(command,check=True)
            try:os.kill(a.coordinator_pid,0)
            except ProcessLookupError:
                subprocess.run(command,check=True);return
            time.sleep(.5)


if __name__=='__main__':main()
