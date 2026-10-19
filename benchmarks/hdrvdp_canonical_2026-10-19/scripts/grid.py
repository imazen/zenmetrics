"""Run an explicit JSON list of oracle argv arrays; every run retains its receipt.
Usage: grid.py GRID_JSON; each element is the nine required oracle arguments plus options.
"""
import json
from pathlib import Path
import subprocess
import sys

runner=Path(__file__).with_name('run_oracle.py')
for args in json.loads(Path(sys.argv[1]).read_text()):
    assert isinstance(args,list) and len(args)>=9
    result=subprocess.run([sys.executable,str(runner),*map(str,args)])
    if result.returncode:
        raise SystemExit(result.returncode)
