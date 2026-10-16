"""Retain selected kernel disassembly and report actual packed arithmetic."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('binary',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
text=subprocess.check_output(['objdump','-Cd',str(a.binary)],text=True)
selected=[];functions={}
for block in re.split(r'(?m)(?=^[0-9a-f]+ <)',text):
    name=block.splitlines()[0] if block else ''
    if not any(key in name for key in ['butteraugli_original::convolve','butteraugli_original::malta','cambi::update_histogram_run','cambi::calculate_c_values_columns','cambi::__arcane_calculate_c_values_row']):continue
    selected.append(block)
    instructions=re.findall(r'^\s*[0-9a-f]+:\s+(?:[0-9a-f]{2}\s+)+\s*([a-z0-9]+)',block,re.M)
    counts=Counter(instructions)
    functions[name]={'packed_arithmetic':{key:counts[key] for key in ['addps','mulps','vaddps','vmulps','paddw','psubw','vpaddw','vpsubw'] if counts[key]},'fma_instructions':{key:value for key,value in counts.items() if key.startswith(('vfmadd','vfmsub','vfnmadd','vfnmsub'))}}
assert selected,'No selected kernel symbols found'
raw=''.join(selected)
with a.output.with_suffix('.asm').open('x') as f:f.write(raw)
receipt={'binary':str(a.binary),'binary_sha256':hashlib.sha256(a.binary.read_bytes()).hexdigest(),'disassembly_path':str(a.output.with_suffix('.asm')),'disassembly_sha256':hashlib.sha256(raw.encode()).hexdigest(),'functions':functions,'counts_are_static_instruction_sites':True}
with a.output.with_suffix('.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt,indent=2))
