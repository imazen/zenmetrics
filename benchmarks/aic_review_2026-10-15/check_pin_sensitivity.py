"""Negative controls for committed oracle pins; only mutate scratch copies."""
import pathlib,subprocess,json
root=pathlib.Path.cwd();out=pathlib.Path.home()/'tmp/aic-variants-scratch/round5/pin-controls';out.mkdir(exist_ok=True)
controls=[
 ('butteraugli_original','0.688059627878','0.689059627878','mask-response constant'),
 ('ssimulacra1','sum += row[0] + row[1] + row[2] + row[3];','for &v in row { sum += v; }','previous sequential 4x4 sum'),
 ('ws_hv','column += similarity * weight;','column = f64::from(column as f32 + (similarity * weight) as f32);','f32 pool reduction'),
]
rows=[]
for name,old,new,label in controls:
 text=(root/f'crates/zenmetrics-cli/src/metrics/{name}.rs').read_text();assert old in text
 source=out/f'{name}.rs';source.write_text(text.replace(old,new));wrapper=out/f'{name}_test.rs';wrapper.write_text(f'#[path = "{source}"]\nmod metric;\n')
 binary=out/name
 subprocess.run(['rustc','--edition','2024','--test','-O',str(wrapper),'-o',str(binary)],check=True)
 result=subprocess.run([str(binary),'metric::oracle_tests::original_author_synthetic_pin','--exact'],capture_output=True,text=True)
 (out/f'{name}.log').write_text(result.stdout+result.stderr)
 assert result.returncode != 0 and 'assertion failed' in result.stdout+result.stderr, label+' did not trip pin'
 rows.append(dict(metric=name,mutation=label,returncode=result.returncode,pin_failed=True))
with (out/'summary.json').open('x') as f:json.dump(rows,f,indent=2)
print(json.dumps(rows,indent=2))
