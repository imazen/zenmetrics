"""Emit a separate benchmark driver copy; never edit a shared producer."""
import argparse
import hashlib
import json
from pathlib import Path


def patched(source):
    needle = '''        run_metric_display(kind, r, d, GpuRuntime::Cpu, Some(&display))
            .map(|v| v[0].1)
            .map_err(|e| e.to_string())'''
    replacement = '''        run_metric_display(kind, r, d, GpuRuntime::Cpu, Some(&display))
            .map_err(|e| e.to_string())
            .and_then(|v| {
                if kind == MetricKind::Brisque {
                    nr_observer::record(&v)
                } else { Ok(v[0].1) }
            })'''
    marker = '        row["ref_score"] = json!(first);'
    insertion = '''        if name == "brisque" {
            let (distorted, reference, calls) = nr_observer::result()?;
            row["nr_result_schema"] = json!("paired-nr-v1");
            row["nr_scores"] = json!({"brisque": distorted, "brisque_ref": reference});
            row["nr_observed_calls"] = json!(calls);
            row["nr_results_stable"] = json!(true);
        }
'''
    if source.count(needle)!=1 or source.count(marker)!=1:
        raise ValueError('benchmark source shape changed')
    return 'mod nr_observer;\n'+source.replace(needle,replacement).replace(marker,insertion+marker)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    source=a.source.read_bytes(); target=a.output/'iqa_compete.rs'
    # Inner crate docs/attributes must precede module declarations.
    result=patched(source.decode()).replace('mod nr_observer;\n','',1)
    anchor='fn make_call('
    result=result.replace(anchor,'mod nr_observer;\n\n'+anchor,1)
    target.write_text(result)
    observer=Path(__file__).with_name('nr_observer.rs').read_bytes()
    (a.output/'nr_observer.rs').write_bytes(observer)
    sha=lambda b:hashlib.sha256(b).hexdigest()
    (a.output/'overlay.json').write_text(json.dumps(dict(schema=1,source_path=str(a.source),
        source_sha256=sha(source),driver_sha256=sha(target.read_bytes()),observer_sha256=sha(observer),
        adapter_sha256=sha(Path(__file__).read_bytes()),compiled=False,scorer_executed=False,
        scope='both outputs extracted from existing timed result vector; observer bookkeeping remains timed; no new metric call'),indent=2))
if __name__=='__main__':main()
