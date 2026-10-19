"""Benchmark-only emitted max-column/scope binding; no metric arithmetic edit."""
import argparse,hashlib,json
from pathlib import Path

def stage(root):
    path=root/'work/zenmetrics/benchmarks/heaptrack/drivers/cpu_profile/src/bin/iqa_compete.rs'
    original=path.read_text();s=original
    old='''run_metric_display(kind, r, d, GpuRuntime::Cpu, Some(&display))
            .map(|v| v[0].1)
            .map_err(|e| e.to_string())'''
    new='''run_metric_display(kind, r, d, GpuRuntime::Cpu, Some(&display))
            .map_err(|e| e.to_string())
            .and_then(|v| v.into_iter().find(|(column, _)| *column == "butteraugli_max")
                .map(|(_, value)| value).ok_or_else(|| "explicit butteraugli_max missing".to_string()))'''
    if s.count(old)!=1:raise ValueError('scalar dispatch anchor')
    s=s.replace(old,new)
    anchor='    let config: Value = serde_json::from_slice(&fs::read(&a[1])?)?;'
    guard='''
    if config["metric_name"] != "butteraugli" || config["width"] != 512
        || config["height"] != 512 || config["threads"] != 1
        || config["only_mode"] != "score" {
        return Err("new native cohort supports only butteraugli_max 512x512 1t score".into());
    }
'''
    if s.count(anchor)!=1:raise ValueError('config anchor')
    s=s.replace(anchor,anchor+guard)
    anchor='    row["headline_declared_column"] = json!(columns[0]);'
    if s.count(anchor)!=1:raise ValueError('emitted scope anchor')
    s=s.replace(anchor,anchor+'''
    row["native_scope_id"] = json!("rgb-input");
    row["native_scalar_column"] = json!("butteraugli_max");
    row["native_scope_qualified"] = json!(false);
''')
    path.write_text(s)
    (root/'rust-overlay.json').write_text(json.dumps(dict(path=str(path),before_sha256=hashlib.sha256(original.encode()).hexdigest(),after_sha256=hashlib.sha256(s.encode()).hexdigest(),scope='benchmark-only explicit scalar selection and emitted protocol',metric_arithmetic_changed=False),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);stage(p.parse_args().root)
