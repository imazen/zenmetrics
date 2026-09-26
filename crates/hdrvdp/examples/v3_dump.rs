//! Debug dumper: runs the v3 pathway on a golden case and writes our
//! intermediates as f64le for numpy comparison.
//! Usage: cargo run -p hdrvdp --example v3_dump -- <case_name>

use hdrvdp::v3::{
    Emission, InputEncoding, Options, Params, Surround, Task, ViewingConditions, hdrvdp3,
};
use std::fs;
use std::path::{Path, PathBuf};

fn dir(case: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/v3")
        .join(case)
}

fn read_f64(path: &Path) -> Vec<f64> {
    fs::read(path)
        .unwrap()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect()
}

fn write_f64(path: &Path, v: &[f64]) {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    fs::write(path, b).unwrap();
}

fn main() {
    let case = std::env::args().nth(1).expect("case name");
    let d = dir(&case);
    // manifest row lookup
    let manifest = fs::read_to_string(d.parent().unwrap().join("manifest.tsv")).unwrap();
    let row = manifest
        .lines()
        .find(|l| l.starts_with(&(case.clone() + "\t")))
        .unwrap();
    let f: Vec<&str> = row.split('\t').collect();
    let (w, h, c): (usize, usize, usize) = (
        f[1].parse().unwrap(),
        f[2].parse().unwrap(),
        f[3].parse().unwrap(),
    );
    let ppd: f64 = f[5].parse().unwrap();
    let enc = match f[4] {
        "luminance" => InputEncoding::Luminance,
        "luma-display" => InputEncoding::LumaDisplay,
        "sRGB-display" => InputEncoding::SrgbDisplay,
        "rgb-bt.709" => InputEncoding::RgbBt709,
        "rgb-bt.2020" => InputEncoding::RgbBt2020,
        "rgb-native" => InputEncoding::RgbNative,
        "xyz" => InputEncoding::Xyz,
        s => panic!("enc {s}"),
    };
    let task = match f[6] {
        "quality" => Task::Quality,
        "side-by-side" | "sbs" => Task::SideBySide,
        "flicker" => Task::Flicker,
        s => panic!("task {s}"),
    };
    let age: u8 = f[7].parse().unwrap();

    let test = read_f64(&d.join("test.f64le"));
    let refr = read_f64(&d.join("ref.f64le"));
    let par = Params::new(
        task,
        ViewingConditions::new(ppd, Surround::None, age),
        enc,
        Emission::default_for(enc),
        Options::reference(task),
    )
    .unwrap();
    let res = hdrvdp3(&test, &refr, w, h, &par).unwrap();
    let _ = c;
    write_f64(&d.join("our_pmap.f64le"), &res.p_map);
    write_f64(&d.join("our_smap.f64le"), &res.s_map);
    eprintln!(
        "Q={} Q_JOD={} Pmean={}",
        res.q,
        res.q_jod,
        res.p_map.iter().sum::<f64>() / res.p_map.len() as f64
    );
}
