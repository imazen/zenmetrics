//! Decode with the exact image::open(...).to_rgb8() ingress used by vdpprobe.
use std::{fs, io::Write, path::Path};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let out = Path::new(&a[2]);
    assert!(!out.exists());
    fs::create_dir(out).unwrap();
    let mut manifest = fs::File::create_new(out.join("manifest.tsv")).unwrap();
    writeln!(manifest, "name\twidth\theight\tbytes").unwrap();
    for path in fs::read_to_string(&a[1]).unwrap().lines() {
        let image = image::open(path).expect(path).to_rgb8();
        let name = Path::new(path).file_name().unwrap().to_str().unwrap();
        writeln!(manifest, "{name}\t{}\t{}\t{}", image.width(), image.height(), image.as_raw().len()).unwrap();
        fs::write(out.join(format!("{name}.rgb")), image.into_raw()).unwrap();
    }
}
