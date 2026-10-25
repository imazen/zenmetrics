//! Compare historical Butteraugli implementations on identical saved RGBA8 pixels.
#![forbid(unsafe_code)]
use std::error::Error;
use std::io::{BufWriter, Write};
use std::path::Path;

macro_rules! score {
    ($module:ident, $a:expr, $b:expr, $w:expr, $h:expr, $params:expr) => {{
        let rgb = |p: &[u8]| {
            p.chunks_exact(4)
                .map(|p| $module::RGB8::new(p[0], p[1], p[2]))
                .collect::<Vec<_>>()
        };
        let (a, b) = (rgb($a), rgb($b));
        let result = $module::butteraugli(
            $module::ImgRef::new(&a, $w, $h),
            $module::ImgRef::new(&b, $w, $h),
            &$params.with_compute_diffmap(true),
        )?;
        result
            .diffmap
            .ok_or("missing map")?
            .pixels()
            .collect::<Vec<f32>>()
    }};
}

fn write_map(path: &Path, values: &[f32]) -> Result<(), Box<dyn Error>> {
    let mut out = BufWriter::new(std::fs::File::create_new(path)?);
    for v in values {
        out.write_all(&v.to_le_bytes())?;
    }
    out.flush()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: margarine-reference-audit PAIRS_DIR NEW_OUTPUT_DIR".into());
    }
    let input = Path::new(&args[0]);
    let output = Path::new(&args[1]);
    std::fs::create_dir(output)?;
    let index = std::fs::read_to_string(input.join("pairs.tsv"))?;
    std::fs::write(output.join("pairs.tsv"), &index)?;
    for line in index.lines().skip(1) {
        let fields: Vec<_> = line.split('\t').collect();
        if fields[2].contains("validation") {
            continue;
        }
        let (id, w, h) = (
            fields[0],
            fields[4].parse::<usize>()?,
            fields[5].parse::<usize>()?,
        );
        let a = std::fs::read(input.join(format!("{id}.ref.rgba")))?;
        let b = std::fs::read(input.join(format!("{id}.dist.rgba")))?;
        if a.len() != w * h * 4 || b.len() != a.len() {
            return Err(format!("invalid RGBA dimensions for {id}").into());
        }
        for mode in ["loop", "teacher", "fast", "exact"] {
            let map = match mode {
                "loop" => score!(
                    loop_butter,
                    &a,
                    &b,
                    w,
                    h,
                    loop_butter::ButteraugliParams::default()
                ),
                "teacher" => score!(
                    teacher_butter,
                    &a,
                    &b,
                    w,
                    h,
                    teacher_butter::ButteraugliParams::default()
                ),
                "fast" => score!(
                    exact_butter,
                    &a,
                    &b,
                    w,
                    h,
                    exact_butter::ButteraugliParams::default()
                ),
                _ => score!(
                    exact_butter,
                    &a,
                    &b,
                    w,
                    h,
                    exact_butter::ButteraugliParams::default()
                        .with_numerics(exact_butter::Numerics::LibjxlExact)
                ),
            };
            write_map(&output.join(format!("{id}.{mode}.f32")), &map)?;
        }
        println!("{id}\t{}\t{}", fields[1], fields[2]);
        std::io::stdout().flush()?;
    }
    Ok(())
}
