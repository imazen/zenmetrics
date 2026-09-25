//! Score the frozen TRAIN colour population after canonical RGB8 decoding.
//! Input: header then pair_key,width,height,ref_rgb,dist_rgb (tab separated).
//! Raw file hashes and dimensions are checked by prepare_colour_rgb.py.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    let mut args = std::env::args().skip(1);
    let input = std::fs::read_to_string(args.next().ok_or("input TSV required")?)?;
    let output = args.next().ok_or("output TSV required")?;
    let mut out = std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)?,
    );
    writeln!(out, "pair_key\tgmsd\tmdsi\tms_gmsdc")?;
    let mut seen = std::collections::HashSet::new();
    for line in input.lines().skip(1) {
        let f: Vec<_> = line.split('\t').collect();
        assert_eq!(f.len(), 5);
        assert!(seen.insert(f[0]));
        let (w, h): (usize, usize) = (f[1].parse()?, f[2].parse()?);
        let (r, d) = (std::fs::read(f[3])?, std::fs::read(f[4])?);
        assert_eq!(r.len(), w * h * 3);
        assert_eq!(d.len(), w * h * 3);
        let gray = gmsd::gmsd_rgb8(&r, &d, w, h, w * 3)?.gmsd;
        let mdsi = gmsd::mdsi_rgb8(&r, &d, w, h, w * 3)?;
        let colour = gmsd::ms_gmsdc_rgb8(&r, &d, w, h, w * 3)?;
        assert!([gray, mdsi, colour].iter().all(|v| v.is_finite()));
        writeln!(out, "{}\t{gray:.17e}\t{mdsi:.17e}\t{colour:.17e}", f[0])?;
    }
    out.flush()?;
    println!("scored_pairs {}", seen.len());
    Ok(())
}
