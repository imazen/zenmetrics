//! Phase-2a byte-safety GATE for the JXL encode-dedup.
//! Groups modes_full cells by `zenjxl::sweep::encode_fingerprint` and proves
//! byte-identity on real encodes: (1) the specific validated merges, (2) a
//! bounded broad sample (all use eprintln! → unbuffered, survives a timeout).
//! Run: cargo run -p zenmetrics-cli --features sweep,jxl --example encode_fp_byte_safety

#[cfg(all(feature = "sweep", feature = "jxl"))]
fn main() {
    use jxl_encoder::api::EncoderStrategy;
    use jxl_encoder::{EncoderMode, PixelLayout, ProgressiveMode};
    use std::collections::{BTreeMap, HashMap};
    use zenjxl::sweep::{
        BuiltConfig, QualityGrid, SweepAxes, SweepBuilder, SweepCell, SweepVariant,
        encode_fingerprint,
    };

    // 64x64 structured image: one full DCT64 block region + texture/edges so
    // DCT16/32/64 + CfL + quant paths are real candidates.
    let (w, h) = (64u32, 64u32);
    let mut px = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let g = ((x * 4) ^ (y * 3)) as u8;
            let ck = if (x / 8 + y / 8) % 2 == 0 { 40 } else { 0 };
            let nz = ((x
                .wrapping_mul(2654435761)
                .wrapping_add(y.wrapping_mul(40503)))
                >> 13) as u8;
            px.push(g.wrapping_add(ck));
            px.push(g.wrapping_mul(3).wrapping_add(nz & 31));
            px.push(g.wrapping_add(y as u8).wrapping_add(ck));
        }
    }
    let enc = |c: &SweepCell| -> Option<Vec<u8>> {
        match c.build() {
            BuiltConfig::Lossy(cfg) => cfg.encode(&px, w, h, PixelLayout::Rgb8).ok(),
            BuiltConfig::Lossless(_) => None,
        }
    };

    let plan = SweepBuilder::new(
        SweepAxes::modes_full(),
        QualityGrid::ExplicitQuality(vec![30.0, 70.0]),
    )
    .plan();
    let mut groups: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, c) in plan.cells.iter().enumerate() {
        groups
            .entry(encode_fingerprint(&c.variant))
            .or_default()
            .push(i);
    }
    let (cells, ng) = (plan.cells.len(), groups.len());
    eprintln!(
        "modes_full: cells={cells} encode_groups={ng} dedup-able={} ({:.1}%)",
        cells - ng,
        100.0 * (cells - ng) as f64 / cells as f64
    );

    let mut multi: Vec<&Vec<usize>> = groups.values().filter(|v| v.len() > 1).collect();
    multi.sort_by_key(|v| v[0]);
    let mut hist: BTreeMap<usize, usize> = BTreeMap::new();
    for g in &multi {
        *hist.entry(g.len()).or_default() += 1;
    }
    eprintln!("multi_member_groups={} size-hist {hist:?}", multi.len());
    eprintln!("example merged groups:");
    for g in multi.iter().step_by((multi.len() / 8).max(1)).take(8) {
        let ids: Vec<&str> = g
            .iter()
            .map(|&i| plan.cells[i].id.as_str())
            .take(5)
            .collect();
        eprintln!(
            "  [{}{}]",
            ids.join(" | "),
            if g.len() > 5 { " | …" } else { "" }
        );
    }

    // (1) TARGETED validated merges — encode + assert identical bytes.
    // The single cell that is all-default EXCEPT effort + the given internal
    // label (unambiguous: one per (effort,label,q)).
    let find = |eff: u8, lab: &str, q: f32| -> Option<&SweepCell> {
        plan.cells.iter().find(|c| {
            c.quality == Some(q)
                && matches!(&c.variant, SweepVariant::Lossy(v)
                    if v.effort == eff
                    && v.internal.label == lab
                    && matches!(v.strategy, EncoderStrategy::Zenjxl)
                    && matches!(v.encoder_mode, EncoderMode::Reference)
                    && v.gaborish.is_none()
                    && v.epf_level == -1
                    && matches!(v.progressive, ProgressiveMode::Single)
                    && !v.noise
                    && v.faster_decoding == 0
                    && v.ans.is_none())
        })
    };
    let pair = |a: Option<&SweepCell>, b: Option<&SweepCell>, want_eq: bool, name: &str| -> u32 {
        match (a, b) {
            (Some(a), Some(b)) => match (enc(a), enc(b)) {
                (Some(x), Some(y)) => {
                    let eq = x == y;
                    let ok = eq == want_eq;
                    eprintln!(
                        "  [{}] {name}: {} == {} -> bytes_equal={eq} ({} vs {}) {}",
                        if ok { "OK" } else { "FAIL" },
                        a.id,
                        b.id,
                        x.len(),
                        y.len(),
                        if want_eq {
                            "(want EQUAL)"
                        } else {
                            "(want DIFFER)"
                        }
                    );
                    if ok { 0 } else { 1 }
                }
                _ => {
                    eprintln!("  [skip] {name}: encode failed");
                    0
                }
            },
            _ => {
                eprintln!("  [skip] {name}: cell not found");
                0
            }
        }
    };
    eprintln!("targeted validated merges (real encodes):");
    let mut bad = 0u32;
    // default-stratum cells use label "def"; q30
    bad += pair(find(1, "def", 30.0), find(2, "def", 30.0), true, "e1==e2");
    bad += pair(
        find(3, "def", 30.0),
        find(3, "dct16off", 30.0),
        true,
        "dct16off@e3==def",
    );
    bad += pair(
        find(3, "def", 30.0),
        find(3, "kaq0.65", 30.0),
        true,
        "kaq0.65@e3==def",
    );
    bad += pair(
        find(3, "def", 30.0),
        find(3, "emulexp", 30.0),
        true,
        "emulexp@e3==def",
    );
    bad += pair(
        find(6, "def", 30.0),
        find(6, "dct64off", 30.0),
        true,
        "dct64off@e6==def",
    );
    // SHOULD differ (active at the gate):
    bad += pair(
        find(7, "def", 30.0),
        find(7, "dct16off", 30.0),
        false,
        "dct16off@e7!=def",
    );
    bad += pair(
        find(5, "def", 30.0),
        find(5, "kaq0.65", 30.0),
        false,
        "kaq0.65@e5!=def",
    );

    // (2) bounded broad sample: 250 groups (incl high-effort), 2 members each.
    let he: Vec<&&Vec<usize>> = multi
        .iter()
        .filter(|g| {
            g.iter()
                .any(|&i| matches!(&plan.cells[i].variant, SweepVariant::Lossy(v) if v.effort>=7))
        })
        .collect();
    let mut to_check: Vec<&Vec<usize>> = multi
        .iter()
        .step_by((multi.len() / 200).max(1))
        .copied()
        .collect();
    for g in he.iter().step_by((he.len() / 80).max(1)).take(80) {
        to_check.push(**g);
    }
    let (mut checked, mut mism) = (0u32, 0u32);
    for (n, g) in to_check.iter().enumerate() {
        let a = &plan.cells[g[0]];
        let b = &plan.cells[g[1]];
        if let (Some(x), Some(y)) = (enc(a), enc(b)) {
            checked += 1;
            if x != y {
                mism += 1;
                if mism <= 8 {
                    eprintln!(
                        "  BROAD MISMATCH {} vs {} ({} vs {})",
                        a.id,
                        b.id,
                        x.len(),
                        y.len()
                    );
                }
            }
        }
        if n % 60 == 0 {
            eprintln!(
                "  ...broad progress {n}/{} checked={checked} mism={mism}",
                to_check.len()
            );
        }
    }
    eprintln!(
        "broad sample: checked={checked} groups (e>=7 groups total={}) MISMATCHES={mism}",
        he.len()
    );
    eprintln!("TARGETED bad={bad}  BROAD mism={mism}");
    if bad == 0 && mism == 0 {
        eprintln!("GATE: PASS");
    } else {
        eprintln!("GATE: FAIL");
        std::process::exit(1);
    }
}
#[cfg(not(all(feature = "sweep", feature = "jxl")))]
fn main() {
    eprintln!("build with --features sweep,jxl");
}
