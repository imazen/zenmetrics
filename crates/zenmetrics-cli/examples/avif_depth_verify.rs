#![forbid(unsafe_code)]
//! **G3 — 10-bit decode-verify.** Reads the bit depth of an AVIF back out of the
//! **stored blob**, by three mutually independent routes, and fails loud when they
//! disagree or when the depth is not the one that was requested.
//!
//! Registration: `benchmarks/avif_hdr_arm_plan_2026-09-02.md` §5 gate **G3**, which
//! exists for hazard **H-BD-3** (§2.2): `AvifEncoder::with_bit_depth` **silently
//! coerces** an unknown depth to 8, so a typo'd depth yields a valid 8-bit encode
//! *labelled* 10-bit, with no error anywhere. **A request for depth 10 is not
//! evidence of a 10-bit stream.** The only admissible evidence is a read from the
//! emitted bitstream — which is what this tool does and all it does.
//!
//! # The three reads
//!
//! | # | route | source | owner |
//! |---|---|---|---|
//! | R1 | **`av1C` container box** | `AvifParser::av1_config()` | `zenavif-parse` |
//! | R2 | **AV1 sequence header (OBU)** | `AvifParser::primary_metadata()` | `zenavif-parse` |
//! | R3 | **decoder `ImageInfo`** | `ManagedAvifDecoder::decode_full()` | `zenavif` (rav1d-safe) |
//!
//! R1 reads a container property; R2 re-parses the compressed payload's own sequence
//! header; R3 is the end-to-end decode the fleet actually runs. R1 and R2 are the two
//! the gate names ("`av1C` + sequence header"); R3 is carried because a depth that
//! survives the container and the header but not the decoder is still a broken cell,
//! and because it is the exact `ImageInfo` the `decode.rs` PQ/HLG tripwire reads.
//!
//! # `high_bitdepth` / `twelve_bit`, and where they are actually coded
//!
//! The gate is phrased over the two AV1 flags. They are **explicit bits in `av1C`**
//! (`zenavif-parse` `read_av1c`: `byte2 >> 6` and `byte2 >> 5`), so R1's pair is a
//! direct read. In the **sequence header** they are not symmetric: `twelve_bit` is
//! coded **only when `seq_profile == 2 && high_bitdepth`** (AV1 spec 5.5.2; see
//! `zenavif-parse` `obu.rs`), so at profile 0/1 a 10-bit stream codes
//! `high_bitdepth = 1` and no `twelve_bit` bit exists at all. This tool therefore
//! reports R2's pair as *derived from the decoded depth* and prints `seq_profile`
//! beside it, so a reader can tell a coded zero from an absent one. The mapping is
//! total and lossless in both directions:
//!
//! ```text
//!  8 <-> high_bitdepth=0, twelve_bit=0
//! 10 <-> high_bitdepth=1, twelve_bit=0
//! 12 <-> high_bitdepth=1, twelve_bit=1
//! ```
//!
//! # Grid (tiled) images
//!
//! A grid item carries no AV1 payload of its own, so R2 falls back to **tile 0**'s
//! sequence header and the `grid_tiles` column reports the tile count. R1 and R3 are
//! unaffected (`av1C` is a property of the primary item; `decode_full` serves the grid
//! and non-grid shapes from one entry).
//!
//! # When R1 is unavailable — MEASURED 2026-09-02, two unrelated causes
//!
//! Verified against 207 conformance vectors (`zenavif/tests/vectors/{link-u,libavif}`)
//! with `--expect-from-name`: **zero depth mismatches and zero disagreements**; every
//! FAIL was an `av1C` that could not be read, never a wrong depth. The two causes are
//! distinguished by the sibling-property list this tool prints beside the failure:
//!
//! 1. **`… resolved: ispe,colr` — a grid, and spec-correct.** A grid *derivation* item
//!    carries no `av1C` of its own; its tiles do. R2 (tile 0) and R3 still read the
//!    depth, so the depth is known — only R1 is missing. 7 of 57 libavif vectors.
//! 2. **`… resolved: none` — a `zenavif-parse` defect, and the primary item's WHOLE
//!    property set is gone.** `read_iprp` (`zenavif-parse` `src/lib.rs`) assigns
//!    `associations = read_ipma(&mut b)?` inside its box loop, so a file carrying
//!    **two `ipma` boxes** keeps only the last: the first box's associations — the
//!    primary item's — are silently discarded. Confirmed by an independent ISOBMFF
//!    walk of `plum-blossom-large.profile0.10bpc.yuv420.alpha-full.avif`: `pitm` = 1
//!    and item 1 *is* associated with `av1C` (ipco index 4), yet `av1_config()`,
//!    `spatial_extents()`, `color_info()` and `pixel_aspect_ratio()` all return `None`.
//!    In this corpus it separates perfectly on alpha (64 of 64 FAILs carry `alpha` in
//!    the name; 0 of 86 PASSes do), because the alpha writer emits the second `ipma`.
//!    **Inert for this arm** — its blobs are opaque, single-`ipma`, non-grid, and read
//!    all three routes — but a fix belongs in `zenavif-parse`, not here.
//!
//! Either way the tool **FAILs rather than softening**: a blob whose container depth
//! cannot be read has not been verified, and a gate that reports PASS on unread
//! evidence is the thing this gate exists to prevent.
//!
//! # Usage
//!
//! ```text
//! avif_depth_verify [OPTIONS] <PATH>...
//!
//!   <PATH>...              .avif files, and/or directories (walked recursively for *.avif)
//!
//!   --expect-depth <N>     every input must read N (8|10|12); any other depth FAILs
//!   --expect-from-name     derive the expected depth per file from a `<N>bpc` token in
//!                          its name (e.g. `fox.profile0.10bpc.yuv420.avif`); a file with
//!                          no such token is reported `-` and is not failed for it
//!   --control <PATH>       assert every input is NOT byte-identical to this blob
//!                          (G3's second half: a bd10 cell must differ from its 8-bit
//!                          control; Stage-A measured 0/288 identical)
//!   --tsv <PATH>           also write the TSV to a file
//!   --quiet                suppress the human-readable summary lines (TSV only)
//! ```
//!
//! Exit codes: **0** all PASS · **1** at least one FAIL · **2** no input files matched
//! (an empty run must not pass silently — the convention `scripts/hdr_corpus_precheck.py`
//! set for G0.2) · **3** bad usage.
//!
//! Every blob's `sha256` is emitted, so a gate script can settle G3's byte-identity half
//! for *arbitrary* (bd10, control) pairings straight from the TSV without re-invoking.

use std::fmt::Write as _;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// depth <-> (high_bitdepth, twelve_bit)
// ---------------------------------------------------------------------------

/// The AV1 flag pair for a bit depth, or `None` if the depth is not one AV1 codes.
///
/// Total and lossless: AV1 admits exactly 8, 10 and 12, so this inversion loses
/// nothing that `read_av1c`'s forward direction encoded.
fn flags_for_depth(depth: u8) -> Option<(u8, u8)> {
    match depth {
        8 => Some((0, 0)),
        10 => Some((1, 0)),
        12 => Some((1, 1)),
        _ => None,
    }
}

/// Expected depth parsed from a `<N>bpc` token in a filename, if there is one.
///
/// The Link-U / libavif conformance vectors name their depth
/// (`fox.profile0.10bpc.yuv420.avif`, `colors-animated-12bpc-keyframes-0-2-3.avif`),
/// which makes them a self-describing verification corpus for this tool. A name with
/// no such token yields `None` and is never failed on that basis.
fn depth_from_name(name: &str) -> Option<u8> {
    let bytes = name.as_bytes();
    let mut from = 0usize;
    while let Some(off) = name[from..].find("bpc") {
        let at = from + off;
        let mut start = at;
        while start > 0 && bytes[start - 1].is_ascii_digit() {
            start -= 1;
        }
        if start < at {
            if let Ok(v) = name[start..at].parse::<u8>() {
                return Some(v);
            }
        }
        from = at + 3;
    }
    None
}

// ---------------------------------------------------------------------------
// per-file record
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Row {
    path: String,
    sha256: String,
    bytes: u64,
    /// R1 — `av1C` container box.
    av1c_depth: Option<u8>,
    /// R2 — AV1 sequence header (primary item, or tile 0 for a grid).
    seq_depth: Option<u8>,
    seq_profile: Option<u8>,
    /// R3 — decoder `ImageInfo`.
    dec_depth: Option<u8>,
    dec_transfer: Option<u8>,
    chroma: String,
    grid_tiles: usize,
    expect: Option<u8>,
    /// Every read that succeeded returned the same depth.
    agree: bool,
    pass: bool,
    detail: String,
}

fn opt_u8(v: Option<u8>) -> String {
    v.map(|x| x.to_string()).unwrap_or_else(|| "-".into())
}

/// `high_bitdepth`/`twelve_bit` columns for a depth read, or `-`/`-` when the read
/// failed or the depth is not an AV1 depth.
fn flag_cols(depth: Option<u8>) -> (String, String) {
    match depth.and_then(flags_for_depth) {
        Some((h, t)) => (h.to_string(), t.to_string()),
        None => ("-".into(), "-".into()),
    }
}

impl Row {
    fn tsv(&self) -> String {
        let (a_h, a_t) = flag_cols(self.av1c_depth);
        let (s_h, s_t) = flag_cols(self.seq_depth);
        let mut s = String::new();
        let _ = write!(
            s,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            self.path,
            self.sha256,
            self.bytes,
            opt_u8(self.av1c_depth),
            a_h,
            a_t,
            opt_u8(self.seq_depth),
            s_h,
            s_t,
            opt_u8(self.dec_depth),
            opt_u8(self.dec_transfer),
            opt_u8(self.seq_profile),
            self.chroma,
            self.grid_tiles,
            opt_u8(self.expect),
            if self.agree { "yes" } else { "NO" },
            if self.pass { "PASS" } else { "FAIL" },
        );
        // `detail` is last so a reader can eyeball it without counting columns.
        let _ = write!(
            s,
            "\t{}",
            if self.detail.is_empty() {
                "-"
            } else {
                &self.detail
            }
        );
        s
    }
}

const TSV_HEADER: &str = "file\tsha256\tbytes\tav1c_depth\tav1c_high_bitdepth\tav1c_twelve_bit\t\
seqhdr_depth\tseqhdr_high_bitdepth\tseqhdr_twelve_bit\tdecoder_depth\tdecoder_transfer\t\
seq_profile\tchroma\tgrid_tiles\texpect\tagree\tstatus\tdetail";

// ---------------------------------------------------------------------------
// the three reads
// ---------------------------------------------------------------------------

fn chroma_label(cfg: &zenavif_parse::AV1Config) -> String {
    if cfg.monochrome {
        return "400".into();
    }
    match (cfg.chroma_subsampling_x, cfg.chroma_subsampling_y) {
        (1, 1) => "420".into(),
        (1, 0) => "422".into(),
        (0, 0) => "444".into(),
        (x, y) => format!("x{x}y{y}"),
    }
}

fn verify_one(path: &Path, data: &[u8], expect: Option<u8>, control: Option<&str>) -> Row {
    let mut row = Row {
        path: path.display().to_string(),
        sha256: format!("{:x}", Sha256::digest(data)),
        bytes: data.len() as u64,
        chroma: "-".into(),
        expect,
        ..Row::default()
    };
    let mut problems: Vec<String> = Vec::new();

    // ---- R1 + R2: the container parse (both gate-named reads come from it) ----
    match zenavif_parse::AvifParser::from_bytes(data) {
        Ok(parser) => {
            match parser.av1_config() {
                Some(cfg) => {
                    row.av1c_depth = Some(cfg.bit_depth);
                    row.chroma = chroma_label(cfg);
                }
                // Not a parse failure: a container that declares no av1C cannot be
                // depth-verified at all, which for a gate is a FAIL, not a shrug.
                //
                // Report whether the OTHER primary-item properties resolved, because
                // that distinguishes the two very different causes: a grid item
                // legitimately carries no av1C of its own (its tiles do), whereas a
                // plain item whose siblings resolved but whose av1C did not is a
                // container-parse problem worth chasing.
                None => {
                    let mut sibs: Vec<&str> = Vec::new();
                    if parser.spatial_extents().is_some() {
                        sibs.push("ispe");
                    }
                    if parser.color_info().is_some() {
                        sibs.push("colr");
                    }
                    if parser.pixel_aspect_ratio().is_some() {
                        sibs.push("pasp");
                    }
                    problems.push(format!(
                        "no av1C property (other primary-item properties resolved: {})",
                        if sibs.is_empty() {
                            "none".to_string()
                        } else {
                            sibs.join(",")
                        }
                    ));
                }
            }
            row.grid_tiles = parser.grid_tile_count();

            // R2 — the payload's own sequence header. A grid item has no AV1 payload
            // of its own, so fall back to tile 0.
            let seq = match parser.primary_metadata() {
                Ok(m) => Ok(m),
                Err(e) if row.grid_tiles > 0 => parser
                    .tile_data(0)
                    .and_then(|d| zenavif_parse::AV1Metadata::parse_av1_bitstream(&d))
                    .map_err(|e2| format!("primary({e}) and tile0({e2})")),
                Err(e) => Err(e.to_string()),
            };
            match seq {
                Ok(m) => {
                    row.seq_depth = Some(m.bit_depth);
                    row.seq_profile = Some(m.seq_profile);
                }
                Err(e) => problems.push(format!("seqhdr parse failed: {e}")),
            }
        }
        Err(e) => problems.push(format!("container parse failed: {e}")),
    }

    // ---- R3: the decode the fleet actually runs ----
    let cfg = zenavif::DecoderConfig::new().threads(1);
    match zenavif::ManagedAvifDecoder::new(data, &cfg)
        .and_then(|mut d| d.decode_full(&enough::Unstoppable).map(|(_, info)| info))
    {
        Ok(info) => {
            row.dec_depth = Some(info.bit_depth);
            row.dec_transfer = Some(info.transfer_characteristics.0);
        }
        Err(e) => problems.push(format!("decode failed: {e}")),
    }

    // ---- adjudication ----
    let reads: Vec<(&str, u8)> = [
        ("av1C", row.av1c_depth),
        ("seqhdr", row.seq_depth),
        ("decoder", row.dec_depth),
    ]
    .into_iter()
    .filter_map(|(n, d)| d.map(|d| (n, d)))
    .collect();

    row.agree = reads.windows(2).all(|w| w[0].1 == w[1].1) && !reads.is_empty();
    if !row.agree && reads.len() > 1 {
        problems.push(format!(
            "reads DISAGREE: {}",
            reads
                .iter()
                .map(|(n, d)| format!("{n}={d}"))
                .collect::<Vec<_>>()
                .join(" ")
        ));
    }

    // A depth outside {8,10,12} is not an AV1 depth; refuse rather than report it.
    for (name, d) in &reads {
        if flags_for_depth(*d).is_none() {
            problems.push(format!("{name} depth {d} is not an AV1 depth (8|10|12)"));
        }
    }

    // The gate proper: was the depth that came back the one that was asked for?
    if let Some(want) = expect {
        if reads.is_empty() {
            problems.push(format!("expected depth {want} but nothing could be read"));
        } else {
            for (name, d) in &reads {
                if *d != want {
                    problems.push(format!("{name} depth {d} != expected {want}"));
                }
            }
        }
    }

    // G3's second half: a bd10 cell must not be byte-identical to its 8-bit control.
    if let Some(c) = control {
        if row.sha256 == c {
            problems.push("BYTE-IDENTICAL to the control blob".into());
        }
    }

    row.pass = problems.is_empty();
    row.detail = problems.join("; ");
    row
}

// ---------------------------------------------------------------------------
// input collection
// ---------------------------------------------------------------------------

fn collect(path: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let md = fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if !md.is_dir() {
        // An explicitly named file is taken whatever its extension — a stored blob
        // in a fleet `blobs/` prefix is content-addressed and often has none.
        out.push(path.to_path_buf());
        return Ok(());
    }
    let mut entries: Vec<PathBuf> = fs::read_dir(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .map(|e| e.map(|e| e.path()).map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    entries.sort();
    for p in entries {
        if p.is_dir() {
            collect(&p, out)?;
        } else if p
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.eq_ignore_ascii_case("avif"))
        {
            out.push(p);
        }
    }
    Ok(())
}

const USAGE: &str = "\
avif_depth_verify — G3 10-bit decode-verify (avif_hdr_arm_plan_2026-09-02.md §5)

USAGE:
  avif_depth_verify [OPTIONS] <PATH>...

OPTIONS:
  --expect-depth <N>   every input must read N (8|10|12)
  --expect-from-name   derive the expected depth per file from a `<N>bpc` name token
  --control <PATH>     assert every input is NOT byte-identical to this blob
  --tsv <PATH>         also write the TSV here
  --quiet              TSV only, no summary lines
  -h, --help           this text

EXIT: 0 all PASS · 1 a FAIL · 2 no input files · 3 bad usage";

fn main() -> std::process::ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut expect_depth: Option<u8> = None;
    let mut expect_from_name = false;
    let mut control_path: Option<String> = None;
    let mut tsv_out: Option<String> = None;
    let mut quiet = false;
    let mut inputs: Vec<String> = Vec::new();

    let mut i = 0;
    while i < argv.len() {
        let a = argv[i].as_str();
        let mut next = |what: &str| -> Option<String> {
            i += 1;
            match argv.get(i) {
                Some(v) => Some(v.clone()),
                None => {
                    eprintln!("avif_depth_verify: {what} needs a value");
                    None
                }
            }
        };
        match a {
            "-h" | "--help" => {
                println!("{USAGE}");
                return std::process::ExitCode::SUCCESS;
            }
            "--expect-depth" => match next("--expect-depth").map(|v| v.parse::<u8>()) {
                Some(Ok(v)) => expect_depth = Some(v),
                _ => {
                    eprintln!("avif_depth_verify: --expect-depth needs an integer");
                    return std::process::ExitCode::from(3);
                }
            },
            "--expect-from-name" => expect_from_name = true,
            "--control" => match next("--control") {
                Some(v) => control_path = Some(v),
                None => return std::process::ExitCode::from(3),
            },
            "--tsv" => match next("--tsv") {
                Some(v) => tsv_out = Some(v),
                None => return std::process::ExitCode::from(3),
            },
            "--quiet" => quiet = true,
            _ if a.starts_with("--") => {
                eprintln!("avif_depth_verify: unknown option {a}\n\n{USAGE}");
                return std::process::ExitCode::from(3);
            }
            _ => inputs.push(a.to_string()),
        }
        i += 1;
    }

    if inputs.is_empty() {
        eprintln!("{USAGE}");
        return std::process::ExitCode::from(3);
    }
    if expect_depth.is_some() && expect_from_name {
        eprintln!("avif_depth_verify: --expect-depth and --expect-from-name are exclusive");
        return std::process::ExitCode::from(3);
    }

    // The control blob is hashed once, not per input.
    let control_sha = match control_path.as_ref() {
        Some(p) => match fs::read(p) {
            Ok(d) => Some(format!("{:x}", Sha256::digest(&d))),
            Err(e) => {
                eprintln!("avif_depth_verify: --control {p}: {e}");
                return std::process::ExitCode::from(3);
            }
        },
        None => None,
    };

    let mut files: Vec<PathBuf> = Vec::new();
    for inp in &inputs {
        if let Err(e) = collect(Path::new(inp), &mut files) {
            eprintln!("avif_depth_verify: {e}");
            return std::process::ExitCode::from(3);
        }
    }
    if files.is_empty() {
        // Never pass silently on an empty run: an empty glob is the classic way a
        // gate reports success without having verified anything.
        eprintln!("avif_depth_verify: no input files matched — refusing to report PASS");
        return std::process::ExitCode::from(2);
    }

    let mut lines: Vec<String> = vec![TSV_HEADER.to_string()];
    let (mut n_pass, mut n_fail) = (0usize, 0usize);
    let mut failures: Vec<String> = Vec::new();

    for f in &files {
        let expect = if expect_from_name {
            f.file_name()
                .and_then(|s| s.to_str())
                .and_then(depth_from_name)
        } else {
            expect_depth
        };
        let row = match fs::read(f) {
            Ok(data) => verify_one(f, &data, expect, control_sha.as_deref()),
            Err(e) => Row {
                path: f.display().to_string(),
                sha256: "-".into(),
                chroma: "-".into(),
                expect,
                pass: false,
                detail: format!("read failed: {e}"),
                ..Row::default()
            },
        };
        if row.pass {
            n_pass += 1;
        } else {
            n_fail += 1;
            failures.push(format!("  FAIL {}  {}", row.path, row.detail));
        }
        lines.push(row.tsv());
    }

    let tsv = lines.join("\n");
    println!("{tsv}");
    if let Some(p) = &tsv_out {
        match fs::File::create(p).and_then(|mut fh| writeln!(fh, "{tsv}")) {
            Ok(()) => {
                if !quiet {
                    eprintln!("avif_depth_verify: TSV written to {p}");
                }
            }
            Err(e) => {
                eprintln!("avif_depth_verify: writing {p}: {e}");
                return std::process::ExitCode::from(3);
            }
        }
    }

    if !quiet {
        eprintln!();
        for l in &failures {
            eprintln!("{l}");
        }
        eprintln!(
            "avif_depth_verify: {} file(s) — {n_pass} PASS, {n_fail} FAIL{}",
            files.len(),
            match (expect_depth, expect_from_name) {
                (Some(d), _) => format!(" (--expect-depth {d})"),
                (_, true) => " (--expect-from-name)".to_string(),
                _ => String::new(),
            }
        );
    }

    if n_fail > 0 {
        std::process::ExitCode::from(1)
    } else {
        std::process::ExitCode::SUCCESS
    }
}

#[cfg(test)]
mod tests {
    use super::{depth_from_name, flags_for_depth};

    /// The AV1 depth <-> flag-pair inversion, pinned in both directions. `read_av1c`
    /// derives the depth from the bits; this tool derives the bits from the depth, so
    /// a divergence here would silently mis-report the two values the gate is written
    /// over.
    #[test]
    fn depth_flag_mapping_is_the_av1c_mapping() {
        assert_eq!(flags_for_depth(8), Some((0, 0)));
        assert_eq!(flags_for_depth(10), Some((1, 0)));
        assert_eq!(flags_for_depth(12), Some((1, 1)));
        // Everything else is refused rather than guessed — including the 16 an
        // `ImageInfo` would carry if a future decoder widened, and the 0 a zeroed
        // struct would carry.
        for d in [0u8, 1, 7, 9, 11, 13, 16, 255] {
            assert_eq!(flags_for_depth(d), None, "depth {d} must not map to flags");
        }
    }

    #[test]
    fn depth_from_name_reads_the_conformance_vector_convention() {
        assert_eq!(depth_from_name("fox.profile0.10bpc.yuv420.avif"), Some(10));
        assert_eq!(
            depth_from_name("fox.profile2.12bpc.yuv444.monochrome.avif"),
            Some(12)
        );
        assert_eq!(depth_from_name("fox.profile0.8bpc.yuv420.avif"), Some(8));
        assert_eq!(
            depth_from_name("cosmos1650_yuv444_10bpc_p3pq.avif"),
            Some(10)
        );
        assert_eq!(
            depth_from_name("colors-animated-12bpc-keyframes-0-2-3.avif"),
            Some(12)
        );
        // No token -> no expectation, rather than a wrong one. `weld_sato_12B_8B_q0`
        // carries digits and a `B` but no `bpc`, and must not be read as a depth.
        assert_eq!(depth_from_name("weld_sato_12B_8B_q0.avif"), None);
        assert_eq!(depth_from_name("kimono.avif"), None);
        assert_eq!(depth_from_name("bpc.avif"), None);
    }
}
