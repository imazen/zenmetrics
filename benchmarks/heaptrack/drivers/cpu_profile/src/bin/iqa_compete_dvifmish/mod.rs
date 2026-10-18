//! Benchmark-only routes. No CLI/public metric API or library arithmetic changes.
use super::maps::Call;
use dvifmish::{Image, Math};
use serde_json::{Value, json};
use zenmetrics_cli::decode::Rgb8Image;

pub(super) const NAMES: [&str; 6] = [
    "dvifmish-port-min-v0.2",
    "dvifmish-port-2d-v0.2",
    "dvifmish-port-lap-v0.2",
    "dvifmish-port-lap-v0.3",
    "dvifmish-talk-faithful-luma-float",
    "dvifmish-talk-faithful-luma-int",
];

pub(super) fn supported(name: &str) -> bool {
    NAMES.contains(&name)
}

fn preset(name: &str) -> Result<(&str, Option<Math>), String> {
    if !supported(name) {
        return Err(format!("unknown dvifmish benchmark route: {name}"));
    }
    Ok(match name {
        "dvifmish-talk-faithful-luma-float" => ("talk-faithful-luma", Some(Math::Float)),
        "dvifmish-talk-faithful-luma-int" => ("talk-faithful-luma", Some(Math::Int)),
        _ => (name.strip_prefix("dvifmish-").unwrap(), None),
    })
}

pub(super) fn inventory() -> Vec<Value> {
    use sha2::{Digest, Sha256};
    NAMES.iter().map(|&name| {
        let (preset, math) = preset(name).expect("known route");
        let constants = if math.is_some() { dvifmish::presets::json(preset) }
            else { dvifmish::port::presets::json(preset) }.expect("pinned embedded preset");
        json!({"metric_name":name,"gpu":false,"compiled":true,"columns":[name],
            "implementation":"dvifmish","source_commit":"74ec5eeae118ead266c7b66b56e56cb5f3bef372",
            "preset":preset,"preset_sha256":Sha256::digest(constants.as_bytes()).iter().map(|byte|format!("{byte:02x}")).collect::<String>(),
            "score_units":if math.is_some(){"raw distortion E"}else{"JND"},
            "native_map":"per-plane/per-level block contributions in raw distortion units; not a per-pixel JND map",
            "threads":"explicit library score_threads/score_with_map_threads budget; small levels may use fewer workers",
            "publish":false})
    }).collect()
}

pub(super) fn make_call(
    name: &str,
    r: &'static Rgb8Image,
    d: &'static Rgb8Image,
    map: bool,
    threads: usize,
) -> Result<Call, String> {
    if threads == 0 {
        return Err("dvifmish benchmark requires a positive explicit thread budget".into());
    }
    let (preset, math) = preset(name)?;
    let a =
        Image::new(r.width as usize, r.height as usize, &r.pixels).map_err(|e| e.to_string())?;
    let b =
        Image::new(d.width as usize, d.height as usize, &d.pixels).map_err(|e| e.to_string())?;
    // Called inside cold_call's clock; parsing/validation and first-call tables
    // are cold setup. Warm rounds reuse these immutable parsed constants.
    if let Some(math) = math {
        let params = dvifmish::presets::get(preset).ok_or("missing embedded talk preset")?;
        Ok(Box::new(move || {
            let score = if map {
                let (score, native) =
                    dvifmish::score_with_map_threads(&params, &a, &b, math, threads)
                        .map_err(|e| e.to_string())?;
                std::hint::black_box(native);
                score
            } else {
                dvifmish::score_threads(&params, &a, &b, math, threads)
                    .map_err(|e| e.to_string())?
            };
            Ok(score.distortion)
        }))
    } else {
        let params = dvifmish::port::presets::get(preset).ok_or("missing embedded port preset")?;
        Ok(Box::new(move || {
            let score = if map {
                let (score, native) =
                    dvifmish::port::score_with_map_threads(&params, &a, &b, threads)
                        .map_err(|e| e.to_string())?;
                std::hint::black_box(native);
                score
            } else {
                dvifmish::port::score_threads(&params, &a, &b, threads)
                    .map_err(|e| e.to_string())?
            };
            Ok(score.jnd)
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapters_keep_library_scalar_map_and_thread_bits() {
        for (w, h) in [(37, 29), (128, 97), (512, 512)] {
            let pixels: Vec<u8> = (0..w * h * 3)
                .map(|i| ((i * 17 + i / 13) % 251) as u8)
                .collect();
            let distorted: Vec<u8> = pixels
                .iter()
                .enumerate()
                .map(|(i, &v)| v.saturating_add(u8::from(i % 7 == 0) * 4))
                .collect();
            let r = Box::leak(Box::new(Rgb8Image {
                width: w as u32,
                height: h as u32,
                pixels,
            }));
            let d = Box::leak(Box::new(Rgb8Image {
                width: w as u32,
                height: h as u32,
                pixels: distorted,
            }));
            let a = Image::new(w, h, &r.pixels).unwrap();
            let b = Image::new(w, h, &d.pixels).unwrap();
            for name in NAMES {
                let (preset, math) = preset(name).unwrap();
                let expected = if let Some(math) = math {
                    dvifmish::score(&dvifmish::presets::get(preset).unwrap(), &a, &b, math)
                        .unwrap()
                        .distortion
                } else {
                    dvifmish::port::score(&dvifmish::port::presets::get(preset).unwrap(), &a, &b)
                        .unwrap()
                        .jnd
                };
                for threads in [1, 8] {
                    for map in [false, true] {
                        let got = make_call(name, r, d, map, threads).unwrap()().unwrap();
                        assert_eq!(
                            got.to_bits(),
                            expected.to_bits(),
                            "{name} {w}x{h} threads={threads} map={map}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn unknown_routes_and_implicit_all_threads_are_refused() {
        let r = Box::leak(Box::new(Rgb8Image {
            width: 1,
            height: 1,
            pixels: vec![1, 2, 3],
        }));
        assert!(make_call("dvifm", r, r, false, 1).is_err());
        assert!(make_call(NAMES[0], r, r, false, 0).is_err());
        assert!(
            inventory()
                .iter()
                .all(|v| v["publish"] == false && v["preset_sha256"].as_str().unwrap().len() == 64)
        );
    }
}
