// Numerical observations for the existing checkpoint fixtures, without timing.
// The caller supplies unmodified `ssimulacra1` and `flip` modules from one
// recorded source revision. Compile this through the generated main.rs wrapper.
fn main() {
    let (w, h) = (64, 64);
    let reference: Vec<u8> = (0..h)
        .flat_map(|y| {
            (0..w).flat_map(move |x| {
                (0..3).map(move |c| ((x * 13 + y * 17 + c * 71 + (x * y) % 29) % 256) as u8)
            })
        })
        .collect();
    let mut distorted = reference.clone();
    for pixel in distorted.iter_mut().step_by(17) {
        *pixel ^= 63;
    }
    let score = ssimulacra1::score(&reference, &distorted, w, h, w * 3, w * 3).unwrap();
    println!("SSIMULACRA1 {score:.17} {:016x}", score.to_bits());
    let reference: Vec<u8> = (0..w * h * 3)
        .map(|i| ((i * 37 + i / 11) % 256) as u8)
        .collect();
    let mut distorted = reference.clone();
    for pixel in distorted.iter_mut().step_by(17) {
        *pixel ^= 63;
    }
    let map = flip::error_map(&reference, &distorted, w, h, w * 3, w * 3).unwrap();
    for (name, score) in [
        ("FLIP", flip::mean(&map)),
        ("FLIP_V17", flip::mean_v17(&map)),
    ] {
        println!("{name} {score:.17} {:016x}", score.to_bits());
    }
    for code in 0u16..256 {
        let value = f32::from(code) / 255.0;
        let linear = if value <= 0.04045 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        };
        println!("LUT {code} {:08x}", linear.to_bits());
    }
}
