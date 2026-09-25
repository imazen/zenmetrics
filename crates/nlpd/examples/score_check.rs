fn main() {
    let w = 97;
    let h = 99;
    let reference = (0..w * h * 3)
        .map(|i| ((i * 37 + i / 97) % 256) as u8)
        .collect::<Vec<_>>();
    let mut changed = reference.clone();
    for p in changed.chunks_exact_mut(3).step_by(19) {
        p[0] = p[0].saturating_add(20);
    }
    println!(
        "{:.15}",
        nlpd::score_rgb_u8(&reference, &changed, w, h).unwrap()
    );
}
