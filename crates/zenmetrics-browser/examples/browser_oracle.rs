//! Native library oracle, independent of the browser binding and worker ingress.
use zensim::{DiffmapWeighting, RgbSlice, Zensim, ZensimProfile};
fn main() {
    let (w, h) = (64, 64);
    let reference: Vec<[u8; 3]> = (0..h)
        .flat_map(|y| {
            (0..w).map(move |x| std::array::from_fn(|c| ((x * 17 + y * 13 + c * 41) % 256) as u8))
        })
        .collect();
    let distorted: Vec<_> = reference
        .iter()
        .map(|p| p.map(|v| v.saturating_sub(7)))
        .collect();
    let result = Zensim::new(ZensimProfile::codec_target())
        .compute_with_diffmap(
            &RgbSlice::new(&reference, w, h),
            &RgbSlice::new(&distorted, w, h),
            DiffmapWeighting::default(),
        )
        .unwrap();
    println!(
        "{}",
        serde_json::json!({"score":result.score(),"map":result.diffmap()})
    );
}
