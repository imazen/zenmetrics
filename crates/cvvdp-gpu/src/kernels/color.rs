//! sRGB packed-u8 → linear → DKLd65 opponent planar f32.
//!
//! Three stages fused into a single kernel pass:
//!
//! 1. **sRGB EOTF**: byte → linear via a 256-entry LUT. The LUT
//!    encodes cvvdp's `srgb2lin`:
//!    `lin = if p > 0.04045 { ((p + 0.055) / 1.055)^2.4 } else { p / 12.92 }`.
//!    Same numbers as `zensim_gpu::kernels::color::SRGB8_TO_LINEARF32_LUT`.
//!
//! 2. **Display model**: linear normalized → cd/m² emitted luminance:
//!    `L = (Y_peak - Y_black) * lin + Y_black + Y_refl`. Constants
//!    come from [`crate::params::DisplayModel`].
//!
//! 3. **DKL transform**: 3×3 matmul with the combined
//!    [`crate::params::SRGB_LINEAR_TO_DKL`] matrix.
//!
//! Output: three planar f32 buffers (A, RG, VY) in absolute DKL units.
//! cvvdp keeps DKL in cd/m²-scaled units (the CSF stage handles
//! sensitivity scaling), so no post-normalization.

// Tick 514: silence missing_docs warnings on items emitted by the
// #[cube(launch)] macro. The macro generates a sibling module +
// launcher struct + associated fn for each annotated function;
// those items don't inherit the user's rustdoc comment and trigger
// 4 warnings per kernel function. Every user-written pub item in
// this file (SRGB8_TO_LINEAR_LUT const, srgb_byte_to_dkl_scalar
// fn, and srgb_to_dkl_kernel fn itself) IS documented, so this
// allow only suppresses the macro-emitted noise.
#![allow(missing_docs)]

use cubecl::prelude::*;

/// 256-entry sRGB byte → linear-normalized f32 LUT. Matches
/// `zensim_gpu::kernels::color::SRGB8_TO_LINEARF32_LUT` byte-for-byte
/// so the upload paths can share scratch.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::kernels::color::SRGB8_TO_LINEAR_LUT;
///
/// // Length + endpoints.
/// assert_eq!(SRGB8_TO_LINEAR_LUT.len(), 256);
/// assert_eq!(SRGB8_TO_LINEAR_LUT[0], 0.0);
/// assert_eq!(SRGB8_TO_LINEAR_LUT[255], 1.0);
///
/// // Strictly monotonic — byte i+1 always maps to a strictly
/// // larger linear value than byte i.
/// for i in 1..256 {
///     assert!(SRGB8_TO_LINEAR_LUT[i] > SRGB8_TO_LINEAR_LUT[i - 1]);
/// }
/// ```
#[rustfmt::skip]
pub const SRGB8_TO_LINEAR_LUT: [f32; 256] = [
    0.0, 0.000303526991, 0.0006070539821, 0.0009105809731, 0.001214107964, 0.001517634955, 0.001821161946, 0.002124688821,
    0.002428215928, 0.002731742803, 0.00303526991, 0.003346535843, 0.003676507389, 0.004024717025, 0.004391442053, 0.004776953254,
    0.005181516521, 0.005605391692, 0.00604883302, 0.006512090564, 0.006995410193, 0.007499032188, 0.008023193106, 0.008568125777,
    0.009134058841, 0.009721217677, 0.01032982301, 0.0109600937, 0.01161224488, 0.01228648797, 0.0129830325, 0.01370208338,
    0.01444384363, 0.01520851441, 0.0159962941, 0.01680737548, 0.01764195412, 0.01850022003, 0.01938236132, 0.0202885624,
    0.02121900953, 0.02217388526, 0.02315336652, 0.02415763214, 0.02518685907, 0.0262412224, 0.02732089162, 0.02842603996,
    0.02955683507, 0.03071344458, 0.03189603239, 0.03310476616, 0.03433980793, 0.03560131416, 0.03688944876, 0.03820437193,
    0.0395462364, 0.04091519862, 0.04231141135, 0.04373503104, 0.04518620297, 0.04666508734, 0.04817182571, 0.04970656708,
    0.05126945674, 0.05286064744, 0.054480277, 0.05612849072, 0.05780543014, 0.05951123685, 0.06124605238, 0.06301001459,
    0.06480326504, 0.06662593782, 0.06847816706, 0.07036009431, 0.07227185369, 0.07421357185, 0.0761853829, 0.07818742096,
    0.08021982014, 0.0822827071, 0.08437620848, 0.08650045842, 0.08865558356, 0.09084171057, 0.0930589661, 0.09530746937,
    0.09758734703, 0.09989872575, 0.1022417322, 0.1046164855, 0.107023105, 0.1094617099, 0.1119324267, 0.1144353747,
    0.1169706658, 0.1195384264, 0.1221387759, 0.1247718185, 0.127437681, 0.130136475, 0.1328683197, 0.1356333345,
    0.1384316087, 0.1412632912, 0.1441284716, 0.147027269, 0.1499597877, 0.152926147, 0.155926466, 0.1589608341,
    0.1620293707, 0.1651321948, 0.1682693958, 0.171441108, 0.1746474057, 0.1778884232, 0.1811642498, 0.1844749898,
    0.1878207773, 0.1912016869, 0.1946178377, 0.1980693191, 0.2015562505, 0.2050787359, 0.208636865, 0.2122307569,
    0.2158605009, 0.2195262015, 0.2232279629, 0.2269658744, 0.2307400554, 0.2345505804, 0.2383975685, 0.242281124,
    0.2462013215, 0.2501582801, 0.2541520894, 0.2581828535, 0.2622506618, 0.2663556039, 0.2704977989, 0.2746773064,
    0.2788942754, 0.2831487358, 0.2874408364, 0.291770637, 0.2961382568, 0.3005437851, 0.3049873114, 0.309468925,
    0.3139887154, 0.318546772, 0.323143214, 0.327778101, 0.3324515224, 0.3371636271, 0.3419144154, 0.3467040658,
    0.3515326083, 0.3564001322, 0.3613067865, 0.3662526011, 0.3712376952, 0.3762621284, 0.3813260198, 0.3864294291,
    0.3915724754, 0.3967552185, 0.4019777775, 0.407240212, 0.4125426114, 0.4178850651, 0.4232676625, 0.4286904931,
    0.4341536462, 0.4396571815, 0.4452011883, 0.4507857859, 0.4564110339, 0.4620769918, 0.4677838087, 0.4735314846,
    0.4793201685, 0.4851499498, 0.4910208583, 0.4969329834, 0.5028864741, 0.5088813305, 0.5149176717, 0.5209955573,
    0.5271151066, 0.5332763791, 0.5394794941, 0.5457244515, 0.5520114303, 0.5583403707, 0.5647115111, 0.5711248517,
    0.577580452, 0.5840784311, 0.5906188488, 0.5972017646, 0.6038273573, 0.6104955673, 0.6172065735, 0.6239603758,
    0.630757153, 0.6375968456, 0.644479692, 0.6514056325, 0.658374846, 0.6653872728, 0.6724431515, 0.6795424819,
    0.6866853237, 0.6938717365, 0.7011018991, 0.708375752, 0.7156934738, 0.7230551243, 0.730460763, 0.7379103899,
    0.7454041839, 0.7529422045, 0.7605245113, 0.7681511641, 0.7758222222, 0.7835378051, 0.7912979126, 0.7991027236,
    0.8069522381, 0.8148465753, 0.8227857351, 0.8307698965, 0.8387989998, 0.8468732238, 0.8549926281, 0.8631572127,
    0.8713670969, 0.8796223998, 0.8879231215, 0.896269381, 0.9046611786, 0.9130986333, 0.9215818644, 0.9301108718,
    0.9386857152, 0.9473065138, 0.9559733272, 0.9646862745, 0.9734452963, 0.9822505713, 0.9911020994, 1.0,
];

/// Host-side scalar reference for the color stage. Bit-exact with
/// `srgb_to_dkl_kernel`'s per-pixel math at f32 precision. Used by
/// unit tests and by host-side debug taps.
///
/// Returns `(dkl_a, dkl_rg, dkl_vy)` for one pixel.
///
/// Hardcodes sRGB EOTF + BT.709 primaries — that's the v1 contract
/// every parity golden was captured under. For a per-`DisplayModel`
/// variant that dispatches on the model's `eotf` + `primaries`
/// fields, see [`display_byte_to_dkl_scalar`].
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::kernels::color::srgb_byte_to_dkl_scalar;
/// use cvvdp_gpu::params::DisplayModel;
///
/// let d = DisplayModel::STANDARD_4K;
///
/// // Pure white → positive achromatic (A), near-zero chroma.
/// let (a_white, rg_white, vy_white) =
///     srgb_byte_to_dkl_scalar(255, 255, 255, d.y_peak, d.y_black, d.y_refl);
/// assert!(a_white > 0.0);
/// // RG / VY rows are mean-zero in DKL — grayscale → tiny chroma.
/// // Actual ratios from the bit-pinned matrix: |RG|/|A| ≈ 0.36%,
/// // |VY|/|A| ≈ 0.98% at (255,255,255) under STANDARD_4K (pinned by
/// // the GOLDENS table in tests/color_scalar.rs). The tolerances
/// // below are ~3× and ~2× the actual values to leave room for
/// // alternate display models that shift `y_peak`/`y_refl`, but
/// // still tight enough to surface a matrix row-sum drift.
/// assert!(rg_white.abs() < a_white * 0.01);
/// assert!(vy_white.abs() < a_white * 0.02);
///
/// // Pure red → positive RG (opposes R against G + B).
/// let (_, rg_red, _) =
///     srgb_byte_to_dkl_scalar(255, 0, 0, d.y_peak, d.y_black, d.y_refl);
/// assert!(rg_red > 0.0);
/// ```
#[inline]
#[must_use]
pub fn srgb_byte_to_dkl_scalar(
    r: u8,
    g: u8,
    b: u8,
    y_peak: f32,
    y_black: f32,
    y_refl: f32,
) -> (f32, f32, f32) {
    use crate::params::SRGB_LINEAR_TO_DKL as M;

    let lin_r = SRGB8_TO_LINEAR_LUT[r as usize];
    let lin_g = SRGB8_TO_LINEAR_LUT[g as usize];
    let lin_b = SRGB8_TO_LINEAR_LUT[b as usize];

    let s = y_peak - y_black;
    let bias = y_black + y_refl;
    let lr = s * lin_r + bias;
    let lg = s * lin_g + bias;
    let lb = s * lin_b + bias;

    let a = M[0][0] * lr + M[0][1] * lg + M[0][2] * lb;
    let rg = M[1][0] * lr + M[1][1] * lg + M[1][2] * lb;
    let vy = M[2][0] * lr + M[2][1] * lg + M[2][2] * lb;
    (a, rg, vy)
}

/// Display-aware host-side scalar reference. Dispatches on the
/// display's [`Eotf`] and [`Primaries`] to handle non-sRGB / non-
/// BT.709 inputs.
///
/// For [`Eotf::Srgb`] + [`Primaries::Bt709`] the output is
/// bit-identical to [`srgb_byte_to_dkl_scalar`] — the two paths
/// share the same LUT, the same constants, and the same matrix
/// row in the dispatch.
///
/// For non-sRGB EOTFs (PQ / HLG / Linear / Gamma / BT.1886) the
/// 8-bit pixel encoding is interpreted as the corresponding
/// 0..255 → 0..1 → EOTF chain. HLG additionally applies the
/// per-pixel OOTF after inverse-OETF; the OOTF needs the RGB
/// triple's `Y_s` so this is the natural API boundary for it.
///
/// Output: `(dkl_a, dkl_rg, dkl_vy)` in cd/m²-scaled DKL.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::kernels::color::{display_byte_to_dkl_scalar, srgb_byte_to_dkl_scalar};
/// use cvvdp_gpu::params::DisplayModel;
///
/// let d = DisplayModel::STANDARD_4K;
/// // Under sRGB / BT.709 the two paths agree bit-for-bit.
/// let (a1, rg1, vy1) = srgb_byte_to_dkl_scalar(200, 50, 100, d.y_peak, d.y_black, d.y_refl);
/// let (a2, rg2, vy2) = display_byte_to_dkl_scalar(200, 50, 100, d);
/// assert_eq!(a1.to_bits(), a2.to_bits());
/// assert_eq!(rg1.to_bits(), rg2.to_bits());
/// assert_eq!(vy1.to_bits(), vy2.to_bits());
/// ```
#[inline]
#[must_use]
pub fn display_byte_to_dkl_scalar(
    r: u8,
    g: u8,
    b: u8,
    display: crate::params::DisplayModel,
) -> (f32, f32, f32) {
    use crate::params::Eotf;

    // sRGB stays bit-identical to srgb_byte_to_dkl_scalar by
    // routing through the precomputed LUT. Other EOTFs normalise
    // to 0..1 first and run their inverse-EOTF formula.
    let (mut lr, mut lg, mut lb) = if matches!(display.eotf, Eotf::Srgb) {
        let lin_r = SRGB8_TO_LINEAR_LUT[r as usize];
        let lin_g = SRGB8_TO_LINEAR_LUT[g as usize];
        let lin_b = SRGB8_TO_LINEAR_LUT[b as usize];
        let s = display.y_peak - display.y_black;
        let bias = display.y_black + display.y_refl;
        (s * lin_r + bias, s * lin_g + bias, s * lin_b + bias)
    } else {
        let vr = (r as f32) * (1.0 / 255.0);
        let vg = (g as f32) * (1.0 / 255.0);
        let vb = (b as f32) * (1.0 / 255.0);
        (
            display.eotf.forward(vr, display.y_peak, display.y_black, display.y_refl),
            display.eotf.forward(vg, display.y_peak, display.y_black, display.y_refl),
            display.eotf.forward(vb, display.y_peak, display.y_black, display.y_refl),
        )
    };

    // HLG OOTF (system gamma applied to the linear triple per
    // BT.2100). Inverse OETF inside Eotf::forward gave us the
    // scene-relative-linear values pre-scaling; we redo the
    // scaling here after the OOTF so the math matches pycvvdp's
    // forward(EOTF='HLG') branch. Other EOTFs are unaffected.
    if matches!(display.eotf, Eotf::Hlg) {
        // Strip the bias added by Eotf::forward so we get back to
        // (y_peak - y_black) * inverse_oetf(v); divide by
        // (y_peak - y_black) to recover inverse_oetf(v).
        let s = display.y_peak - display.y_black;
        let bias = display.y_black + display.y_refl;
        let inv_r = if s > 0.0 { (lr - bias) / s } else { 0.0 };
        let inv_g = if s > 0.0 { (lg - bias) / s } else { 0.0 };
        let inv_b = if s > 0.0 { (lb - bias) / s } else { 0.0 };
        // Compute Y_s using BT.2100 luma coefficients (R, G, B)
        // = (0.2627, 0.6780, 0.0593). Same coefficients as the
        // BT.2020 RGB2Y row in upstream's color_spaces.json.
        let y_s = 0.262_7 * inv_r + 0.678_0 * inv_g + 0.059_3 * inv_b;
        let gamma = crate::params::hlg_system_gamma(display.y_peak, display.e_ambient_lux);
        let factor = if y_s > 0.0 { y_s.powf(gamma - 1.0) } else { 0.0 };
        lr = s * (inv_r * factor) + bias;
        lg = s * (inv_g * factor) + bias;
        lb = s * (inv_b * factor) + bias;
    }

    let m = display.primaries.linear_rgb_to_dkl();
    let a = m[0][0] * lr + m[0][1] * lg + m[0][2] * lb;
    let rg = m[1][0] * lr + m[1][1] * lg + m[1][2] * lb;
    let vy = m[2][0] * lr + m[2][1] * lg + m[2][2] * lb;
    (a, rg, vy)
}

/// Display-aware host-side scalar reference for already-linear
/// inputs (e.g., HDR EXR loaded as linear-light floats). Skips
/// the EOTF entirely (the input is already in the post-EOTF
/// linear-RGB space) and applies the per-primaries DKL matrix
/// directly. The display's `y_peak`, `y_black`, `y_refl` are
/// still applied as the "display scaling" step so the output
/// matches a `[0..1] linear sRGB` input passed through
/// [`Eotf::Linear`].
///
/// Use this for the linear-RGB-planes API entry points; the
/// 8-bit display path goes through [`display_byte_to_dkl_scalar`].
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::kernels::color::display_linear_rgb_to_dkl_scalar;
/// use cvvdp_gpu::params::DisplayModel;
///
/// let d = DisplayModel::STANDARD_4K;
/// // (1.0, 1.0, 1.0) linear → maps to y_peak + y_refl approximately.
/// let (a, _, _) = display_linear_rgb_to_dkl_scalar(1.0, 1.0, 1.0, d);
/// assert!(a > 0.0);
/// ```
#[inline]
#[must_use]
pub fn display_linear_rgb_to_dkl_scalar(
    r: f32,
    g: f32,
    b: f32,
    display: crate::params::DisplayModel,
) -> (f32, f32, f32) {
    let s = display.y_peak - display.y_black;
    let bias = display.y_black + display.y_refl;
    let lr = s * r + bias;
    let lg = s * g + bias;
    let lb = s * b + bias;

    let m = display.primaries.linear_rgb_to_dkl();
    let a = m[0][0] * lr + m[0][1] * lg + m[0][2] * lb;
    let rg = m[1][0] * lr + m[1][1] * lg + m[1][2] * lb;
    let vy = m[2][0] * lr + m[2][1] * lg + m[2][2] * lb;
    (a, rg, vy)
}

/// Runtime EOTF tag passed to GPU color kernels.
///
/// Mirrors [`crate::params::Eotf`] as a packed `u32` because cubecl's
/// `#[cube]` macro can't yet specialize on bool/enum comptime generics
/// (see `docs/CUBECL_GOTCHAS.md` §G1.7). The kernel branches on the
/// tag value; identical to the host-side `Eotf::forward` dispatch.
///
/// Tag values:
/// - `0` → [`Eotf::Srgb`] (default, sRGB EOTF via 256-entry LUT)
/// - `1` → [`Eotf::Pq`] (SMPTE ST 2084, absolute cd/m²)
/// - `2` → [`Eotf::Hlg`] (BT.2100 HLG, inverse-OETF + per-pixel OOTF)
/// - `3` → [`Eotf::Linear`] (input already linear-light)
/// - `4` → [`Eotf::Bt1886`] (BT.1886 display gamma 2.4 with lift)
/// - `5` → [`Eotf::Gamma`] (generic power-law; gamma exponent passed
///   as a separate runtime scalar `gamma_exp` because the variant
///   payload is dynamic)
///
/// [`Eotf::Srgb`]: crate::params::Eotf::Srgb
/// [`Eotf::Pq`]: crate::params::Eotf::Pq
/// [`Eotf::Hlg`]: crate::params::Eotf::Hlg
/// [`Eotf::Linear`]: crate::params::Eotf::Linear
/// [`Eotf::Bt1886`]: crate::params::Eotf::Bt1886
/// [`Eotf::Gamma`]: crate::params::Eotf::Gamma
pub mod eotf_tag {
    /// sRGB / BT.709 EOTF (default).
    pub const SRGB: u32 = 0;
    /// SMPTE ST 2084 PQ.
    pub const PQ: u32 = 1;
    /// BT.2100 Hybrid Log-Gamma.
    pub const HLG: u32 = 2;
    /// Linear-light input.
    pub const LINEAR: u32 = 3;
    /// BT.1886 display gamma 2.4 + black lift.
    pub const BT1886: u32 = 4;
    /// Generic power-law gamma (exponent passed as runtime scalar).
    pub const GAMMA: u32 = 5;
}

/// Resolve a [`crate::params::Eotf`] into its `(tag, gamma_exp)` pair
/// for passing to GPU color kernels. The `gamma_exp` payload is only
/// meaningful when `tag == eotf_tag::GAMMA`; other variants ignore it
/// (any sentinel value passes through the kernel branch).
#[must_use]
pub fn eotf_tag_and_gamma(eotf: crate::params::Eotf) -> (u32, f32) {
    use crate::params::Eotf;
    match eotf {
        Eotf::Srgb => (eotf_tag::SRGB, 0.0),
        Eotf::Pq => (eotf_tag::PQ, 0.0),
        Eotf::Hlg => (eotf_tag::HLG, 0.0),
        Eotf::Linear => (eotf_tag::LINEAR, 0.0),
        Eotf::Bt1886 => (eotf_tag::BT1886, 0.0),
        Eotf::Gamma(g) => (eotf_tag::GAMMA, g),
    }
}

/// In-kernel EOTF apply. Branches on `eotf_tag` to mirror the host
/// [`crate::params::Eotf::forward`] dispatch.
///
/// Input `v` is the byte / linear value in 0..1 normalized space (the
/// caller divides bytes by 255). Linear EOTF accepts values >1
/// (HDR linear-light cd/m² inputs); PQ accepts 0..1 PQ-encoded.
///
/// Output is the per-channel linear-cd/m² scene-light, BEFORE the
/// HLG OOTF for HLG inputs (that step depends on the RGB triple's
/// `Y_s` and is applied separately in the kernel body). For non-HLG
/// EOTFs the output is already in cd/m² and ready for the DKL matmul.
///
/// `#[cube]` doesn't support early `return`, so the dispatch uses
/// chained `if/else` expressions — semantically equivalent to a match
/// on the tag.
#[cube]
fn apply_eotf_branch(
    v: f32,
    eotf_tag: u32,
    gamma_exp: f32,
    y_peak: f32,
    y_black: f32,
    y_refl: f32,
) -> f32 {
    let bias = y_black + y_refl;
    let scale = y_peak - y_black;

    let v_clamped = if v < f32::new(0.0) {
        f32::new(0.0)
    } else if v > f32::new(1.0) {
        f32::new(1.0)
    } else {
        v
    };

    if eotf_tag == 1u32 {
        // PQ (SMPTE ST 2084). Reference: pycvvdp `pq2lin`.
        let l_max = f32::new(10000.0);
        let m1 = f32::new(0.159_301_75);
        let m2 = f32::new(78.843_75);
        let c1 = f32::new(0.835_937_5);
        let c2 = f32::new(18.851_562);
        let c3 = f32::new(18.687_5);
        // PQ accepts raw v (no 0..1 clamp; HDR PQ-encoded can exceed
        // 1 in theory but realistic inputs are clamped upstream).
        let im_t = f32::powf(v, f32::new(1.0) / m2);
        let num_raw = im_t - c1;
        let num = if num_raw < f32::new(0.0) {
            f32::new(0.0)
        } else {
            num_raw
        };
        let den = c2 - c3 * im_t;
        let lin = l_max * f32::powf(num / den, f32::new(1.0) / m1);
        let floor_val = f32::new(0.005);
        let clamped_lo = if lin < floor_val { floor_val } else { lin };
        let clamped = if clamped_lo > y_peak { y_peak } else { clamped_lo };
        clamped + bias
    } else if eotf_tag == 2u32 {
        // HLG inverse OETF. OOTF applied by caller (depends on Y_s).
        let a = f32::new(0.178_832_77);
        let b = f32::new(1.0) - f32::new(4.0) * a;
        let c = f32::new(0.5) - a * f32::ln(f32::new(4.0) * a);
        let lin = if v_clamped <= f32::new(0.5) {
            (v_clamped * v_clamped) / f32::new(3.0)
        } else {
            (f32::exp((v_clamped - c) / a) + b) / f32::new(12.0)
        };
        scale * lin + bias
    } else if eotf_tag == 3u32 {
        // Linear-light input. Clip to [max(0.005, y_black), y_peak]
        // then add y_refl (NOT bias — Linear's path doesn't re-add
        // y_black, per pycvvdp's branch).
        let floor_val = f32::new(0.005);
        let floor_eff = if y_black > floor_val { y_black } else { floor_val };
        let clamped_lo = if v < floor_eff { floor_eff } else { v };
        let clamped = if clamped_lo > y_peak { y_peak } else { clamped_lo };
        clamped + y_refl
    } else if eotf_tag == 4u32 {
        // BT.1886 — gamma 2.4 with black-level lift. L = a · (V + b)^γ.
        let gamma = f32::new(2.4);
        let inv_gamma = f32::new(1.0) / gamma;
        let y_p_g = f32::powf(y_peak, inv_gamma);
        let y_b_g = f32::powf(y_black, inv_gamma);
        let lift_a = f32::powf(y_p_g - y_b_g, gamma);
        let lift_b = y_b_g / (y_p_g - y_b_g);
        let sum = v_clamped + lift_b;
        let sum_pos = if sum < f32::new(0.0) {
            f32::new(0.0)
        } else {
            sum
        };
        let l = lift_a * f32::powf(sum_pos, gamma);
        l + y_refl
    } else if eotf_tag == 5u32 {
        // Generic power-law gamma (Adobe RGB 2.2, Apple RGB 1.8, …).
        let lin = f32::powf(v_clamped, gamma_exp);
        scale * lin + bias
    } else {
        // Default / fallback: sRGB closed-form. The caller takes the
        // LUT path when it knows the EOTF is sRGB; this branch only
        // fires if the linear-planes / non-byte entry routes a tag-0
        // value through here.
        let lin = if v_clamped > f32::new(0.040_45) {
            f32::powf(
                (v_clamped + f32::new(0.055)) / f32::new(1.055),
                f32::new(2.4),
            )
        } else {
            v_clamped / f32::new(12.92)
        };
        scale * lin + bias
    }
}

/// HLG OOTF (system gamma applied to the linear-RGB triple per
/// BT.2100). Computes Y_s from the inverse-OETF values, derives the
/// per-pixel factor `Y_s^(γ-1)`, and re-scales each channel.
///
/// `gamma` is the precomputed HLG system gamma (host-side function
/// `hlg_system_gamma(y_peak, e_ambient_lux)` — passed in as a
/// runtime scalar since it doesn't vary per pixel).
///
/// Returns the OOTF-adjusted `(lr, lg, lb)` already in display-light
/// cd/m² (the scale + bias step is folded in).
#[cube]
fn hlg_ootf(
    lr_pre: f32,
    lg_pre: f32,
    lb_pre: f32,
    y_peak: f32,
    y_black: f32,
    y_refl: f32,
    gamma: f32,
) -> (f32, f32, f32) {
    let scale = y_peak - y_black;
    let bias = y_black + y_refl;
    // Strip the bias / scale applied by apply_eotf_branch so we get
    // back to inverse_oetf(v) in 0..12.
    let inv_r = if scale > f32::new(0.0) {
        (lr_pre - bias) / scale
    } else {
        f32::new(0.0)
    };
    let inv_g = if scale > f32::new(0.0) {
        (lg_pre - bias) / scale
    } else {
        f32::new(0.0)
    };
    let inv_b = if scale > f32::new(0.0) {
        (lb_pre - bias) / scale
    } else {
        f32::new(0.0)
    };
    // BT.2100 luma coefficients (R, G, B) = (0.2627, 0.6780, 0.0593).
    let y_s = f32::new(0.262_7) * inv_r + f32::new(0.678_0) * inv_g + f32::new(0.059_3) * inv_b;
    let factor = if y_s > f32::new(0.0) {
        f32::powf(y_s, gamma - f32::new(1.0))
    } else {
        f32::new(0.0)
    };
    let lr = scale * (inv_r * factor) + bias;
    let lg = scale * (inv_g * factor) + bias;
    let lb = scale * (inv_b * factor) + bias;
    (lr, lg, lb)
}

/// 8-bit packed-RGB → DKL planar f32, with display dispatch on EOTF
/// and primaries.
///
/// Inputs:
/// - `src` — `width × height` packed sRGB bytes (R | G<<8 | B<<16).
/// - `lut` — uploaded [`SRGB8_TO_LINEAR_LUT`] (256 entries). Read only
///   on the sRGB fast path; ignored for non-sRGB EOTFs.
///
/// Outputs:
/// - `out_a`, `out_rg`, `out_vy` — `width × height` planar f32 in
///   DKLd65 opponent space (cd/m²-scaled).
///
/// Runtime dispatch:
/// - `eotf_tag` — see [`eotf_tag`] constants. `0` = sRGB takes the
///   fast LUT path; any other value runs the closed-form EOTF via
///   [`apply_eotf_branch`].
/// - `gamma_exp` — exponent for [`eotf_tag::GAMMA`]; ignored for
///   other tags.
/// - `m00..m22` — 9 runtime scalars carrying the per-primaries
///   linear-RGB→DKL matrix ([`crate::params::Primaries::linear_rgb_to_dkl`]).
///   Pushed as scalars (not constants) so a single kernel binary
///   serves every primaries set; LLVM still folds the linear combo
///   when the values are constant across the launch.
/// - `hlg_gamma` — precomputed HLG system gamma. Only consumed when
///   `eotf_tag == eotf_tag::HLG`.
///
/// The sRGB / BT.709 fast path matches the historical
/// `srgb_to_dkl_kernel` output bit-for-bit (LUT + folded matrix
/// constants come from the same vendored numbers).
#[cube(launch)]
pub fn srgb_to_dkl_kernel(
    src: &Array<u32>,
    lut: &Array<f32>,
    out_a: &mut Array<f32>,
    out_rg: &mut Array<f32>,
    out_vy: &mut Array<f32>,
    width: u32,
    height: u32,
    y_peak: f32,
    y_black: f32,
    y_refl: f32,
    eotf_tag: u32,
    gamma_exp: f32,
    hlg_gamma: f32,
    m00: f32,
    m01: f32,
    m02: f32,
    m10: f32,
    m11: f32,
    m12: f32,
    m20: f32,
    m21: f32,
    m22: f32,
) {
    let idx = ABSOLUTE_POS;
    let total = (width * height) as usize;
    if idx >= total {
        terminate!();
    }

    // T4.L (2026-05-16): packed-RGBA upload. Host packs 3 sRGB bytes
    // per pixel into one u32 (R in low byte, then G, then B; alpha
    // unused). Cuts the H→D transfer 3× vs the prior u8-widened-to-u32
    // path (144 MB → 48 MB at 12 MP); the per-iter `create_from_slice`
    // alloc shrinks in proportion. 3 bit-shifts + 3 ANDs per pixel are
    // free relative to the upload time saved.
    let packed = src[idx];
    let r_byte = packed & 0xffu32;
    let g_byte = (packed >> 8u32) & 0xffu32;
    let b_byte = (packed >> 16u32) & 0xffu32;

    // Per-channel EOTF: sRGB fast path (LUT + scale/bias) on tag=0,
    // closed-form `apply_eotf_branch` on every other tag. Linear-light
    // input is 0..1 byte/255 normalised before the branch (matches the
    // host scalar's `display_byte_to_dkl_scalar` shape).
    let inv_255 = f32::new(1.0) / f32::new(255.0);
    let s = y_peak - y_black;
    let bias = y_black + y_refl;
    let lr_pre = if eotf_tag == 0u32 {
        let lin_r = lut[r_byte as usize];
        s * lin_r + bias
    } else {
        let vr = (r_byte as f32) * inv_255;
        apply_eotf_branch(vr, eotf_tag, gamma_exp, y_peak, y_black, y_refl)
    };
    let lg_pre = if eotf_tag == 0u32 {
        let lin_g = lut[g_byte as usize];
        s * lin_g + bias
    } else {
        let vg = (g_byte as f32) * inv_255;
        apply_eotf_branch(vg, eotf_tag, gamma_exp, y_peak, y_black, y_refl)
    };
    let lb_pre = if eotf_tag == 0u32 {
        let lin_b = lut[b_byte as usize];
        s * lin_b + bias
    } else {
        let vb = (b_byte as f32) * inv_255;
        apply_eotf_branch(vb, eotf_tag, gamma_exp, y_peak, y_black, y_refl)
    };

    // HLG: per-pixel OOTF using the RGB triple's Y_s. Other EOTFs
    // already produced final display-light cd/m².
    let (lr, lg, lb) = if eotf_tag == 2u32 {
        hlg_ootf(lr_pre, lg_pre, lb_pre, y_peak, y_black, y_refl, hlg_gamma)
    } else {
        (lr_pre, lg_pre, lb_pre)
    };

    out_a[idx] = m00 * lr + m01 * lg + m02 * lb;
    out_rg[idx] = m10 * lr + m11 * lg + m12 * lb;
    out_vy[idx] = m20 * lr + m21 * lg + m22 * lb;
}
