//! sRGB packed-u8 → linear → DKLd65 opponent planar f32 (scalar reference).
//!
//! Phase 8c.1-B moved these out of `cvvdp-gpu::kernels::color` so the
//! CPU crate owns the canonical scalar implementation; cvvdp-gpu
//! continues to re-export the same paths. GPU-side `#[cube(launch)]`
//! kernels remain in `cvvdp-gpu::kernels::color`.

/// 256-entry sRGB byte → linear-normalized f32 LUT.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::color::SRGB8_TO_LINEAR_LUT;
///
/// assert_eq!(SRGB8_TO_LINEAR_LUT.len(), 256);
/// assert_eq!(SRGB8_TO_LINEAR_LUT[0], 0.0);
/// assert_eq!(SRGB8_TO_LINEAR_LUT[255], 1.0);
///
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

/// Host-side scalar reference for the color stage. Bit-exact with the
/// GPU per-pixel math at f32 precision.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::color::srgb_byte_to_dkl_scalar;
/// use cvvdp::params::DisplayModel;
///
/// let d = DisplayModel::STANDARD_4K;
///
/// let (a_white, rg_white, vy_white) =
///     srgb_byte_to_dkl_scalar(255, 255, 255, d.y_peak, d.y_black, d.y_refl);
/// assert!(a_white > 0.0);
/// assert!(rg_white.abs() < a_white * 0.01);
/// assert!(vy_white.abs() < a_white * 0.02);
///
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

/// Display-aware host-side scalar reference.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::color::{display_byte_to_dkl_scalar, srgb_byte_to_dkl_scalar};
/// use cvvdp::params::DisplayModel;
///
/// let d = DisplayModel::STANDARD_4K;
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
            display
                .eotf
                .forward(vr, display.y_peak, display.y_black, display.y_refl),
            display
                .eotf
                .forward(vg, display.y_peak, display.y_black, display.y_refl),
            display
                .eotf
                .forward(vb, display.y_peak, display.y_black, display.y_refl),
        )
    };

    if matches!(display.eotf, Eotf::Hlg) {
        let s = display.y_peak - display.y_black;
        let bias = display.y_black + display.y_refl;
        let inv_r = if s > 0.0 { (lr - bias) / s } else { 0.0 };
        let inv_g = if s > 0.0 { (lg - bias) / s } else { 0.0 };
        let inv_b = if s > 0.0 { (lb - bias) / s } else { 0.0 };
        let y_s = 0.262_7 * inv_r + 0.678_0 * inv_g + 0.059_3 * inv_b;
        let gamma = crate::params::hlg_system_gamma(display.y_peak, display.e_ambient_lux);
        let factor = if y_s > 0.0 {
            y_s.powf(gamma - 1.0)
        } else {
            0.0
        };
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

/// Display-aware host-side scalar reference for already-linear inputs.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::color::display_linear_rgb_to_dkl_scalar;
/// use cvvdp::params::DisplayModel;
///
/// let d = DisplayModel::STANDARD_4K;
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

/// Resolve a [`crate::params::Eotf`] into its `(tag, gamma_exp)` pair.
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
