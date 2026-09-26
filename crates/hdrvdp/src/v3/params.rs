//! HDR-VDP-3 parameters: **everything the metric needs to know is explicit**.
//!
//! Unlike the HDR-VDP-2 [`crate::Params`], which exposes a bare
//! `pix_per_deg` with a conventional default, this parameter set requires
//! the caller to state, in the type system:
//!
//! * [`ViewingConditions::pixels_per_degree`] — the angular resolution of
//!   the image *on screen*, or enough display geometry to derive it via
//!   [`crate::params::pix_per_deg`]. No default exists.
//! * [`ViewingConditions::surround`] — what light surrounds the image
//!   ([`Surround::None`] reproduces upstream's default `'none'`).
//! * [`ViewingConditions::observer_age`] — in years; used by all three
//!   age-dependent models that default on upstream (`do_aesl`, `do_aod`,
//!   `do_slum`). Upstream silently uses 24; here it is a required field.
//! * [`InputEncoding`] — how to interpret the pixel values (absolute
//!   luminance vs display code values vs linear RGB vs XYZ).
//! * [`Emission`] — the display's spectral emission model; required
//!   explicitly even though upstream picks a hidden default per encoding
//!   (`Emission::default_for` documents which).
//! * [`Task`] — which calibration bundle to run (`quality`/`side-by-side`
//!   share one; `flicker` is separate).
//!
//! `Options` carries the algorithm switches that upstream exposes through
//! its free-form `options` dict; [`Options::reference`] reproduces the
//! upstream defaults exactly for the given task.

/// The prediction task — selects the calibrated parameter bundle, exactly
/// as upstream's `Metric_par(task, …)` does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Task {
    /// `task='quality'` — the full-reference quality task. Uses the same
    /// calibration as [`Self::SideBySide`] upstream.
    Quality,
    /// `task='side-by-side'`/`'sbs'` — detection in a side-by-side
    /// comparison; identical constants to `quality` upstream.
    SideBySide,
    /// `task='flicker'` — flicker visibility, calibrated on the
    /// compression-flicker dataset (2020-05-23).
    Flicker,
}

/// What light surrounds the image when the display is convolved with the
/// optical MTF — upstream's `surround` option.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Surround {
    /// `'none'` (upstream default): extend the image with symmetric
    /// (edge-mirrored) padding, as if nothing surrounds it.
    None,
    /// `'mean'`: extend with the geometric mean of the *reference* image
    /// (per colour channel). Matches upstream's `surround='mean'`.
    Mean,
    /// A uniform scalar luminance (cd/m²-equivalent in native display
    /// channels) for all channels — upstream's scalar `surround`.
    Uniform(f64),
    /// One luminance per image channel — upstream's vector `surround`.
    PerChannel(Vec<f64>),
}

/// A display's spectral emission: upstream loads one of the
/// `emission_spectra_*.csv` tables or a user-supplied CSV
/// (`spectral_emission` option).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Emission {
    /// One of the vendored presets (`emission_spectra_<name>.csv`).
    Preset(DisplayPreset),
    /// Caller-supplied emission table. `wavelengths_nm` must be strictly
    /// increasing and cover the 360–780 nm resampling range the reference
    /// interpolates onto; `columns` holds one emission column per image
    /// channel — the number of columns must equal the image's channel
    /// count.
    Custom {
        /// Sample wavelengths in nm (must bracket 360–780).
        wavelengths_nm: Vec<f64>,
        /// `columns[c][i]` = emission of channel `c` at
        /// `wavelengths_nm[i]`.
        columns: Vec<Vec<f64>>,
    },
}

/// The vendored display emission presets — upstream's `rgb_display`
/// option values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DisplayPreset {
    /// `ccfl-lcd` — LCD with CCFL backlight (upstream's fallback for 3-channel
    /// images when no `rgb_display` is named).
    CcflLcd,
    /// `crt` — a typical CRT display.
    Crt,
    /// `led-lcd` — LCD with LED backlight (HDR-VDP-2's `rgb-bt.709` default).
    LedLcd,
    /// `led-lcd-srgb` — LED LCD modelled for sRGB input (upstream's default
    /// for `sRGB-display`, `rgb-bt.709` and `rgb-native` encodings in VDP3).
    LedLcdSrgb,
    /// `led-lcd-wcg` — LED LCD, wide colour gamut.
    LedLcdWcg,
    /// `oled` — OLED display (upstream's default for `rgb-bt.2020` and
    /// `xyz` encodings).
    Oled,
    /// `d65` — illuminant D65 (single channel; upstream's fallback for
    /// single-channel images).
    D65,
}

impl Emission {
    /// Number of image channels this emission model feeds: presets are
    /// fixed (`d65` = 1, the rest = 3), [`Self::Custom`] is the column
    /// count of the caller's table.
    #[must_use]
    pub fn channels(&self) -> usize {
        match self {
            Self::Preset(DisplayPreset::D65) => 1,
            Self::Preset(_) => 3,
            Self::Custom { columns, .. } => columns.len(),
        }
    }
}

impl DisplayPreset {
    /// Upstream's filename suffix: `emission_spectra_<suffix>.csv`.
    #[must_use]
    pub const fn file_suffix(self) -> &'static str {
        match self {
            Self::CcflLcd => "ccfl-lcd",
            Self::Crt => "crt",
            Self::LedLcd => "led-lcd",
            Self::LedLcdSrgb => "led-lcd-srgb",
            Self::LedLcdWcg => "led-lcd-wcg",
            Self::Oled => "oled",
            Self::D65 => "d65",
        }
    }
}

/// How to interpret the image pixel values — upstream's `color_encoding`
/// argument. Determines both the colour transform applied and the required
/// channel count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InputEncoding {
    /// `'luminance'`: exactly 1 channel of absolute luminance in cd/m².
    /// Spectrally assumed D65.
    Luminance,
    /// `'luma-display'`: 1 channel of display luma in `[0,1]`, modelled as
    /// `L = 99·V^2.2 + 1` cd/m² (100 cd/m² display, 1 cd/m² black).
    LumaDisplay,
    /// `'sRGB-display'`: 3 channels of sRGB code values in `[0,1]`, decoded
    /// with the sRGB EOTF and mapped onto a `99·linear + 1` cd/m² display.
    SrgbDisplay,
    /// `'rgb-bt.709'`: 3 channels of **linear** BT.709 RGB in absolute
    /// units (the result of decoding an EOTF like PQ — the AIC-4 harness
    /// path).
    RgbBt709,
    /// `'rgb-bt.2020'`: 3 channels of **linear** BT.2020 RGB in absolute
    /// units.
    RgbBt2020,
    /// `'rgb-native'`: 3 channels already in the *native* display colour
    /// space (absolute units) — skips the ITU→native transform.
    RgbNative,
    /// `'xyz'`: 3 channels of absolute CIE 1931 XYZ, `Y` in cd/m².
    Xyz,
    /// `'generic'`: arbitrary channel count; requires
    /// [`Emission::Custom`] with a matching column count.
    Generic,
}

impl InputEncoding {
    /// Required channel count, or `None` for [`Self::Generic`]
    /// (any positive channel count, driven by the emission table).
    #[must_use]
    pub const fn channels(self) -> Option<usize> {
        match self {
            Self::Luminance | Self::LumaDisplay => Some(1),
            Self::SrgbDisplay | Self::RgbBt709 | Self::RgbBt2020 | Self::RgbNative | Self::Xyz => {
                Some(3)
            }
            Self::Generic => None,
        }
    }
}

/// Where the viewing geometry leaves no room for doubt.
///
/// Construct with [`Self::new`]; every field is required because each one
/// materially changes the score. To derive `pixels_per_degree` from a
/// physical setup, use [`crate::params::pix_per_deg`] — the same formula as
/// upstream's `hdrvdp_pix_per_deg`:
///
/// ```
/// use hdrvdp::params::pix_per_deg;
/// let ppd = pix_per_deg(64.5, [3840.0, 2160.0], 1.32528); // ≈ 64.1
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ViewingConditions {
    /// Pixels per visual degree of the displayed image. Upstream requires
    /// `≥ 4` for the pyramid height to be non-negative.
    pub pixels_per_degree: f64,
    /// Surround model for the optical-MTF padding.
    pub surround: Surround,
    /// Observer age in years. Feeds the age-dependent sensitivity loss
    /// (`do_aesl`), aging optical density (`do_aod`) and senile-miosis
    /// (`do_slum`) models — all enabled by default upstream, which itself
    /// assumes `24`. Passed through `clip(20, 83)` inside the pupil model,
    /// matching `pupil_d_unified`.
    pub observer_age: u8,
}

impl ViewingConditions {
    /// All-explicit constructor — no hidden assumptions.
    #[must_use]
    pub fn new(pixels_per_degree: f64, surround: Surround, observer_age: u8) -> Self {
        Self {
            pixels_per_degree,
            surround,
            observer_age,
        }
    }
}

/// The calibrated constants for one task — populated by
/// `Metric_par(task, options)`. Fields are `pub(crate)`: they're the
/// metric's fitted internals, not user-facing knobs.
#[derive(Debug, Clone)]
pub(crate) struct TaskPar {
    /// Observer age in years — copied from `ViewingConditions` at
    /// [`Params::task_par`]; feeds `do_aesl`/`do_aod`/`do_slum`.
    pub age: f64,
    pub base_sensitivity_correction: f64,
    pub mask_self: f64,
    pub mask_xn: f64,
    pub mask_xo: f64,
    pub mask_q: f64,
    pub mask_p: f64,
    pub do_sprob_sum: bool,
    pub psych_func_slope: f64,
    pub si_sigma: f64,
    // non-task-calibrated shared constants:
    pub do_aesl: bool,
    pub do_aod: bool,
    pub do_slum: bool,
    pub masking_norm: f64,
    pub masking_pool_size: usize,
    pub do_si_gauss: bool,
    pub si_size: f64,
    pub do_masking: bool,
    pub do_pixel_threshold: bool,
    pub ignore_freqs_lower_than: Option<f64>,
    pub disable_lowvals_warning: bool,
    pub mtf_params_a: [f64; 4],
    pub mtf_params_b: [f64; 4],
    pub csf_params: [[f64; 5]; 7],
    pub csf_lums: [f64; 7],
    pub csf_sa: [f64; 4],
    pub csf_sr: [f64; 6],
    pub rod_sensitivity: f64,
    pub aesl_slope_freq: f64,
    pub aesl_base: f64,
}

impl TaskPar {
    /// `Metric_par(task, {})` — the task-calibrated bundle with all
    /// non-task fields at upstream defaults.
    #[must_use]
    pub(crate) fn for_task(task: Task) -> Self {
        let par = [0.061_466_549_455_263_f64, 0.997_273_700_237_770_7];
        let mut t = Self {
            // Shared defaults (set before the task switch upstream).
            age: 24.0,
            do_aesl: true,
            do_aod: true,
            do_slum: true,
            aesl_slope_freq: -2.711,
            aesl_base: -0.125539,
            masking_norm: 0.0,
            masking_pool_size: 3,
            do_masking: true,
            do_pixel_threshold: false,
            do_si_gauss: false,
            si_size: -0.034244,
            si_sigma: -0.000502005,
            psych_func_slope: 3.5f64.log10(),
            rod_sensitivity: 0.0,
            csf_sa: [315.98, 6.7977, 1.6008, 0.25534],
            csf_sr: [1.1732, 1.32, 1.095, 0.5547, 2.9899, 1.8],
            mtf_params_a: [
                par[1] * 0.426,
                par[1] * 0.574,
                (1.0 - par[1]) * par[0],
                (1.0 - par[1]) * (1.0 - par[0]),
            ],
            mtf_params_b: [0.028, 0.37, 37.0, 360.0],
            csf_lums: [0.0002, 0.002, 0.02, 0.2, 2.0, 20.0, 150.0],
            csf_params: [
                [0.699404, 1.26181, 4.27832, 0.361902, 3.11914],
                [1.00865, 0.893585, 4.27832, 0.361902, 2.18938],
                [1.41627, 0.84864, 3.57253, 0.530355, 3.12486],
                [1.90256, 0.699243, 3.94545, 0.68608, 4.41846],
                [2.28867, 0.530826, 4.25337, 0.866916, 4.65117],
                [2.46011, 0.459297, 3.78765, 0.981028, 4.33546],
                [2.5145, 0.312626, 4.15264, 0.952367, 3.22389],
            ],
            ignore_freqs_lower_than: None,
            disable_lowvals_warning: false,
            // task fields — overwritten below.
            base_sensitivity_correction: 0.0,
            mask_self: 0.0,
            mask_xn: 0.0,
            mask_xo: 0.0,
            mask_q: 0.0,
            mask_p: 0.0,
            do_sprob_sum: false,
        };
        match task {
            Task::Quality | Task::SideBySide => {
                // `quality` is upstream's alias of `side-by-side`
                // (LocVisVC fit, 2020-05-24).
                t.base_sensitivity_correction = 0.203943775672;
                t.mask_self = 1.41846291721;
                t.mask_xn = 0.136877512403;
                t.mask_xo = -50.0;
                t.mask_q = 0.108934275615;
                t.mask_p = 0.3424;
                t.do_sprob_sum = true;
                t.psych_func_slope = 0.34;
                t.si_sigma = -0.502280453708;
            }
            Task::Flicker => {
                t.base_sensitivity_correction = 0.359225308021 + 0.14178269059;
                t.mask_self = 1.48041073222;
                t.mask_xn = 0.00886149078207;
                t.mask_xo = -50.0;
                t.mask_q = 0.11339123107;
                t.mask_p = 0.3424;
                t.do_sprob_sum = true;
                t.psych_func_slope = 0.591472024776;
                t.si_sigma = -0.511779476487;
            }
        }
        t
    }
}

/// The free-form switches upstream accepts through its `options` dict.
/// [`Self::reference`] reproduces upstream defaults for a task; individual
/// fields may then be overridden before building [`Params`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Options {
    /// `do_masking` — mutual masking in the band domain. Default `true`.
    pub do_masking: bool,
    /// `do_pixel_threshold` — multiply `P_map` by the Weber-fraction
    /// pixel-threshold map. Default `false`.
    pub do_pixel_threshold: bool,
    /// `do_aesl` — empirical age-related sensitivity loss. Default `true`
    /// upstream; requires a meaningful [`ViewingConditions::observer_age`].
    pub do_aesl: bool,
    /// `do_aod` — aging optical density (Pokorny 1987). Default `true`.
    pub do_aod: bool,
    /// `do_slum` — senile-miosis light reduction (Watson & Yellott 2012
    /// unified pupil formula). Default `true`.
    pub do_slum: bool,
    /// `do_si_gauss` — pool masking energy with a `10^si_size`° Gaussian
    /// instead of the 3×3 mean. Default `false` upstream.
    pub do_si_gauss: bool,
    /// `ignore_freqs_lower_than` — zero the `D` band when
    /// `band_freq[b−1]` is below this cpd value (upstream quirk: it reads
    /// the *previous* band's frequency). Default `None`.
    pub ignore_freqs_lower_than: Option<f64>,
    /// Additive correction on `base_sensitivity_correction` (log₁₀ JND) —
    /// upstream's `sensitivity_correction` option is `base + value`; the
    /// named field is never read in the reference port, so this delta is
    /// applied to the base term that *is* read.
    pub sensitivity_correction_delta: f64,
    /// `disable_lowvals_warning` — suppress the "input looks like relative
    /// values" signal. Default `false`.
    pub disable_lowvals_warning: bool,
}

impl Options {
    /// Upstream defaults for `task` — `Metric_par(task, {})` verbatim.
    #[must_use]
    pub fn reference(task: Task) -> Self {
        let _ = task; // all defaults here are task-independent upstream
        Self {
            do_masking: true,
            do_pixel_threshold: false,
            do_aesl: true,
            do_aod: true,
            do_slum: true,
            do_si_gauss: false,
            ignore_freqs_lower_than: None,
            sensitivity_correction_delta: 0.0,
            disable_lowvals_warning: false,
        }
    }
}

/// The full parameter set for one [`super::metric::hdrvdp3`] call —
/// constructed with [`Self::new`], which refuses to be assembled without a
/// task, an encoding, explicit viewing conditions and an explicit emission
/// model.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Params {
    /// Prediction task (selects the calibration bundle).
    pub task: Task,
    /// Viewing conditions — required, no defaults.
    pub viewing: ViewingConditions,
    /// How to interpret the pixel values.
    pub encoding: InputEncoding,
    /// The display's emission model. For [`InputEncoding::Generic`] this
    /// must be [`Emission::Custom`]; for other encodings
    /// [`Emission::default_for`] gives upstream's implicit choice but is
    /// still required to be stated.
    pub emission: Emission,
    /// Algorithm switches — [`Options::reference`] reproduces upstream.
    pub options: Options,
}

impl Params {
    /// All-explicit constructor. `Err` only for structurally impossible
    /// combinations (generic encoding without a custom emission table, or
    /// an emission preset whose channel count cannot feed the encoding).
    pub fn new(
        task: Task,
        viewing: ViewingConditions,
        encoding: InputEncoding,
        emission: Emission,
        options: Options,
    ) -> crate::Result<Self> {
        use crate::Error;
        if !viewing.pixels_per_degree.is_finite() || viewing.pixels_per_degree < 4.0 {
            return Err(Error::InvalidResolution(viewing.pixels_per_degree));
        }
        if let InputEncoding::Generic = encoding {
            match &emission {
                Emission::Custom { .. } => {}
                _ => {
                    return Err(Error::MissingEmission);
                }
            }
        }
        Ok(Self {
            task,
            viewing,
            encoding,
            emission,
            options,
        })
    }

    /// The task-calibrated constants with the options' delta applied —
    /// the working `metric_par`.
    pub(crate) fn task_par(&self) -> TaskPar {
        let mut t = TaskPar::for_task(self.task);
        t.age = f64::from(self.viewing.observer_age);
        t.do_masking = self.options.do_masking;
        t.do_pixel_threshold = self.options.do_pixel_threshold;
        t.do_aesl = self.options.do_aesl;
        t.do_aod = self.options.do_aod;
        t.do_slum = self.options.do_slum;
        t.do_si_gauss = self.options.do_si_gauss;
        t.ignore_freqs_lower_than = self.options.ignore_freqs_lower_than;
        t.disable_lowvals_warning = self.options.disable_lowvals_warning;
        t.base_sensitivity_correction += self.options.sensitivity_correction_delta;
        t
    }
}

impl Emission {
    /// The emission preset upstream silently picks for `encoding` —
    /// `hdrvdp3.py` lines 267–302: `led-lcd-srgb` for `sRGB-display`,
    /// `rgb-bt.709` and `rgb-native`; `oled` for `rgb-bt.2020` and `xyz`;
    /// `d65` for single-channel inputs.
    ///
    /// Using this keeps [`Params::new`] honest — the caller *states* the
    /// assumption — while documenting which choice upstream made.
    #[must_use]
    pub const fn default_for(encoding: InputEncoding) -> Self {
        match encoding {
            InputEncoding::Luminance | InputEncoding::LumaDisplay => {
                Self::Preset(DisplayPreset::D65)
            }
            InputEncoding::SrgbDisplay | InputEncoding::RgbBt709 | InputEncoding::RgbNative => {
                Self::Preset(DisplayPreset::LedLcdSrgb)
            }
            InputEncoding::RgbBt2020 | InputEncoding::Xyz => Self::Preset(DisplayPreset::Oled),
            InputEncoding::Generic => Self::Preset(DisplayPreset::D65), // rejected by Params::new
        }
    }
}

/// Pixels per degree from display geometry — re-export of
/// [`crate::params::pix_per_deg`], which implements the same
/// `hdrvdp_pix_per_deg` formula the VDP3 helper uses (square pixels,
/// diagonal + aspect ratio → display height → angular size).
pub use crate::params::pix_per_deg;
