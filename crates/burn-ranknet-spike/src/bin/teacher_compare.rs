//! gbdt-teacher-compare: is a tree-ensemble TEACHER worth distilling into the
//! shippable GD-MLP student?
//!
//! On interaction-heavy synthetic data (the regime where GBDT tends to beat a
//! small MLP), compares three rankers by HELD-OUT pair-ranking accuracy against
//! the clean ground-truth ordering:
//!
//!   A. GD-MLP student trained DIRECTLY    — burn RankNet on true-label pairs
//!   B. GBDT teacher                       — forust, squared-loss regression
//!   C. GD-MLP student DISTILLED           — burn RankNet on pairs ordered by the
//!                                           TEACHER's predictions (never sees
//!                                           the true labels)
//!
//! Read it as: if B > A, the tree ensemble has a tabular/interaction edge the
//! small MLP misses; if C closes the gap toward B, distillation transfers that
//! edge into the tiny model we can actually ship via zenpredict. Both students
//! are the SAME architecture (10->16->1) — the only difference is what they learn
//! to rank by.
//!
//! THIS IS A METHODOLOGY DEMO ON SYNTHETIC DATA. It is NOT a claim about the real
//! picker ceiling — that needs real labeled sweep data. It exists to make the
//! "gbdt vs gradient-descent teacher" question concrete with runnable numbers.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

use forust_ml::objective::ObjectiveType;
use forust_ml::{GradientBooster, Matrix};

type B = burn::backend::Autodiff<burn::backend::NdArray>;

const D_IN: usize = 10;
const D_HID: usize = 16;
const N: usize = 1500;
const N_TRAIN: usize = 1000;
const N_TEST: usize = N - N_TRAIN;
const TRAIN_PAIRS: usize = 4000;
const TEST_PAIRS: usize = 3000;
const EPOCHS: usize = 600;
const LR: f64 = 5e-3;

// ---------------------------------------------------------------------------
#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    act: Relu,
}
impl<B: Backend> Mlp<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            l1: LinearConfig::new(D_IN, D_HID).init(device),
            l2: LinearConfig::new(D_HID, 1).init(device),
            act: Relu::new(),
        }
    }
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.l2.forward(self.act.forward(self.l1.forward(x)))
    }
}

// ---------------------------------------------------------------------------
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn unit(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0 // [-1, 1]
    }
    /// approx-normal noise via central limit (sum of 4 uniforms).
    fn noise(&mut self) -> f32 {
        (self.unit() + self.unit() + self.unit() + self.unit()) * 0.5
    }
}

fn step(v: f32) -> f32 {
    if v > 0.0 { 1.0 } else { -1.0 }
}

/// Interaction-heavy target: linear terms + a multiplicative interaction + a
/// sign-XOR (axis-aligned, tree-friendly), with feats 7..10 pure noise.
fn target(x: &[f32]) -> f32 {
    let lin = 1.5 * x[0] - 1.0 * x[1] + 0.5 * x[6];
    let mult = 3.0 * x[2] * x[3];
    let xorish = 2.0 * step(x[4]) * step(x[5]);
    lin + mult + xorish
}

/// feats (row-major [N*D_IN]), clean targets, noisy training targets.
fn synth() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = Lcg(0xABCD_1234_5678_9F01);
    let mut feats = Vec::with_capacity(N * D_IN);
    let mut y_clean = Vec::with_capacity(N);
    let mut y_noisy = Vec::with_capacity(N);
    for _ in 0..N {
        let mut row = [0f32; D_IN];
        for r in row.iter_mut() {
            *r = rng.unit();
        }
        feats.extend_from_slice(&row);
        let yc = target(&row);
        y_clean.push(yc);
        y_noisy.push(yc + 0.6 * rng.noise());
    }
    (feats, y_clean, y_noisy)
}

/// Sample ordered pairs (hi, lo) within `range`, ordered by `score` (hi = bigger).
/// Returned indices are GLOBAL (into the full feats/score arrays).
fn pairs_by(score: &[f32], range: std::ops::Range<usize>, n_pairs: usize, seed: u64) -> (Vec<i64>, Vec<i64>) {
    let mut rng = Lcg(seed);
    let span = range.len();
    let base = range.start;
    let (mut hi, mut lo) = (Vec::with_capacity(n_pairs), Vec::with_capacity(n_pairs));
    let mut made = 0;
    while made < n_pairs {
        let a = base + (rng.next_f32() * span as f32) as usize % span;
        let b = base + (rng.next_f32() * span as f32) as usize % span;
        if (score[a] - score[b]).abs() < 1e-5 {
            continue;
        }
        let (h, l) = if score[a] > score[b] { (a, b) } else { (b, a) };
        hi.push(h as i64);
        lo.push(l as i64);
        made += 1;
    }
    (hi, lo)
}

/// Train a RankNet MLP on `feats` using the given (global-indexed) ordered pairs.
/// Self-creates the (inferred) backend device — see main.rs for the same pattern;
/// naming `B::Device` is ambiguous since the autodiff backend satisfies two traits.
fn train_ranknet(feats: &[f32], hi: &[i64], lo: &[i64]) -> Mlp<B> {
    let device = Default::default();
    // NOTE: burn weight-init is unseeded here, so the student numbers wobble ~±0.003
    // run-to-run. The teacher edge is small vs that variance — use multi-seed
    // averaging (workspace 5-seed-CI protocol) for a real recovered-% on real data.
    let x = Tensor::<B, 2>::from_data(TensorData::new(feats.to_vec(), [N, D_IN]), &device);
    let hi_idx = Tensor::<B, 1, Int>::from_data(TensorData::new(hi.to_vec(), [hi.len()]), &device);
    let lo_idx = Tensor::<B, 1, Int>::from_data(TensorData::new(lo.to_vec(), [lo.len()]), &device);

    let mut model = Mlp::<B>::new(&device);
    let mut optim = AdamConfig::new().init();
    for _ in 0..EPOCHS {
        let s = model.forward(x.clone());
        let s_hi = s.clone().select(0, hi_idx.clone());
        let s_lo = s.clone().select(0, lo_idx.clone());
        let diff = s_hi - s_lo;
        // RankNet pairwise-logistic loss = softplus(-diff), numerically stable.
        let z = diff.mul_scalar(-1.0);
        let sp = z.clone().clamp_min(0.0) + z.abs().mul_scalar(-1.0).exp().add_scalar(1.0).log();
        let loss = sp.mean();
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(LR, model, grads);
    }
    model
}

fn mlp_scores(model: &Mlp<B>, feats: &[f32]) -> Vec<f32> {
    let device = Default::default();
    let x = Tensor::<B, 2>::from_data(TensorData::new(feats.to_vec(), [N, D_IN]), &device);
    model.forward(x).to_data().to_vec::<f32>().unwrap()
}

/// Fraction of test pairs the `score` orders the same way as ground truth.
fn pair_acc(score: &[f32], hi: &[i64], lo: &[i64]) -> f32 {
    let mut ok = 0usize;
    for k in 0..hi.len() {
        if score[hi[k] as usize] > score[lo[k] as usize] {
            ok += 1;
        }
    }
    ok as f32 / hi.len() as f32
}

fn main() {
    println!("=== gbdt-teacher-compare: GBDT teacher vs GD-MLP student (+distillation) ===");
    println!("data: {N} samples, {D_IN} feats (3 pure noise), interaction-heavy target + label noise");
    println!("split: {N_TRAIN} train / {N_TEST} test   students: {D_IN}->{D_HID}->1 RankNet, {EPOCHS} ep");
    println!("teacher: forust GBDT (SquaredLoss, 100 trees, depth 5)");
    println!();

    let (feats, y_clean, y_noisy) = synth();

    // --- B. GBDT teacher (forust): regress the noisy target on train ---
    // forust wants COLUMN-major data.
    let col_major = |range: std::ops::Range<usize>| -> Vec<f64> {
        let n = range.len();
        let mut v = Vec::with_capacity(n * D_IN);
        for c in 0..D_IN {
            for r in range.clone() {
                v.push(feats[r * D_IN + c] as f64);
            }
        }
        v
    };
    let train_cm = col_major(0..N_TRAIN);
    let all_cm = col_major(0..N);
    let m_train = Matrix::new(&train_cm, N_TRAIN, D_IN);
    let m_all = Matrix::new(&all_cm, N, D_IN);
    let y_train_f64: Vec<f64> = y_noisy[0..N_TRAIN].iter().map(|&v| v as f64).collect();

    let mut teacher = GradientBooster::default()
        .set_objective_type(ObjectiveType::SquaredLoss)
        .set_iterations(100)
        .set_learning_rate(0.3)
        .set_max_depth(5)
        .set_seed(1);
    teacher.fit_unweighted(&m_train, &y_train_f64, None).expect("forust fit");
    // Measure the serialized teacher. forust serializes to JSON only (no binary
    // format) — this is the "how big is the GBDT model file" answer.
    teacher.save_booster("/tmp/gbdt_teacher_model.json").expect("save booster");
    let model_json_bytes = teacher.json_dump().expect("json dump").len();
    println!(
        "GBDT teacher model file: 100 trees x depth 5 -> JSON {} bytes ({:.1} KB), at /tmp/gbdt_teacher_model.json",
        model_json_bytes,
        model_json_bytes as f64 / 1024.0
    );
    println!();
    let teacher_all: Vec<f32> = teacher.predict(&m_all, true).iter().map(|&v| v as f32).collect();

    // --- held-out test pairs, ordered by CLEAN ground truth ---
    let (test_hi, test_lo) = pairs_by(&y_clean, N_TRAIN..N, TEST_PAIRS, 0xFEED);

    // --- A. student trained DIRECTLY on noisy-label order ---
    let (dir_hi, dir_lo) = pairs_by(&y_noisy, 0..N_TRAIN, TRAIN_PAIRS, 0x11);
    let student_direct = train_ranknet(&feats, &dir_hi, &dir_lo);
    let acc_direct = pair_acc(&mlp_scores(&student_direct, &feats), &test_hi, &test_lo);

    // --- C. student DISTILLED from teacher's train-set predictions ---
    let (dis_hi, dis_lo) = pairs_by(&teacher_all, 0..N_TRAIN, TRAIN_PAIRS, 0x22);
    let student_distilled = train_ranknet(&feats, &dis_hi, &dis_lo);
    let acc_distilled = pair_acc(&mlp_scores(&student_distilled, &feats), &test_hi, &test_lo);

    // --- B. teacher's own held-out ranking ---
    let acc_teacher = pair_acc(&teacher_all, &test_hi, &test_lo);

    println!("=== held-out test pair-ranking accuracy (vs clean ground truth) ===");
    println!("  A. GD-MLP student, direct       : {acc_direct:.4}");
    println!("  B. GBDT teacher                 : {acc_teacher:.4}");
    println!("  C. GD-MLP student, distilled    : {acc_distilled:.4}");
    println!();
    let gap_ba = acc_teacher - acc_direct;
    let recovered = if gap_ba.abs() > 1e-6 {
        (acc_distilled - acc_direct) / gap_ba * 100.0
    } else {
        0.0
    };
    println!("teacher edge over direct student (B-A): {gap_ba:+.4}");
    println!("distillation recovered {recovered:.0}% of that edge (C vs A, toward B)");
    println!();
    println!("=== Reading ===");
    if gap_ba > 0.01 {
        println!("GBDT teacher beats the same-size MLP trained directly — the tree ensemble");
        println!("captures interaction structure the small MLP misses on this data.");
        if acc_distilled > acc_direct + 0.005 {
            println!("Distillation transfers part of that edge into the SAME tiny MLP (which is");
            println!("what ships via zenpredict) — the teacher earns its keep as a soft-label");
            println!("source. On real picker data, run this A/B to size the real gap.");
        } else {
            println!("But distillation did NOT help the student here — the MLP's capacity, not");
            println!("its training signal, is the bottleneck. A teacher won't fix that.");
        }
    } else {
        println!("On THIS data the small MLP already matches the GBDT — no teacher needed here.");
        println!("That's an honest negative: distillation pays off only when B-A is real. The");
        println!("real test is the same comparison on actual labeled picker/metric data.");
    }
    println!();
    println!("NOTE: synthetic methodology demo, not a real-picker ceiling. See README.");
}
