//! Spike: can `burn` train a RankNet-style perceptual-metric / picker MLP
//! with a CUSTOM pairwise-ranking + monotonicity loss, via autodiff?
//!
//! This is the TRAINING-framework counterpart to `burn-conv-spike`, which
//! answered a DIFFERENT question — burn/cubek *conv kernels* for GPU metrics
//! (verdict: ABANDONED, 4.32x slower than the hand-written separable stencil).
//! Training small MLPs is orthogonal to that result: here we test autodiff +
//! custom multi-term loss + optimizer *ergonomics*, not kernel throughput.
//!
//! What it proves (or disproves):
//!   1. burn 0.21 compiles + runs in a standalone crate on vanilla cubecl 0.10.
//!   2. zensim/zenpicker's RankNet (pairwise-logistic) loss + a monotonicity
//!      hinge express cleanly as a few tensor ops, and autodiff handles the
//!      gradients we currently HAND-ROLL in zensim-train-core.
//!   3. The optimizer/train-loop loop is no harder than the hand-rolled Adam.
//!
//! Backend: ndarray + Autodiff (CPU). burn is generic over `Backend`, so the
//! same `Mlp<B>` swaps to the cubecl CUDA backend by changing the `B` alias.
//!
//! Task: synthetic monotone ground-truth `y = sigmoid(w·x)`; the net must learn
//! to RANK samples by `y` from sampled ordered pairs. Success = pair-ranking
//! accuracy climbing from ~0.5 (random) toward ~1.0.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

/// Autodiff-wrapped CPU backend. One line to swap to `Cuda`/`Wgpu`.
type B = burn::backend::Autodiff<burn::backend::NdArray>;

const D_IN: usize = 8;
const D_HID: usize = 16;
const N: usize = 512; // training samples
const N_PAIRS: usize = 2000; // sampled ordered pairs per epoch (fixed set here)
const EPOCHS: usize = 400;
const LR: f64 = 5e-3;
const LAMBDA_MONO: f64 = 0.5; // weight on the monotonicity hinge term
const MARGIN: f64 = 0.0;

// ---------------------------------------------------------------------------
// Model: a 2-layer MLP, the shape zensim/zenpicker actually ship.
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

    /// `[n, D_IN] -> [n, 1]` raw scores (kept 2-D to avoid rank gymnastics).
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.act.forward(self.l1.forward(x));
        self.l2.forward(h)
    }
}

// ---------------------------------------------------------------------------
// Deterministic synthetic data (no rand dep; tiny LCG for reproducibility).
// ---------------------------------------------------------------------------
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        // Numerical Recipes LCG; take top bits for a uniform in [0,1).
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// Returns (features [N*D_IN], true_quality [N]).
fn synth() -> (Vec<f32>, Vec<f32>) {
    let mut rng = Lcg(0x9E37_79B9_7F4A_7C15);
    // Fixed "true" weight vector the net must recover the ranking of.
    let w: [f32; D_IN] = [1.3, -0.8, 0.5, 2.0, -1.1, 0.3, 0.9, -0.4];
    let mut feats = Vec::with_capacity(N * D_IN);
    let mut y = Vec::with_capacity(N);
    for _ in 0..N {
        let mut dot = 0.0f32;
        for &wj in w.iter() {
            let xj = rng.next_f32() * 2.0 - 1.0; // [-1, 1]
            feats.push(xj);
            dot += wj * xj;
        }
        y.push(sigmoid(dot)); // monotone scalar quality in (0,1)
    }
    (feats, y)
}

/// Sample N_PAIRS ordered pairs (i, j) with true_quality[i] > true_quality[j].
fn sample_pairs(y: &[f32]) -> (Vec<i64>, Vec<i64>) {
    let mut rng = Lcg(0xD1B5_4A32_D192_ED03);
    let mut hi = Vec::with_capacity(N_PAIRS);
    let mut lo = Vec::with_capacity(N_PAIRS);
    let mut made = 0;
    while made < N_PAIRS {
        let a = (rng.next_f32() * N as f32) as usize % N;
        let b = (rng.next_f32() * N as f32) as usize % N;
        if (y[a] - y[b]).abs() < 1e-4 {
            continue; // skip near-ties
        }
        let (h, l) = if y[a] > y[b] { (a, b) } else { (b, a) };
        hi.push(h as i64);
        lo.push(l as i64);
        made += 1;
    }
    (hi, lo)
}

fn ranking_accuracy(scores: &[f32], hi: &[i64], lo: &[i64]) -> f32 {
    let mut correct = 0usize;
    for k in 0..hi.len() {
        if scores[hi[k] as usize] > scores[lo[k] as usize] {
            correct += 1;
        }
    }
    correct as f32 / hi.len() as f32
}

fn main() {
    println!("=== burn-ranknet-spike: train a RankNet MLP via burn autodiff ===");
    println!("backend: Autodiff<NdArray> (CPU)   model: {D_IN}->{D_HID}->1");
    println!("samples: {N}   pairs: {N_PAIRS}   epochs: {EPOCHS}   lr: {LR}");
    println!("loss: RankNet pairwise-logistic + {LAMBDA_MONO} * monotonicity hinge");
    println!();

    // Device type is inferred as `<B as Backend>::Device` from the `from_data`
    // / `Mlp::new` call sites below (NdArrayDevice for this backend).
    let device = Default::default();

    let (feats, y) = synth();
    let (hi, lo) = sample_pairs(&y);

    let x = Tensor::<B, 2>::from_data(TensorData::new(feats, [N, D_IN]), &device);
    let hi_idx = Tensor::<B, 1, Int>::from_data(TensorData::new(hi.clone(), [N_PAIRS]), &device);
    let lo_idx = Tensor::<B, 1, Int>::from_data(TensorData::new(lo.clone(), [N_PAIRS]), &device);

    let mut model = Mlp::<B>::new(&device);
    let mut optim = AdamConfig::new().init();

    for epoch in 0..=EPOCHS {
        // Forward: [N, 1] scores.
        let scores = model.forward(x.clone());

        // Gather paired scores: [N_PAIRS, 1] each.
        let s_hi = scores.clone().select(0, hi_idx.clone());
        let s_lo = scores.clone().select(0, lo_idx.clone());
        let diff = s_hi - s_lo; // want > 0

        // RankNet pairwise-logistic loss = softplus(-diff), numerically stable:
        //   softplus(z) = relu(z) + log1p(exp(-|z|)),  with z = -diff.
        let z = diff.clone().mul_scalar(-1.0);
        let softplus = z.clone().clamp_min(0.0)
            + z.abs().mul_scalar(-1.0).exp().add_scalar(1.0).log();
        let loss_rank = softplus.mean();

        // Monotonicity hinge: penalize diff <= margin, i.e. relu(margin - diff).
        let loss_mono = diff
            .mul_scalar(-1.0)
            .add_scalar(MARGIN)
            .clamp_min(0.0)
            .mean();

        let loss = loss_rank.clone() + loss_mono.clone().mul_scalar(LAMBDA_MONO);

        // Backward + Adam step — the whole hand-rolled adam.rs replaced by this.
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(LR, model, grads);

        if epoch % 50 == 0 || epoch == EPOCHS {
            let sv = scores.to_data().to_vec::<f32>().unwrap();
            let acc = ranking_accuracy(&sv, &hi, &lo);
            let l = loss.to_data().to_vec::<f32>().unwrap()[0];
            let lr_ = loss_rank.to_data().to_vec::<f32>().unwrap()[0];
            let lm = loss_mono.to_data().to_vec::<f32>().unwrap()[0];
            println!(
                "epoch {epoch:4}  loss {l:.4} (rank {lr_:.4}, mono {lm:.4})  pair-acc {acc:.4}"
            );
        }
    }

    println!();
    let sv = model.forward(x.clone()).to_data().to_vec::<f32>().unwrap();
    let final_acc = ranking_accuracy(&sv, &hi, &lo);
    println!("=== Verdict ===");
    println!("final pair-ranking accuracy: {final_acc:.4}");
    if final_acc > 0.95 {
        println!(
            "PASS: burn autodiff trained a RankNet MLP with a custom 2-term loss to >0.95"
        );
        println!("ranking accuracy. The loss that zensim-train-core HAND-ROLLS (pairwise");
        println!("logistic + monotonicity) is ~15 lines of tensor ops here; gradients are");
        println!("autodiff'd, not manually derived. Training ergonomics confirmed viable.");
    } else if final_acc > 0.8 {
        println!("MARGINAL: learned a partial ranking ({final_acc:.3}). Ergonomics work;");
        println!("tune lr/epochs. Still confirms the autodiff + custom-loss path.");
    } else {
        println!("FAIL: ranking did not converge ({final_acc:.3}). Investigate the loss");
        println!("wiring or optimizer config before trusting the ergonomics conclusion.");
    }
}
