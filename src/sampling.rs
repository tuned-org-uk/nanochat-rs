// src/sampling.rs

//! Sampling strategies for text generation
//!
//! Returns token IDs as [B, 1] shaped tensors for consistent batch handling

use burn::tensor::{activation, backend::Backend, Bool, Int, Tensor};
use log::debug;

// ═════════════════════════════════════════════════════════════════════════════
// Temperature scaling
// ═════════════════════════════════════════════════════════════════════════════

pub fn apply_temperature<B: Backend>(logits: Tensor<B, 2>, temperature: f64) -> Tensor<B, 2> {
    if temperature == 1.0 {
        return logits;
    }
    assert!(
        temperature > 0.0,
        "Temperature must be positive, got {}",
        temperature
    );
    debug!("Applying temperature scaling: {}", temperature);
    logits / temperature
}

// ═════════════════════════════════════════════════════════════════════════════
// Top-k filtering
// ═════════════════════════════════════════════════════════════════════════════

pub fn top_k_filter<B: Backend>(logits: Tensor<B, 2>, k: usize) -> Tensor<B, 2> {
    let [batch, vocab] = logits.dims();

    if k == 0 || k >= vocab {
        return logits;
    }

    debug!("Applying top-k filter: k={}, vocab={}", k, vocab);

    let topk_vals = logits.clone().sort_with_indices(1).0;
    let kth_val = topk_vals.narrow(1, vocab - k, 1);
    let mask = logits.clone().greater_equal(kth_val.expand([batch, vocab]));

    logits.mask_fill(mask.bool_not(), f64::NEG_INFINITY)
}

// ═════════════════════════════════════════════════════════════════════════════
// Top-p (nucleus) filtering
// ═════════════════════════════════════════════════════════════════════════════

pub fn top_p_filter<B: Backend>(logits: Tensor<B, 2>, p: f64) -> Tensor<B, 2> {
    assert!(p > 0.0 && p <= 1.0, "top_p must be in (0, 1]");
    let [batch, vocab] = logits.dims();

    if p >= 0.9999 {
        return logits;
    }

    debug!("Applying top-p filter: p={}", p);

    let probs = activation::softmax(logits.clone(), 1);
    let probs_host: Vec<f32> = probs.to_data().to_vec().unwrap();
    let mut keep_mask: Vec<f32> = vec![0.0; batch * vocab];

    for b in 0..batch {
        let mut pairs: Vec<(f32, usize)> = (0..vocab)
            .map(|v| (probs_host[b * vocab + v], v))
            .collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let mut cum = 0.0f32;
        for (prob, idx) in pairs {
            if cum < p as f32 {
                keep_mask[b * vocab + idx] = 1.0;
                cum += prob;
            } else {
                break;
            }
        }
    }

    let keep_mask_tensor =
        Tensor::<B, 2>::from_floats(keep_mask.as_slice(), &probs.device()).reshape([batch, vocab]);
    
    // Create threshold tensor - pass ownership via as_slice()
    let threshold_data = vec![0.5; batch * vocab];
    let threshold = Tensor::<B, 2>::from_floats(threshold_data.as_slice(), &probs.device())
        .reshape([batch, vocab]);
    
    let keep_bool: Tensor<B, 2, Bool> = keep_mask_tensor.greater(threshold);

    logits.mask_fill(keep_bool.bool_not(), f64::NEG_INFINITY)
}

// ═════════════════════════════════════════════════════════════════════════════
// Sampling functions - All return [B, 1] Int tensors
// ═════════════════════════════════════════════════════════════════════════════

/// Greedy sampling: argmax on vocab dimension
/// Input: [B, V], Output: [B, 1] Int
pub fn sample_greedy<B: Backend>(logits: Tensor<B, 2>) -> Tensor<B, 2, Int> {
    let [b, _v] = logits.dims();
    // argmax(1) keeps dim by default in Burn, resulting in [B, 1]
    let indices = logits.argmax(1);
    debug!("Greedy sample output shape: {:?}", indices.dims());
    indices
}

/// Sample with policy - unified interface
/// Input: [B, V], Output: [B, 1] Int
pub fn sample_with_policy<B: Backend>(
    logits_last: Tensor<B, 2>,
    policy: SamplingPolicy,
) -> Tensor<B, 2, Int> {
    use SamplingPolicy::*;
    debug!("Sampling with policy: {:?}", policy);

    match policy {
        Greedy => sample_greedy(logits_last),
        Temperature { t } => {
            let logits = apply_temperature(logits_last, t);
            let probs = activation::softmax(logits, 1);
            probs.argmax(1)
        }
        TopK { k } => {
            let logits = top_k_filter(logits_last, k);
            let probs = activation::softmax(logits, 1);
            probs.argmax(1)
        }
        TopP { p } => {
            let logits = top_p_filter(logits_last, p);
            let probs = activation::softmax(logits, 1);
            probs.argmax(1)
        }
        TempTopK { t, k } => {
            let logits = apply_temperature(logits_last, t);
            let logits = top_k_filter(logits, k);
            let probs = activation::softmax(logits, 1);
            probs.argmax(1)
        }
        TempTopP { t, p } => {
            let logits = apply_temperature(logits_last, t);
            let logits = top_p_filter(logits, p);
            let probs = activation::softmax(logits, 1);
            probs.argmax(1)
        }
        TempTopKTopP { t, k, p } => {
            let logits = apply_temperature(logits_last, t);
            let logits = top_k_filter(logits, k);
            let logits = top_p_filter(logits, p);
            let probs = activation::softmax(logits, 1);
            probs.argmax(1)
        }
    }
}

/// Sample with temperature and optional top-k
/// Input: [B, V], Output: [B, 1] Int
pub fn sample_with_temperature_topk<B: Backend>(
    logits: Tensor<B, 2>,
    temperature: f64,
    top_k: Option<usize>,
) -> Tensor<B, 2, Int> {
    let mut logits = apply_temperature(logits, temperature);
    if let Some(k) = top_k {
        logits = top_k_filter(logits, k);
    }
    let probs = activation::softmax(logits, 1);
    probs.argmax(1)
}

/// Main entry point for sampling
/// Input: [B, V], Output: [B, 1] Int
pub fn sample_next_token<B: Backend>(
    logits: Tensor<B, 2>,
    temperature: f64,
    top_k: Option<usize>,
) -> Tensor<B, 2, Int> {
    if temperature == 0.0 {
        sample_greedy(logits)
    } else {
        sample_with_temperature_topk(logits, temperature, top_k)
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Utility functions
// ═════════════════════════════════════════════════════════════════════════════

/// Extract last timestep logits from [B, T, V] -> [B, V]
pub fn extract_last_logits<B: Backend>(logits: Tensor<B, 3>) -> Tensor<B, 2> {
    let [b, t, v] = logits.dims();
    logits.slice([0..b, (t - 1)..t, 0..v]).reshape([b, v])
}

// ═════════════════════════════════════════════════════════════════════════════
// Sampling policy enum
// ═════════════════════════════════════════════════════════════════════════════

/// Sampling policy to inject into the engine
#[derive(Clone, Copy, Debug)]
pub enum SamplingPolicy {
    Greedy,
    Temperature { t: f64 },
    TopK { k: usize },
    TopP { p: f64 },
    TempTopK { t: f64, k: usize },
    TempTopP { t: f64, p: f64 },
    TempTopKTopP { t: f64, k: usize, p: f64 },
}
