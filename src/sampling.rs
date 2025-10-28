// src/sampling.rs

//! Sampling strategies for text generation

use burn::tensor::{activation, backend::Backend, Int, Tensor};
use log::debug;

// ═════════════════════════════════════════════════════════════════════════════
// Temperature scaling
// ═════════════════════════════════════════════════════════════════════════════

pub fn apply_temperature<B: Backend>(logits: Tensor<B, 2>, temperature: f64) -> Tensor<B, 2> {
    if temperature == 1.0 {
        return logits;
    }
    
    assert!(temperature > 0.0, "Temperature must be positive, got {}", temperature);
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
    let mask = logits.clone().greater_equal(kth_val.clone().expand([batch, vocab]));
    
    logits.mask_fill(mask.bool_not(), f64::NEG_INFINITY)
}

// ═════════════════════════════════════════════════════════════════════════════
// Sampling functions
// ═════════════════════════════════════════════════════════════════════════════

/// Greedy sampling: argmax on last dimension
/// Input: [B, V], Output: [B] as Int
pub fn sample_greedy<B: Backend>(logits: Tensor<B, 2>) -> Tensor<B, 1, Int> {
    logits.argmax(1).squeeze(1)
}

/// Sample with temperature and top-k
pub fn sample_with_temperature_topk<B: Backend>(
    logits: Tensor<B, 2>,
    temperature: f64,
    top_k: Option<usize>,
) -> Tensor<B, 1, Int> {
    let mut logits = apply_temperature(logits, temperature);
    
    if let Some(k) = top_k {
        logits = top_k_filter(logits, k);
    }
    
    let probs = activation::softmax(logits, 1);
    probs.argmax(1).squeeze(1)
}

/// Main entry point for sampling
pub fn sample_next_token<B: Backend>(
    logits: Tensor<B, 2>,
    temperature: f64,
    top_k: Option<usize>,
) -> Tensor<B, 1, Int> {
    if temperature == 0.0 {
        sample_greedy(logits)
    } else {
        sample_with_temperature_topk(logits, temperature, top_k)
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Utility functions
// ═════════════════════════════════════════════════════════════════════════════

pub fn extract_last_logits<B: Backend>(logits: Tensor<B, 3>) -> Tensor<B, 2> {
    let [b, t, v] = logits.dims();
    logits.slice([0..b, (t - 1)..t, 0..v]).reshape([b, v])
}
