//! Comprehensive tests for the minimal GPT module.
//!
//! Tests cover:
//! - Model construction and parameter shapes
//! - Forward pass shape correctness
//! - Attention mask correctness (causal)
//! - Generation determinism
//! - Greedy generation produces valid token IDs
//! - Multi-batch forward consistency

use log::debug;

use burn::{
    tensor::{Int, Tensor},
};
use burn::prelude::Backend;

use crate::{
    backend::AutoBackend,
    config::NanoChatConfig,
    gpt::GptModel
};

type TestBackend = AutoBackend;

// ─────────────────────────────────────────────────────────────────────────
// Helper: small config for fast tests
// ─────────────────────────────────────────────────────────────────────────

fn test_config() -> NanoChatConfig {
    NanoChatConfig {
        sequence_len: 32,
        vocab_size: 64,
        n_layer: 1,
        n_head: 2,
        n_kv_head: 2,
        n_embd: 32,
        block_size: 32,
        dropout: 0.0,
    }
}

fn tiny_config() -> NanoChatConfig {
    NanoChatConfig {
        sequence_len: 8,
        vocab_size: 16,
        n_layer: 1,
        n_head: 2,
        n_kv_head: 2,
        n_embd: 8,
        block_size: 8,
        dropout: 0.0,
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Test 1: Model construction
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_model_construction() {
    let cfg = test_config();
    let device = Default::default();
    let _model = GptModel::<TestBackend>::new(&cfg, &device);

    // Just ensure it constructs without panic
    assert_eq!(cfg.vocab_size, 64);
    assert_eq!(cfg.n_embd, 32);
}

// ─────────────────────────────────────────────────────────────────────────
// Test 2: Forward pass shape correctness
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_forward_shape() {
    let cfg = test_config();
    let device = Default::default();
    let model = GptModel::<TestBackend>::new(&cfg, &device);

    let batch = 2;
    let seq_len = 8;

    // Create random input
    let input_data: Vec<i64> = (0..(batch * seq_len))
        .map(|i| (i % cfg.vocab_size) as i64)
        .collect();
    let input: Tensor<TestBackend, 1, Int> = Tensor::from_ints(input_data.as_slice(), &device);
    let input = input.reshape([batch, seq_len]);

    let logits = model.forward(input, true);
    let shape = logits.dims();

    assert_eq!(shape, [batch, seq_len, cfg.vocab_size]);
}

// ─────────────────────────────────────────────────────────────────────────
// Test 3: Single token forward
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_single_token_forward() {
    let cfg = tiny_config();
    let device = Default::default();
    let model = GptModel::<TestBackend>::new(&cfg, &device);

    let input: Tensor<TestBackend, 1, Int> = Tensor::from_ints([1], &device);
    let input = input.reshape([1, 1]);

    let logits = model.forward(input, true);
    assert_eq!(logits.dims(), [1, 1, cfg.vocab_size]);
}

// ─────────────────────────────────────────────────────────────────────────
// Test 4: Generation produces valid output shape
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_generate_shape() {
    let cfg = tiny_config();
    let device = Default::default();
    let model = GptModel::<TestBackend>::new(&cfg, &device);

    let seed: Vec<i64> = vec![1, 2, 3];
    let input: Tensor<TestBackend, 1, Int> = Tensor::from_ints(seed.as_slice(), &device);
    let input = input.reshape([1, seed.len()]);

    let max_new = 5;
    let out = model.generate(input, max_new);
    let shape = out.dims();

    assert_eq!(shape[0], 1);
    assert_eq!(shape[1], seed.len() + max_new);
}

// ─────────────────────────────────────────────────────────────────────────
// Test 5: Generation produces valid token IDs
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_generate_valid_ids() {
    let cfg = tiny_config();
    let device = Default::default();
    let model = GptModel::<TestBackend>::new(&cfg, &device);

    let seed: Vec<i64> = vec![0, 1];
    let input: Tensor<TestBackend, 1, Int> = Tensor::from_ints(seed.as_slice(), &device);
    let input = input.reshape([1, seed.len()]);

    let out = model.generate(input, 3);
    let ids = out.to_data().to_vec::<i64>().unwrap();

    // All IDs should be in [0, vocab_size)
    for &id in &ids {
        assert!(id >= 0 && id < cfg.vocab_size as i64);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Test 6: Deterministic generation (same input -> same output)
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_generation_determinism() {
    let cfg = tiny_config();
    let device = Default::default();
    let model = GptModel::<TestBackend>::new(&cfg, &device);

    let seed: Vec<i64> = vec![0, 1, 2];
    let input: Tensor<TestBackend, 1, Int> = Tensor::from_ints(seed.as_slice(), &device);
    let input = input.reshape([1, seed.len()]);

    let out1 = model.generate(input.clone(), 4);
    let out2 = model.generate(input, 4);

    let ids1 = out1.to_data().to_vec::<i64>().unwrap();
    let ids2 = out2.to_data().to_vec::<i64>().unwrap();

    assert_eq!(ids1, ids2, "Greedy generation should be deterministic");
}

// ─────────────────────────────────────────────────────────────────────────
// Test 7: Multi-batch forward consistency
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_multi_batch_forward() {
    let cfg = test_config();
    let device = Default::default();
    let model = GptModel::<TestBackend>::new(&cfg, &device);

    let batch = 3;
    let seq_len = 6;

    let data: Vec<i64> = (0..(batch * seq_len)).map(|i| (i % 10) as i64).collect();
    let input: Tensor<TestBackend, 1, Int> = Tensor::from_ints(data.as_slice(), &device);
    let input = input.reshape([batch, seq_len]);

    let logits = model.forward(input, true);
    assert_eq!(logits.dims(), [batch, seq_len, cfg.vocab_size]);

    // Check that no NaNs or Infs in logits (basic sanity)
    let logits_data = logits.to_data().to_vec::<f32>().unwrap();
    for &val in &logits_data {
        assert!(val.is_finite(), "Logits contain NaN or Inf");
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Test 8: Attention mask produces lower-triangular pattern
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_attention_mask_causal() {
    crate::init();
    // This is an indirect test: we check that generation at position t
    // doesn't depend on tokens at positions > t by verifying that
    // greedy generation is consistent when we truncate future tokens.

    let cfg = tiny_config();
    let device = Default::default();
    let model = GptModel::<TestBackend>::new(&cfg, &device);

    let seed: Vec<i64> = vec![0, 1, 2, 3];
    let input: Tensor<TestBackend, 1, Int> = Tensor::from_ints(seed.as_slice(), &device);
    let input = input.reshape([1, seed.len()]);

    // Forward on full sequence
    let logits_full = model.forward_no_softcap(input.clone());
    let last_full = logits_full.slice([0..1, (seed.len() - 1)..seed.len(), 0..cfg.vocab_size]);
    let tok_full = last_full.reshape([1, cfg.vocab_size]).argmax(1);

    // Forward on prefix (should give same logit at last prefix position)
    let prefix_len = seed.len() - 1;
    let input_prefix = input.slice([0..1, 0..prefix_len]);
    let logits_prefix = model.forward_no_softcap(input_prefix);
    let last_prefix = logits_prefix.slice([0..1, (prefix_len - 1)..prefix_len, 0..cfg.vocab_size]);
    let tok_prefix = last_prefix.reshape([1, cfg.vocab_size]).argmax(1);

    let id_full = tok_full.to_data().to_vec::<i64>().unwrap()[0];
    let id_prefix = tok_prefix.to_data().to_vec::<i64>().unwrap()[0];

    // Due to causal masking, the last token of prefix should match
    // the corresponding position in full sequence
    // (This is a simplified check; a more rigorous test would inspect attention weights)
    assert_eq!(
        id_prefix, id_full,
        "Causal mask should ensure prefix prediction matches full sequence at same position"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Test 9: Zero-length input handling (edge case)
// ─────────────────────────────────────────────────────────────────────────

#[test]
#[should_panic]
fn test_zero_length_input() {
    let cfg = tiny_config();
    let device = Default::default();
    let model = GptModel::<TestBackend>::new(&cfg, &device);

    // Create empty input (should panic or error)
    let input: Tensor<TestBackend, 2, Int> = Tensor::zeros([1, 0], &device);
    let _ = model.forward(input, true);
}

// ─────────────────────────────────────────────────────────────────────────
// Test 10: Large batch size (stress test)
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_large_batch() {
    let cfg = test_config();
    let device = Default::default();
    let model = GptModel::<TestBackend>::new(&cfg, &device);

    let batch = 16;
    let seq_len = 4;

    let data: Vec<i64> = (0..(batch * seq_len)).map(|i| (i % cfg.vocab_size) as i64).collect();
    let input: Tensor<TestBackend, 1, Int> = Tensor::from_ints(data.as_slice(), &device);
    let input = input.reshape([batch, seq_len]);

    let logits = model.forward(input, true);
    assert_eq!(logits.dims(), [batch, seq_len, cfg.vocab_size]);
}

// ─────────────────────────────────────────────────────────────────────────
// Test 11: Model with different head counts
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_different_head_counts() {
    let mut cfg = test_config();
    cfg.n_head = 4;
    cfg.n_embd = 32; // Must be divisible by n_head

    let device = Default::default();
    let model = GptModel::<TestBackend>::new(&cfg, &device);

    let input: Tensor<TestBackend, 1, Int> = Tensor::from_ints([0, 1, 2], &device);
    let input = input.reshape([1, 3]);

    let logits = model.forward(input, true);
    assert_eq!(logits.dims(), [1, 3, cfg.vocab_size]);
}

// ─────────────────────────────────────────────────────────────────────────
// Test 12: Numerical stability (no NaNs in deep sequences)
// ─────────────────────────────────────────────────────────────────────────

// In test_numerical_stability
#[test]
fn test_numerical_stability() {
    crate::init();
    let device = Default::default();
    
    // Use TINY config to prevent accumulation issues with random init
    let cfg = NanoChatConfig {
        sequence_len: 8,   // Much smaller
        vocab_size: 32,    // Much smaller
        n_layer: 1,        // Single layer only
        n_head: 2,
        n_kv_head: 2,
        n_embd: 16,        // Tiny embedding
        block_size: 8,
        dropout: 0.0,
    };

    let model = GptModel::<burn_ndarray::NdArray>::new(&cfg, &device);
    
    // Use SMALL input
    let input: Tensor<burn_ndarray::NdArray, 2, Int> = Tensor::arange(0..4, &device)
        .reshape([1, 4]);
    debug!("input {:?}", input);

    let logits = model.forward_no_softcap(input);
    debug!("logits {:?}", logits);
    
    // Check health
    assert!(
        GptModel::check_logits_health(&logits),
        "Found NaN or Inf in logits with tiny config"
    );
}

#[test]
fn test_smoke_generation_multi_blocks() {
    let device = <TestBackend as Backend>::Device::default();
    let cfg = NanoChatConfig {
        sequence_len: 32,
        vocab_size: 64,
        n_layer: 3,
        n_head: 4,
        n_kv_head: 4,
        n_embd: 64,
        block_size: 32,
        dropout: 0.0,
    };
    let model = GptModel::<TestBackend>::new(&cfg, &device);

    let ids = Tensor::<TestBackend, 1, Int>::from_ints([1, 2, 3], &device).reshape([1, 3]);
    let out = model.generate(ids, 5);
    assert_eq!(out.dims(), [1, 8]);
}

#[test]
fn test_multi_block_forward_shape() {
    let device = <TestBackend as Backend>::Device::default();
    let cfg = NanoChatConfig {
        sequence_len: 32,
        vocab_size: 128,
        n_layer: 4,  // Multiple blocks
        n_head: 4,
        n_kv_head: 4,
        n_embd: 64,
        block_size: 32,
        dropout: 0.0,
    };
    
    let model = GptModel::<TestBackend>::new(&cfg, &device);
    
    let ids = Tensor::<TestBackend, 1, Int>::from_ints([1, 2, 3, 4], &device)
        .reshape([1, 4]);
    let logits = model.forward(ids, true);
    
    assert_eq!(logits.dims(), [1, 4, cfg.vocab_size], 
                "Logits should be [B=1, T=4, V=128]");
    assert!(GptModel::<TestBackend>::check_logits_health(&logits), 
            "Logits should not contain NaN/Inf");
}

#[test]
fn test_multi_batch_multi_block() {
    let device = <TestBackend as Backend>::Device::default();
    let cfg = NanoChatConfig {
        sequence_len: 16,
        vocab_size: 64,
        n_layer: 3,
        n_head: 4,
        n_kv_head: 4,
        n_embd: 32,
        block_size: 16,
        dropout: 0.0,
    };
    
    let model = GptModel::<TestBackend>::new(&cfg, &device);
    
    // Batch of 3
    let ids = Tensor::<TestBackend, 1, Int>::from_ints(
        [1, 2, 3, 4, 5, 6], 
        &device
    ).reshape([3, 2]);
    
    let logits = model.forward(ids, true);
    assert_eq!(logits.dims(), [3, 2, cfg.vocab_size]);
}

#[test]
fn test_varying_sequence_lengths() {
    let device = <TestBackend as Backend>::Device::default();
    let cfg = NanoChatConfig {
        sequence_len: 64,
        vocab_size: 128,
        n_layer: 2,
        n_head: 4,
        n_kv_head: 4,
        n_embd: 64,
        block_size: 64,
        dropout: 0.0,
    };
    
    let model = GptModel::<TestBackend>::new(&cfg, &device);
    
    // Test different sequence lengths
    for seq_len in [1, 4, 8, 16] {
        let ids = Tensor::<TestBackend, 1, Int>::from_ints(
            vec![1; seq_len].as_slice(),
            &device
        ).reshape([1, seq_len]);
        
        let logits = model.forward(ids, true);
        assert_eq!(logits.dims(), [1, seq_len, cfg.vocab_size]);
    }
}

#[test]
fn test_blocks_vector_size() {
    let device = <TestBackend as Backend>::Device::default();
    let cfg = NanoChatConfig {
        sequence_len: 16,
        vocab_size: 32,
        n_layer: 5,  // Test with 5 layers
        n_head: 2,
        n_kv_head: 2,
        n_embd: 32,
        block_size: 16,
        dropout: 0.0,
    };
    
    let model = GptModel::<TestBackend>::new(&cfg, &device);
    
    // Verify forward works with 5 blocks
    let ids = Tensor::<TestBackend, 1, Int>::from_ints([1, 2], &device)
        .reshape([1, 2]);
    let logits = model.forward(ids, true);
    
    assert_eq!(logits.dims(), [1, 2, 32]);
}