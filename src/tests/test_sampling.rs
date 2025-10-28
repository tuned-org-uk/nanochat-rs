// ═════════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════════

use crate::sampling::*;
use crate::backend::AutoBackend;
use burn::prelude::Backend;
use burn::tensor::{activation, Tensor};
    
type TestBackend = AutoBackend;

fn create_test_logits() -> Tensor<TestBackend, 2> {
    let data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0,
        5.0, 4.0, 3.0, 2.0, 1.0,
    ];
    Tensor::<TestBackend, 1>::from_floats(data.as_slice(), &Default::default())
        .reshape([2, 5])
}

#[test]
fn test_greedy_sampling() {
    let logits = create_test_logits();
    let sampled = sample_greedy(logits);
    
    let ids = sampled.to_data().to_vec::<i64>().unwrap();
    assert_eq!(ids[0], 4);
    assert_eq!(ids[1], 0);
}

#[test]
fn test_temperature_scaling() {
    let logits = create_test_logits();
    
    let scaled_high = apply_temperature(logits.clone(), 2.0);
    let scaled_low = apply_temperature(logits.clone(), 0.5);
    
    let original_data = logits.to_data().to_vec::<f32>().unwrap();
    let high_data = scaled_high.to_data().to_vec::<f32>().unwrap();
    let low_data = scaled_low.to_data().to_vec::<f32>().unwrap();
    
    assert!((high_data[4] - original_data[4] / 2.0).abs() < 1e-5);
    assert!((low_data[4] - original_data[4] * 2.0).abs() < 1e-5);
}

#[test]
fn test_top_k_filter() {
    let logits = create_test_logits();
    let filtered = top_k_filter(logits, 2);
    
    let probs = activation::softmax(filtered, 1);
    let prob_data = probs.to_data().to_vec::<f32>().unwrap();
    
    assert!(prob_data[0] < 0.01);
    assert!(prob_data[3] > 0.1);
    assert!(prob_data[4] > 0.1);
}

#[test]
fn test_top_k_one_equals_greedy() {
    let logits = create_test_logits();
    
    let top1_sample = sample_with_temperature_topk(logits.clone(), 1.0, Some(1));
    let greedy_sample = sample_greedy(logits);
    
    let top1_ids = top1_sample.to_data().to_vec::<i64>().unwrap();
    let greedy_ids = greedy_sample.to_data().to_vec::<i64>().unwrap();
    
    assert_eq!(top1_ids, greedy_ids);
}

#[test]
fn test_temperature_zero_is_greedy() {
    let logits = create_test_logits();
    
    let temp_zero = sample_next_token(logits.clone(), 0.0, None);
    let greedy = sample_greedy(logits);
    
    let temp_ids = temp_zero.to_data().to_vec::<i64>().unwrap();
    let greedy_ids = greedy.to_data().to_vec::<i64>().unwrap();
    
    assert_eq!(temp_ids, greedy_ids);
}

#[test]
fn test_extract_last_logits() {
    let data: Vec<f32> = (0..30).map(|i| i as f32).collect();
    let logits_3d = Tensor::<TestBackend, 1>::from_floats(data.as_slice(), &Default::default())
        .reshape([2, 3, 5]);
    
    let last_logits = extract_last_logits(logits_3d);
    assert_eq!(last_logits.dims(), [2, 5]);
    
    let extracted = last_logits.to_data().to_vec::<f32>().unwrap();
    assert_eq!(extracted[0], 10.0);
    assert_eq!(extracted[9], 29.0);
}

#[test]
#[should_panic(expected = "Temperature must be positive")]
fn test_negative_temperature_panics() {
    let logits = create_test_logits();
    let _ = apply_temperature(logits, -1.0);
}

#[test]
fn test_batch_size_one() {
    let data = vec![1.0_f32, 2.0, 3.0];
    let logits = Tensor::<TestBackend, 1>::from_floats(data.as_slice(), &Default::default())
        .reshape([1, 3]);
    
    let sampled = sample_greedy(logits);
    let ids = sampled.to_data().to_vec::<i64>().unwrap();
    
    assert_eq!(ids.len(), 1);
    assert_eq!(ids[0], 2);
}

#[test]
fn test_large_vocab() {
    let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    let logits = Tensor::<TestBackend, 1>::from_floats(data.as_slice(), &Default::default())
        .reshape([1, 1000]);
    
    let sampled = sample_greedy(logits);
    let ids = sampled.to_data().to_vec::<i64>().unwrap();
    
    assert_eq!(ids[0], 999);
}