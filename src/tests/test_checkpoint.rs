use crate::checkpoint::*;
use crate::gpt::GptModel;
use crate::NanoChatConfig;
use tempfile::TempDir;
use burn::prelude::*;

type TestBackend = crate::backend::AutoBackend;

#[test]
fn test_save_and_load_checkpoint() {
    let device = <TestBackend as Backend>::Device::default();
    let config = NanoChatConfig {
        sequence_len: 16,
        vocab_size: 64,
        n_layer: 2,
        n_head: 2,
        n_kv_head: 2,
        n_embd: 32,
        block_size: 16,
        dropout: 0.0,
    };

    let model = GptModel::<TestBackend>::new(&config, &device);

    // Save to temp dir
    let temp_dir = TempDir::new().unwrap();
    save_checkpoint(&model, &config, temp_dir.path()).unwrap();

    // Load back
    let (loaded_model, loaded_config) =
        load_checkpoint::<TestBackend>(temp_dir.path(), &device).unwrap();

    assert_eq!(config.vocab_size, loaded_config.vocab_size);
    assert_eq!(config.n_layer, loaded_config.n_layer);

    // Verify same forward output (weights preserved)
    use burn::tensor::{Int, Tensor};
    let ids = Tensor::<TestBackend, 1, Int>::from_ints([1, 2, 3], &device)
        .reshape([1, 3]);

    let logits1 = model.forward(ids.clone(), false);
    let logits2 = loaded_model.forward(ids, false);

    let diff: Vec<f32> = (logits1.clone() - logits2.clone())
        .abs()
        .to_data()
        .to_vec()
        .unwrap();

    assert!(
        diff.iter().all(|&x| x < 1e-5),
        "Loaded model weights should match original"
    );
}

