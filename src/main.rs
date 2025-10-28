// src/bin/minimal.rs
// Minimal smoke test for Milestone 0 GPT

use anyhow::Result;
use burn::tensor::{backend::Backend, Int, Tensor};

use nanochat::{
    backend::{get_device, print_backend_info, AutoBackend},
    config::NanoChatConfig,
    gpt::GptModel,
};

fn main() -> Result<()> {
    // 1) Print backend config and get device
    print_backend_info();
    let device = get_device();

    // 2) Minimal model config (keep tiny for a fast GPU/CPU smoke test)
    let cfg = NanoChatConfig {
        sequence_len: 64,
        vocab_size: 128,  // tiny vocab for smoke
        n_layer: 1,
        n_head: 4,
        n_kv_head: 4,
        n_embd: 64,
        block_size: 64,
        dropout: 0.0,
    };

    // 3) Instantiate the minimal model
    let model = GptModel::<AutoBackend>::new(&cfg, &device);

    // 4) Create a tiny fake input sequence (batch = 1)
    //    Use small token ids within [0, vocab_size)
    let seed_tokens: Vec<i64> = vec![1, 2, 3, 4];
    let input: Tensor<AutoBackend, 1, Int> =
        Tensor::from_ints(seed_tokens.as_slice(), &device);
    let input = input.reshape([1, seed_tokens.len()]); // [B=1, T=4]

    // 5) Forward once to check shapes
    let logits = model.forward(input.clone()); // [1, 4, vocab]
    let shape = logits.dims();
    println!("Forward OK. Logits shape = {:?}", shape);

    // 6) Greedy-generate a few tokens
    let out = model.generate(input, 10); // append 10 tokens
    let out_ids = out.to_data().to_vec::<i32>().unwrap();
    println!("Generated token ids (batch-major): {:?}", out_ids);

    Ok(())
}
