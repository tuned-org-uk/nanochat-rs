// src/bin/minimal.rs
use anyhow::Result;
use burn::tensor::{Int, Tensor};

use nanochat::{
    backend::{get_device, print_backend_info, AutoBackend},
    config::NanoChatConfig,
    gpt::GptModel,
};

fn main() -> Result<()> {
    env_logger::init();
    
    print_backend_info();
    let device = get_device();

    // Milestone 3 config: Multiple blocks
    let cfg = NanoChatConfig {
        sequence_len: 64,
        vocab_size: 128,
        n_layer: 4,  // ← Changed from 1 to 4
        n_head: 4,
        n_kv_head: 4,
        n_embd: 64,
        block_size: 64,
        dropout: 0.0,
    };

    println!("\n🔨 Testing Milestone 3: Multi-block GPT");
    println!("   n_layer = {}", cfg.n_layer);
    
    let model = GptModel::<AutoBackend>::new(&cfg, &device);

    let seed_tokens: Vec<i64> = vec![1, 2, 3, 4];
    let input = Tensor::<AutoBackend, 1, Int>::from_ints(seed_tokens.as_slice(), &device)
        .reshape([1, seed_tokens.len()]);

    println!("\n1️⃣ Forward pass test...");
    let logits = model.forward(input.clone(), true);
    let shape = logits.dims();
    println!("   ✓ Logits shape = {:?}", shape);
    assert_eq!(shape, [1, 4, cfg.vocab_size], "Shape mismatch");

    println!("\n2️⃣ Generation test (10 tokens)...");
    let out = model.generate(input, 10);
    let out_ids = out.to_data().to_vec::<i32>().unwrap();
    println!("   ✓ Generated {} tokens", out_ids.len());
    println!("   Token IDs: {:?}", out_ids);

    println!("\n✅ Milestone 3 complete: N-block GPT validated!\n");
    Ok(())
}
