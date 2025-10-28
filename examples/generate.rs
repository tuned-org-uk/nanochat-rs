// // examples/generate.rs

// //! Basic text generation example

// use anyhow::Result;
// use nanochat_burn::prelude::*;

// fn main() -> Result<()> {
//     // Print backend info
//     nanochat_burn::print_backend_info();
//     let device = get_device();
    
//     // Load configuration
//     let config = NanoChatConfig::default();
//     println!("Model config: {:?}", config);
    
//     // Initialize model
//     println!("Initializing model...");
//     let model = GptModel::<AutoBackend>::new(config.clone(), &device);
    
//     // Load tokenizer
//     println!("Loading tokenizer...");
//     let tokenizer = NanoChatTokenizer::from_pretrained("gpt2")?;
    
//     // Create engine
//     let engine = Engine::new(model, tokenizer, device);
    
//     // Generate text
//     let prompt = "Once upon a time";
//     println!("\nPrompt: {}", prompt);
//     println!("Generating...\n");
    
//     let tokens = engine.tokenizer.encode(prompt)?;
    
//     // Stream generation
//     print!("{}", prompt);
//     for (token_column, _masks) in engine.generate(
//         tokens,
//         1,              // num_samples
//         Some(100),      // max_tokens
//         0.8,            // temperature
//         Some(40),       // top_k
//         42,             // seed
//     ) {
//         let text = engine.tokenizer.decode(&[token_column[0]])?;
//         print!("{}", text);
//     }
//     println!("\n");
    
//     Ok(())
// }
fn main() {}