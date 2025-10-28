// // examples/chat.rs

// //! Interactive chat example with conversation history

// use anyhow::Result;
// use nanochat::prelude::*;
// use std::io::{self, Write};

// fn main() -> Result<()> {
//     nanochat::print_backend_info();
//     let device = get_device();
    
//     // Initialize
//     let config = NanoChatConfig::default();
//     let model = GptModel::<AutoBackend>::new(config, &device);
//     let tokenizer = NanoChatTokenizer::from_pretrained("gpt2")?;
//     let engine = Engine::new(model, tokenizer, device);
    
//     println!("\n╔═══════════════════════════════════════╗");
//     println!("║         NanoChat Interactive          ║");
//     println!("╚═══════════════════════════════════════╝\n");
//     println!("Type 'quit' to exit\n");
    
//     // Conversation history
//     let mut conversation = ConversationBuilder::new()
//         .system("You are a helpful AI assistant.")
//         .build();
    
//     loop {
//         // Get user input
//         print!("\n🧑 You: ");
//         io::stdout().flush()?;
        
//         let mut input = String::new();
//         io::stdin().read_line(&mut input)?;
//         let input = input.trim();
        
//         if input.eq_ignore_ascii_case("quit") {
//             println!("Goodbye!");
//             break;
//         }
        
//         if input.is_empty() {
//             continue;
//         }
        
//         // Add user message
//         conversation.push(ChatMessage::user(input));
        
//         // Tokenize conversation
//         let tokens = engine.tokenizer.apply_chat_template_for_generation(&conversation)?;
        
//         // Generate response
//         print!("🤖 Assistant: ");
//         io::stdout().flush()?;
        
//         let mut response_tokens = Vec::new();
//         for (token_column, _masks) in engine.generate(
//             tokens,
//             1,
//             Some(200),
//             0.8,
//             Some(40),
//             42,
//         ) {
//             let token = token_column[0];
//             response_tokens.push(token);
            
//             let text = engine.tokenizer.decode(&[token])?;
//             print!("{}", text);
//             io::stdout().flush()?;
//         }
//         println!();
        
//         // Add assistant response to history
//         let response = engine.tokenizer.decode(&response_tokens)?;
//         conversation.push(ChatMessage::assistant(response));
//     }
    
//     Ok(())
// }
fn main() {}