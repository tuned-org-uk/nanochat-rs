// src/lib.rs

//! NanoChat - A GPT implementation in Rust using Burn
//!
//! This library provides a complete implementation of the NanoChat model
//! with modern features like rotary embeddings, Multi-Query Attention,
//! and efficient KV caching.

pub mod backend;
pub mod config;
pub mod gpt;
pub mod tokenizer;

#[cfg(test)]
mod tests;


pub use backend::{get_device, print_backend_info, AutoBackend};
pub use config::NanoChatConfig;

use std::sync::Once;

static INIT: Once = Once::new();

pub fn init() {
    INIT.call_once(|| {
        // Read RUST_LOG env variable, default to "info" if not set
        let env = env_logger::Env::default().default_filter_or("debug");

        // don't panic if called multiple times across binaries
        let _ = env_logger::Builder::from_env(env)
            .is_test(true) // nicer formatting for tests
            .try_init();
    });
}


// pub mod tokenizer;
// pub mod model {
//     pub mod gpt;
// }
// pub mod engine;
// pub mod sampling;
// pub mod loss;
// pub mod inference;

// // Re-export commonly used types
// pub use backend::{AutoBackend, get_device, print_backend_info, is_gpu_available};
// pub use config::NanoChatConfig;
// pub use tokenizer::{NanoChatTokenizer, ChatMessage, ConversationBuilder};
// pub use model::gpt::GptModel;
// pub use engine::{Engine, KVCache};
// pub use sampling::sample_next_token;
// pub use loss::{language_modeling_loss, next_token_loss};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        backend::{AutoBackend, get_device},
        config::NanoChatConfig,
        gpt::GptModel,
    };
}
