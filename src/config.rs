//! Configuration structures for NanoChat model and training

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NanoChatConfig {
    // Model architecture
    pub sequence_len: usize,
    pub vocab_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub n_embd: usize,
    pub block_size: usize,
    
    // Training
    pub dropout: f64,
}

impl Default for NanoChatConfig {
    fn default() -> Self {
        Self {
            sequence_len: 1024,
            vocab_size: 50304,
            n_layer: 12,
            n_head: 6,
            n_kv_head: 6,
            n_embd: 768,
            block_size: 1024,
            dropout: 0.0,
        }
    }
}

impl NanoChatConfig {
    pub fn small() -> Self {
        Self::default()
    }

    pub fn medium() -> Self {
        Self {
            n_layer: 24,
            n_head: 12,
            n_kv_head: 12,
            n_embd: 1024,
            ..Default::default()
        }
    }

    pub fn large() -> Self {
        Self {
            n_layer: 36,
            n_head: 20,
            n_kv_head: 20,
            n_embd: 1280,
            ..Default::default()
        }
    }

    /// Load from JSON file
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }

    /// Save to JSON file
    pub fn to_file(&self, path: &str) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}
