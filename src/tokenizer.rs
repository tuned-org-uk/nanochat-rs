// src/tokenizer.rs

//! Tokenizer for NanoChat with special token support and chat templating
//! Mirrors the Python tokenizer.py implementation

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ═════════════════════════════════════════════════════════════════════════════
// Special tokens for NanoChat
// ═════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialTokens {
    pub bos_token: String,
    pub eos_token: String,
    pub user_start: String,
    pub user_end: String,
    pub assistant_start: String,
    pub assistant_end: String,
    pub system_start: String,
    pub system_end: String,
    pub python_start: String,
    pub python_end: String,
    pub output_start: String,
    pub output_end: String,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            bos_token: "<|bos|>".to_string(),
            eos_token: "<|eos|>".to_string(),
            user_start: "<|user_start|>".to_string(),
            user_end: "<|user_end|>".to_string(),
            assistant_start: "<|assistant_start|>".to_string(),
            assistant_end: "<|assistant_end|>".to_string(),
            system_start: "<|system_start|>".to_string(),
            system_end: "<|system_end|>".to_string(),
            python_start: "<|python_start|>".to_string(),
            python_end: "<|python_end|>".to_string(),
            output_start: "<|output_start|>".to_string(),
            output_end: "<|output_end|>".to_string(),
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Message structure for chat templating
// ═════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Main tokenizer struct
// ═════════════════════════════════════════════════════════════════════════════

pub struct NanoChatTokenizer {
    base_tokenizer: tokenizers::Tokenizer,
    special_tokens: SpecialTokens,
    special_token_map: HashMap<String, u32>,
    vocab_size: usize,
    bos_token_id: u32,
    eos_token_id: u32,
}

impl NanoChatTokenizer {
    /// Load tokenizer by pointing to tokenizer.json file OR a directory containing it.
    pub fn from_path(model_path: impl AsRef<Path>) -> Result<Self> {
        let path = model_path.as_ref();
        let resolved = resolve_tokenizer_file(path)?;
        let base = tokenizers::Tokenizer::from_file(&resolved)
            .map_err(|e| anyhow!("Failed to load tokenizer from {}: {e}", resolved.display()))?;
        Self::with_base(base)
    }

    /// Strictly from a tokenizer.json file path.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let base = tokenizers::Tokenizer::from_file(path.as_ref())
            .map_err(|e| anyhow!("Failed to load tokenizer from file: {e}"))?;
        Self::with_base(base)
    }

    /// Optionally from in-memory bytes of a tokenizer.json file.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let base = tokenizers::Tokenizer::from_bytes(bytes)
            .map_err(|e| anyhow!("Failed to load tokenizer from bytes: {e}"))?;
        Self::with_base(base)
    }

    fn with_base(mut base_tokenizer: tokenizers::Tokenizer) -> Result<Self> {
        let special_tokens = SpecialTokens::default();
        let mut special_token_map = HashMap::new();

        let special_token_strings: Vec<String> = vec![
            special_tokens.bos_token.clone(),
            special_tokens.eos_token.clone(),
            special_tokens.user_start.clone(),
            special_tokens.user_end.clone(),
            special_tokens.assistant_start.clone(),
            special_tokens.assistant_end.clone(),
            special_tokens.system_start.clone(),
            special_tokens.system_end.clone(),
            special_tokens.python_start.clone(),
            special_tokens.python_end.clone(),
            special_tokens.output_start.clone(),
            special_tokens.output_end.clone(),
        ];

        let added: Vec<tokenizers::AddedToken> = special_token_strings
            .iter()
            .map(|s| tokenizers::AddedToken::from(s.clone(), true))
            .collect();

        let _n_added = base_tokenizer.add_special_tokens(&added);

        for token_str in &special_token_strings {
            if let Some(id) = base_tokenizer.token_to_id(token_str) {
                special_token_map.insert(token_str.clone(), id);
            }
        }

        let vocab_size = base_tokenizer.get_vocab_size(true);
        let bos_token_id = special_token_map
            .get(&special_tokens.bos_token)
            .copied()
            .ok_or_else(|| anyhow!("BOS token not found after registration"))?;
        let eos_token_id = special_token_map
            .get(&special_tokens.eos_token)
            .copied()
            .ok_or_else(|| anyhow!("EOS token not found after registration"))?;

        Ok(Self {
            base_tokenizer,
            special_tokens,
            special_token_map,
            vocab_size,
            bos_token_id,
            eos_token_id,
        })
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Core encoding/decoding
    // ─────────────────────────────────────────────────────────────────────────

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .base_tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!("Encoding failed: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn encode_with_bos(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = vec![self.bos_token_id];
        ids.extend(self.encode(text)?);
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.base_tokenizer
            .decode(ids, false)
            .map_err(|e| anyhow!("Decoding failed: {e}"))
    }

    pub fn decode_skip_special(&self, ids: &[u32]) -> Result<String> {
        self.base_tokenizer
            .decode(ids, true)
            .map_err(|e| anyhow!("Decoding failed: {e}"))
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Special token utilities
    // ─────────────────────────────────────────────────────────────────────────

    pub fn encode_special(&self, token: &str) -> Option<u32> {
        self.special_token_map.get(token).copied()
    }

    pub fn get_bos_token_id(&self) -> u32 {
        self.bos_token_id
    }

    pub fn get_eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn is_special_token(&self, id: u32) -> bool {
        self.special_token_map.values().any(|&v| v == id)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Chat templating
    // ─────────────────────────────────────────────────────────────────────────

    pub fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        let mut tokens = vec![self.bos_token_id];

        for msg in messages {
            let (start_token, end_token) = match msg.role.as_str() {
                "user" => (
                    &self.special_tokens.user_start,
                    &self.special_tokens.user_end,
                ),
                "assistant" => (
                    &self.special_tokens.assistant_start,
                    &self.special_tokens.assistant_end,
                ),
                "system" => (
                    &self.special_tokens.system_start,
                    &self.special_tokens.system_end,
                ),
                _ => return Err(anyhow!("Unknown role: {}", msg.role)),
            };

            if let Some(start_id) = self.encode_special(start_token) {
                tokens.push(start_id);
            }
            tokens.extend(self.encode(&msg.content)?);
            if let Some(end_id) = self.encode_special(end_token) {
                tokens.push(end_id);
            }
        }

        Ok(tokens)
    }

    pub fn apply_chat_template_for_generation(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        let mut tokens = self.apply_chat_template(messages)?;
        if let Some(assistant_start) = self.encode_special(&self.special_tokens.assistant_start) {
            tokens.push(assistant_start);
        }
        Ok(tokens)
    }

    pub fn format_user_message(&self, content: &str) -> Result<Vec<u32>> {
        let messages = vec![ChatMessage::user(content)];
        self.apply_chat_template_for_generation(&messages)
    }

    pub fn format_conversation(
        &self,
        system_prompt: Option<&str>,
        messages: &[ChatMessage],
    ) -> Result<Vec<u32>> {
        let mut all_messages = Vec::new();
        if let Some(system) = system_prompt {
            all_messages.push(ChatMessage::system(system));
        }
        all_messages.extend_from_slice(messages);
        self.apply_chat_template_for_generation(&all_messages)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Tool use formatting
    // ─────────────────────────────────────────────────────────────────────────

    pub fn format_python_call(&self, expression: &str) -> Result<Vec<u32>> {
        let mut tokens = Vec::new();
        if let Some(start_id) = self.encode_special(&self.special_tokens.python_start) {
            tokens.push(start_id);
        }
        tokens.extend(self.encode(expression)?);
        if let Some(end_id) = self.encode_special(&self.special_tokens.python_end) {
            tokens.push(end_id);
        }
        Ok(tokens)
    }

    pub fn format_python_output(&self, output: &str) -> Result<Vec<u32>> {
        let mut tokens = Vec::new();
        if let Some(start_id) = self.encode_special(&self.special_tokens.output_start) {
            tokens.push(start_id);
        }
        tokens.extend(self.encode(output)?);
        if let Some(end_id) = self.encode_special(&self.special_tokens.output_end) {
            tokens.push(end_id);
        }
        Ok(tokens)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Batch ops and introspection
    // ─────────────────────────────────────────────────────────────────────────

    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>> {
        texts.iter().map(|text| self.encode(text)).collect()
    }

    pub fn decode_batch(&self, ids_batch: &[&[u32]]) -> Result<Vec<String>> {
        ids_batch.iter().map(|ids| self.decode(ids)).collect()
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.base_tokenizer.id_to_token(id)
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.base_tokenizer.token_to_id(token)
    }

    pub fn get_special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    pub fn get_special_token_map(&self) -> &HashMap<String, u32> {
        &self.special_token_map
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Save
    // ─────────────────────────────────────────────────────────────────────────

    pub fn save<P: AsRef<Path>>(&self, path: P, pretty: bool) -> Result<()> {
        self.base_tokenizer
            .save(path.as_ref().to_str().unwrap(), pretty)
            .map_err(|e| anyhow!("Failed to save tokenizer: {e}"))
    }
}

// Helper: resolve tokenizer.json from a file or directory path.
fn resolve_tokenizer_file(path: &Path) -> Result<PathBuf> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    if path.is_dir() {
        let candidate = path.join("tokenizer.json");
        if candidate.exists() && candidate.is_file() {
            return Ok(candidate);
        }
        return Err(anyhow!(
            "Directory '{}' does not contain tokenizer.json",
            path.display()
        ));
    }
    Err(anyhow!("Path '{}' not found", path.display()))
}

// ═════════════════════════════════════════════════════════════════════════════
// Convenience builder for chat messages
// ═════════════════════════════════════════════════════════════════════════════

pub struct ConversationBuilder {
    messages: Vec<ChatMessage>,
}

impl ConversationBuilder {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    pub fn system(mut self, content: impl Into<String>) -> Self {
        self.messages.push(ChatMessage::system(content));
        self
    }

    pub fn user(mut self, content: impl Into<String>) -> Self {
        self.messages.push(ChatMessage::user(content));
        self
    }

    pub fn assistant(mut self, content: impl Into<String>) -> Self {
        self.messages.push(ChatMessage::assistant(content));
        self
    }

    pub fn push(&mut self, message: ChatMessage) {
        self.messages.push(message);
    }

    pub fn build(self) -> Vec<ChatMessage> {
        self.messages
    }
}

impl Default for ConversationBuilder {
    fn default() -> Self {
        Self::new()
    }
}
