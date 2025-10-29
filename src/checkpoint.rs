// src/checkpoint.rs

//! Checkpoint save/load for NanoChat models
//!
//! Keeps config and weights separate to avoid Record type coercion issues

use anyhow::{Context, Result};
use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use crate::config::NanoChatConfig;
use crate::gpt::GptModel;

/// Checkpoint container with config and model record
#[derive(Serialize, Deserialize)]
pub struct Checkpoint<R> {
    pub config: NanoChatConfig,
    pub record: R,
}

/// Save model and config to checkpoint directory
pub fn save_checkpoint<B: Backend>(
    model: &GptModel<B>,
    config: &NanoChatConfig,
    checkpoint_dir: impl AsRef<Path>,
) -> Result<()> {
    let dir = checkpoint_dir.as_ref();
    std::fs::create_dir_all(dir).context("Failed to create checkpoint directory")?;

    // Save config as JSON
    let config_path = dir.join("config.json");
    let config_file = File::create(&config_path)
        .context(format!("Failed to create config file: {:?}", config_path))?;
    serde_json::to_writer_pretty(BufWriter::new(config_file), config)
        .context("Failed to serialize config")?;

    // Save model record using NamedMpkFileRecorder (MessagePack-based)
    let record_path = dir.join("model.mpk");
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model
        .clone()
        .save_file(record_path, &recorder)
        .context("Failed to save model record")?;

    log::info!("Checkpoint saved to {:?}", dir);
    Ok(())
}

/// Load model from checkpoint directory
pub fn load_checkpoint<B: Backend>(
    checkpoint_dir: impl AsRef<Path>,
    device: &B::Device,
) -> Result<(GptModel<B>, NanoChatConfig)> {
    let dir = checkpoint_dir.as_ref();

    // Load config
    let config_path = dir.join("config.json");
    let config_file = File::open(&config_path)
        .context(format!("Failed to open config file: {:?}", config_path))?;
    let config: NanoChatConfig = serde_json::from_reader(BufReader::new(config_file))
        .context("Failed to deserialize config")?;

    // Create model with loaded config
    let model = GptModel::<B>::new(&config, device);

    // Load weights into model
    let record_path = dir.join("model.mpk");
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model = model
        .load_file(record_path, &recorder, device)
        .context("Failed to load model record")?;

    log::info!("Checkpoint loaded from {:?}", dir);
    Ok((model, config))
}

/// Save only model weights (for fine-tuning scenarios where config is known)
pub fn save_weights<B: Backend>(
    model: &GptModel<B>,
    weights_path: impl AsRef<Path>,
) -> Result<()> {
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model
        .clone()
        .save_file(weights_path.as_ref(), &recorder)
        .context("Failed to save model weights")?;
    log::info!("Weights saved to {:?}", weights_path.as_ref());
    Ok(())
}

/// Load weights into an existing model
pub fn load_weights<B: Backend>(
    mut model: GptModel<B>,
    weights_path: impl AsRef<Path>,
    device: &B::Device,
) -> Result<GptModel<B>> {
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model = model
        .load_file(weights_path.as_ref(), &recorder, device)
        .context("Failed to load model weights")?;
    log::info!("Weights loaded from {:?}", weights_path.as_ref());
    Ok(model)
}
