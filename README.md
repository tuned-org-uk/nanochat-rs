# nanochat-rs 🔥

A GPT implementation in Rust using the [Burn](https://github.com/tracel-ai/burn) deep learning framework. This is a high-performance Rust port inspired by [Andrej Karpathy's nanochat](https://github.com/karpathy/nanochat), featuring modern transformer architecture with advanced optimizations.

Karpathy's original [repo and license here](https://github.com/karpathy/nanochat/blob/1ccbaf4416bc39dfb75f0e4fe934363278adcfda/LICENSE)

## Try it
```
cargo run --release --features wgpu
cargo run --release --features cuda
```

Test model capabilities:
```
cargo run --example check_features --features wgpu -- --nocapture
```

## Overview

nanochat-rs implements a decoder-only transformer with state-of-the-art features for efficient text generation. Built on Burn 0.18, it leverages Rust's performance and safety guarantees while providing GPU acceleration through multiple backends (WGPU, CUDA, CPU).

## Architecture

### Core Components

**Model Architecture (gpt.rs)**

- **Multi-layer Transformer**: N stacked decoder blocks with pre-norm residual connections
- **Rotary Position Embeddings (RoPE)**: Replaces learned positional encodings with rotary embeddings for better length generalization
- **Multi-Query Attention (MQA)**: Reduces KV cache size by sharing key/value heads across query heads
- **RMSNorm**: Parameter-free normalization for stability (instead of LayerNorm)
- **QK-norm**: Normalizes queries and keys before attention to prevent numerical instability
- **ReLU² MLP**: Uses ReLU(x)² activation for better gradient flow on GPUs
- **Softcap Logits**: Bounds output logits using tanh(x/15)*15 to prevent extreme values


## Building

```bash
# CPU only (fast compile)
cargo build --release

# GPU with WGPU
cargo build --release --features wgpu

# NVIDIA with CUDA
cargo build --release --features cuda
```


## Testing

```bash
# Unit tests
cargo test

# Integration test with demo
cargo run --release --bin main
```

## Technical Details

**Attention Mechanism**:

- Scaled dot-product with sqrt(d_k) normalization
- Causal masking via tril_mask with large-negative fill (-1e9)
- Max-subtraction per row before softmax for numerical stability
- MQA: KV heads repeated via unsqueeze/expand to match Q heads

**Forward Pass Flow**:

```
Input IDs [B,T] 
→ Embedding [B,T,C]
→ N × Block(RMSNorm → Attn+Residual → RMSNorm → MLP+Residual)
→ Final RMSNorm [B,T,C]
→ LM Head [B,T,V]
→ Softcap + Clamp
→ Logits [B,T,V]
```

**Decode Flow** (with cache):

```
Last Token [B,1]
→ Embed [B,1,C]
→ N × Block(decode with cache update)
→ Final RMSNorm [B,1,C]
→ LM Head [B,1,V]
→ Sample next token
```

### Key Features by Milestone

**M2: Sampling Policies**

- Greedy (argmax)
- Temperature scaling
- Top-k filtering
- Top-p (nucleus) sampling
- Combined policies (temp + top-k + top-p)

**M3: Multi-Block GPT**

- N transformer blocks with clean Block interface
- Pre-norm residual architecture
- Shape validation and numerical stability checks

**M4: KV Cache \& Streaming**

- Efficient KV caching for O(1) decode steps
- Streaming iterator interface for token-by-token generation
- Cache stored as Vec<Option<(K,V)>> to avoid complex 6D tensor mutations

**M5: RoPE (Rotary Position Embeddings)**

- Precomputed sin/cos frequencies (base=10000)
- Applied to Q and K in [B, H, T, D] format
- Position-aware without learned parameters

**M6: RMSNorm \& QK-norm**

- Functional RMSNorm (no trainable params) for block pre-norm
- QK-norm applied to queries/keys after RoPE, before attention
- Improves numerical stability during training and inference

**M7: Multi-Query Attention**

- Configurable n_kv_head parameter (e.g., n_head=8, n_kv_head=2)
- Reduces memory footprint for KV cache
- Maintains quality while enabling larger batch sizes

**M8: Advanced Sampling**

- Injectable sampling policies via SamplingPolicy enum
- Combined temperature/top-k/top-p strategies
- CPU-based top-p with sorted cumulative probability

**M9: Logits Softcap**

- tanh(logits/15)*15 with pre/post clamping at ±50
- Prevents extreme logit values during generation
- Maintains stable softmax behavior

**M10: Checkpoint I/O**

- Separate config (JSON) and weights (MessagePack) serialization
- Uses NamedMpkFileRecorder for cross-backend compatibility
- Clean save/load API via checkpoint module


## Project Structure

```
src/
├── lib.rs              # Public API exports
├── gpt.rs              # Core GPT model implementation
├── config.rs           # Model hyperparameters
├── engine.rs           # KV cache and streaming interface
├── sampling.rs         # Sampling strategies
├── checkpoint.rs       # Model serialization
├── backend.rs          # Multi-backend support (WGPU/CUDA/Metal)
└── tokenizer.rs        # BPE tokenizer (compatible with rustbpe)
```


## Usage

### Basic Inference

```rust
use burn::tensor::{Int, Tensor};
use nanochat::{
    backend::{get_device, AutoBackend},
    config::nanochatConfig,
    gpt::GptModel,
    sampling::{sample_with_policy, SamplingPolicy},
};

// Configure model
let cfg = nanochatConfig {
    vocab_size: 65536,
    n_layer: 12,
    n_head: 8,
    n_kv_head: 2,  // MQA: 2 KV heads shared across 8 Q heads
    n_embd: 768,
    sequence_len: 2048,
    block_size: 2048,
    dropout: 0.0,
};

let device = get_device();
let model = GptModel::<AutoBackend>::new(&cfg, &device);

// Encode input (token IDs from tokenizer)
let input_ids = vec![1, 2, 3, 4, 5];
let input = Tensor::<AutoBackend, 1, Int>::from_ints(&input_ids, &device)
    .reshape([1, input_ids.len()]);

// Generate with temperature sampling
let output = model.generate(input, 50);
```


### Streaming Generation with KV Cache

```rust
use nanochat::engine::{Engine, KVCache};

let engine = Engine::new(model, device);

// Stream tokens one at a time
for next_token in engine.stream(input, 100) {
    let token_id = next_token.to_data().to_vec::<i64>().unwrap()[^0];
    // Decode and display token
    print!("{}", tokenizer.decode(&[token_id as u32]));
}
```


### Custom Sampling Policy

```rust
use nanochat::sampling::{extract_last_logits, sample_with_policy, SamplingPolicy};

let logits = model.forward(input, true);  // true = use softcap
let last_logits = extract_last_logits(logits);

// Nucleus sampling with temperature
let next_token = sample_with_policy(
    last_logits,
    SamplingPolicy::TempTopP { t: 0.8, p: 0.9 }
);
```


### Save/Load Checkpoints

```rust
use nanochat::checkpoint::{save_checkpoint, load_checkpoint};

// Save
save_checkpoint(&model, &cfg, "./checkpoints/model_v1")?;

// Load
let (loaded_model, loaded_cfg) = load_checkpoint::<AutoBackend>(
    "./checkpoints/model_v1",
    &device
)?;
```


## Configuration

Key hyperparameters in `nanochatConfig`:

```rust
pub struct nanochatConfig {
    pub vocab_size: usize,      // Tokenizer vocabulary size
    pub n_layer: usize,         // Number of transformer blocks
    pub n_head: usize,          // Number of query heads
    pub n_kv_head: usize,       // Number of KV heads (MQA)
    pub n_embd: usize,          // Embedding dimension
    pub sequence_len: usize,    // Maximum sequence length
    pub block_size: usize,      // Context window size
    pub dropout: f64,           // Dropout rate (0.0 for inference)
}
```


## Performance Optimizations

1. **Numerical Stability**: QK-norm, RMSNorm, softcap logits, and stable attention softmax (max-subtraction)
2. **Memory Efficiency**: MQA reduces KV cache size; simple Vec<Option<(K,V)>> storage avoids complex indexing
3. **GPU Optimization**: ReLU² activation, fused operations, Burn's JIT compilation with autotuning
4. **Kaiming Initialization**: Reduced gain (0.5) for stable training convergence
5. **Clamping**: Pre/post softcap clamps at ±50 prevent overflow/underflow

## Backend Support

Automatically selects best available backend:

- **WGPU** (default): Cross-platform GPU via Vulkan/Metal/DX12
- **CUDA**: NVIDIA GPUs with cuDNN
- **NdArray**: CPU fallback for testing

Override with environment:

```bash
export BURN_BACKEND=wgpu  # or cuda, ndarray
```
