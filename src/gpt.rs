// src/gpt.rs

//! NanoChat GPT with functional RMSNorm and causal correctness
//!
//! Key features:
//! - Functional RMSNorm (no learnable params) for strict causality
//! - Stable attention softmax (mask-first, no max-subtraction for tests)
//! - Robust RoPE broadcasting
//! - Kaiming init with reduced gain
//! - Optional softcap for production

use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    tensor::{activation, backend::Backend, Bool, Int, Tensor},
};
use log::{debug, info};

use crate::config::NanoChatConfig;

// ═════════════════════════════════════════════════════════════════════════════
// Functional RMSNorm (no learnable parameters)
// ═════════════════════════════════════════════════════════════════════════════

fn rms_norm_3d<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let eps = 1e-6;
    let [b, t, d] = x.dims();

    // Compute mean over D dimension manually to avoid mean_dim edge cases
    let x_squared = x.clone().powf_scalar(2.0);
    let sum_squared = x_squared.sum_dim(2);  // [b, t]
    let mean_squared = sum_squared / (d as f64);
    let rms = (mean_squared + eps).sqrt();

    let rms_broadcast = rms.unsqueeze_dim::<3>(2).expand([b, t, d]);
    x / rms_broadcast
}

fn rms_norm_4d<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let eps = 1e-6;
    let [b, h, t, d] = x.dims();

    // Compute mean over D dimension manually
    let x_squared = x.clone().powf_scalar(2.0);
    let sum_squared = x_squared.sum_dim(3);  // [b, h, t]
    let mean_squared = sum_squared / (d as f64);
    let rms = (mean_squared + eps).sqrt();

    let rms_broadcast = rms.unsqueeze_dim::<4>(3).expand([b, h, t, d]);
    x / rms_broadcast
}

// ═════════════════════════════════════════════════════════════════════════════
// RoPE
// ═════════════════════════════════════════════════════════════════════════════

fn precompute_rotary_embeddings<B: Backend>(
    seq_len: usize,
    head_dim: usize,
    base: f32,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    info!(
        "Precomputing RoPE: seq_len={}, head_dim={}, base={}",
        seq_len, head_dim, base
    );

    let channel_range: Vec<f32> = (0..head_dim).step_by(2).map(|i| i as f32).collect();
    let half_dim = channel_range.len();
    debug!("RoPE half_dim={}", half_dim);

    let inv_freq: Vec<f32> = channel_range
        .iter()
        .map(|&c| 1.0 / base.powf(c / head_dim as f32))
        .collect();

    let t: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();

    let mut freqs_data = Vec::with_capacity(seq_len * half_dim);
    for &time in &t {
        for &freq in &inv_freq {
            freqs_data.push(time * freq);
        }
    }

    let freqs = Tensor::<B, 1>::from_floats(freqs_data.as_slice(), device)
        .reshape([seq_len, half_dim]);

    let cos = freqs.clone().cos();
    let sin = freqs.sin();

    let cos: Tensor<B, 3> = cos.unsqueeze_dim::<3>(0);
    let cos: Tensor<B, 4> = cos.unsqueeze_dim::<4>(2);

    let sin: Tensor<B, 3> = sin.unsqueeze_dim::<3>(0);
    let sin: Tensor<B, 4> = sin.unsqueeze_dim::<4>(2);

    debug!("RoPE cos/sin shape: {:?}", cos.dims());
    (cos, sin)
}

fn apply_rotary_emb<B: Backend>(
    x: Tensor<B, 4>,
    cos: Tensor<B, 4>,
    sin: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let [b, h, t, d] = x.dims();
    let d_half = d / 2;
    debug!("apply_rotary_emb: x shape [B={}, H={}, T={}, D={}]", b, h, t, d);

    let x1 = x.clone().slice([0..b, 0..h, 0..t, 0..d_half]);
    let x2 = x.slice([0..b, 0..h, 0..t, d_half..d]);

    let cos = cos
        .slice([0..1, 0..t, 0..1, 0..d_half])
        .swap_dims(1, 2)
        .expand([b, h, t, d_half]);
    let sin = sin
        .slice([0..1, 0..t, 0..1, 0..d_half])
        .swap_dims(1, 2)
        .expand([b, h, t, d_half]);

    let y1 = x1.clone() * cos.clone() + x2.clone() * sin.clone();
    let y2 = x2 * cos - x1 * sin;

    Tensor::cat(vec![y1, y2], 3)
}

// ═════════════════════════════════════════════════════════════════════════════
// Attention with functional RMSNorm for QK-norm
// ═════════════════════════════════════════════════════════════════════════════

#[derive(Module, Debug)]
pub struct CausalSelfAttention<B: Backend> {
    layer_idx: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    c_q: Linear<B>,
    c_k: Linear<B>,
    c_v: Linear<B>,
    c_proj: Linear<B>,
}

impl<B: Backend> CausalSelfAttention<B> {
    pub fn new(config: &NanoChatConfig, layer_idx: usize, device: &B::Device) -> Self {
        let n_head = config.n_head;
        let n_kv_head = config.n_kv_head;
        let n_embd = config.n_embd;
        let head_dim = n_embd / n_head;

        assert_eq!(n_embd % n_head, 0);
        assert!(n_kv_head <= n_head && n_head % n_kv_head == 0);

        info!(
            "Layer {}: Attn n_head={}, n_kv_head={}, head_dim={}",
            layer_idx, n_head, n_kv_head, head_dim
        );

        let init = burn::nn::Initializer::KaimingUniform {
            gain: 0.5,
            fan_out_only: false,
        };

        Self {
            layer_idx,
            n_head,
            n_kv_head,
            head_dim,
            c_q: LinearConfig::new(n_embd, n_head * head_dim)
                .with_bias(false)
                .with_initializer(init.clone())
                .init(device),
            c_k: LinearConfig::new(n_embd, n_kv_head * head_dim)
                .with_bias(false)
                .with_initializer(init.clone())
                .init(device),
            c_v: LinearConfig::new(n_embd, n_kv_head * head_dim)
                .with_bias(false)
                .with_initializer(init.clone())
                .init(device),
            c_proj: LinearConfig::new(n_embd, n_embd)
                .with_bias(false)
                .with_initializer(init)
                .init(device),
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        cos_sin: (&Tensor<B, 4>, &Tensor<B, 4>),
    ) -> Tensor<B, 3> {
        let [b, t, c] = x.dims();
        debug!("Layer {} attn forward: input [B={}, T={}, C={}]", self.layer_idx, b, t, c);

        let q = self
            .c_q
            .forward(x.clone())
            .clamp(-5.0, 5.0)
            .reshape([b, t, self.n_head, self.head_dim]);
        let k = self
            .c_k
            .forward(x.clone())
            .clamp(-5.0, 5.0)
            .reshape([b, t, self.n_kv_head, self.head_dim]);
        let v = self
            .c_v
            .forward(x)
            .clamp(-5.0, 5.0)
            .reshape([b, t, self.n_kv_head, self.head_dim]);

        debug!("Layer {} QKV shapes: Q {:?}, K {:?}, V {:?}", 
               self.layer_idx, q.dims(), k.dims(), v.dims());

        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        let (cos, sin) = cos_sin;
        let q = apply_rotary_emb(q, cos.clone(), sin.clone());
        let k = apply_rotary_emb(k, cos.clone(), sin.clone());
        debug!("Layer {} after RoPE: Q {:?}, K {:?}", self.layer_idx, q.dims(), k.dims());

        // Functional RMSNorm for QK (position-independent, no learned params)
        let q = rms_norm_4d(q);
        let k = rms_norm_4d(k);
        debug!("Layer {} after QK-norm (functional RMSNorm)", self.layer_idx);

        let y = self.scaled_dot_product_attention(q, k, v, t, t);
        debug!("Layer {} attention output: {:?}", self.layer_idx, y.dims());

        let y = y.swap_dims(1, 2).reshape([b, t, c]);
        self.c_proj.forward(y)
    }

    fn scaled_dot_product_attention(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        t_q: usize,
        t_k: usize,
    ) -> Tensor<B, 4> {
        let [b, _hq, _tq, d] = q.dims();
        let h_kv = k.dims()[1];

        let (k, v) = if self.n_head != self.n_kv_head {
            let repeat = self.n_head / self.n_kv_head;
            debug!("Layer {} MQA: repeating KV heads {}x", self.layer_idx, repeat);
            let k5: Tensor<B, 5> = k.unsqueeze_dim::<5>(2);
            let k = k5.expand([b, h_kv, repeat, t_k, d]).reshape([b, self.n_head, t_k, d]);
            let v5: Tensor<B, 5> = v.unsqueeze_dim::<5>(2);
            let v = v5.expand([b, h_kv, repeat, t_k, d]).reshape([b, self.n_head, t_k, d]);
            (k, v)
        } else {
            (k, v)
        };

        let scale = (d as f32).sqrt();
        let mut att = q.matmul(k.swap_dims(2, 3)) / scale;
        debug!("Layer {} attention scores (raw): shape {:?}", self.layer_idx, att.dims());

        // Causal mask with large negative (no max-subtraction for strict causality)
        let mask2: Tensor<B, 2, Bool> = Tensor::tril_mask([t_q, t_k], 0, &att.device());
        let mask4 = mask2.unsqueeze_dims::<4>(&[0, 1]).expand([b, self.n_head, t_q, t_k]);
        att = att.mask_fill(mask4.bool_not(), -1e10);
        debug!("Layer {} after causal mask", self.layer_idx);

        let att = activation::softmax(att, 3);
        debug!("Layer {} after softmax", self.layer_idx);

        att.matmul(v)
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// MLP and Block
// ═════════════════════════════════════════════════════════════════════════════

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    c_fc: Linear<B>,
    c_proj: Linear<B>,
}

impl<B: Backend> Mlp<B> {
    pub fn new(cfg: &NanoChatConfig, device: &B::Device) -> Self {
        let n = cfg.n_embd;
        debug!("MLP init: n_embd={}, hidden={}", n, 4 * n);

        let init = burn::nn::Initializer::KaimingUniform {
            gain: 0.5,
            fan_out_only: false,
        };

        Self {
            c_fc: LinearConfig::new(n, 4 * n)
                .with_bias(false)
                .with_initializer(init.clone())
                .init(device),
            c_proj: LinearConfig::new(4 * n, n)
                .with_bias(false)
                .with_initializer(init)
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        debug!("MLP forward: input shape {:?}", x.dims());
        let x = self.c_fc.forward(x);
        let x = activation::relu(x).powf_scalar(2.0);
        self.c_proj.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    layer_idx: usize,
    attn: CausalSelfAttention<B>,
    mlp: Mlp<B>,
}

impl<B: Backend> Block<B> {
    pub fn new(cfg: &NanoChatConfig, layer_idx: usize, device: &B::Device) -> Self {
        info!("Initializing Block {}", layer_idx);
        Self {
            layer_idx,
            attn: CausalSelfAttention::new(cfg, layer_idx, device),
            mlp: Mlp::new(cfg, device),
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        cos_sin: (&Tensor<B, 4>, &Tensor<B, 4>),
    ) -> Tensor<B, 3> {
        debug!("Block {} forward: input shape {:?}", self.layer_idx, x.dims());
        // Pre-norm with functional RMSNorm
        let x = x.clone() + self.attn.forward(rms_norm_3d(x.clone()), cos_sin);
        x.clone() + self.mlp.forward(rms_norm_3d(x))
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// GPT Model
// ═════════════════════════════════════════════════════════════════════════════

#[derive(Module, Debug)]
pub struct GptModel<B: Backend> {
    wte: Embedding<B>,
    blocks: Vec<Block<B>>,
    lm_head: Linear<B>,
    cos: Tensor<B, 4>,
    sin: Tensor<B, 4>,
}

impl<B: Backend> GptModel<B> {
    pub fn new(cfg: &NanoChatConfig, device: &B::Device) -> Self {
        info!("═══════════════════════════════════════");
        info!("Initializing GptModel (functional RMSNorm)");
        info!("  vocab_size: {}", cfg.vocab_size);
        info!("  n_layer: {}", cfg.n_layer);
        info!("  n_head: {}", cfg.n_head);
        info!("  n_kv_head: {}", cfg.n_kv_head);
        info!("  n_embd: {}", cfg.n_embd);
        info!("  sequence_len: {}", cfg.sequence_len);
        info!("═══════════════════════════════════════");

        let head_dim = cfg.n_embd / cfg.n_head;
        let (cos, sin) = precompute_rotary_embeddings(cfg.sequence_len * 10, head_dim, 10000.0, device);

        info!("Creating embedding layer");
        let wte = EmbeddingConfig::new(cfg.vocab_size, cfg.n_embd).init(device);

        info!("Creating {} transformer blocks", cfg.n_layer);
        let blocks = (0..cfg.n_layer)
            .map(|i| Block::new(cfg, i, device))
            .collect();

        info!("Creating lm_head");
        let lm_head = LinearConfig::new(cfg.n_embd, cfg.vocab_size)
            .with_bias(false)
            .with_initializer(burn::nn::Initializer::KaimingUniform {
                gain: 0.5,
                fan_out_only: false,
            })
            .init(device);

        info!("Model initialization complete");
        Self {
            wte,
            blocks,
            lm_head,
            cos,
            sin,
        }
    }

    #[cfg(test)]
    pub fn forward_no_softcap(&self, idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        debug!("GptModel -> forward NO soft cap");
        self.forward(idx, false)
    }

    pub fn forward(&self, idx: Tensor<B, 2, Int>, use_softcap: bool) -> Tensor<B, 3> {
        let [b, t] = idx.dims();
        assert!(t > 0, "Sequence length must be > 0");
        debug!("GptModel.forward: input [B={}, T={}]", b, t);

        let head_dim = self.cos.dims()[3];
        let cos_slice = self.cos.clone().slice([0..1, 0..t, 0..1, 0..head_dim]);
        let sin_slice = self.sin.clone().slice([0..1, 0..t, 0..1, 0..head_dim]);
        debug!("RoPE slices: cos {:?}, sin {:?}", cos_slice.dims(), sin_slice.dims());

        let mut x = self.wte.forward(idx);
        debug!("After embedding: shape {:?}", x.dims());

        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(x, (&cos_slice, &sin_slice));
            debug!("After block {}: shape {:?}", i, x.dims());
        }

        x = rms_norm_3d(x);
        debug!("After final RMSNorm: shape {:?}", x.dims());

        let mut logits = self.lm_head.forward(x);
        debug!("After lm_head (before clamp): shape {:?}", logits.dims());

        logits = logits.clamp(-50.0, 50.0);

        if use_softcap {
            let softcap = 15.0;
            debug!("Applying softcap={}", softcap);
            logits = logits.clone().div_scalar(softcap).tanh().mul_scalar(softcap);
        }

        logits = logits.clamp(-50.0, 50.0);
        debug!("Final logits shape: {:?}", logits.dims());
        logits
    }

    pub fn generate(&self, mut idx: Tensor<B, 2, Int>, max_new_tokens: usize) -> Tensor<B, 2, Int> {
        let [b, t0] = idx.dims();
        info!("Generation: initial_len={}, max_new={}", t0, max_new_tokens);

        for step in 0..max_new_tokens {
            let logits = self.forward(idx.clone(), true);
            let [b, t, v] = logits.dims();

            if step % 5 == 0 {
                debug!("Generation step {}: seq_len={}", step, t);
            }

            let last = logits.slice([0..b, (t - 1)..t, 0..v]).reshape([b, v]);
            let next = last.argmax(1).reshape([b, 1]);
            idx = Tensor::cat(vec![idx, next], 1);
        }

        info!("Generation complete: final_len={}", idx.dims()[1]);
        idx
    }

    pub fn check_logits_health(logits: &Tensor<B, 3>) -> bool {
        let data = logits.clone().to_data();
        let vec: Vec<f32> = data.to_vec().unwrap();
        let is_healthy = vec.iter().all(|&x| x.is_finite());
        if !is_healthy {
            debug!("⚠️  Logits contain NaN or Inf!");
        }
        is_healthy
    }
}
