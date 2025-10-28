//! NanoChat GPT with numerically stable attention and logits
//!
//! Key stability measures:
//! - Stable attention softmax (max-subtraction + large-negative mask)
//! - Robust RoPE broadcasting (align [1,T,1,D/2] → [B,H,T,D/2])
//! - LayerNorm (stable) for QK-norm and block prenorm
//! - Kaiming init with reduced gain, QKV projection clamping
//! - Logits clamped before/after softcap tanh

use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{activation, backend::Backend, Bool, Int, Tensor},
};
use log::{debug, info};

use crate::config::NanoChatConfig;

// ─────────────────────────────────────────────────────────────────────────────
// RoPE
// ─────────────────────────────────────────────────────────────────────────────

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

    // [1, seq_len, 1, D/2]
    let cos: Tensor<B, 3> = cos.unsqueeze_dim::<3>(0);
    let cos: Tensor<B, 4> = cos.unsqueeze_dim::<4>(2);

    let sin: Tensor<B, 3> = sin.unsqueeze_dim::<3>(0);
    let sin: Tensor<B, 4> = sin.unsqueeze_dim::<4>(2);

    debug!("RoPE cos/sin shape: {:?}", cos.dims());
    (cos, sin)
}

fn apply_rotary_emb<B: Backend>(
    x: Tensor<B, 4>,           // [B, H, T, D]
    cos: Tensor<B, 4>,         // [1, T, 1, D/2]
    sin: Tensor<B, 4>,         // [1, T, 1, D/2]
) -> Tensor<B, 4> {
    let [b, h, t, d] = x.dims();
    let d_half = d / 2;
    debug!("apply_rotary_emb: x shape [B={}, H={}, T={}, D={}]", b, h, t, d);

    let x1 = x.clone().slice([0..b, 0..h, 0..t, 0..d_half]);
    let x2 = x.slice([0..b, 0..h, 0..t, d_half..d]);

    // Slice to current time, move time to axis 2, then expand
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

// ─────────────────────────────────────────────────────────────────────────────
// Attention (LayerNorm-based QK-norm for stability)
// ─────────────────────────────────────────────────────────────────────────────

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
    qk_norm_q: LayerNorm<B>,
    qk_norm_k: LayerNorm<B>,
}

impl<B: Backend> CausalSelfAttention<B> {
    pub fn new(config: &NanoChatConfig, layer_idx: usize, device: &B::Device) -> Self {
        let n_head = config.n_head;
        let n_kv_head = config.n_kv_head;
        let n_embd = config.n_embd;
        let head_dim = n_embd / n_head;

        assert_eq!(n_embd % n_head, 0, "n_embd must be divisible by n_head");
        assert!(
            n_kv_head <= n_head && n_head % n_kv_head == 0,
            "Invalid MQA config"
        );

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
            qk_norm_q: LayerNormConfig::new(head_dim).init(device),
            qk_norm_k: LayerNormConfig::new(head_dim).init(device),
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,                         // [B, T, C]
        cos_sin: (&Tensor<B, 4>, &Tensor<B, 4>),
    ) -> Tensor<B, 3> {
        let [b, t, c] = x.dims();
        debug!("Layer {} attn forward: input [B={}, T={}, C={}]", self.layer_idx, b, t, c);

        // Projections with clipping
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

        // [B, H, T, D]
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        // RoPE
        let (cos, sin) = cos_sin;
        let q = apply_rotary_emb(q, cos.clone(), sin.clone());
        let k = apply_rotary_emb(k, cos.clone(), sin.clone());
        debug!("Layer {} after RoPE: Q {:?}, K {:?}", self.layer_idx, q.dims(), k.dims());

        // QK-norm via LayerNorm over D
        let q = self.norm_heads(q, &self.qk_norm_q);
        let k = self.norm_heads(k, &self.qk_norm_k);
        debug!("Layer {} after QK-norm", self.layer_idx);

        // Attention
        let y = self.scaled_dot_product_attention(q, k, v, t, t);
        debug!("Layer {} attention output: {:?}", self.layer_idx, y.dims());

        // [B, T, C]
        let y = y.swap_dims(1, 2).reshape([b, t, c]);
        self.c_proj.forward(y)
    }

    fn norm_heads(&self, x: Tensor<B, 4>, ln: &LayerNorm<B>) -> Tensor<B, 4> {
        let [b, h, t, d] = x.dims();
        let x_flat = x.reshape([b * h * t, d]);
        let out = ln.forward(x_flat);
        out.reshape([b, h, t, d])
    }

    fn scaled_dot_product_attention(
        &self,
        q: Tensor<B, 4>, // [B, H, Tq, D]
        k: Tensor<B, 4>, // [B, H, Tk, D]
        v: Tensor<B, 4>, // [B, H, Tk, D]
        t_q: usize,
        t_k: usize,
    ) -> Tensor<B, 4> {
        let [b, _hq, _tq, d] = q.dims();
        let h_kv = k.dims()[1];

        // MQA repeat if needed
        let (k, v) = if self.n_head != self.n_kv_head {
            let repeat = self.n_head / self.n_kv_head;
            let k5: Tensor<B, 5> = k.unsqueeze_dim::<5>(2);
            let k = k5.expand([b, h_kv, repeat, t_k, d]).reshape([b, self.n_head, t_k, d]);
            let v5: Tensor<B, 5> = v.unsqueeze_dim::<5>(2);
            let v = v5.expand([b, h_kv, repeat, t_k, d]).reshape([b, self.n_head, t_k, d]);
            (k, v)
        } else {
            (k, v)
        };

        // Compute raw attention scores
        let scale = (d as f32).sqrt();
        let mut att = q.matmul(k.swap_dims(2, 3)) / scale;   // [B, H, Tq, Tk]

        // 1) Apply causal mask FIRST with large negative
        let mask2: Tensor<B, 2, Bool> = Tensor::tril_mask([t_q, t_k], 0, &att.device());
        let mask4 = mask2.unsqueeze_dims::<4>(&[0, 1]).expand([b, self.n_head, t_q, t_k]);
        att = att.mask_fill(mask4.bool_not(), -1.0e9);

        // 2) Subtract per-row max along keys axis (dimension 3)
        // max_dim(3) returns [B, H, Tq, 1]; squeeze to [B, H, Tq], then unsqueeze back
        let att_max = att.clone().max_dim(3).squeeze::<3>(3);        // [B, H, Tq]
        att = att - att_max.unsqueeze_dim::<4>(3);                   // [B, H, Tq, Tk]

        // 3) Softmax
        let att = activation::softmax(att, 3);

        // 4) Weighted sum
        att.matmul(v)  // [B, H, Tq, D]
    }

    // Decode-time forward: Tq = 1, appends K/V to cache for this layer and attends to full past.
    pub fn forward_decode(
        &self,
        x_step: Tensor<B, 3>,                         // [B, 1, C]
        cos_sin_step: (&Tensor<B, 4>, &Tensor<B, 4>), // [1,1,1,D/2]
        cache_layer: &mut Option<(Tensor<B, 4>, Tensor<B, 4>)>, // K/V store for this layer
    ) -> Tensor<B, 3> {
        let [b, tq, c] = x_step.dims();
        debug_assert_eq!(tq, 1, "forward_decode expects T=1 input");

        // Project Q,K,V then reshape
        let q = self
            .c_q
            .forward(x_step.clone())
            .clamp(-5.0, 5.0)
            .reshape([b, 1, self.n_head, self.head_dim])
            .swap_dims(1, 2); // [B, Hq, 1, D]

        let k_new = self
            .c_k
            .forward(x_step.clone())
            .clamp(-5.0, 5.0)
            .reshape([b, 1, self.n_kv_head, self.head_dim])
            .swap_dims(1, 2); // [B, Hkv, 1, D]

        let v_new = self
            .c_v
            .forward(x_step)
            .clamp(-5.0, 5.0)
            .reshape([b, 1, self.n_kv_head, self.head_dim])
            .swap_dims(1, 2); // [B, Hkv, 1, D]

        // Apply RoPE to Q and to K_new
        let (cos_step, sin_step) = cos_sin_step;
        let q = apply_rotary_emb(q, cos_step.clone(), sin_step.clone());
        let k_new = apply_rotary_emb(k_new, cos_step.clone(), sin_step.clone());

        // QK norm via LayerNorm over D
        let q = self.norm_heads(q, &self.qk_norm_q);
        let k_new = self.norm_heads(k_new, &self.qk_norm_k);

        // Append new K,V into cache (time concat on dim=2)
        let (k_full, v_full): (Tensor<B, 4>, Tensor<B, 4>) = match cache_layer.take() {
            Some((k_all, v_all)) => {
                // concat on time axis=2
                let k_cat = Tensor::cat(vec![k_all, k_new.clone()], 2);
                let v_cat = Tensor::cat(vec![v_all, v_new.clone()], 2);
                (k_cat, v_cat)
            }
            None => (k_new.clone(), v_new.clone()),
        };

        // Store back into the cache layer
        *cache_layer = Some((k_full.clone(), v_full.clone()));

        // Use full K,V from cache for attention
        let tk = k_full.dims()[2];
        let y = self.scaled_dot_product_attention(q, k_full, v_full, 1, tk);

        // Merge heads to [B,1,C] then project
        let y = y.swap_dims(1, 2).reshape([b, 1, c]);
        self.c_proj.forward(y)
    }

}

// ─────────────────────────────────────────────────────────────────────────────
// MLP (ReLU²) and Block (pre-norm)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    c_fc: Linear<B>,
    c_proj: Linear<B>,
}

impl<B: Backend> Mlp<B> {
    pub fn new(cfg: &NanoChatConfig, device: &B::Device) -> Self {
        let n = cfg.n_embd;
        debug!("MLP init: n_embd={}, hidden=4*n_embd={}", n, 4 * n);
        
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
    ln1: LayerNorm<B>,
    attn: CausalSelfAttention<B>,
    ln2: LayerNorm<B>,
    mlp: Mlp<B>,
}

impl<B: Backend> Block<B> {
    pub fn new(cfg: &NanoChatConfig, layer_idx: usize, device: &B::Device) -> Self {
        info!("Initializing Block {}", layer_idx);
        Self {
            layer_idx,
            ln1: LayerNormConfig::new(cfg.n_embd).init(device),
            attn: CausalSelfAttention::new(cfg, layer_idx, device),
            ln2: LayerNormConfig::new(cfg.n_embd).init(device),
            mlp: Mlp::new(cfg, device),
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        cos_sin: (&Tensor<B, 4>, &Tensor<B, 4>),
    ) -> Tensor<B, 3> {
        debug!("Block {} forward: input shape {:?}", self.layer_idx, x.dims());
        let x = x.clone() + self.attn.forward(self.ln1.forward(x.clone()), cos_sin);
        x.clone() + self.mlp.forward(self.ln2.forward(x))
    }

    pub fn forward_decode(
        &self,
        x_step: Tensor<B, 3>,                         // [B,1,C]
        cos_sin_step: (&Tensor<B, 4>, &Tensor<B, 4>), // [1,1,1,D/2]
        cache_layer: &mut Option<(Tensor<B, 4>, Tensor<B, 4>)>,
    ) -> Tensor<B, 3> {
        let x = x_step.clone() + self
            .attn
            .forward_decode(self.ln1.forward(x_step.clone()), cos_sin_step, cache_layer);
        x.clone() + self.mlp.forward(self.ln2.forward(x))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GPT
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct GptModel<B: Backend> {
    wte: Embedding<B>,
    blocks: Vec<Block<B>>,
    ln_f: LayerNorm<B>,
    lm_head: Linear<B>,
    cos: Tensor<B, 4>,
    sin: Tensor<B, 4>,
    n_embd: usize,
}

impl<B: Backend> GptModel<B> {
    pub fn new(cfg: &NanoChatConfig, device: &B::Device) -> Self {
        info!("═══════════════════════════════════════");
        info!("Initializing GptModel (stable version)");
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
        
        info!("Creating final LayerNorm and lm_head");
        let ln_f = LayerNormConfig::new(cfg.n_embd).init(device);
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
            ln_f,
            lm_head,
            cos,
            sin,
            n_embd: cfg.n_embd,
        }
    }

    pub fn forward(&self, idx: Tensor<B, 2, Int>, use_softcap: bool) -> Tensor<B, 3> {
        let [b, t] = idx.dims();
        assert!(t > 0, "Sequence length must be > 0");
        debug!("GptModel.forward: input [B={}, T={}]", b, t);

        let head_dim = self.cos.dims()[3];
        let cos_slice = self.cos.clone().slice([0..1, 0..t, 0..1, 0..head_dim]);
        let sin_slice = self.sin.clone().slice([0..1, 0..t, 0..1, 0..head_dim]);
        debug!("RoPE slices: cos {:?}, sin {:?}", cos_slice.dims(), sin_slice.dims());

        // Embed
        let mut x = self.wte.forward(idx);
        debug!("After embedding: shape {:?}", x.dims());

        // Blocks
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(x, (&cos_slice, &sin_slice));
            debug!("After block {}: shape {:?}", i, x.dims());
        }

        // Final norm
        x = self.ln_f.forward(x);
        debug!("After final LayerNorm: shape {:?}", x.dims());

        // Head → logits
        let mut logits = self.lm_head.forward(x);
        debug!("After lm_head (before clamp): shape {:?}", logits.dims());

        // Safety clamp before softcap
        logits = logits.clamp(-50.0, 50.0);

        // Softcap
        if use_softcap {
            let softcap = 15.0;
            debug!("Applying softcap={}", softcap);
            logits = logits.clone().div_scalar(softcap).tanh().mul_scalar(softcap);
        }

        // Final clamp
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

    // Test-only: no softcap
    #[cfg(test)]
    pub fn forward_no_softcap(&self, idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        debug!("GptModel -> forward NO soft cap");
        self.forward(idx, false)
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

        // Number of blocks (for KVCache sizing)
    pub fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    // Decode one step using KV cache: last_ids [B,1] -> logits [B,1,V]
    pub fn forward_decode(
        &self,
        last_ids: Tensor<B, 2, Int>, // [B,1]
        cache: &mut crate::engine::KVCache<B>,
        use_softcap: bool,
    ) -> Tensor<B, 3> {
        let [b, tq] = last_ids.dims();
        debug_assert_eq!(tq, 1);

        // Determine current time position from cache
        let t_pos = cache.position();

        // Slice RoPE for current position: [1,1,1,D/2]
        let head_dim = self.cos.dims()[3];
        let cos_step = self
            .cos
            .clone()
            .slice([0..1, t_pos..(t_pos + 1), 0..1, 0..head_dim]);
        let sin_step = self
            .sin
            .clone()
            .slice([0..1, t_pos..(t_pos + 1), 0..1, 0..head_dim]);

        // Embed last token
        let mut x = self.wte.forward(last_ids).reshape([b, 1, self.n_embd]);

        // One-step through blocks with KV cache
        for (i, block) in self.blocks.iter().enumerate() {
            let layer_cache = &mut cache.store[i];
            x = block.forward_decode(x, (&cos_step, &sin_step), layer_cache);
        }

        // Final LayerNorm and head
        x = self.ln_f.forward(x);
        let mut logits = self.lm_head.forward(x);

        // Stability clamps and optional softcap
        logits = logits.clamp(-50.0, 50.0);
        if use_softcap {
            let softcap = 15.0;
            logits = logits.clone().div_scalar(softcap).tanh().mul_scalar(softcap);
        }
        logits = logits.clamp(-50.0, 50.0);

        logits
    }
}
