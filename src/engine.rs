use burn::tensor::{backend::Backend, Int, Tensor};
use log::{debug, info};

use crate::gpt::GptModel;

// Simple per-layer KV cache.
// Shapes:
//   K: [B, H_kv, T_total, D]
//   V: [B, H_kv, T_total, D]
pub struct KVCache<B: Backend> {
    pub(crate) store: Vec<Option<(Tensor<B, 4>, Tensor<B, 4>)>>,
    t_pos: usize,
}

impl<B: Backend> KVCache<B> {
    pub fn new(n_layer: usize) -> Self {
        info!("KVCache: initializing with {} layers", n_layer);
        Self {
            store: vec![None; n_layer],
            t_pos: 0,
        }
    }

    pub fn clear(&mut self) {
        info!("KVCache: clearing all layers and resetting position");
        for (i, slot) in self.store.iter_mut().enumerate() {
            if slot.is_some() {
                debug!("KVCache: clearing layer {}", i);
            }
            *slot = None;
        }
        self.t_pos = 0;
    }

    pub fn position(&self) -> usize {
        self.t_pos
    }

    pub fn get(&self, layer_idx: usize) -> Option<&(Tensor<B, 4>, Tensor<B, 4>)> {
        self.store[layer_idx].as_ref()
    }

    pub fn append_step(
        &mut self,
        layer_idx: usize,
        k_step: Tensor<B, 4>, // [B, H_kv, 1, D]
        v_step: Tensor<B, 4>, // [B, H_kv, 1, D]
    ) {
        let [b, h, t_new, d] = k_step.dims();
        debug!(
            "KVCache: append_step layer={} step_dims(K)=[B={},H={},T={},D={}] t_pos={}",
            layer_idx, b, h, t_new, d, self.t_pos
        );

        match &mut self.store[layer_idx] {
            Some((k_all, v_all)) => {
                let t_prev = k_all.dims()[2];
                debug!(
                    "KVCache: concatenating existing K/V at layer {} (prev T={} -> new T={})",
                    layer_idx,
                    t_prev,
                    t_prev + t_new
                );
                // concat on time axis=2
                let new_k = Tensor::cat(vec![k_all.clone(), k_step], 2);
                let new_v = Tensor::cat(vec![v_all.clone(), v_step], 2);
                *k_all = new_k;
                *v_all = new_v;
            }
            slot @ None => {
                debug!(
                    "KVCache: initializing layer {} with first K/V chunk (T={})",
                    layer_idx, t_new
                );
                *slot = Some((k_step, v_step));
            }
        }
    }

    pub fn advance(&mut self) {
        self.t_pos += 1;
        debug!("KVCache: advanced position to t_pos={}", self.t_pos);
    }
}

// Minimal inference engine to run prefill + decode streaming without changing model API.
pub struct Engine<B: Backend> {
    model: GptModel<B>,
    _device: B::Device,
}

impl<B: Backend> Engine<B> {
    pub fn new(model: GptModel<B>, device: B::Device) -> Self {
        info!("Engine: new with {} layers", model.num_layers());
        Self { model, device }
    }

    // Prefill runs full forward once to compute logits (no cache used).
    // Caller is responsible for starting decode at last token.
    pub fn prefill(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [b, t] = input_ids.dims();
        info!("Engine: prefill forward [B={},T={}]", b, t);
        let logits = self.model.forward(input_ids, true);
        debug!("Engine: prefill logits shape {:?}", logits.dims());
        logits
    }

    // Decode next token given last token id and existing cache.
    // Returns logits for that step [B, 1, V].
    pub fn decode_next(
        &self,
        last_id: Tensor<B, 2, Int>, // [B, 1]
        cache: &mut KVCache<B>,
    ) -> Tensor<B, 3> {
        let [b, t] = last_id.dims();
        debug_assert_eq!(t, 1, "decode_next expects [B,1], got T={}", t);
        info!("Engine: decode_next at t_pos={} [B={},T=1]", cache.position(), b);
        let logits = self.model.forward_decode(last_id, cache, true);
        debug!("Engine: decode_next logits_step shape {:?}", logits.dims());
        logits
    }

    pub fn stream<'a>(
        &'a self,
        ids: Tensor<B, 2, Int>,
        max_new_tokens: usize,
    ) -> Streamer<'a, B> {
        let [b, t] = ids.dims();
        info!(
            "Engine: streaming start [B={},T0={}] max_new_tokens={}",
            b, t, max_new_tokens
        );
        let cache = KVCache::<B>::new(self.model.num_layers());
        Streamer {
            engine: self,
            ids: Some(ids),
            cache,
            steps_left: max_new_tokens,
            finished: false,
        }
    }
}

// Streaming iterator that yields one token id [B,1] each step.
pub struct Streamer<'a, B: Backend> {
    engine: &'a Engine<B>,
    ids: Option<Tensor<B, 2, Int>>,
    cache: KVCache<B>,
    steps_left: usize,
    finished: bool,
}

impl<'a, B: Backend> Iterator for Streamer<'a, B> {
    type Item = Tensor<B, 2, Int>; // [B, 1]

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished || self.steps_left == 0 {
            info!(
                "Streamer: finished (steps_left={}, finished={})",
                self.steps_left, self.finished
            );
            self.finished = true;
            return None;
        }
        let ids = self.ids.as_ref().unwrap();

        // Take the last token id [B,1]
        let [b, t] = ids.dims();
        debug!("Streamer: step begin [B={},T_current={}]", b, t);
        let last_id = ids.clone().slice([0..b, (t - 1)..t]);

        // Decode next logits [B,1,V]
        let logits_step = self.engine.decode_next(last_id.clone(), &mut self.cache);
        let [_, _, v] = logits_step.dims();
        debug!("Streamer: logits_step [B={},T=1,V={}]", b, v);

        // Greedy next token
        let next = logits_step
            .reshape([b, v])
            .argmax(1)
            .reshape([b, 1]);

        // Append to ids
        let new_ids = Tensor::cat(vec![ids.clone(), next.clone()], 1);
        self.ids = Some(new_ids);
        self.cache.advance();
        self.steps_left -= 1;

        info!(
            "Streamer: emitted next token, steps_left={}, t_pos={}",
            self.steps_left,
            self.cache.position()
        );
        Some(next)
    }
}
