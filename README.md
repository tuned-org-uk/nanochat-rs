# nanochat-rs

Karpathy's GPT model nanochat ported to Rust.

## Try it
```
cargo run --release --features wgpu
cargo run --release --features cuda
```

Test model capabilities:
```
cargo run --example check_features --features wgpu -- --nocapture
```