# Tiny Chat-Bot from Scratch (C++)

A **tiny chat-bot** built **from scratch in C++** — no PyTorch, no TensorFlow, no external ML libraries. Everything is hand-written: custom `Tensor`, `Linear`, `RMSNorm`, `RoPE`, `SwiGLU`, `AdamW`, custom BPE tokenizer, and manual backward pass.

> **This is a learning project, not a product.** The model is ~1.1M parameters — far too small to be called an LLM. The goal is to understand every line of the pipeline, especially the manual backward for RoPE / RMSNorm / SwiGLU. Success criterion: the bot should **memorize the dialogue dataset** and produce **coherent (if repetitive) answers on seen prompts**.

---

## Status

- ✅ Architecture: Transformer, 5 layers, `dim=192`, 6 heads, `head_dim=32`, RoPE, RMSNorm, SwiGLU
- ✅ Custom BPE tokenizer (vocab=2000)
- ✅ Manual backward for all layers
- ✅ AdamW with warmup
- ✅ Pretrain + finetune + chat pipeline
- ⚠️ **Known issue:** loss drops to ~1.0, but generation is gibberish. See "Known Issue" below.

---

## Architecture

| Component | Details |
|-----------|---------|
| Model | Transformer, **5 layers**, `dim=192`, `6 heads`, `head_dim=32` |
| Parameters | **~1.1M** |
| Positional encoding | **RoPE** (rotary), applied to Q and K only |
| Normalization | **RMSNorm** (pre-norm) |
| FFN | **SwiGLU**, `hidden = 4 * dim` |
| Tokenizer | Custom **BPE**, `vocab=2000`, trained on 500K chars |
| Optimizer | **AdamW**, warmup (100 steps pretrain, 50 finetune) |
| Loss | **Cross-entropy**, masked on answer tokens (finetune) |
| Context | `BLOCK_SIZE = 64` |
| Batch | `BATCH_SIZE = 2` |

---

## Pipeline
