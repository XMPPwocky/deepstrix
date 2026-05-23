//! v4flash-core — non-GPU model handling.
//!
//! Currently: GGUF v3 parser (no model-architecture knowledge yet). Long
//! term, this crate grows to hold the tensor inventory, tokenizer, and
//! anything else that doesn't directly touch the GPU.

pub mod gguf;

pub use gguf::{Gguf, GgufError, GgufTensor, GgufType, GgufValue};
