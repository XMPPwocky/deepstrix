//! v4flash-core — non-GPU model handling.
//!
//! Currently: GGUF v3 parser (no model-architecture knowledge yet). Long
//! term, this crate grows to hold the tensor inventory, tokenizer, and
//! anything else that doesn't directly touch the GPU.

pub mod engram_hash;
pub mod engram_table;
pub mod gguf;
pub mod hf_v41;
pub mod heap;
pub mod iq3_s_ref;
pub mod kquants;
pub mod mapped;
pub mod safetensors;
pub mod tokenizer;
pub mod weight_src;

pub use gguf::{Gguf, GgufError, GgufTensor, GgufType, GgufValue};
pub use engram_hash::EngramHash;
pub use engram_table::EngramTable;
pub use hf_v41::V41HfWeights;
pub use mapped::MappedGguf;
pub use safetensors::SafetensorsDir;
pub use weight_src::{ModelShape, WeightSrc};
pub use tokenizer::BpeVocab;
