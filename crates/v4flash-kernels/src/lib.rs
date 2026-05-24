//! v4flash-kernels — HIP kernels for V4 Flash inference + a per-kernel
//! oracle-based validation framework.
//!
//! Layout:
//! - [`oracle`]   — loads the M2 activation dump tree (manifest + binary blobs)
//!                  produced by `external/ds4-dump/ds4-dump-activations`
//! - [`rms_norm`] — first ported kernel, `rms_norm_weighted`
//!
//! Each ported kernel ships a Rust wrapper around its HIP `.hip` source
//! (compiled to per-arch `.hsaco` by `build.rs`) plus an `#[ignore]`-gated
//! oracle test under `tests/`. The test loads the relevant tag slices
//! from the activation dump and asserts `max_abs_diff < threshold`.

pub mod attention;
pub mod oracle;
pub mod q8_0;
pub mod rms_norm;
pub mod rope;
pub mod weights;

pub use attention::{AttentionMixed, AttentionSwa, ATTN_MIXED_MAX_KEYS, ATTN_SWA_MAX_KV};
pub use oracle::{ActivationDump, Dtype, TensorEntry};
pub use q8_0::{Q8_0GroupedMatvec, Q8_0Matvec};
pub use rms_norm::{RmsNorm, RmsNormNoWeight};
pub use rope::{RopeParams, RopeTail};
pub use weights::{load_to_device, DeviceWeight};
