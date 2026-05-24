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

pub mod oracle;
pub mod rms_norm;

pub use oracle::{ActivationDump, Dtype, TensorEntry};
pub use rms_norm::RmsNorm;
