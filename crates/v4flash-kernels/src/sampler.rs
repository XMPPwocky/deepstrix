//! On-device token sampling. Replaces per-token `copy_to_host(logits)`
//! + CPU argmax with kernels that write a single i32 to a device buffer
//! the host reads back (4 bytes vs 517 KB).
//!
//! For V4-Flash, the recommended sampling parameters are
//! `temperature = 1.0`, `top_p = 1.0` (multinomial from raw softmax) —
//! so the default path is two partial-reduce kernels + a chunked-scan
//! sample walk. Argmax mode is kept as a deterministic alternative
//! useful for tests, A/B benchmarks, and `temperature == 0`.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const SOFTMAX_SAMPLE_GFX1201: &[u8] = include_bytes!(env!("KERNEL_SOFTMAX_SAMPLE_GFX1201"));
const SOFTMAX_SAMPLE_GFX1151: &[u8] = include_bytes!(env!("KERNEL_SOFTMAX_SAMPLE_GFX1151"));

/// Number of WGs for the partial-reduce stages. 64 matches the
/// rms_norm_no_weight_multiwg geometry; partials arrays of [N_WG] f32
/// are tiny so this is just "enough WGs to spread across the dGPU".
pub const SAMPLER_N_WG: u32 = 64;

pub struct Sampler {
    module: Module,
}

impl Sampler {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            SOFTMAX_SAMPLE_GFX1201
        } else if arch.starts_with("gfx1151") {
            SOFTMAX_SAMPLE_GFX1151
        } else {
            return Err(eyre!("unsupported arch for sampler kernel: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// Deterministic argmax. Single-WG parallel reduce, ties broken by
    /// lowest index.
    pub fn launch_argmax(
        &self,
        stream: &Stream,
        next_token_out: &mut DeviceBuffer<i32>, // [1]
        logits: &DeviceBuffer<f32>,             // [n]
        n: u32,
    ) -> eyre::Result<()> {
        if next_token_out.len() < 1 || logits.len() < n as usize {
            return Err(eyre!("argmax: buffer too small"));
        }
        let f = self.module.get_function("argmax_one")?;
        let cfg = LaunchConfig {
            grid: (1, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(f, cfg, stream, [next_token_out.raw(), logits.raw(), n])
    }

    /// Multinomial sample from softmax(logits / temperature).
    ///
    /// Three kernels:
    ///   1. logits_max_partial    — per-WG max over (logits * inv_T)
    ///   2. logits_expsum_partial — per-WG sum(exp(x*inv_T - gmax))
    ///   3. softmax_sample_one    — single-WG cumulative-walk picker
    ///
    /// `u01[0]` is the host-supplied uniform sample in [0, 1).
    /// `min_p_rel` is the min-p threshold relative to the most-likely
    /// token (e.g. 0.05 to prune anything <5% of p_max). Use 0.0 for the
    /// official V4-Flash recommendation (no pruning).
    pub fn launch_multinomial(
        &self,
        stream: &Stream,
        next_token_out: &mut DeviceBuffer<i32>, // [1]
        logits: &DeviceBuffer<f32>,             // [n]
        partials_max: &mut DeviceBuffer<f32>,   // [N_WG]
        partials_z: &mut DeviceBuffer<f32>,     // [N_WG]
        u01: &DeviceBuffer<f32>,                // [1]
        n: u32,
        temperature: f32,
        min_p_rel: f32,
    ) -> eyre::Result<()> {
        if temperature <= 0.0 {
            return Err(eyre!(
                "multinomial: temperature must be > 0 (got {temperature}); use argmax for T=0"
            ));
        }
        let n_wg = SAMPLER_N_WG;
        if n % n_wg != 0 {
            return Err(eyre!("multinomial: n={n} not divisible by N_WG={n_wg}"));
        }
        if partials_max.len() < n_wg as usize || partials_z.len() < n_wg as usize {
            return Err(eyre!("multinomial: partials buffer too small"));
        }
        if u01.len() < 1 {
            return Err(eyre!("multinomial: u01 buffer empty"));
        }
        let inv_t = 1.0f32 / temperature;

        let f_max = self.module.get_function("logits_max_partial")?;
        let f_z   = self.module.get_function("logits_expsum_partial")?;
        let f_s   = self.module.get_function("softmax_sample_one")?;

        let cfg_partial = LaunchConfig {
            grid: (n_wg, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let cfg_single = LaunchConfig {
            grid: (1, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        launch_kernel!(f_max, cfg_partial, stream, [
            partials_max.raw(), logits.raw(), n, inv_t
        ])?;
        launch_kernel!(f_z, cfg_partial, stream, [
            partials_z.raw(), partials_max.raw(), logits.raw(), n, n_wg, inv_t
        ])?;
        launch_kernel!(f_s, cfg_single, stream, [
            next_token_out.raw(), logits.raw(),
            partials_max.raw(), partials_z.raw(),
            u01.raw(),
            n, n_wg, inv_t, min_p_rel
        ])
    }
}

/// Tiny host-side PRNG for the per-token uniform `u01`. xoshiro128**
/// is overkill for 1 draw/token but it's stateless-cheap and avoids
/// pulling in the `rand` crate. Seedable for reproducibility.
#[derive(Clone)]
pub struct SamplerRng {
    s: [u32; 4],
}

impl SamplerRng {
    /// Seed from a u64. `seed = 0` picks a deterministic baseline.
    pub fn new(seed: u64) -> Self {
        // SplitMix64 to expand the seed.
        let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut next = || {
            z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            x ^ (x >> 31)
        };
        let a = next();
        let b = next();
        let s = [
            (a & 0xFFFF_FFFF) as u32,
            (a >> 32) as u32,
            (b & 0xFFFF_FFFF) as u32,
            (b >> 32) as u32,
        ];
        // Avoid all-zero state.
        let s = if s == [0; 4] { [0xDEAD_BEEFu32, 1, 2, 3] } else { s };
        Self { s }
    }

    fn next_u32(&mut self) -> u32 {
        // xoshiro128** core.
        let result = self.s[1]
            .wrapping_mul(5)
            .rotate_left(7)
            .wrapping_mul(9);
        let t = self.s[1] << 9;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(11);
        result
    }

    /// Uniform in [0, 1). 24 bits of mantissa precision (f32 native).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / 16_777_216.0)
    }
}
