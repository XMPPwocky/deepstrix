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

/// Number of equal log-space sub-intervals the top-p threshold search
/// splits its bracket into per level. Must match `TOPP_NBINS` in
/// `kernels/softmax_sample.hip`.
pub const SAMPLER_TOPP_NBINS: u32 = 32;
/// Edges evaluated per level (`NBINS + 1`); also the per-WG stride of the
/// `topp_mass_partial` output.
pub const SAMPLER_TOPP_NEDGE: u32 = SAMPLER_TOPP_NBINS + 1;
/// Refinement levels. Final bracket width in log space is
/// `SAMPLER_TOPP_LOG_RANGE / NBINS^LEVELS` = 40 / 32^4 = 3.8e-5, i.e. the
/// cutoff probability is resolved to a factor of 1.0000382.
pub const SAMPLER_TOPP_LEVELS: u32 = 4;
/// Initial search bracket is `[-LOG_RANGE, 0]` in log space, i.e. tokens
/// whose probability is below `exp(-40) = 4.2e-18` of the most likely one
/// are never admitted to the nucleus. Their combined mass is under 5.4e-13
/// of Z, so this only matters for `top_p > 1 - 5.4e-13`.
/// Must match `TOPP_LOG_RANGE` in `kernels/softmax_sample.hip`, which the
/// search kernels use to seed the bracket at level 0.
pub const SAMPLER_TOPP_LOG_RANGE: f32 = 40.0;

/// Elements the `topp_mass` scratch buffer must hold.
pub const fn sampler_topp_mass_len() -> usize {
    (SAMPLER_N_WG * SAMPLER_TOPP_NEDGE) as usize
}

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

    /// [`Self::launch_argmax`] over `b` rows of logits (`[b, n]`) into
    /// `next_token_out[0..b]` in one launch — the multi-stream decode sampler.
    /// Same kernel body per row (`argmax_rows`), so each row is bit-identical to
    /// `argmax_one` on its slice.
    pub fn launch_argmax_rows(
        &self,
        stream: &Stream,
        next_token_out: &mut DeviceBuffer<i32>, // [b]
        logits: &DeviceBuffer<f32>,             // [b, n]
        n: u32,
        b: u32,
    ) -> eyre::Result<()> {
        if b == 0 {
            return Ok(());
        }
        if next_token_out.len() < b as usize || logits.len() < (b as usize) * (n as usize) {
            return Err(eyre!("argmax_rows: buffer too small (b={b}, n={n})"));
        }
        let f = self.module.get_function("argmax_rows")?;
        let cfg = LaunchConfig {
            grid: (1, b, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(f, cfg, stream, [next_token_out.raw(), logits.raw(), n])
    }

    /// Multinomial sample from softmax(logits / temperature).
    ///
    /// Three kernels:
    ///   1. logits_max_partial    — per-WG max over (logits * inv_T)
    ///   2. logits_expsum_partial — per-WG sum(exp(x*inv_T - gmax)),
    ///      excluding tokens below `min_p_rel` so Z renormalises over the
    ///      surviving set (ds4.c sample_full_vocab)
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
            partials_z.raw(), partials_max.raw(), logits.raw(), n, n_wg, inv_t, min_p_rel,
            NULL_F32
        ])?;
        launch_kernel!(f_s, cfg_single, stream, [
            next_token_out.raw(), logits.raw(),
            partials_max.raw(), partials_z.raw(),
            u01.raw(),
            n, n_wg, inv_t, min_p_rel,
            NULL_F32
        ])
    }

    /// Multinomial sample with top-p (nucleus) truncation composed on top of
    /// min-p, in OpenAI/vLLM ordering: TEMPERATURE first, then `top_p` over
    /// the full (tempered) distribution, then `min_p`, then renormalise over
    /// the survivors.
    ///
    /// NOTE: llama.cpp's *default* sampler chain runs the temperature sampler
    /// LAST, so its nucleus is taken on the un-tempered `T = 1`
    /// probabilities. The two orderings coincide exactly at
    /// `temperature == 1.0` (the server default) and diverge otherwise; this
    /// implementation follows the OpenAI-compatible ordering on purpose.
    ///
    /// `top_p >= 1.0` is a no-op: it delegates verbatim to
    /// [`Self::launch_multinomial`], so the three-kernel legacy chain runs
    /// with byte-identical arguments and produces bit-identical tokens.
    ///
    /// Otherwise the chain is
    ///   1. `logits_max_partial`                        — gmax
    ///   2. `logits_expsum_partial(min_p_rel = 0)`      — Z over the full vocab
    ///   3. `topp_mass_partial` + `topp_bracket_step`   — x LEVELS, the
    ///      log-space threshold search (see the kernel file for the proof and
    ///      the convergence bound); publishes `thr = max(t*, min_p_rel)`.
    ///      Both kernels take the level index and seed the bracket to
    ///      `[-SAMPLER_TOPP_LOG_RANGE, 0]` themselves at level 0, so the
    ///      chain is enqueued without a host round trip.
    ///   4. `logits_expsum_partial(thr)`                — Z over the nucleus
    ///   5. `softmax_sample_one(thr)`                   — cumulative walk
    ///
    /// Extra cost over the legacy chain: `LEVELS` full-vocabulary passes plus
    /// one extra exp-sum pass and `LEVELS` single-WG reductions.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_multinomial_topp(
        &self,
        stream: &Stream,
        next_token_out: &mut DeviceBuffer<i32>, // [1]
        logits: &DeviceBuffer<f32>,             // [n]
        partials_max: &mut DeviceBuffer<f32>,   // [N_WG]
        partials_z: &mut DeviceBuffer<f32>,     // [N_WG]
        topp_mass: &mut DeviceBuffer<f32>,      // [N_WG * TOPP_NEDGE]
        topp_bracket: &mut DeviceBuffer<f32>,   // [2]
        topp_thr: &mut DeviceBuffer<f32>,       // [1]
        u01: &DeviceBuffer<f32>,                // [1]
        n: u32,
        temperature: f32,
        min_p_rel: f32,
        top_p: f32,
    ) -> eyre::Result<()> {
        if top_p.is_nan() || top_p <= 0.0 || top_p > 1.0 {
            return Err(eyre!(
                "multinomial: top_p must be in (0, 1] (got {top_p})"
            ));
        }
        if top_p >= 1.0 {
            // Exact legacy path — no extra kernels, no extra arithmetic.
            return self.launch_multinomial(
                stream,
                next_token_out,
                logits,
                partials_max,
                partials_z,
                u01,
                n,
                temperature,
                min_p_rel,
            );
        }
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
        if topp_mass.len() < sampler_topp_mass_len() {
            return Err(eyre!(
                "multinomial: topp_mass buffer too small ({} < {})",
                topp_mass.len(),
                sampler_topp_mass_len()
            ));
        }
        if topp_bracket.len() < 2 || topp_thr.len() < 1 || u01.len() < 1 {
            return Err(eyre!("multinomial: top-p scratch buffer too small"));
        }
        let inv_t = 1.0f32 / temperature;

        let f_max = self.module.get_function("logits_max_partial")?;
        let f_z = self.module.get_function("logits_expsum_partial")?;
        let f_mass = self.module.get_function("topp_mass_partial")?;
        let f_step = self.module.get_function("topp_bracket_step")?;
        let f_s = self.module.get_function("softmax_sample_one")?;

        let cfg_partial = LaunchConfig {
            grid: (n_wg, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let cfg_step = LaunchConfig {
            grid: (1, 1, 1),
            block: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        let cfg_single = LaunchConfig {
            grid: (1, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        // 1. global max.
        launch_kernel!(f_max, cfg_partial, stream, [
            partials_max.raw(), logits.raw(), n, inv_t
        ])?;
        // 2. Z over the FULL vocabulary — top_p's cumulative is taken over the
        //    untruncated (but temperature-scaled) distribution; min_p is
        //    applied afterwards, in step 4.
        launch_kernel!(f_z, cfg_partial, stream, [
            partials_z.raw(), partials_max.raw(), logits.raw(), n, n_wg, inv_t, 0.0f32,
            NULL_F32
        ])?;
        // 3. bracket the cutoff in log space and refine it. `level` is passed
        //    to both kernels so level 0 seeds [-LOG_RANGE, 0] on the device:
        //    a host-side seed upload would be a blocking null-stream
        //    `hipMemcpy` in the middle of the chain (the compute stream is a
        //    legacy blocking stream), draining the device once per token.
        for level in 0..SAMPLER_TOPP_LEVELS {
            launch_kernel!(f_mass, cfg_partial, stream, [
                topp_mass.raw(), partials_max.raw(), logits.raw(), topp_bracket.raw(),
                n, n_wg, inv_t, level
            ])?;
            launch_kernel!(f_step, cfg_step, stream, [
                topp_bracket.raw(), topp_thr.raw(), topp_mass.raw(), partials_z.raw(),
                n_wg, top_p, min_p_rel, level
            ])?;
        }
        // 4. Z renormalised over the surviving set.
        launch_kernel!(f_z, cfg_partial, stream, [
            partials_z.raw(), partials_max.raw(), logits.raw(), n, n_wg, inv_t, min_p_rel,
            topp_thr.raw()
        ])?;
        // 5. cumulative walk restricted to the same set.
        launch_kernel!(f_s, cfg_single, stream, [
            next_token_out.raw(), logits.raw(),
            partials_max.raw(), partials_z.raw(),
            u01.raw(),
            n, n_wg, inv_t, min_p_rel,
            topp_thr.raw()
        ])
    }
}

/// `nullptr` for the optional `const float *thr_dev` kernel parameter.
/// Kept as a `const` so every `launch_kernel!` site stores a value of the
/// same ABI type as `DeviceBuffer::raw()`.
const NULL_F32: v4flash_hip::sys::hipDeviceptr_t = std::ptr::null_mut();

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

/// Host reference for the nucleus cutoff, in the same weight space the
/// kernels use: `weights[i] = exp(logit_i * inv_T - gmax)`, so the most
/// likely token has weight `1.0`.
///
/// Returns the cutoff weight `t*` such that the nucleus is
/// `{ i : weights[i] >= t* }`:
///
/// * sort descending, accumulate, and take the first index `k` whose
///   inclusive cumulative mass reaches `top_p * Z`; `t* = weights[k]`.
/// * **Ties**: because the answer is a threshold, every token sharing the
///   boundary weight is kept. A sorting implementation would cut after
///   exactly `k + 1` tokens and drop the rest of the tied group in whatever
///   order the sort happened to produce; keeping them all is a superset that
///   still satisfies "cumulative >= top_p" and is order-independent.
/// * `top_p >= 1.0` returns `0.0` (no truncation); `top_p <= 0` returns the
///   maximum weight (greedy).
///
/// This is the specification the GPU search converges to — see
/// `kernels/softmax_sample.hip`.
pub fn top_p_cutoff(weights: &[f64], top_p: f64) -> f64 {
    if weights.is_empty() {
        return 0.0;
    }
    if top_p >= 1.0 {
        return 0.0;
    }
    let mut sorted: Vec<f64> = weights.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    if top_p <= 0.0 {
        return sorted[0];
    }
    let z: f64 = sorted.iter().sum();
    let target = top_p * z;
    let mut cum = 0.0f64;
    for &w in &sorted {
        cum += w;
        if cum >= target {
            return w;
        }
    }
    *sorted.last().unwrap()
}

/// The single threshold that top-p and min-p compose into.
///
/// The chain is OpenAI/vLLM ordering — temperature, then `top_p` over the
/// full (already temperature-scaled) distribution, then `min_p`. `min_p` is
/// itself the rule `weight >= min_p_rel`, so the intersection of the two
/// filters is `weight >= max(t*, min_p_rel)`. The surviving mass is then
/// renormalised over exactly that set.
///
/// (`weights` are expected to be the tempered weights
/// `exp(logit/T - gmax)`. llama.cpp's default chain instead applies
/// temperature last, so it truncates the `T = 1` distribution; the two agree
/// at `T = 1` and diverge otherwise.)
pub fn top_p_min_p_threshold(weights: &[f64], top_p: f64, min_p_rel: f64) -> f64 {
    top_p_cutoff(weights, top_p).max(min_p_rel)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Weights normalised into probabilities, then the nucleus by threshold.
    fn nucleus(weights: &[f64], top_p: f64) -> Vec<usize> {
        let t = top_p_cutoff(weights, top_p);
        (0..weights.len()).filter(|&i| weights[i] >= t).collect()
    }

    fn mass(weights: &[f64], set: &[usize]) -> f64 {
        let z: f64 = weights.iter().sum();
        set.iter().map(|&i| weights[i]).sum::<f64>() / z
    }

    #[test]
    fn top_p_one_is_no_truncation() {
        let w = [1.0, 0.5, 0.25, 0.125];
        assert_eq!(top_p_cutoff(&w, 1.0), 0.0);
        assert_eq!(nucleus(&w, 1.0), vec![0, 1, 2, 3]);
    }

    #[test]
    fn top_p_zero_is_greedy() {
        let w = [0.25, 1.0, 0.5];
        assert_eq!(top_p_cutoff(&w, 0.0), 1.0);
        assert_eq!(nucleus(&w, 0.0), vec![1]);
    }

    #[test]
    fn single_dominant_token() {
        // p = [0.98, 0.01, 0.005, 0.005]; top_p 0.95 needs only the first.
        let w = [0.98, 0.01, 0.005, 0.005];
        assert_eq!(nucleus(&w, 0.95), vec![0]);
        assert!(mass(&w, &nucleus(&w, 0.95)) >= 0.95);
    }

    #[test]
    fn boundary_just_below_and_just_above() {
        // Every weight (and every partial sum) is exact in binary f64, so
        // the `>=` comparisons at the boundaries are not fp-fragile.
        // p = [0.5, 0.25, 0.125, 0.0625, 0.0625], Z = 1.0
        // cumulative:  0.5, 0.75, 0.875, 0.9375, 1.0
        let w = [0.5, 0.25, 0.125, 0.0625, 0.0625];
        assert_eq!(nucleus(&w, 0.4), vec![0]);
        // Just under the 2-token boundary.
        assert_eq!(nucleus(&w, 0.74), vec![0, 1]);
        // Exactly at it: `>=` is inclusive, so 2 tokens still suffice.
        assert_eq!(nucleus(&w, 0.75), vec![0, 1]);
        // Just above: the third token is pulled in.
        assert_eq!(nucleus(&w, 0.7501), vec![0, 1, 2]);
        assert_eq!(nucleus(&w, 0.875), vec![0, 1, 2]);
        // Above the 3-token boundary the 4th is needed — and the 5th ties
        // with it at 0.0625, so the threshold rule keeps both.
        assert_eq!(nucleus(&w, 0.8751), vec![0, 1, 2, 3, 4]);
        for &p in &[0.4, 0.74, 0.75, 0.7501, 0.875, 0.8751] {
            assert!(
                mass(&w, &nucleus(&w, p)) >= p - 1e-12,
                "top_p={p} must keep at least p of the mass"
            );
        }
    }

    #[test]
    fn ties_at_the_boundary_are_all_kept() {
        // Four equal runners-up: a sorted cut would take exactly one of them,
        // the threshold rule takes all four.
        let w = [0.6, 0.1, 0.1, 0.1, 0.1];
        // target = 0.65: 0.6 alone is short, +0.1 = 0.7 reaches it, so the
        // boundary weight is 0.1 — and every 0.1 qualifies.
        assert_eq!(top_p_cutoff(&w, 0.65), 0.1);
        assert_eq!(nucleus(&w, 0.65), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn flat_distribution() {
        // 100 equal tokens: any top_p < 1 keeps all of them, because the
        // boundary weight is shared by every token.
        let w = vec![1.0f64; 100];
        assert_eq!(top_p_cutoff(&w, 0.95), 1.0);
        assert_eq!(nucleus(&w, 0.95).len(), 100);
        // ... and 1/100 of the mass is enough for top_p = 0.005.
        assert_eq!(nucleus(&w, 0.005).len(), 100);
    }

    #[test]
    fn hand_computed_geometric() {
        // w = 1, 1/2, 1/4, 1/8, 1/16 ; Z = 1.9375
        // cumulative fractions: 0.5161, 0.7742, 0.9032, 0.9677, 1.0
        let w = [1.0, 0.5, 0.25, 0.125, 0.0625];
        assert_eq!(nucleus(&w, 0.5), vec![0]);
        assert_eq!(nucleus(&w, 0.52), vec![0, 1]);
        assert_eq!(nucleus(&w, 0.78), vec![0, 1, 2]);
        assert_eq!(nucleus(&w, 0.91), vec![0, 1, 2, 3]);
        assert_eq!(nucleus(&w, 0.97), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn min_p_composes_as_max_of_thresholds() {
        let w = [1.0, 0.5, 0.25, 0.125, 0.0625];
        // top_p alone would keep 4 tokens (cutoff 0.125); min_p 0.3 is
        // stricter, so the intersection is the 2-token set.
        let t = top_p_min_p_threshold(&w, 0.91, 0.3);
        assert_eq!(t, 0.3);
        let set: Vec<usize> = (0..w.len()).filter(|&i| w[i] >= t).collect();
        assert_eq!(set, vec![0, 1]);
        // The other direction: a loose min_p leaves top_p in charge.
        assert_eq!(top_p_min_p_threshold(&w, 0.91, 0.01), 0.125);
    }
}
