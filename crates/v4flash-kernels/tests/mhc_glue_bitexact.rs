//! Bit-exactness of the 2026-09-26 F_mhc_glue sweep kernels against the
//! kernels they replace, on synthetic data (no model load; < 60 MB of device
//! memory per test, so it runs beside a live hub):
//!
//!   1. `rms_norm_weighted_batched_fast` vs `rms_norm_weighted_batched` and the
//!      per-row `rms_norm_weighted` (grid 1), n in the fast set {512, 1280,
//!      5120} and the generic fallback (4864, 100), incl. zero / huge rows,
//!      plus the `RmsNorm` wrappers' default pick (`V41_RMS_FAST`).
//!   2. `f16_matvec_batched_h20` vs `f16_matvec_batched` at the router shape
//!      384x5120 (b = 1..16, 64) and tail / other shapes (generic path), plus
//!      `F16Matvec::matvec_batched_router` (`V41_ROUTER_MV_H20`).
//!   3. `router_topk_wfred` vs `router_topk_par` on every output (selected,
//!      weights, alts, alt_w, orig_sel, range) in the plain, alternatives,
//!      cache-prior (n_protect 0/2/6) and dry-run configs, random / tie-heavy /
//!      all-equal logits, b = 1..16, plus `launch_batched_ex` (`V41_TOPK_WFRED`).
//!   4. `f16_gemm_narrow_n16_bk128_pf2` vs `f16_gemm_wmma_lds_tiled` at the mHC
//!      pre-mix shape M = 24, K = 20480 for B = 1..513 (tails around 16 / 64),
//!      plus `gemm_batched_wmma`'s gate (`V41_MHC_GEMM_NARROW`): narrow shapes
//!      routed, K % 128 != 0 and M = 384 kept on the old kernel. gfx12 only.
//!
//! Run: `cargo test --release --features v41 -p v4flash-kernels --test
//! mhc_glue_bitexact -- --ignored --test-threads=1 --nocapture`. With every
//! knob = 0 the wrapper comparisons pit the OLD path against the old symbols.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, launch_kernel, sys, Device, DeviceBuffer, LaunchConfig, Stream};
use v4flash_kernels::config::{HC_DIM, HC_MIX_DIM, N_EMBD, N_EXPERT, RMS_EPS};
use v4flash_kernels::router_topk::{RouterEx, RouterTopk, ROUTER_MAX_EXPERTS};
use v4flash_kernels::{F16Matvec, RmsNorm};

fn pick_dgpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") {
            return Ok(d);
        }
    }
    Device::all()?.into_iter().next().ok_or_else(|| eyre!("no HIP devices"))
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next() & 0xFFFFFF) as f32 / 16777216.0
    }
    /// Uniform in [-a, a).
    fn sym(&mut self, a: f32) -> f32 {
        (self.unit() * 2.0 - 1.0) * a
    }
    /// A finite f16 bit pattern, |v| in ~[2^-9, 2^0), random sign.
    fn f16_bits(&mut self) -> u16 {
        let exp = 6 + (self.next() % 9) as u16; // 6..=14
        let man = (self.next() & 0x3FF) as u16;
        (((self.next() & 1) as u16) << 15) | (exp << 10) | man
    }
}

fn upload<T: Copy>(id: i32, host: &[T]) -> eyre::Result<DeviceBuffer<T>> {
    let mut d: DeviceBuffer<T> = DeviceBuffer::new(id, host.len())?;
    d.copy_from_host(host)?;
    Ok(d)
}

fn download<T: Copy + Default>(d: &DeviceBuffer<T>) -> eyre::Result<Vec<T>> {
    let mut v = vec![T::default(); d.len()];
    d.copy_to_host(&mut v)?;
    Ok(v)
}

const SENTINEL: f32 = f32::from_bits(0x7FC0_BEEF);

fn sentinel_buf(id: i32, n: usize) -> eyre::Result<DeviceBuffer<f32>> {
    upload(id, &vec![SENTINEL; n])
}

/// Count of bit-different f32 lanes; a sentinel left in place counts too.
fn bit_diff(a: &[f32], b: &[f32]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

fn unwritten(a: &[f32]) -> usize {
    a.iter().filter(|x| x.to_bits() == SENTINEL.to_bits()).count()
}

fn f16_bytes(bits: &[u16]) -> Vec<u8> {
    bits.iter().flat_map(|b| b.to_le_bytes()).collect()
}

// ---------------------------------------------------------------- 1. rms_fast

#[test]
#[ignore]
fn rms_fast_matches_rms_norm_weighted() -> eyre::Result<()> {
    install_panic_handler()?;
    let dev = pick_dgpu()?;
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let id = dev.id;
    let rms = RmsNorm::for_arch(&arch)?;
    let stream = Stream::new(id)?;
    let mut rng = Lcg(0x5EED_F001);
    let mut cases = 0usize;
    for &n in &[512u32, 1280, N_EMBD, 4864, 100] {
        let w_h: Vec<f32> = (0..n).map(|_| rng.sym(2.0)).collect();
        let w = upload(id, &w_h)?;
        for &b in &[1u32, 3, 4, 8, 512] {
            if b == 512 && n != N_EMBD && n != 1280 {
                continue;
            }
            let nn = (b * n) as usize;
            let mut x_h: Vec<f32> = (0..nn).map(|_| rng.sym(4.0)).collect();
            // Adversarial rows: all-zero (scale = rsqrt(eps)), huge (sum of squares
            // overflows f32 after the double accumulation), tiny.
            if b >= 3 {
                x_h[..n as usize].iter_mut().for_each(|v| *v = 0.0);
                x_h[n as usize..2 * n as usize].iter_mut().enumerate().for_each(|(i, v)| *v = if i % 2 == 0 { 3e19 } else { -1e18 });
                x_h[2 * n as usize..3 * n as usize].iter_mut().for_each(|v| *v *= 1e-20);
            }
            let x = upload(id, &x_h)?;
            let cfg = LaunchConfig { grid: (b, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
            let tail = 257usize;
            let out_ref = sentinel_buf(id, nn + tail)?;
            let f = rms.module().get_function("rms_norm_weighted_batched")?;
            launch_kernel!(f, cfg, &stream, [out_ref.raw(), x.raw(), w.raw(), n, RMS_EPS])?;
            let out_fast = sentinel_buf(id, nn + tail)?;
            let f = rms.module().get_function("rms_norm_weighted_batched_fast")?;
            launch_kernel!(f, cfg, &stream, [out_fast.raw(), x.raw(), w.raw(), n, RMS_EPS])?;
            let mut out_wr = sentinel_buf(id, nn + tail)?;
            rms.launch_weighted_batched(&stream, &mut out_wr, &x, &w, n, RMS_EPS, b)?;
            stream.synchronize()?;
            let (r, fa, wr) = (download(&out_ref)?, download(&out_fast)?, download(&out_wr)?);
            assert_eq!(unwritten(&r[..nn]), 0, "old kernel left outputs unwritten at n={n} b={b}");
            assert_eq!(unwritten(&r[nn..]), tail, "old kernel wrote past the end at n={n} b={b}");
            let (d_fa, d_wr) = (bit_diff(&r, &fa), bit_diff(&r, &wr));
            eprintln!("rms n={n} b={b}: fast bit_diff={d_fa}, wrapper bit_diff={d_wr}");
            assert_eq!(d_fa, 0, "rms_norm_weighted_batched_fast not bit-exact at n={n} b={b}");
            assert_eq!(d_wr, 0, "launch_weighted_batched not bit-exact at n={n} b={b}");
            cases += 1;
        }
        // Per-row head prep: rms_norm_weighted (grid 1) vs the fast kernel at grid 1
        // and the launch_weighted wrapper.
        let x_h: Vec<f32> = (0..n).map(|_| rng.sym(4.0)).collect();
        let x = upload(id, &x_h)?;
        let cfg = LaunchConfig { grid: (1, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        let out_ref = sentinel_buf(id, n as usize)?;
        let f = rms.module().get_function("rms_norm_weighted")?;
        launch_kernel!(f, cfg, &stream, [out_ref.raw(), x.raw(), w.raw(), n, RMS_EPS])?;
        let out_fast = sentinel_buf(id, n as usize)?;
        let f = rms.module().get_function("rms_norm_weighted_batched_fast")?;
        launch_kernel!(f, cfg, &stream, [out_fast.raw(), x.raw(), w.raw(), n, RMS_EPS])?;
        let mut out_wr = sentinel_buf(id, n as usize)?;
        rms.launch_weighted(&stream, &mut out_wr, &x, &w, n, RMS_EPS)?;
        stream.synchronize()?;
        let (r, fa, wr) = (download(&out_ref)?, download(&out_fast)?, download(&out_wr)?);
        assert_eq!(unwritten(&r), 0);
        let (d_fa, d_wr) = (bit_diff(&r, &fa), bit_diff(&r, &wr));
        eprintln!("rms 1-row n={n}: fast bit_diff={d_fa}, wrapper bit_diff={d_wr}");
        assert_eq!(d_fa, 0, "fast kernel at grid 1 not bit-exact vs rms_norm_weighted at n={n}");
        assert_eq!(d_wr, 0, "launch_weighted not bit-exact at n={n}");
        cases += 1;
    }
    eprintln!("PASS: {cases} rms cases bit-exact");
    Ok(())
}

// ------------------------------------------------------------ 2. router mv h20

#[test]
#[ignore]
fn router_mv_h20_matches_f16_matvec_batched() -> eyre::Result<()> {
    install_panic_handler()?;
    let dev = pick_dgpu()?;
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let id = dev.id;
    let f16 = F16Matvec::for_arch(&arch)?;
    let stream = Stream::new(id)?;
    let mut rng = Lcg(0x5EED_F002);
    // (n_rows, k): router; row tail; generic-path k tails; 1 and 2 chunks; the
    // other matvec_batched shapes (indexer q, proj, compressor, index-K).
    let shapes: [(u32, u32); 9] = [
        (N_EXPERT, N_EMBD), (383, N_EMBD), (384, 4992), (41, 5152), (8, 640), (16, 1280),
        (4096, 1280), (32, 5120), (128, 512),
    ];
    let mut cases = 0usize;
    for &(n_rows, k) in &shapes {
        let w_bits: Vec<u16> = (0..(n_rows * k)).map(|_| rng.f16_bits()).collect();
        let w = upload(id, &f16_bytes(&w_bits))?;
        let bs: &[u32] = if (n_rows, k) == (N_EXPERT, N_EMBD) { &[1, 2, 3, 4, 5, 6, 7, 8, 16, 64] } else { &[1, 3, 8] };
        for &b in bs {
            let x_h: Vec<f32> = (0..(b * k)).map(|_| rng.sym(3.0)).collect();
            let x = upload(id, &x_h)?;
            let n_out = (b * n_rows) as usize;
            let cfg = LaunchConfig { grid: (n_rows.div_ceil(8), 1, b), block: (256, 1, 1), shared_mem_bytes: 0 };
            let out_ref = sentinel_buf(id, n_out)?;
            let f = f16.wide_module().get_function("f16_matvec_batched")?;
            launch_kernel!(f, cfg, &stream, [out_ref.raw(), w.raw(), x.raw(), k, n_rows])?;
            let out_h20 = sentinel_buf(id, n_out)?;
            let f = f16.wide_module().get_function("f16_matvec_batched_h20")?;
            launch_kernel!(f, cfg, &stream, [out_h20.raw(), w.raw(), x.raw(), k, n_rows])?;
            let mut out_wr = sentinel_buf(id, n_out)?;
            f16.matvec_batched_router(&stream, &mut out_wr, &w, &x, n_rows, k, b)?;
            stream.synchronize()?;
            let (r, h, wr) = (download(&out_ref)?, download(&out_h20)?, download(&out_wr)?);
            assert_eq!(unwritten(&r), 0, "f16_matvec_batched left outputs unwritten at {n_rows}x{k} b={b}");
            assert!(r.iter().all(|v| v.is_finite()));
            let (d_h, d_wr) = (bit_diff(&r, &h), bit_diff(&r, &wr));
            eprintln!("mv {n_rows}x{k} b={b}: h20 bit_diff={d_h}, router wrapper bit_diff={d_wr}");
            assert_eq!(d_h, 0, "f16_matvec_batched_h20 not bit-exact at {n_rows}x{k} b={b}");
            assert_eq!(d_wr, 0, "matvec_batched_router not bit-exact at {n_rows}x{k} b={b}");
            cases += 1;
        }
    }
    eprintln!("PASS: {cases} matvec cases bit-exact");
    Ok(())
}

// -------------------------------------------------------------- 3. topk wfred

struct TopkOut {
    sel: Vec<i32>,
    w: Vec<u32>,
    alts: Vec<i32>,
    alt_w: Vec<u32>,
    orig: Vec<i32>,
    range: Vec<u32>,
}

#[allow(clippy::too_many_arguments)]
fn topk_run(
    r: &RouterTopk, s: &Stream, id: i32, sym: Option<&str>, logits: &DeviceBuffer<f32>, bias: &DeviceBuffer<f32>,
    prior: Option<&DeviceBuffer<f32>>, b: u32, n_used: u32, n_alt: u32, n_protect: u32, dry: bool,
) -> eyre::Result<TopkOut> {
    let bn = b as usize;
    let nu = n_used as usize;
    let na = n_alt.max(1) as usize;
    let fill_i = |n: usize| upload(id, &vec![-7i32; n]);
    let mut sel = fill_i(bn * nu)?;
    let mut w = sentinel_buf(id, bn * nu)?;
    let mut alts = fill_i(bn * na)?;
    let mut alt_w = sentinel_buf(id, bn * na)?;
    let mut orig = fill_i(bn * nu)?;
    let mut range = sentinel_buf(id, bn)?;
    match sym {
        Some(sym) => {
            let f = r.module().get_function(sym)?;
            let cfg = LaunchConfig { grid: (b, 1, 1), block: (ROUTER_MAX_EXPERTS, 1, 1), shared_mem_bytes: 0 };
            let null: sys::hipDeviceptr_t = std::ptr::null_mut();
            let a_ptr = if n_alt > 0 { alts.raw() } else { null };
            let aw_ptr = if n_alt > 0 { alt_w.raw() } else { null };
            let pr_ptr = prior.map_or(null, |p| p.raw());
            launch_kernel!(f, cfg, s, [
                sel.raw(), w.raw(), logits.raw(), bias.raw(),
                N_EXPERT, n_used, 1.5f32, 6.103515625e-5f32, a_ptr, n_alt, aw_ptr,
                pr_ptr, n_protect.min(n_used), dry as u32, orig.raw(), range.raw()
            ])?;
        }
        None => {
            r.launch_batched_ex(
                s, &mut sel, &mut w, logits, Some(bias), N_EXPERT, n_used, 1.5, 6.103515625e-5, b,
                RouterEx {
                    alts: if n_alt > 0 { Some(&mut alts) } else { None },
                    n_alt,
                    alt_w: if n_alt > 0 { Some(&mut alt_w) } else { None },
                    prior,
                    n_protect,
                    prior_dry: dry,
                    orig_sel: Some(&mut orig),
                    range_out: Some(&mut range),
                },
            )?;
        }
    }
    s.synchronize()?;
    let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<u32>>();
    Ok(TopkOut {
        sel: download(&sel)?,
        w: bits(download(&w)?),
        alts: download(&alts)?,
        alt_w: bits(download(&alt_w)?),
        orig: download(&orig)?,
        range: bits(download(&range)?),
    })
}

#[test]
#[ignore]
fn topk_wfred_matches_router_topk_par() -> eyre::Result<()> {
    install_panic_handler()?;
    let dev = pick_dgpu()?;
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let id = dev.id;
    let r = RouterTopk::for_arch(&arch)?;
    let stream = Stream::new(id)?;
    let mut rng = Lcg(0x5EED_F003);
    let ne = N_EXPERT as usize;
    let mut cases = 0usize;
    // Input classes: random; tie-heavy (7 logit levels, zero bias); all-equal.
    for class in 0..3 {
        for &b in &[1u32, 2, 3, 4, 5, 8, 16] {
            let bn = b as usize;
            let logits_h: Vec<f32> = (0..bn * ne)
                .map(|_| match class {
                    0 => rng.sym(3.0),
                    1 => ((rng.next() % 7) as f32 - 3.0) * 0.5,
                    _ => 0.25,
                })
                .collect();
            let bias_h: Vec<f32> = (0..ne).map(|_| if class == 0 { rng.sym(0.1) } else { 0.0 }).collect();
            let prior_h: Vec<f32> = (0..ne).map(|_| if rng.next() % 3 == 0 { 0.3 } else { 0.0 }).collect();
            let logits = upload(id, &logits_h)?;
            let bias = upload(id, &bias_h)?;
            let prior = upload(id, &prior_h)?;
            // (prior, n_used, n_alt, n_protect, dry): plain, plain + alts, the
            // production cache-prior (n_protect 2) with / without alts, protect
            // everything, protect nothing, dry run.
            let cfgs: [(bool, u32, u32, u32, bool); 7] = [
                (false, 6, 0, 0, false), (false, 6, 3, 0, false), (true, 6, 0, 2, false),
                (true, 6, 2, 2, false), (true, 6, 1, 6, false), (true, 6, 0, 0, false), (true, 6, 2, 2, true),
            ];
            for &(use_prior, nu, na, np, dry) in &cfgs {
                let p = if use_prior { Some(&prior) } else { None };
                let old = topk_run(&r, &stream, id, Some("router_topk_par"), &logits, &bias, p, b, nu, na, np, dry)?;
                let new = topk_run(&r, &stream, id, Some("router_topk_wfred"), &logits, &bias, p, b, nu, na, np, dry)?;
                let wr = topk_run(&r, &stream, id, None, &logits, &bias, p, b, nu, na, np, dry)?;
                let tag = format!("class={class} b={b} prior={use_prior} n_alt={na} n_protect={np} dry={dry}");
                assert!(old.sel.iter().all(|&s| s >= 0 && (s as u32) < N_EXPERT), "par picks invalid: {tag}");
                for (name, o) in [("wfred", &new), ("launch_batched_ex", &wr)] {
                    assert_eq!(old.sel, o.sel, "{name} selected differs: {tag}");
                    assert_eq!(old.w, o.w, "{name} weights differ: {tag}");
                    assert_eq!(old.alts, o.alts, "{name} alts differ: {tag}");
                    assert_eq!(old.alt_w, o.alt_w, "{name} alt_w differ: {tag}");
                    assert_eq!(old.orig, o.orig, "{name} orig_sel differs: {tag}");
                    assert_eq!(old.range, o.range, "{name} range differs: {tag}");
                }
                cases += 1;
            }
        }
    }
    eprintln!("PASS: {cases} topk configs bit-exact (selected, weights, alts, alt_w, orig_sel, range)");
    Ok(())
}

// ------------------------------------------------------- 4. mHC narrow GEMM

#[test]
#[ignore]
fn mhc_gemm_narrow_matches_lds_tiled() -> eyre::Result<()> {
    install_panic_handler()?;
    let dev = pick_dgpu()?;
    let arch = dev.properties()?.gcn_arch_name;
    if !arch.starts_with("gfx1201") {
        eprintln!("SKIP: WMMA GEMMs are gfx12-only (device is {arch})");
        return Ok(());
    }
    dev.set_current()?;
    let id = dev.id;
    let f16 = F16Matvec::for_arch(&arch)?;
    let gm = f16.gemm_module().ok_or_else(|| eyre!("no gemm module"))?;
    let stream = Stream::new(id)?;
    let mut rng = Lcg(0x5EED_F004);
    const B_MAX: u32 = 513;
    let (m, k) = (HC_MIX_DIM, HC_DIM);
    let w_bits: Vec<u16> = (0..(m * k)).map(|_| rng.f16_bits()).collect();
    let w = upload(id, &f16_bytes(&w_bits))?;
    // Residual-like activations; row r scaled so tails differ from the body.
    let x_h: Vec<f32> = (0..(B_MAX * k) as usize).map(|i| rng.sym(1.0) * (1.0 + (i / k as usize % 7) as f32)).collect();
    let x = upload(id, &x_h)?;
    let mut cases = 0usize;
    for &b in &[1u32, 7, 15, 16, 17, 33, 63, 64, 65, 100, 129, 511, 512, 513] {
        let n_out = (b * m) as usize;
        let out_ref = sentinel_buf(id, n_out)?;
        let f = gm.get_function("f16_gemm_wmma_lds_tiled")?;
        let cfg = LaunchConfig { grid: (m.div_ceil(64), b.div_ceil(64), 1), block: (128, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, &stream, [out_ref.raw(), w.raw(), x.raw(), k, m, b])?;
        let out_nar = sentinel_buf(id, n_out)?;
        let f = gm.get_function("f16_gemm_narrow_n16_bk128_pf2")?;
        let cfg = LaunchConfig { grid: (1, b.div_ceil(16), 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, &stream, [out_nar.raw(), w.raw(), x.raw(), k, m, b])?;
        let mut out_wr = sentinel_buf(id, n_out)?;
        f16.gemm_batched_wmma(&stream, &mut out_wr, &w, &x, m, k, b)?;
        stream.synchronize()?;
        let (r, n, wr) = (download(&out_ref)?, download(&out_nar)?, download(&out_wr)?);
        assert_eq!(unwritten(&r), 0, "lds_tiled left outputs unwritten at B={b}");
        assert!(r.iter().all(|v| v.is_finite()), "non-finite reference at B={b}");
        let (d_n, d_wr) = (bit_diff(&r, &n), bit_diff(&r, &wr));
        eprintln!("gemm {m}x{k} B={b}: narrow bit_diff={d_n}, wrapper bit_diff={d_wr}");
        assert_eq!(d_n, 0, "f16_gemm_narrow_n16_bk128_pf2 not bit-exact at B={b}");
        assert_eq!(d_wr, 0, "gemm_batched_wmma wrapper not bit-exact at B={b}");
        cases += 1;
    }
    // Gate: shapes outside the narrow preconditions stay on lds_tiled (the
    // wrapper output must equal the old kernel's, which it can only do by
    // running it: the narrow kernel writes nothing there).
    for &(m2, k2, b) in &[(HC_MIX_DIM, HC_DIM - 32, 65u32), (N_EXPERT, N_EMBD, 65)] {
        let w2_bits: Vec<u16> = (0..(m2 * k2)).map(|_| rng.f16_bits()).collect();
        let w2 = upload(id, &f16_bytes(&w2_bits))?;
        let n_out = (b * m2) as usize;
        let out_ref = sentinel_buf(id, n_out)?;
        let f = gm.get_function("f16_gemm_wmma_lds_tiled")?;
        let cfg = LaunchConfig { grid: (m2.div_ceil(64), b.div_ceil(64), 1), block: (128, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, cfg, &stream, [out_ref.raw(), w2.raw(), x.raw(), k2, m2, b])?;
        let mut out_wr = sentinel_buf(id, n_out)?;
        f16.gemm_batched_wmma(&stream, &mut out_wr, &w2, &x, m2, k2, b)?;
        stream.synchronize()?;
        let (r, wr) = (download(&out_ref)?, download(&out_wr)?);
        assert_eq!(unwritten(&r), 0);
        let d = bit_diff(&r, &wr);
        eprintln!("gemm gate {m2}x{k2} B={b}: wrapper bit_diff={d} (must stay on lds_tiled)");
        assert_eq!(d, 0, "gemm_batched_wmma misrouted {m2}x{k2}");
        cases += 1;
    }
    eprintln!("PASS: {cases} gemm cases bit-exact");
    Ok(())
}
