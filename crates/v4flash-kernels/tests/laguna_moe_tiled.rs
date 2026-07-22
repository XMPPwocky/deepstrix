//! Laguna by-expert TILED MoE GEMM — parity vs the proven BxN path + a
//! bandwidth-amortization B-sweep (Milestone 1 of the batched-prefill roofline
//! lever).
//!
//! The tiled kernels ([`LagunaMoeTiled`]) read each active expert's weight tile
//! ONCE per B-chunk (grid.y = expert, stream all its routed tokens) instead of
//! once-per-(token,slot) like the BxN kernels. Laguna's Q4_K/Q6_K dequant is
//! cheap, so the lever is pure weight BANDWIDTH: per-chunk weight traffic drops
//! from ~(B*top_k) expert reads to ~(distinct experts hit) reads.
//!
//! Uses REAL layer-1 expert weights from the Laguna GGUF (exact dtypes/byte
//! layout) with synthetic activations + random distinct top-10 routing (a
//! near-uniform, conservative reuse pattern; real routing is more skewed →
//! more reuse). Validates the tiled output ≈ the BxN output (down uses atomic
//! accumulation → f32 non-associativity, so ~1e-4 rel, not bit-exact), then
//! times both across B ∈ {64, 256, 512} and prints µs/token + the BW roofline.
//!
//! Run (server stopped, GPU free):
//!   nix develop --command cargo test --release -p v4flash-kernels \
//!       --test laguna_moe_tiled -- --ignored --nocapture

use color_eyre::eyre::{self, eyre};
use v4flash_core::gguf::GgufType;
use v4flash_core::MappedGguf;
use v4flash_hip::{Device, DeviceBuffer, Stream};
use v4flash_kernels::{
    LagunaMoeTiled, MoeGroupBuilder, Q4KMatvec, Q6KMatvec, Q8KQuantize,
};

const GGUF_PATH: &str = "/persist/lumi/models/laguna-s-2.1-int4/laguna-s-2.1-Q4_K_M.gguf";

const HIDDEN: usize = 3072;
const N_EXPERT: usize = 256;
const TOPK: usize = 10;
const FF_EXP: usize = 1024;
// first MoE layer by default; override with LAGUNA_TEST_LAYER to exercise a
// layer with a Q4_K down (the new q4_k_down_tiled_reg_w32_col col path).
fn test_layer() -> usize {
    std::env::var("LAGUNA_TEST_LAYER").ok().and_then(|v| v.parse().ok()).unwrap_or(1)
}

fn block_bytes(dt: GgufType) -> usize {
    match dt {
        GgufType::Q4_K => 144,
        GgufType::Q6_K => 210,
        _ => 0,
    }
}

struct Lcg(u64);
impl Lcg {
    fn u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn f32(&mut self) -> f32 {
        (self.u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// Distinct top-10 experts per token (routing selection), + positive weights.
fn gen_routing(b: usize, rng: &mut Lcg) -> (Vec<i32>, Vec<f32>) {
    let mut sel = vec![0i32; b * TOPK];
    let mut ew = vec![0f32; b * TOPK];
    for t in 0..b {
        let mut chosen = Vec::with_capacity(TOPK);
        while chosen.len() < TOPK {
            let e = (rng.u32() as usize) % N_EXPERT;
            if !chosen.contains(&e) {
                chosen.push(e);
            }
        }
        for s in 0..TOPK {
            sel[t * TOPK + s] = chosen[s] as i32;
            ew[t * TOPK + s] = (rng.f32().abs()) * 0.4 + 0.05;
        }
    }
    (sel, ew)
}

fn rel_err(a: &[f32], b: &[f32]) -> (f32, f32) {
    // (max rel err, mean rel err) over elements with non-trivial magnitude.
    let mut max_r = 0f32;
    let mut sum_r = 0f32;
    let mut n = 0usize;
    for (x, y) in a.iter().zip(b) {
        let m = x.abs().max(y.abs());
        if m < 1e-3 {
            continue;
        }
        let r = (x - y).abs() / m;
        max_r = max_r.max(r);
        sum_r += r;
        n += 1;
    }
    (max_r, if n > 0 { sum_r / n as f32 } else { 0.0 })
}

struct Weights {
    gate_all: DeviceBuffer<u8>,
    up_all: DeviceBuffer<u8>,
    down_all: DeviceBuffer<u8>,
    gate_stride: usize,
    up_stride: usize,
    down_stride: usize,
    down_dt: GgufType,
}

#[test]
#[ignore = "drives the GPU + needs the 75GB Laguna GGUF; run explicitly"]
fn laguna_moe_tiled_parity_and_bw() -> eyre::Result<()> {
    v4flash_hip::install_panic_handler();

    // iGPU (gfx1151) — the experts live there in production; measure BW there.
    let dev = Device::all()?
        .into_iter()
        .find(|d| {
            d.properties()
                .map(|p| p.gcn_arch_name.starts_with("gfx1151"))
                .unwrap_or(false)
        })
        .ok_or_else(|| eyre!("no gfx1151 device"))?;
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    println!("device id={} arch={arch}", dev.id);
    let stream = Stream::new(dev.id)?;

    let q8k = Q8KQuantize::for_arch(&arch)?;
    let q4b = Q4KMatvec::for_arch(&arch)?;
    let q6b = Q6KMatvec::for_arch(&arch)?;
    let tiled = LagunaMoeTiled::for_arch(&arch)?;
    let gb = MoeGroupBuilder::for_arch(&arch)?;

    // ----- load real layer-1 expert weights -----
    let gguf = MappedGguf::open(GGUF_PATH)?;
    let g = gguf.gguf();
    let p = |s: &str| format!("blk.{}.{s}", test_layer());
    let gate_t = g.tensor(&p("ffn_gate_exps.weight")).ok_or_else(|| eyre!("no gate_exps"))?;
    let up_t = g.tensor(&p("ffn_up_exps.weight")).ok_or_else(|| eyre!("no up_exps"))?;
    let down_t = g.tensor(&p("ffn_down_exps.weight")).ok_or_else(|| eyre!("no down_exps"))?;
    let gate_stride = FF_EXP * (HIDDEN / 256) * block_bytes(gate_t.dtype);
    let up_stride = FF_EXP * (HIDDEN / 256) * block_bytes(up_t.dtype);
    let down_stride = HIDDEN * (FF_EXP / 256) * block_bytes(down_t.dtype);
    println!(
        "L{} dtypes: gate={:?} up={:?} down={:?} | strides g={gate_stride} u={up_stride} d={down_stride}",
        test_layer(), gate_t.dtype, up_t.dtype, down_t.dtype
    );
    assert_eq!(gate_t.dtype, GgufType::Q4_K);
    assert_eq!(up_t.dtype, GgufType::Q4_K);

    let up_dev = |name: &str| -> eyre::Result<DeviceBuffer<u8>> {
        let t = gguf.gguf().tensor(name).unwrap();
        let bytes = gguf.read_tensor(t)?;
        let mut b = DeviceBuffer::<u8>::new(dev.id, bytes.len())?;
        b.copy_from_host(&bytes)?;
        Ok(b)
    };
    let w = Weights {
        gate_all: up_dev(&p("ffn_gate_exps.weight"))?,
        up_all: up_dev(&p("ffn_up_exps.weight"))?,
        down_all: up_dev(&p("ffn_down_exps.weight"))?,
        gate_stride,
        up_stride,
        down_stride,
        down_dt: down_t.dtype,
    };

    let n_blk_hidden = (HIDDEN / 256) as u32; // 12
    let n_blk_mid = (FF_EXP / 256) as u32; // 4
    let xq_slot_stride = n_blk_mid * 292;

    let per_expert_bytes = (gate_stride + up_stride + down_stride) as f64;
    let ig_bw = 229.0e9f64; // gfx1151 achievable DRAM BW (phase0)

    // ================= parity (B=64) + BW sweep =================
    let mut first = true;
    for &b in &[64usize, 256, 512] {
        let mut rng = Lcg(0xC0FFEE ^ (b as u64));
        let (sel_h, ew_h) = gen_routing(b, &mut rng);
        let acts: Vec<f32> = (0..b * HIDDEN).map(|_| rng.f32()).collect();

        // uploads
        let mut sel = DeviceBuffer::<i32>::new(dev.id, sel_h.len())?;
        sel.copy_from_host(&sel_h)?;
        let mut ew = DeviceBuffer::<f32>::new(dev.id, ew_h.len())?;
        ew.copy_from_host(&ew_h)?;
        let mut acts_d = DeviceBuffer::<f32>::new(dev.id, acts.len())?;
        acts_d.copy_from_host(&acts)?;

        // Q8_K of activations: [B, n_blk_hidden] blocks contiguous.
        let mut xq = DeviceBuffer::<u8>::new(dev.id, b * n_blk_hidden as usize * 292)?;
        q8k.launch(&stream, &mut xq, &acts_d, b as u32 * n_blk_hidden)?;

        // scratch
        let mut mid = DeviceBuffer::<f32>::new(dev.id, b * TOPK * FF_EXP)?;
        let mut xq_mid = DeviceBuffer::<u8>::new(dev.id, b * TOPK * n_blk_mid as usize * 292)?;
        let mut out = DeviceBuffer::<f32>::new(dev.id, b * HIDDEN)?;

        // group-builder inputs
        let mut group_count = DeviceBuffer::<i32>::new(dev.id, N_EXPERT)?;
        let max_per_expert = b as u32; // each expert picked ≤ once per token
        let mut members = DeviceBuffer::<i32>::new(dev.id, N_EXPERT * b)?;

        // ---- closures: one full MoE-expert step (gate/up + q8k + down) ----
        let run_bxn = |stream: &Stream,
                       mid: &mut DeviceBuffer<f32>,
                       xq_mid: &mut DeviceBuffer<u8>,
                       out: &mut DeviceBuffer<f32>|
         -> eyre::Result<()> {
            q4b.launch_pair_swiglu_bxn(
                stream, mid, &w.gate_all, &w.up_all, &xq, &ew, &sel,
                w.gate_stride as u32, w.up_stride as u32, TOPK as u32, 0.0,
                FF_EXP as u32, n_blk_hidden, b as u32,
            )?;
            q8k.launch(stream, xq_mid, mid, b as u32 * TOPK as u32 * n_blk_mid)?;
            match w.down_dt {
                GgufType::Q6_K => q6b.launch_batched_bxn(
                    stream, out, &w.down_all, xq_mid, &sel, w.down_stride as u32,
                    xq_slot_stride, TOPK as u32, HIDDEN as u32, n_blk_mid, b as u32,
                )?,
                GgufType::Q4_K => q4b.launch_batched_bxn(
                    stream, out, &w.down_all, xq_mid, &sel, w.down_stride as u32,
                    xq_slot_stride, TOPK as u32, HIDDEN as u32, n_blk_mid, b as u32,
                )?,
                other => return Err(eyre!("down dtype {other:?}")),
            }
            Ok(())
        };

        let run_tiled = |stream: &Stream,
                         group_count: &mut DeviceBuffer<i32>,
                         members: &mut DeviceBuffer<i32>,
                         mid: &mut DeviceBuffer<f32>,
                         xq_mid: &mut DeviceBuffer<u8>,
                         out: &mut DeviceBuffer<f32>|
         -> eyre::Result<()> {
            group_count.fill_zero_async(stream)?;
            gb.launch(
                stream, group_count, members, &sel, b as u32, TOPK as u32,
                N_EXPERT as u32, max_per_expert,
            )?;
            tiled.gate_up_swiglu(
                stream, mid, &w.gate_all, &w.up_all, &xq, &ew, group_count, members,
                w.gate_stride as u32, w.up_stride as u32, TOPK as u32, max_per_expert,
                0.0, FF_EXP as u32, n_blk_hidden, N_EXPERT as u32,
            )?;
            q8k.launch(stream, xq_mid, mid, b as u32 * TOPK as u32 * n_blk_mid)?;
            out.fill_zero_async(stream)?;
            tiled.down(
                stream, w.down_dt, out, &w.down_all, xq_mid, group_count, members,
                w.down_stride as u32, xq_slot_stride, TOPK as u32, max_per_expert,
                HIDDEN as u32, n_blk_mid, N_EXPERT as u32,
            )?;
            Ok(())
        };

        // register-tiled path (decode weight once, reuse across members).
        // down_reg only implemented for Q6_K; fall back to naive down for Q4_K.
        let run_reg = |stream: &Stream,
                       group_count: &mut DeviceBuffer<i32>,
                       members: &mut DeviceBuffer<i32>,
                       mid: &mut DeviceBuffer<f32>,
                       xq_mid: &mut DeviceBuffer<u8>,
                       out: &mut DeviceBuffer<f32>|
         -> eyre::Result<()> {
            group_count.fill_zero_async(stream)?;
            gb.launch(
                stream, group_count, members, &sel, b as u32, TOPK as u32,
                N_EXPERT as u32, max_per_expert,
            )?;
            tiled.gate_up_swiglu_reg(
                stream, mid, &w.gate_all, &w.up_all, &xq, &ew, group_count, members,
                w.gate_stride as u32, w.up_stride as u32, TOPK as u32, max_per_expert,
                0.0, FF_EXP as u32, n_blk_hidden, N_EXPERT as u32,
            )?;
            q8k.launch(stream, xq_mid, mid, b as u32 * TOPK as u32 * n_blk_mid)?;
            out.fill_zero_async(stream)?;
            match w.down_dt {
                GgufType::Q6_K => tiled.down_reg_q6k(
                    stream, out, &w.down_all, xq_mid, group_count, members,
                    w.down_stride as u32, xq_slot_stride, TOPK as u32, max_per_expert,
                    HIDDEN as u32, n_blk_mid, N_EXPERT as u32,
                )?,
                _ => tiled.down(
                    stream, w.down_dt, out, &w.down_all, xq_mid, group_count, members,
                    w.down_stride as u32, xq_slot_stride, TOPK as u32, max_per_expert,
                    HIDDEN as u32, n_blk_mid, N_EXPERT as u32,
                )?,
            }
            Ok(())
        };

        // column-tiled path (NT_COL members staged per barrier).
        let run_col = |stream: &Stream,
                       group_count: &mut DeviceBuffer<i32>,
                       members: &mut DeviceBuffer<i32>,
                       mid: &mut DeviceBuffer<f32>,
                       xq_mid: &mut DeviceBuffer<u8>,
                       out: &mut DeviceBuffer<f32>|
         -> eyre::Result<()> {
            group_count.fill_zero_async(stream)?;
            gb.launch(
                stream, group_count, members, &sel, b as u32, TOPK as u32,
                N_EXPERT as u32, max_per_expert,
            )?;
            tiled.gate_up_swiglu_reg_col(
                stream, mid, &w.gate_all, &w.up_all, &xq, &ew, group_count, members,
                w.gate_stride as u32, w.up_stride as u32, TOPK as u32, max_per_expert,
                0.0, FF_EXP as u32, n_blk_hidden, N_EXPERT as u32,
            )?;
            q8k.launch(stream, xq_mid, mid, b as u32 * TOPK as u32 * n_blk_mid)?;
            out.fill_zero_async(stream)?;
            match w.down_dt {
                GgufType::Q6_K => tiled.down_reg_q6k_w32_col(
                    stream, out, &w.down_all, xq_mid, group_count, members,
                    w.down_stride as u32, xq_slot_stride, TOPK as u32, max_per_expert,
                    HIDDEN as u32, n_blk_mid, N_EXPERT as u32,
                )?,
                GgufType::Q4_K => tiled.down_reg_q4k_w32_col(
                    stream, out, &w.down_all, xq_mid, group_count, members,
                    w.down_stride as u32, xq_slot_stride, TOPK as u32, max_per_expert,
                    HIDDEN as u32, n_blk_mid, N_EXPERT as u32,
                )?,
                _ => tiled.down(
                    stream, w.down_dt, out, &w.down_all, xq_mid, group_count, members,
                    w.down_stride as u32, xq_slot_stride, TOPK as u32, max_per_expert,
                    HIDDEN as u32, n_blk_mid, N_EXPERT as u32,
                )?,
            }
            Ok(())
        };

        // ---- parity check (once) ----
        if first {
            first = false;
            run_bxn(&stream, &mut mid, &mut xq_mid, &mut out)?;
            stream.synchronize()?;
            let mut out_bxn = vec![0f32; b * HIDDEN];
            out.copy_to_host(&mut out_bxn)?;

            run_tiled(&stream, &mut group_count, &mut members, &mut mid, &mut xq_mid, &mut out)?;
            stream.synchronize()?;
            let mut out_tiled = vec![0f32; b * HIDDEN];
            out.copy_to_host(&mut out_tiled)?;

            run_reg(&stream, &mut group_count, &mut members, &mut mid, &mut xq_mid, &mut out)?;
            stream.synchronize()?;
            let mut out_reg = vec![0f32; b * HIDDEN];
            out.copy_to_host(&mut out_reg)?;

            run_col(&stream, &mut group_count, &mut members, &mut mid, &mut xq_mid, &mut out)?;
            stream.synchronize()?;
            let mut out_col = vec![0f32; b * HIDDEN];
            out.copy_to_host(&mut out_col)?;

            let s_bxn: f64 = out_bxn.iter().map(|&x| x as f64).sum();
            let s_tiled: f64 = out_tiled.iter().map(|&x| x as f64).sum();
            let s_reg: f64 = out_reg.iter().map(|&x| x as f64).sum();
            let s_col: f64 = out_col.iter().map(|&x| x as f64).sum();
            let (max_r, mean_r) = rel_err(&out_bxn, &out_tiled);
            let (max_rr, mean_rr) = rel_err(&out_bxn, &out_reg);
            let (max_rc, mean_rc) = rel_err(&out_bxn, &out_col);
            println!("\n=== PARITY (B={b}, real L{} weights) ===", test_layer());
            println!("  sum   bxn={s_bxn:.5}  tiled={s_tiled:.5}  reg={s_reg:.5}  col={s_col:.5}");
            println!("  tiled rel max={max_r:.3e} mean={mean_r:.3e}");
            println!("  reg   rel max={max_rr:.3e} mean={mean_rr:.3e}");
            println!("  col   rel max={max_rc:.3e} mean={mean_rc:.3e}");
            assert!(max_r < 5e-3, "tiled vs bxn max rel err {max_r:.3e} too high");
            assert!(max_rr < 5e-3, "reg vs bxn max rel err {max_rr:.3e} too high");
            assert!(max_rc < 5e-3, "col vs bxn max rel err {max_rc:.3e} too high");
            println!("  [OK] tiled & reg & col ≈ bxn within atomic-reorder tolerance");
        }

        // ---- timing ----
        const WARM: usize = 3;
        const REPS: usize = 20;
        // BxN
        for _ in 0..WARM {
            run_bxn(&stream, &mut mid, &mut xq_mid, &mut out)?;
        }
        stream.synchronize()?;
        let t0 = std::time::Instant::now();
        for _ in 0..REPS {
            run_bxn(&stream, &mut mid, &mut xq_mid, &mut out)?;
        }
        stream.synchronize()?;
        let bxn_us = t0.elapsed().as_secs_f64() * 1e6 / REPS as f64;

        // tiled
        for _ in 0..WARM {
            run_tiled(&stream, &mut group_count, &mut members, &mut mid, &mut xq_mid, &mut out)?;
        }
        stream.synchronize()?;
        let t1 = std::time::Instant::now();
        for _ in 0..REPS {
            run_tiled(&stream, &mut group_count, &mut members, &mut mid, &mut xq_mid, &mut out)?;
        }
        stream.synchronize()?;
        let tiled_us = t1.elapsed().as_secs_f64() * 1e6 / REPS as f64;

        // register-tiled
        for _ in 0..WARM {
            run_reg(&stream, &mut group_count, &mut members, &mut mid, &mut xq_mid, &mut out)?;
        }
        stream.synchronize()?;
        let t2 = std::time::Instant::now();
        for _ in 0..REPS {
            run_reg(&stream, &mut group_count, &mut members, &mut mid, &mut xq_mid, &mut out)?;
        }
        stream.synchronize()?;
        let reg_us = t2.elapsed().as_secs_f64() * 1e6 / REPS as f64;

        // column-tiled
        for _ in 0..WARM {
            run_col(&stream, &mut group_count, &mut members, &mut mid, &mut xq_mid, &mut out)?;
        }
        stream.synchronize()?;
        let t3 = std::time::Instant::now();
        for _ in 0..REPS {
            run_col(&stream, &mut group_count, &mut members, &mut mid, &mut xq_mid, &mut out)?;
        }
        stream.synchronize()?;
        let col_us = t3.elapsed().as_secs_f64() * 1e6 / REPS as f64;

        // distinct experts hit (host, from sel_h) → BW roofline for tiled
        let mut hit = vec![false; N_EXPERT];
        for &e in &sel_h {
            hit[e as usize] = true;
        }
        let distinct = hit.iter().filter(|&&x| x).count();
        let active_bytes = distinct as f64 * per_expert_bytes;
        let roof_us = active_bytes / ig_bw * 1e6; // whole-chunk weight read
        let roof_us_per_tok = roof_us / b as f64;

        println!(
            "\n=== B={b}  (distinct experts hit = {distinct}/{N_EXPERT}) ===\n  \
             BxN       : {bxn_us:8.1} µs/chunk = {:6.2} µs/token\n  \
             tiled     : {tiled_us:8.1} µs/chunk = {:6.2} µs/token   (speedup {:.2}x vs BxN)\n  \
             tiled-reg : {reg_us:8.1} µs/chunk = {:6.2} µs/token   (speedup {:.2}x vs BxN, {:.2}x vs tiled)\n  \
             col-tiled : {col_us:8.1} µs/chunk = {:6.2} µs/token   (speedup {:.2}x vs BxN, {:.2}x vs reg)\n  \
             BW roofline (weight read once): {roof_us:7.1} µs/chunk = {roof_us_per_tok:5.2} µs/token   \
             reg is {:.0}% of roofline, col is {:.0}% of roofline",
            bxn_us / b as f64,
            tiled_us / b as f64,
            bxn_us / tiled_us,
            reg_us / b as f64,
            bxn_us / reg_us,
            tiled_us / reg_us,
            col_us / b as f64,
            bxn_us / col_us,
            reg_us / col_us,
            roof_us / reg_us * 100.0,
            roof_us / col_us * 100.0,
        );
    }
    Ok(())
}
