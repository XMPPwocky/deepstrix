//! Packed FP8 compressed-KV format: bit-exactness proof against the f16
//! path it replaces (`docs/FP8_KV_IMPL_2026-09.md`, step 1-2 gates).
//!
//! Runs on EVERY HIP device present (production reads the dGPU; the
//! ds4-dump oracles run on the iGPU), with KiB-sized buffers, so it is
//! safe beside the live server. No model load.
//!
//! 1. table check: the arithmetic E4M3 decode equals the __constant__
//!    table for all 128 indices.
//! 2. exhaustive expansion: every (code, e) over the full i8 field through
//!    the OLD path (`dsv4_e4m3fn_dequant * ldexpf(1,e)` then
//!    `__float2half_rn`) equals the NEW expand — bit-identical u16.
//! 3. chain oracle on random rows: old chain
//!    (`fp8_e4m3fn_quantize -> f16_roundtrip -> comp_kv_append`) versus new
//!    chain (`comp_kv_append_fp8 -> {expand, gather}`) — u16-exact, single
//!    and batched, plus the head shadow, the host unpack reference, and the
//!    host f16 -> packed recovery used by the v3 snapshot conversion.
//!
//! `cargo test -p v4flash-kernels --release --test fp8_kv_format -- --nocapture`

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::comp_kv_fp8::{
    expand_half_bits_host, pack_row_from_f16_host, unpack_row_host, FP8_KV_HEAD_DIM,
    FP8_KV_HEAD_ROWS, FP8_KV_N_NOPE, FP8_KV_ROW_BYTES,
};
use v4flash_kernels::{CompKvAppend, CompKvFp8, F16Roundtrip, Fp8E4m3fnQuantize};

const HEAD_DIM: u32 = FP8_KV_HEAD_DIM as u32;
const N_NOPE: u32 = FP8_KV_N_NOPE as u32;

/// Deterministic xorshift so failures reproduce.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Rows with realistic and adversarial block statistics: per-block
/// magnitude from 1e-7 (below the 1e-4 amax floor -> e = -22, values that
/// flush in f16) up to 6e4 (near f16 max), zero blocks, exact powers of
/// two (the log2f corner), values that snap to code 0 with either sign,
/// and RoPE tails with -0.0 / subnormals / large values.
fn gen_rows(rng: &mut Rng, n_rows: usize) -> Vec<f32> {
    let mut rows = vec![0f32; n_rows * FP8_KV_HEAD_DIM];
    for r in 0..n_rows {
        let row = &mut rows[r * FP8_KV_HEAD_DIM..(r + 1) * FP8_KV_HEAD_DIM];
        for blk in 0..7 {
            let kind = rng.next() % 10;
            let mag: f32 = match kind {
                0 => 0.0,
                1 => 1e-7,
                2 => 1e-4 * (0.5 + rng.unit()),
                3 => 2f32.powi((rng.next() % 30) as i32 - 20), // exact power of two amax
                4 => 6.0e4,
                _ => 10f32.powf(rng.unit() * 6.0 - 4.0),
            };
            for j in 0..64 {
                let u = rng.unit() * 2.0 - 1.0;
                let v = match rng.next() % 8 {
                    0 => 0.0,
                    1 => -0.0,
                    2 => u * mag * 1e-3, // snaps to code 0, sign preserved
                    3 => mag * if u < 0.0 { -1.0 } else { 1.0 }, // the amax itself
                    _ => u * mag,
                };
                row[blk * 64 + j] = v;
            }
        }
        for j in 0..64 {
            let u = rng.unit() * 2.0 - 1.0;
            row[FP8_KV_N_NOPE + j] = match rng.next() % 6 {
                0 => -0.0,
                1 => 3.0e-8 * u, // f16 subnormal range
                2 => 6.0e4 * u,
                _ => u * 10f32.powf(rng.unit() * 4.0 - 2.0),
            };
        }
    }
    rows
}

struct Kernels {
    fp8: Fp8E4m3fnQuantize,
    f16rt: F16Roundtrip,
    append: CompKvAppend,
    new: CompKvFp8,
}

fn table_check(k: &Kernels, dev: i32, stream: &Stream) -> eyre::Result<()> {
    let mut t = DeviceBuffer::<f32>::new(dev, 128)?;
    let mut d = DeviceBuffer::<f32>::new(dev, 128)?;
    k.new.launch_table_check(stream, &mut t, &mut d)?;
    stream.synchronize()?;
    let mut th = vec![0f32; 128];
    let mut dh = vec![0f32; 128];
    t.copy_to_host(&mut th)?;
    d.copy_to_host(&mut dh)?;
    let mut bad = 0;
    for i in 0..127 {
        if th[i].to_bits() != dh[i].to_bits() {
            bad += 1;
            eprintln!("  table[{i}] = {} vs decode {}", th[i], dh[i]);
        }
        // Host magnitude too.
        let hm = v4flash_kernels::comp_kv_fp8::e4m3fn_magnitude(i as u8);
        if hm.to_bits() != th[i].to_bits() {
            bad += 1;
            eprintln!("  host magnitude[{i}] = {hm} vs table {}", th[i]);
        }
    }
    if bad != 0 {
        return Err(eyre!("table check: {bad} mismatches"));
    }
    println!("  table check: 127/127 device decode == table == host");
    Ok(())
}

/// Host reference == device expand over the full i8 exponent field.
fn exhaustive_host_vs_device(k: &Kernels, dev: i32, stream: &Stream) -> eyre::Result<()> {
    let e_lo = -128i32;
    let n_e = 256u32;
    let n = (n_e as usize) * 256;
    let mut out = DeviceBuffer::<u16>::new(dev, n)?;
    k.new.launch_expand_exhaustive(stream, &mut out, e_lo, n_e)?;
    stream.synchronize()?;
    let mut oh = vec![0u16; n];
    out.copy_to_host(&mut oh)?;
    let mut bad = 0;
    for i in 0..n {
        let code = (i & 255) as u8;
        let e = e_lo + (i >> 8) as i32;
        if code & 0x7F == 127 {
            continue;
        }
        let hb = expand_half_bits_host(code, e as i8);
        if hb != oh[i] {
            bad += 1;
            if bad <= 10 {
                eprintln!("  HOST code {code:#04x} e {e}: device {:#06x} host {hb:#06x}", oh[i]);
            }
        }
    }
    if bad != 0 {
        return Err(eyre!("host reference vs device expand: {bad} mismatches"));
    }
    println!("  host reference == device expand for all {} (code, e) over the i8 field", n - 512);
    Ok(())
}

/// Every reachable (code, e) through the REAL old kernels versus the new
/// chain. The quantiser picks e from the block amax, so each 64-block
/// carries an anchor `300 * 2^e` (safely inside (224, 448] * 2^e, away
/// from the exact-power-of-two log2f corner) plus 63 probe values
/// `sign * table[idx] * 2^e`; idx 0 negative is a tiny negative (a value
/// that snaps to code 0 with the sign bit). e runs from the 1e-4 amax
/// floor (-22) to past f16 overflow (30).
fn exhaustive_old_vs_new(k: &Kernels, dev: i32, stream: &Stream) -> eyre::Result<()> {
    let e_lo = -22i32;
    let e_hi = 30i32;
    let mut rows: Vec<f32> = Vec::new();
    let mut expect: Vec<(usize, usize, u8, i32)> = Vec::new(); // (row, dim, code, e)
    let mut cur = vec![0f32; FP8_KV_HEAD_DIM];
    let mut blk = 0usize;
    let mut slot = 0usize;
    for e in e_lo..=e_hi {
        let scale = 2f32.powi(e);
        for code in 0u8..=255 {
            let idx = code & 0x7F;
            if idx == 127 {
                continue;
            }
            let mag = if idx == 0 { 1.0e-30f32 * scale } else { v4flash_kernels::comp_kv_fp8::e4m3fn_magnitude(idx) * scale };
            let v = if code & 0x80 != 0 { -mag } else { mag };
            if slot == 0 {
                cur[blk * 64] = 300.0 * scale; // anchor -> this block's e
                slot = 1;
            }
            cur[blk * 64 + slot] = v;
            expect.push((rows.len() / FP8_KV_HEAD_DIM, blk * 64 + slot, code, e));
            slot += 1;
            if slot == 64 {
                slot = 0;
                blk += 1;
                if blk == 7 {
                    rows.extend_from_slice(&cur);
                    cur.iter_mut().for_each(|x| *x = 0.0);
                    blk = 0;
                }
            }
        }
        // Start a fresh block for the next e (the anchor sets e per block).
        if slot != 0 {
            slot = 0;
            blk += 1;
            if blk == 7 {
                rows.extend_from_slice(&cur);
                cur.iter_mut().for_each(|x| *x = 0.0);
                blk = 0;
            }
        }
    }
    if blk != 0 || slot != 0 {
        rows.extend_from_slice(&cur);
    }
    let n_rows = rows.len() / FP8_KV_HEAD_DIM;
    let old = old_chain_batched(k, dev, stream, &rows)?;
    let mut x = DeviceBuffer::<f32>::new(dev, rows.len())?;
    x.copy_from_host(&rows)?;
    let mut packed = DeviceBuffer::<u8>::new(dev, n_rows * FP8_KV_ROW_BYTES)?;
    let mut head = DeviceBuffer::<u16>::new(dev, FP8_KV_HEAD_ROWS * FP8_KV_HEAD_DIM)?;
    k.new.launch_append_batched(stream, &mut packed, &mut head, &x, 0, n_rows as u32, FP8_KV_HEAD_ROWS as u32)?;
    let mut exp = DeviceBuffer::<u16>::new(dev, rows.len())?;
    k.new.launch_expand(stream, &mut exp, &packed, n_rows as u32)?;
    stream.synchronize()?;
    let mut new = vec![0u16; rows.len()];
    exp.copy_to_host(&mut new)?;
    let mut packed_h = vec![0u8; n_rows * FP8_KV_ROW_BYTES];
    packed.copy_to_host(&mut packed_h)?;
    let mut bad = 0usize;
    let mut code_bad = 0usize;
    let mut e_bad = 0usize;
    for &(r, d, code, e) in &expect {
        let i = r * FP8_KV_HEAD_DIM + d;
        if old[i] != new[i] {
            bad += 1;
            if bad <= 10 {
                eprintln!("  code {code:#04x} e {e}: old {:#06x} new {:#06x} (in {})", old[i], new[i], rows[i]);
            }
        }
        // The producer must also have emitted exactly this (code, e) —
        // except where the old value overflowed f16 (both sides inf) or
        // where several codes round to the same f16 (subnormal grid): then
        // only the expansion must match, which the check above covers.
        let got_code = packed_h[r * FP8_KV_ROW_BYTES + d];
        let got_e = packed_h[r * FP8_KV_ROW_BYTES + 448 + (d >> 6)] as i8 as i32;
        if got_e != e {
            e_bad += 1;
            if e_bad <= 5 {
                eprintln!("  block exponent: expected {e} got {got_e} (row {r} dim {d})");
            }
        }
        if got_code != code && old[i] & 0x7FFF != 0x7C00 && (old[i] & 0x7C00) != 0 {
            code_bad += 1;
            if code_bad <= 5 {
                eprintln!("  code: expected {code:#04x} got {got_code:#04x} at e {e} (old {:#06x})", old[i]);
            }
        }
    }
    for i in 0..rows.len() {
        if old[i] != new[i] {
            bad += 1; // anchors / zero fill too
        }
    }
    if bad != 0 || e_bad != 0 || code_bad != 0 {
        return Err(eyre!(
            "exhaustive old-vs-new: {bad} value mismatches, {e_bad} exponent mismatches, {code_bad} code mismatches ({} probes, {n_rows} rows)",
            expect.len()
        ));
    }
    println!(
        "  exhaustive old-vs-new: {} (code, e) probes over e in [{e_lo}, {e_hi}] through the real kernels, bit-identical; producer emitted the expected (code, e) for every normal-range value",
        expect.len()
    );
    Ok(())
}

fn old_chain_batched(k: &Kernels, dev: i32, stream: &Stream, rows: &[f32]) -> eyre::Result<Vec<u16>> {
    let n_rows = rows.len() / FP8_KV_HEAD_DIM;
    let mut x = DeviceBuffer::<f32>::new(dev, rows.len())?;
    x.copy_from_host(rows)?;
    let mut cache = DeviceBuffer::<u16>::new(dev, rows.len())?;
    k.fp8.launch_batched(stream, &mut x, N_NOPE, HEAD_DIM, n_rows as u32)?;
    k.f16rt.launch(stream, &mut x, rows.len() as u32)?;
    k.append.launch_batched(stream, &mut cache, &x, 0, HEAD_DIM, n_rows as u32)?;
    stream.synchronize()?;
    let mut out = vec![0u16; rows.len()];
    cache.copy_to_host(&mut out)?;
    Ok(out)
}

fn old_chain_single(k: &Kernels, dev: i32, stream: &Stream, rows: &[f32]) -> eyre::Result<Vec<u16>> {
    let n_rows = rows.len() / FP8_KV_HEAD_DIM;
    let mut cache = DeviceBuffer::<u16>::new(dev, rows.len())?;
    for r in 0..n_rows {
        let mut x = DeviceBuffer::<f32>::new(dev, FP8_KV_HEAD_DIM)?;
        x.copy_from_host(&rows[r * FP8_KV_HEAD_DIM..(r + 1) * FP8_KV_HEAD_DIM])?;
        k.fp8.launch(stream, &mut x, N_NOPE)?;
        k.f16rt.launch(stream, &mut x, HEAD_DIM)?;
        k.append.launch(stream, &mut cache, &x, r as u32, HEAD_DIM)?;
    }
    stream.synchronize()?;
    let mut out = vec![0u16; rows.len()];
    cache.copy_to_host(&mut out)?;
    Ok(out)
}

fn chain_oracle(k: &Kernels, dev: i32, stream: &Stream, rng: &mut Rng) -> eyre::Result<()> {
    let n_rows = 700usize; // > FP8_KV_HEAD_ROWS so the shadow cut-off is exercised
    let rows = gen_rows(rng, n_rows);
    let old_b = old_chain_batched(k, dev, stream, &rows)?;
    let old_s = old_chain_single(k, dev, stream, &rows[..64 * FP8_KV_HEAD_DIM])?;
    if old_s[..] != old_b[..64 * FP8_KV_HEAD_DIM] {
        return Err(eyre!("old chain: single vs batched differ (test harness problem)"));
    }

    // New chain, batched producer.
    let mut x = DeviceBuffer::<f32>::new(dev, rows.len())?;
    x.copy_from_host(&rows)?;
    let mut packed = DeviceBuffer::<u8>::new(dev, n_rows * FP8_KV_ROW_BYTES)?;
    let mut head = DeviceBuffer::<u16>::new(dev, FP8_KV_HEAD_ROWS * FP8_KV_HEAD_DIM)?;
    // Poison so an unwritten byte shows up.
    packed.copy_from_host(&vec![0xA5u8; n_rows * FP8_KV_ROW_BYTES])?;
    head.copy_from_host(&vec![0xBEEFu16; FP8_KV_HEAD_ROWS * FP8_KV_HEAD_DIM])?;
    // Two batches so n_comp_start != 0 is covered.
    let split = 300u32;
    k.new.launch_append_batched(stream, &mut packed, &mut head, &x, 0, split, FP8_KV_HEAD_ROWS as u32)?;
    let x_tail = x.slice_view((split as usize) * FP8_KV_HEAD_DIM, (n_rows - split as usize) * FP8_KV_HEAD_DIM);
    k.new.launch_append_batched(
        stream, &mut packed, &mut head, &x_tail, split, n_rows as u32 - split, FP8_KV_HEAD_ROWS as u32,
    )?;
    // Expand all rows.
    let mut exp = DeviceBuffer::<u16>::new(dev, rows.len())?;
    k.new.launch_expand(stream, &mut exp, &packed, n_rows as u32)?;
    stream.synchronize()?;
    let mut new_all = vec![0u16; rows.len()];
    exp.copy_to_host(&mut new_all)?;
    let mut head_h = vec![0u16; FP8_KV_HEAD_ROWS * FP8_KV_HEAD_DIM];
    head.copy_to_host(&mut head_h)?;
    let mut packed_h = vec![0u8; n_rows * FP8_KV_ROW_BYTES];
    packed.copy_to_host(&mut packed_h)?;

    let mut bad = 0usize;
    for i in 0..rows.len() {
        if old_b[i] != new_all[i] {
            bad += 1;
            if bad <= 10 {
                let r = i / FP8_KV_HEAD_DIM;
                let d = i % FP8_KV_HEAD_DIM;
                eprintln!(
                    "  row {r} dim {d}: in {} old {:#06x} new {:#06x} code {:#04x} e {}",
                    rows[i], old_b[i], new_all[i], packed_h[r * FP8_KV_ROW_BYTES + d.min(FP8_KV_N_NOPE)],
                    packed_h[r * FP8_KV_ROW_BYTES + 448 + (d >> 6).min(6)] as i8
                );
            }
        }
    }
    if bad != 0 {
        return Err(eyre!("chain oracle (batched producer -> expand): {bad} / {} mismatches", rows.len()));
    }
    println!("  chain oracle: {n_rows} rows x 512 u16-exact (batched producer -> expand)");

    // Head shadow == old rows for row < 512.
    if head_h[..] != old_b[..FP8_KV_HEAD_ROWS * FP8_KV_HEAD_DIM] {
        return Err(eyre!("head shadow differs from the f16 cache prefix"));
    }
    println!("  head shadow: {FP8_KV_HEAD_ROWS} rows identical to the f16 prefix");

    // Pad bytes deterministic.
    for r in 0..n_rows {
        let row = &packed_h[r * FP8_KV_ROW_BYTES..(r + 1) * FP8_KV_ROW_BYTES];
        if row[455] != 0 || row[584..592].iter().any(|&b| b != 0) {
            return Err(eyre!("row {r}: pad bytes not zero"));
        }
    }

    // Single-row producer == batched producer.
    let mut packed1 = DeviceBuffer::<u8>::new(dev, 64 * FP8_KV_ROW_BYTES)?;
    let mut head1 = DeviceBuffer::<u16>::new(dev, FP8_KV_HEAD_ROWS * FP8_KV_HEAD_DIM)?;
    for r in 0..64usize {
        let xr = x.slice_view(r * FP8_KV_HEAD_DIM, FP8_KV_HEAD_DIM);
        k.new.launch_append(stream, &mut packed1, &mut head1, &xr, r as u32, FP8_KV_HEAD_ROWS as u32)?;
    }
    stream.synchronize()?;
    let mut p1 = vec![0u8; 64 * FP8_KV_ROW_BYTES];
    packed1.copy_to_host(&mut p1)?;
    if p1[..] != packed_h[..64 * FP8_KV_ROW_BYTES] {
        return Err(eyre!("single-row producer bytes differ from batched"));
    }
    println!("  single-row producer: 64 rows byte-identical to batched");

    // Gather (single + batched) with a random selection incl. sentinels.
    let top_k = 512u32;
    let batch = 3u32;
    let mut sel_h = vec![0i32; (batch * top_k) as usize];
    for s in sel_h.iter_mut() {
        *s = if rng.next() % 16 == 0 { -1 } else { (rng.next() % n_rows as u64) as i32 };
    }
    let mut sel = DeviceBuffer::<i32>::new(dev, sel_h.len())?;
    sel.copy_from_host(&sel_h)?;
    let mut act_b = DeviceBuffer::<u16>::new(dev, (batch * top_k) as usize * FP8_KV_HEAD_DIM)?;
    act_b.copy_from_host(&vec![0xDEADu16; (batch * top_k) as usize * FP8_KV_HEAD_DIM])?;
    k.new.launch_gather_batched(stream, &mut act_b, &packed, &sel, top_k, batch)?;
    let mut act_s = DeviceBuffer::<u16>::new(dev, top_k as usize * FP8_KV_HEAD_DIM)?;
    act_s.copy_from_host(&vec![0xDEADu16; top_k as usize * FP8_KV_HEAD_DIM])?;
    let sel0 = sel.slice_view(0, top_k as usize);
    k.new.launch_gather(stream, &mut act_s, &packed, &sel0, top_k)?;
    stream.synchronize()?;
    let mut ab = vec![0u16; act_b.len()];
    act_b.copy_to_host(&mut ab)?;
    let mut as_ = vec![0u16; act_s.len()];
    act_s.copy_to_host(&mut as_)?;
    for b in 0..batch as usize {
        for i in 0..top_k as usize {
            let s = sel_h[b * top_k as usize + i];
            let dst = &ab[(b * top_k as usize + i) * FP8_KV_HEAD_DIM..(b * top_k as usize + i + 1) * FP8_KV_HEAD_DIM];
            if s < 0 {
                if dst.iter().any(|&v| v != 0xDEAD) {
                    return Err(eyre!("gather_batched wrote a sentinel slot (b={b} i={i})"));
                }
            } else {
                let src = &old_b[s as usize * FP8_KV_HEAD_DIM..(s as usize + 1) * FP8_KV_HEAD_DIM];
                if dst != src {
                    return Err(eyre!("gather_batched b={b} i={i} (row {s}) differs from f16 cache row"));
                }
            }
            if b == 0 {
                let d1 = &as_[i * FP8_KV_HEAD_DIM..(i + 1) * FP8_KV_HEAD_DIM];
                if d1 != dst {
                    return Err(eyre!("gather single vs batched differ at i={i}"));
                }
            }
        }
    }
    println!("  gather: {batch} x {top_k} selections (with sentinels) identical to f16 cache rows");

    // Host unpack reference == device expand.
    let mut hu = vec![0u16; FP8_KV_HEAD_DIM];
    let mut host_bad = 0;
    for r in 0..n_rows {
        unpack_row_host(&packed_h[r * FP8_KV_ROW_BYTES..], &mut hu);
        if hu[..] != new_all[r * FP8_KV_HEAD_DIM..(r + 1) * FP8_KV_HEAD_DIM] {
            host_bad += 1;
        }
    }
    if host_bad != 0 {
        return Err(eyre!("host unpack: {host_bad} rows differ from device expand"));
    }
    println!("  host unpack reference: {n_rows}/{n_rows} rows identical to device expand");

    // v3 conversion path: recover packed rows from the OLD f16 rows on the
    // host, expand on device, compare to the old rows. Rows the host
    // refuses are reported (expected only for flushed blocks); rows it
    // accepts MUST round-trip exactly.
    let mut rec = vec![0u8; n_rows * FP8_KV_ROW_BYTES];
    let mut refused = 0usize;
    let mut accepted_rows = Vec::new();
    for r in 0..n_rows {
        let row = &old_b[r * FP8_KV_HEAD_DIM..(r + 1) * FP8_KV_HEAD_DIM];
        if pack_row_from_f16_host(row, &mut rec[r * FP8_KV_ROW_BYTES..(r + 1) * FP8_KV_ROW_BYTES]).is_some() {
            accepted_rows.push(r);
        } else {
            refused += 1;
        }
    }
    let mut recd = DeviceBuffer::<u8>::new(dev, rec.len())?;
    recd.copy_from_host(&rec)?;
    let mut rexp = DeviceBuffer::<u16>::new(dev, rows.len())?;
    k.new.launch_expand(stream, &mut rexp, &recd, n_rows as u32)?;
    stream.synchronize()?;
    let mut rh = vec![0u16; rows.len()];
    rexp.copy_to_host(&mut rh)?;
    let mut conv_bad = 0usize;
    for &r in &accepted_rows {
        if rh[r * FP8_KV_HEAD_DIM..(r + 1) * FP8_KV_HEAD_DIM] != old_b[r * FP8_KV_HEAD_DIM..(r + 1) * FP8_KV_HEAD_DIM] {
            conv_bad += 1;
        }
    }
    if conv_bad != 0 {
        return Err(eyre!("v3 conversion: {conv_bad} host-accepted rows do not round-trip on device"));
    }
    println!(
        "  v3 conversion: {} rows recovered and device-verified exact, {refused} refused (flushed blocks)",
        accepted_rows.len()
    );
    Ok(())
}

#[test]
fn fp8_kv_format_bit_exact() -> eyre::Result<()> {
    install_panic_handler()?;
    let devices = Device::all()?;
    if devices.is_empty() {
        return Err(eyre!("no HIP devices"));
    }
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for d in &devices {
        d.set_current()?;
        let props = d.properties()?;
        let arch = props.gcn_arch_name.clone();
        println!("device {} ({arch})", d.id);
        let stream = Stream::new(d.id)?;
        let k = Kernels {
            fp8: Fp8E4m3fnQuantize::for_arch(&arch)?,
            f16rt: F16Roundtrip::for_arch(&arch)?,
            append: CompKvAppend::for_arch(&arch)?,
            new: CompKvFp8::for_arch(&arch)?,
        };
        table_check(&k, d.id, &stream)?;
        exhaustive_host_vs_device(&k, d.id, &stream)?;
        exhaustive_old_vs_new(&k, d.id, &stream)?;
        for _ in 0..3 {
            chain_oracle(&k, d.id, &stream, &mut rng)?;
        }
    }
    Ok(())
}
