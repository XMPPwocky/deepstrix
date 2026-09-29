//! `V41_PUSH_XQ` precondition: `q8_k_quantize` produces the SAME BYTES on the
//! dGPU (gfx1201) and the iGPU (gfx1151). With the knob on, the iGPU's MoE
//! consumes the dGPU's Q8_K of `ffn_input_norm` instead of quantising the f32
//! rows itself, so any byte difference is a numerics change on box 1's routed
//! experts. (Box 2 already consumes the dGPU's bytes -- `pre_moe_route` relies
//! on the same equality.)
//!
//! Synthetic inputs, a few MB per device, no model load. Rows 1..512 of
//! N_EMBD = 5120 (20 blocks of 256) with the awkward blocks mixed in: all-zero,
//! +a/-a ties at several positions, a lone spike, an all-tiny block (max ~3e-39,
//! so -127/max overflows to -inf: out of domain for RMSNorm output, triage a
//! failure on kind 4 alone as such), huge magnitudes, a negative maximum,
//! constant blocks, values on rint's .5 edge, and a normal max with subnormal
//! elements (kind 9, where a flush-to-zero difference would show).
//!
//! Test 2 (`push_xq_transport_matches_igpu_quantize`) runs the PRODUCTION data
//! path end to end -- dGPU quantize on a compute stream, event, u8 peer push on
//! a second dGPU stream (`het::sync::peer_push_u8`), event, D2D copy on the
//! iGPU stream -- and compares the landed bytes with the iGPU's own quantize of
//! the same rows, for many iterations with fresh data into reused buffers, so a
//! stale or zero read (a failed peer copy reads zeros silently here) shows.
//!
//! Run (server DOWN -- touches both GPUs):
//! `cargo test --release --features v41 -p v4flash-kernels --test
//! q8k_cross_device -- --ignored --test-threads=1 --nocapture`.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::Q8KQuantize;

const QK_K: usize = 256;
const BLOCK_BYTES: usize = 292;
const N_EMBD: usize = 5120;

fn pick(prefix: &str) -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with(prefix) {
            return Ok(d);
        }
    }
    Err(eyre!("no {prefix} device"))
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
}

/// One 256-value block of kind `k` (0 = ordinary activations).
fn block(rng: &mut Lcg, k: u32) -> Vec<f32> {
    let mut v: Vec<f32> = (0..QK_K).map(|_| (rng.unit() * 2.0 - 1.0) * 3.0).collect();
    match k {
        1 => v.iter_mut().for_each(|x| *x = 0.0),
        2 => {
            // +a / -a ties for the max at several positions: the tree
            // reduction's tie-break decides the sign of `iscale`.
            let a = 7.25;
            for (i, &p) in [3usize, 64, 130, 255].iter().enumerate() {
                v[p] = if i % 2 == 0 { a } else { -a };
            }
        }
        3 => v[rng.next() as usize % QK_K] = 900.0,
        4 => v.iter_mut().for_each(|x| *x *= 1e-39), // subnormal
        5 => v.iter_mut().for_each(|x| *x *= 1e30),
        6 => {
            let p = rng.next() as usize % QK_K;
            v[p] = -40.0;
        }
        7 => v.iter_mut().for_each(|x| *x = 0.3),
        9 => {
            v[7] = 50.0;
            for (i, x) in v.iter_mut().enumerate() {
                if i != 7 {
                    *x = if i % 2 == 0 { 1e-39 } else { -3e-40 };
                }
            }
        }
        8 => {
            // Values that land near n + 0.5 after scaling by -127/max.
            let max = 127.0;
            v[0] = max;
            for (i, x) in v.iter_mut().enumerate().skip(1) {
                *x = -((i % 120) as f32 + 0.5);
            }
        }
        _ => {}
    }
    v
}

fn quantize(dev: &Device, x: &[f32]) -> eyre::Result<Vec<u8>> {
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    let q = Q8KQuantize::for_arch(&arch)?;
    let stream = Stream::new(dev.id)?;
    let mut xd: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, x.len())?;
    xd.copy_from_host(x)?;
    let n_blocks = x.len() / QK_K;
    let mut out: DeviceBuffer<u8> = DeviceBuffer::new(dev.id, n_blocks * BLOCK_BYTES)?;
    // Sentinel: a block the kernel failed to write shows as a mismatch.
    out.copy_from_host(&vec![0xA5u8; n_blocks * BLOCK_BYTES])?;
    q.launch(&stream, &mut out, &xd, n_blocks as u32)?;
    stream.synchronize()?;
    let mut h = vec![0u8; n_blocks * BLOCK_BYTES];
    out.copy_to_host(&mut h)?;
    Ok(h)
}

#[test]
#[ignore]
fn q8k_quantize_bytes_match_across_dgpu_and_igpu() -> eyre::Result<()> {
    install_panic_handler()?;
    let dgpu = pick("gfx1201")?;
    let igpu = pick("gfx1151")?;
    let mut rng = Lcg(0x0808_0808_5EED);
    let mut total_blocks = 0usize;
    for &rows in &[1usize, 2, 3, 4, 7, 8, 16, 64, 511, 512] {
        let mut x = Vec::with_capacity(rows * N_EMBD);
        let mut kinds = Vec::new();
        for bi in 0..rows * (N_EMBD / QK_K) {
            // Mostly ordinary blocks, every edge kind at least once per row set.
            let kind = if bi < 10 { bi as u32 } else if rng.next() % 5 == 0 { rng.next() % 10 } else { 0 };
            kinds.push(kind);
            x.extend(block(&mut rng, kind));
        }
        let a = quantize(&dgpu, &x)?;
        let b = quantize(&igpu, &x)?;
        let bad: Vec<usize> = (0..a.len() / BLOCK_BYTES)
            .filter(|&k| a[k * BLOCK_BYTES..(k + 1) * BLOCK_BYTES] != b[k * BLOCK_BYTES..(k + 1) * BLOCK_BYTES])
            .collect();
        println!("rows {rows:4}: {} blocks, {} differ", a.len() / BLOCK_BYTES, bad.len());
        if let Some(&k) = bad.first() {
            let (da, db) = (&a[k * BLOCK_BYTES..k * BLOCK_BYTES + 4], &b[k * BLOCK_BYTES..k * BLOCK_BYTES + 4]);
            let bad_kinds: std::collections::BTreeSet<u32> = bad.iter().map(|&k| kinds[k]).collect();
            return Err(eyre!("rows {rows}: block {k} (kind {}) differs (d dgpu {da:02x?} vs igpu {db:02x?}); {} blocks differ, kinds {bad_kinds:?}", kinds[k], bad.len()));
        }
        total_blocks += a.len() / BLOCK_BYTES;
    }
    println!("OK: {total_blocks} Q8_K blocks byte-identical on gfx1201 and gfx1151");
    Ok(())
}

fn random_rows(rng: &mut Lcg, rows: usize) -> Vec<f32> {
    let mut x = Vec::with_capacity(rows * N_EMBD);
    for _ in 0..rows * (N_EMBD / QK_K) {
        let kind = if rng.next() % 7 == 0 { 1 + rng.next() % 9 } else { 0 };
        x.extend(block(rng, kind));
    }
    x
}

#[test]
#[ignore]
fn push_xq_transport_matches_igpu_quantize() -> eyre::Result<()> {
    install_panic_handler()?;
    let dgpu = pick("gfx1201")?;
    let igpu = pick("gfx1151")?;
    // Peer access both ways, as `HeterogeneousEngine::new` sets it up.
    dgpu.set_current()?;
    if !dgpu.can_access_peer(igpu)? {
        return Err(eyre!("dGPU cannot access the iGPU as a peer"));
    }
    let _ = dgpu.enable_peer_access(igpu);
    igpu.set_current()?;
    if !igpu.can_access_peer(dgpu)? {
        return Err(eyre!("iGPU cannot access the dGPU as a peer"));
    }
    let _ = igpu.enable_peer_access(dgpu);

    const MAX_ROWS: usize = 512;
    let max_bytes = MAX_ROWS * (N_EMBD / QK_K) * BLOCK_BYTES;
    dgpu.set_current()?;
    let qd = Q8KQuantize::for_arch(&dgpu.properties()?.gcn_arch_name)?;
    let d_compute = Stream::new(dgpu.id)?;
    let d_xfer = Stream::new(dgpu.id)?;
    let mut x_d: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, MAX_ROWS * N_EMBD)?;
    let mut xq_d: DeviceBuffer<u8> = DeviceBuffer::new(dgpu.id, max_bytes)?;
    // Both events are recorded on dGPU streams, so they belong to the dGPU
    // (as `LayerSyncEvents` do); the iGPU stream waits on them cross-device.
    let ready = Event::new_no_timing()?;
    let pushed = Event::new_no_timing()?;
    igpu.set_current()?;
    let qi = Q8KQuantize::for_arch(&igpu.properties()?.gcn_arch_name)?;
    let i_compute = Stream::new(igpu.id)?;
    let mut x_i: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, MAX_ROWS * N_EMBD)?;
    let mut recv_i: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, max_bytes)?;
    let mut head_i: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, max_bytes)?;
    let mut ref_i: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, max_bytes)?;

    let mut rng = Lcg(0x9E37_79B9_7F4A_7C15);
    let row_set = [1usize, 2, 3, 4, 5, 8, 4, 4, 3, 512, 1, 4, 256, 7, 4, 4];
    let mut checked = 0usize;
    for iter in 0..200usize {
        let rows = row_set[iter % row_set.len()];
        let n_blocks = rows * (N_EMBD / QK_K);
        let nbytes = n_blocks * BLOCK_BYTES;
        let x = random_rows(&mut rng, rows);
        // Sentinels differ per iteration, so a skipped or stale copy cannot
        // match by accident.
        let sentinel = vec![(iter as u8) ^ 0x5A; max_bytes];
        igpu.set_current()?;
        head_i.copy_from_host(&sentinel)?;
        recv_i.copy_from_host(&sentinel)?;
        x_i.slice_view_mut(0, x.len()).copy_from_host(&x)?;
        // Reference: the iGPU's own quantize (today's path).
        qi.launch(&i_compute, &mut ref_i, &x_i.slice_view(0, x.len()), n_blocks as u32)?;

        // dGPU: quantize on compute, event, push on xfer (source-device stream).
        dgpu.set_current()?;
        x_d.slice_view_mut(0, x.len()).copy_from_host(&x)?;
        qd.launch(&d_compute, &mut xq_d, &x_d.slice_view(0, x.len()), n_blocks as u32)?;
        ready.record(&d_compute)?;
        d_xfer.wait_event(&ready)?;
        v4flash_kernels::het::sync::peer_push_u8(
            &xq_d.slice_view(0, nbytes),
            &mut recv_i.slice_view_mut(0, nbytes),
            &d_xfer,
        )?;
        pushed.record(&d_xfer)?;

        // iGPU: wait for the push, then the same D2D copy `pre_moe_launch` does.
        igpu.set_current()?;
        i_compute.wait_event(&pushed)?;
        head_i
            .slice_view_mut(0, nbytes)
            .copy_from_buffer_async(&recv_i.slice_view(0, nbytes), &i_compute)?;
        i_compute.synchronize()?;

        let mut got = vec![0u8; nbytes];
        head_i.slice_view(0, nbytes).copy_to_host(&mut got)?;
        let mut want = vec![0u8; nbytes];
        ref_i.slice_view(0, nbytes).copy_to_host(&mut want)?;
        if got != want {
            let k = (0..n_blocks)
                .find(|&k| got[k * BLOCK_BYTES..(k + 1) * BLOCK_BYTES] != want[k * BLOCK_BYTES..(k + 1) * BLOCK_BYTES])
                .unwrap_or(0);
            let all_sentinel = got.iter().all(|&b| b == sentinel[0]);
            let all_zero = got.iter().all(|&b| b == 0);
            return Err(eyre!(
                "iter {iter} rows {rows}: landed Q8_K != iGPU quantize at block {k} (all sentinel: {all_sentinel}, all zero: {all_zero})"
            ));
        }
        checked += n_blocks;
    }
    println!("OK: {checked} Q8_K blocks through quantize(dGPU) -> push -> copy(iGPU) == quantize(iGPU)");
    Ok(())
}
