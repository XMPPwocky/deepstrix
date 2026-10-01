//! `V41_MTP_MOE_GROUPED`: the drafter's routed MoE grouped by expert
//! (`mxfp4_pair_matvec_fused_swiglu_grouped` + one q8k + `mxfp4_matvec_par_grouped`)
//! vs the per-row chain it replaces in `MtpState::moe` (B x [gate/up hetsplit,
//! q8k, down hetsplit]), on synthetic MXFP4 experts at the drafter's shapes (no
//! model load; ~600 MB of iGPU memory).
//!
//! The claim is BIT-IDENTITY by construction, so this asserts it bit for bit on
//! every buffer the chain writes: `mid` (sentinel-filled, so the written set must
//! match too, including the zeros of non-resident picks), `midq` and `ffn_out`,
//! over selection regimes that stress the grouping: one expert set in the same
//! slot order for every token, the same set ROTATED across slots (a token's 3
//! experts are summed inside each lane in slot order, so a cross-slot repeat is
//! where a by-expert reduce would re-associate), 15 distinct, a 6-expert pool,
//! a 9-expert pool (~7.8 distinct of 15, the production mean is 7.6), and picks
//! past the resident set (remap 0 = not ours). Then it times both chains on
//! rotating 9-of-32 pools (cold-ish weights: 600 MB rotates through the 32 MB
//! MALL) -- informational, the bar is the parity run.
//!
//! gfx1151 only. Run (hub DOWN): `cargo test --release --features v41 -p
//! v4flash-kernels --test mtp_moe_grouped -- --ignored --test-threads=1 --nocapture`.

use color_eyre::eyre;
use v4flash_core::gguf::GgufType;
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::config::{BLOCKS_Q8K_DOWN_IN, BLOCKS_Q8K_GATE_IN, N_EMBD, N_FF_EXP, SWIGLU_CLAMP_EXP};
use v4flash_kernels::het::dispatch;
use v4flash_kernels::het::engine::DeviceEngine;
use v4flash_kernels::het::mtp::{MTP_BLOCK, MTP_TOPK};
use v4flash_kernels::het::remote_experts::{MIDQ_BYTES_PER_SLOT, SENTINEL_EXPERT, XQ_BYTES_PER_TOKEN};
use v4flash_kernels::mxfp4_tables::SUPER_MXFP4_BYTES;
use v4flash_kernels::q8_k::BLOCK_Q8_K_BYTES;

fn pick_igpu() -> eyre::Result<Option<Device>> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(Some(d));
        }
    }
    Ok(None)
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
}

/// MXFP4 v2 super-blocks, E8M0 scales 2^(s-128) with s in [sc_lo, sc_lo + span).
fn mxfp4_weights(rng: &mut Lcg, n_experts: usize, n_rows: usize, nb: usize, sc_lo: u8, span: u32) -> Vec<u8> {
    let mut w = vec![0u8; n_experts * n_rows * nb * SUPER_MXFP4_BYTES];
    for sb in w.chunks_exact_mut(SUPER_MXFP4_BYTES) {
        for b8 in 0..8 {
            sb[128 + b8] = sc_lo + (rng.next() % span) as u8;
            for j in 0..16 {
                sb[b8 * 16 + j] = rng.next() as u8;
            }
        }
    }
    w
}

/// Q8_K blocks with a positive scale, random int8s and consistent bsums.
fn q8k_blocks(rng: &mut Lcg, n_blocks: usize) -> Vec<u8> {
    let mut out = vec![0u8; n_blocks * BLOCK_Q8_K_BYTES];
    for blk in out.chunks_exact_mut(BLOCK_Q8_K_BYTES) {
        let d = 0.001f32 + (rng.next() % 1000) as f32 * 1e-5;
        blk[..4].copy_from_slice(&d.to_le_bytes());
        let mut bsums = [0i16; 16];
        for k in 0..256 {
            let q = (rng.next() % 255) as i32 - 127;
            blk[4 + k] = q as i8 as u8;
            bsums[k / 16] += q as i16;
        }
        for (j, s) in bsums.iter().enumerate() {
            blk[260 + 2 * j..262 + 2 * j].copy_from_slice(&s.to_le_bytes());
        }
    }
    out
}

fn upload<T: Copy>(id: i32, h: &[T]) -> eyre::Result<DeviceBuffer<T>> {
    let mut d = DeviceBuffer::new(id, h.len().max(1))?;
    if !h.is_empty() {
        d.copy_from_host(h)?;
    }
    Ok(d)
}

fn bits(d: &DeviceBuffer<f32>) -> eyre::Result<Vec<u32>> {
    let mut v = vec![0f32; d.len()];
    d.copy_to_host(&mut v)?;
    Ok(v.iter().map(|x| x.to_bits()).collect())
}

fn bytes(d: &DeviceBuffer<u8>) -> eyre::Result<Vec<u8>> {
    let mut v = vec![0u8; d.len()];
    d.copy_to_host(&mut v)?;
    Ok(v)
}

const SENTINEL: f32 = f32::from_bits(0x7FC0_BEEF);

/// Drafter shapes and buffers for one block.
struct Rig {
    id: i32,
    gate: DeviceBuffer<u8>,
    up: DeviceBuffer<u8>,
    down: DeviceBuffer<u8>,
    gbpe: u32,
    dbpe: u32,
    remap: DeviceBuffer<i32>,
    xq: DeviceBuffer<u8>,
    ew: DeviceBuffer<f32>,
}

/// One block's outputs: `mid`, `midq`, `ffn_out`.
struct Out {
    mid: DeviceBuffer<f32>,
    midq: DeviceBuffer<u8>,
    out: DeviceBuffer<f32>,
}

impl Out {
    fn new(id: i32) -> eyre::Result<Self> {
        let (b, tk) = (MTP_BLOCK, MTP_TOPK as usize);
        Ok(Self {
            mid: upload(id, &vec![SENTINEL; b * tk * N_FF_EXP as usize])?,
            midq: upload(id, &vec![0xA5u8; b * tk * MIDQ_BYTES_PER_SLOT])?,
            out: upload(id, &vec![SENTINEL; b * N_EMBD as usize])?,
        })
    }
}

/// `MtpState::moe`'s per-row chain, verbatim.
fn per_row(e: &DeviceEngine, s: &Stream, r: &Rig, sel: &DeviceBuffer<i32>, o: &mut Out) -> eyre::Result<()> {
    let tk = MTP_TOPK as usize;
    let (ffe, ne) = (N_FF_EXP as usize, N_EMBD as usize);
    for j in 0..MTP_BLOCK {
        let xq_j = r.xq.slice_view(j * XQ_BYTES_PER_TOKEN, XQ_BYTES_PER_TOKEN);
        let ew_j = r.ew.slice_view(j * tk, tk);
        let sel_j = sel.slice_view(j * tk, tk);
        let mut mid_j = o.mid.slice_view_mut(j * tk * ffe, tk * ffe);
        dispatch::moe_gate_up_batch_hetsplit(
            e, GgufType::MXFP4, s, &mut mid_j, &r.gate, &r.up, &xq_j, &ew_j, &sel_j, &r.remap, 0, MTP_TOPK,
            r.gbpe, r.gbpe, MTP_TOPK, SWIGLU_CLAMP_EXP, N_FF_EXP, BLOCKS_Q8K_GATE_IN,
        )?;
        let mut midq_j = o.midq.slice_view_mut(j * tk * MIDQ_BYTES_PER_SLOT, tk * MIDQ_BYTES_PER_SLOT);
        e.q8k.launch(s, &mut midq_j, &mid_j, BLOCKS_Q8K_DOWN_IN * MTP_TOPK)?;
        let mut out_j = o.out.slice_view_mut(j * ne, ne);
        dispatch::moe_down_batched_hetsplit(
            e, GgufType::MXFP4, s, &mut out_j, &r.down, &midq_j, &sel_j, &r.remap, 0, MTP_TOPK, r.dbpe,
            MIDQ_BYTES_PER_SLOT as u32, MTP_TOPK, N_EMBD, BLOCKS_Q8K_DOWN_IN,
        )?;
    }
    Ok(())
}

/// The `V41_MTP_MOE_GROUPED` chain, as `MtpState::moe` issues it.
fn grouped(e: &DeviceEngine, s: &Stream, r: &Rig, sel: &DeviceBuffer<i32>, o: &mut Out) -> eyre::Result<()> {
    let b = MTP_BLOCK as u32;
    e.mxfp4pair.launch_fused_swiglu_grouped(
        s, &mut o.mid, &r.gate, &r.up, &r.xq, &r.ew, sel, &r.remap, r.gbpe, r.gbpe, SWIGLU_CLAMP_EXP,
        N_FF_EXP, BLOCKS_Q8K_GATE_IN, b, MTP_TOPK,
    )?;
    e.q8k.launch(s, &mut o.midq, &o.mid, BLOCKS_Q8K_DOWN_IN * b * MTP_TOPK)?;
    e.mxfp4.launch_grouped(
        s, &mut o.out, &r.down, &o.midq, sel, &r.remap, r.dbpe, MIDQ_BYTES_PER_SLOT as u32, N_EMBD,
        BLOCKS_Q8K_DOWN_IN, b, MTP_TOPK,
    )
}

/// `MTP_BLOCK` tokens x `MTP_TOPK` DISTINCT picks from `pool`, token-major.
fn picks_from(rng: &mut Lcg, pool: &[i32]) -> Vec<i32> {
    let mut v = Vec::new();
    for _ in 0..MTP_BLOCK {
        let mut t: Vec<i32> = Vec::new();
        while t.len() < MTP_TOPK as usize {
            let x = pool[rng.below(pool.len() as u32) as usize];
            if !t.contains(&x) {
                t.push(x);
            }
        }
        v.extend(t);
    }
    v
}

fn distinct(sel: &[i32]) -> usize {
    let mut d: Vec<i32> = sel.to_vec();
    d.sort_unstable();
    d.dedup();
    d.len()
}

const N_PHYS: usize = 32;

#[test]
#[ignore]
fn grouped_moe_bit_identical_to_per_row() -> eyre::Result<()> {
    install_panic_handler()?;
    let Some(dev) = pick_igpu()? else {
        eprintln!("SKIP: no gfx1151 device");
        return Ok(());
    };
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    let id = dev.id;
    let e = DeviceEngine::for_arch(dev, &arch)?;
    let s = Stream::new(id)?;
    let mut rng = Lcg(0x6D74_7067);

    let (ffe, ne) = (N_FF_EXP as usize, N_EMBD as usize);
    let (nb_in, nb_dn) = (BLOCKS_Q8K_GATE_IN as usize, BLOCKS_Q8K_DOWN_IN as usize);
    let gbpe = ffe * nb_in * SUPER_MXFP4_BYTES;
    let dbpe = ne * nb_dn * SUPER_MXFP4_BYTES;
    // The drafter's resident identity remap: -(e)-1 for resident experts, 0 = not ours.
    let mut remap_h = vec![0i32; SENTINEL_EXPERT as usize + 1];
    for (x, r) in remap_h.iter_mut().enumerate().take(N_PHYS) {
        *r = -(x as i32) - 1;
    }
    let rig = Rig {
        id,
        gate: upload(id, &mxfp4_weights(&mut rng, N_PHYS, ffe, nb_in, 118, 8))?,
        up: upload(id, &mxfp4_weights(&mut rng, N_PHYS, ffe, nb_in, 118, 8))?,
        down: upload(id, &mxfp4_weights(&mut rng, N_PHYS, ne, nb_dn, 118, 8))?,
        gbpe: gbpe as u32,
        dbpe: dbpe as u32,
        remap: upload(id, &remap_h)?,
        xq: upload(id, &q8k_blocks(&mut rng, MTP_BLOCK * nb_in))?,
        ew: upload(id, &(0..MTP_BLOCK * MTP_TOPK as usize).map(|_| 0.05 + (rng.next() % 1000) as f32 * 1e-3).collect::<Vec<_>>())?,
    };

    let all: Vec<i32> = (0..N_PHYS as i32).collect();
    let mut cases: Vec<(String, Vec<i32>)> = Vec::new();
    cases.push(("same 3, same slots".into(), [4, 9, 17].repeat(MTP_BLOCK)));
    cases.push(("same 3, rotated slots".into(), [4, 9, 17, 9, 17, 4, 17, 4, 9, 4, 17, 9, 9, 4, 17].to_vec()));
    cases.push(("15 distinct".into(), (0..15).map(|x| (x * 2 + 1) % N_PHYS as i32).collect()));
    for t in 0..8 {
        let pool: Vec<i32> = (0..6).map(|_| all[rng.below(N_PHYS as u32) as usize]).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        if pool.len() >= MTP_TOPK as usize {
            cases.push((format!("pool6 #{t}"), picks_from(&mut rng, &pool)));
        }
        let mut pool9: Vec<i32> = Vec::new();
        while pool9.len() < 9 {
            let x = rng.below(N_PHYS as u32) as i32;
            if !pool9.contains(&x) {
                pool9.push(x);
            }
        }
        cases.push((format!("pool9 #{t}"), picks_from(&mut rng, &pool9)));
        // Picks past the resident set: ids N_PHYS.. have remap 0 (not ours).
        let mixed: Vec<i32> = (0..12).map(|i| if i < 8 { rng.below(N_PHYS as u32) as i32 } else { N_PHYS as i32 + i }).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        cases.push((format!("non-resident #{t}"), picks_from(&mut rng, &mixed)));
    }
    cases.push(("all non-resident".into(), picks_from(&mut rng, &[100, 101, 102, 103])));

    for (name, sel_h) in &cases {
        let sel = upload(id, sel_h)?;
        let mut a = Out::new(rig.id)?;
        let mut g = Out::new(rig.id)?;
        per_row(&e, &s, &rig, &sel, &mut a)?;
        grouped(&e, &s, &rig, &sel, &mut g)?;
        s.synchronize()?;
        let (ma, mg) = (bits(&a.mid)?, bits(&g.mid)?);
        let d_mid = ma.iter().zip(&mg).filter(|(x, y)| x != y).count();
        let unwritten = mg.iter().filter(|&&x| x == SENTINEL.to_bits()).count();
        let (qa, qg) = (bytes(&a.midq)?, bytes(&g.midq)?);
        let d_midq = qa.iter().zip(&qg).filter(|(x, y)| x != y).count();
        let (oa, og) = (bits(&a.out)?, bits(&g.out)?);
        let d_out = oa.iter().zip(&og).filter(|(x, y)| x != y).count();
        let nz = oa.iter().filter(|&&x| f32::from_bits(x) != 0.0).count();
        eprintln!(
            "{name:24} distinct={:2}  mid diff {d_mid} (unwritten {unwritten})  midq diff {d_midq}  out diff {d_out} (nonzero {nz}/{})",
            distinct(sel_h),
            oa.len()
        );
        assert_eq!((d_mid, unwritten, d_midq, d_out), (0, 0, 0, 0), "{name}: grouped != per-row, sel={sel_h:?}");
    }

    // Shape guards are refusals, not silent no-ops.
    let mut o = Out::new(id)?;
    let sel = upload(id, &[0i32; 18])?;
    assert!(e.mxfp4.launch_grouped(&s, &mut o.out, &rig.down, &o.midq, &sel, &rig.remap, rig.dbpe,
        MIDQ_BYTES_PER_SLOT as u32, N_EMBD, BLOCKS_Q8K_DOWN_IN, 6, MTP_TOPK).is_err(), "18 picks > 16 must be refused");
    assert!(e.mxfp4.launch_grouped(&s, &mut o.out, &rig.down, &o.midq, &sel, &rig.remap, rig.dbpe,
        MIDQ_BYTES_PER_SLOT as u32, N_EMBD, 17, MTP_BLOCK as u32, MTP_TOPK).is_err(), "n_blocks_in 17 must be refused");

    // Timing (informational): rotating 9-of-32 pools, ~7.8 distinct of 15.
    const K: usize = 24;
    let mut sels = Vec::new();
    let mut dsum = 0usize;
    for _ in 0..K {
        let mut pool9: Vec<i32> = Vec::new();
        while pool9.len() < 9 {
            let x = rng.below(N_PHYS as u32) as i32;
            if !pool9.contains(&x) {
                pool9.push(x);
            }
        }
        let p = picks_from(&mut rng, &pool9);
        dsum += distinct(&p);
        sels.push(upload(id, &p)?);
    }
    let mut a = Out::new(id)?;
    let mut g = Out::new(id)?;
    for round in 0..3 {
        s.synchronize()?;
        let t0 = std::time::Instant::now();
        for sel in &sels {
            per_row(&e, &s, &rig, sel, &mut a)?;
        }
        s.synchronize()?;
        let t_row = t0.elapsed().as_secs_f64() * 1e6 / K as f64;
        let t1 = std::time::Instant::now();
        for sel in &sels {
            grouped(&e, &s, &rig, sel, &mut g)?;
        }
        s.synchronize()?;
        let t_grp = t1.elapsed().as_secs_f64() * 1e6 / K as f64;
        eprintln!(
            "timing round {round}: per-row {t_row:.0} us/layer, grouped {t_grp:.0} us/layer ({:.2}x), mean distinct {:.1}/15",
            t_row / t_grp,
            dsum as f64 / K as f64
        );
    }
    Ok(())
}
