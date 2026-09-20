//! Multi-stream M1a, G5a at the kernel level: the batched KV kernels with a
//! PER-ROW BASE (each row its own sequence somewhere in one arena buffer) must
//! produce exactly what the single-sequence form produces when each row is run
//! alone on a slice of the same buffer. Synthetic data, dGPU only (WMMA
//! kernels), a few MB — runs beside a live server.
//!
//!   cargo test -p v4flash-kernels --features v41 --release --test multistream_row_bases -- --ignored --nocapture
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer};
use v4flash_kernels::het::engine::DeviceEngine;
use v4flash_kernels::het::remote_experts::f32_to_f16_bits;
use v4flash_kernels::index_kv_e2m1::{E2M1_KEY_DIM, E2M1_KEY_ROW_BYTES};

const N_HEAD: u32 = 64;
const HEAD_DIM: u32 = 512;
const B: usize = 4;

struct Rng(u64);
impl Rng {
    fn f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
    }
    fn u(&mut self, n: usize) -> usize { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; (self.0 % n as u64) as usize }
}

fn pick_dgpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") { return Ok(d); }
    }
    Err(eyre!("no gfx1201 device"))
}

fn dev_f32(id: i32, v: &[f32]) -> DeviceBuffer<f32> { let mut b = DeviceBuffer::<f32>::new(id, v.len().max(1)).unwrap(); if !v.is_empty() { b.copy_from_host(v).unwrap(); } b }
fn dev_u16(id: i32, v: &[u16]) -> DeviceBuffer<u16> { let mut b = DeviceBuffer::<u16>::new(id, v.len().max(1)).unwrap(); if !v.is_empty() { b.copy_from_host(v).unwrap(); } b }
fn dev_i32(id: i32, v: &[i32]) -> DeviceBuffer<i32> { let mut b = DeviceBuffer::<i32>::new(id, v.len().max(1)).unwrap(); if !v.is_empty() { b.copy_from_host(v).unwrap(); } b }
fn dev_u32(id: i32, v: &[u32]) -> DeviceBuffer<u32> { let mut b = DeviceBuffer::<u32>::new(id, v.len().max(1)).unwrap(); if !v.is_empty() { b.copy_from_host(v).unwrap(); } b }
fn host<T: Copy + Default>(b: &DeviceBuffer<T>) -> Vec<T> { let mut v = vec![T::default(); b.len()]; b.copy_to_host(&mut v).unwrap(); v }
fn bits_eq(a: &[f32], b: &[f32]) -> usize { a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count() }

#[test]
#[ignore = "needs the dGPU (gfx1201)"]
fn per_row_bases_match_single_sequence() -> eyre::Result<()> {
    install_panic_handler().ok();
    let dev = pick_dgpu()?;
    let arch = dev.properties()?.gcn_arch_name;
    let e = DeviceEngine::for_arch(dev, &arch)?;
    let s = &e.compute;
    let id = dev.id;
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let hd = HEAD_DIM as usize;
    let nh = N_HEAD as usize;

    // Four rows = four sequences: different raw-window sizes/offsets and comp counts.
    let n_raw = [17usize, 128, 64, 1];
    let raw_off = [3usize, 0, 900, 1140];      // within a 1152-row window region
    let n_comp = [40usize, 512, 300, 7];
    let raw_rows_per_seq = 1152usize;          // KV_CACHE_ROWS
    let comp_base = [100usize, 700, 1500, 2100]; // arena rows (scattered)
    let top_k = 512usize;

    // ---- arena buffers: one raw buffer with per-row regions, one comp buffer with scattered bases
    let mut raw_host = vec![0u16; B * raw_rows_per_seq * hd];
    for v in raw_host.iter_mut() { *v = f32_to_f16_bits(rng.f32()); }
    let mut comp_host = vec![0u16; 2700 * hd];
    for v in comp_host.iter_mut() { *v = f32_to_f16_bits(rng.f32()); }
    let raw_kv = dev_u16(id, &raw_host);
    let comp_kv = dev_u16(id, &comp_host);
    let q_host: Vec<f32> = (0..B * nh * hd).map(|_| rng.f32()).collect();
    let q = dev_f32(id, &q_host);
    let sinks_host: Vec<f32> = (0..nh).map(|_| rng.f32()).collect();
    let sinks = dev_f32(id, &sinks_host);
    let scores_stride = 1024u32;
    let n_total_max = (0..B).map(|b| n_raw[b] + n_comp[b]).max().unwrap() as u32;

    // ---- (1) attention score + wsum: new path (batched, per-row comp bases) vs per-row single-sequence runs
    let n_raw_per = dev_i32(id, &n_raw.iter().map(|&x| x as i32).collect::<Vec<_>>());
    let n_raw_offset_per = dev_i32(id, &(0..B).map(|b| (b * raw_rows_per_seq + raw_off[b]) as i32).collect::<Vec<_>>());
    let n_comp_per = dev_i32(id, &n_comp.iter().map(|&x| x as i32).collect::<Vec<_>>());
    let comp_base_per = dev_i32(id, &comp_base.iter().map(|&x| x as i32).collect::<Vec<_>>());
    let mut scores = DeviceBuffer::<f32>::new(id, B * nh * scores_stride as usize)?;
    let mut out_new = DeviceBuffer::<f32>::new(id, B * nh * hd)?;
    scores.fill_zero()?; out_new.fill_zero()?;
    e.attn_mixed.launch_score_batched_htiled_wmma_f16s_rows(s, &mut scores, &q, &raw_kv, Some(&comp_kv), &n_raw_per, &n_raw_offset_per, &n_comp_per, None, N_HEAD, HEAD_DIM, n_total_max, B as u32, 0, scores_stride, Some(&comp_base_per))?;
    e.attn_mixed.launch_softmax_wsum_batched_htiled_wmma_ldsv_f16s_rows(s, &mut out_new, &mut scores, &sinks, &raw_kv, Some(&comp_kv), &n_raw_per, &n_raw_offset_per, &n_comp_per, N_HEAD, HEAD_DIM, B as u32, 0, scores_stride, Some(&comp_base_per))?;
    s.synchronize()?;
    let out_new_h = host(&out_new);
    let mut mism = 0usize;
    for b in 0..B {
        // legacy single-sequence run of row b alone: raw base = this row's region, comp = a slice starting at its base
        let q_b = q.slice_view(b * nh * hd, nh * hd);
        let raw_b = raw_kv.slice_view(b * raw_rows_per_seq * hd, raw_rows_per_seq * hd);
        let comp_b = comp_kv.slice_view(comp_base[b] * hd, n_comp[b] * hd);
        let nr = dev_i32(id, &[n_raw[b] as i32]); let ro = dev_i32(id, &[raw_off[b] as i32]); let nc = dev_i32(id, &[n_comp[b] as i32]);
        let mut sc1 = DeviceBuffer::<f32>::new(id, nh * scores_stride as usize)?; sc1.fill_zero()?;
        let mut out1 = DeviceBuffer::<f32>::new(id, nh * hd)?; out1.fill_zero()?;
        e.attn_mixed.launch_score_batched_htiled_wmma_f16s(s, &mut sc1, &q_b, &raw_b, Some(&comp_b), &nr, &ro, &nc, None, N_HEAD, HEAD_DIM, (n_raw[b] + n_comp[b]) as u32, 1, 0, scores_stride)?;
        e.attn_mixed.launch_softmax_wsum_batched_htiled_wmma_ldsv_f16s(s, &mut out1, &mut sc1, &sinks, &raw_b, Some(&comp_b), &nr, &ro, &nc, N_HEAD, HEAD_DIM, 1, 0, scores_stride)?;
        s.synchronize()?;
        let o1 = host(&out1);
        let m = bits_eq(&o1, &out_new_h[b * nh * hd..(b + 1) * nh * hd]);
        eprintln!("attention row {b}: n_raw {} n_comp {} -> {m} f32 mismatches of {}", n_raw[b], n_comp[b], nh * hd);
        mism += m;
    }
    assert_eq!(mism, 0, "attention per-row bases are not bit-identical");

    // ---- (2) indexer gather: relative indices + per-row base vs absolute indices on the shared buffer
    let mut sel_rel = vec![0i32; B * top_k];
    let mut sel_abs = vec![0i32; B * top_k];
    for b in 0..B { for i in 0..top_k { if i < n_comp[b] { let r = rng.u(n_comp[b]); sel_rel[b * top_k + i] = r as i32; sel_abs[b * top_k + i] = (comp_base[b] + r) as i32; } else { sel_rel[b * top_k + i] = -1; sel_abs[b * top_k + i] = -1; } } }
    let sel_rel_d = dev_i32(id, &sel_rel); let sel_abs_d = dev_i32(id, &sel_abs);
    let mut g_new = DeviceBuffer::<u16>::new(id, B * top_k * hd)?; g_new.fill_zero()?;
    let mut g_ref = DeviceBuffer::<u16>::new(id, B * top_k * hd)?; g_ref.fill_zero()?;
    e.indexer_gather.launch_batched_rows(s, &mut g_new, &comp_kv, &sel_rel_d, top_k as u32, HEAD_DIM, B as u32, Some(&comp_base_per))?;
    e.indexer_gather.launch_batched(s, &mut g_ref, &comp_kv, &sel_abs_d, top_k as u32, HEAD_DIM, B as u32)?;
    s.synchronize()?;
    let (a, r) = (host(&g_new), host(&g_ref));
    let m = a.iter().zip(&r).filter(|(x, y)| x != y).count();
    eprintln!("indexer gather: {m} mismatches of {}", a.len());
    assert_eq!(m, 0);

    // ---- (3) index-key append with per-row destination rows + indexer score with per-row key bases
    let n_keys = [40usize, 512, 300, 7];
    let key_base = [50usize, 400, 1000, 1400];
    let key_rows_f32: Vec<f32> = (0..B * 512 * E2M1_KEY_DIM).map(|_| rng.f32()).collect();
    let rows_d = dev_f32(id, &key_rows_f32);
    let mut packed_new = DeviceBuffer::<u8>::new(id, 1500 * E2M1_KEY_ROW_BYTES)?; packed_new.fill_zero()?;
    let mut packed_ref = DeviceBuffer::<u8>::new(id, 1500 * E2M1_KEY_ROW_BYTES)?; packed_ref.fill_zero()?;
    for b in 0..B {
        let rows_b = rows_d.slice_view(b * 512 * E2M1_KEY_DIM, n_keys[b] * E2M1_KEY_DIM);
        let dst: Vec<i32> = (0..n_keys[b]).map(|k| (key_base[b] + k) as i32).collect();
        let dst_d = dev_i32(id, &dst);
        e.index_kv_e2m1.launch_append_batched_rows(s, &mut packed_new, &rows_b, 0, n_keys[b] as u32, Some(&dst_d))?;
        e.index_kv_e2m1.launch_append_batched(s, &mut packed_ref, &rows_b, key_base[b] as u32, n_keys[b] as u32)?;
    }
    s.synchronize()?;
    let (a, r) = (host(&packed_new), host(&packed_ref));
    let m = a.iter().zip(&r).filter(|(x, y)| x != y).count();
    eprintln!("index-key append: {m} byte mismatches of {}", a.len());
    assert_eq!(m, 0);
    let isw = e.indexer_score_wmma.as_ref().ok_or_else(|| eyre!("no indexer_score_wmma on this arch"))?;
    let iq_host: Vec<f32> = (0..B * 32 * 128).map(|_| rng.f32()).collect();
    let iq = dev_f32(id, &iq_host);
    let hw_host: Vec<f32> = (0..B * 32).map(|_| rng.f32().abs()).collect();
    let hw = dev_f32(id, &hw_host);
    let n_idx_stride = 1024u32;
    let n_idx_per = dev_u32(id, &n_keys.iter().map(|&x| x as u32).collect::<Vec<_>>());
    let keys_base_per = dev_u32(id, &key_base.iter().map(|&x| x as u32).collect::<Vec<_>>());
    let mut isc_new = DeviceBuffer::<f32>::new(id, B * n_idx_stride as usize)?; isc_new.fill_zero()?;
    isw.launch_batched_mw_e2m1_rows(s, &mut isc_new, &iq, &hw, &packed_new, &n_idx_per, 512, n_idx_stride, B as u32, Some(&keys_base_per))?;
    s.synchronize()?;
    let a = host(&isc_new);
    let mut m = 0usize;
    for b in 0..B {
        let keys_b = packed_new.slice_view(key_base[b] * E2M1_KEY_ROW_BYTES, n_keys[b] * E2M1_KEY_ROW_BYTES);
        let iq_b = iq.slice_view(b * 32 * 128, 32 * 128); let hw_b = hw.slice_view(b * 32, 32);
        let np = dev_u32(id, &[n_keys[b] as u32]);
        let mut sc1 = DeviceBuffer::<f32>::new(id, n_idx_stride as usize)?; sc1.fill_zero()?;
        isw.launch_batched_mw_e2m1(s, &mut sc1, &iq_b, &hw_b, &keys_b, &np, n_keys[b] as u32, n_idx_stride, 1)?;
        s.synchronize()?;
        let r = host(&sc1);
        m += bits_eq(&r[..n_keys[b]], &a[b * n_idx_stride as usize..b * n_idx_stride as usize + n_keys[b]]);
    }
    eprintln!("indexer score (per-row key bases): {m} f32 mismatches");
    assert_eq!(m, 0);

    // ---- (4) raw kv append with per-row slots; comp append with per-row rows
    let kv_new_host: Vec<f32> = (0..B * hd).map(|_| rng.f32()).collect();
    let kv_new = dev_f32(id, &kv_new_host);
    let slots: Vec<i32> = (0..B).map(|b| (b * raw_rows_per_seq + raw_off[b] + n_raw[b]) as i32).collect();
    let slots_d = dev_i32(id, &slots);
    let mut cache_new = DeviceBuffer::<u16>::new(id, B * raw_rows_per_seq * hd)?; cache_new.fill_zero()?;
    let mut cache_ref = DeviceBuffer::<u16>::new(id, B * raw_rows_per_seq * hd)?; cache_ref.fill_zero()?;
    e.kv_append.launch_batched_rows(s, &mut cache_new, &kv_new, 0, HEAD_DIM, B as u32, Some(&slots_d))?;
    for b in 0..B {
        let mut dst = cache_ref.slice_view_mut(b * raw_rows_per_seq * hd, raw_rows_per_seq * hd);
        let src = kv_new.slice_view(b * hd, hd);
        e.kv_append.launch_batched(s, &mut dst, &src, (raw_off[b] + n_raw[b]) as u32, HEAD_DIM, 1)?;
    }
    s.synchronize()?;
    let (a, r) = (host(&cache_new), host(&cache_ref));
    let m = a.iter().zip(&r).filter(|(x, y)| x != y).count();
    eprintln!("raw kv append (per-row slots): {m} mismatches");
    assert_eq!(m, 0);
    let comp_rows_host: Vec<f32> = (0..B * hd).map(|_| rng.f32()).collect();
    let comp_rows = dev_f32(id, &comp_rows_host);
    let dst_rows: Vec<i32> = (0..B).map(|b| (comp_base[b] + n_comp[b]) as i32).collect();
    let dst_rows_d = dev_i32(id, &dst_rows);
    let mut ck_new = DeviceBuffer::<u16>::new(id, 2700 * hd)?; ck_new.fill_zero()?;
    let mut ck_ref = DeviceBuffer::<u16>::new(id, 2700 * hd)?; ck_ref.fill_zero()?;
    e.comp_kv_append.launch_batched_rows(s, &mut ck_new, &comp_rows, 0, HEAD_DIM, B as u32, Some(&dst_rows_d))?;
    for b in 0..B {
        let src = comp_rows.slice_view(b * hd, hd);
        e.comp_kv_append.launch_batched(s, &mut ck_ref, &src, (comp_base[b] + n_comp[b]) as u32, HEAD_DIM, 1)?;
    }
    s.synchronize()?;
    let (a, r) = (host(&ck_new), host(&ck_ref));
    let m = a.iter().zip(&r).filter(|(x, y)| x != y).count();
    eprintln!("comp kv append (per-row rows): {m} mismatches");
    assert_eq!(m, 0);

    // ---- (5) compressor state write with per-row state bases + pool with per-boundary state index
    let ratio = 2u32; let width = HEAD_DIM; let state_per = (ratio * width) as usize;
    let n_states = 6usize; // arena of 6 state blocks; rows use blocks 4,1,5,0
    let state_idx = [4usize, 1, 5, 0];
    let kv_cur_h: Vec<f32> = (0..B * width as usize).map(|_| rng.f32()).collect();
    let sc_cur_h: Vec<f32> = (0..B * width as usize).map(|_| rng.f32()).collect();
    let kv_cur = dev_f32(id, &kv_cur_h); let sc_cur = dev_f32(id, &sc_cur_h);
    let ape_h: Vec<u16> = (0..ratio as usize * width as usize).map(|_| f32_to_f16_bits(rng.f32())).collect();
    let mut ape = DeviceBuffer::<u8>::new(id, ape_h.len() * 2)?;
    { let bytes: Vec<u8> = ape_h.iter().flat_map(|v| v.to_le_bytes()).collect(); ape.copy_from_host(&bytes)?; }
    let row_per_b = dev_i32(id, &[1, 0, 1, 0]); let pos_mod_per_b = dev_i32(id, &[1, 0, 1, 0]);
    let state_base_per = dev_i32(id, &state_idx.iter().map(|&i| (i * state_per) as i32).collect::<Vec<_>>());
    let mut skv_new = DeviceBuffer::<f32>::new(id, n_states * state_per)?; skv_new.fill_zero()?;
    let mut ssc_new = DeviceBuffer::<f32>::new(id, n_states * state_per)?; ssc_new.fill_zero()?;
    let mut skv_ref = DeviceBuffer::<f32>::new(id, n_states * state_per)?; skv_ref.fill_zero()?;
    let mut ssc_ref = DeviceBuffer::<f32>::new(id, n_states * state_per)?; ssc_ref.fill_zero()?;
    e.compressor_state_write.launch_batched_rows(s, &mut skv_new, &mut ssc_new, &kv_cur, &sc_cur, &ape, &row_per_b, &pos_mod_per_b, width, B as u32, Some(&state_base_per))?;
    for b in 0..B {
        let mut skv_b = skv_ref.slice_view_mut(state_idx[b] * state_per, state_per);
        let mut ssc_b = ssc_ref.slice_view_mut(state_idx[b] * state_per, state_per);
        let kv_b = kv_cur.slice_view(b * width as usize, width as usize); let sc_b = sc_cur.slice_view(b * width as usize, width as usize);
        let rp = dev_i32(id, &[[1, 0, 1, 0][b]]); let pm = dev_i32(id, &[[1, 0, 1, 0][b]]);
        e.compressor_state_write.launch_batched(s, &mut skv_b, &mut ssc_b, &kv_b, &sc_b, &ape, &rp, &pm, width, 1)?;
    }
    s.synchronize()?;
    let m = bits_eq(&host(&skv_new), &host(&skv_ref)) + bits_eq(&host(&ssc_new), &host(&ssc_ref));
    eprintln!("compressor state write (per-row bases): {m} mismatches");
    assert_eq!(m, 0);
    // fill the whole state arena with data, then pool: new path picks blocks by index, reference pools each block via a slice
    let fill_kv: Vec<f32> = (0..n_states * state_per).map(|_| rng.f32()).collect();
    let fill_sc: Vec<f32> = (0..n_states * state_per).map(|_| rng.f32()).collect();
    let skv = dev_f32(id, &fill_kv); let ssc = dev_f32(id, &fill_sc);
    let state_idx_d = dev_i32(id, &state_idx.iter().map(|&i| i as i32).collect::<Vec<_>>());
    let mut pool_new = DeviceBuffer::<f32>::new(id, B * hd)?; pool_new.fill_zero()?;
    e.compressor_pool.launch_batched_rows(s, &mut pool_new, &skv, &ssc, HEAD_DIM, ratio, B as u32, Some(&state_idx_d))?;
    let mut m = 0usize;
    for b in 0..B {
        let skv_b = skv.slice_view(state_idx[b] * state_per, state_per); let ssc_b = ssc.slice_view(state_idx[b] * state_per, state_per);
        let mut o1 = DeviceBuffer::<f32>::new(id, hd)?; o1.fill_zero()?;
        e.compressor_pool.launch_batched(s, &mut o1, &skv_b, &ssc_b, HEAD_DIM, ratio, 1)?;
        s.synchronize()?;
        let a = host(&pool_new);
        m += bits_eq(&host(&o1), &a[b * hd..(b + 1) * hd]);
    }
    eprintln!("compressor pool (per-boundary state index): {m} mismatches");
    assert_eq!(m, 0);
    eprintln!("ALL per-row-base kernels bit-identical to their single-sequence runs");
    Ok(())
}
