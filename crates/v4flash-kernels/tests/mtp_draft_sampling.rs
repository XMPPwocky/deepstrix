//! DSpark sampled drafts (docs/v41/DSPARK_ARENA_PLAN.md M6): the exit's device
//! half -- the top-M of one position's biased logits (`indexer_topk_bitonic` at
//! k = MTP_DRAFT_TOP_M over the full 129,280-entry vocabulary; production runs
//! it at k = 512 over comp rows) and the `gather_f32` of their logits -- against
//! a host sort. The host half (`draft_dist`: q and the draw) has unit tests in
//! mtp.rs. No model weights; runs on the dGPU, where the exit lives.
//!
//! ```text
//! HIP_VISIBLE_DEVICES=0,1 CARGO_TARGET_DIR=target-v41 nix develop -c cargo test -p v4flash-kernels \
//!   --release --features v41 --test mtp_draft_sampling -- --ignored --nocapture
//! ```
#![cfg(feature = "v41")]

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::config::N_VOCAB;
use v4flash_kernels::het::mtp::{draft_dist, MTP_DRAFT_TOP_M};
use v4flash_kernels::{IndexerTopkBitonic, VecAddInplace};

fn pick(prefix: &str) -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with(prefix) {
            return Ok(d);
        }
    }
    Err(eyre!("no {prefix} device"))
}

#[test]
#[ignore]
fn top_m_and_gather_match_a_host_sort() -> eyre::Result<()> {
    install_panic_handler()?;
    let dev = pick("gfx1201")?;
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    let topk = IndexerTopkBitonic::for_arch(&arch)?;
    let gather = VecAddInplace::for_arch(&arch)?;
    let s = Stream::new(dev.id)?;
    let nv = N_VOCAB as usize;
    let m = MTP_DRAFT_TOP_M;
    let mut logits = DeviceBuffer::<f32>::new(dev.id, nv)?;
    let mut sel = DeviceBuffer::<i32>::new(dev.id, m)?;
    let mut bits = DeviceBuffer::<u32>::new(dev.id, nv.div_ceil(32))?;
    let scratch_n: usize = v4flash_kernels::indexer::topk_merge_levels(N_VOCAB, m as u32).iter().map(|&n| n as usize).sum();
    let mut scratch = DeviceBuffer::<u32>::new(dev.id, scratch_n.max(1))?;
    let mut cand = DeviceBuffer::<f32>::new(dev.id, m)?;
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut rnd = || {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (x >> 40) as f32 / (1u64 << 24) as f32
    };
    // Cases: uniform noise, heavy ties (coarse values), and a realistic peaked
    // row (a few large logits over a broad low tail).
    for case in 0..12 {
        let host: Vec<f32> = (0..nv)
            .map(|i| match case % 3 {
                0 => rnd() * 20.0 - 10.0,
                1 => (rnd() * 8.0).floor(),
                _ => if i % 9973 == 0 { 15.0 + rnd() * 5.0 } else { rnd() * 4.0 - 8.0 },
            })
            .collect();
        logits.copy_from_host(&host)?;
        topk.launch(&s, &mut sel, &mut bits, &mut scratch, &logits, N_VOCAB, m as u32)?;
        gather.launch_gather(&s, &mut cand, &logits, &sel, m as u32)?;
        s.synchronize()?;
        let mut got_id = vec![0i32; m];
        let mut got_lg = vec![0f32; m];
        sel.copy_to_host(&mut got_id)?;
        cand.copy_to_host(&mut got_lg)?;
        // Host reference: descending by value, ties to the lower index.
        let mut idx: Vec<usize> = (0..nv).collect();
        idx.sort_by(|&a, &b| host[b].partial_cmp(&host[a]).unwrap().then(a.cmp(&b)));
        let want: Vec<i32> = idx[..m].iter().map(|&i| i as i32).collect();
        assert_eq!(got_id, want, "case {case}: top-{m} ids differ from the host sort");
        for k in 0..m {
            assert_eq!(got_lg[k].to_bits(), host[got_id[k] as usize].to_bits(), "case {case}: gathered logit {k}");
        }
        let (q, d) = draft_dist(&got_id, &got_lg, 1.0, 0.95, 0.0, 0.5)?;
        assert!(q.iter().any(|&(t, _)| t == d) && (q.iter().map(|e| e.1).sum::<f64>() - 1.0).abs() < 1e-9);
    }
    // Timing: the per-position device cost the sampled exit adds.
    let t = std::time::Instant::now();
    let reps = 200;
    for _ in 0..reps {
        topk.launch(&s, &mut sel, &mut bits, &mut scratch, &logits, N_VOCAB, m as u32)?;
        gather.launch_gather(&s, &mut cand, &logits, &sel, m as u32)?;
    }
    s.synchronize()?;
    println!("top-{m} + gather over {nv} logits: {:.1} us per position", t.elapsed().as_secs_f64() * 1e6 / reps as f64);
    Ok(())
}
