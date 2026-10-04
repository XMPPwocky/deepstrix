//! DSpark drafter PARITY against DeepSeek's unmodified reference drafter.
//!
//! Every earlier acceptance number compared the engine's drafts to the engine's
//! own decode, on text the engine generated itself, so "engine vs reference"
//! was never measured like for like (acceptance varies ~2x with content). This
//! test feeds OUR drafter exactly what the reference drafter saw in
//! `scripts/dspark_accept.py` (the oracle's own main-model residuals, the same
//! transcript, the same seeding, the same positions) and compares draft by
//! draft with the reference's recorded drafts and greedy targets.
//!
//! Protocol (mirrors the oracle script, `seed` = 256 by default):
//!   * seed: the reference runs `forward_spec(tok[seed], mh[0..seed], 0)`, which
//!     writes the window rows of positions 0..seed-1 (only the last 128 survive,
//!     at slot `pos % 128`). Here: `ring_write_only` for positions
//!     `seed-128 .. seed`, in order (ring slot == pos % 128 while in lockstep).
//!   * step i = seed ..: the drafter takes token `tok[i+1]` and `mh[i]` at
//!     `start_pos = i`, writes the ring row for position i, and drafts
//!     positions i+2 ..= i+6. Greedy acceptance of draft k is
//!     `draft[k] == main_argmax[i+1+k]` (the reference's `greedy` field).
//!
//! Inputs (made by `scripts/v41_oracle/parity_convert.py <gen2 dump>` from the oracle's gen2 dump, no torch):
//!   $PARITY_DIR/{mh.bin f32[T,3,5120], tokens.bin i32[T], argmax.bin i32[T],
//!                ref_<name>.bin i32[steps, 11], ref_<name>_conf.bin f32[steps,5]}
//!
//! Env: PARITY_DIR (default ~/.cache/deepstrix/v41/agentic/gen2/parity),
//!      PARITY_REF = base (default) | noseed | nomarkov | incseed,
//!      PARITY_MH_BF16 = 1 (default; the oracle rounds mh to bf16) | 0,
//!      PARITY_MH = alternative main-hidden file (e.g. the engine's own),
//!      PARITY_SHOW = N steps printed in detail (default 6),
//!      PARITY_ASSERT = 1 applies THE pass rule: E(ours) >= E(ref) - 0.2 (~4.2
//!      for `base`) AND the paired 90% bootstrap lower bound >= -0.45,
//!      PARITY_OUT = csv path for our per-step drafts.
//!
//! `PARITY_ASSERT` is meaningful for `base` / `incseed` only. `noseed` is NOT like
//! for like (the reference attends 127 zeroed window rows; ours attends only the
//! written ones). 89 overlapping blocks are few (effective n ~19), so read the
//! PAIRED numbers first: draft agreement given identical earlier drafts, and the
//! moving-block bootstrap interval of E(ours) - E(ref).
//! The exit runs on the iGPU here; production runs it on the dGPU (gfx1201), so
//! this does not cover production exit numerics (M-A A3 / M3 do).
//! `V41_MTP_KV_QUANT=v4` selects the legacy ring quantizer for an A/B.
//!
//! Needs the checkpoint and the iGPU (drafter + exit on gfx1151, ~8.7 GB), so
//! run it with the hub DOWN:
//!
//!   HIP_VISIBLE_DEVICES=0,1 CARGO_TARGET_DIR=target-v41 nix develop -c \
//!     cargo test -p v4flash-kernels --release --features v41 --test dspark_parity \
//!     -- --ignored --nocapture

use color_eyre::eyre::{self, eyre};
use v4flash_core::{V41HfWeights, WeightSrc};
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::config::{HC_DIM, N_EMBD};
use v4flash_kernels::het::engine::DeviceEngine;
use v4flash_kernels::het::drafter::{drafter_rope, DrafterExit, DrafterState, DRAFT_BLOCK, DRAFT_NOISE_TOKEN, DRAFT_WINDOW};
use v4flash_kernels::het::weights::{DrafterExitWeights, DrafterWeights};

const NSRC: usize = 3;

fn pick_igpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1151"))
}

fn read_f32(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn read_i32(path: &str) -> Vec<i32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// torch's f32 -> bf16 cast: round to nearest, ties to even (NaN not expected).
fn bf16_round(x: f32) -> f32 {
    let b = x.to_bits();
    let r = b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(r)
}

fn env_or(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}

struct RefStep {
    i: usize,
    drafts: [i32; DRAFT_BLOCK],
    greedy: [i32; DRAFT_BLOCK],
    conf: [f32; DRAFT_BLOCK],
}

fn load_ref(dir: &str, name: &str) -> Vec<RefStep> {
    let r = read_i32(&format!("{dir}/ref_{name}.bin"));
    let c = read_f32(&format!("{dir}/ref_{name}_conf.bin"));
    let w = 1 + 2 * DRAFT_BLOCK;
    assert_eq!(r.len() % w, 0, "ref_{name}.bin: {} ints is not a multiple of {w}", r.len());
    let n = r.len() / w;
    assert_eq!(c.len(), n * DRAFT_BLOCK, "ref_{name}_conf.bin length");
    (0..n)
        .map(|s| {
            let rec = &r[s * w..(s + 1) * w];
            let mut st = RefStep { i: rec[0] as usize, drafts: [0; DRAFT_BLOCK], greedy: [0; DRAFT_BLOCK], conf: [0.0; DRAFT_BLOCK] };
            st.drafts.copy_from_slice(&rec[1..1 + DRAFT_BLOCK]);
            st.greedy.copy_from_slice(&rec[1 + DRAFT_BLOCK..w]);
            st.conf.copy_from_slice(&c[s * DRAFT_BLOCK..(s + 1) * DRAFT_BLOCK]);
            st
        })
        .collect()
}

#[test]
#[ignore = "needs the checkpoint and the iGPU; run with the hub down"]
fn drafter_matches_reference() {
    install_panic_handler().ok();
    let home = std::env::var("HOME").unwrap_or_default();
    let dir = env_or("PARITY_DIR", &format!("{home}/.cache/deepstrix/v41/agentic/gen2/parity"));
    let refname = env_or("PARITY_REF", "base");
    let seeded = refname != "noseed";
    let use_plain = refname == "nomarkov";
    let mh_bf16 = env_or("PARITY_MH_BF16", "1") != "0";
    let show: usize = env_or("PARITY_SHOW", "6").parse().unwrap_or(6);
    let model = env_or("V41_MODEL", &format!("{home}/.cache/deepstrix/models/dsv4.1f"));

    let ne = N_EMBD as usize;
    let row = NSRC * ne;
    // PARITY_MH: an alternative [T, 3, 5120] f32 file, e.g. the ENGINE's own
    // residuals at layers 36-38 dumped from a teacher-forced run of the same
    // transcript (M-A step A3), to score our drafter on our main model.
    let mh_path = env_or("PARITY_MH", &format!("{dir}/mh.bin"));
    let mut mh = read_f32(&mh_path);
    let tok = read_i32(&format!("{dir}/tokens.bin"));
    let argmax = read_i32(&format!("{dir}/argmax.bin"));
    let t = tok.len();
    assert_eq!(mh.len(), t * row, "mh.bin is not [T={t}, 3, {ne}]");
    assert_eq!(argmax.len(), t);
    if mh_bf16 {
        for v in mh.iter_mut() {
            *v = bf16_round(*v);
        }
    }
    let refs = load_ref(&dir, &refname);
    let seed = refs[0].i;
    println!(
        "parity: ref={refname} T={t} seed={seed} steps={} seeded={seeded} markov={} mh_bf16={mh_bf16} kv_quant={} mh={mh_path}",
        refs.len(),
        !use_plain,
        std::env::var("V41_MTP_KV_QUANT").unwrap_or_else(|_| "v41".into())
    );
    if refname == "noseed" {
        println!(
            "  NOTE: not like for like. The reference's unseeded window attends all 128 slots \
             (127 zero rows: get_dspark_topk_idxs uses min(win, start_pos+1) over a zeroed cache); \
             ours attends only the rows actually written."
        );
    }

    let hf = V41HfWeights::open(&model, None).expect("open checkpoint");
    let dev = pick_igpu().expect("igpu");
    let arch = dev.properties().expect("props").gcn_arch_name;
    let e = DeviceEngine::for_arch(dev, &arch).expect("engine");
    let w = DrafterWeights::load(&hf, dev, 40).expect("load drafter");
    let xw = DrafterExitWeights::load(&hf, dev).expect("load exit weights");
    let head = v4flash_kernels::weights::load_to_device(&hf, "output.weight", dev.id).expect("load tied head");
    let src: WeightSrc = (&hf).into();
    let mk = src.tensor("mtp.2.markov_embd.weight").expect("markov embed tensor");
    let mk_dtype = mk.dtype;
    let mk_bytes = src.read_tensor(mk).expect("read markov embed");
    let te = src.tensor("token_embd.weight").expect("token_embd");
    let te_dtype = te.dtype;
    let te_bytes = src.read_tensor(te).expect("read token_embd");
    let mut noise_row = vec![0.0f32; HC_DIM as usize];
    v4flash_kernels::embed::embed_lookup(&te_bytes, te_dtype, DRAFT_NOISE_TOKEN, &mut noise_row).expect("embed noise");

    let mut st = DrafterState::alloc(dev.id).expect("alloc state");
    let mut ex = DrafterExit::alloc(dev.id).expect("alloc exit");
    let rope = drafter_rope();

    if seeded {
        let win = DRAFT_WINDOW as usize;
        // PARITY_SEED_ROWS=n (1..=8): seed through the batched writer
        // (`ring_write_rows`, n rows per call, as the arena writes a verify
        // block's kept rows); the drafts must be identical to the per-row seed.
        let batch = std::env::var("PARITY_SEED_ROWS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
        let lo = seed.saturating_sub(win);
        if batch > 0 {
            let mut p = lo;
            while p < seed {
                let q = (p + batch).min(seed);
                st.ring_write_rows(&e, &e.compute, &w, &rope, p as u32, &mh[p * row..q * row]).expect("ring rows");
                p = q;
            }
            println!("seeded {} ring rows through ring_write_rows, {batch} per call", seed - lo);
        } else {
            for p in lo..seed {
                st.inject_main_hidden(&mh[p * row..(p + 1) * row]).expect("inject");
                st.ring_write_only(&e, &e.compute, &w, &rope, p as u32).expect("ring write");
            }
        }
        e.compute.synchronize().expect("sync");
    }

    let mut agree = [0usize; DRAFT_BLOCK];
    let mut acc = [0usize; DRAFT_BLOCK];
    let mut ref_acc = [0usize; DRAFT_BLOCK];
    let mut prefix = [0usize; DRAFT_BLOCK];
    let mut ref_prefix = [0usize; DRAFT_BLOCK];
    let mut conf_err = [0.0f64; DRAFT_BLOCK];
    // Paired per-step statistics (89 overlapping blocks are too few for an
    // absolute bar: lag-1 autocorrelation ~0.65, effective n ~19).
    let mut agree_cond = [0usize; DRAFT_BLOCK]; // draft k equal GIVEN drafts < k equal
    let mut agree_cond_n = [0usize; DRAFT_BLOCK];
    let mut diff_per_step: Vec<f64> = Vec::with_capacity(refs.len()); // accepted(ours) - accepted(ref)
    let mut csv = String::from("i,d0,d1,d2,d3,d4,p0,p1,p2,p3,p4,c0,c1,c2,c3,c4\n");
    let mut token_row = vec![0.0f32; HC_DIM as usize];
    let mut h_host = vec![0.0f32; st.h.len()];
    let mut pre_host = vec![0.0f32; st.pre_carry().len()];
    // Wall split per draft (ms): drafter enqueue, drafter device (to sync),
    // exit. V41_MTP_GRAPH=1 replays the drafter as one graph; the CSV
    // (PARITY_OUT, full-precision conf) must be byte-identical either way.
    let (mut t_enq, mut t_dev, mut t_exit) = (Vec::new(), Vec::new(), Vec::new());
    for (n, r) in refs.iter().enumerate() {
        let i = r.i;
        assert!(i + 1 + DRAFT_BLOCK < t, "step {i} runs past the transcript");
        let next = tok[i + 1];
        v4flash_kernels::embed::embed_lookup(&te_bytes, te_dtype, next, &mut token_row).expect("embed token");
        st.inject_main_hidden(&mh[i * row..(i + 1) * row]).expect("inject");
        let t0 = std::time::Instant::now();
        st.forward(&e, &e.compute, &w, &rope, i as u32, &token_row, &noise_row).expect("drafter forward");
        let t1 = std::time::Instant::now();
        e.compute.synchronize().expect("sync");
        let t2 = std::time::Instant::now();
        st.h.copy_to_host(&mut h_host).expect("read h");
        st.pre_carry().copy_to_host(&mut pre_host).expect("read pre");
        let (ids, plain) = ex
            .forward(&e, &e.compute, &h_host, &pre_host, &xw, &head, &mk_bytes, mk_dtype, next)
            .expect("exit forward");
        e.compute.synchronize().expect("sync");
        let ms = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1e3;
        t_enq.push(ms(t0, t1));
        t_dev.push(ms(t1, t2));
        t_exit.push(ms(t2, std::time::Instant::now()));
        let ours = if use_plain { plain } else { ids };

        let (mut ok, mut rok) = (true, true);
        let mut same_so_far = true;
        let (mut n_ours, mut n_ref) = (0usize, 0usize);
        for k in 0..DRAFT_BLOCK {
            if same_so_far {
                agree_cond_n[k] += 1;
                agree_cond[k] += usize::from(ours[k] == r.drafts[k]);
            }
            same_so_far &= ours[k] == r.drafts[k];
            let g = argmax[i + 1 + k];
            // Hard assert (the run script builds --release): misaligned targets
            // would push BOTH E values down and pass the relative bar vacuously.
            assert_eq!(g, r.greedy[k], "argmax.bin and the record disagree at i={i} k={k}");
            agree[k] += usize::from(ours[k] == r.drafts[k]);
            acc[k] += usize::from(ours[k] == g);
            ref_acc[k] += usize::from(r.drafts[k] == g);
            ok &= ours[k] == g;
            rok &= r.drafts[k] == g;
            prefix[k] += usize::from(ok);
            ref_prefix[k] += usize::from(rok);
            n_ours += usize::from(ok);
            n_ref += usize::from(rok);
            conf_err[k] += (ex.conf[k] as f64 - r.conf[k] as f64).abs();
        }
        diff_per_step.push(n_ours as f64 - n_ref as f64);
        if n < show {
            println!(
                "  i={i} next={next}\n    ours  {ours:?} conf {:?}\n    ref   {:?} conf {:?}\n    greedy {:?}",
                ex.conf.map(|c| (c * 100.0).round() / 100.0),
                r.drafts,
                r.conf.map(|c| (c * 100.0).round() / 100.0),
                r.greedy
            );
        }
        csv += &format!(
            "{i},{},{},{}\n",
            ids.map(|x| x.to_string()).join(","),
            plain.map(|x| x.to_string()).join(","),
            ex.conf.map(|x| format!("{x}")).join(",")
        );
    }
    // The first draft captures the graph (when on): report it apart.
    let med = |v: &[f64]| {
        let mut w = v[1.min(v.len())..].to_vec();
        w.sort_by(|a, b| a.partial_cmp(b).unwrap());
        w.get(w.len() / 2).copied().unwrap_or(f64::NAN)
    };
    println!(
        "draft timing (median ms over {} steps after the first; first {:.2}): drafter enqueue {:.2}, drafter device-wait {:.2}, exit {:.2} (V41_MTP_GRAPH={})",
        t_enq.len().saturating_sub(1),
        t_enq.first().copied().unwrap_or(f64::NAN) + t_dev.first().copied().unwrap_or(f64::NAN),
        med(&t_enq), med(&t_dev), med(&t_exit),
        std::env::var("V41_MTP_GRAPH").unwrap_or_default()
    );

    let n = refs.len() as f64;
    let e_of = |p: &[usize; DRAFT_BLOCK]| 1.0 + p.iter().map(|&x| x as f64 / n).sum::<f64>();
    println!("\ndepth  draft==ref  ours greedy-acc  ref greedy-acc  ours prefix  ref prefix  |conf-ref|");
    for k in 0..DRAFT_BLOCK {
        println!(
            "  d{}     {:.3}         {:.3}           {:.3}          {:.3}        {:.3}      {:.3}",
            k + 1,
            agree[k] as f64 / n,
            acc[k] as f64 / n,
            ref_acc[k] as f64 / n,
            prefix[k] as f64 / n,
            ref_prefix[k] as f64 / n,
            conf_err[k] / n
        );
    }
    let (eo, er) = (e_of(&prefix), e_of(&ref_prefix));
    println!("E (K=5): ours {eo:.3}   reference {er:.3}   gap {:.3}", er - eo);
    println!(
        "draft agreement given identical earlier drafts: {}",
        (0..DRAFT_BLOCK)
            .map(|k| format!("d{} {:.3} (n={})", k + 1, agree_cond[k] as f64 / agree_cond_n[k].max(1) as f64, agree_cond_n[k]))
            .collect::<Vec<_>>()
            .join("  ")
    );
    // Moving-block bootstrap (block 6 >= the block overlap) of the PAIRED
    // per-step difference: accepted(ours) - accepted(reference).
    let (lo, hi) = {
        let d = &diff_per_step;
        let (m, bl) = (d.len(), 6usize.min(d.len().max(1)));
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut means: Vec<f64> = (0..4000)
            .map(|_| {
                let (mut s, mut c) = (0.0, 0usize);
                while c < m {
                    let start = (next() % (m - bl + 1) as u64) as usize;
                    for x in &d[start..start + bl] {
                        if c < m {
                            s += x;
                            c += 1;
                        }
                    }
                }
                s / m as f64
            })
            .collect();
        means.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (means[(0.05 * means.len() as f64) as usize], means[(0.95 * means.len() as f64) as usize])
    };
    println!("E(ours) - E(ref), 90% moving-block bootstrap: [{lo:+.3}, {hi:+.3}]");
    // The reference's E recomputed from its records must equal what its JSON
    // recorded; otherwise the targets or the record parsing are off.
    let json = format!("{dir}/../dspark_accept_{refname}.json");
    let asserting = std::env::var("PARITY_ASSERT").as_deref() == Ok("1");
    match std::fs::read(&json) {
        Ok(bytes) => {
            let v: serde_json::Value = serde_json::from_slice(&bytes).expect("reference json");
            let want = v["expected_tokens"].get(DRAFT_BLOCK - 1).and_then(|x| x.as_f64());
            let want = want.unwrap_or_else(|| panic!("{json}: no expected_tokens[{}]", DRAFT_BLOCK - 1));
            assert!((want - er).abs() < 1e-6, "reference E from records {er:.6} != its JSON {want:.6}");
        }
        Err(e) if asserting => panic!("{json}: {e} (the alignment check needs the reference JSON)"),
        Err(e) => println!("WARNING: {json}: {e}; reference E NOT cross-checked against its JSON"),
    }
    if let Ok(p) = std::env::var("PARITY_OUT") {
        std::fs::write(&p, csv).expect("write PARITY_OUT");
        println!("per-step drafts written to {p}");
    }
    if asserting {
        // THE pass rule (plan M-A): owner's bar -- a numerics-level shortfall
        // (~4.2 vs 4.38) is fine -- as a point gap, AND the paired interval's
        // lower bound within ~one reference SE (0.45; effective n ~19).
        assert!(eo >= er - 0.2, "E {eo:.3} is more than 0.2 below the reference's {er:.3}");
        assert!(lo >= -0.45, "paired 90% interval lower bound {lo:+.3} < -0.45");
    }
}
