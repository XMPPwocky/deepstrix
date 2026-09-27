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
//!      PARITY_ASSERT = 1 to fail below the bars (E no more than 0.2 below the
//!      reference, i.e. ~4.2 for `base`; d1 draft agreement >= 0.90),
//!      PARITY_OUT = csv path for our per-step drafts.
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
use v4flash_kernels::het::mtp::{mtp_rope, MtpExit, MtpState, MTP_BLOCK, MTP_NOISE_TOKEN, MTP_WINDOW};
use v4flash_kernels::het::weights::{MtpExitWeights, MtpWeights};

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
    drafts: [i32; MTP_BLOCK],
    greedy: [i32; MTP_BLOCK],
    conf: [f32; MTP_BLOCK],
}

fn load_ref(dir: &str, name: &str) -> Vec<RefStep> {
    let r = read_i32(&format!("{dir}/ref_{name}.bin"));
    let c = read_f32(&format!("{dir}/ref_{name}_conf.bin"));
    let w = 1 + 2 * MTP_BLOCK;
    assert_eq!(r.len() % w, 0, "ref_{name}.bin: {} ints is not a multiple of {w}", r.len());
    let n = r.len() / w;
    assert_eq!(c.len(), n * MTP_BLOCK, "ref_{name}_conf.bin length");
    (0..n)
        .map(|s| {
            let rec = &r[s * w..(s + 1) * w];
            let mut st = RefStep { i: rec[0] as usize, drafts: [0; MTP_BLOCK], greedy: [0; MTP_BLOCK], conf: [0.0; MTP_BLOCK] };
            st.drafts.copy_from_slice(&rec[1..1 + MTP_BLOCK]);
            st.greedy.copy_from_slice(&rec[1 + MTP_BLOCK..w]);
            st.conf.copy_from_slice(&c[s * MTP_BLOCK..(s + 1) * MTP_BLOCK]);
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
        "parity: ref={refname} T={t} seed={seed} steps={} seeded={seeded} markov={} mh_bf16={mh_bf16}",
        refs.len(),
        !use_plain
    );

    let hf = V41HfWeights::open(&model, None).expect("open checkpoint");
    let dev = pick_igpu().expect("igpu");
    let arch = dev.properties().expect("props").gcn_arch_name;
    let e = DeviceEngine::for_arch(dev, &arch).expect("engine");
    let w = MtpWeights::load(&hf, dev, 40).expect("load drafter");
    let xw = MtpExitWeights::load(&hf, dev).expect("load exit weights");
    let head = v4flash_kernels::weights::load_to_device(&hf, "output.weight", dev.id).expect("load tied head");
    let src: WeightSrc = (&hf).into();
    let mk = src.tensor("mtp.2.markov_embd.weight").expect("markov embed tensor");
    let mk_dtype = mk.dtype;
    let mk_bytes = src.read_tensor(mk).expect("read markov embed");
    let te = src.tensor("token_embd.weight").expect("token_embd");
    let te_dtype = te.dtype;
    let te_bytes = src.read_tensor(te).expect("read token_embd");
    let mut noise_row = vec![0.0f32; HC_DIM as usize];
    v4flash_kernels::embed::embed_lookup(&te_bytes, te_dtype, MTP_NOISE_TOKEN, &mut noise_row).expect("embed noise");

    let mut st = MtpState::alloc(dev.id).expect("alloc state");
    let mut ex = MtpExit::alloc(dev.id).expect("alloc exit");
    let rope = mtp_rope();

    if seeded {
        let win = MTP_WINDOW as usize;
        for p in seed.saturating_sub(win)..seed {
            st.inject_main_hidden(&mh[p * row..(p + 1) * row]).expect("inject");
            st.ring_write_only(&e, &e.compute, &w, &rope, p as u32).expect("ring write");
        }
        e.compute.synchronize().expect("sync");
    }

    let mut agree = [0usize; MTP_BLOCK];
    let mut acc = [0usize; MTP_BLOCK];
    let mut ref_acc = [0usize; MTP_BLOCK];
    let mut prefix = [0usize; MTP_BLOCK];
    let mut ref_prefix = [0usize; MTP_BLOCK];
    let mut conf_err = [0.0f64; MTP_BLOCK];
    let mut csv = String::from("i,d0,d1,d2,d3,d4,p0,p1,p2,p3,p4,c0,c1,c2,c3,c4\n");
    let mut token_row = vec![0.0f32; HC_DIM as usize];
    let mut h_host = vec![0.0f32; st.h.len()];
    let mut pre_host = vec![0.0f32; st.pre_carry().len()];
    for (n, r) in refs.iter().enumerate() {
        let i = r.i;
        assert!(i + 1 + MTP_BLOCK < t, "step {i} runs past the transcript");
        let next = tok[i + 1];
        v4flash_kernels::embed::embed_lookup(&te_bytes, te_dtype, next, &mut token_row).expect("embed token");
        st.inject_main_hidden(&mh[i * row..(i + 1) * row]).expect("inject");
        st.forward(&e, &e.compute, &w, &rope, i as u32, &token_row, &noise_row).expect("drafter forward");
        e.compute.synchronize().expect("sync");
        st.h.copy_to_host(&mut h_host).expect("read h");
        st.pre_carry().copy_to_host(&mut pre_host).expect("read pre");
        let (ids, plain) = ex
            .forward(&e, &e.compute, &h_host, &pre_host, &xw, &head, &mk_bytes, mk_dtype, next)
            .expect("exit forward");
        e.compute.synchronize().expect("sync");
        let ours = if use_plain { plain } else { ids };

        let (mut ok, mut rok) = (true, true);
        for k in 0..MTP_BLOCK {
            let g = argmax[i + 1 + k];
            debug_assert_eq!(g, r.greedy[k], "argmax.bin and the record disagree at i={i} k={k}");
            agree[k] += usize::from(ours[k] == r.drafts[k]);
            acc[k] += usize::from(ours[k] == g);
            ref_acc[k] += usize::from(r.drafts[k] == g);
            ok &= ours[k] == g;
            rok &= r.drafts[k] == g;
            prefix[k] += usize::from(ok);
            ref_prefix[k] += usize::from(rok);
            conf_err[k] += (ex.conf[k] as f64 - r.conf[k] as f64).abs();
        }
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
            ex.conf.map(|x| format!("{x:.4}")).join(",")
        );
    }

    let n = refs.len() as f64;
    let e_of = |p: &[usize; MTP_BLOCK]| 1.0 + p.iter().map(|&x| x as f64 / n).sum::<f64>();
    println!("\ndepth  draft==ref  ours greedy-acc  ref greedy-acc  ours prefix  ref prefix  |conf-ref|");
    for k in 0..MTP_BLOCK {
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
    if let Ok(p) = std::env::var("PARITY_OUT") {
        std::fs::write(&p, csv).expect("write PARITY_OUT");
        println!("per-step drafts written to {p}");
    }
    if std::env::var("PARITY_ASSERT").as_deref() == Ok("1") {
        // Owner's bar: a numerics-level shortfall (~4.2 vs 4.38) is fine.
        assert!(eo >= er - 0.2, "E {eo:.3} is more than 0.2 below the reference's {er:.3}");
        assert!(agree[0] as f64 / n >= 0.90, "d1 draft agreement {:.3}", agree[0] as f64 / n);
    }
}
