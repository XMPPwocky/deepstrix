//! Laguna-S-2.1 — routed-expert selection histogram + hot-set CAPTURE FRACTION.
//!
//! The het-decode "hot experts on dGPU" overlap can only move as much iGPU work
//! to the dGPU as the per-token top-10 selection actually lands on the K
//! dGPU-resident experts. This test MEASURES that ceiling: it greedily generates
//! real text and accumulates a per-layer expert-selection histogram, then reports
//! the top-K capture fraction for K = 4/8/16/32.
//!
//! Run (server stopped; GPUs free):
//!   LAGUNA_EXPERT_HIST=1 nix develop --command cargo test --release \
//!       -p v4flash-kernels --test laguna_expert_hist -- --ignored --nocapture

use std::time::Instant;

use color_eyre::eyre::{self, eyre};
use v4flash_core::gguf::Gguf;
use v4flash_core::tokenizer::BpeVocab;
use v4flash_hip::Device;
use v4flash_kernels::laguna_het::LagunaHetModel;

const GGUF_PATH: &str = "/persist/lumi/models/laguna-s-2.1-int4/laguna-s-2.1-Q4_K_M.gguf";

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

#[test]
#[ignore = "drives BOTH GPUs + needs the 75GB Laguna GGUF; run explicitly"]
fn laguna_expert_hist() -> eyre::Result<()> {
    let _ = v4flash_hip::install_panic_handler();
    std::env::set_var("LAGUNA_EXPERT_HIST", "1");

    if !std::path::Path::new(GGUF_PATH).exists() {
        eprintln!("SKIP: {GGUF_PATH} not present");
        return Ok(());
    }

    let n_gen = env_usize("HIST_NGEN", 256);

    let devs = Device::all()?;
    let dgpu = devs
        .iter()
        .find(|d| d.properties().map(|p| p.gcn_arch_name.starts_with("gfx1201")).unwrap_or(false))
        .cloned()
        .ok_or_else(|| eyre!("no gfx1201 (dGPU) device"))?;
    let igpu = devs
        .iter()
        .find(|d| d.properties().map(|p| p.gcn_arch_name.starts_with("gfx1151")).unwrap_or(false))
        .cloned()
        .ok_or_else(|| eyre!("no gfx1151 (Strix iGPU) device"))?;
    let dgpu_arch = dgpu.properties()?.gcn_arch_name;
    let igpu_arch = igpu.properties()?.gcn_arch_name;

    let g = Gguf::open(GGUF_PATH)?;
    let vocab = BpeVocab::from_gguf(&g)?;
    // Greedy decode of a base model loops after a few tokens (measured: 2
    // distinct tokens over 256 steps) which would badly BIAS the histogram.
    // Instead TEACHER-FORCE a long, lexically diverse passage through the
    // decode path: feed each real token at its real position and collect the
    // routing it induces. This exercises the router over genuinely varied
    // content, giving an honest capture-fraction ceiling.
    let passage = "The history of science is a long and winding story that spans many \
        cultures and centuries. In the ancient world, astronomers in Babylon, Egypt, Greece, \
        China, and India charted the motions of the planets and the changing phases of the moon, \
        recording their observations on clay tablets, papyrus scrolls, and carved stone. \
        Philosophers argued about whether matter was continuous or made of tiny indivisible \
        particles, whether the earth stood still at the center of the cosmos or wheeled around \
        the sun, and whether disease arose from imbalanced humors or invisible contagions. \
        During the medieval period, scholars in Baghdad, Cordoba, and Timbuktu translated, \
        preserved, and extended this inheritance, advancing algebra, optics, chemistry, and \
        medicine while Europe slowly rediscovered the classical texts. The Renaissance brought \
        a renewed appetite for direct observation: Vesalius dissected human cadavers, Galileo \
        turned his telescope toward Jupiter's moons, and Kepler wrestled elliptical orbits out \
        of Tycho Brahe's meticulous tables. Newton unified terrestrial and celestial mechanics \
        under a single law of gravitation, and the Enlightenment that followed applied reason \
        and experiment to electricity, combustion, geology, and the classification of living \
        things. The nineteenth century industrialized discovery: thermodynamics explained steam \
        engines, Maxwell's equations wove together electricity, magnetism, and light, Darwin \
        proposed natural selection, and Mendel quietly counted his pea plants in a monastery \
        garden. The twentieth century shattered old certainties, with relativity bending space \
        and time, quantum mechanics dissolving determinism, and molecular biology decoding the \
        double helix that carries the instructions of life itself.";
    let ids: Vec<usize> = vocab.encode_laguna(passage).into_iter().map(|i| i as usize).collect();
    let n_feed = ids.len().min(n_gen + 1);
    println!("passage tokens: {} (feeding {})", ids.len(), n_feed);

    let max_kv = ids.len() + 8;
    let t_load = Instant::now();
    let mut model =
        LagunaHetModel::load(GGUF_PATH, dgpu.clone(), &dgpu_arch, igpu.clone(), &igpu_arch, max_kv)?;
    println!("model loaded in {:.1}s", t_load.elapsed().as_secs_f32());

    // TEACHER-FORCE: feed each real token through the decode path at its real
    // position (predictions ignored). Histogram accumulates over all of them.
    model.reset();
    for (pos, &tok) in ids.iter().take(n_feed).enumerate() {
        let _ = model.decode_step(tok, pos)?;
    }
    let distinct = {
        let mut v: Vec<usize> = ids.iter().take(n_feed).copied().collect();
        v.sort_unstable();
        v.dedup();
        v.len()
    };
    println!("distinct fed tokens: {distinct}/{n_feed}");

    // ---- capture fraction ----
    println!("\n=== HOT-SET CAPTURE FRACTION (per-token top-10 slots on K resident experts) ===");
    println!("(uniform baseline for K experts of 256 = K/256)");
    for &k in &[4usize, 8, 16, 24, 32] {
        let (overall, per_layer) = model.hot_capture_fraction(k);
        let uniform = k as f64 / 256.0;
        let min = per_layer.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = per_layer.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!(
            "  K={:>2}: overall capture {:5.1}%  (uniform {:4.1}%, skew {:.1}x)  per-layer[min {:.1}% max {:.1}%]",
            k,
            overall * 100.0,
            uniform * 100.0,
            overall / uniform,
            min * 100.0,
            max * 100.0,
        );
    }

    // dump a couple of per-layer top-8 lists for sanity
    let hot8 = model.hot_experts_per_layer(8);
    println!("\nlayer 1 top-8 hot experts: {:?}", hot8.get(1));
    println!("layer 24 top-8 hot experts: {:?}", hot8.get(24));
    println!("layer 47 top-8 hot experts: {:?}", hot8.get(47));

    // Optionally emit a hot-experts residency file for LAGUNA_HOT_EXPERTS_DGPU:
    // N_LAYER lines, line `il` = space-separated hot expert ids (blank for the
    // dense layer 0). HIST_K controls K.
    if let Ok(out) = std::env::var("LAGUNA_HOT_EXPERTS_OUT") {
        let k = env_usize("HIST_K", 8);
        let hot = model.hot_experts_per_layer(k);
        let mut s = String::new();
        for (il, ids) in hot.iter().enumerate() {
            if il == 0 {
                s.push('\n'); // dense layer, no experts
                continue;
            }
            let line: Vec<String> = ids.iter().map(|e| e.to_string()).collect();
            s.push_str(&line.join(" "));
            s.push('\n');
        }
        std::fs::write(&out, s)?;
        println!("wrote hot-experts file (K={k}) -> {out}");
    }

    Ok(())
}
