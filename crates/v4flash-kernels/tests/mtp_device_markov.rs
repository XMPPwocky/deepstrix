//! `V41_MTP_DEVICE_MARKOV`: the DSpark exit's markov loop on the device vs the host loop it replaces,
//! on the dGPU (where the exit runs), from identical pre-markov logits and pre-norm `x`: drafts, every
//! position's q (the verifier's input), the confidence head, and the biased logits must be identical,
//! and the device draw must never disagree with the host re-draw. Greedy and three sampling settings.
//! Synthetic weights at the real shapes (markov table 129280 x 256 f16 = 66 MB, markov head Q8_0 35 MB);
//! runs beside a live hub. Also prints the wall time of each loop.
//!
//!   cargo test --release --features v41 -p v4flash-kernels --test mtp_device_markov -- --ignored --nocapture
#![cfg(feature = "v41")]
use color_eyre::eyre::{self, eyre};
use v4flash_core::gguf::GgufType;
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::config::{N_EMBD, N_VOCAB};
use v4flash_kernels::het::engine::DeviceEngine;
use v4flash_kernels::het::mtp::{DraftSampling, MtpExit, MARKOV_DRAW_MISMATCH, MTP_BLOCK, MTP_DRAFT_TOP_M, MTP_MARKOV_RANK};
use v4flash_kernels::het::weights::MtpExitWeights;
use v4flash_kernels::weights::DeviceWeight;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next() & 0xFFFFFF) as f32 / 16777216.0
    }
    fn sym(&mut self, a: f32) -> f32 {
        (self.unit() * 2.0 - 1.0) * a
    }
}

fn upload<T: Copy>(id: i32, host: &[T]) -> eyre::Result<DeviceBuffer<T>> {
    let mut d: DeviceBuffer<T> = DeviceBuffer::new(id, host.len())?;
    d.copy_from_host(host)?;
    Ok(d)
}

fn f16_bits(rng: &mut Lcg, lo_exp: u32, span: u32) -> u16 {
    let r = rng.next();
    ((r >> 31) << 15) as u16 | ((lo_exp + (r >> 10) % span) << 10) as u16 | (r & 0x3FF) as u16
}

#[test]
#[ignore]
fn device_markov_matches_host_loop() -> eyre::Result<()> {
    install_panic_handler()?;
    let dev = Device::all()?
        .into_iter()
        .find(|d| d.properties().map(|p| p.gcn_arch_name.starts_with("gfx1201")).unwrap_or(false))
        .ok_or_else(|| eyre!("no gfx1201"))?;
    let arch = dev.properties()?.gcn_arch_name;
    dev.set_current()?;
    let id = dev.id;
    let e = DeviceEngine::for_arch(dev, &arch)?;
    let s = Stream::new(id)?;
    let (nv, ne, r) = (N_VOCAB as usize, N_EMBD as usize, MTP_MARKOV_RANK);
    let mut rng = Lcg(0x3A_4C0F);

    // Markov table [N_VOCAB, 256] f16, |v| ~ 2^-6 .. 2^1.
    let mut table = Vec::with_capacity(nv * r * 2);
    for _ in 0..nv * r {
        table.extend_from_slice(&f16_bits(&mut rng, 9, 8).to_le_bytes());
    }
    // Markov head [N_VOCAB, 256] Q8_0 in the kernels' REPACKED row layout (M18, q8_0_matvec.hip):
    // each row = its 8 f16 scales, then its 256 i8 quants. Scales ~2^-13 so the bias (~0.3 nats)
    // competes with the logits instead of deciding every draft.
    let mut head = Vec::with_capacity(nv * (r / 32) * 34);
    for _ in 0..nv {
        for _ in 0..r / 32 {
            head.extend_from_slice(&f16_bits(&mut rng, 1, 2).to_le_bytes());
        }
        for _ in 0..r {
            head.push((rng.next() % 255) as u8);
        }
    }
    let w = MtpExitWeights {
        norm: upload(id, &vec![1.0f32; ne])?,
        markov_head: DeviceWeight { buffer: upload(id, &head)?, n_elements: (nv * r) as u64, dtype: GgufType::Q8_0, shape: vec![r as u64, nv as u64] },
        confidence: upload(id, &(0..ne + r).map(|_| rng.sym(0.01)).collect::<Vec<f32>>())?,
    };
    let mut host_exit = MtpExit::alloc(id)?;
    let mut dev_exit = MtpExit::alloc(id)?;

    let cases: [(&str, Option<(f32, f32, f32)>); 4] = [
        ("argmax", None),
        ("T1 p1 m0", Some((1.0, 1.0, 0.0))),
        ("T0.7 p0.9 m0.05", Some((0.7, 0.9, 0.05))),
        ("T1.3 p0.95 m0", Some((1.3, 0.95, 0.0))),
    ];
    let before = MARKOV_DRAW_MISMATCH.load(std::sync::atomic::Ordering::Relaxed);
    let (mut t_host, mut t_dev, mut runs) = (0f64, 0f64, 0u32);
    for (ci, (name, cfg)) in cases.iter().enumerate() {
        for rep in 0..6u32 {
            // Flat-ish logits (spread 0.8 nats) so q spans many of the top-M and the draw really decides.
            let logits: Vec<f32> = (0..MTP_BLOCK * nv).map(|_| rng.sym(0.8)).collect();
            let x: Vec<f32> = (0..MTP_BLOCK * ne).map(|_| rng.sym(1.0)).collect();
            let first = (rng.next() % N_VOCAB) as i32;
            let sampling = cfg.map(|(t, p, m)| {
                let mut u = [0f32; MTP_BLOCK];
                for v in u.iter_mut() {
                    *v = rng.unit();
                }
                DraftSampling { temperature: t, top_p: p, min_p_rel: m, m: MTP_DRAFT_TOP_M, u }
            });
            let t0 = std::time::Instant::now();
            let a = host_exit.markov_loop_for_test(&e, &s, &w, &table, GgufType::F16, first, sampling.as_ref(), &logits, &x, false)?;
            let t1 = std::time::Instant::now();
            let b = dev_exit.markov_loop_for_test(&e, &s, &w, &table, GgufType::F16, first, sampling.as_ref(), &logits, &x, true)?;
            let t2 = std::time::Instant::now();
            if rep > 0 {
                t_host += (t1 - t0).as_secs_f64() * 1e3;
                t_dev += (t2 - t1).as_secs_f64() * 1e3;
                runs += 1;
            }
            assert_eq!(a, b, "{name} rep {rep}: drafts differ");
            assert_eq!(host_exit.q, dev_exit.q, "{name} rep {rep}: q differs");
            assert!(host_exit.conf.iter().zip(&dev_exit.conf).all(|(x, y)| x.to_bits() == y.to_bits()), "{name} rep {rep}: confidence differs");
            let (mut la, mut lb) = (vec![0f32; MTP_BLOCK * nv], vec![0f32; MTP_BLOCK * nv]);
            host_exit.logits.slice_view(0, MTP_BLOCK * nv).copy_to_host(&mut la)?;
            dev_exit.logits.slice_view(0, MTP_BLOCK * nv).copy_to_host(&mut lb)?;
            let d = la.iter().zip(&lb).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            assert_eq!(d, 0, "{name} rep {rep}: biased logits differ in {d} values");
            if rep == 0 {
                eprintln!("{name}: drafts {a:?} (q sizes {:?})", dev_exit.q.iter().map(|q| q.len()).collect::<Vec<_>>());
            }
        }
    }
    let mism = MARKOV_DRAW_MISMATCH.load(std::sync::atomic::Ordering::Relaxed) - before;
    assert_eq!(mism, 0, "device draw disagreed with the host re-draw {mism} times");
    eprintln!("PASS: {} cases x 6 reps identical (drafts, q, confidence, biased logits), 0 draw mismatches", cases.len());
    eprintln!("markov loop wall time (incl. sync + readbacks): host {:.2} ms, device {:.2} ms", t_host / runs as f64, t_dev / runs as f64);
    let qsz: Vec<usize> = dev_exit.q.iter().map(|q| q.len()).collect();
    assert!(qsz.iter().any(|&n| n > 4), "sampling never spanned several candidates: q sizes {qsz:?}");
    dev_exit.time_markov_kernels(&e, &s, &w, 200)?;
    Ok(())
}
