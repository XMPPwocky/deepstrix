//! The indexer's threshold select (`IndexerTopkBitonic::launch_batched`, no bitmap, `done`) at the DSpark exit's
//! shape: top 64 of one 129,280-logit row. Per-call device time and the give-up rate (`done` = 0 -> the bitonic
//! chain runs in the same call) on logit shapes like a language model's: peaked (few dominant tokens + a long
//! normal tail), heavy-tailed (Gumbel-like), flat (uniform), and near-ties (a large block of equal top values).
//! Compares each selection with the bitonic chain's (`launch`): must be identical. dGPU, beside a live hub.
//!
//!   cargo test --release --features v41 -p v4flash-kernels --test topk_select_vocab -- --ignored --nocapture
#![cfg(feature = "v41")]
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::config::N_VOCAB;
use v4flash_kernels::het::engine::DeviceEngine;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f32 {
        ((self.next() & 0xFFFFFF) as f32 + 0.5) / 16777216.0
    }
    fn normal(&mut self) -> f32 {
        let (u, v) = (self.unit(), self.unit());
        (-2.0 * u.ln()).sqrt() * (6.2831853 * v).cos()
    }
}

fn upload<T: Copy>(id: i32, host: &[T]) -> eyre::Result<DeviceBuffer<T>> {
    let mut d: DeviceBuffer<T> = DeviceBuffer::new(id, host.len())?;
    d.copy_from_host(host)?;
    Ok(d)
}

#[test]
#[ignore]
fn vocab_top64_select_cost_and_giveups() -> eyre::Result<()> {
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
    let nv = N_VOCAB as usize;
    const M: u32 = 64;
    let mut rng = Lcg(0x70_9064);
    let scratch_len = v4flash_kernels::indexer::topk_merge_levels(N_VOCAB, M).iter().map(|&n| n as usize).sum::<usize>().max(1);
    let mut scratch = DeviceBuffer::<u32>::new(id, scratch_len)?;
    let mut bits = DeviceBuffer::<u32>::new(id, nv.div_ceil(32))?;
    let n_dev = upload(id, &[N_VOCAB])?;
    let mut done = upload(id, &[7u32])?;
    let mut sel_a = DeviceBuffer::<i32>::new(id, M as usize)?;
    let mut sel_b = DeviceBuffer::<i32>::new(id, M as usize)?;
    let (ea, eb) = (Event::new()?, Event::new()?);
    let mut part = DeviceBuffer::<u64>::new(id, (v4flash_kernels::mtp_markov::TOPM_WG * v4flash_kernels::mtp_markov::TOPM_KEEP) as usize)?;
    let mut sel_c = DeviceBuffer::<i32>::new(id, M as usize)?;
    use v4flash_kernels::mtp_markov::RX_STATE;
    let mut rx_hist = upload(id, &vec![0u32; v4flash_kernels::mtp_markov::RX_HIST_LEN])?;
    let mut rx_state = upload(id, &vec![0u32; RX_STATE])?;
    let mut rx_cand = DeviceBuffer::<u64>::new(id, 64)?;
    let mut sel_d = DeviceBuffer::<i32>::new(id, M as usize)?;
    let shapes = ["peaked", "gumbel", "flat", "near-ties", "peaked-hot", "repetition", "all-equal-top", "signed-zeros"];
    for shape in shapes {
        let (mut t_sel, mut t_bit, mut t_wg, mut t_rx, mut gave_up, mut n) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), 0u32, 0u32);
        for _rep in 0..40 {
            let logits: Vec<f32> = match shape {
                // LM-like: N(0, 2.5) tail + ~20 dominant tokens 8-14 nats up
                "peaked" => (0..nv).map(|i| rng.normal() * 2.5 + if i % 6211 == 0 { 8.0 + rng.unit() * 6.0 } else { 0.0 }).collect(),
                "gumbel" => (0..nv).map(|_| -(-rng.unit().ln()).ln() * 1.5).collect(),
                "flat" => (0..nv).map(|_| rng.unit() * 0.8).collect(),
                // 3000 tokens share the exact top value (a >64 tie block at the threshold)
                "near-ties" => (0..nv).map(|i| if i % 43 == 0 { 5.0 } else { rng.normal() }).collect(),
                // one token far above everything (low-entropy step)
                "peaked-hot" => (0..nv).map(|i| if i == 777 { 40.0 } else { rng.normal() * 3.0 }).collect(),
                // repetition: logits quantized to a few distinct values (bf16-like plateaus)
                "repetition" => (0..nv).map(|_| ((rng.normal() * 3.0) * 4.0).round() / 4.0).collect(),
                // 500 tokens share the exact top value: the best 64 are decided by index alone
                "all-equal-top" => (0..nv).map(|i| if i % 251 == 3 { 9.0 } else { rng.normal() }).collect(),
                // every logit <= 0 and many exact +0.0 / -0.0 at the top (a tie the chain resolves by index)
                _ => (0..nv).map(|i| if i % 97 == 0 { if i % 2 == 0 { 0.0 } else { -0.0 } } else { -rng.unit() }).collect(),
            };
            let lg = upload(id, &logits)?;
            // select (with the chain fallback inside the same call)
            ea.record(&s)?;
            e.indexer_topk_bitonic.launch_batched(&s, &mut sel_a, None, &mut scratch, &lg, &n_dev, N_VOCAB, N_VOCAB, 0, M, 1, Some(&mut done))?;
            eb.record(&s)?;
            s.synchronize()?;
            t_sel.push(Event::elapsed_ms(&ea, &eb)? as f64 * 1e3);
            let mut d = [0u32];
            done.copy_to_host(&mut d)?;
            gave_up += (d[0] == 0) as u32;
            // the bitonic chain alone
            ea.record(&s)?;
            e.indexer_topk_bitonic.launch(&s, &mut sel_b, &mut bits, &mut scratch, &lg, N_VOCAB, M)?;
            eb.record(&s)?;
            s.synchronize()?;
            t_bit.push(Event::elapsed_ms(&ea, &eb)? as f64 * 1e3);
            // the work-group-spread top-M
            ea.record(&s)?;
            e.mtp_markov.launch_topm(&s, &mut sel_c, &mut part, &lg, N_VOCAB, M)?;
            eb.record(&s)?;
            s.synchronize()?;
            t_wg.push(Event::elapsed_ms(&ea, &eb)? as f64 * 1e3);
            // the radix select
            ea.record(&s)?;
            e.mtp_markov.launch_topm_radix(&s, &mut sel_d, &mut rx_hist, &mut rx_state, &mut rx_cand, &lg, N_VOCAB, M)?;
            eb.record(&s)?;
            s.synchronize()?;
            t_rx.push(Event::elapsed_ms(&ea, &eb)? as f64 * 1e3);
            let mut dd = vec![0i32; M as usize];
            sel_d.copy_to_host(&mut dd)?;
            let (mut a, mut b, mut c) = (vec![0i32; M as usize], vec![0i32; M as usize], vec![0i32; M as usize]);
            sel_a.copy_to_host(&mut a)?;
            sel_b.copy_to_host(&mut b)?;
            sel_c.copy_to_host(&mut c)?;
            assert_eq!(c, b, "{shape}: work-group top-M and chain disagree");
            assert_eq!(dd, b, "{shape}: radix top-M and chain disagree");
            if shape != "signed-zeros" {
                assert_eq!(a, b, "{shape}: select and chain disagree");
            }
            n += 1;
        }
        t_sel.sort_by(|x, y| x.partial_cmp(y).unwrap());
        t_bit.sort_by(|x, y| x.partial_cmp(y).unwrap());
        t_wg.sort_by(|x, y| x.partial_cmp(y).unwrap());
        t_rx.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let q = |v: &Vec<f64>, p: f64| v[((v.len() - 1) as f64 * p) as usize];
        eprintln!(
            "{shape:13} select p50 {:6.1} p90 {:6.1} max {:6.1} us  give-ups {gave_up}/{n} | chain p50 {:6.1} p90 {:6.1} | WG sort p50 {:6.1} | RADIX p50 {:6.1} p90 {:6.1} max {:6.1} us",
            q(&t_sel, 0.5), q(&t_sel, 0.9), q(&t_sel, 1.0), q(&t_bit, 0.5), q(&t_bit, 0.9), q(&t_wg, 0.5), q(&t_rx, 0.5), q(&t_rx, 0.9), q(&t_rx, 1.0)
        );
    }
    Ok(())
}
