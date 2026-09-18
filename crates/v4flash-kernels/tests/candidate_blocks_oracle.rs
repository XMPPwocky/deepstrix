//! ARCH_SPEC §1.5 candidate pool: the GPU mask must match the CPU transcription
//! of `inference/model.py::select_candidate_blocks` exactly.
//!
//! A wrong mask here is silent: level-two index sources (24/28/32/36) would pick
//! their top-512 from the wrong candidate set, which changes which positions
//! attention sees. Only visible as worse tokens at long context.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{Device, DeviceBuffer, Stream};
use v4flash_kernels::candidate_blocks::{select_candidate_blocks_cpu, CandidateBlocks};
use v4flash_kernels::config::CANDIDATE_BLOCK_SIZE;

fn pick_dgpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        let a = d.properties()?.gcn_arch_name;
        if a.starts_with("gfx1201") || a.starts_with("gfx1151") {
            return Ok(d);
        }
    }
    Err(eyre!("no supported GPU"))
}

struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u32 << 31) as f32) * 20.0 - 10.0
    }
}

fn case(dev: &Device, stream: &Stream, cb: &CandidateBlocks, n_per_host: &[u32], seed: u64) -> eyre::Result<()> {
    let batch = n_per_host.len() as u32;
    let n_max = *n_per_host.iter().max().unwrap();
    let stride = n_max;
    let nb_stride = CandidateBlocks::n_blocks(n_max);

    let mut rng = Lcg(seed);
    // Distinct scores: ties are broken arbitrarily by torch.topk too, so an
    // oracle comparison is only well defined without them.
    let mut scores = vec![f32::NEG_INFINITY; (batch * stride) as usize];
    for (b, &n) in n_per_host.iter().enumerate() {
        for p in 0..n as usize {
            scores[b * stride as usize + p] = rng.next_f32();
        }
    }

    let mut want: Vec<f32> = scores.clone();
    for (b, &n) in n_per_host.iter().enumerate() {
        let row = &scores[b * stride as usize..b * stride as usize + n as usize];
        let keep = select_candidate_blocks_cpu(row, n as usize);
        for (p, &k) in keep.iter().enumerate() {
            if !k {
                want[b * stride as usize + p] = f32::NEG_INFINITY;
            }
        }
    }

    let mut d_scores = DeviceBuffer::<f32>::new(dev.id, scores.len())?;
    d_scores.copy_from_host(&scores)?;
    let mut d_block = DeviceBuffer::<f32>::new(dev.id, (batch * nb_stride) as usize)?;
    let mut d_thr = DeviceBuffer::<u32>::new(dev.id, batch as usize)?;
    let mut d_n = DeviceBuffer::<u32>::new(dev.id, batch as usize)?;
    d_n.copy_from_host(n_per_host)?;

    cb.launch_apply(stream, &mut d_scores, &mut d_block, &mut d_thr, &d_n, stride, nb_stride, n_max, batch)?;
    stream.synchronize()?;

    let mut got = vec![0f32; scores.len()];
    d_scores.copy_to_host(&mut got)?;

    for (b, &n) in n_per_host.iter().enumerate() {
        let mut kept_gpu = 0usize;
        let mut kept_cpu = 0usize;
        for p in 0..n as usize {
            let i = b * stride as usize + p;
            let g = got[i].is_finite();
            let w = want[i].is_finite();
            if g { kept_gpu += 1; }
            if w { kept_cpu += 1; }
            if g != w {
                return Err(eyre!(
                    "row {b} (n={n}) pos {p} block {}: gpu kept={g} cpu kept={w} (score {})",
                    p / CANDIDATE_BLOCK_SIZE as usize, scores[i]
                ));
            }
            if w && got[i] != want[i] {
                return Err(eyre!("row {b} pos {p}: kept but value changed {} -> {}", want[i], got[i]));
            }
        }
        println!("  row {b}: n={n} kept {kept_gpu} positions (cpu {kept_cpu})");
        assert_eq!(kept_gpu, kept_cpu);
    }
    Ok(())
}

#[test]
#[ignore = "needs a HIP device"]
fn candidate_blocks_match_cpu() -> eyre::Result<()> {
    let dev = pick_dgpu()?;
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    let stream = Stream::new(dev.id)?;
    let cb = CandidateBlocks::for_arch(&arch)?;
    println!("candidate_blocks oracle on {arch}");

    // Vacuous regime: n_comp <= 2048*8, every block kept, mask is a no-op.
    println!("vacuous (<=16384):");
    case(&dev, &stream, &cb, &[4096, 16384], 0x1111)?;
    // Just past the cut, where the mask starts to bite.
    println!("just past the cut:");
    case(&dev, &stream, &cb, &[16385, 20000], 0x2222)?;
    // Decode shape (batch 1) at long context — the regime that matters.
    println!("decode @ long ctx:");
    case(&dev, &stream, &cb, &[65536], 0x3333)?;
    case(&dev, &stream, &cb, &[307200], 0x4444)?;
    // Ragged batch: rows of different reachable lengths, as in prefill.
    println!("ragged prefill batch:");
    case(&dev, &stream, &cb, &[17000, 65536, 3000, 120000], 0x5555)?;
    Ok(())
}
