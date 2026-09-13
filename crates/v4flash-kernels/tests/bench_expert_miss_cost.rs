//! **Residency prototype, phase A** — what does ONE cold expert miss actually
//! cost, end to end, on the decode critical path?
//!
//! `docs/v41/COLD_EXPERT_CACHING.md` predicts ~1.4 SSD reads/token for an
//! all-native V4.1 at 75.7% residency, and converts that to decode time using
//! an ASSUMED 2.7 ms per miss (18.8 MB / 7 GB/s). That assumption is the
//! weakest link in the whole all-native plan: it ignores page-fault cost, the
//! host->device copy, and any serialisation against the decode stream. If a
//! real miss costs 8 ms instead of 2.7, all-native decode drops from ~33 tok/s
//! to ~20 and the third box comes back on the table.
//!
//! This measures the miss path for the *explicit LRU* design (read from the
//! mmap'd/NVMe-resident weight file into pinned host memory, then DMA to the
//! device). Phase B — whether the OS page cache can serve GTT-mapped expert
//! reads directly, skipping the copy — is a separate experiment.
//!
//! Run:
//!   MISS_FILE=/persist/lumi/models/... MISS_MB=18.8 \
//!   HIP_VISIBLE_DEVICES=0,1 nix develop -c cargo test --release \
//!     -p v4flash-kernels --test bench_expert_miss_cost -- --ignored --nocapture

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::time::Instant;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{
    install_panic_handler, Device, DeviceBuffer, PinnedBuffer, Stream,
    HIP_HOST_MALLOC_NON_COHERENT,
};

// Declared directly rather than pulling in `libc` — one syscall does not justify
// a new dependency (repo rule: deps need sign-off).
extern "C" {
    fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
}
const POSIX_FADV_DONTNEED: i32 = 4;

/// Tell the kernel to forget these pages so the next read is a real disk read.
fn evict(file: &File, off: u64, len: usize) {
    unsafe {
        posix_fadvise(file.as_raw_fd(), off as i64, len as i64, POSIX_FADV_DONTNEED);
    }
}

fn pick_igpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(d);
        }
    }
    Err(eyre!("no iGPU"))
}

#[test]
#[ignore]
fn bench_expert_miss_cost() -> eyre::Result<()> {
    install_panic_handler()?;
    let path = std::env::var("MISS_FILE").unwrap_or_else(|_| {
        "/persist/lumi/models/dsv4f-exp-iq3-xxs/UD-IQ3_XXS/\
         DeepSeek-V4-Flash-Vision-Exp-UD-IQ3_XXS-00001-of-00004.gguf"
            .to_string()
    });
    let n_iters: usize = std::env::var("MISS_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    // Expert sizes worth testing: 8.69 MB = V4-Flash IQ3_XXS expert,
    // 18.8 MB = V4.1 native MXFP4 expert (the number the plan assumes).
    let sizes_mb: Vec<f64> = std::env::var("MISS_MB")
        .unwrap_or_else(|_| "8.69,18.8".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    let file = File::open(&path)?;
    let flen = file.metadata()?.len();
    eprintln!("miss-cost bench: {path}\n  file {:.1} GiB, {n_iters} iters/size", flen as f64 / 2f64.powi(30));

    let igpu = pick_igpu()?;
    let stream = Stream::new(igpu.id)?;

    println!(
        "\n{:>8} {:>10} {:>12} {:>12} {:>12} {:>12}",
        "size MB", "phase", "read ms", "H2D ms", "total ms", "eff GB/s"
    );
    for &mb in &sizes_mb {
        let bytes = (mb * 1e6) as usize;
        let mut host: PinnedBuffer<u8> = PinnedBuffer::new(bytes)?;
        let mut dev: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, bytes)?;

        for phase in ["cold", "warm"] {
            let mut t_read = 0f64;
            let mut t_copy = 0f64;
            for i in 0..n_iters {
                // Random-ish offset, page aligned, inside the file.
                let span = flen.saturating_sub(bytes as u64 + 4096);
                let off = ((i as u64).wrapping_mul(0x9E3779B97F4A7C15) % span.max(1)) & !4095u64;
                if phase == "cold" {
                    evict(&file, off, bytes);
                }
                let t0 = Instant::now();
                file.read_exact_at(host.as_mut_slice(), off)?;
                t_read += t0.elapsed().as_secs_f64();

                let t1 = Instant::now();
                dev.copy_from_host_async(host.as_slice(), &stream)?;
                stream.synchronize()?;
                t_copy += t1.elapsed().as_secs_f64();
            }
            let n = n_iters as f64;
            let rd = t_read / n * 1e3;
            let cp = t_copy / n * 1e3;
            let tot = rd + cp;
            println!(
                "{:>8.2} {:>10} {:>12.3} {:>12.3} {:>12.3} {:>12.2}",
                mb,
                phase,
                rd,
                cp,
                tot,
                (bytes as f64 / 1e9) / (tot / 1e3)
            );
        }
    }
    // ---- parallel reader: is 1 GB/s the device, or just one synchronous
    // reader with no queue depth? A real miss path would issue the extent as
    // N concurrent chunks (io_uring / thread pool) and madvise ahead.
    println!("\n{:>8} {:>10} {:>12} {:>12}", "size MB", "threads", "cold ms", "eff GB/s");
    for &mb in &sizes_mb {
        let bytes = (mb * 1e6) as usize;
        for threads in [1usize, 4, 16, 32, 64, 128] {
            let chunk = bytes.div_ceil(threads);
            let mut total = 0f64;
            let reps = 16;
            for i in 0..reps {
                let span = flen.saturating_sub(bytes as u64 + 4096);
                let off = ((i as u64).wrapping_mul(0x9E3779B97F4A7C15) % span.max(1)) & !4095u64;
                evict(&file, off, bytes);
                let t0 = Instant::now();
                std::thread::scope(|sc| {
                    for t in 0..threads {
                        let f = &file;
                        let lo = t * chunk;
                        let hi = ((t + 1) * chunk).min(bytes);
                        if lo >= hi {
                            continue;
                        }
                        sc.spawn(move || {
                            let mut buf = vec![0u8; hi - lo];
                            let _ = f.read_exact_at(&mut buf, off + lo as u64);
                            std::hint::black_box(&buf[0]);
                        });
                    }
                });
                total += t0.elapsed().as_secs_f64();
            }
            let ms = total / reps as f64 * 1e3;
            println!(
                "{:>8.2} {:>10} {:>12.3} {:>12.2}",
                mb, threads, ms, (bytes as f64 / 1e9) / (ms / 1e3)
            );
        }
    }

    println!(
        "\nInterpretation: the plan assumes 2.7 ms per miss at 18.8 MB. Compare the\n\
         COLD total. Decode impact = misses/token x total ms; at the measured\n\
         1.4 misses/token, every extra 1 ms per miss costs ~1.4 ms/token\n\
         (~1.5 tok/s at a ~30 ms/token baseline)."
    );
    Ok(())
}


/// **Residency prototype, phase B** — can we delete the host->device copy?
///
/// On Strix the iGPU's "device memory" IS system memory (GTT). The phase-A
/// miss path pays ~2.6 ms copying an expert from a host buffer into a device
/// buffer, i.e. moving bytes from LPDDR5X to LPDDR5X for no reason. If the GPU
/// can read `hipHostMalloc` memory at comparable bandwidth, we `pread()`
/// straight into GPU-visible pages and skip the copy — the GTT pages become
/// the cache.
///
/// The catch is coherence granularity: default (coherent / fine-grained) host
/// memory is typically uncached on the GPU, while non-coherent
/// (coarse-grained) may be cached. This measures both.
///
/// XNACK is OFF on this hardware (`rocminfo`: "XNACK enabled: NO", and
/// HSA_XNACK=1 does not change it), so letting kernels fault on an mmap'd file
/// is not an option; explicit staging is the only route.
#[test]
#[ignore]
fn bench_gtt_as_page_cache() -> eyre::Result<()> {
    install_panic_handler()?;
    let mb: f64 = std::env::var("MISS_MB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(18.8);
    let iters: usize = std::env::var("MISS_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let bytes = (mb * 1e6) as usize;

    let igpu = pick_igpu()?;
    let stream = Stream::new(igpu.id)?;
    let mut dst: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, bytes)?;

    let dev_src: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, bytes)?;
    let pin_coh: PinnedBuffer<u8> = PinnedBuffer::new(bytes)?;
    let pin_nc: PinnedBuffer<u8> =
        PinnedBuffer::new_with_flags(bytes, HIP_HOST_MALLOC_NON_COHERENT)?;

    // SAFETY: both pointers come from hipHostMalloc and outlive the views.
    let view_coh = unsafe { DeviceBuffer::<u8>::from_raw_parts(pin_coh.device_ptr(), bytes, igpu.id) };
    let view_nc = unsafe { DeviceBuffer::<u8>::from_raw_parts(pin_nc.device_ptr(), bytes, igpu.id) };

    println!("\nGPU reads {mb:.1} MB from each source ({iters} iters):");
    println!("{:>28} {:>12} {:>12}", "source", "ms", "GB/s");
    for (name, src) in [
        ("device memory (baseline)", &dev_src),
        ("host pinned, coherent", &view_coh),
        ("host pinned, NON-coherent", &view_nc),
    ] {
        // warm
        for _ in 0..4 {
            dst.copy_from_buffer_async(src, &stream)?;
        }
        stream.synchronize()?;
        let t0 = Instant::now();
        for _ in 0..iters {
            dst.copy_from_buffer_async(src, &stream)?;
        }
        stream.synchronize()?;
        let ms = t0.elapsed().as_secs_f64() / iters as f64 * 1e3;
        println!("{:>28} {:>12.3} {:>12.2}", name, ms, (bytes as f64 / 1e9) / (ms / 1e3));
    }
    println!(
        "\nIf non-coherent host memory is within ~2x of device memory, the phase-A\n\
         H2D copy (~2.6 ms, 41% of a cold miss) can be deleted: pread straight\n\
         into GPU-visible pages and let the iGPU read them in place."
    );
    Ok(())
}


/// **Phase B, take 2** — can the CPU `pread()` STRAIGHT INTO a device buffer?
///
/// Take 1 showed `hipHostMalloc` memory is only readable by the iGPU at
/// ~7.2 GB/s (vs device memory), and the coherence flag makes no difference —
/// so the 2.6 ms "H2D copy" is not overhead, it is that read rate. The copy
/// cannot be flagged away.
///
/// But on an APU, `hipMalloc` memory may itself be CPU-addressable (unified
/// memory / large BAR). If so, `pread()` can write directly into it: the
/// kernel's copy_to_user lands in GTT and the separate staging step disappears
/// entirely — one copy instead of two, and the GPU then reads at full device
/// bandwidth.
///
/// If the pointer is not CPU-mappable this will fault; that is a definitive
/// answer too, and it is contained in the test process.
#[test]
#[ignore]
fn bench_pread_into_device() -> eyre::Result<()> {
    install_panic_handler()?;
    let path = std::env::var("MISS_FILE").unwrap_or_else(|_| {
        "/persist/lumi/models/dsv4.1f-full/model-00009-of-00048.safetensors".to_string()
    });
    let mb: f64 = std::env::var("MISS_MB").ok().and_then(|s| s.parse().ok()).unwrap_or(18.8);
    let iters: usize = std::env::var("MISS_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
    let bytes = (mb * 1e6) as usize;

    let file = File::open(&path)?;
    let flen = file.metadata()?.len();
    let igpu = pick_igpu()?;
    let dev: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, bytes)?;

    // Probe CPU addressability with a single byte before doing anything big.
    let ptr = dev.raw() as *mut u8;
    println!("device ptr = {ptr:?}; probing CPU write access...");
    let probe = std::panic::catch_unwind(|| unsafe {
        std::ptr::write_volatile(ptr, 0xA5u8);
        std::ptr::read_volatile(ptr)
    });
    match probe {
        Ok(v) if v == 0xA5 => println!("  CPU CAN write device memory (readback 0x{v:02X})"),
        Ok(v) => {
            println!("  wrote 0xA5, read back 0x{v:02X} — not coherent; aborting");
            return Ok(());
        }
        Err(_) => {
            println!("  CPU CANNOT address device memory — pread-into-device is out");
            return Ok(());
        }
    }

    let slice = unsafe { std::slice::from_raw_parts_mut(ptr, bytes) };
    let mut total = 0f64;
    for i in 0..iters {
        let span = flen.saturating_sub(bytes as u64 + 4096);
        let off = ((i as u64).wrapping_mul(0x9E3779B97F4A7C15) % span.max(1)) & !4095u64;
        evict(&file, off, bytes);
        let t0 = Instant::now();
        file.read_exact_at(slice, off)?;
        total += t0.elapsed().as_secs_f64();
    }
    let ms = total / iters as f64 * 1e3;
    println!(
        "\npread COLD straight into device memory: {ms:.3} ms  ({:.2} GB/s)",
        (bytes as f64 / 1e9) / (ms / 1e3)
    );
    println!(
        "Compare phase A: cold read into host ~3.7-7.6 ms PLUS a 2.6 ms copy.\n\
         If this lands near the read-only number, the copy is deleted."
    );
    Ok(())
}


/// Does `pread()`-into-device-memory scale with reader threads the way
/// pread-into-host does? Single-threaded it is 2.34 GB/s vs host's 2.73 —
/// a ~14% penalty for the CPU writing into GTT. If it scales like the host
/// path (2.73 -> 5.11 GB/s at 128 threads) it would BEAT the two-step
/// "parallel read + 2.6 ms copy" and collapse the miss path to one copy.
#[test]
#[ignore]
fn bench_pread_into_device_threads() -> eyre::Result<()> {
    install_panic_handler()?;
    let path = std::env::var("MISS_FILE").expect("set MISS_FILE");
    let mb: f64 = std::env::var("MISS_MB").ok().and_then(|s| s.parse().ok()).unwrap_or(18.8);
    let bytes = (mb * 1e6) as usize;
    let file = File::open(&path)?;
    let flen = file.metadata()?.len();
    let igpu = pick_igpu()?;
    let dev: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, bytes)?;
    let ptr = dev.raw() as *mut u8;

    println!("\n{:>10} {:>12} {:>12}", "threads", "cold ms", "GB/s");
    for threads in [1usize, 4, 16, 64, 128] {
        let chunk = bytes.div_ceil(threads);
        let mut tot = 0f64;
        let reps = 8;
        for i in 0..reps {
            let span = flen.saturating_sub(bytes as u64 + 4096);
            let off = ((i as u64).wrapping_mul(0x9E3779B97F4A7C15) % span.max(1)) & !4095u64;
            evict(&file, off, bytes);
            let t0 = Instant::now();
            std::thread::scope(|sc| {
                for t in 0..threads {
                    let f = &file;
                    let lo = t * chunk;
                    let hi = ((t + 1) * chunk).min(bytes);
                    if lo >= hi { continue; }
                    let p = unsafe { ptr.add(lo) } as usize;
                    sc.spawn(move || {
                        let sl = unsafe { std::slice::from_raw_parts_mut(p as *mut u8, hi - lo) };
                        let _ = f.read_exact_at(sl, off + lo as u64);
                    });
                }
            });
            tot += t0.elapsed().as_secs_f64();
        }
        let ms = tot / reps as f64 * 1e3;
        println!("{:>10} {:>12.3} {:>12.2}", threads, ms, (bytes as f64 / 1e9) / (ms / 1e3));
    }
    println!("\nBeat = under 6.28 ms (parallel host read 3.68 + copy 2.6).");
    Ok(())
}


/// Models the ACTUAL iGPU expert-load loop, which is now 28.6 s of a 47.5 s
/// model load (60%): ~241 reads of `bpe` bytes at strided offsets within one
/// stacked tensor, currently serial into a single reused staging buffer.
///
/// Measures serial vs T threads with **one reused buffer per thread** — NOT a
/// fresh allocation per extent, which is what made the earlier "mode 2" attempt
/// 2.3x slower. Predicts the gain before touching the loader again.
#[test]
#[ignore]
fn bench_expert_loop_shape() -> eyre::Result<()> {
    install_panic_handler()?;
    let path = std::env::var("MISS_FILE").expect("set MISS_FILE");
    // One expert of one tensor: ~2.9 MB for V4-Flash UD-IQ3_XXS.
    let bpe: usize = std::env::var("BPE_MB")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(2.9) as usize * 1_000_000;
    let n_extents: usize = std::env::var("N_EXTENTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(241);
    let file = File::open(&path)?;
    let flen = file.metadata()?.len();
    let span = flen.saturating_sub((n_extents * bpe) as u64 + 4096);
    let base = (span / 2) & !4095u64;
    let total = (n_extents * bpe) as f64;

    println!(
        "\nexpert-loop shape: {n_extents} x {:.2} MB extents = {:.2} GiB",
        bpe as f64 / 1e6,
        total / 2f64.powi(30)
    );
    println!("{:>10} {:>12} {:>12}", "threads", "s", "GB/s");
    for threads in [1usize, 2, 4, 8, 16, 32] {
        // cold: forget the whole region first
        evict(&file, base, n_extents * bpe);
        let t0 = Instant::now();
        if threads == 1 {
            let mut buf = vec![0u8; bpe];
            for i in 0..n_extents {
                file.read_exact_at(&mut buf, base + (i * bpe) as u64)?;
                std::hint::black_box(&buf[0]);
            }
        } else {
            let per = n_extents.div_ceil(threads);
            std::thread::scope(|sc| {
                for t in 0..threads {
                    let f = &file;
                    let lo = t * per;
                    let hi = ((t + 1) * per).min(n_extents);
                    if lo >= hi {
                        continue;
                    }
                    sc.spawn(move || {
                        // ONE reused buffer per thread.
                        let mut buf = vec![0u8; bpe];
                        for i in lo..hi {
                            let _ = f.read_exact_at(&mut buf, base + (i * bpe) as u64);
                            std::hint::black_box(&buf[0]);
                        }
                    });
                }
            });
        }
        let s = t0.elapsed().as_secs_f64();
        println!("{:>10} {:>12.3} {:>12.2}", threads, s, total / 1e9 / s);
    }
    println!("\nCurrent loader does the `threads=1` row. Scale the 28.6 s by the ratio.");
    Ok(())
}
