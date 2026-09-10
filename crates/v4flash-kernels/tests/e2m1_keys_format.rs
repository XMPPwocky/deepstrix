//! Packed E2M1 indexer-key format: bit-exactness proof against the f16 path
//! it replaces (`docs/E2M1_INDEXER_KEYS_2026-09.md`, step 1 gate).
//!
//! Runs on EVERY HIP device present with KiB-sized buffers; no model.
//!
//! 1. table check: device magnitude table == host.
//! 2. host expand == device expand for every (nibble, e) over the i8 field.
//! 3. constructed rows: for each block exponent e in [-40, 20], rows whose
//!    QAT output is every nibble value at that e (built by the inverse
//!    Hadamard, with a 6 x 2^e anchor per block so the QAT picks e), pushed
//!    through the REAL old chain (indexer_qat -> f16_roundtrip ->
//!    comp_kv_append) and the new chain (indexer_qat -> index_kv_append_e2m1
//!    -> expand): u16-exact, and the packed bytes must contain every nonzero
//!    nibble at every e (coverage of the proof).
//! 4. random-row chain oracle: block magnitudes from below the 6 x 2^-126
//!    floor to 1e30, zero blocks, tiny negatives (the -0.0 case), single and
//!    batched producers, host unpack == device expand.
//!
//! `cargo test -p v4flash-kernels --release --test e2m1_keys_format -- --nocapture`

use std::collections::HashSet;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::index_kv_e2m1::{
    expand_half_bits_host, unpack_row_host, E2M1_KEY_DIM, E2M1_KEY_OFF_EXP, E2M1_KEY_ROW_BYTES, E2M1_MAGNITUDES,
};
use v4flash_kernels::{CompKvAppend, F16Roundtrip, IndexKvE2m1, IndexerQat};

const DIM: u32 = E2M1_KEY_DIM as u32;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

struct Kernels {
    qat: IndexerQat,
    f16rt: F16Roundtrip,
    append: CompKvAppend,
    new: IndexKvE2m1,
}

/// Host Hadamard128 butterfly (same pairwise tree as the kernel).
fn hadamard128(v: &mut [f32]) {
    let mut stride = 1usize;
    while stride < 128 {
        for base in (0..128).step_by(2 * stride) {
            for j in 0..stride {
                let a = v[base + j];
                let b = v[base + j + stride];
                v[base + j] = a + b;
                v[base + j + stride] = a - b;
            }
        }
        stride <<= 1;
    }
}

/// Input row whose QAT output (Hadamard then 1/sqrt(128)) is `target`.
fn inverse_row(target: &[f32]) -> Vec<f32> {
    let mut x = target.to_vec();
    hadamard128(&mut x);
    for v in x.iter_mut() {
        *v *= 0.088_388_347_648_318_45;
    }
    x
}

fn old_chain(k: &Kernels, dev: i32, stream: &Stream, rows: &[f32]) -> eyre::Result<Vec<u16>> {
    let n_rows = rows.len() / E2M1_KEY_DIM;
    let mut x = DeviceBuffer::<f32>::new(dev, rows.len())?;
    x.copy_from_host(rows)?;
    let mut cache = DeviceBuffer::<u16>::new(dev, rows.len())?;
    k.qat.launch(stream, &mut x, n_rows as u32)?;
    k.f16rt.launch(stream, &mut x, rows.len() as u32)?;
    k.append.launch_batched(stream, &mut cache, &x, 0, DIM, n_rows as u32)?;
    stream.synchronize()?;
    let mut out = vec![0u16; rows.len()];
    cache.copy_to_host(&mut out)?;
    Ok(out)
}

/// New chain: returns (expanded f16 rows, packed bytes).
fn new_chain(k: &Kernels, dev: i32, stream: &Stream, rows: &[f32], single: bool) -> eyre::Result<(Vec<u16>, Vec<u8>)> {
    let n_rows = rows.len() / E2M1_KEY_DIM;
    let mut x = DeviceBuffer::<f32>::new(dev, rows.len())?;
    x.copy_from_host(rows)?;
    k.qat.launch(stream, &mut x, n_rows as u32)?;
    let mut packed = DeviceBuffer::<u8>::new(dev, n_rows * E2M1_KEY_ROW_BYTES)?;
    packed.copy_from_host(&vec![0xA5u8; n_rows * E2M1_KEY_ROW_BYTES])?;
    if single {
        for r in 0..n_rows {
            let xr = x.slice_view(r * E2M1_KEY_DIM, E2M1_KEY_DIM);
            k.new.launch_append(stream, &mut packed, &xr, r as u32)?;
        }
    } else {
        // Two batches so n_comp_start != 0 is exercised.
        let split = (n_rows / 2) as u32;
        k.new.launch_append_batched(stream, &mut packed, &x, 0, split, )?;
        let tail = x.slice_view(split as usize * E2M1_KEY_DIM, (n_rows - split as usize) * E2M1_KEY_DIM);
        k.new.launch_append_batched(stream, &mut packed, &tail, split, n_rows as u32 - split)?;
    }
    let mut exp = DeviceBuffer::<u16>::new(dev, rows.len())?;
    k.new.launch_expand(stream, &mut exp, &packed, n_rows as u32)?;
    stream.synchronize()?;
    let mut out = vec![0u16; rows.len()];
    exp.copy_to_host(&mut out)?;
    let mut pb = vec![0u8; n_rows * E2M1_KEY_ROW_BYTES];
    packed.copy_to_host(&mut pb)?;
    Ok((out, pb))
}

fn compare(label: &str, rows: &[f32], old: &[u16], new: &[u16], packed: &[u8]) -> eyre::Result<()> {
    let mut bad = 0usize;
    for i in 0..old.len() {
        if old[i] != new[i] {
            bad += 1;
            if bad <= 8 {
                let r = i / E2M1_KEY_DIM;
                let d = i % E2M1_KEY_DIM;
                let byte = packed[r * E2M1_KEY_ROW_BYTES + d / 2];
                let nib = if d % 2 == 0 { byte & 0xF } else { byte >> 4 };
                let e = packed[r * E2M1_KEY_ROW_BYTES + E2M1_KEY_OFF_EXP + d / 32] as i8;
                eprintln!("  {label}: row {r} dim {d}: in {:e} old {:#06x} new {:#06x} nib {nib:#x} e {e}", rows[i], old[i], new[i]);
            }
        }
    }
    if bad != 0 {
        return Err(eyre!("{label}: {bad} / {} mismatches", old.len()));
    }
    for r in 0..old.len() / E2M1_KEY_DIM {
        let row = &packed[r * E2M1_KEY_ROW_BYTES..(r + 1) * E2M1_KEY_ROW_BYTES];
        if row[68..80].iter().any(|&b| b != 0) {
            return Err(eyre!("{label}: row {r} pad bytes not zero"));
        }
    }
    Ok(())
}

fn table_and_exhaustive(k: &Kernels, dev: i32, stream: &Stream) -> eyre::Result<()> {
    let mut t = DeviceBuffer::<f32>::new(dev, 8)?;
    k.new.launch_table_check(stream, &mut t)?;
    stream.synchronize()?;
    let mut th = vec![0f32; 8];
    t.copy_to_host(&mut th)?;
    for i in 0..8 {
        if th[i].to_bits() != E2M1_MAGNITUDES[i].to_bits() {
            return Err(eyre!("table[{i}] device {} != host {}", th[i], E2M1_MAGNITUDES[i]));
        }
    }
    println!("  table check: 8/8");
    let (e_lo, n_e) = (-128i32, 256u32);
    let n = n_e as usize * 16;
    let mut out = DeviceBuffer::<u16>::new(dev, n)?;
    k.new.launch_expand_exhaustive(stream, &mut out, e_lo, n_e)?;
    stream.synchronize()?;
    let mut oh = vec![0u16; n];
    out.copy_to_host(&mut oh)?;
    let mut bad = 0;
    for i in 0..n {
        let hb = expand_half_bits_host((i & 15) as u8, (e_lo + (i >> 4) as i32) as i8);
        if hb != oh[i] {
            bad += 1;
            if bad <= 8 {
                eprintln!("  HOST nib {:#x} e {}: device {:#06x} host {hb:#06x}", i & 15, e_lo + (i >> 4) as i32, oh[i]);
            }
        }
    }
    if bad != 0 {
        return Err(eyre!("host reference vs device expand: {bad} mismatches"));
    }
    println!("  host reference == device expand for all {n} (nibble, e) over the i8 field");

    // The fast byte-permute path (what the score kernels use) vs the general
    // path: every nibble value at every one of the 8 positions (others zero,
    // and others all-ones), plus random words, at every e in the i8 field.
    let mut words: Vec<u32> = Vec::new();
    for pos in 0..8 {
        for v in 0..16u32 {
            words.push(v << (4 * pos));
            words.push((v << (4 * pos)) | (0xFFFF_FFFFu32 & !(0xFu32 << (4 * pos))));
        }
    }
    let mut seed = 0x1357_9BDF_2468_ACE0u64;
    for _ in 0..768 {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        words.push(seed as u32);
    }
    let n_words = words.len();
    let mut dw = DeviceBuffer::<u32>::new(dev, n_words)?;
    dw.copy_from_host(&words)?;
    let n_out = 256 * n_words * 4;
    let mut fast = DeviceBuffer::<u32>::new(dev, n_out)?;
    let mut refd = DeviceBuffer::<u32>::new(dev, n_out)?;
    k.new.launch_expand8_check(stream, &mut fast, &mut refd, &dw, -128, 256)?;
    stream.synchronize()?;
    let mut fh = vec![0u32; n_out];
    let mut rh = vec![0u32; n_out];
    fast.copy_to_host(&mut fh)?;
    refd.copy_to_host(&mut rh)?;
    let mut bad = 0;
    for i in 0..n_out {
        if fh[i] != rh[i] {
            bad += 1;
            if bad <= 8 {
                let g = i / 4;
                eprintln!("  FAST word {:#010x} e {} lane {}: fast {:#010x} ref {:#010x}", words[g % n_words], -128 + (g / n_words) as i32, i % 4, fh[i], rh[i]);
            }
        }
    }
    if bad != 0 {
        return Err(eyre!("fast expand8 vs general: {bad} mismatches"));
    }
    println!("  fast byte-permute expand8 == general path for {n_words} words x 256 exponents ({} f16)", n_out * 2);
    Ok(())
}

fn constructed(k: &Kernels, dev: i32, stream: &Stream, rng: &mut Rng) -> eyre::Result<()> {
    let (e_lo, e_hi) = (-40i32, 20i32);
    let mut rows: Vec<f32> = Vec::new();
    for e in e_lo..=e_hi {
        // Rows per e: variant 0/1 spread the block exponents (e+b / e-b) with
        // a 5.9 anchor (block max snaps to 6: e' = e0); variants 2/3 use a
        // 3.0 anchor (max 3: the QAT re-derivation lands on e0 - 1 with
        // doubled codes) and a 3.5 anchor (the tie, rounds to 4: e' = e0).
        for variant in 0..4 {
            let mut target = vec![0f32; E2M1_KEY_DIM];
            for blk in 0..4 {
                let eb = if variant % 2 == 0 { e + blk as i32 } else { e - blk as i32 };
                let scale = 2f32.powi(eb);
                for j in 0..32 {
                    let nib = if j < 16 { j as u8 } else { (rng.next() % 16) as u8 };
                    let m = E2M1_MAGNITUDES[(nib & 7) as usize] * scale;
                    target[blk * 32 + j] = if nib & 8 != 0 { -m } else { m };
                }
                // Anchor: block max in (3, 6] * 2^eb makes the QAT exponent eb. 5.9
                // (snaps to code 6) rather than exactly 6.0, so a one-ulp
                // reconstruction error in the device Hadamard cannot push the
                // max past 6 and bump the exponent (it did on gfx1151).
                target[blk * 32 + 16] = match variant { 2 => 3.0, 3 => 3.5, _ => 5.9 } * scale;
                if variant >= 2 {
                    // Keep the anchor the block max: cap the other targets at 3.
                    for j in 0..32 {
                        if j != 16 && target[blk * 32 + j].abs() > 3.0 * scale {
                            target[blk * 32 + j] = target[blk * 32 + j].signum() * 3.0 * scale;
                        }
                    }
                }
            }
            rows.extend(inverse_row(&target));
        }
    }
    let old = old_chain(k, dev, stream, &rows)?;
    let (new, packed) = new_chain(k, dev, stream, &rows, false)?;
    compare("constructed", &rows, &old, &new, &packed)?;
    // Coverage: every nonzero nibble at every e in [e_lo, e_hi] must occur.
    let mut seen: HashSet<(u8, i8)> = HashSet::new();
    let (mut pz, mut nz) = (0usize, 0usize);
    for r in 0..rows.len() / E2M1_KEY_DIM {
        let row = &packed[r * E2M1_KEY_ROW_BYTES..(r + 1) * E2M1_KEY_ROW_BYTES];
        for i in 0..E2M1_KEY_DIM {
            let byte = row[i / 2];
            let nib = if i % 2 == 0 { byte & 0xF } else { byte >> 4 };
            let e = row[E2M1_KEY_OFF_EXP + i / 32] as i8;
            match nib {
                0 => pz += 1,
                8 => nz += 1,
                _ => {
                    seen.insert((nib, e));
                }
            }
        }
    }
    let mut missing = 0;
    for e in e_lo..=e_hi {
        for nib in (1u8..16).filter(|&n| n != 8) {
            if !seen.contains(&(nib, e as i8)) {
                missing += 1;
                if missing <= 6 {
                    eprintln!("  coverage gap: nib {nib:#x} e {e}");
                }
            }
        }
    }
    if missing != 0 {
        return Err(eyre!("constructed rows: {missing} (nibble, e) pairs never produced by the QAT"));
    }
    println!(
        "  constructed: {} rows, every nonzero nibble at every e in [{e_lo}, {e_hi}] through the real chain, bit-identical (+0: {pz}, -0: {nz})",
        rows.len() / E2M1_KEY_DIM
    );
    Ok(())
}

fn random_rows(k: &Kernels, dev: i32, stream: &Stream, rng: &mut Rng) -> eyre::Result<()> {
    let n_rows = 1500usize;
    let mut rows = vec![0f32; n_rows * E2M1_KEY_DIM];
    for r in 0..n_rows {
        let row = &mut rows[r * E2M1_KEY_DIM..(r + 1) * E2M1_KEY_DIM];
        let kind = rng.next() % 8;
        let mag: f32 = match kind {
            0 => 0.0,
            1 => 1e-40,  // below the 6 x 2^-126 amax floor
            2 => 1e-36,
            3 => 1e30,
            4 => 2f32.powi((rng.next() % 40) as i32 - 20),
            _ => 10f32.powf(rng.unit() * 4.0 - 2.0),
        };
        for v in row.iter_mut() {
            let u = rng.unit() * 2.0 - 1.0;
            *v = match rng.next() % 10 {
                0 => 0.0,
                1 => -0.0,
                2 => u * mag * 1e-4, // snaps to code 0 after the Hadamard, either sign
                _ => u * mag,
            };
        }
    }
    let old = old_chain(k, dev, stream, &rows)?;
    let (new_b, packed_b) = new_chain(k, dev, stream, &rows, false)?;
    compare("random/batched", &rows, &old, &new_b, &packed_b)?;
    let (new_s, packed_s) = new_chain(k, dev, stream, &rows[..64 * E2M1_KEY_DIM], true)?;
    compare("random/single", &rows[..64 * E2M1_KEY_DIM], &old[..64 * E2M1_KEY_DIM], &new_s, &packed_s)?;
    if packed_s[..] != packed_b[..64 * E2M1_KEY_ROW_BYTES] {
        return Err(eyre!("single-row producer bytes differ from batched"));
    }
    let mut hu = vec![0u16; E2M1_KEY_DIM];
    let mut nz = 0usize;
    for r in 0..n_rows {
        unpack_row_host(&packed_b[r * E2M1_KEY_ROW_BYTES..], &mut hu);
        if hu[..] != new_b[r * E2M1_KEY_DIM..(r + 1) * E2M1_KEY_DIM] {
            return Err(eyre!("host unpack differs from device expand at row {r}"));
        }
        nz += old[r * E2M1_KEY_DIM..(r + 1) * E2M1_KEY_DIM].iter().filter(|&&w| w == 0x8000).count();
    }
    println!("  random rows: {n_rows} x 128 u16-exact (batched + single producers), host unpack == device expand, {nz} stored -0.0 words");
    Ok(())
}

#[test]
fn e2m1_keys_format_bit_exact() -> eyre::Result<()> {
    install_panic_handler()?;
    let devices = Device::all()?;
    if devices.is_empty() {
        return Err(eyre!("no HIP devices"));
    }
    let mut rng = Rng(0x5EED_5EED_1234_ABCD);
    for d in &devices {
        d.set_current()?;
        let arch = d.properties()?.gcn_arch_name.clone();
        println!("device {} ({arch})", d.id);
        let stream = Stream::new(d.id)?;
        let k = Kernels {
            qat: IndexerQat::for_arch(&arch)?,
            f16rt: F16Roundtrip::for_arch(&arch)?,
            append: CompKvAppend::for_arch(&arch)?,
            new: IndexKvE2m1::for_arch(&arch)?,
        };
        table_and_exhaustive(&k, d.id, &stream)?;
        constructed(&k, d.id, &stream, &mut rng)?;
        for _ in 0..2 {
            random_rows(&k, d.id, &stream, &mut rng)?;
        }
    }
    Ok(())
}
