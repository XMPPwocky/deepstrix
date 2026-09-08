//! gfx11 wave32 WMMA operand-lane probe: are A / B read from lanes 16..31?
//! Run: `cargo test --release -p v4flash-kernels --test iq2_xs_wmma_probe -- --ignored --nocapture`
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::iq2_xs::Iq2XsPairMatvec;

#[test]
#[ignore]
fn wmma_upper_lane_probe() -> eyre::Result<()> {
    install_panic_handler()?;
    let mut igpu = None;
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") { igpu = Some(d); }
    }
    let igpu = igpu.ok_or_else(|| eyre!("no gfx1151"))?;
    igpu.set_current()?;
    let arch = igpu.properties()?.gcn_arch_name;
    let stream = Stream::new(igpu.id)?;
    let m = Iq2XsPairMatvec::for_arch(&arch)?;
    let mut out: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, 9 * 256)?;
    out.fill_zero()?;
    m.launch_upper_lane_probe(&stream, &mut out)?;
    stream.synchronize()?;
    let mut h = vec![0f32; 9 * 256];
    out.copy_to_host(&mut h)?;
    // CPU reference of D0.
    let mut want = vec![0f32; 256];
    for mm in 0..16 { for n in 0..16 { let mut s = 0f32; for k in 0..16 {
        let a = 0.25 * (((mm * 7 + k * 3) % 11) as f32) - 1.0;
        let b = 0.5 * (((n * 5 + k * 2) % 7) as f32) - 1.5;
        s += a * b; } want[mm * 16 + n] = s; } }
    let md = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let d0 = md(&h[..256], &want);
    eprintln!("D0 (all groups valid) vs CPU: max_diff={d0:.3e}");
    let names = ["lo lanes e0-7", "lo lanes e8-15", "hi lanes e0-7", "hi lanes e8-15"];
    for op in 0..2 {
        for g in 0..4 {
            let t = 1 + op * 4 + g;
            let d = md(&h[t * 256..(t + 1) * 256], &h[..256]);
            eprintln!("{} garbage in {:16}: max_diff={d:.3e} -> {}", if op == 0 { "A" } else { "B" }, names[g],
                if d == 0.0 { "NOT READ" } else { "READ" });
        }
    }
    if d0 > 1e-3 { return Err(eyre!("probe reference mismatch {d0}")); }
    Ok(())
}
