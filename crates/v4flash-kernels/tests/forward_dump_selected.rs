//! Quick test: dump expert_selected from the activation dump for L=30..42, T=0.

use std::path::PathBuf;
use color_eyre::eyre;
use v4flash_kernels::ActivationDump;

#[test]
#[ignore]
fn dump_selected() -> eyre::Result<()> {
    let dump = ActivationDump::open(PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("reference/v4flash-cpu-activations"))?;
    for l in 0..43i32 {
        let bytes = dump.read_bytes(dump.tensor("expert_selected", l, 0).unwrap())?;
        let ids: Vec<i32> = bytes.chunks_exact(4).map(|c| i32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
        let w = dump.read_f32(dump.tensor("expert_weight_out", l, 0).unwrap())?;
        println!("dump L{l} selected={:?} weights={:?}", &ids[..6], &w[..6]);
    }
    Ok(())
}
