//! Compile every `kernels/*.hip` source to a per-arch standalone code object
//! (hsaco) for each target gfx arch. Emits
//! `cargo:rustc-env=KERNEL_<NAME>_<ARCH>=<path>` so the lib can
//! `include_bytes!(env!("KERNEL_<NAME>_<ARCH>"))`.
//!
//! Mirrors `crates/phase0/build.rs` so the convention is uniform across the
//! workspace. Targets come from `$DEEPSTRIX_GFX_TARGETS` (space-separated);
//! defaults to "gfx1201 gfx1151" if unset.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let kernels_dir = manifest_dir.join("kernels");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", kernels_dir.display());
    println!("cargo:rerun-if-env-changed=HIPCC");
    println!("cargo:rerun-if-env-changed=DEEPSTRIX_GFX_TARGETS");
    println!("cargo:rerun-if-env-changed=DEEPSTRIX_KERNEL_CFLAGS");

    export_kw_max_chunk();

    if !kernels_dir.exists() {
        return;
    }

    let hipcc = env::var("HIPCC").unwrap_or_else(|_| "hipcc".to_string());
    let targets_env =
        env::var("DEEPSTRIX_GFX_TARGETS").unwrap_or_else(|_| "gfx1201 gfx1151".to_string());
    let targets: Vec<&str> = targets_env.split_whitespace().collect();

    for entry in fs::read_dir(&kernels_dir).expect("read kernels dir") {
        let entry = entry.expect("dirent");
        let path = entry.path();
        if path.extension().map(|e| e == "hip").unwrap_or(false) {
            for arch in &targets {
                compile(&hipcc, &path, arch, &out_dir);
            }
        }
    }
}

/// Mirror the kwide `*_KW_MAX_CHUNK` bounds into `rustc-env` so the Rust
/// launch guards can never disagree with the kernel that was actually
/// compiled.
///
/// These macros size the kwide kernels' LDS staging (`s_q8v`, `s_yd`,
/// `s_member_packed`) and register accumulators. They are `#ifndef`-guarded
/// in the .hip so a `DEEPSTRIX_KERNEL_CFLAGS=-DIQ2S_KW_MAX_CHUNK=16u`
/// occupancy probe takes effect — which means a hardcoded `32` on the Rust
/// side would then let production's `chunk_size = 32` through the guard and
/// silently overrun LDS. Parsing the same flag string here keeps the two in
/// lockstep by construction.
fn export_kw_max_chunk() {
    let extra = env::var("DEEPSTRIX_KERNEL_CFLAGS").unwrap_or_default();
    for macro_name in ["IQ2S_KW_MAX_CHUNK", "IQ3S_KW_MAX_CHUNK"] {
        let prefix = format!("-D{macro_name}=");
        // Last -D wins, matching the preprocessor.
        let val = extra
            .split_whitespace()
            .filter_map(|tok| tok.strip_prefix(prefix.as_str()))
            .last()
            .map(|v| v.trim_end_matches(['u', 'U']).to_string())
            .unwrap_or_else(|| "32".to_string());
        assert!(
            !val.is_empty() && val.bytes().all(|b| b.is_ascii_digit()),
            "{macro_name}: expected a plain decimal (optionally `u`-suffixed), got {val:?}"
        );
        println!("cargo:rustc-env=DEEPSTRIX_{macro_name}={val}");
    }
}

fn compile(hipcc: &str, src: &Path, arch: &str, out_dir: &Path) {
    let stem = src
        .file_stem()
        .expect("stem")
        .to_str()
        .expect("utf8 stem");
    let out_path = out_dir.join(format!("{stem}_{arch}.hsaco"));

    // Extra compile flags (e.g. -DOCC_PAD_FLOATS=N) for kernel experiments.
    let extra: Vec<String> = env::var("DEEPSTRIX_KERNEL_CFLAGS")
        .unwrap_or_default()
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();
    let status = Command::new(hipcc)
        .arg("-O3")
        .arg("--genco")
        .arg(format!("--offload-arch={arch}"))
        .args(&extra)
        .arg(src)
        .arg("-o")
        .arg(&out_path)
        .status()
        .unwrap_or_else(|e| panic!("failed to invoke {hipcc}: {e}"));

    if !status.success() {
        panic!("hipcc failed for {} @ {arch}", src.display());
    }

    let env_name = format!(
        "KERNEL_{}_{}",
        stem.to_uppercase().replace('-', "_"),
        arch.to_uppercase()
    );
    println!("cargo:rustc-env={env_name}={}", out_path.display());
}
