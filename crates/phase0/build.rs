//! Compile every `kernels/*.hip` file to a standalone code object (hsaco) for
//! each target gfx arch. Emits `cargo:rustc-env=KERNEL_<NAME>_<ARCH>=<path>` so
//! the binary can `include_bytes!(env!("KERNEL_<NAME>_<ARCH>"))`.
//!
//! Targets come from `$DEEPSTRIX_GFX_TARGETS` (space-separated) — set by the
//! Nix dev shell. Defaults to "gfx1201 gfx1151" if unset.

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

fn compile(hipcc: &str, src: &Path, arch: &str, out_dir: &Path) {
    let stem = src
        .file_stem()
        .expect("stem")
        .to_str()
        .expect("utf8 stem");
    let out_path = out_dir.join(format!("{stem}_{arch}.hsaco"));

    let status = Command::new(hipcc)
        .arg("-O3")
        .arg("--genco")
        .arg(format!("--offload-arch={arch}"))
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
