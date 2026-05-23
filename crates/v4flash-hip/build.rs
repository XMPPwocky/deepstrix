//! Link against libamdhip64 from the ROCm install. Path comes from
//! $ROCM_PATH (set by the Nix dev shell). No headers are read; we declare
//! the FFI surface by hand in src/sys.rs.

use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");

    let rocm = env::var("HIP_PATH")
        .or_else(|_| env::var("ROCM_PATH"))
        .unwrap_or_else(|_| "/opt/rocm".to_string());

    println!("cargo:rustc-link-search=native={rocm}/lib");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
}
