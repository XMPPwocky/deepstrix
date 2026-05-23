//! Phase 0 gate dispatcher. Subcommands are added incrementally; this
//! commit (#2) only verifies that the build.rs pipeline embeds an hsaco for
//! each target gfx arch.

const HELLO_GFX1201: &[u8] = include_bytes!(env!("KERNEL_HELLO_GFX1201"));
const HELLO_GFX1151: &[u8] = include_bytes!(env!("KERNEL_HELLO_GFX1151"));

fn main() {
    println!("phase0 build artifacts:");
    println!("  hello@gfx1201: {} bytes", HELLO_GFX1201.len());
    println!("  hello@gfx1151: {} bytes", HELLO_GFX1151.len());
    println!("(subcommands land in commit 4)");
}
