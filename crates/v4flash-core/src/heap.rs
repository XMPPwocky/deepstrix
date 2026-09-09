//! glibc heap introspection + trim for the long-running server.
//!
//! Why: deepstrix-server's host RSS grew 0.44 → 9.2 GiB over 17 h of
//! snapshot traffic (2026-09-08) in freed-but-retained glibc arena heaps,
//! and ~2.4 GiB of main-heap growth survived the GLIBC_TUNABLES fix. The
//! server calls [`trim_and_stats`] at the end of every request so that
//! (a) free pages inside every arena are returned to the kernel
//! (`malloc_trim` walks all arenas since glibc 2.8, not just the top
//! chunk) and (b) the log records in-use vs free heap bytes, which is the
//! only way to tell fragmentation from a real leak without a debugger.
//!
//! Linux/glibc only; a no-op elsewhere.

/// Heap counters from `mallinfo2` after a `malloc_trim(0)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct HeapStats {
    /// Bytes obtained from the system via brk for the main arena
    /// (mallinfo2 `arena`).
    pub arena: u64,
    /// Bytes in mmap'd chunks (`hblkhd`).
    pub mmapped: u64,
    /// Bytes in in-use chunks across all arenas (`uordblks`).
    pub in_use: u64,
    /// Bytes in free chunks across all arenas (`fordblks`).
    pub free: u64,
    /// Bytes at the top of the main arena that could be trimmed
    /// (`keepcost`).
    pub keepcost: u64,
    /// Whether `malloc_trim` reported releasing any memory.
    pub trimmed: bool,
}

/// Release free heap pages back to the kernel and report heap counters.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
pub fn trim_and_stats() -> HeapStats {
    // SAFETY: both are plain glibc calls with no preconditions; mallinfo2
    // returns by value.
    let trimmed = unsafe { libc::malloc_trim(0) } != 0;
    let mi = unsafe { libc::mallinfo2() };
    HeapStats {
        arena: mi.arena as u64,
        mmapped: mi.hblkhd as u64,
        in_use: mi.uordblks as u64,
        free: mi.fordblks as u64,
        keepcost: mi.keepcost as u64,
        trimmed,
    }
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
pub fn trim_and_stats() -> HeapStats {
    HeapStats::default()
}

/// Resident set size of this process in bytes (Linux `/proc/self/statm`),
/// or 0 if unavailable.
pub fn rss_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| s.split_whitespace().nth(1)?.parse::<u64>().ok())
        .map(|pages| pages * 4096)
        .unwrap_or(0)
}
