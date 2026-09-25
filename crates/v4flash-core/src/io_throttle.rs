//! Preemptible BACKGROUND reads (box 2's expert prefetch readers).
//!
//! An 18.8 MB expert read goes to the drive as ~147 commands of 128 KB
//! (`max_sectors_kb=128`), all dispatched at once under the `none` scheduler,
//! and neither io_uring cancel nor I/O priority classes can recall or reorder
//! them there. So a background read that is already in flight delays an urgent
//! one by up to its whole length. The fix is to not hand the drive much
//! background work at a time: a thread marked BACKGROUND (`set_background`)
//! reads in `chunk_bytes()` pieces, and calls the installed pause hook before
//! each one. The daemon's hook waits while an urgent read is active. An
//! urgent read then waits for at most the chunks already in flight.
//!
//! Unmarked threads (every demand and urgent read) are unaffected.

use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::OnceLock;

thread_local! {
    /// `Some(token)` while this thread reads for a background job; the token
    /// is opaque here (the daemon uses `layer << 16 | expert`).
    static BACKGROUND: Cell<Option<u64>> = const { Cell::new(None) };
}

static CHUNK_BYTES: AtomicUsize = AtomicUsize::new(0);
static PAUSE: OnceLock<Box<dyn Fn(u64) + Send + Sync>> = OnceLock::new();

/// Mark (or unmark) the current thread's reads as background. Threads a
/// background read spawns must copy it (`background()` before, `set_background`
/// inside).
pub fn set_background(token: Option<u64>) {
    BACKGROUND.with(|b| b.set(token));
}

pub fn background() -> Option<u64> {
    BACKGROUND.with(|b| b.get())
}

/// Background read chunk size in bytes (0 = do not chunk). Rounded down to a
/// multiple of 4096 so O_DIRECT offsets stay aligned.
pub fn set_chunk_bytes(n: usize) {
    CHUNK_BYTES.store(n / 4096 * 4096, Relaxed);
}

pub fn chunk_bytes() -> usize {
    CHUNK_BYTES.load(Relaxed)
}

/// Install the hook a background chunk calls before it is issued (first call
/// wins). It gets the thread's token and returns when the chunk may go.
pub fn install_pause(f: impl Fn(u64) + Send + Sync + 'static) {
    let _ = PAUSE.set(Box::new(f));
}

/// Called before each background chunk.
pub fn pause(token: u64) {
    if let Some(f) = PAUSE.get() {
        f(token);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn background_is_per_thread() {
        set_background(Some(7));
        assert_eq!(background(), Some(7));
        let other = std::thread::spawn(background).join().unwrap();
        assert_eq!(other, None, "spawned threads must copy the mark explicitly");
        set_background(None);
        assert_eq!(background(), None);
    }

    #[test]
    fn chunk_is_page_aligned() {
        set_chunk_bytes(1_000_000);
        assert_eq!(chunk_bytes(), 1_000_000 / 4096 * 4096);
        set_chunk_bytes(0);
        assert_eq!(chunk_bytes(), 0);
    }
}
