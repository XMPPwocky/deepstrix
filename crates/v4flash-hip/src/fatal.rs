//! STICKY HIP ERRORS (HIP 700 instrumentation, owner-approved 2026-10-19).
//!
//! HIP 7.2.3 (CLR `rocclr/device/rocm/rocdevice.cpp`, `HIP_SKIP_ABORT_ON_GPU_ERROR` defaults to TRUE): a GPU queue
//! error (`callbackQueue`: e.g. `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`) or a system GPU event
//! (`BackendErrorCallBackHandler`: memory fault / HW exception / memory error) no longer aborts the process -- it sets the
//! PROCESS-WIDE `amd::Device::gpu_error_`, which is never cleared. From then on EVERY HIP API returns the converted code
//! (`HIP_INIT_API` and `HIP_RETURN` in `hipamd/src/hip_internal.hpp`), on every device and thread, `hipGetDevice`
//! included. The conversion (`ConvertHSAErrorIntoCLError` -> `ConvertCLErrorIntoHIPError`):
//!   * memory fault / aperture violation / invalid allocation        -> 700 `hipErrorIllegalAddress`
//!   * queue exception / invalid packet / illegal instruction / HW exception -> 719 `hipErrorLaunchFailure`
//!   * `HSA_STATUS_ERROR`, GPU memory error event, anything else       -> 999 `hipErrorUnknown`
//!   * incompatible / invalid argument -> 1, invalid code object -> 300, invalid ISA -> 218 (rare)
//!
//! 719 is ALSO returned non-sticky by a launch over the kernel's launch bounds (`hip_module.cpp`), and 1 / 300 / 218 are
//! ordinary errors, so the code alone does not tell: [`is_sticky`] PROBES -- after any failed call (700 included: the
//! probe runs only on a failure, so it is free) it asks `hipGetDevice`, which can fail only in that state.
//!
//! The first sticky error prints ONE machine-greppable line at once, before any grace period ([`marker`]):
//! `HIP-FATAL: code=700 name=hipErrorIllegalAddress call="hipMemcpyAsync(HtoD)" at=<file>:<line> thread=<name>
//! exiting=rc87` (`exiting=no` in a process that does not exit on it), so an external watchdog can tell "the hub is
//! exiting on its own" and stand down.
//!
//! A hub that saw one kept answering every request with HTTP 500 and never exited (10-18 23:50 UTC), so its supervisor
//! never restarted it. Now [`check`](crate::check) records the FIRST sticky error (code, call, `#[track_caller]`
//! location, thread) and logs it once; a process that called [`install_exit_on_fatal`] (the hub) then exits with
//! [`HIP_FATAL_EXIT_CODE`]: the engine via [`exit_now`] right after it fails its in-flight requests, or the backstop
//! thread [`install_exit_on_fatal`] arms after its grace period, whichever comes first. The exit dumps the
//! launch / copy ring ([`crate::ring`]) and uses `_exit` (no destructors run against a dead GPU, no core dump of a
//! 100 GB address space).
use std::panic::Location;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use crate::sys;

/// The process exit code after a sticky HIP error. The hub supervisor (`~/run_v41_server.sh`) relaunches on every code
/// but 0 / 130 / 143; chosen to be distinct from the hang watchdog's abort (134) in its `exited rc=` line.
pub const HIP_FATAL_EXIT_CODE: i32 = 87;

/// The first sticky HIP error this process saw.
#[derive(Clone, Debug)]
pub struct FatalHip {
    pub code: i32,
    pub name: String,
    pub message: String,
    /// The HIP call that returned it.
    pub context: &'static str,
    pub location: &'static Location<'static>,
    pub thread: String,
    pub at: Instant,
}

impl std::fmt::Display for FatalHip {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sticky HIP error {} ({}: {}) first returned by {} at {}:{} on thread {} {:.1} ms ago",
            self.code,
            self.name,
            self.message,
            self.context,
            self.location.file(),
            self.location.line(),
            self.thread,
            self.at.elapsed().as_secs_f64() * 1e3
        )
    }
}

static FATAL: OnceLock<FatalHip> = OnceLock::new();
static EXIT_ON_FATAL: AtomicBool = AtomicBool::new(false);
static GRACE_MS: AtomicU64 = AtomicU64::new(5000);
static EXITING: AtomicBool = AtomicBool::new(false);

/// The first sticky HIP error, once one was seen.
pub fn get() -> Option<&'static FatalHip> {
    FATAL.get()
}

/// Is the process in HIP's sticky GPU-error state? `code` is the error a call just returned (not `HIP_SUCCESS`).
pub fn is_sticky(code: sys::hipError_t) -> bool {
    if code == 600 {
        // hipErrorNotReady: a status, never the error state.
        return false;
    }
    let mut d = 0;
    // SAFETY: `d` is a valid out-pointer; hipGetDevice has no other preconditions.
    unsafe { sys::hipGetDevice(&mut d) != sys::HIP_SUCCESS }
}

/// `check`'s hook: a failed HIP call. Records and logs the first sticky error; arms the backstop exit.
pub(crate) fn observe(code: sys::hipError_t, err: &crate::HipError, location: &'static Location<'static>) {
    if FATAL.get().is_some() || !is_sticky(code) {
        return;
    }
    let f = FatalHip {
        code: err.code,
        name: err.name.clone(),
        message: err.message.clone(),
        context: err.context,
        location,
        thread: std::thread::current().name().unwrap_or("?").to_string(),
        at: Instant::now(),
    };
    if FATAL.set(f).is_err() {
        return; // another thread recorded first
    }
    let f = FATAL.get().expect("just set");
    let exiting = EXIT_ON_FATAL.load(Ordering::Acquire);
    // The marker first, unbuffered, before anything can wait: an external watchdog greps for it.
    eprintln!("{}", marker(f, exiting));
    eprintln!("HIP-FATAL detail: {f}; the GPU is unusable for the life of this process");
    tracing::error!(code = f.code, context = f.context, file = f.location.file(), line = f.location.line(), "HIP-FATAL: {f}");
    if exiting {
        let grace = GRACE_MS.load(Ordering::Relaxed);
        let spawned = std::thread::Builder::new().name("hip-fatal-backstop".into()).spawn(move || {
            std::thread::sleep(Duration::from_millis(grace));
            exit_now(&format!("backstop: nothing exited within {grace} ms of the sticky error"));
        });
        if let Err(e) = spawned {
            eprintln!("HIP-FATAL: cannot spawn the backstop thread ({e}); exiting now");
            exit_now("backstop thread spawn failed");
        }
    }
}

/// The one-line machine-greppable marker of the first sticky error (module doc).
pub fn marker(f: &FatalHip, exiting: bool) -> String {
    format!(
        "HIP-FATAL: code={} name={} call=\"{}\" at={}:{} thread={} exiting={}",
        f.code,
        f.name,
        f.context,
        f.location.file(),
        f.location.line(),
        f.thread.replace(char::is_whitespace, "_"),
        if exiting { format!("rc{HIP_FATAL_EXIT_CODE}") } else { "no".to_string() }
    )
}

/// Exit the process on a sticky HIP error (the hub): the engine calls [`exit_now`] once it has failed its in-flight
/// requests; a backstop thread exits `grace` after the first sticky error regardless. `V41_HIP_FATAL_GRACE_MS`
/// overrides `grace_ms`.
pub fn install_exit_on_fatal(grace_ms: u64) {
    let g = std::env::var("V41_HIP_FATAL_GRACE_MS").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(grace_ms);
    GRACE_MS.store(g, Ordering::Relaxed);
    EXIT_ON_FATAL.store(true, Ordering::Release);
    // A sticky error seen before the install (startup) still exits.
    if let Some(f) = get() {
        eprintln!("HIP-FATAL: {f} (seen before install_exit_on_fatal)");
        exit_now("sticky error before install");
    }
}

/// Did this process ask to exit on a sticky error?
pub fn exit_on_fatal_installed() -> bool {
    EXIT_ON_FATAL.load(Ordering::Acquire)
}

/// Log the one-line verdict, dump the ring and `_exit(HIP_FATAL_EXIT_CODE)`. A second caller (the backstop racing the
/// engine) parks: the first one is exiting.
pub fn exit_now(reason: &str) -> ! {
    if EXITING.swap(true, Ordering::AcqRel) {
        loop {
            std::thread::sleep(Duration::from_secs(1));
        }
    }
    let what = get().map_or_else(|| "no sticky error recorded".to_string(), |f| f.to_string());
    let line = format!("HIP-FATAL EXIT rc={HIP_FATAL_EXIT_CODE} ({reason}): {what}; exiting for supervisor restart");
    tracing::error!("{line}");
    eprintln!("{line}");
    crate::ring::dump(reason);
    eprintln!("{line}");
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    extern "C" {
        fn _exit(code: i32) -> !;
    }
    // SAFETY: `_exit` ends the process without running atexit handlers or static destructors (libamdhip64's would
    // touch the dead GPU from under the still-running threads).
    unsafe { _exit(HIP_FATAL_EXIT_CODE) }
}

#[cfg(test)]
mod tests {
    /// The one code decided without the `hipGetDevice` probe (no HIP call on a CPU).
    #[test]
    fn not_ready_is_never_sticky() {
        assert!(!super::is_sticky(600));
        assert!(super::get().is_none(), "is_sticky only answers; recording is observe's");
    }

    #[test]
    fn marker_is_one_greppable_line() {
        let f = super::FatalHip {
            code: 700,
            name: "hipErrorIllegalAddress".into(),
            message: "an illegal memory access was encountered".into(),
            context: "hipMemcpyAsync(HtoD)",
            location: std::panic::Location::caller(),
            thread: "deepstrix engine".into(),
            at: std::time::Instant::now(),
        };
        let m = super::marker(&f, true);
        assert!(m.starts_with("HIP-FATAL: code=700 name=hipErrorIllegalAddress call=\"hipMemcpyAsync(HtoD)\" at="), "{m}");
        assert!(m.ends_with("thread=deepstrix_engine exiting=rc87") && !m.contains('\n'), "{m}");
        assert!(super::marker(&f, false).ends_with("exiting=no"));
    }
}
