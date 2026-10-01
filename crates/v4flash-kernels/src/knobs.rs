//! RUNTIME KNOBS: one implementation for every `V41_*` setting (owner,
//! 2026-10-01: "ad-hoc code per knob, or a more sensible single
//! implementation?"). Until then each knob parsed its own env var at its use
//! (some on every decode step), and each of the eight run-time-file knobs was
//! its own ~20-line copy of "env + file path + mutex cache + 2 s re-read",
//! with subtly different truthiness and fallbacks.
//!
//! A knob is a `static Knob` in its crate's table (`knobs!`; this file's
//! `ALL` at the bottom, `deepstrix_server::knobs::ALL`), named by its env var.
//! Its value lives in an atomic: a read is one load -- no env lookup, no lock.
//!
//! Sources, lowest to highest:
//! 1. the default;
//! 2. the env var (`V41_X=...`);
//! 3. LIVE knobs only: the legacy one-value file named by `<legacy>` (the
//!    10-01 `V41_X_FILE=/dev/shm/x.txt` scheme, kept for the A/B scripts);
//! 4. LIVE knobs only: the process's knob file `V41_KNOBS_FILE` (else the
//!    default `start_with` names, e.g. box 2's `~/expertd-knobs.txt`) --
//!    `NAME=value` lines (the env names, or a knob's short `alias`), `#`
//!    comments.
//!
//! `start` (once per process, at startup) resolves every knob, logs the ones
//! off their default, and runs a watcher thread that re-resolves the live ones
//! every second (at once after `request_pass`, e.g. box 2's SIGUSR2). A key
//! removed from a file reverts its knob to the next source down; an INVALID
//! value warns and keeps the current one (a typo must not silently reset a
//! knob mid-A/B); an EMPTY value -- and an existing but empty knob file, what a
//! truncating `printf ... > file` shows a reader for an instant -- keeps the
//! current values silently (clear the file on purpose by deleting it or
//! leaving a comment line; or write a temp file and `mv` it); an unknown key,
//! or a static knob's key, warns. Every change is logged (`knob changed`), and the whole effective
//! table goes to `<V41_KNOBS_FILE>.effective` after every change. A process
//! that never calls `start` (tests, tools) resolves each knob once, at first
//! use.
//!
//! STATIC knobs (sizing, drivers, anything allocated or chosen once) read
//! their env at first use and never change; a knob is LIVE only where a
//! change between two reads is safe.

use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};

/// Where a knob's current value came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Default,
    Env,
    /// The legacy one-value file (`Knob::legacy`).
    LegacyFile,
    /// `V41_KNOBS_FILE`.
    File,
    /// `Knob::set` (tests, in-process harnesses).
    Set,
}

impl Source {
    fn from_u8(v: u8) -> Self {
        match v {
            1 => Source::Env,
            2 => Source::LegacyFile,
            3 => Source::File,
            4 => Source::Set,
            _ => Source::Default,
        }
    }
    pub(crate) fn as_u8(self) -> u8 {
        match self {
            Source::Default => 0,
            Source::Env => 1,
            Source::LegacyFile => 2,
            Source::File => 3,
            Source::Set => 4,
        }
    }
}

/// A knob's type, default and bounds.
#[derive(Debug)]
pub enum Kind {
    /// `1|on|true|yes` / `0|off|false|no` (any case).
    Flag(bool),
    /// An unsigned integer, clamped into `min..=max`.
    Int { default: u64, min: u64, max: u64 },
    /// A finite real, clamped into `min..=max`.
    Real { default: f64, min: f64, max: f64 },
    /// One of `names` (each option with its aliases, the first is shown; any
    /// case); the value is the option's index.
    Choice { default: usize, names: &'static [&'static [&'static str]] },
    /// Free text, STATIC only (ladders, seeds): `None` when unset.
    Text,
}

impl Kind {
    /// The value's bits for `s`, or `None` when `s` is not a valid value.
    pub fn parse(&self, s: &str) -> Option<u64> {
        let s = s.trim();
        match *self {
            Kind::Flag(_) => match s.to_ascii_lowercase().as_str() {
                "1" | "on" | "true" | "yes" => Some(1),
                "0" | "off" | "false" | "no" => Some(0),
                _ => None,
            },
            Kind::Int { min, max, .. } => s.parse::<u64>().ok().map(|v| v.clamp(min, max)),
            Kind::Real { min, max, .. } => s.parse::<f64>().ok().filter(|v| v.is_finite()).map(|v| v.clamp(min, max).to_bits()),
            Kind::Choice { names, .. } => names.iter().position(|o| o.iter().any(|n| n.eq_ignore_ascii_case(s))).map(|i| i as u64),
            Kind::Text => Some(0),
        }
    }

    fn default_bits(&self) -> u64 {
        match *self {
            Kind::Flag(d) => d as u64,
            Kind::Int { default, .. } => default,
            Kind::Real { default, .. } => default.to_bits(),
            Kind::Choice { default, .. } => default as u64,
            Kind::Text => 0,
        }
    }

    fn show(&self, bits: u64) -> String {
        match *self {
            Kind::Flag(_) => if bits != 0 { "1" } else { "0" }.to_string(),
            Kind::Int { .. } => bits.to_string(),
            Kind::Real { .. } => format!("{}", f64::from_bits(bits)),
            Kind::Choice { names, .. } => names.get(bits as usize).and_then(|o| o.first()).copied().unwrap_or("?").to_string(),
            Kind::Text => String::new(),
        }
    }
}

/// One setting: declare it in a `knobs!` table, read it with the accessor of
/// its kind (`on`, `get` / `usize`, `f64`, `pick`, `str`).
pub struct Knob {
    /// The env var; also its key in `V41_KNOBS_FILE`.
    pub name: &'static str,
    pub kind: Kind,
    /// Re-read while running (`start`'s watcher).
    pub live: bool,
    /// The env var naming this knob's legacy one-value file (live knobs only).
    pub legacy: Option<&'static str>,
    /// A second key for it in the knob file (box 2's pre-10-01 short keys).
    pub alias: Option<&'static str>,
    /// Called after the value is resolved or changes (`start`'s watcher, a
    /// first use): pushes it into state another crate owns.
    hook: Option<fn(&Knob)>,
    bits: AtomicU64,
    /// 0 = not yet resolved; else 1 + `Source`.
    state: AtomicU8,
    text: OnceLock<Option<String>>,
}

impl Knob {
    const fn new(name: &'static str, kind: Kind) -> Self {
        Self { name, kind, live: false, legacy: None, alias: None, hook: None, bits: AtomicU64::new(0), state: AtomicU8::new(0), text: OnceLock::new() }
    }

    pub const fn flag(name: &'static str, default: bool) -> Self {
        Self::new(name, Kind::Flag(default))
    }

    pub const fn int(name: &'static str, default: u64, min: u64, max: u64) -> Self {
        Self::new(name, Kind::Int { default, min, max })
    }

    pub const fn real(name: &'static str, default: f64, min: f64, max: f64) -> Self {
        Self::new(name, Kind::Real { default, min, max })
    }

    pub const fn choice(name: &'static str, default: usize, names: &'static [&'static [&'static str]]) -> Self {
        Self::new(name, Kind::Choice { default, names })
    }

    pub const fn text(name: &'static str) -> Self {
        Self::new(name, Kind::Text)
    }

    /// Re-read while running.
    pub const fn live(mut self) -> Self {
        self.live = true;
        self
    }

    /// Also read the legacy one-value file named by the env var `file_env`
    /// (implies `live`).
    pub const fn legacy(mut self, file_env: &'static str) -> Self {
        self.live = true;
        self.legacy = Some(file_env);
        self
    }

    /// Also accept the knob-file key `key` (implies `live`).
    pub const fn alias(mut self, key: &'static str) -> Self {
        self.live = true;
        self.alias = Some(key);
        self
    }

    /// Call `f` after the value is resolved or changes.
    pub const fn hook(mut self, f: fn(&Knob)) -> Self {
        self.hook = Some(f);
        self
    }

    fn load(&self) -> u64 {
        if self.state.load(Ordering::Acquire) == 0 {
            self.resolve_first();
        }
        self.bits.load(Ordering::Relaxed)
    }

    /// A `Flag`'s value.
    pub fn on(&self) -> bool {
        debug_assert!(matches!(self.kind, Kind::Flag(_)), "{}: not a flag", self.name);
        self.load() != 0
    }

    /// An `Int`'s value.
    pub fn get(&self) -> u64 {
        debug_assert!(matches!(self.kind, Kind::Int { .. }), "{}: not an int", self.name);
        self.load()
    }

    /// An `Int`'s value as a `usize`.
    pub fn usize(&self) -> usize {
        self.get() as usize
    }

    /// A `Real`'s value.
    pub fn f64(&self) -> f64 {
        debug_assert!(matches!(self.kind, Kind::Real { .. }), "{}: not a real", self.name);
        f64::from_bits(self.load())
    }

    /// A `Choice`'s option index.
    pub fn pick(&self) -> usize {
        debug_assert!(matches!(self.kind, Kind::Choice { .. }), "{}: not a choice", self.name);
        self.load() as usize
    }

    /// A `Text`'s value (`None` = unset).
    pub fn str(&self) -> Option<&str> {
        debug_assert!(matches!(self.kind, Kind::Text), "{}: not text", self.name);
        self.text.get_or_init(|| std::env::var(self.name).ok()).as_deref()
    }

    /// Where the current value came from (e.g. a default derived from another
    /// knob applies only while this one is `Default`).
    pub fn source(&self) -> Source {
        self.load();
        Source::from_u8(self.state.load(Ordering::Acquire).saturating_sub(1))
    }

    /// Set the value in-process (tests, A/B harnesses). A LIVE knob's value is
    /// replaced by its sources' at the running watcher's next pass (`start`):
    /// set live knobs only in processes without one.
    pub fn set(&self, s: &str) -> bool {
        match self.kind.parse(s) {
            Some(bits) => {
                self.store(bits, Source::Set);
                true
            }
            None => false,
        }
    }

    fn store(&self, bits: u64, src: Source) {
        self.bits.store(bits, Ordering::Relaxed);
        self.state.store(1 + src.as_u8(), Ordering::Release);
    }

    /// The value now, for logs and the effective table.
    pub fn show(&self) -> String {
        match self.kind {
            Kind::Text => self.str().unwrap_or("").to_string(),
            _ => self.kind.show(self.load()),
        }
    }

    /// First use: every source once (no watcher needed for a static knob, and
    /// a live one read before `start` -- or without it -- still sees its files).
    /// Under `RESOLVE`, like the watcher's pass: a first use cannot overwrite
    /// a value the watcher stored meanwhile.
    fn resolve_first(&self) {
        {
            let _g = RESOLVE.lock().unwrap_or_else(|p| p.into_inner());
            if self.state.load(Ordering::Acquire) != 0 {
                return;
            }
            self.resolve_first_locked();
        }
        if let Some(h) = self.hook {
            h(self);
        }
    }

    fn resolve_first_locked(&self) {
        let file = knob_file().and_then(|p| std::fs::read_to_string(p).ok()).map(|t| parse_file(&t)).unwrap_or_default();
        let env = |k: &str| std::env::var(k).ok();
        let read = |p: &str| std::fs::read_to_string(p).ok();
        let (bits, src, bad) = self.resolve(&file, &env, &read);
        if let Some(raw) = bad {
            tracing::warn!(knob = self.name, value = %raw, using = %self.kind.show(bits), "knobs: invalid value");
        }
        self.store(bits, src);
    }

    /// The value this knob's sources give, highest first; an invalid value at
    /// a source keeps the CURRENT value (`Some(raw)` reports it).
    fn resolve(&self, file: &BTreeMap<String, String>, env: &dyn Fn(&str) -> Option<String>, read: &dyn Fn(&str) -> Option<String>) -> (u64, Source, Option<String>) {
        let mut cands: Vec<(Source, String)> = Vec::new();
        if self.live {
            if let Some(v) = file.get(self.name).or_else(|| self.alias.and_then(|a| file.get(a))) {
                cands.push((Source::File, v.clone()));
            }
            if let Some(v) = self.legacy.and_then(|f| env(f)).and_then(|p| read(&p)) {
                cands.push((Source::LegacyFile, v));
            }
        }
        if let Some(v) = env(self.name) {
            cands.push((Source::Env, v));
        }
        // First resolution: an empty source has no current value to keep -- the
        // next source down decides (a hub starting while a legacy file is empty
        // must take its env, not the default; review round 18).
        if self.state.load(Ordering::Acquire) == 0 && !matches!(self.kind, Kind::Text) {
            cands.retain(|(_, v)| !v.trim().is_empty());
        }
        match cands.into_iter().next() {
            None => (self.kind.default_bits(), Source::Default, None),
            // Empty (a truncating write in progress, `V41_X=`): keep the
            // current value, silently (the default on first use).
            Some((_, raw)) if raw.trim().is_empty() && !matches!(self.kind, Kind::Text) => {
                let st = self.state.load(Ordering::Acquire);
                if st == 0 {
                    (self.kind.default_bits(), Source::Default, None)
                } else {
                    (self.bits.load(Ordering::Relaxed), Source::from_u8(st - 1), None)
                }
            }
            Some((src, raw)) => match self.kind.parse(&raw) {
                Some(bits) => (bits, src, None),
                None => {
                    // Invalid: keep the current value (the default on first use).
                    let st = self.state.load(Ordering::Acquire);
                    if st == 0 {
                        (self.kind.default_bits(), Source::Default, Some(raw))
                    } else {
                        (self.bits.load(Ordering::Relaxed), Source::from_u8(st - 1), Some(raw))
                    }
                }
            },
        }
    }
}

/// Declare a crate's knob table: `static`s, plus `ALL` (every knob of the
/// table, for `start`).
///
/// ```ignore
/// v4flash_kernels::knobs! {
///     /// Doc.
///     pub static MS_SLOTS = Knob::int("V41_MS_SLOTS", 8, 1, 64);
/// }
/// ```
#[macro_export]
macro_rules! knobs {
    ($($(#[$m:meta])* $v:vis static $n:ident = $e:expr;)*) => {
        $($(#[$m])* $v static $n: $crate::knobs::Knob = { use $crate::knobs::Knob; $e };)*
        /// Every knob of this table (`v4flash_kernels::knobs::start`).
        pub static ALL: &[&$crate::knobs::Knob] = &[$(&$n),*];
    };
}

/// The process's knob file: `V41_KNOBS_FILE`, else `start_with`'s default.
pub fn knob_file() -> Option<String> {
    std::env::var("V41_KNOBS_FILE").ok().filter(|s| !s.is_empty()).or_else(|| DEFAULT_FILE.get().cloned())
}

static DEFAULT_FILE: OnceLock<String> = OnceLock::new();
/// Also print the watcher's lines to stderr (a process without a tracing
/// subscriber: box 2's daemon).
static ECHO: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// `NAME=value` lines; `#` starts a comment; the last of a repeated key wins.
pub fn parse_file(text: &str) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if let Some((k, v)) = line.split_once('=') {
            m.insert(k.trim().to_string(), v.trim().to_string());
        }
    }
    m
}

/// The non-empty, non-comment lines of a knob file that are not `NAME=value`
/// (`parse_file` skips them; the watcher warns).
pub fn bad_lines(text: &str) -> Vec<String> {
    text.lines().map(|l| l.split('#').next().unwrap_or("").trim()).filter(|l| !l.is_empty() && !l.contains('=')).map(str::to_string).collect()
}

/// What one pass of the watcher found.
#[derive(Default)]
pub struct Pass {
    /// `(name, old, new, source)` per changed knob.
    pub changed: Vec<(&'static str, String, String, Source)>,
    /// Knobs resolved or changed whose `hook` is due (the caller runs them,
    /// outside `RESOLVE`).
    pub hooks: Vec<&'static Knob>,
    /// One line per problem (unknown / static key, invalid value).
    pub warnings: Vec<String>,
}

/// One pass over `knobs` (the watcher's step; pure but for the knobs it
/// stores into). `warned` remembers problems already reported, so each is
/// logged once while it persists.
pub fn pass(knobs: &[&'static Knob], file: &BTreeMap<String, String>, env: &dyn Fn(&str) -> Option<String>, read: &dyn Fn(&str) -> Option<String>, warned: &mut HashSet<String>) -> Pass {
    let mut out = Pass::default();
    let mut seen = HashSet::new();
    for k in knobs {
        seen.insert(k.name);
        if let Some(a) = k.alias {
            seen.insert(a);
        }
        if file.contains_key(k.name) && !k.live {
            let w = format!("{}: static (restart to change); the knob file's value is ignored", k.name);
            if warned.insert(w.clone()) {
                out.warnings.push(w);
            }
        }
        // `NAME=` in the knob file is more likely a mistake than a truncating
        // write (an empty FILE is handled whole, `step`): say so, once.
        let file_val = file.get(k.name).or_else(|| k.alias.and_then(|a| file.get(a)));
        if k.live && file_val.is_some_and(|v| v.trim().is_empty()) && warned.insert(format!("empty:{}", k.name)) {
            // (Not `k.show()`: on a first resolution that would resolve -- and
            // take `RESOLVE`, which the watcher's pass holds.)
            let now = if k.state.load(Ordering::Acquire) != 0 { k.kind.show(k.bits.load(Ordering::Relaxed)) } else { "its env/default".to_string() };
            out.warnings.push(format!("{}: empty value in the knob file; keeping {now}", k.name));
        }
        let first = k.state.load(Ordering::Acquire) == 0;
        if !first && !k.live {
            continue;
        }
        let (bits, src, bad) = k.resolve(file, env, read);
        if bad.is_none() {
            // A later return of the same bad value is reported again.
            let prefix = format!("{}=", k.name);
            warned.retain(|w| !w.starts_with(&prefix));
        }
        if first {
            // First resolution (`start`): no change to report.
            if let Some(raw) = bad {
                out.warnings.push(format!("{}: invalid value {raw:?}; using {}", k.name, k.kind.show(bits)));
            }
            k.store(bits, src);
            if k.hook.is_some() {
                out.hooks.push(*k);
            }
            continue;
        }
        if let Some(raw) = bad {
            let w = format!("{}: invalid value {raw:?}; keeping {}", k.name, k.kind.show(bits));
            if warned.insert(format!("{}={raw}", k.name)) {
                out.warnings.push(w);
            }
            continue;
        }
        let (old_bits, old_src) = (k.bits.load(Ordering::Relaxed), k.source());
        if old_bits != bits || old_src != src {
            k.store(bits, src);
            if old_bits != bits {
                out.changed.push((k.name, k.kind.show(old_bits), k.kind.show(bits), src));
                if k.hook.is_some() {
                    out.hooks.push(*k);
                }
            }
        }
    }
    for key in file.keys() {
        if !seen.contains(key.as_str()) && warned.insert(format!("unknown:{key}")) {
            out.warnings.push(format!("{key}: no such knob in this process; ignored"));
        }
    }
    out
}

/// The effective table, one knob per line (`<V41_KNOBS_FILE>.effective`).
pub fn effective(knobs: &[&'static Knob]) -> String {
    let mut s = String::from("# NAME=value  # source, live|static, default\n");
    for k in knobs {
        let default = match k.kind {
            Kind::Text => String::new(),
            _ => k.kind.show(k.kind.default_bits()),
        };
        s.push_str(&format!("{}={}  # {:?}, {}, default {}\n", k.name, k.show(), k.source(), if k.live { "live" } else { "static" }, default));
    }
    s
}

static REGISTERED: Mutex<Vec<&'static Knob>> = Mutex::new(Vec::new());

/// Held by a first resolution and by the watcher's pass (never by a read).
static RESOLVE: Mutex<()> = Mutex::new(());

/// A logged change of a live knob (`changes_since`; the perfetto `knobs` track).
#[derive(Clone, Debug)]
pub struct Change {
    /// 1, 2, ... in order of the changes.
    pub seq: u64,
    /// CLOCK_REALTIME ns (the perfetto exporters' clock).
    pub t_ns: u64,
    /// CLOCK_MONOTONIC_RAW ns (evtrace's clock).
    pub t_raw: u64,
    pub name: &'static str,
    pub from: String,
    pub to: String,
    pub source: Source,
}

/// The last `CHANGES_KEPT` changes.
static CHANGES: Mutex<(u64, std::collections::VecDeque<Change>)> = Mutex::new((0, std::collections::VecDeque::new()));
const CHANGES_KEPT: usize = 4096;

/// The seq of the latest change (0 = none yet).
pub fn change_seq() -> u64 {
    CHANGES.lock().unwrap_or_else(|p| p.into_inner()).0
}

/// The changes after `seq` still kept, oldest first.
pub fn changes_since(seq: u64) -> Vec<Change> {
    CHANGES.lock().unwrap_or_else(|p| p.into_inner()).1.iter().filter(|c| c.seq > seq).cloned().collect()
}

/// Every registered knob now: `(name, value, source, live)` (traces, dumps).
/// Without `start` (a tool, a bench), the kernels' own table (`ALL`).
pub fn snapshot() -> Vec<(&'static str, String, Source, bool)> {
    let mut knobs = REGISTERED.lock().unwrap_or_else(|p| p.into_inner()).clone();
    if knobs.is_empty() {
        knobs = ALL.to_vec();
    }
    knobs.iter().map(|k| (k.name, k.show(), k.source(), k.live)).collect()
}

fn wall_ns() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0)
}

/// Resolve every knob of `tables` now, log the ones off their default, and
/// (once per process) start the watcher: every second, re-read
/// `V41_KNOBS_FILE` and the legacy files, store and log what changed, and
/// rewrite the effective table. Call once, at startup.
pub fn start(tables: &[&'static [&'static Knob]]) {
    start_with(tables, None, false)
}

/// `start` with a knob file to use when `V41_KNOBS_FILE` is unset, and with
/// the watcher's lines also on stderr when `echo` (box 2's daemon).
pub fn start_with(tables: &[&'static [&'static Knob]], default_file: Option<String>, echo: bool) {
    if let Some(f) = default_file {
        let _ = DEFAULT_FILE.set(f);
    }
    if echo {
        ECHO.store(true, Ordering::Relaxed);
    }
    let knobs: Vec<&'static Knob> = tables.iter().flat_map(|t| t.iter().copied()).collect();
    let mut names = HashSet::new();
    for k in &knobs {
        if !names.insert(k.name) {
            tracing::warn!(knob = k.name, "knobs: declared twice");
        }
    }
    let first = {
        let mut reg = REGISTERED.lock().unwrap_or_else(|p| p.into_inner());
        let first = reg.is_empty();
        reg.extend(knobs.iter().copied());
        first
    };
    step_now();
    let set: Vec<String> = knobs.iter().filter(|k| k.source() != Source::Default).map(|k| format!("{}={}({:?})", k.name, k.show(), k.source())).collect();
    tracing::info!(knobs = knobs.len(), live = knobs.iter().filter(|k| k.live).count(), file = ?knob_file(), set = set.join(" "), "knobs: resolved");
    if ECHO.load(Ordering::Relaxed) {
        eprintln!("knobs: resolved {} ({} live) file={:?} set: {}", knobs.len(), knobs.iter().filter(|k| k.live).count(), knob_file(), set.join(" "));
    }
    if first {
        let spawned = std::thread::Builder::new().name("knobs".into()).spawn(|| {
            let mut ticks = 0u32;
            loop {
                std::thread::sleep(std::time::Duration::from_millis(100));
                ticks += 1;
                if REQUEST.swap(false, Ordering::Relaxed) || ticks >= 10 {
                    ticks = 0;
                    step_now();
                }
            }
        });
        if let Err(e) = spawned {
            tracing::error!(error = %e, "knobs: watcher not started; live knobs keep their startup values");
        }
    }
}

static REQUEST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Ask the watcher for a pass now (within ~100 ms) instead of at its next
/// second. One atomic store: safe from a signal handler (box 2's SIGUSR2), and
/// keeps the pass's I/O off the caller's thread.
pub fn request_pass() {
    REQUEST.store(true, Ordering::Relaxed);
}

/// One watcher pass over every registered knob, now, on the calling thread
/// (the watcher; startup).
pub fn step_now() {
    static WARNED: Mutex<Option<HashSet<String>>> = Mutex::new(None);
    let mut g = WARNED.lock().unwrap_or_else(|p| p.into_inner());
    step(g.get_or_insert_with(HashSet::new));
}

fn step(warned: &mut HashSet<String>) {
    let knobs = REGISTERED.lock().unwrap_or_else(|p| p.into_inner()).clone();
    // An existing but EMPTY knob file is a truncating write in progress: the
    // last file's keys stand (a missing file clears them).
    static LAST: Mutex<Option<BTreeMap<String, String>>> = Mutex::new(None);
    let read = knob_file().and_then(|p| std::fs::read_to_string(p).ok());
    let text = read.clone().unwrap_or_default();
    let file = {
        let mut last = LAST.lock().unwrap_or_else(|p| p.into_inner());
        let file = match read {
            Some(t) if t.trim().is_empty() => last.clone().unwrap_or_default(),
            Some(t) => parse_file(&t),
            None => BTreeMap::new(),
        };
        *last = Some(file.clone());
        file
    };
    let mut p = {
        let _g = RESOLVE.lock().unwrap_or_else(|p| p.into_inner());
        pass(&knobs, &file, &|k| std::env::var(k).ok(), &|p| std::fs::read_to_string(p).ok(), warned)
    };
    for l in bad_lines(&text) {
        if warned.insert(format!("line:{l}")) {
            p.warnings.push(format!("{l:?}: not NAME=value; ignored"));
        }
    }
    for k in &p.hooks {
        if let Some(h) = k.hook {
            h(k);
        }
    }
    let echo = ECHO.load(Ordering::Relaxed);
    for w in &p.warnings {
        tracing::warn!("knobs: {w}");
        if echo {
            eprintln!("knobs: WARN {w}");
        }
    }
    if !p.changed.is_empty() {
        let (t_ns, t_raw) = (wall_ns(), crate::het::evtrace::now() as u64);
        {
            let mut log = CHANGES.lock().unwrap_or_else(|p| p.into_inner());
            for (name, old, new, src) in &p.changed {
                tracing::info!(knob = *name, from = %old, to = %new, source = ?src, "knob changed");
                if echo {
                    eprintln!("knobs: {name} {old} -> {new} ({src:?})");
                }
                log.0 += 1;
                let seq = log.0;
                log.1.push_back(Change { seq, t_ns, t_raw, name, from: old.clone(), to: new.clone(), source: *src });
                if log.1.len() > CHANGES_KEPT {
                    log.1.pop_front();
                }
            }
        }
        // evtrace `knob` records AFTER `CHANGES` is released: no nested locks
        // (the interner is a leaf, docs/v41/EVTRACE_REBUILD_PLAN.md 2.6).
        for (name, _, new, src) in &p.changed {
            crate::het::evtrace::emit_knob(t_raw as f64, name, new, src.as_u8());
        }
    }
    static WRITTEN: Mutex<bool> = Mutex::new(false);
    let mut written = WRITTEN.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(f) = knob_file() {
        if !*written || !p.changed.is_empty() {
            let _ = std::fs::write(format!("{f}.effective"), effective(&knobs));
            *written = true;
        }
    }
}

// The kernels' knobs (the server's: `deepstrix_server::knobs`).
crate::knobs! {
    /// `V41_LM_PREFILL` (default off): layer-major CED prefill for NEW jobs
    /// (`PrefillJob::new`; a job keeps the mode it started with). Live; the
    /// 10-01 `V41_MS_LM_FILE` file still works.
    pub static LM_PREFILL = Knob::flag("V41_LM_PREFILL", false).legacy("V41_MS_LM_FILE");
    /// `V41_SUB_LAMBDA` (default 0.1, 0..=1): the cache prior's strength
    /// (`b2_mirror::lambda`). Live; `V41_SUB_LAMBDA_FILE` still works.
    pub static SUB_LAMBDA = Knob::real("V41_SUB_LAMBDA", 0.1, 0.0, 1.0).legacy("V41_SUB_LAMBDA_FILE");
    /// `V41_B2_PIN_PREFILL_BAND` (default 2048): `b2_mirror::pin_prefill_band`.
    /// Live; `V41_B2_PIN_PREFILL_BAND_FILE` still works.
    pub static B2_PIN_PREFILL_BAND = Knob::int("V41_B2_PIN_PREFILL_BAND", 2048, 0, u32::MAX as u64).legacy("V41_B2_PIN_PREFILL_BAND_FILE");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_of(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |k| pairs.iter().find(|p| p.0 == k).map(|p| p.1.to_string())
    }

    #[test]
    fn kinds_parse_like_the_old_readers() {
        let f = Kind::Flag(true);
        for (s, v) in [("1", Some(1)), ("on", Some(1)), ("TRUE", Some(1)), (" 0 ", Some(0)), ("off", Some(0)), ("2", None), ("", None)] {
            assert_eq!(f.parse(s), v, "{s:?}");
        }
        let i = Kind::Int { default: 6, min: 2, max: 64 };
        assert_eq!(i.parse("1"), Some(2));
        assert_eq!(i.parse("4"), Some(4));
        assert_eq!(i.parse("99"), Some(64));
        assert_eq!(i.parse("-1"), None);
        let r = Kind::Real { default: 0.1, min: 0.0, max: 1.0 };
        assert_eq!(r.parse("0.25").map(f64::from_bits), Some(0.25));
        assert_eq!(r.parse("7").map(f64::from_bits), Some(1.0));
        assert_eq!(r.parse("nan"), None);
        let c = Kind::Choice { default: 0, names: &[&["1", "on"], &["0", "off"], &["check"]] };
        assert_eq!(c.parse("CHECK"), Some(2));
        assert_eq!(c.parse("0"), Some(1));
        assert_eq!(c.parse("x"), None);
        assert_eq!(c.show(1), "0");
    }

    #[test]
    fn sources_stack_and_a_removed_key_reverts() {
        static K: Knob = Knob::int("T_MIN_ROWS", 6, 2, 64).legacy("T_MIN_ROWS_FILE");
        static S: Knob = Knob::int("T_SLOTS", 8, 1, 64);
        let knobs: [&'static Knob; 2] = [&K, &S];
        let mut warned = HashSet::new();
        let read = |p: &str| if p == "/legacy" { Some("3\n".to_string()) } else { None };
        // env only
        let env = env_of(&[("T_MIN_ROWS", "4"), ("T_SLOTS", "6")]);
        let p = pass(&knobs, &BTreeMap::new(), &env, &read, &mut warned);
        assert_eq!((K.get(), K.source()), (4, Source::Env));
        assert_eq!((S.get(), S.source()), (6, Source::Env));
        assert!(p.warnings.is_empty());
        // + the legacy file
        let env = env_of(&[("T_MIN_ROWS", "4"), ("T_MIN_ROWS_FILE", "/legacy")]);
        let p = pass(&knobs, &BTreeMap::new(), &env, &read, &mut warned);
        assert_eq!((K.get(), K.source()), (3, Source::LegacyFile));
        assert_eq!(p.changed, vec![("T_MIN_ROWS", "4".to_string(), "3".to_string(), Source::LegacyFile)]);
        // + the knob file (wins); a static knob's key is ignored, with a warning
        let file = parse_file("T_MIN_ROWS = 2  # two lanes from 2 rows\nT_SLOTS=4\n");
        let p = pass(&knobs, &file, &env, &read, &mut warned);
        assert_eq!((K.get(), K.source()), (2, Source::File));
        assert_eq!(S.get(), 6, "static: never re-read");
        assert_eq!(p.warnings.len(), 1, "{:?}", p.warnings);
        // the same problem is reported once
        assert!(pass(&knobs, &file, &env, &read, &mut warned).warnings.is_empty());
        // an invalid value keeps the current one (warned once)
        let bad = parse_file("T_MIN_ROWS=two\n");
        let p = pass(&knobs, &bad, &env, &read, &mut warned);
        assert_eq!((K.get(), K.source()), (2, Source::File));
        assert_eq!(p.warnings.len(), 1);
        assert!(p.changed.is_empty());
        // key removed: back to the legacy file, then to the env, then the default
        pass(&knobs, &BTreeMap::new(), &env, &read, &mut warned);
        assert_eq!((K.get(), K.source()), (3, Source::LegacyFile));
        let env2 = env_of(&[("T_MIN_ROWS", "4")]);
        pass(&knobs, &BTreeMap::new(), &env2, &read, &mut warned);
        assert_eq!((K.get(), K.source()), (4, Source::Env));
        pass(&knobs, &BTreeMap::new(), &env_of(&[]), &read, &mut warned);
        assert_eq!((K.get(), K.source()), (6, Source::Default));
        // an empty value keeps the current one; in the knob file that is
        // probably a mistake, so it warns (once)
        let p = pass(&knobs, &parse_file("T_MIN_ROWS=\n"), &env_of(&[]), &read, &mut warned);
        assert_eq!((K.get(), p.warnings.len(), p.changed.len()), (6, 1, 0));
        assert!(pass(&knobs, &parse_file("T_MIN_ROWS=\n"), &env_of(&[]), &read, &mut warned).warnings.is_empty());
        // ... and an empty LEGACY file (a truncating `echo 4 > f`) keeps it silently
        let env_l = env_of(&[("T_MIN_ROWS_FILE", "/empty")]);
        let read_e = |p: &str| if p == "/empty" { Some(String::new()) } else { None };
        let p = pass(&knobs, &BTreeMap::new(), &env_l, &read_e, &mut warned);
        assert_eq!((K.get(), p.warnings.len(), p.changed.len()), (6, 0, 0));
        // a bad value that was fixed (the passes above resolved valid) and
        // comes back is reported again, once
        assert_eq!(pass(&knobs, &bad, &env, &read, &mut warned).warnings.len(), 1);
        assert!(pass(&knobs, &bad, &env, &read, &mut warned).warnings.is_empty());
        pass(&knobs, &BTreeMap::new(), &env_of(&[]), &read, &mut warned);
        // lines that are not NAME=value
        assert_eq!(bad_lines("# c\nT_MIN_ROWS 3\n\nT_SLOTS=2 # ok\n"), vec!["T_MIN_ROWS 3".to_string()]);
        // unknown keys warn once
        let p = pass(&knobs, &parse_file("T_NOPE=1\n"), &env_of(&[]), &read, &mut warned);
        assert_eq!(p.warnings, vec!["T_NOPE: no such knob in this process; ignored".to_string()]);
        // the effective table lists both
        let e = effective(&knobs);
        assert!(e.contains("T_MIN_ROWS=6  # Default, live, default 6") && e.contains("T_SLOTS=6  # Env, static, default 8"), "{e}");
    }

    #[test]
    fn an_alias_is_a_file_key_and_a_hook_runs_on_resolve_and_change() {
        use std::sync::atomic::AtomicU64;
        static SEEN: AtomicU64 = AtomicU64::new(0);
        fn hook(k: &Knob) {
            SEEN.store(k.get(), Ordering::Relaxed);
        }
        static A: Knob = Knob::int("T_B2_MISS_PAR", 1, 1, 16).alias("miss_par").hook(hook);
        let knobs: [&'static Knob; 1] = [&A];
        let mut warned = HashSet::new();
        let read = |_: &str| None;
        let p = pass(&knobs, &parse_file("miss_par=4\n"), &env_of(&[("T_B2_MISS_PAR", "2")]), &read, &mut warned);
        assert_eq!((A.get(), A.source()), (4, Source::File));
        assert!(p.warnings.is_empty(), "the alias is a known key: {:?}", p.warnings);
        assert_eq!(p.hooks.len(), 1, "first resolution runs the hook");
        p.hooks[0].hook.unwrap()(p.hooks[0]);
        assert_eq!(SEEN.load(Ordering::Relaxed), 4);
        // the full name wins over the alias; a change runs the hook again
        let p = pass(&knobs, &parse_file("miss_par=4\nT_B2_MISS_PAR=8\n"), &env_of(&[]), &read, &mut warned);
        assert_eq!(A.get(), 8);
        assert_eq!(p.hooks.len(), 1);
        // no change, no hook
        assert!(pass(&knobs, &parse_file("T_B2_MISS_PAR=8\n"), &env_of(&[]), &read, &mut warned).hooks.is_empty());
    }

    #[test]
    fn a_first_resolution_skips_an_empty_source() {
        // The hub starts while /dev/shm/lm_prefill.txt is empty (created, or
        // mid-truncating-write): the env decides, not the default.
        static L: Knob = Knob::flag("T_LM_PREFILL", false).legacy("T_LM_FILE");
        let knobs: [&'static Knob; 1] = [&L];
        let env = env_of(&[("T_LM_PREFILL", "1"), ("T_LM_FILE", "/shm/lm")]);
        let read = |p: &str| if p == "/shm/lm" { Some(" \n".to_string()) } else { None };
        let p = pass(&knobs, &BTreeMap::new(), &env, &read, &mut HashSet::new());
        assert_eq!((L.on(), L.source()), (true, Source::Env));
        assert!(p.warnings.is_empty());
    }

    #[test]
    fn set_and_choice_and_real() {
        static C: Knob = Knob::choice("T_HEAD", 0, &[&["1", "on"], &["0", "off"], &["check"]]).live();
        static R: Knob = Knob::real("T_LAMBDA", 0.1, 0.0, 1.0);
        assert_eq!(C.pick(), 0);
        assert!(C.set("check"));
        assert_eq!((C.pick(), C.source(), C.show()), (2, Source::Set, "check".to_string()));
        assert!(!C.set("nope"));
        assert_eq!(R.f64(), 0.1);
        assert!(R.set("0.25"));
        assert_eq!(R.f64(), 0.25);
    }
}
