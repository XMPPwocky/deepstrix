//! The arena stage-graph CONTEXT (docs/v41/GRAPH_KEYS_DESIGN.md 2.1-2.3).
//!
//! One pointer-only device struct holds every per-layer / per-lane operand a captured
//! arena stage reads. Before a stage replays or captures, the host writes the entry of
//! its (lane, layer) into ONE fixed device slot with `arena_ctx_store` (a one-workgroup
//! kernel taking the entry BY VALUE: stream-ordered on the stages' own queue, nothing
//! for the host to keep alive). `_ind` kernel twins read their operands from the slot,
//! so one graph per (stage, rows) serves every layer and every lane.
//!
//! Layout mirrors `kernels/arena_ctx.inc` (`ArenaCtx`): 32 pointers, a sequence number
//! and the canary log pointer (design 2.8).
use color_eyre::eyre::{self, eyre};
use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use v4flash_hip::{launch_kernel, DeviceBuffer, GraphExec, LaunchConfig, Module, Stream};

const ARENA_CTX_GFX1201: &[u8] = include_bytes!(env!("KERNEL_ARENA_CTX_GFX1201"));
const ARENA_CTX_GFX1151: &[u8] = include_bytes!(env!("KERNEL_ARENA_CTX_GFX1151"));

/// Context slots (`ARENA_CTX_SLOTS` in arena_ctx.inc).
pub const ARENA_CTX_SLOTS: usize = 32;

/// u64 words of one `ArenaCtx` (`ARENA_CTX_WORDS`): the slot buffer's length.
pub const ARENA_CTX_WORDS: usize = ARENA_CTX_SLOTS + 2;

/// The context entry, by value (`#[repr(C)]` = `struct ArenaCtx` in arena_ctx.inc).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaCtx {
    /// Device addresses, by slot (0 = unused).
    pub p: [u64; ARENA_CTX_SLOTS],
    pub seq: u64,
    /// Device address of the canary log (`ArenaCanaryLog`), 0 = none.
    pub log: u64,
}

// `arena_ctx_store`'s by-value kernel argument (static_asserts in arena_ctx.inc mirror this).
const _: () = assert!(std::mem::size_of::<ArenaCtx>() == 8 * ARENA_CTX_WORDS);

impl Default for ArenaCtx {
    fn default() -> Self {
        Self { p: [0; ARENA_CTX_SLOTS], seq: 0, log: 0 }
    }
}

/// Most operands an `_ind` twin resolves through the context.
pub const IND_MAX_OPERANDS: usize = 16;

/// How an `_ind` twin finds its operands (design 2.3): the device address of the context
/// slot buffer, and per operand the slot it reads (`None` = direct). An indirect operand's
/// pointer argument is the ADDRESS of its slot (`ctx + 8 x slot`: process-static, safe to
/// bake into a graph) and its `mask` bit tells the kernel to dereference it.
///
/// `canary` (design 2.8) is the context slot's address when the kernel should log the `seq` it
/// read (0 = off, the production value), `tag` the stage id. A non-zero `canary` selects the
/// `_canary` twin symbols; the production twins carry no canary code (a record after the body
/// reshaped it: Step 0 run 2, gemv b=8 +9.8%).
#[derive(Clone, Copy, Debug)]
pub struct Ind {
    pub ctx: u64,
    pub slots: [Option<u8>; IND_MAX_OPERANDS],
    pub canary: u64,
    pub tag: u32,
}

impl Ind {
    pub fn new(ctx: u64) -> Self {
        Self { ctx, slots: [None; IND_MAX_OPERANDS], canary: 0, tag: 0 }
    }

    /// Log (seq, `tag`) per launch into the log the context entry names.
    pub fn with_canary(mut self, tag: u16) -> Self {
        self.canary = self.ctx;
        self.tag = tag as u32;
        self
    }

    /// Operand `operand` reads context slot `slot`.
    pub fn with(mut self, operand: usize, slot: usize) -> Self {
        assert!(operand < IND_MAX_OPERANDS && slot < ARENA_CTX_SLOTS);
        self.slots[operand] = Some(slot as u8);
        self
    }

    /// The device address of context slot `slot` (a value slot's argument, e.g. the rope pairs).
    pub fn slot_addr(&self, slot: usize) -> u64 {
        assert!(slot < ARENA_CTX_SLOTS);
        self.ctx + 8 * slot as u64
    }

    /// The twin symbol for `base` (e.g. "q8_0_gemv_bpack_tB4"): `base_ind`, or `base_ind_canary`
    /// when the canary is on (production twins carry no canary code, design 2.8).
    pub fn symbol(&self, base: &str) -> String {
        if self.canary != 0 { format!("{base}_ind_canary") } else { format!("{base}_ind") }
    }

    /// The kernel's `ind_mask`.
    pub fn mask(&self) -> u32 {
        self.slots.iter().enumerate().filter(|(_, s)| s.is_some()).fold(0, |m, (i, _)| m | (1 << i))
    }

    /// The pointer argument of operand `operand`: its slot's address when indirect, else
    /// `direct`.
    pub fn ptr(&self, operand: usize, direct: u64) -> u64 {
        match self.slots[operand] {
            Some(slot) => self.ctx + 8 * slot as u64,
            None => direct,
        }
    }
}

/// The context kernels (`kernels/arena_ctx.hip`).
pub struct ArenaCtxKernels {
    module: Module,
}

impl ArenaCtxKernels {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            ARENA_CTX_GFX1201
        } else if arch.starts_with("gfx1151") {
            ARENA_CTX_GFX1151
        } else {
            return Err(eyre!("unsupported arch for arena_ctx: {arch}"));
        };
        Ok(Self { module: Module::load_data(image)? })
    }

    /// Enqueue the write of `entry` into the context slot at device address `dst`.
    pub fn store_raw(&self, stream: &Stream, entry: &ArenaCtx, dst: u64) -> eyre::Result<()> {
        let function = self.module.get_function("arena_ctx_store")?;
        let cfg = LaunchConfig { grid: (1, 1, 1), block: (32, 1, 1), shared_mem_bytes: 0 };
        let e = *entry;
        launch_kernel!(function, cfg, stream, [e, dst])
    }

    /// Enqueue the write of `entry` into the device slot `dst` (one `ArenaCtx`).
    pub fn store(&self, stream: &Stream, entry: &ArenaCtx, dst: &mut DeviceBuffer<u64>) -> eyre::Result<()> {
        if dst.len() < ARENA_CTX_WORDS {
            return Err(eyre!("arena_ctx store: slot buffer of {} u64, need {}", dst.len(), ARENA_CTX_WORDS));
        }
        let function = self.module.get_function("arena_ctx_store")?;
        let cfg = LaunchConfig { grid: (1, 1, 1), block: (32, 1, 1), shared_mem_bytes: 0 };
        let e = *entry;
        launch_kernel!(function, cfg, stream, [e, dst.raw()])
    }

    /// A do-nothing launch (Step 0's empty-launch baseline).
    pub fn nop(&self, stream: &Stream, sink: &mut DeviceBuffer<u32>) -> eyre::Result<()> {
        let function = self.module.get_function("arena_ctx_nop")?;
        let cfg = LaunchConfig { grid: (1, 1, 1), block: (32, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(function, cfg, stream, [sink.raw(), 0u32])
    }
}

/// The context slots of a lane-layer entry (design 2.1, 2.11; inventory 2026-10-05: the four
/// multi-node stages read ten per-layer weights, three per-lane buffers, the layer's rope values).
pub mod slot {
    pub const Q_A: usize = 0;
    pub const Q_A_NORM: usize = 1;
    pub const Q_B: usize = 2;
    pub const KV: usize = 3;
    pub const KV_NORM: usize = 4;
    pub const WO_A: usize = 5;
    pub const WO_B: usize = 6;
    pub const SH_GATE: usize = 7;
    pub const SH_UP: usize = 8;
    pub const SH_DOWN: usize = 9;
    /// per lane
    pub const POS: usize = 10;
    pub const FFN_IN: usize = 11;
    pub const FFN_SHARED: usize = 12;
    /// value slots: the rope arguments as f32 pairs (`RopeTail::arena_ctx_rope_words`)
    pub const ROPE: [usize; 3] = [20, 21, 22];
}

// ---- process-static operand ranges (design 2.5's whitelist, 2.7's fingerprint) -------------------

static STATIC_RANGES: RwLock<Vec<(u64, u64)>> = RwLock::new(Vec::new());
static RANGES_GENERATION: AtomicU64 = AtomicU64::new(0);

/// Register the process-static device ranges (base, bytes) a `stage_b` capture may bake (`sd.*`,
/// engine scratch). Replaces the set; a CHANGED set bumps the generation, which makes every cached
/// stage graph stale (they bake `sd` pointers: 2.7).
pub fn register_static_ranges(mut ranges: Vec<(u64, u64)>) {
    ranges.retain(|r| r.1 > 0);
    ranges.sort_unstable();
    let mut cur = STATIC_RANGES.write().unwrap();
    if *cur != ranges {
        *cur = ranges;
        RANGES_GENERATION.fetch_add(1, Ordering::SeqCst);
    }
}

/// The registered set's generation (0 = nothing registered).
pub fn ranges_generation() -> u64 {
    RANGES_GENERATION.load(Ordering::SeqCst)
}

/// Whether device address `ptr` lies inside a registered process-static range.
pub fn is_static(ptr: u64) -> bool {
    let r = STATIC_RANGES.read().unwrap();
    // ranges are sorted by base: the last base <= ptr
    let i = r.partition_point(|&(base, _)| base <= ptr);
    i > 0 && ptr < r[i - 1].0 + r[i - 1].1
}

/// Vet the next launch when every pointer in `direct` is process-static (design 2.5 rule (b)): a
/// direct launch inside a `stage_b` capture whose operands are all `sd` may be baked. Returns
/// whether it vetted.
pub fn vet_static(direct: &[u64]) -> bool {
    let ok = direct.iter().all(|&p| p == 0 || is_static(p));
    if ok {
        v4flash_hip::vet_next_launch();
    }
    ok
}

/// Vet an `_ind` twin's launch: every operand NOT read through the context is process-static.
pub fn vet_ind(ind: &Ind, direct: &[u64]) -> bool {
    let ok = direct.iter().enumerate().all(|(i, &p)| ind.slots[i].is_some() || p == 0 || is_static(p));
    if ok {
        v4flash_hip::vet_next_launch();
    }
    ok
}

/// How a stage body launches (design 2.4): `Direct` (legacy graphs, uncaptured) or through the
/// arena context (`stage_b`; the `Ind` carries the slot's address, canary and stage tag, no slots).
#[derive(Clone, Copy, Debug)]
pub enum StageSrc {
    Direct,
    Ctx(Ind),
}

impl StageSrc {
    /// The context template (no operand slots set) when the body reads through the context.
    pub fn ind(&self) -> Option<Ind> {
        match self {
            StageSrc::Direct => None,
            StageSrc::Ctx(i) => Some(*i),
        }
    }
}

/// The arena graph mode of a step (`V41_MS_GRAPH_KEYS`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphMode {
    Legacy = 0,
    StageB = 1,
}

/// Per stage name: (direct, replayed, captured stage_b, captured legacy, tainted, uncaptured).
pub type StageCounters = [u64; 6];
pub const C_DIRECT: usize = 0;
pub const C_REPLAYED: usize = 1;
pub const C_CAPTURED_B: usize = 2;
pub const C_CAPTURED_LEGACY: usize = 3;
pub const C_TAINTED: usize = 4;
pub const C_UNCAPTURED: usize = 5;

/// The engine's `stage_b` state (design 2.2-2.8, 2.11): the context slot on the dGPU, the host
/// shadow of its last enqueued entry, the legacy-marked (stage, rows, topo) set, the tainted
/// executables (kept: one per legacy mark, so at most ~32 per process), the topology classes, the
/// counters the gates read.
pub struct StageB {
    pub kernels: ArenaCtxKernels,
    /// The (stage, rows, topo)-keyed graphs (a cache of their own: no key collides with the
    /// legacy (stage, layer, rows, lane) keys; room decisions follow `dgpu_graphs`).
    pub graphs: super::graph_cache::GraphCache,
    /// one `ArenaCtx` on the dGPU
    slot: DeviceBuffer<u64>,
    shadow: Mutex<Option<ArenaCtx>>,
    seq: AtomicU64,
    mode: AtomicU8,
    legacy: Mutex<HashSet<(&'static str, u32, u8)>>,
    retired: Mutex<Vec<Arc<GraphExec>>>,
    topo: Mutex<Vec<u64>>,
    /// Per layer: the static part of its entries (weights, rope values) and its topology class,
    /// computed on first use (weights do not change after load).
    per_layer: Mutex<Vec<Option<(ArenaCtx, u8)>>>,
    counters: Mutex<BTreeMap<&'static str, StageCounters>>,
    /// (carrier writes, standalone writes)
    writes: [AtomicU64; 2],
    /// `ranges_generation()` the cached stage graphs were captured under
    generation: AtomicU64,
    /// `V41_MS_CTX_CHECK`: the canary log the `_ind_canary` twins append to, and the (seq, stage
    /// tag) of every context-reading graph launch in enqueue order (design 2.8).
    canary: Option<(Mutex<CanaryLog>, Mutex<Vec<(u64, u16)>>)>,
}

impl StageB {
    /// On the CURRENT device (the dGPU).
    pub fn new(device: i32, arch: &str) -> eyre::Result<Self> {
        let mut slot = DeviceBuffer::<u64>::new(device, ARENA_CTX_WORDS)?;
        slot.fill_zero()?;
        Ok(Self {
            kernels: ArenaCtxKernels::for_arch(arch)?,
            graphs: super::graph_cache::GraphCache::new(),
            slot,
            shadow: Mutex::new(None),
            seq: AtomicU64::new(0),
            mode: AtomicU8::new(GraphMode::Legacy as u8),
            legacy: Mutex::new(HashSet::new()),
            retired: Mutex::new(Vec::new()),
            topo: Mutex::new(Vec::new()),
            per_layer: Mutex::new(Vec::new()),
            counters: Mutex::new(BTreeMap::new()),
            writes: [AtomicU64::new(0), AtomicU64::new(0)],
            generation: AtomicU64::new(0),
            canary: if crate::knobs::MS_CTX_CHECK.on() {
                Some((Mutex::new(CanaryLog::new(device, 1 << 16)?), Mutex::new(Vec::new())))
            } else {
                None
            },
        })
    }

    /// The canary log's device address (0 = off): every entry carries it (`ArenaCtx::log`).
    pub fn canary_log_addr(&self) -> u64 {
        self.canary.as_ref().map_or(0, |(l, _)| l.lock().unwrap().buf.raw() as u64)
    }

    /// A context-reading graph launch was enqueued for stage `tag` under the current entry.
    pub fn canary_expect(&self, tag: u16) {
        if let Some((_, exp)) = &self.canary {
            let seq = self.shadow.lock().unwrap().map_or(0, |e| e.seq);
            exp.lock().unwrap().push((seq, tag));
        }
    }

    /// Check the canary (design 2.8) after the device finished every expected launch (the caller
    /// synchronized): the records' (seq, tag) sequence, consecutive repeats collapsed (one graph
    /// appends one record per `_ind` launch), must equal the expected launches' sequence collapsed
    /// the same way. Resets the log and the expectations. `None` = the canary is off; else
    /// (records, expected launches, the first mismatch if any).
    pub fn canary_check(&self) -> eyre::Result<Option<(usize, usize, Option<String>)>> {
        let Some((log, exp)) = &self.canary else { return Ok(None) };
        let mut log = log.lock().unwrap();
        let mut exp = exp.lock().unwrap();
        let (cursor, recs) = log.read()?;
        let collapse = |v: &mut Vec<(u64, u16)>| v.dedup();
        let mut got: Vec<(u64, u16)> = recs.iter().map(|r| (r.seq, r.tag)).collect();
        let mut want = exp.clone();
        collapse(&mut got);
        collapse(&mut want);
        let mismatch = if cursor > log.cap {
            Some(format!("log overflow: {cursor} records > {}", log.cap))
        } else if got != want {
            let i = got.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
            Some(format!(
                "at launch group {i}: read {:?}, expected {:?} ({} groups read, {} expected)",
                got.get(i),
                want.get(i),
                got.len(),
                want.len()
            ))
        } else {
            None
        };
        let n = (recs.len(), exp.len(), mismatch);
        exp.clear();
        log.reset()?;
        Ok(Some(n))
    }

    /// The context slot's device address (process-static).
    pub fn slot_addr(&self) -> u64 {
        self.slot.raw() as u64
    }

    /// Read `V41_MS_GRAPH_KEYS` for this step (design 2.4: once per step, carried down) and forget
    /// the shadow (the next lane-layer writes its entry for sure).
    pub fn begin_step(&self) -> GraphMode {
        let mode = if crate::knobs::MS_GRAPH_KEYS.pick() == 1 && cfg!(feature = "v41") { GraphMode::StageB } else { GraphMode::Legacy };
        self.mode.store(mode as u8, Ordering::Relaxed);
        *self.shadow.lock().unwrap() = None;
        mode
    }

    pub fn mode(&self) -> GraphMode {
        if self.mode.load(Ordering::Relaxed) == GraphMode::StageB as u8 { GraphMode::StageB } else { GraphMode::Legacy }
    }

    /// The `Ind` template of this process (slot address; canary per `V41_MS_CTX_CHECK`).
    pub fn ind(&self, tag: u16) -> Ind {
        let i = Ind::new(self.slot_addr());
        if crate::knobs::MS_CTX_CHECK.on() { i.with_canary(tag) } else { i }
    }

    fn same_payload(a: &ArenaCtx, b: &ArenaCtx) -> bool {
        a.p == b.p && a.log == b.log
    }

    /// Make the slot hold `entry` on `stream` (design 2.2): compared with the shadow of the last
    /// enqueued entry (payload, not `seq`); on a change, stamps a new `seq` and enqueues the
    /// standalone write. Must not run inside a capture (the caller's precondition).
    pub fn ensure(&self, stream: &Stream, entry: &ArenaCtx) -> eyre::Result<()> {
        let mut sh = self.shadow.lock().unwrap();
        if sh.as_ref().is_some_and(|s| Self::same_payload(s, entry)) {
            return Ok(());
        }
        let mut e = *entry;
        e.seq = self.seq.fetch_add(1, Ordering::Relaxed) + 1;
        self.kernels.store_raw(stream, &e, self.slot_addr())?;
        *sh = Some(e);
        self.writes[1].fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// The entry a CARRIER launch writes (design 2.11 R2): stamped, and the shadow set as if
    /// `ensure` had written it. `None` when the slot already holds this payload (no carrier needed).
    pub fn carry(&self, entry: &ArenaCtx) -> Option<ArenaCtx> {
        let mut sh = self.shadow.lock().unwrap();
        if sh.as_ref().is_some_and(|s| Self::same_payload(s, entry)) {
            return None;
        }
        let mut e = *entry;
        e.seq = self.seq.fetch_add(1, Ordering::Relaxed) + 1;
        *sh = Some(e);
        self.writes[0].fetch_add(1, Ordering::Relaxed);
        Some(e)
    }

    /// Forget the shadow (an abnormal end: whatever was enqueued is not trusted).
    pub fn forget(&self) {
        *self.shadow.lock().unwrap() = None;
    }

    /// The canary tag of a stage (design 2.8): which stage a record came from.
    pub fn tag_of(stage: &str) -> u16 {
        match stage {
            "g.q_chain" => 1,
            "g.kv_chain" => 2,
            "g.output_proj" => 3,
            "g.shared_expert" => 4,
            _ => 15,
        }
    }

    pub fn is_legacy(&self, stage: &'static str, b: u32, topo: u8) -> bool {
        self.legacy.lock().unwrap().contains(&(stage, b, topo))
    }

    /// A tainted capture (design 2.5): keep its executable alive (its one launch may be queued)
    /// and send (stage, b, topo) to legacy keys from now on.
    pub fn taint(&self, stage: &'static str, b: u32, topo: u8, exec: Arc<GraphExec>) {
        self.legacy.lock().unwrap().insert((stage, b, topo));
        self.retired.lock().unwrap().push(exec);
    }

    /// Layer `layer`'s static entry part and topology class, computed by `make` on first use.
    pub fn layer_part(&self, layer: usize, make: impl FnOnce() -> (ArenaCtx, u64)) -> (ArenaCtx, u8) {
        let mut v = self.per_layer.lock().unwrap();
        if v.len() <= layer {
            v.resize(layer + 1, None);
        }
        if let Some(p) = v[layer] {
            return p;
        }
        let (e, fp) = make();
        let p = (e, self.topo_class(fp));
        v[layer] = Some(p);
        p
    }

    /// The topology class of a fingerprint of a layer's topology-relevant facts (design 2.6).
    pub fn topo_class(&self, fingerprint: u64) -> u8 {
        let mut t = self.topo.lock().unwrap();
        if let Some(i) = t.iter().position(|&f| f == fingerprint) {
            return i as u8;
        }
        t.push(fingerprint);
        (t.len() - 1).min(255) as u8
    }

    pub fn count(&self, stage: &'static str, what: usize) {
        self.counters.lock().unwrap().entry(stage).or_insert([0; 6])[what] += 1;
    }

    /// (per-stage counters, carrier writes, standalone writes)
    pub fn counters(&self) -> (BTreeMap<&'static str, StageCounters>, u64, u64) {
        (
            self.counters.lock().unwrap().clone(),
            self.writes[0].load(Ordering::Relaxed),
            self.writes[1].load(Ordering::Relaxed),
        )
    }

    pub fn reset_counters(&self) {
        self.counters.lock().unwrap().clear();
        self.writes[0].store(0, Ordering::Relaxed);
        self.writes[1].store(0, Ordering::Relaxed);
    }

    /// Design 2.7: the cached stage graphs bake `sd`; a changed registered set (a different `sd`)
    /// makes them stale. Returns true when the caller must synchronize the device and clear the
    /// graph caches (stage_b AND legacy) before any replay.
    pub fn generation_changed(&self) -> bool {
        let g = ranges_generation();
        self.generation.swap(g, Ordering::SeqCst) != g
    }
}

/// A device canary log (`ArenaCanaryLog`): word 0 = cursor (low u32) | cap (high u32), then
/// `cap` records of two words: `seq << 16 | tag`, the XOR of the launch's resolved operand
/// pointers.
pub struct CanaryLog {
    pub buf: DeviceBuffer<u64>,
    pub cap: u32,
}

/// One canary record: what an `_ind` launch read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CanaryRec {
    pub seq: u64,
    pub tag: u16,
    /// XOR of the operand pointers the launch resolved.
    pub ptr_xor: u64,
}

impl CanaryLog {
    pub fn new(device: i32, cap: u32) -> eyre::Result<Self> {
        let buf = DeviceBuffer::<u64>::new(device, 1 + 2 * cap as usize)?;
        let mut log = Self { buf, cap };
        log.reset()?;
        Ok(log)
    }

    /// Reset the cursor (synchronous; between steps).
    pub fn reset(&mut self) -> eyre::Result<()> {
        self.buf.slice_view_mut(0, 1).copy_from_host(&[(self.cap as u64) << 32])
    }

    /// (launches logged, including any past `cap`; the records written), synchronous.
    pub fn read(&self) -> eyre::Result<(u32, Vec<CanaryRec>)> {
        let mut host = vec![0u64; 1 + 2 * self.cap as usize];
        self.buf.copy_to_host(&mut host)?;
        let cursor = host[0] as u32;
        let n = cursor.min(self.cap) as usize;
        let recs = host[1..1 + 2 * n]
            .chunks(2)
            .map(|r| CanaryRec { seq: r[0] >> 16, tag: (r[0] & 0xffff) as u16, ptr_xor: r[1] })
            .collect();
        Ok((cursor, recs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ind_mask_and_slot_addresses() {
        let ctx = 0x7f00_0000_1000u64;
        let ind = Ind::new(ctx).with(0, 0).with(3, 27).with(12, 31);
        assert_eq!(ind.mask(), 1 | 1 << 3 | 1 << 12);
        assert_eq!(ind.ptr(0, 0xdead), ctx);
        assert_eq!(ind.ptr(3, 0xdead), ctx + 8 * 27);
        assert_eq!(ind.ptr(12, 0xdead), ctx + 8 * 31);
        assert_eq!(ind.ptr(1, 0xbeef), 0xbeef, "direct operands pass through");
        assert_eq!((ind.canary, ind.tag), (0, 0), "canary off by default (production)");
        let c = ind.with_canary(9);
        assert_eq!((c.canary, c.tag), (ctx, 9));
        // seq / log sit right after the slots (arena_ctx.inc static_asserts the same).
        assert_eq!(std::mem::offset_of!(ArenaCtx, seq), 8 * ARENA_CTX_SLOTS);
        assert_eq!(std::mem::offset_of!(ArenaCtx, log), 8 * ARENA_CTX_SLOTS + 8);
        assert_eq!(c.symbol("q8_0_gemv_bpack_tB4"), "q8_0_gemv_bpack_tB4_ind_canary");
        assert_eq!(ind.symbol("q8_0_gemv_bpack_tB4"), "q8_0_gemv_bpack_tB4_ind");
    }

    #[test]
    fn static_ranges_contain_and_generation() {
        // One test (the registry is process-global).
        let g0 = ranges_generation();
        register_static_ranges(vec![(0x2000, 0x100), (0x1000, 0x10), (0x9000, 0)]);
        let g1 = ranges_generation();
        assert_eq!(g1, g0 + 1, "a new set bumps the generation");
        assert!(is_static(0x1000) && is_static(0x100f) && !is_static(0x1010));
        assert!(is_static(0x2000) && is_static(0x20ff) && !is_static(0x2100) && !is_static(0xfff));
        assert!(!is_static(0x9000), "empty ranges are dropped");
        register_static_ranges(vec![(0x1000, 0x10), (0x2000, 0x100)]);
        assert_eq!(ranges_generation(), g1, "the same set (any order) keeps the generation");
        let ind = Ind::new(0x5000).with(1, 3);
        assert!(vet_ind(&ind, &[0x1004, 0xdead_0000, 0x2010]), "operand 1 is read through the context");
        assert!(!vet_ind(&ind, &[0xdead_0000, 0x1004]), "a direct operand outside the set");
        assert!(vet_static(&[0x1000, 0, 0x20ff]) && !vet_static(&[0x1000, 0x3000]));
        // consume the vetted marks this test set (no launch follows)
        let _ = v4flash_hip::launch_audit();
        register_static_ranges(vec![(0x1000, 0x20)]);
        assert_eq!(ranges_generation(), g1 + 1);
    }
}
