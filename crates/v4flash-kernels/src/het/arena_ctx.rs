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
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

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
/// read (0 = off, the production value: one uniform branch in every twin), `tag` the stage id.
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

    /// Enqueue the write of `entry` into the device slot `dst` (one `ArenaCtx`).
    pub fn store(&self, stream: &Stream, entry: &ArenaCtx, dst: &mut DeviceBuffer<u64>) -> eyre::Result<()> {
        if dst.len() < ARENA_CTX_WORDS {
            return Err(eyre!("arena_ctx store: slot buffer of {} u64, need {}", dst.len(), ARENA_CTX_WORDS));
        }
        let function = self.module.get_function("arena_ctx_store")?;
        let cfg = LaunchConfig { grid: (1, 1, 1), block: (64, 1, 1), shared_mem_bytes: 0 };
        let e = *entry;
        launch_kernel!(function, cfg, stream, [e, dst.raw()])
    }

    /// A do-nothing launch (Step 0's empty-launch baseline).
    pub fn nop(&self, stream: &Stream, sink: &mut DeviceBuffer<u32>) -> eyre::Result<()> {
        let function = self.module.get_function("arena_ctx_nop")?;
        let cfg = LaunchConfig { grid: (1, 1, 1), block: (64, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(function, cfg, stream, [sink.raw(), 0u32])
    }
}

/// A device canary log (`ArenaCanaryLog`): word 0 = cursor (low u32) | cap (high u32), then
/// `cap` records of `seq << 16 | tag`.
pub struct CanaryLog {
    pub buf: DeviceBuffer<u64>,
    pub cap: u32,
}

impl CanaryLog {
    pub fn new(device: i32, cap: u32) -> eyre::Result<Self> {
        let mut buf = DeviceBuffer::<u64>::new(device, 1 + cap as usize)?;
        buf.copy_from_host(&[(cap as u64) << 32])?;
        Ok(Self { buf, cap })
    }

    /// Reset the cursor (synchronous; between steps).
    pub fn reset(&mut self) -> eyre::Result<()> {
        self.buf.copy_from_host(&[(self.cap as u64) << 32])
    }

    /// (launches logged, including any past `cap`; the records written), synchronous.
    pub fn read(&self) -> eyre::Result<(u32, Vec<(u64, u16)>)> {
        let mut host = vec![0u64; 1 + self.cap as usize];
        self.buf.copy_to_host(&mut host)?;
        let cursor = host[0] as u32;
        let n = cursor.min(self.cap) as usize;
        Ok((cursor, host[1..1 + n].iter().map(|&r| (r >> 16, (r & 0xffff) as u16)).collect()))
    }
}
