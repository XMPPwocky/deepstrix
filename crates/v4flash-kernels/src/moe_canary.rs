//! MoE LAUNCH-ARG CANARY (HIP 700 instrumentation, owner-approved 2026-10-19).
//!
//! 10-18 23:50:58 UTC the iGPU queue died with `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` in
//! `mxfp4_pair_matvec_fused_swiglu_wmma` (prefill MoE gate+up). Every in-kernel index is 16-bit bounded, so a POINTER or
//! SCALAR argument was wrong. The MXFP4 MoE launchers (gate+up: kwide / WMMA / decode; down: kwide2 / small-b / dn3 /
//! WMMA) now validate their arguments before launching ([`check`]):
//!
//! * every pointer argument lies inside a live registered allocation of the launch device (`v4flash_hip::ptr_registry`);
//! * a weight base inside a registered EXPERT POOL ([`register_pool`], captured when the pager pool / box-2 shard was
//!   built) sits in the right role's buffer, on a slot boundary, with the pool's bytes-per-expert, the view inside the
//!   pool, gate and up on the SAME slot, and the pool's shape (`n_rows`, `n_blocks`);
//! * `n_work_items <= work_items.len()` (a garbage host read-back of the count otherwise sends grid.y past the work
//!   items and the kernel reads group ids from beyond the buffer), `chunk_size` in range, and the grid consistent with
//!   rows / work items.
//!
//! A failure is an error `MoE launch arg corrupted: <arg> = <value>, expected <...> (<kernel>, layer L, lane N, call
//! site)`, never a clamp. Layer / lane come from the thread's `v4flash_hip::ring::scope`. The validation itself
//! ([`validate`]) is pure, so it is unit-tested on a CPU.
use std::panic::Location;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

use color_eyre::eyre::{self, eyre};

use crate::model_weights::RoutedExpertWeights;

/// Q8_K elements per block (`n_blocks = k / QK_K`).
const QK_K: u64 = 256;

/// Which expert role a weight argument is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Gate,
    Up,
    Down,
}

/// One role's buffer of a registered expert pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RolePool {
    pub base: usize,
    pub bytes: usize,
    /// Bytes per expert slot.
    pub bpe: usize,
    /// Output rows of one expert (`n_rows` of its launch).
    pub rows: u32,
    /// Q8_K input blocks (`n_blocks` / `n_blocks_in` of its launch).
    pub n_blocks: u32,
}

/// A registered expert pool: the base pointers captured when it was built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolBases {
    pub id: u64,
    pub name: &'static str,
    pub device: i32,
    pub n_slots: u32,
    pub gate: RolePool,
    pub up: RolePool,
    pub down: RolePool,
}

impl PoolBases {
    fn role(&self, r: Role) -> &RolePool {
        match r {
            Role::Gate => &self.gate,
            Role::Up => &self.up,
            Role::Down => &self.down,
        }
    }

    /// The role buffer containing `addr`, if any.
    fn find(&self, addr: usize) -> Option<(Role, &RolePool)> {
        [Role::Gate, Role::Up, Role::Down]
            .into_iter()
            .map(|r| (r, self.role(r)))
            .find(|(_, p)| addr >= p.base && addr < p.base + p.bytes)
    }
}

static POOLS: RwLock<Vec<PoolBases>> = RwLock::new(Vec::new());
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Deregisters its pool on drop (keep it beside the pool's buffers).
#[derive(Debug)]
pub struct PoolGuard {
    id: u64,
}

impl Drop for PoolGuard {
    fn drop(&mut self) {
        POOLS.write().unwrap_or_else(|p| p.into_inner()).retain(|p| p.id != self.id);
    }
}

/// Capture `routed`'s base pointers and shapes as an expert pool on `device` (the box-1 pager pool, box 2's shard). The
/// pool's layout must be `n_slots` slots of `*_bytes_per_expert` per role, and each role's `shape` `[slots, rows, k]`.
pub fn register_pool(name: &'static str, device: i32, routed: &RoutedExpertWeights) -> eyre::Result<PoolGuard> {
    let role = |what: &str, w: &crate::weights::DeviceWeight, bpe: usize| -> eyre::Result<RolePool> {
        let bytes = w.buffer.byte_len();
        if bpe == 0 || bytes != routed.n_slots as usize * bpe {
            return Err(eyre!("MoE canary: pool {name} {what}: {bytes} B != {} slots x {bpe} B", routed.n_slots));
        }
        let (rows, k) = match w.shape.as_slice() {
            [_, rows, k] if *k % QK_K == 0 => (*rows as u32, (*k / QK_K) as u32),
            s => return Err(eyre!("MoE canary: pool {name} {what}: shape {s:?} is not [slots, rows, k] with k % 256 == 0")),
        };
        Ok(RolePool { base: w.buffer.raw() as usize, bytes, bpe, rows, n_blocks: k })
    };
    let p = PoolBases {
        id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        name,
        device,
        n_slots: routed.n_slots,
        gate: role("gate", &routed.gate, routed.gate_bytes_per_expert)?,
        up: role("up", &routed.up, routed.up_bytes_per_expert)?,
        down: role("down", &routed.down, routed.down_bytes_per_expert)?,
    };
    POOLS.write().unwrap_or_else(|p| p.into_inner()).push(p);
    tracing::info!(
        pool = name, device, slots = p.n_slots, gate = format_args!("{:#x}", p.gate.base), up = format_args!("{:#x}", p.up.base),
        down = format_args!("{:#x}", p.down.base), "MoE canary: expert pool base pointers captured"
    );
    Ok(PoolGuard { id: p.id })
}

/// The registered pools (a copy).
pub fn pools() -> Vec<PoolBases> {
    POOLS.read().unwrap_or_else(|p| p.into_inner()).clone()
}

/// A pointer argument: `bytes` from `ptr` are what the buffer argument covers.
#[derive(Clone, Copy, Debug)]
pub struct Ptr {
    pub name: &'static str,
    pub ptr: usize,
    pub bytes: usize,
}

impl Ptr {
    pub fn of<T>(name: &'static str, b: &v4flash_hip::DeviceBuffer<T>) -> Self {
        Ptr { name, ptr: b.raw() as usize, bytes: b.byte_len() }
    }
}

/// An expert weight argument.
#[derive(Clone, Copy, Debug)]
pub struct Weight {
    pub arg: Ptr,
    pub role: Role,
    /// The launch's bytes-per-expert argument for this role.
    pub bpe: u32,
}

/// Grid shape the launcher used.
#[derive(Clone, Copy, Debug)]
pub enum Grid {
    /// `(ceil(n_rows / rows_per_wg), n_work_items, 1)`.
    RowsByItems { rows_per_wg: u32 },
    /// A persistent grid `(wgs, 1, 1)`, `wgs > 0`.
    Persistent,
}

/// One MXFP4 MoE launch, as the launcher is about to issue it.
#[derive(Clone, Debug)]
pub struct MoeLaunch<'a> {
    pub kernel: &'static str,
    pub device: i32,
    /// Every pointer argument but the weights (an absent optional pointer is left out).
    pub ptrs: &'a [Ptr],
    pub weights: &'a [Weight],
    pub n_rows: u32,
    pub n_blocks: u32,
    pub n_work_items: u32,
    pub work_items_len: usize,
    pub chunk_size: u32,
    pub grid: (u32, u32, u32),
    pub grid_kind: Grid,
}

fn bad(arg: &str, value: impl std::fmt::Display, expected: impl std::fmt::Display) -> String {
    format!("{arg} = {value}, expected {expected}")
}

/// The checks of the module doc, on `l` against `pools` and the liveness oracle `live(name, ptr, bytes)` (the pointer
/// registry in production). `Err` = `"<arg> = <value>, expected <...>"`.
pub fn validate(l: &MoeLaunch<'_>, pools: &[PoolBases], live: &dyn Fn(&str, usize, usize) -> Result<(), String>) -> Result<(), String> {
    if l.n_rows == 0 {
        return Err(bad("n_rows", 0, "> 0"));
    }
    if l.n_blocks == 0 {
        return Err(bad("n_blocks", 0, "> 0"));
    }
    if l.chunk_size == 0 || l.chunk_size > crate::mxfp4_pair::MXFP4_WMMA_MAX_CHUNK {
        return Err(bad("chunk_size", l.chunk_size, format!("1..={}", crate::mxfp4_pair::MXFP4_WMMA_MAX_CHUNK)));
    }
    if l.n_work_items as usize > l.work_items_len {
        return Err(bad("n_work_items", l.n_work_items, format!("<= work_items capacity {}", l.work_items_len)));
    }
    match l.grid_kind {
        Grid::RowsByItems { rows_per_wg } => {
            let want = (l.n_rows.div_ceil(rows_per_wg.max(1)), l.n_work_items, 1);
            if l.grid != want {
                return Err(bad("grid", format!("{:?}", l.grid), format!("{want:?} (n_rows {} / {rows_per_wg} rows per WG, n_work_items)", l.n_rows)));
            }
        }
        Grid::Persistent => {
            if l.grid.0 == 0 || l.grid.1 != 1 || l.grid.2 != 1 {
                return Err(bad("grid", format!("{:?}", l.grid), "(wgs > 0, 1, 1) (persistent)"));
            }
        }
    }
    // Pool-backed weights: role, alignment, extent, shape; gate and up on one slot.
    let mut slot_of: Vec<(Role, usize, usize)> = Vec::new(); // (role, first slot, slots)
    for w in l.weights {
        let a = w.arg;
        if w.bpe == 0 {
            return Err(bad(&format!("{} bytes-per-expert", a.name), 0, "> 0"));
        }
        let Some((p, (role, rp))) = pools.iter().filter(|p| p.device == l.device).find_map(|p| p.find(a.ptr).map(|f| (p, f))) else {
            continue; // resident (non-pool) weights: liveness below
        };
        let at = |what: &str| format!("{what} of pool {} {:?} [{:#x}, +{} B)", p.name, role, rp.base, rp.bytes);
        if role != w.role {
            return Err(bad(a.name, format!("{:#x}", a.ptr), format!("inside the {:?} buffer, but it lies in the {}", w.role, at("buffer"))));
        }
        if w.bpe as usize != rp.bpe {
            return Err(bad(&format!("{} bytes-per-expert", a.name), w.bpe, format!("{} (pool {})", rp.bpe, p.name)));
        }
        let off = a.ptr - rp.base;
        if off % rp.bpe != 0 {
            return Err(bad(a.name, format!("{:#x}", a.ptr), format!("a slot boundary ({:#x} + k x {}) of the {}", rp.base, rp.bpe, at("buffer"))));
        }
        if a.bytes % rp.bpe != 0 || off + a.bytes > rp.bytes {
            return Err(bad(
                &format!("{} view", a.name),
                format!("[{:#x}, +{} B)", a.ptr, a.bytes),
                format!("whole slots inside the {}", at("buffer")),
            ));
        }
        if l.n_rows != rp.rows {
            return Err(bad("n_rows", l.n_rows, format!("{} (rows of pool {} {:?})", rp.rows, p.name, role)));
        }
        if l.n_blocks != rp.n_blocks {
            return Err(bad("n_blocks", l.n_blocks, format!("{} (k / 256 of pool {} {:?})", rp.n_blocks, p.name, role)));
        }
        slot_of.push((role, off / rp.bpe, a.bytes / rp.bpe));
    }
    if let (Some(g), Some(u)) = (slot_of.iter().find(|s| s.0 == Role::Gate), slot_of.iter().find(|s| s.0 == Role::Up)) {
        if (g.1, g.2) != (u.1, u.2) {
            return Err(bad("up_w_base slot window", format!("[{}, +{})", u.1, u.2), format!("the gate's [{}, +{})", g.1, g.2)));
        }
    }
    for a in l.ptrs.iter().chain(l.weights.iter().map(|w| &w.arg)) {
        live(a.name, a.ptr, a.bytes).map_err(|e| bad(a.name, format!("{:#x}", a.ptr), format!("inside a live dev{} allocation ({e})", l.device)))?;
    }
    Ok(())
}

/// [`validate`] against the registered pools and the pointer registry; the error names the kernel, the thread's
/// (layer, lane) and the call site.
pub fn check(l: &MoeLaunch<'_>, loc: &'static Location<'static>) -> eyre::Result<()> {
    let pools = POOLS.read().unwrap_or_else(|p| p.into_inner());
    let live = |name: &str, ptr: usize, bytes: usize| {
        v4flash_hip::ptr_registry::check_device(name, ptr, bytes, l.device, loc).map_err(|e| e.to_string())
    };
    validate(l, &pools, &live).map_err(|e| {
        let (layer, lane) = v4flash_hip::ring::ctx();
        eyre!(
            "MoE launch arg corrupted: {e} ({}, layer {layer}, lane {lane}, call site {}:{})",
            l.kernel,
            loc.file(),
            loc.line()
        )
    })
}

/// The CALL-SITE half of the canary (forward_prefill's iGPU MoE): the expert weights a layer is about to launch on.
/// `n_slots x bytes-per-expert` must fit each role's buffer; with `pool = Some((name, slot0))` (a view of the pager
/// pool) each role's base must EQUAL the base captured when that pool was built + `slot0` slots, with the pool's
/// bytes-per-expert, and the view must end inside the pool.
pub fn check_routed(routed: &RoutedExpertWeights, pool: Option<(&str, usize)>, device: i32, loc: &'static Location<'static>) -> eyre::Result<()> {
    let fail = |e: String| {
        let (layer, lane) = v4flash_hip::ring::ctx();
        eyre!("MoE launch arg corrupted: {e} (routed experts, layer {layer}, lane {lane}, call site {}:{})", loc.file(), loc.line())
    };
    let roles = [
        ("gate_w_base", Role::Gate, &routed.gate, routed.gate_bytes_per_expert),
        ("up_w_base", Role::Up, &routed.up, routed.up_bytes_per_expert),
        ("down_w_base", Role::Down, &routed.down, routed.down_bytes_per_expert),
    ];
    for (name, _, w, bpe) in roles {
        let len = w.buffer.byte_len();
        if bpe == 0 || (routed.n_slots as usize).checked_mul(bpe).is_none_or(|need| need > len) {
            return Err(fail(bad(
                &format!("{name} n_slots"),
                routed.n_slots,
                format!("<= {} (buffer {len} B / {bpe} B per expert)", len.checked_div(bpe).unwrap_or(0)),
            )));
        }
    }
    let Some((pname, slot0)) = pool else { return Ok(()) };
    let pools = POOLS.read().unwrap_or_else(|p| p.into_inner());
    let p = pools
        .iter()
        .find(|p| p.name == pname && p.device == device)
        .ok_or_else(|| fail(format!("expert pool `{pname}` on dev{device} = unregistered, expected registered at pool build")))?;
    for (name, role, w, bpe) in roles {
        let rp = p.role(role);
        if bpe != rp.bpe {
            return Err(fail(bad(&format!("{name} bytes-per-expert"), bpe, format!("{} (pool {pname} {role:?})", rp.bpe))));
        }
        let want = rp.base + slot0 * rp.bpe;
        let got = w.buffer.raw() as usize;
        if got != want {
            return Err(fail(bad(
                name,
                format!("{got:#x}"),
                format!("{want:#x} (pool {pname} {role:?} base {:#x} captured at build + slot {slot0} x {bpe} B)", rp.base),
            )));
        }
    }
    if slot0 + routed.n_slots as usize > p.n_slots as usize {
        return Err(fail(bad("routed view slots", format!("[{slot0}, +{})", routed.n_slots), format!("inside pool {pname}'s {} slots", p.n_slots))));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const GBPE: usize = 1000;
    const DBPE: usize = 900;

    fn pool() -> PoolBases {
        let rp = |base: usize, bpe: usize, rows: u32, nb: u32| RolePool { base, bytes: 16 * bpe, bpe, rows, n_blocks: nb };
        PoolBases {
            id: 1,
            name: "test",
            device: 1,
            n_slots: 16,
            gate: rp(0x10_0000, GBPE, 2304, 20),
            up: rp(0x20_0000, GBPE, 2304, 20),
            down: rp(0x30_0000, DBPE, 5120, 9),
        }
    }

    fn live_all(_: &str, ptr: usize, _: usize) -> Result<(), String> {
        if ptr >= 0x10_0000 && ptr < 0x80_0000 { Ok(()) } else { Err("not live".into()) }
    }

    struct Args {
        ptrs: Vec<Ptr>,
        weights: Vec<Weight>,
    }

    fn gate_up(gate_slot: usize, up_slot: usize, slots: usize) -> Args {
        Args {
            ptrs: vec![
                Ptr { name: "mid", ptr: 0x40_0000, bytes: 4096 },
                Ptr { name: "xq", ptr: 0x41_0000, bytes: 4096 },
                Ptr { name: "work_items", ptr: 0x42_0000, bytes: 400 },
            ],
            weights: vec![
                Weight { arg: Ptr { name: "gate_w_base", ptr: 0x10_0000 + gate_slot * GBPE, bytes: slots * GBPE }, role: Role::Gate, bpe: GBPE as u32 },
                Weight { arg: Ptr { name: "up_w_base", ptr: 0x20_0000 + up_slot * GBPE, bytes: slots * GBPE }, role: Role::Up, bpe: GBPE as u32 },
            ],
        }
    }

    fn launch<'a>(a: &'a Args, n_rows: u32, n_blocks: u32, n_wi: u32) -> MoeLaunch<'a> {
        MoeLaunch {
            kernel: "mxfp4_pair_matvec_fused_swiglu_wmma",
            device: 1,
            ptrs: &a.ptrs,
            weights: &a.weights,
            n_rows,
            n_blocks,
            n_work_items: n_wi,
            work_items_len: 100,
            chunk_size: 32,
            grid: (n_rows.div_ceil(128), n_wi, 1),
            grid_kind: Grid::RowsByItems { rows_per_wg: 128 },
        }
    }

    fn err(l: &MoeLaunch<'_>) -> String {
        validate(l, &[pool()], &live_all).unwrap_err()
    }

    #[test]
    fn good_launches_pass() {
        let full = gate_up(0, 0, 16);
        assert_eq!(validate(&launch(&full, 2304, 20, 50), &[pool()], &live_all), Ok(()));
        let window = gate_up(4, 4, 8);
        assert_eq!(validate(&launch(&window, 2304, 20, 100), &[pool()], &live_all), Ok(()));
        // resident weights outside any pool: only liveness + scalars
        let mut res = gate_up(0, 0, 16);
        res.weights[0].arg.ptr = 0x50_0000;
        res.weights[1].arg.ptr = 0x60_0000;
        assert_eq!(validate(&launch(&res, 64, 3, 1), &[pool()], &live_all), Ok(()));
        // a pool on ANOTHER device is not consulted
        let mut other = pool();
        other.device = 0;
        assert_eq!(validate(&launch(&gate_up(0, 0, 16), 64, 3, 1), &[other], &live_all), Ok(()));
    }

    #[test]
    fn each_corruption_is_named() {
        let a = gate_up(0, 0, 16);
        let e = err(&launch(&a, 2304, 20, 101));
        assert!(e.starts_with("n_work_items = 101, expected <= work_items capacity 100"), "{e}");
        let e = err(&launch(&a, 2048, 20, 5));
        assert!(e.starts_with("n_rows = 2048, expected 2304"), "{e}");
        let e = err(&launch(&a, 2304, 16, 5));
        assert!(e.starts_with("n_blocks = 16, expected 20"), "{e}");
        let mut l = launch(&a, 2304, 20, 5);
        l.grid = (18, 6, 1);
        assert!(err(&l).starts_with("grid = (18, 6, 1), expected (18, 5, 1)"), "{}", err(&l));
        let mut l = launch(&a, 2304, 20, 5);
        l.chunk_size = 33;
        assert!(err(&l).starts_with("chunk_size = 33"), "{}", err(&l));

        let mut mis = gate_up(0, 0, 16);
        mis.weights[0].arg.ptr += 8;
        mis.weights[0].arg.bytes -= GBPE;
        assert!(err(&launch(&mis, 2304, 20, 5)).starts_with("gate_w_base = 0x100008, expected a slot boundary"), "{}", err(&launch(&mis, 2304, 20, 5)));
        let over = gate_up(8, 8, 9);
        assert!(err(&launch(&over, 2304, 20, 5)).starts_with("gate_w_base view = [0x101f40, +9000 B), expected whole slots inside"));
        let skew = gate_up(4, 5, 8);
        assert!(err(&launch(&skew, 2304, 20, 5)).starts_with("up_w_base slot window = [5, +8), expected the gate's [4, +8)"));
        let mut swapped = gate_up(0, 0, 16);
        swapped.weights.swap(0, 1);
        swapped.weights[0].role = Role::Gate;
        swapped.weights[1].role = Role::Up;
        assert!(err(&launch(&swapped, 2304, 20, 5)).contains("expected inside the Gate buffer, but it lies in the buffer of pool test Up"));
        let mut bpe = gate_up(0, 0, 16);
        bpe.weights[1].bpe = 999;
        assert!(err(&launch(&bpe, 2304, 20, 5)).starts_with("up_w_base bytes-per-expert = 999, expected 1000"));
        let mut garbage = gate_up(0, 0, 16);
        garbage.ptrs[1].ptr = 0x6408_6bd4_6000;
        let e = err(&launch(&garbage, 2304, 20, 5));
        assert!(e.starts_with("xq = 0x64086bd46000, expected inside a live dev1 allocation (not live)"), "{e}");
    }

    #[test]
    fn down_and_persistent() {
        let a = Args {
            ptrs: vec![Ptr { name: "partials", ptr: 0x40_0000, bytes: 64 }],
            weights: vec![Weight { arg: Ptr { name: "w_base", ptr: 0x30_0000, bytes: 16 * DBPE }, role: Role::Down, bpe: DBPE as u32 }],
        };
        let mut l = MoeLaunch {
            kernel: "mxfp4_matvec_par_by_expert_kwide2",
            device: 1,
            ptrs: &a.ptrs,
            weights: &a.weights,
            n_rows: 5120,
            n_blocks: 9,
            n_work_items: 7,
            work_items_len: 7,
            chunk_size: 32,
            grid: (320, 7, 1),
            grid_kind: Grid::RowsByItems { rows_per_wg: 16 },
        };
        assert_eq!(validate(&l, &[pool()], &live_all), Ok(()));
        l.n_blocks = 20;
        assert!(validate(&l, &[pool()], &live_all).unwrap_err().starts_with("n_blocks = 20, expected 9 (k / 256 of pool test Down)"));
        l.n_blocks = 9;
        l.grid = (24, 1, 1);
        l.grid_kind = Grid::Persistent;
        assert_eq!(validate(&l, &[pool()], &live_all), Ok(()));
        l.grid = (24, 2, 1);
        assert!(validate(&l, &[pool()], &live_all).unwrap_err().starts_with("grid = (24, 2, 1), expected (wgs > 0, 1, 1)"));
    }

    /// `register_pool` + `check_routed` over fake (non-owning, never dereferenced) views: no GPU.
    #[test]
    fn call_site_check_against_the_captured_pool() {
        use crate::weights::DeviceWeight;
        use v4flash_core::gguf::GgufType;
        let dw = |base: usize, slots: usize, bpe: usize, rows: u64, k: u64| DeviceWeight {
            // SAFETY: a non-owning view that is only compared, never dereferenced or freed.
            buffer: unsafe { v4flash_hip::DeviceBuffer::<u8>::from_raw_parts(base as v4flash_hip::sys::hipDeviceptr_t, slots * bpe, 5) },
            n_elements: slots as u64 * rows * k,
            dtype: GgufType::MXFP4,
            shape: vec![slots as u64, rows, k],
        };
        let routed = |gate: usize, up: usize, down: usize, slots: u32| RoutedExpertWeights {
            gate: dw(gate, slots as usize, GBPE, 2304, 5120),
            up: dw(up, slots as usize, GBPE, 2304, 5120),
            down: dw(down, slots as usize, DBPE, 5120, 2304),
            gate_bytes_per_expert: GBPE,
            up_bytes_per_expert: GBPE,
            down_bytes_per_expert: DBPE,
            n_slots: slots,
        };
        let (g, u, d) = (0x4d00_0000_0000_0000usize, 0x4d00_0001_0000_0000usize, 0x4d00_0002_0000_0000usize);
        let pool = routed(g, u, d, 16);
        let _guard = register_pool("canary test pool", 5, &pool).unwrap();
        let p = pools().into_iter().find(|p| p.name == "canary test pool").unwrap();
        assert_eq!((p.gate.rows, p.gate.n_blocks, p.down.rows, p.down.n_blocks), (2304, 20, 5120, 9));
        let here = Location::caller();
        // the whole pool, and a window view at slot 4
        check_routed(&pool, Some(("canary test pool", 0)), 5, here).unwrap();
        let win = routed(g + 4 * GBPE, u + 4 * GBPE, d + 4 * DBPE, 8);
        check_routed(&win, Some(("canary test pool", 4)), 5, here).unwrap();
        // a view whose up base drifted by one slot
        let bad_up = routed(g + 4 * GBPE, u + 5 * GBPE, d + 4 * DBPE, 8);
        let e = check_routed(&bad_up, Some(("canary test pool", 4)), 5, here).unwrap_err().to_string();
        assert!(e.starts_with("MoE launch arg corrupted: up_w_base = 0x4d00000100001388, expected 0x4d00000100000fa0 (pool canary test pool Up base"), "{e}");
        // a window past the pool's end
        let e = check_routed(&routed(g + 12 * GBPE, u + 12 * GBPE, d + 12 * DBPE, 8), Some(("canary test pool", 12)), 5, here).unwrap_err().to_string();
        assert!(e.contains("routed view slots = [12, +8), expected inside pool canary test pool's 16 slots"), "{e}");
        // n_slots larger than the buffers
        let mut big = routed(g, u, d, 8);
        big.n_slots = 9;
        let e = check_routed(&big, None, 5, here).unwrap_err().to_string();
        assert!(e.contains("gate_w_base n_slots = 9, expected <= 8"), "{e}");
        // the wrong device has no such pool
        assert!(check_routed(&pool, Some(("canary test pool", 0)), 6, here).unwrap_err().to_string().contains("unregistered"));
        drop(_guard);
        assert!(pools().iter().all(|p| p.name != "canary test pool"), "the guard deregisters");
    }
}
