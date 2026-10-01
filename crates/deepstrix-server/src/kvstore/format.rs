//! File format of chunks (`.kvc`) and tails (`.kvt`) (design 4.1, 4.5, 5.1).
//!
//! ```text
//! chunk:  header (256 B) | token ids (C × le32) | image records (40 B each) | payload
//! tail:   header (512 B) | open token ids       | image records             | section E | section D
//! ```
//!
//! WHY fixed binary headers instead of v6's `meta.json`:
//! - the startup scan reads one block per file (~38K files at the cap);
//! - the IO thread updates `hits` with one 4-byte `pwrite` at a fixed offset
//!   ([`TAIL_HITS_OFFSET`]), which is why the header checksum skips those 4
//!   bytes: a touch must not have to rewrite the whole header;
//! - section D (decoder windows + DSpark ring) is LAST, so demoting a full tail
//!   to an encoder tail is an `ftruncate` plus a header rewrite (8.2).
//!
//! Every read checks the header checksum, the namespace, the exact file length,
//! the key (recomputed from the stored parent / base, token ids and image
//! records, which also proves the ids), and each payload section's blake3. A
//! file that fails any of these is a miss, never data (G5).

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

use super::keys::{self, ImageRecord, Key};
use super::C;

pub const CHUNK_MAGIC: [u8; 5] = *b"DSKVC";
pub const TAIL_MAGIC: [u8; 5] = *b"DSKVT";
pub const FORMAT_VERSION: u8 = 1;
pub const CHUNK_HEADER_LEN: usize = 256;
pub const TAIL_HEADER_LEN: usize = 512;
pub const IMAGE_RECORD_LEN: usize = 40;
/// The four KV-source stores of V4.1 (layers 2, 8, 14, 20).
pub const MAX_STORES: usize = 4;
/// `session_id` is telemetry only (6.4); longer ids are truncated.
pub const SESSION_ID_MAX: usize = 64;
const CHECKSUM_LEN: usize = 16;
/// Offset of the tail's `hits` (le32), excluded from the header checksum.
pub const TAIL_HITS_OFFSET: u64 = 160;
/// `hits` is clamped to this on read (log2(1 + 2^16) × 6 h ≈ 4 days of score).
pub const MAX_HITS: u32 = 1 << 16;

/// Row encodings recorded in [`StoreLayout`] (namespace ABI).
pub const ENC_F16: u8 = 1;
pub const ENC_E2M1_B32: u8 = 2;

/// One KV-source store's layout: part of the namespace (4.4) and of the
/// section-E shape check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoreLayout {
    pub layer: u16,
    pub ratio: u16,
    pub row_encoding: u8,
    pub row_bytes: u32,
    pub key_encoding: u8,
    pub key_bytes: u32,
    /// `state_kv` + `state_score` (f32) of this store's compressor.
    pub state_bytes: u32,
}

impl StoreLayout {
    pub fn encode(&self) -> [u8; 18] {
        let mut b = [0u8; 18];
        b[0..2].copy_from_slice(&self.layer.to_le_bytes());
        b[2..4].copy_from_slice(&self.ratio.to_le_bytes());
        b[4] = self.row_encoding;
        b[5..9].copy_from_slice(&self.row_bytes.to_le_bytes());
        b[9] = self.key_encoding;
        b[10..14].copy_from_slice(&self.key_bytes.to_le_bytes());
        b[14..18].copy_from_slice(&self.state_bytes.to_le_bytes());
        b
    }

    /// Compressed rows of positions `[0, t)`: `n_comp = ⌊t/ratio⌋`.
    pub fn rows_at(&self, t: u32) -> u32 {
        t / self.ratio as u32
    }

    /// Bytes of one compressed row plus its index key.
    pub fn row_and_key_bytes(&self) -> u64 {
        self.row_bytes as u64 + self.key_bytes as u64
    }
}

/// The V4.1 stores from `config` (3.): comp rows f16 × N_HEAD_DIM, E2M1 index
/// keys, accumulators of `ratio` f32 rows of width N_HEAD_DIM, kv + score.
pub fn v41_stores() -> [StoreLayout; MAX_STORES] {
    use v4flash_kernels::config::{COMPRESS_RATIOS, KV_SOURCE_LAYERS, N_HEAD_DIM};
    use v4flash_kernels::index_kv_e2m1::E2M1_KEY_ROW_BYTES;
    assert_eq!(KV_SOURCE_LAYERS.len(), MAX_STORES);
    core::array::from_fn(|i| {
        let layer = KV_SOURCE_LAYERS[i] as usize;
        let ratio = COMPRESS_RATIOS[layer];
        StoreLayout {
            layer: layer as u16,
            ratio: ratio as u16,
            row_encoding: ENC_F16,
            row_bytes: N_HEAD_DIM * 2,
            key_encoding: ENC_E2M1_B32,
            key_bytes: E2M1_KEY_ROW_BYTES as u32,
            // HetCompressorState::alloc: state_rows = ratio * coff (coff = 1 off
            // ratio 4), width = head_dim; two f32 buffers.
            state_bytes: 2 * ratio * N_HEAD_DIM * 4,
        }
    })
}

/// Bytes per position of the chunk payload (3.: 2,760 B for V4.1).
pub fn bytes_per_position(stores: &[StoreLayout]) -> u64 {
    // Exact only when every ratio divides C, which v41_stores guarantees.
    stores.iter().map(|s| s.row_and_key_bytes() * (C as u64 / s.ratio as u64)).sum::<u64>() / C as u64
}

/// Bytes of one chunk's payload: store by store, the rows `[kC/ratio,
/// (k+1)C/ratio)` and then their keys (4.1).
pub fn chunk_payload_len(stores: &[StoreLayout]) -> u64 {
    stores.iter().map(|s| (C / s.ratio as u32) as u64 * s.row_and_key_bytes()).sum()
}

/// Per-layer raw window row: f16 × N_HEAD_DIM (`HetLayerState::kv_cache`).
pub fn window_row_bytes() -> u64 {
    v4flash_kernels::config::N_HEAD_DIM as u64 * 2
}

/// Encoder layers (L0..CED_DECODER_START), whose windows go in section E.
pub fn encoder_layers() -> u64 {
    v4flash_kernels::config::CED_DECODER_START as u64
}

/// Section E's length for a tail at `t` with `n_raw` encoder window rows: per
/// store the open rows `[⌊t/C⌋C/ratio, ⌊t/ratio⌋)` with their keys and the
/// accumulators, then the encoder windows (4.5). M2's capture writes exactly
/// this; a restore refuses a section of any other length (a shape check).
pub fn section_e_len(stores: &[StoreLayout], t: u32, n_raw: u32) -> u64 {
    let b0 = t / C * C;
    let open: u64 = stores
        .iter()
        .map(|s| (s.rows_at(t) - s.rows_at(b0)) as u64 * s.row_and_key_bytes() + s.state_bytes as u64)
        .sum();
    open + encoder_layers() * n_raw as u64 * window_row_bytes()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Provenance {
    Prefill = 0,
    Decode = 1,
    Mixed = 2,
    /// Copied from a v6 snapshot in shadow phase A (11.1).
    BackfillV6 = 3,
}

impl Provenance {
    fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => Self::Prefill,
            1 => Self::Decode,
            2 => Self::Mixed,
            3 => Self::BackfillV6,
            _ => return None,
        })
    }
}

/// Full tails carry section D (decoder windows + ring); encoder tails do not.
/// `Full > Enc`: a full tail replaces an encoder tail at the same key (5.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum TailKind {
    Enc = 0,
    Full = 1,
}

/// Why a tail was written (5.2). Recorded so thinning can tell a demoted
/// prompt-end tail from a waypoint, and for `kv.write kind=`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TailOrigin {
    PromptEnd = 0,
    Waypoint = 1,
    Anchor = 2,
    Cancel = 3,
    /// M4.
    TurnEnd = 4,
}

impl TailOrigin {
    fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => Self::PromptEnd,
            1 => Self::Waypoint,
            2 => Self::Anchor,
            3 => Self::Cancel,
            4 => Self::TurnEnd,
            _ => return None,
        })
    }
}

pub const TAIL_FLAG_ANCHOR: u8 = 1;
pub const TAIL_FLAG_DEMOTED: u8 = 2;

/// The writer's build (4.4: headers only, never the namespace).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BuildId {
    /// Raw git sha1.
    pub sha: [u8; 20],
    pub dirty: bool,
    /// Bits copied from a v6 snapshot: build id `v6` (11.1).
    pub v6: bool,
}

impl BuildId {
    pub const UNKNOWN: BuildId = BuildId { sha: [0; 20], dirty: false, v6: false };
    pub const V6: BuildId = BuildId { sha: [0; 20], dirty: false, v6: true };

    /// `<40 hex>` with an optional `-dirty` suffix.
    pub fn parse(s: &str) -> Option<Self> {
        let (hex, dirty) = match s.strip_suffix("-dirty") {
            Some(h) => (h, true),
            None => (s, false),
        };
        if hex.len() != 40 {
            return None;
        }
        let mut sha = [0u8; 20];
        for (i, o) in sha.iter_mut().enumerate() {
            *o = u8::from_str_radix(hex.get(2 * i..2 * i + 2)?, 16).ok()?;
        }
        Some(Self { sha, dirty, v6: false })
    }

    fn flags(&self) -> u8 {
        self.dirty as u8 | (self.v6 as u8) << 1
    }

    fn from_parts(sha: [u8; 20], flags: u8) -> Self {
        Self { sha, dirty: flags & 1 != 0, v6: flags & 2 != 0 }
    }
}

impl std::fmt::Display for BuildId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.v6 {
            return f.write_str("v6");
        }
        f.write_str(&keys::hex(&self.sha))?;
        if self.dirty {
            f.write_str("-dirty")?;
        }
        Ok(())
    }
}

/// `KV_NUMERICS_GEN` for bits copied from v6 snapshots (11.1: "generation v6").
pub const GEN_V6: u32 = u32::MAX;

/// The effective numerics generation (4.4): the code constant AND the hash of
/// the numerics-relevant env knobs. A knob changed only in the launch env
/// changes no commit, so the constant alone would hide it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GenPair {
    pub gen: u32,
    pub knob: [u8; 16],
}

impl std::fmt::Display for GenPair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.gen == GEN_V6 {
            write!(f, "v6/{}", keys::hex(&self.knob[..4]))
        } else {
            write!(f, "{}/{}", self.gen, keys::hex(&self.knob[..4]))
        }
    }
}

/// Rows of one store in a chunk ("per store: rows, row bytes, key bytes").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StoreRows {
    pub rows: u32,
    pub row_bytes: u16,
    pub key_bytes: u16,
}

impl StoreRows {
    /// The rows of one chunk of `s`.
    pub fn chunk_of(s: &StoreLayout) -> Self {
        Self { rows: C / s.ratio as u32, row_bytes: s.row_bytes as u16, key_bytes: s.key_bytes as u16 }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatError {
    /// An IO failure, with its kind: only some kinds say the FILE is bad
    /// ([`FormatError::evicts`]).
    Io(std::io::ErrorKind, String),
    /// Shorter than its header claims (torn or truncated write).
    Short { want: u64, got: u64 },
    /// Longer than its header claims.
    Long { want: u64, got: u64 },
    BadMagic,
    BadVersion(u8),
    BadField(&'static str),
    HeaderChecksum,
    Namespace,
    /// The key recomputed from the stored parent, ids and images differs, or
    /// the header names another key than the one asked for.
    Key,
    /// A payload section's blake3 differs ("payload", "E", "D").
    Checksum(&'static str),
    /// Section D was asked for, but the file is an encoder tail.
    NoSectionD,
}

impl FormatError {
    /// Is this a verdict on the file's DATA (evict it, 6.2, 9.4), rather
    /// than a transient failure of this read (EMFILE, ENOMEM, EIO, EACCES...)
    /// that says nothing about the file? A file that is gone or shorter than
    /// a read (UnexpectedEof) is data-attributable; any other IO error is not,
    /// and evicting on it would turn an infrastructure fault into cold
    /// re-prefills (risk 10).
    pub fn evicts(&self) -> bool {
        match self {
            Self::Io(kind, _) => matches!(kind, std::io::ErrorKind::UnexpectedEof | std::io::ErrorKind::NotFound),
            _ => true,
        }
    }
}

impl std::fmt::Display for FormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(k, e) => write!(f, "io ({k:?}): {e}"),
            Self::Short { want, got } => write!(f, "short: want {want} B, file has {got}"),
            Self::Long { want, got } => write!(f, "long: want {want} B, file has {got}"),
            Self::BadMagic => f.write_str("bad magic"),
            Self::BadVersion(v) => write!(f, "format version {v}"),
            Self::BadField(w) => write!(f, "bad header field {w}"),
            Self::HeaderChecksum => f.write_str("header checksum"),
            Self::Namespace => f.write_str("namespace mismatch"),
            Self::Key => f.write_str("key mismatch"),
            Self::Checksum(s) => write!(f, "section {s} checksum"),
            Self::NoSectionD => f.write_str("section D asked of an encoder tail"),
        }
    }
}

impl std::error::Error for FormatError {}

impl From<std::io::Error> for FormatError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.kind(), e.to_string())
    }
}

/// Little-endian cursor over a header block.
struct Put<'a>(&'a mut [u8], usize);
impl Put<'_> {
    fn bytes(&mut self, b: &[u8]) {
        self.0[self.1..self.1 + b.len()].copy_from_slice(b);
        self.1 += b.len();
    }
    fn u8(&mut self, v: u8) {
        self.bytes(&[v]);
    }
    fn u16(&mut self, v: u16) {
        self.bytes(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.bytes(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }
    fn skip(&mut self, n: usize) {
        self.1 += n;
    }
}

struct Get<'a>(&'a [u8], usize);
impl Get<'_> {
    fn bytes<const N: usize>(&mut self) -> [u8; N] {
        let mut o = [0u8; N];
        o.copy_from_slice(&self.0[self.1..self.1 + N]);
        self.1 += N;
        o
    }
    fn u8(&mut self) -> u8 {
        self.bytes::<1>()[0]
    }
    fn u16(&mut self) -> u16 {
        u16::from_le_bytes(self.bytes())
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.bytes())
    }
    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.bytes())
    }
    fn skip(&mut self, n: usize) {
        self.1 += n;
    }
}

/// blake3 of the header block minus its trailing checksum, with the tail's
/// `hits` zeroed, truncated to 16 bytes.
fn header_checksum(block: &[u8], hits_at: Option<usize>) -> [u8; CHECKSUM_LEN] {
    let body = &block[..block.len() - CHECKSUM_LEN];
    let mut h = blake3::Hasher::new_derive_key("deepstrix kvstore v1 header");
    match hits_at {
        Some(at) => {
            h.update(&body[..at]);
            h.update(&[0u8; 4]);
            h.update(&body[at + 4..]);
        }
        None => {
            h.update(body);
        }
    }
    let mut out = [0u8; CHECKSUM_LEN];
    out.copy_from_slice(&h.finalize().as_bytes()[..CHECKSUM_LEN]);
    out
}

fn check_prelude(block: &[u8], magic: &[u8; 5], len: usize) -> Result<(), FormatError> {
    if block[0..5] != magic[..] {
        return Err(FormatError::BadMagic);
    }
    if block[5] != FORMAT_VERSION {
        return Err(FormatError::BadVersion(block[5]));
    }
    if u16::from_le_bytes([block[6], block[7]]) as usize != len {
        return Err(FormatError::BadField("header_len"));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkHeader {
    pub ns: Key,
    pub key: Key,
    /// chain_k (the root for k = 0).
    pub parent: Key,
    pub k: u32,
    pub n_tokens: u16,
    pub n_images: u16,
    pub stores: Vec<StoreRows>,
    pub provenance: Provenance,
    pub gen: GenPair,
    pub build: BuildId,
    /// Unix seconds.
    pub created: u64,
    pub payload_len: u64,
    pub payload_hash: [u8; 32],
}

impl ChunkHeader {
    pub fn encode(&self) -> [u8; CHUNK_HEADER_LEN] {
        assert!(self.stores.len() <= MAX_STORES);
        let mut b = [0u8; CHUNK_HEADER_LEN];
        let mut p = Put(&mut b, 0);
        p.bytes(&CHUNK_MAGIC);
        p.u8(FORMAT_VERSION);
        p.u16(CHUNK_HEADER_LEN as u16);
        p.bytes(&self.ns);
        p.bytes(&self.key);
        p.bytes(&self.parent);
        p.u32(self.k);
        p.u16(self.n_tokens);
        p.u16(self.n_images);
        p.u8(self.stores.len() as u8);
        p.u8(self.provenance as u8);
        p.skip(2);
        for i in 0..MAX_STORES {
            let s = self.stores.get(i).copied().unwrap_or_default();
            p.u32(s.rows);
            p.u16(s.row_bytes);
            p.u16(s.key_bytes);
        }
        p.u32(self.gen.gen);
        p.bytes(&self.gen.knob);
        p.bytes(&self.build.sha);
        p.u8(self.build.flags());
        p.skip(3);
        p.u64(self.created);
        p.u64(self.payload_len);
        p.bytes(&self.payload_hash);
        debug_assert_eq!(p.1, CHUNK_HEADER_LEN - CHECKSUM_LEN);
        let sum = header_checksum(&b, None);
        b[CHUNK_HEADER_LEN - CHECKSUM_LEN..].copy_from_slice(&sum);
        b
    }

    pub fn decode(b: &[u8]) -> Result<Self, FormatError> {
        if b.len() < CHUNK_HEADER_LEN {
            return Err(FormatError::Short { want: CHUNK_HEADER_LEN as u64, got: b.len() as u64 });
        }
        let b = &b[..CHUNK_HEADER_LEN];
        check_prelude(b, &CHUNK_MAGIC, CHUNK_HEADER_LEN)?;
        if header_checksum(b, None)[..] != b[CHUNK_HEADER_LEN - CHECKSUM_LEN..] {
            return Err(FormatError::HeaderChecksum);
        }
        let mut g = Get(b, 8);
        let ns = g.bytes();
        let key = g.bytes();
        let parent = g.bytes();
        let k = g.u32();
        let n_tokens = g.u16();
        let n_images = g.u16();
        let n_stores = g.u8() as usize;
        let provenance = Provenance::from_u8(g.u8()).ok_or(FormatError::BadField("provenance"))?;
        g.skip(2);
        if n_stores > MAX_STORES {
            return Err(FormatError::BadField("n_stores"));
        }
        let mut stores = Vec::with_capacity(n_stores);
        for i in 0..MAX_STORES {
            let s = StoreRows { rows: g.u32(), row_bytes: g.u16(), key_bytes: g.u16() };
            if i < n_stores {
                stores.push(s);
            }
        }
        let gen = GenPair { gen: g.u32(), knob: g.bytes() };
        let sha = g.bytes();
        let build = BuildId::from_parts(sha, g.u8());
        g.skip(3);
        let created = g.u64();
        let payload_len = g.u64();
        let payload_hash = g.bytes();
        let h = Self { ns, key, parent, k, n_tokens, n_images, stores, provenance, gen, build, created, payload_len, payload_hash };
        let rows: u64 = h.stores.iter().map(|s| s.rows as u64 * (s.row_bytes as u64 + s.key_bytes as u64)).sum();
        if rows != h.payload_len {
            return Err(FormatError::BadField("payload_len"));
        }
        if h.n_tokens as u32 != C {
            return Err(FormatError::BadField("n_tokens"));
        }
        Ok(h)
    }

    pub fn data_offset(&self) -> u64 {
        CHUNK_HEADER_LEN as u64 + 4 * self.n_tokens as u64 + IMAGE_RECORD_LEN as u64 * self.n_images as u64
    }

    pub fn file_len(&self) -> u64 {
        self.data_offset() + self.payload_len
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TailHeader {
    pub ns: Key,
    pub key: Key,
    /// chain_⌊t/C⌋.
    pub base: Key,
    pub t: u32,
    pub kind: TailKind,
    pub anchor: bool,
    /// A full tail truncated to encoder-only (8.2); only these are thinned.
    pub demoted: bool,
    pub origin: TailOrigin,
    pub n_open: u16,
    pub n_images: u16,
    /// Encoder window rows (min(t, 128)).
    pub n_raw: u32,
    /// Decoder window rows (section D; 0 for an encoder tail).
    pub n_raw_dec: u32,
    /// DSpark drafter fingerprint; all-zero = no ring in section D.
    pub drafter: [u8; 32],
    /// Restores of this tail itself (8.3); mutable, outside the checksum.
    pub hits: u32,
    pub gen: GenPair,
    pub build: BuildId,
    pub created: u64,
    pub sec_e_len: u64,
    pub sec_e_hash: [u8; 32],
    pub sec_d_len: u64,
    pub sec_d_hash: [u8; 32],
    /// Telemetry only (6.4).
    pub session_id: String,
}

impl TailHeader {
    pub fn encode(&self) -> [u8; TAIL_HEADER_LEN] {
        let mut b = [0u8; TAIL_HEADER_LEN];
        let sid = truncate_utf8(&self.session_id, SESSION_ID_MAX);
        let mut p = Put(&mut b, 0);
        p.bytes(&TAIL_MAGIC);
        p.u8(FORMAT_VERSION);
        p.u16(TAIL_HEADER_LEN as u16);
        p.bytes(&self.ns);
        p.bytes(&self.key);
        p.bytes(&self.base);
        p.u32(self.t);
        p.u8(self.kind as u8);
        p.u8((self.anchor as u8) * TAIL_FLAG_ANCHOR | (self.demoted as u8) * TAIL_FLAG_DEMOTED);
        p.u8(self.origin as u8);
        p.u8(sid.len() as u8);
        p.u16(self.n_open);
        p.u16(self.n_images);
        p.u32(self.n_raw);
        p.u32(self.n_raw_dec);
        p.skip(4);
        p.bytes(&self.drafter);
        debug_assert_eq!(p.1 as u64, TAIL_HITS_OFFSET);
        p.u32(self.hits);
        p.u32(self.gen.gen);
        p.bytes(&self.gen.knob);
        p.bytes(&self.build.sha);
        p.u8(self.build.flags());
        p.skip(3);
        p.u64(self.created);
        p.u64(self.sec_e_len);
        p.bytes(&self.sec_e_hash);
        p.u64(self.sec_d_len);
        p.bytes(&self.sec_d_hash);
        let mut sid_buf = [0u8; SESSION_ID_MAX];
        sid_buf[..sid.len()].copy_from_slice(sid.as_bytes());
        p.bytes(&sid_buf);
        let sum = header_checksum(&b, Some(TAIL_HITS_OFFSET as usize));
        b[TAIL_HEADER_LEN - CHECKSUM_LEN..].copy_from_slice(&sum);
        b
    }

    pub fn decode(b: &[u8]) -> Result<Self, FormatError> {
        if b.len() < TAIL_HEADER_LEN {
            return Err(FormatError::Short { want: TAIL_HEADER_LEN as u64, got: b.len() as u64 });
        }
        let b = &b[..TAIL_HEADER_LEN];
        check_prelude(b, &TAIL_MAGIC, TAIL_HEADER_LEN)?;
        if header_checksum(b, Some(TAIL_HITS_OFFSET as usize))[..] != b[TAIL_HEADER_LEN - CHECKSUM_LEN..] {
            return Err(FormatError::HeaderChecksum);
        }
        let mut g = Get(b, 8);
        let ns = g.bytes();
        let key = g.bytes();
        let base = g.bytes();
        let t = g.u32();
        let kind = match g.u8() {
            0 => TailKind::Enc,
            1 => TailKind::Full,
            _ => return Err(FormatError::BadField("kind")),
        };
        let flags = g.u8();
        let origin = TailOrigin::from_u8(g.u8()).ok_or(FormatError::BadField("origin"))?;
        let sid_len = g.u8() as usize;
        let n_open = g.u16();
        let n_images = g.u16();
        let n_raw = g.u32();
        let n_raw_dec = g.u32();
        g.skip(4);
        let drafter = g.bytes();
        // `hits` is outside the checksum: a damaged value must not make a
        // tail immortal (score = path_last_used + 6 h × log2(1 + hits)).
        let hits = g.u32().min(MAX_HITS);
        let gen = GenPair { gen: g.u32(), knob: g.bytes() };
        let sha = g.bytes();
        let build = BuildId::from_parts(sha, g.u8());
        g.skip(3);
        let created = g.u64();
        let sec_e_len = g.u64();
        let sec_e_hash = g.bytes();
        let sec_d_len = g.u64();
        let sec_d_hash = g.bytes();
        let sid_buf: [u8; SESSION_ID_MAX] = g.bytes();
        if sid_len > SESSION_ID_MAX || flags & !(TAIL_FLAG_ANCHOR | TAIL_FLAG_DEMOTED) != 0 {
            return Err(FormatError::BadField("flags"));
        }
        let session_id = String::from_utf8_lossy(&sid_buf[..sid_len]).into_owned();
        let h = Self {
            ns,
            key,
            base,
            t,
            kind,
            anchor: flags & TAIL_FLAG_ANCHOR != 0,
            demoted: flags & TAIL_FLAG_DEMOTED != 0,
            origin,
            n_open,
            n_images,
            n_raw,
            n_raw_dec,
            drafter,
            hits,
            gen,
            build,
            created,
            sec_e_len,
            sec_e_hash,
            sec_d_len,
            sec_d_hash,
            session_id,
        };
        if h.n_open as u32 != h.t % C {
            return Err(FormatError::BadField("n_open"));
        }
        if h.kind == TailKind::Enc && (h.sec_d_len != 0 || h.n_raw_dec != 0) {
            return Err(FormatError::BadField("enc tail with section D"));
        }
        if h.anchor && h.demoted {
            return Err(FormatError::BadField("demoted anchor"));
        }
        Ok(h)
    }

    pub fn e_offset(&self) -> u64 {
        TAIL_HEADER_LEN as u64 + 4 * self.n_open as u64 + IMAGE_RECORD_LEN as u64 * self.n_images as u64
    }

    /// Where section E ends: the length of a demoted (or encoder) tail.
    pub fn e_end(&self) -> u64 {
        self.e_offset() + self.sec_e_len
    }

    pub fn file_len(&self) -> u64 {
        self.e_end() + self.sec_d_len
    }

    /// This header after demotion (8.2): encoder-only, section D gone.
    pub fn demoted(&self, hits: u32) -> Self {
        let mut h = self.clone();
        h.kind = TailKind::Enc;
        h.demoted = true;
        h.n_raw_dec = 0;
        h.sec_d_len = 0;
        h.sec_d_hash = [0; 32];
        h.hits = hits;
        h
    }
}

fn truncate_utf8(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// `header ‖ token ids ‖ image records`: everything before a chunk's payload.
pub fn encode_chunk_prefix(h: &ChunkHeader, tokens: &[i32], images: &[ImageRecord]) -> Vec<u8> {
    debug_assert_eq!(tokens.len(), h.n_tokens as usize);
    debug_assert_eq!(images.len(), h.n_images as usize);
    let mut v = Vec::with_capacity(h.data_offset() as usize);
    v.extend_from_slice(&h.encode());
    append_ids_and_images(&mut v, tokens, images, h.k * C);
    v
}

/// `header ‖ open token ids ‖ image records`: everything before section E.
pub fn encode_tail_prefix(h: &TailHeader, open: &[i32], images: &[ImageRecord]) -> Vec<u8> {
    debug_assert_eq!(open.len(), h.n_open as usize);
    debug_assert_eq!(images.len(), h.n_images as usize);
    let mut v = Vec::with_capacity(h.e_offset() as usize);
    v.extend_from_slice(&h.encode());
    append_ids_and_images(&mut v, open, images, h.t / C * C);
    v
}

fn append_ids_and_images(v: &mut Vec<u8>, ids: &[i32], images: &[ImageRecord], a: u32) {
    for t in ids {
        v.extend_from_slice(&t.to_le_bytes());
    }
    for r in images {
        v.extend_from_slice(&r.to_bytes(a));
    }
}

fn parse_ids_and_images(b: &[u8], n_ids: usize, n_images: usize, a: u32) -> Result<(Vec<i32>, Vec<ImageRecord>), FormatError> {
    if b.len() < 4 * n_ids + IMAGE_RECORD_LEN * n_images {
        return Err(FormatError::Short { want: (4 * n_ids + IMAGE_RECORD_LEN * n_images) as u64, got: b.len() as u64 });
    }
    let ids: Vec<i32> = b[..4 * n_ids].chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
    let mut images = Vec::with_capacity(n_images);
    for c in b[4 * n_ids..4 * n_ids + IMAGE_RECORD_LEN * n_images].chunks_exact(IMAGE_RECORD_LEN) {
        let r = ImageRecord::from_bytes(c.try_into().unwrap(), a).ok_or(FormatError::BadField("image"))?;
        images.push(r);
    }
    // Strictly increasing starts inside [a, a + C): what `images_in` produced.
    // (A tail's tighter bound, start < t, is checked by `read_tail`.)
    if images.windows(2).any(|w| w[0].start >= w[1].start) || images.iter().any(|r| r.start as u64 >= a as u64 + C as u64) {
        return Err(FormatError::BadField("image order"));
    }
    Ok((ids, images))
}

fn read_exact_at(f: &File, len: u64, off: u64) -> Result<Vec<u8>, FormatError> {
    let mut v = vec![0u8; len as usize];
    f.read_exact_at(&mut v, off)?;
    Ok(v)
}

/// Read `len` bytes at `off` into `buf` (reused: its capacity is kept). No
/// `clear()`: `resize` zero-fills only growth, and the read overwrites all
/// of it (zeroing 911 MB would cost a third of the blake3 pass).
fn read_into(f: &File, buf: &mut Vec<u8>, len: u64, off: u64) -> Result<(), FormatError> {
    buf.resize(len as usize, 0);
    f.read_exact_at(buf, off)?;
    Ok(())
}

/// A verified chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkFile {
    pub header: ChunkHeader,
    pub tokens: Vec<i32>,
    /// Absolute positions.
    pub images: Vec<ImageRecord>,
    pub payload: Vec<u8>,
}

/// A chunk's verified header, ids and images (its payload went into the
/// caller's buffer).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkMeta {
    pub header: ChunkHeader,
    pub tokens: Vec<i32>,
    pub images: Vec<ImageRecord>,
}

/// Read and verify a chunk into `payload` (a staging buffer the caller
/// reuses): header checksum, the requested key, exact length, namespace, the
/// key recomputed from parent + ids + images, and the payload blake3.
pub fn read_chunk_into(path: &Path, ns: &Key, key: Option<&Key>, payload: &mut Vec<u8>) -> Result<ChunkMeta, FormatError> {
    let f = File::open(path)?;
    let size = f.metadata()?.len();
    let hb = read_exact_at(&f, (CHUNK_HEADER_LEN as u64).min(size), 0)?;
    let header = ChunkHeader::decode(&hb)?;
    if key.is_some_and(|k| *k != header.key) {
        return Err(FormatError::Key);
    }
    check_len(header.file_len(), size)?;
    let prefix = read_exact_at(&f, header.data_offset() - CHUNK_HEADER_LEN as u64, CHUNK_HEADER_LEN as u64)?;
    let (tokens, images) = verify_chunk_prefix(&header, &prefix, ns)?;
    read_into(&f, payload, header.payload_len, header.data_offset())?;
    verify_section(payload, &header.payload_hash, "payload")?;
    Ok(ChunkMeta { header, tokens, images })
}

/// Read and verify a whole chunk file ([`read_chunk_into`] with a fresh
/// buffer and no key expectation).
pub fn read_chunk(path: &Path, ns: &Key) -> Result<ChunkFile, FormatError> {
    let mut payload = Vec::new();
    let m = read_chunk_into(path, ns, None, &mut payload)?;
    Ok(ChunkFile { header: m.header, tokens: m.tokens, images: m.images, payload })
}

/// The blake3 check of one staged section (payload, E or D).
pub fn verify_section(bytes: &[u8], want: &[u8; 32], which: &'static str) -> Result<(), FormatError> {
    if blake3::hash(bytes).as_bytes() != want {
        return Err(FormatError::Checksum(which));
    }
    Ok(())
}

/// Namespace + ids + images + key of a chunk whose header is already decoded.
pub fn verify_chunk_prefix(h: &ChunkHeader, prefix: &[u8], ns: &Key) -> Result<(Vec<i32>, Vec<ImageRecord>), FormatError> {
    if &h.ns != ns {
        return Err(FormatError::Namespace);
    }
    let a = h.k.checked_mul(C).ok_or(FormatError::BadField("k"))?;
    let (tokens, images) = parse_ids_and_images(prefix, h.n_tokens as usize, h.n_images as usize, a)?;
    if keys::chunk_step(&h.parent, a, &tokens, &images) != h.key {
        return Err(FormatError::Key);
    }
    Ok((tokens, images))
}

fn check_len(want: u64, got: u64) -> Result<(), FormatError> {
    match got.cmp(&want) {
        std::cmp::Ordering::Less => Err(FormatError::Short { want, got }),
        std::cmp::Ordering::Greater => Err(FormatError::Long { want, got }),
        std::cmp::Ordering::Equal => Ok(()),
    }
}

/// A verified tail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TailFile {
    pub header: TailHeader,
    pub open: Vec<i32>,
    pub images: Vec<ImageRecord>,
    pub sec_e: Vec<u8>,
    /// `Some` exactly when section D was asked for (t ≤ 128 restores, 6.6).
    pub sec_d: Option<Vec<u8>>,
}

/// A tail's verified header, open ids and images (its sections went into
/// the caller's buffers).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TailMeta {
    pub header: TailHeader,
    pub open: Vec<i32>,
    pub images: Vec<ImageRecord>,
}

/// Read and verify a tail into `sec_e` (and `sec_d` when `want_d`). Asking
/// for section D of an encoder tail is an error, never an empty D: a restore
/// that keeps rings it never got is the KNOWN_BUGS #25 signature.
pub fn read_tail_into(
    path: &Path,
    ns: &Key,
    key: Option<&Key>,
    want_d: bool,
    sec_e: &mut Vec<u8>,
    sec_d: &mut Vec<u8>,
) -> Result<TailMeta, FormatError> {
    let f = File::open(path)?;
    let size = f.metadata()?.len();
    let hb = read_exact_at(&f, (TAIL_HEADER_LEN as u64).min(size), 0)?;
    let header = TailHeader::decode(&hb)?;
    if key.is_some_and(|k| *k != header.key) {
        return Err(FormatError::Key);
    }
    // A demotion whose truncate ran but whose header rewrite failed leaves a
    // full header over a file that ends after section E: section E is
    // intact, so a read that does not want D is served (the startup scan
    // finishes the demotion).
    let half_demoted = !want_d && header.kind == TailKind::Full && !header.anchor && size == header.e_end();
    if !half_demoted {
        check_len(header.file_len(), size)?;
    }
    if &header.ns != ns {
        return Err(FormatError::Namespace);
    }
    if want_d && header.kind != TailKind::Full {
        return Err(FormatError::NoSectionD);
    }
    let prefix = read_exact_at(&f, header.e_offset() - TAIL_HEADER_LEN as u64, TAIL_HEADER_LEN as u64)?;
    let a = header.t / C * C;
    let (open, images) = parse_ids_and_images(&prefix, header.n_open as usize, header.n_images as usize, a)?;
    if images.iter().any(|r| r.start >= header.t) {
        return Err(FormatError::BadField("image past t"));
    }
    if keys::tail_step(&header.base, a, &open, &images) != header.key {
        return Err(FormatError::Key);
    }
    read_into(&f, sec_e, header.sec_e_len, header.e_offset())?;
    verify_section(sec_e, &header.sec_e_hash, "E")?;
    sec_d.clear();
    if want_d {
        read_into(&f, sec_d, header.sec_d_len, header.e_end())?;
        verify_section(sec_d, &header.sec_d_hash, "D")?;
    }
    Ok(TailMeta { header, open, images })
}

/// Read and verify a whole tail ([`read_tail_into`] with fresh buffers and
/// no key expectation).
pub fn read_tail(path: &Path, ns: &Key, want_d: bool) -> Result<TailFile, FormatError> {
    let (mut e, mut d) = (Vec::new(), Vec::new());
    let m = read_tail_into(path, ns, None, want_d, &mut e, &mut d)?;
    Ok(TailFile { header: m.header, open: m.open, images: m.images, sec_e: e, sec_d: want_d.then_some(d) })
}

/// The ids and image records of a file (no payload), for the startup scan's
/// ancestry pass. Verifies only the header checksum.
pub fn read_ids(path: &Path, is_tail: bool) -> Result<(u32, Vec<i32>, Vec<ImageRecord>), FormatError> {
    let f = File::open(path)?;
    let size = f.metadata()?.len();
    let hlen = if is_tail { TAIL_HEADER_LEN } else { CHUNK_HEADER_LEN } as u64;
    let hb = read_exact_at(&f, hlen.min(size), 0)?;
    let (a, n_ids, n_images, data_off) = if is_tail {
        let h = TailHeader::decode(&hb)?;
        (h.t / C * C, h.n_open as usize, h.n_images as usize, h.e_offset())
    } else {
        let h = ChunkHeader::decode(&hb)?;
        (h.k * C, h.n_tokens as usize, h.n_images as usize, h.data_offset())
    };
    if size < data_off {
        return Err(FormatError::Short { want: data_off, got: size });
    }
    let prefix = read_exact_at(&f, data_off - hlen, hlen)?;
    let (ids, images) = parse_ids_and_images(&prefix, n_ids, n_images, a)?;
    Ok((a, ids, images))
}

/// Tests: recompute a mutated chunk header's checksum in place.
#[cfg(test)]
pub(crate) fn reseal_chunk(b: &mut [u8]) {
    let sum = header_checksum(&b[..CHUNK_HEADER_LEN], None);
    b[CHUNK_HEADER_LEN - CHECKSUM_LEN..CHUNK_HEADER_LEN].copy_from_slice(&sum);
}

/// Tests: recompute a mutated tail header's checksum in place.
#[cfg(test)]
pub(crate) fn reseal_tail(b: &mut [u8]) {
    let sum = header_checksum(&b[..TAIL_HEADER_LEN], Some(TAIL_HITS_OFFSET as usize));
    b[TAIL_HEADER_LEN - CHECKSUM_LEN..TAIL_HEADER_LEN].copy_from_slice(&sum);
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub fn sample_chunk_header() -> ChunkHeader {
        let stores: Vec<StoreRows> = v41_stores().iter().map(StoreRows::chunk_of).collect();
        ChunkHeader {
            ns: [1; 32],
            key: [2; 32],
            parent: [3; 32],
            k: 7,
            n_tokens: C as u16,
            n_images: 0,
            payload_len: chunk_payload_len(&v41_stores()),
            stores,
            provenance: Provenance::Prefill,
            gen: GenPair { gen: 3, knob: [4; 16] },
            build: BuildId::parse("0123456789abcdef0123456789abcdef01234567-dirty").unwrap(),
            created: 1_790_000_000,
            payload_hash: [5; 32],
        }
    }

    pub fn sample_tail_header() -> TailHeader {
        TailHeader {
            ns: [1; 32],
            key: [2; 32],
            base: [3; 32],
            t: 5 * C + 77,
            kind: TailKind::Full,
            anchor: true,
            demoted: false,
            origin: TailOrigin::Anchor,
            n_open: 77,
            n_images: 0,
            n_raw: 128,
            n_raw_dec: 128,
            drafter: [6; 32],
            hits: 9,
            gen: GenPair { gen: 3, knob: [4; 16] },
            build: BuildId::V6,
            created: 1_790_000_000,
            sec_e_len: 1000,
            sec_e_hash: [7; 32],
            sec_d_len: 500,
            sec_d_hash: [8; 32],
            session_id: "sess-é".into(),
        }
    }

    #[test]
    fn design_sizes() {
        let s = v41_stores();
        assert_eq!(bytes_per_position(&s), 2760, "3.: 3 × (512 + 40) + (1,024 + 80)");
        assert_eq!(chunk_payload_len(&s), 2_826_240, "4.1 payload");
        let h = sample_chunk_header();
        assert_eq!(h.file_len(), 2_830_592, "4.1: header + 4 KiB ids + payload");
        assert_eq!(s.iter().map(|s| s.state_bytes).sum::<u32>(), 28 * 1024, "3.: accumulators 28 KiB");
        // 5.1: an encoder tail at a chunk boundary is 2.65 MB (28 KiB + 20 × 128 rows × 1 KiB).
        assert_eq!(section_e_len(&s, 8 * C, 128), 2_650_112);
        // ... plus 2,760 B per open position (exact at even t).
        assert_eq!(section_e_len(&s, 8 * C + 100, 128) - section_e_len(&s, 8 * C, 128), 100 * 2760);
        // Odd t: the open ratio-2 pair is in the accumulators, not a row.
        assert_eq!(section_e_len(&s, 8 * C + 101, 128), section_e_len(&s, 8 * C + 100, 128) + 1104);
        // A short prompt has a short window.
        assert_eq!(section_e_len(&s, 64, 64), 28 * 1024 + 64 * 2760 + 20 * 64 * 1024);
    }

    #[test]
    fn chunk_header_roundtrip_and_checksum() {
        let h = sample_chunk_header();
        let b = h.encode();
        assert_eq!(ChunkHeader::decode(&b).unwrap(), h);
        for i in 0..CHUNK_HEADER_LEN {
            let mut c = b;
            c[i] ^= 0x10;
            assert!(ChunkHeader::decode(&c).is_err(), "flip at byte {i} accepted");
        }
        assert!(matches!(ChunkHeader::decode(&b[..100]), Err(FormatError::Short { .. })));
    }

    #[test]
    fn tail_header_roundtrip_hits_outside_checksum() {
        let h = sample_tail_header();
        let b = h.encode();
        assert_eq!(TailHeader::decode(&b).unwrap(), h);
        // A hits pwrite leaves the checksum valid.
        let mut c = b;
        c[TAIL_HITS_OFFSET as usize..TAIL_HITS_OFFSET as usize + 4].copy_from_slice(&1234u32.to_le_bytes());
        let d = TailHeader::decode(&c).unwrap();
        assert_eq!(d.hits, 1234);
        assert_eq!(TailHeader { hits: h.hits, ..d }, h);
        // Any other flipped byte is caught.
        for i in (0..TAIL_HEADER_LEN).filter(|&i| !(160..164).contains(&i)) {
            let mut c = b;
            c[i] ^= 0x01;
            assert!(TailHeader::decode(&c).is_err(), "flip at byte {i} accepted");
        }
        // Long session ids are cut at a char boundary.
        let long = TailHeader { session_id: "é".repeat(40), ..h.clone() };
        let back = TailHeader::decode(&long.encode()).unwrap();
        assert_eq!(back.session_id.len(), 64);
        let dem = h.demoted(42);
        assert!(dem.kind == TailKind::Enc && dem.demoted && dem.sec_d_len == 0 && dem.hits == 42);
        assert_eq!(dem.file_len(), h.e_end());
    }

    #[test]
    fn build_id_parse() {
        let b = BuildId::parse("0123456789abcdef0123456789abcdef01234567").unwrap();
        assert!(!b.dirty && !b.v6);
        assert_eq!(b.to_string(), "0123456789abcdef0123456789abcdef01234567");
        assert!(BuildId::parse("0123").is_none());
        assert_eq!(BuildId::V6.to_string(), "v6");
    }

    /// 9b: every header validator, reached. A random flip trips the checksum
    /// first, so each case mutates ONE field and recomputes the checksum.
    #[test]
    fn header_validators_each_fire() {
        let good = sample_chunk_header().encode();
        let chunk_case = |at: usize, bytes: &[u8]| {
            let mut b = good;
            b[at..at + bytes.len()].copy_from_slice(bytes);
            reseal_chunk(&mut b);
            ChunkHeader::decode(&b)
        };
        assert_eq!(chunk_case(0, b"X"), Err(FormatError::BadMagic));
        assert_eq!(chunk_case(5, &[2]), Err(FormatError::BadVersion(2)));
        assert_eq!(chunk_case(6, &255u16.to_le_bytes()), Err(FormatError::BadField("header_len")));
        assert_eq!(chunk_case(112, &[5]), Err(FormatError::BadField("n_stores")));
        assert_eq!(chunk_case(113, &[9]), Err(FormatError::BadField("provenance")));
        assert_eq!(chunk_case(200, &7u64.to_le_bytes()), Err(FormatError::BadField("payload_len")));
        assert_eq!(chunk_case(108, &1000u16.to_le_bytes()), Err(FormatError::BadField("n_tokens")));
        let mut b = good;
        b[150] ^= 1; // inside the checksummed body, not resealed
        assert_eq!(ChunkHeader::decode(&b), Err(FormatError::HeaderChecksum));

        let good = sample_tail_header().encode();
        let tail_case = |edits: &[(usize, &[u8])]| {
            let mut b = good;
            for (at, bytes) in edits {
                b[*at..*at + bytes.len()].copy_from_slice(bytes);
            }
            reseal_tail(&mut b);
            TailHeader::decode(&b)
        };
        assert_eq!(tail_case(&[(108, &[7])]), Err(FormatError::BadField("kind")));
        assert_eq!(tail_case(&[(110, &[9])]), Err(FormatError::BadField("origin")));
        assert_eq!(tail_case(&[(109, &[0x80])]), Err(FormatError::BadField("flags")));
        assert_eq!(tail_case(&[(111, &[65])]), Err(FormatError::BadField("flags")));
        assert_eq!(tail_case(&[(112, &78u16.to_le_bytes())]), Err(FormatError::BadField("n_open")));
        assert_eq!(tail_case(&[(108, &[0])]), Err(FormatError::BadField("enc tail with section D")));
        assert_eq!(tail_case(&[(109, &[TAIL_FLAG_ANCHOR | TAIL_FLAG_DEMOTED])]), Err(FormatError::BadField("demoted anchor")));
        // hits is outside the checksum and clamped on read.
        let h = tail_case(&[(160, &u32::MAX.to_le_bytes())]).unwrap();
        assert_eq!(h.hits, MAX_HITS);
    }

    /// The file-level validators of the readers (ids, images, key, lengths,
    /// namespace, section D), each reached with a self-consistent header.
    #[test]
    fn reader_validators_each_fire() {
        let dir = crate::kvstore::io::tests::unique_dir("fmt-val");
        let stores = v41_stores();
        let ns = [1u8; 32];
        let tokens: Vec<i32> = (0..C as i32).collect();
        let write_chunk = |name: &str, k: u32, images: &[ImageRecord], key_images: &[ImageRecord]| -> std::path::PathBuf {
            let payload = vec![3u8; chunk_payload_len(&stores) as usize];
            let mut h = sample_chunk_header();
            h.k = k;
            h.n_images = images.len() as u16;
            h.key = keys::chunk_step(&h.parent, k.wrapping_mul(C), &tokens, key_images);
            h.payload_hash = *blake3::hash(&payload).as_bytes();
            let mut b = Vec::new();
            b.extend_from_slice(&h.encode());
            for t in &tokens {
                b.extend_from_slice(&t.to_le_bytes());
            }
            for r in images {
                // Raw records (relative to k*C, wrapping): may be malformed on purpose.
                let mut rec = [0u8; 40];
                rec[0..4].copy_from_slice(&r.start.wrapping_sub(k.wrapping_mul(C)).to_le_bytes());
                rec[4..8].copy_from_slice(&r.len.to_le_bytes());
                rec[8..].copy_from_slice(&r.hash);
                b.extend_from_slice(&rec);
            }
            b.extend_from_slice(&payload);
            let p = dir.join(name);
            std::fs::write(&p, &b).unwrap();
            p
        };
        let img = |start: u32| ImageRecord { start, len: 5, hash: [4; 32] };
        // Baseline: valid.
        let ok_imgs = [img(7 * C + 10), img(7 * C + 20)];
        let p = write_chunk("ok.kvc", 7, &ok_imgs, &ok_imgs);
        let f = read_chunk(&p, &ns).unwrap();
        assert_eq!(f.images, ok_imgs);
        assert_eq!(read_chunk(&p, &[2; 32]), Err(FormatError::Namespace));
        assert_eq!(read_chunk_into(&p, &ns, Some(&[9; 32]), &mut Vec::new()).map(|_| ()), Err(FormatError::Key));
        // Out-of-order records, and one past the chunk.
        let rev = [img(7 * C + 20), img(7 * C + 10)];
        let p = write_chunk("order.kvc", 7, &rev, &[]);
        assert_eq!(read_chunk(&p, &ns), Err(FormatError::BadField("image order")));
        let p = write_chunk("past.kvc", 7, &[img(8 * C + 1)], &[]);
        assert_eq!(read_chunk(&p, &ns), Err(FormatError::BadField("image order")));
        // A record whose absolute start overflows u32.
        let k_big = u32::MAX / C;
        let p = write_chunk("ovf.kvc", k_big, &[ImageRecord { start: k_big * C - 1, len: 1, hash: [0; 32] }], &[]);
        assert_eq!(read_chunk(&p, &ns), Err(FormatError::BadField("image")));
        // k * C overflows.
        let p = write_chunk("k.kvc", u32::MAX / C + 1, &[], &[]);
        assert_eq!(read_chunk(&p, &ns), Err(FormatError::BadField("k")));
        // Ids changed after the key was taken.
        let p = write_chunk("ids.kvc", 7, &[], &[]);
        let mut b = std::fs::read(&p).unwrap();
        b[CHUNK_HEADER_LEN + 4] ^= 1;
        std::fs::write(&p, &b).unwrap();
        assert_eq!(read_chunk(&p, &ns), Err(FormatError::Key));
        // Long and short files.
        let p = write_chunk("long.kvc", 7, &[], &[]);
        let mut b = std::fs::read(&p).unwrap();
        b.push(0);
        std::fs::write(&p, &b).unwrap();
        assert!(matches!(read_chunk(&p, &ns), Err(FormatError::Long { .. })));
        b.truncate(b.len() - 100);
        std::fs::write(&p, &b).unwrap();
        assert!(matches!(read_chunk(&p, &ns), Err(FormatError::Short { .. })));

        // Tails: an image record past t, section D asked of an encoder tail.
        let write_tail = |name: &str, kind: TailKind, images: &[ImageRecord]| -> std::path::PathBuf {
            let t = 3 * C + 50;
            let open: Vec<i32> = (0..50).collect();
            let sec_e = vec![1u8; 100];
            let sec_d = if kind == TailKind::Full { vec![2u8; 60] } else { vec![] };
            let mut h = sample_tail_header();
            h.anchor = false;
            h.kind = kind;
            h.t = t;
            h.n_open = 50;
            h.n_images = images.len() as u16;
            h.n_raw_dec = if kind == TailKind::Full { 128 } else { 0 };
            h.key = keys::tail_step(&h.base, 3 * C, &open, &[]);
            h.sec_e_len = sec_e.len() as u64;
            h.sec_e_hash = *blake3::hash(&sec_e).as_bytes();
            h.sec_d_len = sec_d.len() as u64;
            h.sec_d_hash = if sec_d.is_empty() { [0; 32] } else { *blake3::hash(&sec_d).as_bytes() };
            let mut b = encode_tail_prefix(&h, &open, images);
            b.extend_from_slice(&sec_e);
            b.extend_from_slice(&sec_d);
            let p = dir.join(name);
            std::fs::write(&p, &b).unwrap();
            p
        };
        let p = write_tail("past.kvt", TailKind::Full, &[img(3 * C + 60)]);
        assert_eq!(read_tail(&p, &ns, false), Err(FormatError::BadField("image past t")));
        let p = write_tail("enc.kvt", TailKind::Enc, &[]);
        assert!(read_tail(&p, &ns, false).is_ok());
        assert_eq!(read_tail(&p, &ns, true), Err(FormatError::NoSectionD));
        let p = write_tail("full.kvt", TailKind::Full, &[]);
        let t = read_tail(&p, &ns, true).unwrap();
        assert_eq!(t.sec_d.unwrap().len(), 60);
        // Half a demotion (truncate ran, header rewrite did not): section E is
        // served, section D is not.
        let half = dir.join("half.kvt");
        std::fs::copy(&p, &half).unwrap();
        std::fs::OpenOptions::new().write(true).open(&half).unwrap().set_len(t.header.e_end()).unwrap();
        assert_eq!(read_tail(&half, &ns, false).unwrap().sec_e, t.sec_e);
        assert!(matches!(read_tail(&half, &ns, true), Err(FormatError::Short { .. })));
        assert_eq!(read_tail_into(&p, &ns, Some(&[0; 32]), false, &mut Vec::new(), &mut Vec::new()).map(|_| ()), Err(FormatError::Key));
        // Transient IO errors are not verdicts; data errors are.
        assert!(!FormatError::Io(std::io::ErrorKind::PermissionDenied, String::new()).evicts());
        assert!(!FormatError::Io(std::io::ErrorKind::Other, String::new()).evicts());
        assert!(FormatError::Io(std::io::ErrorKind::UnexpectedEof, String::new()).evicts());
        assert!(FormatError::Io(std::io::ErrorKind::NotFound, String::new()).evicts());
        assert!(FormatError::Key.evicts());
    }
}
