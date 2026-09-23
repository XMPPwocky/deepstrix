//! On-disk KV-cache snapshots + LRU eviction.
//!
//! A snapshot is keyed by BLAKE3 over the token-id sequence it
//! represents. On disk it lives at
//! `~/.cache/deepstrix/snapshots/<hex(blake3)>/`:
//!
//!   meta.json       — schema-version'd metadata + per-layer counts
//!   tokens.bin      — i32-LE token sequence (token_count entries)
//!   kv.bin          — concatenated raw kv_cache, per layer
//!   comp_kv.bin     — concatenated cumulative comp_kv, per compressor layer
//!   comp_state.bin  — concatenated (state_kv ++ state_score) f32 blocks,
//!                     per compressor layer
//!
//! Snapshots are written when the worker is about to switch live
//! conversations (and the live state hasn't been saved since its
//! last change), and at shutdown. They are looked up by walking
//! per-turn EOS boundaries of an incoming request, hashing each
//! prefix, and picking the longest match.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use color_eyre::eyre::{self, eyre};
use serde::{Deserialize, Serialize};
use v4flash_core::tokenizer::BpeVocab;
use v4flash_hip::Device;
use v4flash_kernels::config::{COMPRESS_RATIOS, N_HEAD_DIM, N_LAYER, NEG_INF, SWA_WINDOW};
use v4flash_kernels::het::HetModelState;

use v4flash_kernels::comp_kv_fp8::{FP8_KV_HEAD_ROWS, FP8_KV_ROW_BYTES};
use v4flash_kernels::index_kv_e2m1::E2M1_KEY_ROW_BYTES;
use v4flash_kernels::het::state::CompKvStore;

use crate::embed::gpt2_decode_token;
use crate::vision_prompt::{span_hash_at, synthetic_token_bytes, ImageSpan};

/// Bumped to 2 when snapshot keys switched from `blake3(token_id_LE_bytes)`
/// to `blake3(decoded_byte_stream)` — bytes are the source of truth and
/// survive tokenizer-roundtrip splits.
// v2 → v3: added per-layer indexer compressor state (`has_indexer_compressor`,
// `n_index_comp`, `index_*` shape fields) + index_comp_kv.bin +
// index_comp_state.bin blobs. v2 snapshots get evicted at startup since
// they lack the indexer state needed for correct ratio==4 attention at
// long context.
// v3 → v4: per-layer `comp_kv_format` / `comp_kv_row_bytes` (the ratio-4
// main compressors store packed FP8 rows, `comp_kv_fp8.rs`; -42% on
// comp_kv.bin).
// v4 → v5: per-layer `index_comp_kv_format` / `index_comp_kv_row_bytes`
// (the ratio-4 indexer compressors store packed E2M1 rows,
// `index_kv_e2m1.rs`; -69% on index_comp_kv.bin). Snapshots are a CACHE:
// no cross-format conversion exists any more (v4's f16->FP8 conversion was
// removed with it) — a file whose encoding does not match the live store,
// or an older version, is refused and evicted, and the session prefills
// from scratch once.
// v5 -> v6: per-layer V4.1 sparse-indexer KEY store (`has_index_k`, `n_index_k`)
// + index_k.bin. V4.1 keeps its index keys in the MAIN compressor
// (`HetCompressorState::index_k` / `n_index_comp`), not in a separate
// `indexer_compressor` — that struct is only allocated at ratio==4 and V4.1 has
// no ratio-4 layer, so the v3 indexer blobs are always empty under V4.1 and the
// keys were silently NOT persisted. A restored session therefore came back with
// `n_index_comp == 0`, which fails the decode gate
// (`n_index_comp > INDEXER_TOP_K`), so no index-source layer gathered, nothing
// was published for S2, and all 40 layers scored densely — `V41_INDEX_K=1` was
// inert on any prefix-cache hit. MEASURED before this fix: sel_sync 22.6 ms at
// 8K -> 44.8 ms at 46K, i.e. attention scaling linearly with context, which is
// the dense signature.
const FORMAT_VERSION: u32 = 6;
/// Oldest format `restore` accepts.
const MIN_FORMAT_VERSION: u32 = 6;

/// On-disk encoding of one compressor's `comp_kv` rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CompKvFormat {
    /// `[n_comp, head_dim]` f16 little-endian (v3, and the ratio-128 /
    /// indexer compressors in v4).
    #[default]
    F16,
    /// Packed rows of `FP8_KV_ROW_BYTES` (E4M3 codes, block exponents,
    /// f16 RoPE tail) — `v4flash_kernels::comp_kv_fp8`.
    Fp8E4m3B64,
    /// Packed rows of `E2M1_KEY_ROW_BYTES` (E2M1 nibbles + per-32 block
    /// exponents) — `v4flash_kernels::index_kv_e2m1` (indexer keys only).
    E2m1B32,
}

impl CompKvFormat {
    fn of(store: &CompKvStore) -> CompKvFormat {
        match store {
            CompKvStore::F16(_) => CompKvFormat::F16,
            CompKvStore::Fp8 { .. } => CompKvFormat::Fp8E4m3B64,
            CompKvStore::E2m1(_) => CompKvFormat::E2m1B32,
        }
    }
    /// Bytes per row of `head_dim` elements in this encoding.
    fn row_bytes(self, head_dim: usize) -> usize {
        match self {
            CompKvFormat::F16 => head_dim * 2,
            CompKvFormat::Fp8E4m3B64 => FP8_KV_ROW_BYTES,
            CompKvFormat::E2m1B32 => E2M1_KEY_ROW_BYTES,
        }
    }
}

/// Decode a token-id sequence to the raw byte stream the model would
/// see at the surface level. Used for snapshot keys + byte-level
/// prefix matching across tokenization differences.
pub fn decode_tokens_to_bytes(
    tokens: &[i32],
    vocab: &BpeVocab,
    byte_decoder: &std::collections::HashMap<char, u8>,
) -> Vec<u8> {
    decode_tokens_to_bytes_vl(tokens, &[], vocab, byte_decoder)
}

/// [`decode_tokens_to_bytes`] for a stream that may contain image blocks.
/// Synthetic image ids have no vocab text; they decode to a per-type
/// marker, and IMAGE_START additionally carries the image's content hash
/// (looked up in `image_spans` by position) — so the byte stream, and
/// therefore the snapshot key, is different for different pixels.
pub fn decode_tokens_to_bytes_vl(
    tokens: &[i32],
    image_spans: &[ImageSpan],
    vocab: &BpeVocab,
    byte_decoder: &std::collections::HashMap<char, u8>,
) -> Vec<u8> {
    decode_tokens_to_bytes_with(tokens, image_spans, |id| {
        vocab
            .token_text(id)
            .map(|b| gpt2_decode_token(b, byte_decoder))
            .unwrap_or_default()
    })
}

/// [`decode_tokens_to_bytes_vl`] with an injectable per-id text decoder
/// (unit-testable without a vocab). `decode` is only consulted for
/// non-synthetic ids.
pub fn decode_tokens_to_bytes_with(
    tokens: &[i32],
    image_spans: &[ImageSpan],
    decode: impl Fn(i32) -> Vec<u8>,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(tokens.len() * 2);
    for (i, &id) in tokens.iter().enumerate() {
        if let Some(b) = synthetic_token_bytes(id, span_hash_at(image_spans, i)) {
            out.extend(b);
        } else {
            out.extend(decode(id));
        }
    }
    out
}

/// Identifies which model produced a snapshot — guards against
/// silently restoring weights from a different GGUF. We don't blake
/// the whole weights file (multi-GB), just enough fields to fail
/// loud when a different model is loaded.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelFingerprint {
    pub n_layer: u32,
    pub n_head_dim: u32,
    pub vocab_size: u32,
    /// blake3 of the first 4 KiB of `token_embd.weight` bytes.
    pub token_embd_prefix_blake3: String,
    /// blake3 over every tensor's (name, dtype id, byte_size), in
    /// directory order. The embd prefix alone can collide between two
    /// quants of the same checkpoint that share an embedding table (e.g.
    /// a requant touching only expert tensors); the directory hash is
    /// guaranteed to differ whenever any tensor's dtype or size changes.
    /// `serde(default)`: indexes written before this field deserialize
    /// with "" and simply mismatch → snapshots quarantined once.
    #[serde(default)]
    pub tensor_directory_blake3: String,
}

impl ModelFingerprint {
    pub fn compute(
        vocab_size: u32,
        token_embd_bytes: &[u8],
        tensors: &[v4flash_core::gguf::GgufTensor],
    ) -> Self {
        let prefix_len = token_embd_bytes.len().min(4096);
        let hash = blake3::hash(&token_embd_bytes[..prefix_len]);
        let mut dir = blake3::Hasher::new();
        for t in tensors {
            dir.update(t.name.as_bytes());
            dir.update(&[0]);
            dir.update(t.dtype.name().as_bytes());
            dir.update(&t.byte_size.to_le_bytes());
        }
        Self {
            n_layer: N_LAYER as u32,
            n_head_dim: N_HEAD_DIM,
            vocab_size,
            token_embd_prefix_blake3: hash.to_hex().to_string(),
            tensor_directory_blake3: dir.finalize().to_hex().to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerLayerMeta {
    pub n_raw: u32,
    /// Number of raw KV rows actually written to `kv.bin` for this layer.
    /// = min(n_raw, SWA_WINDOW).
    pub kv_rows: u32,
    pub has_compressor: bool,
    pub n_comp: u32,
    pub ratio: u32,
    pub coff: u32,
    pub width: u32,
    pub head_dim: u32,
    pub state_rows: u32,
    /// CSA indexer compressor state (only on ratio==4 layers). When
    /// `has_indexer_compressor` is false the `index_*` fields are 0 /
    /// undefined and no indexer bytes are written for this layer.
    #[serde(default)]
    pub has_indexer_compressor: bool,
    #[serde(default)]
    pub n_index_comp: u32,
    #[serde(default)]
    pub index_coff: u32,
    #[serde(default)]
    pub index_width: u32,
    #[serde(default)]
    pub index_head_dim: u32,
    #[serde(default)]
    pub index_state_rows: u32,
    /// v6: V4.1 sparse-indexer key store, which lives in the MAIN compressor
    /// (`index_k` / `n_index_comp`) rather than in `indexer_compressor`.
    /// Without this the indexer cannot engage after a restore — see the
    /// version history above.
    #[serde(default)]
    pub has_index_k: bool,
    #[serde(default)]
    pub n_index_k: u32,
    /// v4: encoding of this layer's main-compressor rows in comp_kv.bin.
    /// Absent (v3) = f16.
    #[serde(default)]
    pub comp_kv_format: CompKvFormat,
    /// v4: bytes per row in comp_kv.bin for this layer (0 = `head_dim * 2`
    /// f16).
    #[serde(default)]
    pub comp_kv_row_bytes: u32,
    /// v5: encoding of this layer's indexer-compressor rows in
    /// index_comp_kv.bin. Absent = f16.
    #[serde(default)]
    pub index_comp_kv_format: CompKvFormat,
    #[serde(default)]
    pub index_comp_kv_row_bytes: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotMeta {
    pub format_version: u32,
    pub fingerprint: ModelFingerprint,
    pub token_count: u32,
    pub n_kv_max: u32,
    pub created_at_unix: u64,
    pub last_used_unix: u64,
    pub layers: Vec<PerLayerMeta>,
    /// Total bytes on disk for this snapshot (sum of files).
    pub disk_bytes: u64,
    /// Conversation lineage this snapshot belongs to (letta `session_id`).
    /// Used for per-lineage retention (keep the N most-recent per session).
    /// Absent on pre-retention snapshots → treated as a singleton lineage.
    #[serde(default)]
    pub session_id: Option<String>,
    /// Image spans inside `tokens.bin` (Vision-Exp). Needed to re-derive
    /// the byte-stream key and to compare image content on restore.
    /// Absent on pre-vision snapshots (which contain no image tokens).
    #[serde(default)]
    pub image_spans: Vec<ImageSpan>,
}

/// In-memory record of one on-disk snapshot.
/// Output of [`SnapshotIndex::diag_largest_divergence`].
#[derive(Debug, Clone)]
pub struct DiagDivergence {
    pub snap_token_count: u32,
    pub snap_byte_len: usize,
    pub req_byte_len: usize,
    pub common_byte_len: usize,
    pub before: String,
    pub snap_after: String,
    pub req_after: String,
}

#[derive(Debug, Clone)]
pub struct IndexEntry {
    pub hash: [u8; 32],
    pub token_count: u32,
    pub last_used_unix: u64,
    pub disk_bytes: u64,
    pub dir: PathBuf,
    /// Lineage key (letta `session_id`); `None` = singleton lineage.
    pub session_id: Option<String>,
}

pub struct SnapshotIndex {
    root: PathBuf,
    /// The fingerprint of the model currently loaded; entries on disk
    /// that don't match are skipped at load time.
    #[allow(dead_code)]
    fingerprint: ModelFingerprint,
    /// Hash → entry.
    by_hash: HashMap<[u8; 32], IndexEntry>,
    /// (last_used_unix, hash) sorted by time for LRU eviction.
    by_last_used: BTreeMap<(u64, [u8; 32]), ()>,
    /// session_id → hashes of that lineage's snapshots (unordered;
    /// recency comes from `by_hash[h].last_used_unix`). Drives the
    /// per-lineage retention cap (R1).
    by_session: HashMap<String, Vec<[u8; 32]>>,
    /// Max snapshots retained per lineage (R1). Env-tunable via
    /// `DEEPSTRIX_SNAPSHOT_KEEP_PER_SESSION`, default 3.
    max_per_lineage: usize,
    total_bytes: u64,
    /// Soft cap; eviction targets get to ≤ this.
    pub cap_bytes: u64,
    /// Hint cache: sessionId → most recent hash for that conversation.
    /// Letta passes `session_id` per request; we use it as a fast-path
    /// for the common single-conversation pattern.
    pub session_to_hash: HashMap<String, [u8; 32]>,
}

impl SnapshotIndex {
    pub fn new(root: PathBuf, fingerprint: ModelFingerprint, cap_bytes: u64) -> Self {
        let max_per_lineage = std::env::var("DEEPSTRIX_SNAPSHOT_KEEP_PER_SESSION")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(3);
        Self {
            root,
            fingerprint,
            by_hash: HashMap::new(),
            by_last_used: BTreeMap::new(),
            by_session: HashMap::new(),
            max_per_lineage,
            total_bytes: 0,
            cap_bytes,
            session_to_hash: HashMap::new(),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    pub fn len(&self) -> usize {
        self.by_hash.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_hash.is_empty()
    }

    /// Walk an existing snapshot root, populating the index. Entries
    /// whose fingerprint mismatches the current model are quarantined
    /// (logged + skipped, not deleted — user might be switching back).
    pub fn load(
        root: PathBuf,
        fingerprint: ModelFingerprint,
        cap_bytes: u64,
    ) -> eyre::Result<Self> {
        let mut idx = Self::new(root.clone(), fingerprint.clone(), cap_bytes);
        if !root.exists() {
            return Ok(idx);
        }
        let mut skipped_fingerprint = 0usize;
        for entry in fs::read_dir(&root)? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let meta_path = path.join("meta.json");
            if !meta_path.exists() {
                continue;
            }
            let Ok(bytes) = fs::read(&meta_path) else { continue };
            let Ok(meta): Result<SnapshotMeta, _> = serde_json::from_slice(&bytes) else {
                tracing::warn!(path = ?path, "snapshot meta.json unparseable; skipping");
                continue;
            };
            if meta.format_version < MIN_FORMAT_VERSION {
                // An older-format entry is an invalid cache entry: delete
                // it now, or it would sit outside the disk cap's accounting
                // forever (the index only counts entries it accepted).
                match fs::remove_dir_all(&path) {
                    Ok(()) => tracing::warn!(
                        path = ?path,
                        saw = meta.format_version,
                        min = MIN_FORMAT_VERSION,
                        "snapshot format too old; deleted"
                    ),
                    Err(e) => tracing::warn!(path = ?path, error = %e, "failed to delete old-format snapshot"),
                }
                continue;
            }
            if meta.format_version > FORMAT_VERSION {
                tracing::warn!(
                    path = ?path,
                    saw = meta.format_version,
                    want = FORMAT_VERSION,
                    "snapshot format newer than this binary; skipping"
                );
                continue;
            }
            if meta.fingerprint != fingerprint {
                skipped_fingerprint += 1;
                continue;
            }
            let Some(hash_hex) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let Some(hash) = hex_to_blake3(hash_hex) else {
                continue;
            };
            let e = IndexEntry {
                hash,
                token_count: meta.token_count,
                last_used_unix: meta.last_used_unix,
                disk_bytes: meta.disk_bytes,
                dir: path,
                session_id: meta.session_id.clone(),
            };
            idx.total_bytes += e.disk_bytes;
            idx.by_last_used.insert((e.last_used_unix, e.hash), ());
            if let Some(sid) = &e.session_id {
                idx.by_session.entry(sid.clone()).or_default().push(e.hash);
            }
            idx.by_hash.insert(e.hash, e);
        }
        tracing::info!(
            count = idx.by_hash.len(),
            total_bytes = idx.total_bytes,
            cap_bytes = idx.cap_bytes,
            skipped_fingerprint,
            "snapshot index loaded"
        );
        Ok(idx)
    }

    /// Update last_used_unix on an entry (LRU touch).
    pub fn touch(&mut self, hash: &[u8; 32]) -> eyre::Result<()> {
        let Some(entry) = self.by_hash.get_mut(hash) else {
            return Ok(());
        };
        let old_key = (entry.last_used_unix, entry.hash);
        let now = unix_now();
        entry.last_used_unix = now;
        self.by_last_used.remove(&old_key);
        self.by_last_used.insert((now, *hash), ());
        // Update meta.json's last_used_unix on disk too.
        let meta_path = entry.dir.join("meta.json");
        if let Ok(bytes) = fs::read(&meta_path) {
            if let Ok(mut meta) = serde_json::from_slice::<SnapshotMeta>(&bytes) {
                meta.last_used_unix = now;
                if let Ok(out) = serde_json::to_vec_pretty(&meta) {
                    let _ = fs::write(&meta_path, out);
                }
            }
        }
        Ok(())
    }

    /// Fully remove one snapshot: drop it from every index, subtract its
    /// bytes, and delete its on-disk dir. Central path for both eviction
    /// rules so the maps never drift.
    /// Drop one snapshot (index entry + directory), e.g. after a failed
    /// restore, so the next matching request does not re-read it.
    pub fn evict(&mut self, hash: &[u8; 32], reason: &str) {
        self.remove_entry(hash, reason);
    }

    fn remove_entry(&mut self, hash: &[u8; 32], reason: &str) {
        let Some(entry) = self.by_hash.remove(hash) else {
            return;
        };
        self.by_last_used.remove(&(entry.last_used_unix, *hash));
        if let Some(sid) = &entry.session_id {
            if let Some(v) = self.by_session.get_mut(sid) {
                v.retain(|h| h != hash);
                if v.is_empty() {
                    self.by_session.remove(sid);
                }
            }
            // Drop the hot-cache hint if it pointed at this snapshot.
            if self.session_to_hash.get(sid) == Some(hash) {
                self.session_to_hash.remove(sid);
            }
        }
        self.total_bytes = self.total_bytes.saturating_sub(entry.disk_bytes);
        if let Err(e) = fs::remove_dir_all(&entry.dir) {
            tracing::warn!(dir = ?entry.dir, error = %e, "failed to remove evicted snapshot dir");
        } else {
            tracing::info!(
                hash = %hex::encode(entry.hash),
                bytes = entry.disk_bytes,
                reason,
                "evicted snapshot"
            );
        }
    }

    /// R1: cap a lineage at `max_per_lineage`, evicting its oldest
    /// (lowest `last_used_unix`) snapshots beyond the cap. The newest
    /// (continuation tip) is always retained.
    fn evict_lineage_overflow(&mut self, session_id: &str) {
        loop {
            let to_evict = {
                let Some(members) = self.by_session.get(session_id) else {
                    return;
                };
                if members.len() <= self.max_per_lineage {
                    return;
                }
                members
                    .iter()
                    .copied()
                    .min_by_key(|h| self.by_hash.get(h).map(|e| e.last_used_unix).unwrap_or(0))
            };
            match to_evict {
                Some(h) => self.remove_entry(&h, "lineage cap"),
                None => return,
            }
        }
    }

    /// R4: global backstop. Pop globally-LRU entries until
    /// total_bytes <= cap_bytes (evicts whole least-recently-used
    /// lineages first, since their snapshots carry the oldest timestamps).
    pub fn evict_to_fit(&mut self) {
        while self.total_bytes > self.cap_bytes {
            let Some((ts, hash)) = self.by_last_used.iter().next().map(|(k, _)| *k) else {
                break;
            };
            if self.by_hash.contains_key(&hash) {
                self.remove_entry(&hash, "global cap");
            } else {
                // Stale key with no entry — drop it and continue.
                self.by_last_used.remove(&(ts, hash));
            }
        }
    }

    pub fn insert(&mut self, entry: IndexEntry) {
        // De-dup: re-saving the same hash replaces the prior record so we
        // never double-count bytes or duplicate lineage membership.
        if let Some(old) = self.by_hash.remove(&entry.hash) {
            self.by_last_used.remove(&(old.last_used_unix, old.hash));
            if let Some(sid) = &old.session_id {
                if let Some(v) = self.by_session.get_mut(sid) {
                    v.retain(|h| *h != old.hash);
                }
            }
            self.total_bytes = self.total_bytes.saturating_sub(old.disk_bytes);
        }
        let session_id = entry.session_id.clone();
        self.total_bytes += entry.disk_bytes;
        self.by_last_used
            .insert((entry.last_used_unix, entry.hash), ());
        if let Some(sid) = &session_id {
            self.by_session
                .entry(sid.clone())
                .or_default()
                .push(entry.hash);
        }
        self.by_hash.insert(entry.hash, entry);
        // R1: per-lineage cap, then R4: global cap backstop.
        if let Some(sid) = &session_id {
            self.evict_lineage_overflow(sid);
        }
        self.evict_to_fit();
    }

    /// Diagnostic: pick the snapshot in the index whose stored
    /// token_count is largest, load its tokens.bin, decode to bytes,
    /// and compute the byte-position at which the snapshot's byte
    /// stream first diverges from `req_tokens`' byte stream. Returns
    /// the snapshot's token count, the divergence byte offset, and
    /// short hex slices of the bytes before/at/after divergence on
    /// each side. Used to investigate why save-every-turn snapshots
    /// don't byte-match what letta replays. Returns None when there
    /// is no candidate larger than `min_token_count`.
    pub fn diag_largest_divergence(
        &self,
        req_tokens: &[i32],
        min_token_count: u32,
        vocab: &BpeVocab,
        byte_decoder: &std::collections::HashMap<char, u8>,
    ) -> Option<DiagDivergence> {
        let entry = self
            .by_hash
            .values()
            .filter(|e| e.token_count > min_token_count)
            .max_by_key(|e| e.token_count)?;
        let tokens_path = entry.dir.join("tokens.bin");
        let raw = std::fs::read(&tokens_path).ok()?;
        if raw.len() % 4 != 0 {
            return None;
        }
        let mut snap_tokens: Vec<i32> = Vec::with_capacity(raw.len() / 4);
        for c in raw.chunks_exact(4) {
            snap_tokens.push(i32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        }
        let snap_bytes = decode_tokens_to_bytes(&snap_tokens, vocab, byte_decoder);
        let req_bytes = decode_tokens_to_bytes(req_tokens, vocab, byte_decoder);
        let common_len = snap_bytes
            .iter()
            .zip(req_bytes.iter())
            .take_while(|(a, b)| a == b)
            .count();
        let ctx_before = 64usize.min(common_len);
        let snap_after_end = (common_len + 64).min(snap_bytes.len());
        let req_after_end = (common_len + 64).min(req_bytes.len());
        Some(DiagDivergence {
            snap_token_count: entry.token_count,
            snap_byte_len: snap_bytes.len(),
            req_byte_len: req_bytes.len(),
            common_byte_len: common_len,
            before: String::from_utf8_lossy(
                &snap_bytes[common_len.saturating_sub(ctx_before)..common_len],
            )
            .into_owned(),
            snap_after: String::from_utf8_lossy(&snap_bytes[common_len..snap_after_end])
                .into_owned(),
            req_after: String::from_utf8_lossy(&req_bytes[common_len..req_after_end])
                .into_owned(),
        })
    }

    /// Walk `tokens` looking for the longest prefix `tokens[..i]` whose
    /// byte-decoded form (`blake3(decode(tokens[..i]))`) matches an
    /// on-disk snapshot. Probes at turn-boundary indices: positions
    /// `i` where `tokens[i-1]` is `TOK_EOS` (end-of-message),
    /// `TOK_ASSISTANT` (start of an assistant turn — the canonical
    /// save point for the start-of-think snapshot), or `TOK_USER`
    /// (start of a user turn). `TOK_ASSISTANT` is essential:
    /// post-`<think>`-design snapshots end at `<Assistant>` (not
    /// EOS), and probing only at EOS would never find them.
    /// `TOK_USER` is what makes the *system prompt* reusable across
    /// conversations: a fresh conversation shares only
    /// `[BOS + system + tools + <User>]` with anything already on
    /// disk, and diverges at the first byte of the user message. That
    /// boundary is the sole probe point a new conversation can hit,
    /// and it pairs with the system-prefix save in
    /// `handle_generate_stream`'s full-prefill path.
    ///
    /// Returns `(req_prefix_len, hash, dir)` of the largest match —
    /// where `req_prefix_len` is the req-side token count for the
    /// byte boundary that hashed (the snapshot's stored token count
    /// may differ, since the same bytes can split differently).
    ///
    /// `image_spans` (Vision-Exp): the request's image spans, so the
    /// probed byte prefix carries each image's content hash exactly as
    /// `save` hashed it. Image blocks live strictly inside a user turn
    /// (after `<User>`, before `<Assistant>`), so the probe points are
    /// never inside a block and a snapshot saved at `<Assistant>` after
    /// an image turn hashes identically here.
    pub fn find_longest_prefix(
        &self,
        tokens: &[i32],
        image_spans: &[ImageSpan],
        tok_eos: i32,
        tok_assistant: i32,
        tok_user: i32,
        vocab: &BpeVocab,
        byte_decoder: &std::collections::HashMap<char, u8>,
    ) -> Option<(usize, [u8; 32], PathBuf)> {
        let mut best: Option<(usize, [u8; 32], PathBuf)> = None;
        // ONE running hasher, cloned at each boundary.
        //
        // This used to accumulate a `byte_prefix: Vec<u8>` and call
        // `blake3::hash(&byte_prefix)` at EVERY boundary token -- re-hashing the
        // whole prefix from byte zero each time, so O(prefix x boundaries). On a
        // 348-message request that is ~700 boundaries against a prefix reaching
        // ~1.4 MB: ~500 MB hashed to produce 700 hashes, ~0.3-0.5 s single
        // threaded, which is about the measured p50 of the whole pre-prefill
        // window. Incremental hashing makes it ~1.4 MB.
        //
        // blake3 guarantees `Hasher::new().update(x).finalize() == hash(x)`, so
        // the keys are BIT-IDENTICAL -- which they must be: these hashes address
        // every snapshot already on disk (108 GB, 521 entries). Guarded by
        // `incremental_hash_matches_whole_prefix_hash` below.
        let mut hasher = blake3::Hasher::new();
        let decode = |id: i32| -> Vec<u8> {
            vocab
                .token_text(id)
                .map(|b| gpt2_decode_token(b, byte_decoder))
                .unwrap_or_default()
        };
        for (i, &tok) in tokens.iter().enumerate() {
            match synthetic_token_bytes(tok, span_hash_at(image_spans, i)) {
                Some(b) => hasher.update(&b),
                None => hasher.update(&decode(tok)),
            };
            if tok == tok_eos || tok == tok_assistant || tok == tok_user {
                let h = *hasher.clone().finalize().as_bytes();
                if let Some(entry) = self.by_hash.get(&h) {
                    let req_prefix_len = i + 1;
                    if best.as_ref().map(|(b, _, _)| *b).unwrap_or(0) < req_prefix_len {
                        best = Some((req_prefix_len, h, entry.dir.clone()));
                    }
                }
            }
        }
        best
    }

    /// `O(1)` probe for a snapshot tied to a sessionId. Verifies the
    /// snapshot's tokens are a prefix of `tokens` before returning.
    pub fn lookup_session(
        &self,
        session_id: &str,
        tokens: &[i32],
    ) -> Option<(usize, [u8; 32], PathBuf)> {
        let h = self.session_to_hash.get(session_id)?;
        let entry = self.by_hash.get(h)?;
        let n = entry.token_count as usize;
        if n > tokens.len() {
            return None;
        }
        // We don't have the full token sequence here without re-reading
        // tokens.bin; the caller can compare after restore. For now we
        // trust the sessionId hint and verify in the caller (cheap:
        // restore + check before generating).
        Some((n, *h, entry.dir.clone()))
    }
}

fn token_ids_as_bytes(tokens: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(tokens.len() * 4);
    for &t in tokens {
        out.extend_from_slice(&t.to_le_bytes());
    }
    out
}

fn blake3_decoded(
    tokens: &[i32],
    image_spans: &[ImageSpan],
    vocab: &BpeVocab,
    byte_decoder: &std::collections::HashMap<char, u8>,
) -> [u8; 32] {
    let bytes = decode_tokens_to_bytes_vl(tokens, image_spans, vocab, byte_decoder);
    *blake3::hash(&bytes).as_bytes()
}

fn hex_to_blake3(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).ok()?;
    }
    Some(out)
}

/// Snapshot blob file that is created on first write, so a blob that ends
/// up empty leaves no file behind (the on-disk contract `restore` relies
/// on: `comp_kv.bin` etc. exist iff non-empty).
struct BlobWriter {
    path: PathBuf,
    w: Option<std::io::BufWriter<fs::File>>,
    bytes: u64,
}

impl BlobWriter {
    fn new(path: PathBuf) -> Self {
        Self { path, w: None, bytes: 0 }
    }
    fn write(&mut self, bytes: &[u8]) -> eyre::Result<()> {
        use std::io::Write;
        if bytes.is_empty() {
            return Ok(());
        }
        if self.w.is_none() {
            self.w = Some(std::io::BufWriter::with_capacity(1 << 20, fs::File::create(&self.path)?));
        }
        self.w.as_mut().unwrap().write_all(bytes)?;
        self.bytes += bytes.len() as u64;
        Ok(())
    }
    fn finish(mut self) -> eyre::Result<u64> {
        use std::io::Write;
        if let Some(mut w) = self.w.take() {
            w.flush()?;
        }
        Ok(self.bytes)
    }
}

/// Reusable host staging for the per-layer device reads. Peak host cost of
/// a save is now one layer's live prefix (a few MiB) plus the 1 MiB
/// BufWriters, instead of every blob in memory at once (~1.5 GB at 177K
/// tokens) plus a full-capacity copy of each device buffer — the pattern
/// that grew the server heap by GiBs per day (see v4flash_core::heap).
#[derive(Default)]
struct SaveScratch {
    u16s: Vec<u16>,
    f32s: Vec<f32>,
    bytes: Vec<u8>,
}

impl SaveScratch {
    /// Copy `n` leading elements of `buf` to the host and append them,
    /// little-endian, to `out`.
    fn stream_u16(
        &mut self,
        buf: &v4flash_hip::DeviceBuffer<u16>,
        n: usize,
        out: &mut BlobWriter,
    ) -> eyre::Result<()> {
        self.stream_u16_at(buf, 0, n, out)
    }
    /// Copy elements `[off, off + n)` of `buf` to the host and append them,
    /// little-endian, to `out`.
    fn stream_u16_at(
        &mut self,
        buf: &v4flash_hip::DeviceBuffer<u16>,
        off: usize,
        n: usize,
        out: &mut BlobWriter,
    ) -> eyre::Result<()> {
        if n == 0 {
            return Ok(());
        }
        if self.u16s.len() < n {
            self.u16s.resize(n, 0);
        }
        buf.slice_view(off, n).copy_to_host(&mut self.u16s[..n])?;
        self.bytes.clear();
        self.bytes.reserve(n * 2);
        for v in &self.u16s[..n] {
            self.bytes.extend_from_slice(&v.to_le_bytes());
        }
        out.write(&self.bytes)
    }
    /// Copy `n` leading bytes of `buf` to the host and append them to `out`.
    fn stream_u8(
        &mut self,
        buf: &v4flash_hip::DeviceBuffer<u8>,
        n: usize,
        out: &mut BlobWriter,
    ) -> eyre::Result<()> {
        if n == 0 {
            return Ok(());
        }
        self.bytes.resize(n, 0);
        buf.slice_view(0, n).copy_to_host(&mut self.bytes[..n])?;
        out.write(&self.bytes[..n])
    }
    fn stream_f32(
        &mut self,
        buf: &v4flash_hip::DeviceBuffer<f32>,
        n: usize,
        out: &mut BlobWriter,
    ) -> eyre::Result<()> {
        if n == 0 {
            return Ok(());
        }
        if self.f32s.len() < n {
            self.f32s.resize(n, 0.0);
        }
        buf.slice_view(0, n).copy_to_host(&mut self.f32s[..n])?;
        self.bytes.clear();
        self.bytes.reserve(n * 4);
        for v in &self.f32s[..n] {
            self.bytes.extend_from_slice(&v.to_le_bytes());
        }
        out.write(&self.bytes)
    }
}

/// Streaming reader for a snapshot blob: per-layer exact-size reads into
/// a reusable host buffer, then a prefix `copy_from_host` through a
/// `slice_view_mut`. A missing file reads as empty (the save-side contract:
/// a blob file exists iff it has bytes).
struct BlobReader {
    r: Option<std::io::BufReader<fs::File>>,
    remaining: u64,
    name: &'static str,
}

impl BlobReader {
    fn open(dir: &Path, name: &'static str, required: bool) -> eyre::Result<Self> {
        match fs::File::open(dir.join(name)) {
            Ok(f) => {
                let remaining = f.metadata()?.len();
                Ok(Self { r: Some(std::io::BufReader::with_capacity(1 << 20, f)), remaining, name })
            }
            Err(e) if !required && e.kind() == std::io::ErrorKind::NotFound => {
                Ok(Self { r: None, remaining: 0, name })
            }
            Err(e) => Err(eyre!("snapshot.restore: {name}: {e}")),
        }
    }
    fn has(&self, bytes: usize) -> bool {
        self.remaining >= bytes as u64
    }
    fn read_exact(&mut self, dst: &mut [u8]) -> eyre::Result<()> {
        use std::io::Read;
        if dst.is_empty() {
            return Ok(());
        }
        let Some(r) = self.r.as_mut() else {
            return Err(eyre!("snapshot.restore: {} missing", self.name));
        };
        r.read_exact(dst)
            .map_err(|e| eyre!("snapshot.restore: {}: {e}", self.name))?;
        self.remaining -= dst.len() as u64;
        Ok(())
    }
}

/// Reusable host staging for restore (mirror of `SaveScratch`).
#[derive(Default)]
struct RestoreScratch {
    bytes: Vec<u8>,
    u16s: Vec<u16>,
    f32s: Vec<f32>,
}

impl RestoreScratch {
    /// Read `n` little-endian u16 from `src` and write them to the first
    /// `n` elements of `dst` (the rest of `dst` is left as is: its valid
    /// extent is gated by the row counters, see HetModelState::reset_in_place).
    fn load_u16(
        &mut self,
        src: &mut BlobReader,
        n: usize,
        dst: &mut v4flash_hip::DeviceBuffer<u16>,
    ) -> eyre::Result<()> {
        if n == 0 {
            return Ok(());
        }
        self.bytes.resize(n * 2, 0);
        src.read_exact(&mut self.bytes[..n * 2])?;
        if self.u16s.len() < n {
            self.u16s.resize(n, 0);
        }
        for (i, c) in self.bytes[..n * 2].chunks_exact(2).enumerate() {
            self.u16s[i] = u16::from_le_bytes([c[0], c[1]]);
        }
        dst.slice_view_mut(0, n).copy_from_host(&self.u16s[..n])
    }
    /// Read `n` bytes from `src` straight into the first `n` bytes of `dst`.
    fn load_u8(
        &mut self,
        src: &mut BlobReader,
        n: usize,
        dst: &mut v4flash_hip::DeviceBuffer<u8>,
    ) -> eyre::Result<()> {
        if n == 0 {
            return Ok(());
        }
        self.bytes.resize(n, 0);
        src.read_exact(&mut self.bytes[..n])?;
        dst.slice_view_mut(0, n).copy_from_host(&self.bytes[..n])
    }
    fn load_f32(
        &mut self,
        src: &mut BlobReader,
        n: usize,
        dst: &mut v4flash_hip::DeviceBuffer<f32>,
    ) -> eyre::Result<()> {
        if n == 0 {
            return Ok(());
        }
        self.bytes.resize(n * 4, 0);
        src.read_exact(&mut self.bytes[..n * 4])?;
        if self.f32s.len() < n {
            self.f32s.resize(n, 0.0);
        }
        for (i, c) in self.bytes[..n * 4].chunks_exact(4).enumerate() {
            self.f32s[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }
        dst.slice_view_mut(0, n).copy_from_host(&self.f32s[..n])
    }
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Serialize and write a snapshot of `state` + `tokens` to a fresh
/// directory under `index.root()`. Returns the new IndexEntry on
/// success, but doesn't insert it — the caller does that after
/// confirming a successful write.
pub fn save(
    state: &HetModelState,
    tokens: &[i32],
    image_spans: &[ImageSpan],
    dgpu: Device,
    igpu: Device,
    fingerprint: &ModelFingerprint,
    root: &Path,
    vocab: &BpeVocab,
    byte_decoder: &std::collections::HashMap<char, u8>,
    session_id: Option<&str>,
) -> eyre::Result<IndexEntry> {
    if state.layers.len() != N_LAYER as usize {
        return Err(eyre!(
            "snapshot.save: layer count {} != N_LAYER {}",
            state.layers.len(),
            N_LAYER
        ));
    }
    // Key is hash of DECODED bytes, not token IDs — survives tokenizer
    // round-trips (sampled tokens vs. re-encoded history may split
    // differently but produce identical byte streams).
    let hash = blake3_decoded(tokens, image_spans, vocab, byte_decoder);
    let dir = root.join(hex::encode(hash));
    fs::create_dir_all(&dir)?;

    // tokens.bin is still i32-LE token IDs — we DO want to restore the
    // actual sampled tokens into the KV cache (the K/V vectors were
    // built from those exact IDs). The hash key just makes lookup
    // byte-stable across re-encodings.
    let tokens_bytes = token_ids_as_bytes(tokens);
    fs::write(dir.join("tokens.bin"), &tokens_bytes)?;

    // Pull device data per layer.
    let mut layers = Vec::with_capacity(N_LAYER as usize);
    let mut kv_blob = BlobWriter::new(dir.join("kv.bin"));
    let mut comp_kv_blob = BlobWriter::new(dir.join("comp_kv.bin"));
    let mut comp_state_blob = BlobWriter::new(dir.join("comp_state.bin"));
    let mut index_comp_kv_blob = BlobWriter::new(dir.join("index_comp_kv.bin"));
    let mut index_comp_state_blob = BlobWriter::new(dir.join("index_comp_state.bin"));
    let mut index_k_blob = BlobWriter::new(dir.join("index_k.bin"));
    let mut scratch = SaveScratch::default();
    for (li, layer) in state.layers.iter().enumerate() {
        let ratio = COMPRESS_RATIOS[li];
        let kv_rows = layer.n_raw.min(SWA_WINDOW);
        // Only the live window is read (slice_view) and it is streamed
        // straight to the file — no full-capacity host copy, no blob. It
        // starts at `raw_off`, not slot 0: the decode append is monotonic and
        // slides `raw_off` past SWA_WINDOW, so a state saved after a long
        // decode (the legacy `save_live_if_dirty`) used to write the rows that
        // had already been evicted. Restore puts them at slot 0, `raw_off = 0`.
        dgpu.set_current()?;
        let kv_used_n = (kv_rows as usize) * (N_HEAD_DIM as usize);
        let kv_first = (layer.raw_off + layer.n_raw - kv_rows) as usize * (N_HEAD_DIM as usize);
        scratch.stream_u16_at(&layer.kv_cache, kv_first, kv_used_n, &mut kv_blob)?;

        let mut comp_kv_format = CompKvFormat::F16;
        let mut comp_kv_row_bytes = 0u32;
        let mut has_index_k = false;
        let mut n_index_k = 0u32;
        let mut index_comp_kv_format = CompKvFormat::F16;
        let mut index_comp_kv_row_bytes = 0u32;
        let (has_compressor, n_comp, width, head_dim, state_rows, coff) = if let Some(comp) =
            &layer.compressor
        {
            let coff_local = if ratio == 4 { 2u32 } else { 1u32 };
            let state_rows = ratio * coff_local;
            // comp_kv on dGPU — live prefix only, streamed, in the
            // store's native encoding (no re-encoding on save).
            dgpu.set_current()?;
            match &comp.comp_kv {
                CompKvStore::F16(buf) => {
                    let ck_used_n = (comp.n_comp as usize) * (comp.head_dim as usize);
                    scratch.stream_u16(buf, ck_used_n, &mut comp_kv_blob)?;
                    comp_kv_format = CompKvFormat::F16;
                    comp_kv_row_bytes = comp.head_dim * 2;
                }
                CompKvStore::Fp8 { rows, .. } => {
                    let ck_used_b = (comp.n_comp as usize) * FP8_KV_ROW_BYTES;
                    scratch.stream_u8(rows, ck_used_b, &mut comp_kv_blob)?;
                    comp_kv_format = CompKvFormat::Fp8E4m3B64;
                    comp_kv_row_bytes = FP8_KV_ROW_BYTES as u32;
                }
                CompKvStore::E2m1(_) => {
                    return Err(eyre!("snapshot.save: layer {li} main compressor store cannot be E2M1"));
                }
            }
            // v6: V4.1's sparse-indexer KEYS (E2M1 rows) live here, not in
            // `indexer_compressor`. Stream the live prefix only, same shape
            // rule as comp_kv above.
            if let Some(ik) = comp.index_k.as_ref() {
                let used = (comp.n_index_comp as usize) * E2M1_KEY_ROW_BYTES;
                if used > 0 {
                    scratch.stream_u8(ik, used, &mut index_k_blob)?;
                }
                has_index_k = true;
                n_index_k = comp.n_index_comp;
            }
            // state_kv + state_score on iGPU — these ARE allocated at
            // exactly state_rows*width so no slicing needed.
            let n_state = comp.state_kv.len();
            igpu.set_current()?;
            scratch.stream_f32(&comp.state_kv, n_state, &mut comp_state_blob)?;
            scratch.stream_f32(&comp.state_score, n_state, &mut comp_state_blob)?;
            (
                true,
                comp.n_comp,
                comp.width,
                comp.head_dim,
                state_rows,
                coff_local,
            )
        } else {
            (false, 0, 0, 0, 0, 0)
        };

        // CSA indexer compressor (only on ratio==4 layers). State lives
        // on dGPU (per HetCompressorState::alloc(dgpu, dgpu, …)) so all
        // reads happen with dgpu current.
        let (
            has_indexer_compressor,
            n_index_comp,
            index_width,
            index_head_dim,
            index_state_rows,
            index_coff,
        ) = if let Some(icomp) = &layer.indexer_compressor {
            let coff_local = 2u32; // ratio==4 only
            let state_rows = ratio * coff_local;
            dgpu.set_current()?;
            index_comp_kv_format = CompKvFormat::of(&icomp.comp_kv);
            index_comp_kv_row_bytes = index_comp_kv_format.row_bytes(icomp.head_dim as usize) as u32;
            match &icomp.comp_kv {
                CompKvStore::F16(ibuf) => {
                    let ck_used_n = (icomp.n_comp as usize) * (icomp.head_dim as usize);
                    scratch.stream_u16(ibuf, ck_used_n, &mut index_comp_kv_blob)?;
                }
                CompKvStore::E2m1(rows) => {
                    let ck_used_b = (icomp.n_comp as usize) * E2M1_KEY_ROW_BYTES;
                    scratch.stream_u8(rows, ck_used_b, &mut index_comp_kv_blob)?;
                }
                CompKvStore::Fp8 { .. } => {
                    return Err(eyre!("snapshot.save: layer {li} indexer store cannot be FP8"));
                }
            }
            let n_state = icomp.state_kv.len();
            scratch.stream_f32(&icomp.state_kv, n_state, &mut index_comp_state_blob)?;
            scratch.stream_f32(&icomp.state_score, n_state, &mut index_comp_state_blob)?;
            (
                true,
                icomp.n_comp,
                icomp.width,
                icomp.head_dim,
                state_rows,
                coff_local,
            )
        } else {
            (false, 0, 0, 0, 0, 0)
        };

        layers.push(PerLayerMeta {
            n_raw: layer.n_raw,
            kv_rows,
            has_compressor,
            n_comp,
            ratio,
            coff,
            width,
            head_dim,
            state_rows,
            has_indexer_compressor,
            n_index_comp,
            index_coff,
            index_width,
            index_head_dim,
            index_state_rows,
            comp_kv_format,
            has_index_k,
            n_index_k,
            comp_kv_row_bytes,
            index_comp_kv_format,
            index_comp_kv_row_bytes,
        });
    }
    // Restore dgpu as current (callers expect that).
    dgpu.set_current()?;

    // Finish the streamed blobs (kv.bin always exists, even if empty —
    // restore reads it unconditionally).
    let kv_bytes = kv_blob.finish()?;
    if kv_bytes == 0 {
        fs::write(dir.join("kv.bin"), b"")?;
    }
    let comp_kv_bytes = comp_kv_blob.finish()?;
    let comp_state_bytes = comp_state_blob.finish()?;
    // `index_k.bin` was NEVER finish()ed and neither it nor `index_comp_kv.bin`
    // fed `total_bytes` (2026-09-22 audit A7). Consequences, measured on the live
    // cache (416 entries): the index believed it held 99.92 GiB against a 100 GiB
    // cap while `du` said 107.57 -- `index_k.bin` alone was 7.64 GiB -- so the LRU
    // under-evicted and the gap grew with every save. And an unfinished BlobWriter
    // flushes in Drop, which SWALLOWS io errors: an ENOSPC left a short
    // `index_k.bin`, which `load` silently degrades to `n_index_comp = 0`, i.e.
    // the dense-attention regression the v5->v6 note at the top of this file
    // measured. Bind both, and count both.
    let index_comp_kv_bytes = index_comp_kv_blob.finish()?;
    let index_k_bytes = index_k_blob.finish()?;
    let _ = index_comp_state_blob.finish()?;
    drop(scratch);

    // Total disk bytes (including meta.json's eventual size — we
    // approximate by writing meta first and summing).
    let now = unix_now();
    let mut meta = SnapshotMeta {
        format_version: FORMAT_VERSION,
        fingerprint: fingerprint.clone(),
        token_count: tokens.len() as u32,
        n_kv_max: state.n_kv_max,
        created_at_unix: now,
        last_used_unix: now,
        layers,
        disk_bytes: 0,
        session_id: session_id.map(|s| s.to_string()),
        image_spans: image_spans.to_vec(),
    };
    let mut total_bytes: u64 = tokens_bytes.len() as u64 + kv_bytes;
    total_bytes += comp_kv_bytes;
    total_bytes += comp_state_bytes;
    total_bytes += index_comp_kv_bytes;
    total_bytes += index_k_bytes;
    let meta_initial = serde_json::to_vec_pretty(&meta).map_err(|e| eyre!("meta encode: {e}"))?;
    total_bytes += meta_initial.len() as u64;
    meta.disk_bytes = total_bytes;
    let meta_final = serde_json::to_vec_pretty(&meta).map_err(|e| eyre!("meta encode: {e}"))?;
    fs::write(dir.join("meta.json"), &meta_final)?;

    Ok(IndexEntry {
        hash,
        token_count: tokens.len() as u32,
        last_used_unix: now,
        disk_bytes: total_bytes,
        dir,
        session_id: session_id.map(|s| s.to_string()),
    })
}

/// Restore a snapshot into `state`. The caller must have already
/// `reset_in_place`'d `state` (so it's at alloc-time defaults). Returns
/// the loaded token sequence.
pub fn restore(
    state: &mut HetModelState,
    src: &Path,
    dgpu: Device,
    igpu: Device,
    fingerprint: &ModelFingerprint,
    fp8: RestoreKernels<'_>,
) -> eyre::Result<Vec<i32>> {
    restore_vl(state, src, dgpu, igpu, fingerprint, fp8).map(|r| r.tokens)
}

/// Device-side helpers `restore` needs for the packed-FP8 compressed-KV
/// store: the expand kernel (head-shadow rebuild) and the dGPU stream to
/// run it on. Both live in the engine (`engine.dgpu.comp_kv_fp8`,
/// `engine.dgpu.compute`).
#[derive(Clone, Copy)]
pub struct RestoreKernels<'a> {
    pub fp8: &'a v4flash_kernels::CompKvFp8,
    pub stream: &'a v4flash_hip::Stream,
}

/// What [`restore_vl`] loaded.
#[derive(Debug, Clone)]
pub struct RestoredSnapshot {
    pub tokens: Vec<i32>,
    /// Image spans recorded at save time (empty for pre-vision snapshots).
    pub image_spans: Vec<ImageSpan>,
}

/// [`restore`] that also returns the snapshot's image spans.
pub fn restore_vl(
    state: &mut HetModelState,
    src: &Path,
    dgpu: Device,
    igpu: Device,
    fingerprint: &ModelFingerprint,
    kernels: RestoreKernels<'_>,
) -> eyre::Result<RestoredSnapshot> {
    let meta_bytes = fs::read(src.join("meta.json"))
        .map_err(|e| eyre!("snapshot.restore: read meta.json: {e}"))?;
    let meta: SnapshotMeta = serde_json::from_slice(&meta_bytes)
        .map_err(|e| eyre!("snapshot.restore: parse meta.json: {e}"))?;
    if meta.format_version < MIN_FORMAT_VERSION || meta.format_version > FORMAT_VERSION {
        return Err(eyre!(
            "snapshot.restore: format_version mismatch (saw {}, want {}..={})",
            meta.format_version,
            MIN_FORMAT_VERSION,
            FORMAT_VERSION
        ));
    }
    if meta.fingerprint != *fingerprint {
        return Err(eyre!(
            "snapshot.restore: model fingerprint mismatch (snapshot is from a different model)"
        ));
    }
    if meta.layers.len() != N_LAYER as usize {
        return Err(eyre!(
            "snapshot.restore: layer count mismatch (saw {}, want {})",
            meta.layers.len(),
            N_LAYER
        ));
    }
    if state.layers.len() != N_LAYER as usize {
        return Err(eyre!("snapshot.restore: state has wrong layer count"));
    }
    // Snapshots from a smaller n_kv_max fit inside a larger state's
    // buffers — restore copies row data and zero-pads the rest. Only
    // reject when the snapshot is LARGER than the live state (would
    // overflow). This lets `--ctx` be bumped without invalidating the
    // disk cache.
    if meta.n_kv_max > state.n_kv_max {
        return Err(eyre!(
            "snapshot.restore: snapshot n_kv_max {} exceeds state n_kv_max {}",
            meta.n_kv_max,
            state.n_kv_max
        ));
    }

    let tokens_bytes = fs::read(src.join("tokens.bin"))
        .map_err(|e| eyre!("snapshot.restore: tokens.bin: {e}"))?;
    if tokens_bytes.len() % 4 != 0 {
        return Err(eyre!("snapshot.restore: tokens.bin not i32-aligned"));
    }
    let token_count = tokens_bytes.len() / 4;
    let mut tokens = Vec::with_capacity(token_count);
    for chunk in tokens_bytes.chunks_exact(4) {
        tokens.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }

    // Blobs are streamed per layer (exact-size reads into reusable host
    // staging, prefix copies onto the device). Peak host cost is one
    // layer's slice instead of every blob in memory plus a full-capacity
    // zero-padded copy of each device buffer.
    let mut kv_rd = BlobReader::open(src, "kv.bin", true)?;
    let mut comp_kv_rd = BlobReader::open(src, "comp_kv.bin", false)?;
    let mut comp_state_rd = BlobReader::open(src, "comp_state.bin", false)?;
    let mut index_comp_kv_rd = BlobReader::open(src, "index_comp_kv.bin", false)?;
    let mut index_comp_state_rd = BlobReader::open(src, "index_comp_state.bin", false)?;
    let mut index_k_rd = BlobReader::open(src, "index_k.bin", false)?;
    let mut scratch = RestoreScratch::default();

    for (li, layer) in state.layers.iter_mut().enumerate() {
        let m = &meta.layers[li];

        // raw KV — live prefix only; the rest of the buffer is gated by
        // n_raw (see HetModelState::reset_in_place).
        let kv_count = (m.kv_rows as usize) * (N_HEAD_DIM as usize);
        let kv_bytes_len = kv_count * 2;
        if !kv_rd.has(kv_bytes_len) {
            return Err(eyre!(
                "snapshot.restore: kv.bin truncated at layer {li}"
            ));
        }
        let kv_full_n = layer.kv_cache.len();
        if kv_count > kv_full_n {
            return Err(eyre!(
                "snapshot.restore: kv_count {kv_count} > layer buffer {kv_full_n}"
            ));
        }
        dgpu.set_current()?;
        scratch.load_u16(&mut kv_rd, kv_count, &mut layer.kv_cache)?;
        layer.n_raw = m.n_raw;
        layer.raw_off = 0; // saved rows land at slot 0 (see the save side)

        if m.has_compressor {
            let Some(comp) = &mut layer.compressor else {
                return Err(eyre!(
                    "snapshot.restore: layer {li} has compressor in snapshot but not in state"
                ));
            };
            comp.n_comp = m.n_comp;

            // comp_kv — live prefix only (gated by n_comp). The file's
            // encoding must equal the live store's; anything else is a
            // cache miss for the caller to evict (no conversion, see the
            // version note at the top).
            let n_comp = m.n_comp as usize;
            let head_dim = m.head_dim as usize;
            let live_fmt = CompKvFormat::of(&comp.comp_kv);
            if m.comp_kv_format != live_fmt {
                return Err(eyre!(
                    "snapshot.restore: layer {li} comp_kv encoding {:?} does not match the live store {:?}",
                    m.comp_kv_format, live_fmt
                ));
            }
            let file_row_bytes = m.comp_kv_format.row_bytes(head_dim);
            if m.comp_kv_format != CompKvFormat::F16 && m.comp_kv_row_bytes as usize != file_row_bytes {
                return Err(eyre!(
                    "snapshot.restore: layer {li} comp_kv rows of {} B (expected {file_row_bytes})",
                    m.comp_kv_row_bytes
                ));
            }
            if !comp_kv_rd.has(n_comp * file_row_bytes) {
                return Err(eyre!(
                    "snapshot.restore: comp_kv.bin truncated at layer {li}"
                ));
            }
            if n_comp > comp.comp_kv.capacity_rows(m.head_dim) {
                return Err(eyre!(
                    "snapshot.restore: n_comp {n_comp} > buffer capacity {} rows",
                    comp.comp_kv.capacity_rows(m.head_dim)
                ));
            }
            dgpu.set_current()?;
            match &mut comp.comp_kv {
                CompKvStore::F16(buf) => {
                    scratch.load_u16(&mut comp_kv_rd, n_comp * head_dim, buf)?;
                }
                CompKvStore::Fp8 { rows, head } => {
                    scratch.load_u8(&mut comp_kv_rd, n_comp * FP8_KV_ROW_BYTES, rows)?;
                    // Rebuild the dense-path f16 head shadow through the
                    // production expand kernel.
                    let head_n = n_comp.min(FP8_KV_HEAD_ROWS) as u32;
                    kernels.fp8.launch_expand(kernels.stream, head, rows, head_n)?;
                    kernels.stream.synchronize()?;
                }
                CompKvStore::E2m1(_) => {
                    return Err(eyre!("snapshot.restore: layer {li} main compressor store cannot be E2M1"));
                }
            }

            // state_kv + state_score (each n_state floats, packed back-to-back).
            let n_state = (m.state_rows as usize) * (m.width as usize);
            let block_bytes = n_state * 4;
            let block_total = 2 * block_bytes;
            igpu.set_current()?;
            if !comp_state_rd.has(block_total) {
                // Fall back to alloc-time defaults if missing.
                comp.state_kv.copy_from_host(&vec![0f32; n_state])?;
                comp.state_score.copy_from_host(&vec![NEG_INF; n_state])?;
            } else {
                scratch.load_f32(&mut comp_state_rd, n_state, &mut comp.state_kv)?;
                scratch.load_f32(&mut comp_state_rd, n_state, &mut comp.state_score)?;
            }

            // v6: V4.1 sparse-indexer keys. Without these `n_index_comp` comes
            // back 0 and the decode gate (`n_index_comp > INDEXER_TOP_K`) can
            // never fire on a restored session, so every layer scores densely.
            //
            // A key store that does not cover every compressed row is REFUSED:
            // the snapshot is a cache miss (callers evict it and prefill fully).
            // It used to "degrade" to `n_index_comp = 0` with `n_comp` kept, on
            // the theory that the session would run dense -- but the next
            // prefill appends keys from `n_comp` on and then declares the whole
            // store valid (`n_index_comp = n_comp_start + fired`), while the
            // indexer scores `n_comp` rows: rows [0, n_comp) were whatever the
            // reused scratch state last held, i.e. ANOTHER request's keys, and
            // the next snapshot saved them as valid for the rest of the chain.
            match (comp.index_k.as_mut(), m.has_index_k) {
                (Some(ik), true) => {
                    let want = (m.n_index_k as usize) * E2M1_KEY_ROW_BYTES;
                    if m.n_index_k != m.n_comp {
                        return Err(eyre!(
                            "snapshot.restore: layer {li} holds {} index keys for {} compressed rows",
                            m.n_index_k, m.n_comp
                        ));
                    }
                    if want > 0 {
                        if !index_k_rd.has(want) {
                            return Err(eyre!(
                                "snapshot.restore: layer {li} index_k.bin is short ({} key rows expected)",
                                m.n_index_k
                            ));
                        }
                        dgpu.set_current()?;
                        scratch.load_u8(&mut index_k_rd, want, ik)?;
                    }
                    comp.n_index_comp = m.n_index_k;
                }
                (Some(_), false) if m.n_comp > 0 => {
                    return Err(eyre!(
                        "snapshot.restore: layer {li} has {} compressed rows but no index keys",
                        m.n_comp
                    ));
                }
                (Some(_), false) => comp.n_index_comp = 0,
                (None, _) => {}
            }
        } else if let Some(comp) = layer.compressor.as_mut() {
            // State expects a compressor but snapshot doesn't have one;
            // re-init defaults.
            comp.n_comp = 0;
            comp.n_index_comp = 0;
            let n_state = comp.state_kv.len();
            igpu.set_current()?;
            comp.state_kv.copy_from_host(&vec![0f32; n_state])?;
            comp.state_score.copy_from_host(&vec![NEG_INF; n_state])?;
        }

        // CSA indexer compressor restore. Mirrors the main-compressor
        // block above with the indexer's smaller dims. State lives on
        // dGPU (per HetCompressorState::alloc(dgpu, dgpu, …)).
        if m.has_indexer_compressor {
            let Some(icomp) = &mut layer.indexer_compressor else {
                return Err(eyre!(
                    "snapshot.restore: layer {li} has indexer_compressor in snapshot but not in state"
                ));
            };
            icomp.n_comp = m.n_index_comp;

            let n_icomp = m.n_index_comp as usize;
            let ihd = m.index_head_dim as usize;
            let live_fmt = CompKvFormat::of(&icomp.comp_kv);
            if m.index_comp_kv_format != live_fmt {
                return Err(eyre!(
                    "snapshot.restore: layer {li} index_comp_kv encoding {:?} does not match the live store {:?}",
                    m.index_comp_kv_format, live_fmt
                ));
            }
            let irow_bytes = m.index_comp_kv_format.row_bytes(ihd);
            if !index_comp_kv_rd.has(n_icomp * irow_bytes) {
                return Err(eyre!(
                    "snapshot.restore: index_comp_kv.bin truncated at layer {li}"
                ));
            }
            if n_icomp > icomp.comp_kv.capacity_rows(m.index_head_dim) {
                return Err(eyre!(
                    "snapshot.restore: n_index_comp {n_icomp} > buffer capacity {} rows",
                    icomp.comp_kv.capacity_rows(m.index_head_dim)
                ));
            }
            dgpu.set_current()?;
            match &mut icomp.comp_kv {
                CompKvStore::F16(ibuf) => scratch.load_u16(&mut index_comp_kv_rd, n_icomp * ihd, ibuf)?,
                CompKvStore::E2m1(rows) => scratch.load_u8(&mut index_comp_kv_rd, n_icomp * E2M1_KEY_ROW_BYTES, rows)?,
                CompKvStore::Fp8 { .. } => {
                    return Err(eyre!("snapshot.restore: layer {li} indexer store cannot be FP8"));
                }
            }

            let in_state = (m.index_state_rows as usize) * (m.index_width as usize);
            let in_block_bytes = in_state * 4;
            let in_block_total = 2 * in_block_bytes;
            if !index_comp_state_rd.has(in_block_total) {
                icomp.state_kv.copy_from_host(&vec![0f32; in_state])?;
                icomp.state_score.copy_from_host(&vec![NEG_INF; in_state])?;
            } else {
                scratch.load_f32(&mut index_comp_state_rd, in_state, &mut icomp.state_kv)?;
                scratch.load_f32(&mut index_comp_state_rd, in_state, &mut icomp.state_score)?;
            }
        } else if let Some(icomp) = layer.indexer_compressor.as_mut() {
            // State expects an indexer_compressor but snapshot doesn't —
            // re-init defaults.
            icomp.n_comp = 0;
            let n_state = icomp.state_kv.len();
            dgpu.set_current()?;
            icomp.state_kv.copy_from_host(&vec![0f32; n_state])?;
            icomp.state_score.copy_from_host(&vec![NEG_INF; n_state])?;
        }
    }

    // Leave dgpu current for the caller's subsequent prefill.
    dgpu.set_current()?;
    Ok(RestoredSnapshot { tokens, image_spans: meta.image_spans })
}

/// hex encode/decode for the snapshot directory names. We avoid pulling
/// in the `hex` crate by inlining the trivial implementation here.
mod hex {
    pub fn encode(bytes: [u8; 32]) -> String {
        let mut s = String::with_capacity(64);
        for b in &bytes {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }
}

#[cfg(test)]
mod retention_tests {
    //! Per-lineage retention (R1) + global-cap backstop (R4). Pure index
    //! logic — no GPU. Uses real temp dirs so `remove_dir_all` exercises.
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn unique_root() -> PathBuf {
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("deepstrix-rt-{}-{n}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }
    fn fp() -> ModelFingerprint {
        ModelFingerprint {
            n_layer: 1,
            n_head_dim: 1,
            vocab_size: 1,
            token_embd_prefix_blake3: "x".into(),
            tensor_directory_blake3: "y".into(),
        }
    }
    fn h(tag: u8) -> [u8; 32] {
        let mut x = [0u8; 32];
        x[0] = tag;
        x
    }
    fn mk(root: &Path, tag: u8, last_used: u64, bytes: u64, session: Option<&str>) -> IndexEntry {
        let hash = h(tag);
        let dir = root.join(hex::encode(hash));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("tokens.bin"), b"x").unwrap();
        IndexEntry {
            hash,
            token_count: bytes as u32,
            last_used_unix: last_used,
            disk_bytes: bytes,
            dir,
            session_id: session.map(|s| s.to_string()),
        }
    }

    #[test]
    fn r1_keeps_last_n_per_lineage_and_protects_tip() {
        let root = unique_root();
        let mut idx = SnapshotIndex::new(root.clone(), fp(), 1 << 40); // huge cap → isolate R1
        idx.max_per_lineage = 3;
        for i in 0..5u8 {
            idx.insert(mk(&root, i, i as u64 + 1, 100, Some("A")));
        }
        assert_eq!(idx.len(), 3, "lineage capped at 3");
        assert_eq!(idx.by_session.get("A").unwrap().len(), 3);
        // Oldest two (tags 0,1) evicted incl. their dirs; newest tip (4) kept.
        assert!(idx.by_hash.get(&h(0)).is_none());
        assert!(idx.by_hash.get(&h(1)).is_none());
        assert!(!root.join(hex::encode(h(0))).exists());
        assert!(idx.by_hash.contains_key(&h(4)), "tip retained");
    }

    #[test]
    fn lineages_are_independent() {
        let root = unique_root();
        let mut idx = SnapshotIndex::new(root.clone(), fp(), 1 << 40);
        idx.max_per_lineage = 3;
        for i in 0..5u8 {
            idx.insert(mk(&root, i, i as u64 + 1, 100, Some("A")));
        }
        for i in 5..10u8 {
            idx.insert(mk(&root, i, i as u64 + 1, 100, Some("B")));
        }
        assert_eq!(idx.len(), 6, "3 per lineage × 2 lineages");
        assert_eq!(idx.by_session.get("A").unwrap().len(), 3);
        assert_eq!(idx.by_session.get("B").unwrap().len(), 3);
    }

    #[test]
    fn r4_global_cap_evicts_lru_across_lineages() {
        let root = unique_root();
        let mut idx = SnapshotIndex::new(root.clone(), fp(), 250); // 250-byte cap
        idx.max_per_lineage = 10; // disable R1 so R4 is the actor
        idx.insert(mk(&root, 0, 1, 100, Some("A")));
        idx.insert(mk(&root, 1, 2, 100, Some("B")));
        idx.insert(mk(&root, 2, 3, 100, Some("A"))); // 300>250 → drop globally-oldest (tag0)
        assert!(idx.total_bytes() <= 250);
        assert!(idx.by_hash.get(&h(0)).is_none(), "oldest across lineages evicted");
        assert!(idx.by_hash.contains_key(&h(2)), "newest kept");
    }

    #[test]
    fn resave_same_hash_does_not_double_count() {
        let root = unique_root();
        let mut idx = SnapshotIndex::new(root.clone(), fp(), 1 << 40);
        idx.max_per_lineage = 3;
        idx.insert(mk(&root, 0, 1, 100, Some("A")));
        idx.insert(mk(&root, 0, 2, 100, Some("A"))); // same hash, re-saved
        assert_eq!(idx.len(), 1);
        assert_eq!(idx.total_bytes(), 100, "bytes not double-counted");
        assert_eq!(idx.by_session.get("A").unwrap().len(), 1, "membership not duplicated");
    }

    // ---- image-aware byte stream / snapshot key ---------------------

    #[test]
    fn snapshot_key_differs_for_different_image_bytes() {
        use crate::vision_prompt::{expand_prepared, ImageSpan};
        use v4flash_vision::PreprocessedImage;
        let fake = |seed: u8| PreprocessedImage {
            patches: vec![seed as f32; 4],
            n_vit_h: 37,
            n_vit_w: 37,
            content_hash: *blake3::hash(&[seed]).as_bytes(),
        };
        const PH: i32 = 129264;
        // Same prompt text, same image DIMENSIONS (identical token ids),
        // different pixels.
        let base = vec![0, 128803, 11, PH, 12, 128804];
        let a = expand_prepared(base.clone(), PH, vec![fake(1)]).unwrap();
        let b = expand_prepared(base.clone(), PH, vec![fake(2)]).unwrap();
        assert_eq!(a.tokens, b.tokens, "identical layouts → identical ids");
        let dec = |id: i32| -> Vec<u8> { format!("[{id}]").into_bytes() };
        let ba = decode_tokens_to_bytes_with(&a.tokens, &a.spans, dec);
        let bb = decode_tokens_to_bytes_with(&b.tokens, &b.spans, dec);
        assert_ne!(ba, bb);
        assert_ne!(blake3::hash(&ba), blake3::hash(&bb));
        // Same pixels → same key (the property snapshot reuse relies on).
        let a2 = expand_prepared(base.clone(), PH, vec![fake(1)]).unwrap();
        let ba2 = decode_tokens_to_bytes_with(&a2.tokens, &a2.spans, dec);
        assert_eq!(ba, ba2);
        // The image bytes sit at the START position: the stream before the
        // block is shared, and the pre-image text-only prefix hashes the
        // same as a text-only prompt would (role-boundary probing works).
        let prefix_a = decode_tokens_to_bytes_with(&a.tokens[..3], &a.spans, dec);
        let prefix_text = decode_tokens_to_bytes_with(&base[..3], &[], dec);
        assert_eq!(prefix_a, prefix_text);
        // Without spans (e.g. diag paths) the stream still reflects the
        // layout, just not the content.
        let no_spans: &[ImageSpan] = &[];
        let ba_nospan = decode_tokens_to_bytes_with(&a.tokens, no_spans, dec);
        let bb_nospan = decode_tokens_to_bytes_with(&b.tokens, no_spans, dec);
        assert_eq!(ba_nospan, bb_nospan);
        assert_ne!(ba_nospan, ba);
    }

    #[test]
    fn snapshot_meta_image_spans_roundtrip_and_default() {
        let spans = vec![crate::vision_prompt::ImageSpan { start: 7, len: 198, hash: [9u8; 32] }];
        let meta = SnapshotMeta {
            format_version: FORMAT_VERSION,
            fingerprint: fp(),
            token_count: 1,
            n_kv_max: 1,
            created_at_unix: 0,
            last_used_unix: 0,
            layers: Vec::new(),
            disk_bytes: 0,
            session_id: None,
            image_spans: spans.clone(),
        };
        let j = serde_json::to_string(&meta).unwrap();
        let back: SnapshotMeta = serde_json::from_str(&j).unwrap();
        assert_eq!(back.image_spans, spans);
        // Pre-vision meta.json (no field) → empty.
        let v: serde_json::Value = serde_json::from_str(&j).unwrap();
        let mut o = v.as_object().unwrap().clone();
        o.remove("image_spans");
        let back2: SnapshotMeta = serde_json::from_value(serde_json::Value::Object(o)).unwrap();
        assert!(back2.image_spans.is_empty());
    }

    /// `find_longest_prefix` keys every on-disk snapshot by the blake3 of the
    /// byte prefix at a boundary token. It used to re-hash the WHOLE prefix at
    /// each boundary (O(prefix x boundaries), ~500 MB of hashing on a long
    /// conversation); it now keeps one running `Hasher` and clones it.
    ///
    /// These two must agree byte for byte forever, or every snapshot already on
    /// disk becomes unreachable and every request silently falls back to a full
    /// prefill. Covers the cases the real loop produces: EMPTY chunks (an
    /// unknown token decodes to `unwrap_or_default()`), single bytes, and a run
    /// long enough to cross blake3's 1024-byte chunk boundary.
    #[test]
    fn incremental_hash_matches_whole_prefix_hash() {
        let chunks: Vec<Vec<u8>> = vec![
            b"hello".to_vec(),
            Vec::new(),
            b" world".to_vec(),
            vec![0xEF],
            vec![7u8; 1500],
            Vec::new(),
            vec![0xFFu8; 3000],
            b"tail".to_vec(),
        ];
        let mut whole: Vec<u8> = Vec::new();
        let mut running = blake3::Hasher::new();
        for c in &chunks {
            whole.extend_from_slice(c);
            running.update(c);
            assert_eq!(
                *running.clone().finalize().as_bytes(),
                *blake3::hash(&whole).as_bytes(),
                "incremental hash diverged after {} bytes -- this would orphan \
                 every snapshot on disk",
                whole.len()
            );
        }
        // Cloning must not disturb the running state: the next update has to
        // continue the same stream, not a finalized one.
        running.update(b"more");
        whole.extend_from_slice(b"more");
        assert_eq!(*running.finalize().as_bytes(), *blake3::hash(&whole).as_bytes());
    }
}
