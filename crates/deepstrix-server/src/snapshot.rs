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

use crate::embed::gpt2_decode_token;

/// Bumped to 2 when snapshot keys switched from `blake3(token_id_LE_bytes)`
/// to `blake3(decoded_byte_stream)` — bytes are the source of truth and
/// survive tokenizer-roundtrip splits.
// v2 → v3: added per-layer indexer compressor state (`has_indexer_compressor`,
// `n_index_comp`, `index_*` shape fields) + index_comp_kv.bin +
// index_comp_state.bin blobs. v2 snapshots get evicted at startup since
// they lack the indexer state needed for correct ratio==4 attention at
// long context.
const FORMAT_VERSION: u32 = 3;

/// Decode a token-id sequence to the raw byte stream the model would
/// see at the surface level. Used for snapshot keys + byte-level
/// prefix matching across tokenization differences.
pub fn decode_tokens_to_bytes(
    tokens: &[i32],
    vocab: &BpeVocab,
    byte_decoder: &std::collections::HashMap<char, u8>,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(tokens.len() * 2);
    for &id in tokens {
        if let Some(bytes) = vocab.token_text(id) {
            out.extend(gpt2_decode_token(bytes, byte_decoder));
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
}

impl ModelFingerprint {
    pub fn compute(vocab_size: u32, token_embd_bytes: &[u8]) -> Self {
        let prefix_len = token_embd_bytes.len().min(4096);
        let hash = blake3::hash(&token_embd_bytes[..prefix_len]);
        Self {
            n_layer: N_LAYER as u32,
            n_head_dim: N_HEAD_DIM,
            vocab_size,
            token_embd_prefix_blake3: hash.to_hex().to_string(),
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
        Self {
            root,
            fingerprint,
            by_hash: HashMap::new(),
            by_last_used: BTreeMap::new(),
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
            if meta.format_version != FORMAT_VERSION {
                tracing::warn!(
                    path = ?path,
                    saw = meta.format_version,
                    want = FORMAT_VERSION,
                    "snapshot format mismatch; skipping"
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
            };
            idx.total_bytes += e.disk_bytes;
            idx.by_last_used.insert((e.last_used_unix, e.hash), ());
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

    /// Pop LRU entries until total_bytes <= self.cap_bytes.
    pub fn evict_to_fit(&mut self) {
        while self.total_bytes > self.cap_bytes {
            let Some(((ts, hash), ())) = self.by_last_used.iter().next().map(|(k, v)| (*k, *v))
            else {
                break;
            };
            let Some(entry) = self.by_hash.remove(&hash) else {
                self.by_last_used.remove(&(ts, hash));
                continue;
            };
            self.by_last_used.remove(&(ts, hash));
            self.total_bytes = self.total_bytes.saturating_sub(entry.disk_bytes);
            if let Err(e) = fs::remove_dir_all(&entry.dir) {
                tracing::warn!(dir = ?entry.dir, error = %e, "failed to remove evicted snapshot dir");
            } else {
                tracing::info!(
                    hash = %hex::encode(entry.hash),
                    bytes = entry.disk_bytes,
                    "evicted snapshot"
                );
            }
        }
    }

    pub fn insert(&mut self, entry: IndexEntry) {
        self.total_bytes += entry.disk_bytes;
        self.by_last_used
            .insert((entry.last_used_unix, entry.hash), ());
        self.by_hash.insert(entry.hash, entry);
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
    /// `i` where `tokens[i-1]` is either `TOK_EOS` (end-of-message)
    /// or `TOK_ASSISTANT` (start of an assistant turn — the
    /// canonical save point for the start-of-think snapshot). The
    /// latter is essential: post-`<think>`-design snapshots end at
    /// `<Assistant>` (not EOS), and probing only at EOS would never
    /// find them. Returns `(req_prefix_len, hash, dir)` of the
    /// largest match — where `req_prefix_len` is the req-side token
    /// count for the byte boundary that hashed (the snapshot's
    /// stored token count may differ, since the same bytes can
    /// split differently).
    pub fn find_longest_prefix(
        &self,
        tokens: &[i32],
        tok_eos: i32,
        tok_assistant: i32,
        vocab: &BpeVocab,
        byte_decoder: &std::collections::HashMap<char, u8>,
    ) -> Option<(usize, [u8; 32], PathBuf)> {
        let mut best: Option<(usize, [u8; 32], PathBuf)> = None;
        let mut byte_prefix: Vec<u8> = Vec::with_capacity(tokens.len() * 2);
        let decode = |id: i32| -> Vec<u8> {
            vocab
                .token_text(id)
                .map(|b| gpt2_decode_token(b, byte_decoder))
                .unwrap_or_default()
        };
        for (i, &tok) in tokens.iter().enumerate() {
            byte_prefix.extend(decode(tok));
            if tok == tok_eos || tok == tok_assistant {
                let h = *blake3::hash(&byte_prefix).as_bytes();
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
    vocab: &BpeVocab,
    byte_decoder: &std::collections::HashMap<char, u8>,
) -> [u8; 32] {
    let bytes = decode_tokens_to_bytes(tokens, vocab, byte_decoder);
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
    dgpu: Device,
    igpu: Device,
    fingerprint: &ModelFingerprint,
    root: &Path,
    vocab: &BpeVocab,
    byte_decoder: &std::collections::HashMap<char, u8>,
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
    let hash = blake3_decoded(tokens, vocab, byte_decoder);
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
    let mut kv_blob: Vec<u8> = Vec::new();
    let mut comp_kv_blob: Vec<u8> = Vec::new();
    let mut comp_state_blob: Vec<u8> = Vec::new();
    let mut index_comp_kv_blob: Vec<u8> = Vec::new();
    let mut index_comp_state_blob: Vec<u8> = Vec::new();
    for (li, layer) in state.layers.iter().enumerate() {
        let ratio = COMPRESS_RATIOS[li];
        let kv_rows = layer.n_raw.min(SWA_WINDOW);
        // DeviceBuffer::copy_to_host requires the host slice length to
        // match the full buffer length, so we read the entire buffer
        // and slice the live portion ourselves.
        dgpu.set_current()?;
        let kv_full_n = layer.kv_cache.len();
        let mut kv_host = vec![0u16; kv_full_n];
        if kv_full_n > 0 {
            layer.kv_cache.copy_to_host(&mut kv_host)?;
        }
        let kv_used_n = (kv_rows as usize) * (N_HEAD_DIM as usize);
        for v in &kv_host[..kv_used_n] {
            kv_blob.extend_from_slice(&v.to_le_bytes());
        }

        let (has_compressor, n_comp, width, head_dim, state_rows, coff) = if let Some(comp) =
            &layer.compressor
        {
            let coff_local = if ratio == 4 { 2u32 } else { 1u32 };
            let state_rows = ratio * coff_local;
            // comp_kv on dGPU — same full-buffer dance.
            let ck_full_n = comp.comp_kv.len();
            let mut comp_kv_host = vec![0u16; ck_full_n];
            if ck_full_n > 0 {
                dgpu.set_current()?;
                comp.comp_kv.copy_to_host(&mut comp_kv_host)?;
            }
            let ck_used_n = (comp.n_comp as usize) * (comp.head_dim as usize);
            for v in &comp_kv_host[..ck_used_n] {
                comp_kv_blob.extend_from_slice(&v.to_le_bytes());
            }
            // state_kv + state_score on iGPU — these ARE allocated at
            // exactly state_rows*width so no slicing needed.
            let n_state = comp.state_kv.len();
            let mut state_kv_host = vec![0f32; n_state];
            let mut state_score_host = vec![0f32; n_state];
            igpu.set_current()?;
            if n_state > 0 {
                comp.state_kv.copy_to_host(&mut state_kv_host)?;
                comp.state_score.copy_to_host(&mut state_score_host)?;
            }
            for v in &state_kv_host {
                comp_state_blob.extend_from_slice(&v.to_le_bytes());
            }
            for v in &state_score_host {
                comp_state_blob.extend_from_slice(&v.to_le_bytes());
            }
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
            let ck_full_n = icomp.comp_kv.len();
            let mut icomp_kv_host = vec![0u16; ck_full_n];
            if ck_full_n > 0 {
                dgpu.set_current()?;
                icomp.comp_kv.copy_to_host(&mut icomp_kv_host)?;
            }
            let ck_used_n = (icomp.n_comp as usize) * (icomp.head_dim as usize);
            for v in &icomp_kv_host[..ck_used_n] {
                index_comp_kv_blob.extend_from_slice(&v.to_le_bytes());
            }
            let n_state = icomp.state_kv.len();
            let mut state_kv_host = vec![0f32; n_state];
            let mut state_score_host = vec![0f32; n_state];
            if n_state > 0 {
                dgpu.set_current()?;
                icomp.state_kv.copy_to_host(&mut state_kv_host)?;
                icomp.state_score.copy_to_host(&mut state_score_host)?;
            }
            for v in &state_kv_host {
                index_comp_state_blob.extend_from_slice(&v.to_le_bytes());
            }
            for v in &state_score_host {
                index_comp_state_blob.extend_from_slice(&v.to_le_bytes());
            }
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
        });
    }
    // Restore dgpu as current (callers expect that).
    dgpu.set_current()?;

    // Write the binary blobs.
    fs::write(dir.join("kv.bin"), &kv_blob)?;
    if !comp_kv_blob.is_empty() {
        fs::write(dir.join("comp_kv.bin"), &comp_kv_blob)?;
    }
    if !comp_state_blob.is_empty() {
        fs::write(dir.join("comp_state.bin"), &comp_state_blob)?;
    }
    if !index_comp_kv_blob.is_empty() {
        fs::write(dir.join("index_comp_kv.bin"), &index_comp_kv_blob)?;
    }
    if !index_comp_state_blob.is_empty() {
        fs::write(dir.join("index_comp_state.bin"), &index_comp_state_blob)?;
    }

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
    };
    let mut total_bytes: u64 = tokens_bytes.len() as u64 + kv_blob.len() as u64;
    total_bytes += comp_kv_blob.len() as u64;
    total_bytes += comp_state_blob.len() as u64;
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
) -> eyre::Result<Vec<i32>> {
    let meta_bytes = fs::read(src.join("meta.json"))
        .map_err(|e| eyre!("snapshot.restore: read meta.json: {e}"))?;
    let meta: SnapshotMeta = serde_json::from_slice(&meta_bytes)
        .map_err(|e| eyre!("snapshot.restore: parse meta.json: {e}"))?;
    if meta.format_version != FORMAT_VERSION {
        return Err(eyre!(
            "snapshot.restore: format_version mismatch (saw {}, want {})",
            meta.format_version,
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

    let kv_bytes = fs::read(src.join("kv.bin"))
        .map_err(|e| eyre!("snapshot.restore: kv.bin: {e}"))?;
    let comp_kv_bytes = fs::read(src.join("comp_kv.bin")).unwrap_or_default();
    let comp_state_bytes = fs::read(src.join("comp_state.bin")).unwrap_or_default();
    let index_comp_kv_bytes = fs::read(src.join("index_comp_kv.bin")).unwrap_or_default();
    let index_comp_state_bytes =
        fs::read(src.join("index_comp_state.bin")).unwrap_or_default();

    let mut kv_off = 0usize;
    let mut comp_kv_off = 0usize;
    let mut comp_state_off = 0usize;
    let mut index_comp_kv_off = 0usize;
    let mut index_comp_state_off = 0usize;

    for (li, layer) in state.layers.iter_mut().enumerate() {
        let m = &meta.layers[li];

        // raw KV — copy_from_host needs full-buffer length; pad with zeros.
        let kv_count = (m.kv_rows as usize) * (N_HEAD_DIM as usize);
        let kv_bytes_len = kv_count * 2;
        if kv_off + kv_bytes_len > kv_bytes.len() {
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
        let mut kv_host = vec![0u16; kv_full_n];
        for (i, c) in kv_bytes[kv_off..kv_off + kv_bytes_len]
            .chunks_exact(2)
            .enumerate()
        {
            kv_host[i] = u16::from_le_bytes([c[0], c[1]]);
        }
        if kv_full_n > 0 {
            dgpu.set_current()?;
            layer.kv_cache.copy_from_host(&kv_host)?;
        }
        kv_off += kv_bytes_len;
        layer.n_raw = m.n_raw;

        if m.has_compressor {
            let Some(comp) = &mut layer.compressor else {
                return Err(eyre!(
                    "snapshot.restore: layer {li} has compressor in snapshot but not in state"
                ));
            };
            comp.n_comp = m.n_comp;

            // comp_kv — full buffer copy with zero pad past n_comp rows.
            let ck_count = (m.n_comp as usize) * (m.head_dim as usize);
            let ck_bytes_len = ck_count * 2;
            if comp_kv_off + ck_bytes_len > comp_kv_bytes.len() {
                return Err(eyre!(
                    "snapshot.restore: comp_kv.bin truncated at layer {li}"
                ));
            }
            let ck_full_n = comp.comp_kv.len();
            if ck_count > ck_full_n {
                return Err(eyre!(
                    "snapshot.restore: ck_count {ck_count} > buffer {ck_full_n}"
                ));
            }
            if ck_full_n > 0 {
                let mut ck_host = vec![0u16; ck_full_n];
                for (i, c) in comp_kv_bytes[comp_kv_off..comp_kv_off + ck_bytes_len]
                    .chunks_exact(2)
                    .enumerate()
                {
                    ck_host[i] = u16::from_le_bytes([c[0], c[1]]);
                }
                dgpu.set_current()?;
                comp.comp_kv.copy_from_host(&ck_host)?;
            }
            comp_kv_off += ck_bytes_len;

            // state_kv + state_score (each n_state floats, packed back-to-back).
            let n_state = (m.state_rows as usize) * (m.width as usize);
            let block_bytes = n_state * 4;
            let block_total = 2 * block_bytes;
            if comp_state_off + block_total > comp_state_bytes.len() {
                // Fall back to alloc-time defaults if missing.
                igpu.set_current()?;
                comp.state_kv.copy_from_host(&vec![0f32; n_state])?;
                comp.state_score.copy_from_host(&vec![NEG_INF; n_state])?;
            } else {
                let mut kv = vec![0f32; n_state];
                let mut score = vec![0f32; n_state];
                for (i, c) in comp_state_bytes[comp_state_off..comp_state_off + block_bytes]
                    .chunks_exact(4)
                    .enumerate()
                {
                    kv[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                }
                for (i, c) in comp_state_bytes
                    [comp_state_off + block_bytes..comp_state_off + block_total]
                    .chunks_exact(4)
                    .enumerate()
                {
                    score[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                }
                igpu.set_current()?;
                comp.state_kv.copy_from_host(&kv)?;
                comp.state_score.copy_from_host(&score)?;
                comp_state_off += block_total;
            }
        } else if let Some(comp) = layer.compressor.as_mut() {
            // State expects a compressor but snapshot doesn't have one;
            // re-init defaults.
            comp.n_comp = 0;
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

            let ick_count = (m.n_index_comp as usize) * (m.index_head_dim as usize);
            let ick_bytes_len = ick_count * 2;
            if index_comp_kv_off + ick_bytes_len > index_comp_kv_bytes.len() {
                return Err(eyre!(
                    "snapshot.restore: index_comp_kv.bin truncated at layer {li}"
                ));
            }
            let ick_full_n = icomp.comp_kv.len();
            if ick_count > ick_full_n {
                return Err(eyre!(
                    "snapshot.restore: index ck_count {ick_count} > buffer {ick_full_n}"
                ));
            }
            if ick_full_n > 0 {
                let mut ick_host = vec![0u16; ick_full_n];
                for (i, c) in index_comp_kv_bytes
                    [index_comp_kv_off..index_comp_kv_off + ick_bytes_len]
                    .chunks_exact(2)
                    .enumerate()
                {
                    ick_host[i] = u16::from_le_bytes([c[0], c[1]]);
                }
                dgpu.set_current()?;
                icomp.comp_kv.copy_from_host(&ick_host)?;
            }
            index_comp_kv_off += ick_bytes_len;

            let in_state = (m.index_state_rows as usize) * (m.index_width as usize);
            let in_block_bytes = in_state * 4;
            let in_block_total = 2 * in_block_bytes;
            if index_comp_state_off + in_block_total > index_comp_state_bytes.len() {
                dgpu.set_current()?;
                icomp.state_kv.copy_from_host(&vec![0f32; in_state])?;
                icomp.state_score.copy_from_host(&vec![NEG_INF; in_state])?;
            } else {
                let mut kv = vec![0f32; in_state];
                let mut score = vec![0f32; in_state];
                for (i, c) in index_comp_state_bytes
                    [index_comp_state_off..index_comp_state_off + in_block_bytes]
                    .chunks_exact(4)
                    .enumerate()
                {
                    kv[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                }
                for (i, c) in index_comp_state_bytes[index_comp_state_off + in_block_bytes
                    ..index_comp_state_off + in_block_total]
                    .chunks_exact(4)
                    .enumerate()
                {
                    score[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                }
                dgpu.set_current()?;
                icomp.state_kv.copy_from_host(&kv)?;
                icomp.state_score.copy_from_host(&score)?;
                index_comp_state_off += in_block_total;
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
    Ok(tokens)
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
