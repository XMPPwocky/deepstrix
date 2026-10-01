//! Keys: the namespace (design 4.4) and the chained chunk / tail keys (4.3).
//!
//! Every key the store ever compares is produced by [`chunk_step`] or
//! [`tail_step`]. The walk ([`super::index::Index::walk`]) and the writer
//! ([`super::Store::write_chunk`] / `write_tail`) both go through them, and the
//! golden vector in the tests pins their output. That is the whole mitigation
//! for risk 2 ("a key-derivation bug orphans the whole store"): there is no
//! second implementation that could drift.
//!
//! WHY token ids and not decoded bytes (the v6 snapshot key): KV is a function
//! of the ids, chunk boundaries are token positions, and multistream already
//! refuses a token mismatch (4.3). Images are folded in by content hash at their
//! IMAGE_START position, so a cut anywhere after IMAGE_START (inside the block
//! too) carries the picture's identity: the synthetic ids inside a block only
//! encode its layout.

use super::format::StoreLayout;
use super::C;

/// A 32-byte blake3 output: a namespace, a chain key, or a tail key.
pub type Key = [u8; 32];

/// Keys are uniform blake3 outputs: their first 8 bytes are already a hash,
/// so the maps keyed by them skip SipHash (38K chunks, ~10K tails, lookups on
/// every walk step).
#[derive(Default, Clone, Copy)]
pub struct KeyHasher(u64);

impl std::hash::Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        // `[u8; 32]` hashes as a length prefix (write_usize) then its bytes.
        if bytes.len() >= 8 {
            self.0 ^= u64::from_le_bytes(bytes[..8].try_into().unwrap());
        } else {
            for b in bytes {
                self.0 = self.0.rotate_left(8) ^ *b as u64;
            }
        }
    }
    fn write_usize(&mut self, _len: usize) {}
}

pub type KeyBuild = std::hash::BuildHasherDefault<KeyHasher>;
pub type KeyMap<V> = std::collections::HashMap<Key, V, KeyBuild>;
pub type KeySet = std::collections::HashSet<Key, KeyBuild>;

/// `blake3::derive_key` contexts. Separate contexts keep the three kinds of
/// hash apart: no chunk key can ever equal a tail key or a namespace.
pub const CTX_NAMESPACE: &str = "deepstrix kvstore v1 namespace";
pub const CTX_CHUNK: &str = "deepstrix kvstore v1 chunk";
pub const CTX_TAIL: &str = "deepstrix kvstore v1 tail";

/// One image of a request, at an absolute token position.
///
/// `start` is the IMAGE_START index (as `vision_prompt::ImageSpan::start`),
/// `len` the block length through IMAGE_END, `hash` the tower input's content
/// hash. Requests hold these sorted by `start`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageRecord {
    pub start: u32,
    pub len: u32,
    pub hash: [u8; 32],
}

impl From<&crate::vision_prompt::ImageSpan> for ImageRecord {
    fn from(s: &crate::vision_prompt::ImageSpan) -> Self {
        Self { start: s.start, len: s.len, hash: s.hash }
    }
}

impl ImageRecord {
    /// `le32(p − a) ‖ le32(len) ‖ content_hash`: the bytes that are hashed
    /// into a key AND stored in the file (4.5), relative to the start `a` of
    /// the chunk or tail that carries the record.
    pub fn to_bytes(&self, a: u32) -> [u8; 40] {
        debug_assert!(self.start >= a);
        let mut out = [0u8; 40];
        out[0..4].copy_from_slice(&(self.start - a).to_le_bytes());
        out[4..8].copy_from_slice(&self.len.to_le_bytes());
        out[8..40].copy_from_slice(&self.hash);
        out
    }

    pub fn from_bytes(b: &[u8; 40], a: u32) -> Option<Self> {
        let rel = u32::from_le_bytes(b[0..4].try_into().unwrap());
        let len = u32::from_le_bytes(b[4..8].try_into().unwrap());
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&b[8..40]);
        Some(Self { start: a.checked_add(rel)?, len, hash })
    }
}

/// The images whose IMAGE_START lies in `[a, b)`, from a list sorted by
/// `start`. This is `img(a, b)` in 4.3.
pub fn images_in(images: &[ImageRecord], a: u32, b: u32) -> &[ImageRecord] {
    debug_assert!(images.windows(2).all(|w| w[0].start < w[1].start), "images must be sorted by start");
    let lo = images.partition_point(|r| r.start < a);
    let hi = images.partition_point(|r| r.start < b);
    &images[lo..hi.max(lo)]
}

/// Fold token ids into a hasher as `le32` each, in one `update` per 1 KiB
/// of ids (per-id updates would cost ~10x on a 4 KiB chunk).
fn update_tokens(h: &mut blake3::Hasher, tokens: &[i32]) {
    let mut buf = [0u8; 4096];
    for part in tokens.chunks(buf.len() / 4) {
        for (i, t) in part.iter().enumerate() {
            buf[4 * i..4 * i + 4].copy_from_slice(&t.to_le_bytes());
        }
        h.update(&buf[..4 * part.len()]);
    }
}

fn update_images(h: &mut blake3::Hasher, images: &[ImageRecord], a: u32) {
    for r in images {
        h.update(&r.to_bytes(a));
    }
}

/// `chain_0 = derive_key(CTX_CHUNK, ns)`: the root of a namespace's trie.
pub fn chain_root(ns: &Key) -> Key {
    blake3::derive_key(CTX_CHUNK, ns)
}

/// `chain_{k+1} = derive_key(CTX_CHUNK, chain_k ‖ le32(tok[kC..(k+1)C]) ‖ img(kC, (k+1)C))`.
///
/// `a` is kC. `tokens` must be exactly C ids. `images` are the request's
/// images whose start is in `[a, a + C)` (absolute positions).
pub fn chunk_step(parent: &Key, a: u32, tokens: &[i32], images: &[ImageRecord]) -> Key {
    assert_eq!(tokens.len(), C as usize, "a chunk key covers exactly C positions");
    debug_assert!(a % C == 0);
    debug_assert!(images.iter().all(|r| r.start >= a && r.start < a + C));
    let mut h = blake3::Hasher::new_derive_key(CTX_CHUNK);
    h.update(parent);
    update_tokens(&mut h, tokens);
    update_images(&mut h, images, a);
    *h.finalize().as_bytes()
}

/// `tail(T) = derive_key(CTX_TAIL, chain_b ‖ le32(T − bC) ‖ le32(tok[bC..T]) ‖ img(bC, T))`
/// with b = ⌊T/C⌋.
///
/// `a` is bC; `open` is `tok[bC..T]` (0..C−1 ids; empty for a tail exactly at a
/// chunk boundary, e.g. a waypoint). The open length is hashed explicitly, so
/// the open-id bytes alone never have to disambiguate it.
pub fn tail_step(base: &Key, a: u32, open: &[i32], images: &[ImageRecord]) -> Key {
    assert!(open.len() < C as usize, "a tail's open part is shorter than one chunk");
    debug_assert!(a % C == 0);
    let t = a + open.len() as u32;
    debug_assert!(images.iter().all(|r| r.start >= a && r.start < t));
    let mut h = blake3::Hasher::new_derive_key(CTX_TAIL);
    h.update(base);
    h.update(&(open.len() as u32).to_le_bytes());
    update_tokens(&mut h, open);
    update_images(&mut h, images, a);
    *h.finalize().as_bytes()
}

/// Per-namespace key derivation: the root plus helpers over a whole request.
#[derive(Debug, Clone)]
pub struct KeyChain {
    ns: Key,
    root: Key,
}

impl KeyChain {
    pub fn new(ns: Key) -> Self {
        Self { ns, root: chain_root(&ns) }
    }

    pub fn ns(&self) -> &Key {
        &self.ns
    }

    /// chain_0.
    pub fn root(&self) -> &Key {
        &self.root
    }

    /// A lazily extended list of chain keys for one request. A job keeps one
    /// for its request so each chunk write costs one `chunk_step`, not k.
    pub fn cursor(&self) -> ChainCursor {
        ChainCursor { keys: vec![self.root] }
    }

    /// `tail(t)` over `tokens[..t]`, with chain_b from `cursor` (b = ⌊t/C⌋).
    /// Returns `(chain_b, tail key)`.
    pub fn tail_key(&self, cursor: &mut ChainCursor, tokens: &[i32], images: &[ImageRecord], t: u32) -> (Key, Key) {
        assert!(t as usize <= tokens.len(), "tail at {t} past the request's {} tokens", tokens.len());
        let b = t / C;
        let base = cursor.chain(tokens, images, b);
        let a = b * C;
        let key = tail_step(&base, a, &tokens[a as usize..t as usize], images_in(images, a, t));
        (base, key)
    }
}

/// chain_0..chain_m of one request, computed on demand.
///
/// It records only keys, not the tokens they were computed from: a cursor must
/// be used with the request it was created for (one per job).
#[derive(Debug, Clone)]
pub struct ChainCursor {
    keys: Vec<Key>,
}

impl ChainCursor {
    /// chain_b over `tokens` (needs `b * C <= tokens.len()`).
    pub fn chain(&mut self, tokens: &[i32], images: &[ImageRecord], b: u32) -> Key {
        assert!((b as usize) * (C as usize) <= tokens.len(), "chain_{b} needs {} tokens, have {}", b * C, tokens.len());
        while self.keys.len() <= b as usize {
            let k = (self.keys.len() - 1) as u32;
            let a = k * C;
            let next = chunk_step(
                &self.keys[k as usize],
                a,
                &tokens[a as usize..(a + C) as usize],
                images_in(images, a, a + C),
            );
            self.keys.push(next);
        }
        self.keys[b as usize]
    }

    /// Seed with keys already known for this request (the walk's matched
    /// chunks, chain_1..chain_m), so a job never re-hashes its restored prefix.
    ///
    /// Panics if the cursor already holds keys that disagree with `matched`:
    /// it was created for a different request (a caller bug that would write
    /// files under keys no walk can reach).
    pub fn seed(&mut self, matched: &[Key]) {
        for (i, k) in matched.iter().enumerate() {
            match self.keys.get(i + 1) {
                Some(have) => assert_eq!(have, k, "ChainCursor::seed: chain_{} differs; cursor of another request", i + 1),
                None => self.keys.push(*k),
            }
        }
    }

    /// chain_0..chain_b (extended over `tokens` as needed).
    pub fn keys_to(&mut self, tokens: &[i32], images: &[ImageRecord], b: u32) -> &[Key] {
        self.chain(tokens, images, b);
        &self.keys[..=b as usize]
    }

    pub fn known(&self) -> u32 {
        (self.keys.len() - 1) as u32
    }
}

/// Everything that decides what stored KV MEANS (4.4). A change in any field
/// starts a new namespace, i.e. a cold start for every conversation (~350 s
/// each, serial), so only semantics and ABI belong here. Numerics knobs and the
/// build id go into file headers instead ([`super::format::GenPair`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamespaceInputs {
    /// The model fingerprint (v6's `ModelFingerprint`, hashed by
    /// [`model_fingerprint_hash`]).
    pub model: [u8; 32],
    /// The vision tower fingerprint: tower rows depend on its weights and the
    /// keys carry only the pixel hash. All-zero when no tower is loaded.
    pub tower: [u8; 32],
    /// The Engram identity: hasher config + the tables' tensor directory.
    pub engram: [u8; 32],
    /// Per KV-source store: layer, ratio, encodings, row/key/state bytes.
    pub stores: Vec<StoreLayout>,
    /// C.
    pub chunk_positions: u32,
    pub swa_window: u32,
    pub ced_decoder_start: u32,
    /// `V41_CED`: decides what a full tail's decoder windows mean (5.3).
    pub ced: bool,
    /// `V41_INDEX_K`: whether chunks carry index keys at all.
    pub index_k: bool,
    /// [`super::KV_EPOCH`] (4.4.1).
    pub kv_epoch: u32,
    /// [`super::format::FORMAT_VERSION`]: a format bump starts a new namespace,
    /// so neither the new binary's scan nor a rolled-back one unlinks the
    /// other's files as "bad version"; the old namespace is kept as the
    /// inactive one and evicted first.
    pub format: u32,
}

impl NamespaceInputs {
    /// The canonical byte string hashed into the namespace: a version tag,
    /// then every field tagged and fixed-width. Field order is part of the
    /// ABI; never reorder, only append (with a new tag).
    pub fn encode(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(256);
        v.extend_from_slice(b"ns-v1\0");
        v.extend_from_slice(b"model\0");
        v.extend_from_slice(&self.model);
        v.extend_from_slice(b"tower\0");
        v.extend_from_slice(&self.tower);
        v.extend_from_slice(b"engram\0");
        v.extend_from_slice(&self.engram);
        v.extend_from_slice(b"stores\0");
        v.extend_from_slice(&(self.stores.len() as u32).to_le_bytes());
        for s in &self.stores {
            v.extend_from_slice(&s.encode());
        }
        v.extend_from_slice(b"c\0");
        v.extend_from_slice(&self.chunk_positions.to_le_bytes());
        v.extend_from_slice(b"swa\0");
        v.extend_from_slice(&self.swa_window.to_le_bytes());
        v.extend_from_slice(b"ced_start\0");
        v.extend_from_slice(&self.ced_decoder_start.to_le_bytes());
        v.extend_from_slice(b"ced\0");
        v.push(self.ced as u8);
        v.extend_from_slice(b"index_k\0");
        v.push(self.index_k as u8);
        v.extend_from_slice(b"epoch\0");
        v.extend_from_slice(&self.kv_epoch.to_le_bytes());
        v.extend_from_slice(b"format\0");
        v.extend_from_slice(&self.format.to_le_bytes());
        v
    }

    pub fn key(&self) -> Key {
        blake3::derive_key(CTX_NAMESPACE, &self.encode())
    }

    /// The V4.1 inputs for this binary: layout constants from `config`, the
    /// three identities from the caller (M2 computes them at load time).
    pub fn v41(model: [u8; 32], tower: [u8; 32], engram: [u8; 32], ced: bool, index_k: bool) -> Self {
        use v4flash_kernels::config::{CED_DECODER_START, SWA_WINDOW};
        Self {
            model,
            tower,
            engram,
            stores: super::format::v41_stores().to_vec(),
            chunk_positions: C,
            swa_window: SWA_WINDOW,
            ced_decoder_start: CED_DECODER_START as u32,
            ced,
            index_k,
            kv_epoch: super::KV_EPOCH,
            format: super::format::FORMAT_VERSION as u32,
        }
    }
}

/// A canonical hash of v6's [`crate::snapshot::ModelFingerprint`] (the
/// identity the old store already trusts), for [`NamespaceInputs::model`].
pub fn model_fingerprint_hash(fp: &crate::snapshot::ModelFingerprint) -> [u8; 32] {
    let mut h = blake3::Hasher::new_derive_key("deepstrix kvstore v1 model fingerprint");
    h.update(&fp.n_layer.to_le_bytes());
    h.update(&fp.n_head_dim.to_le_bytes());
    h.update(&fp.vocab_size.to_le_bytes());
    for s in [&fp.token_embd_prefix_blake3, &fp.tensor_directory_blake3] {
        h.update(&(s.len() as u32).to_le_bytes());
        h.update(s.as_bytes());
    }
    *h.finalize().as_bytes()
}

/// Lowercase hex of a key (file names, logs).
pub fn hex(key: &[u8]) -> String {
    const D: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(key.len() * 2);
    for b in key {
        s.push(D[(b >> 4) as usize] as char);
        s.push(D[(b & 15) as usize] as char);
    }
    s
}

pub fn parse_hex_key(s: &str) -> Option<Key> {
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, o) in out.iter_mut().enumerate() {
        *o = u8::from_str_radix(s.get(2 * i..2 * i + 2)?, 16).ok()?;
    }
    Some(out)
}

/// `<ns16>`: the namespace's directory name under `kvstore-v1/` (4.4).
pub fn ns16(ns: &Key) -> String {
    hex(&ns[..8])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(n: usize, seed: i32) -> Vec<i32> {
        (0..n as i32).map(|i| i.wrapping_mul(7919).wrapping_add(seed) % 129_000).collect()
    }

    /// The reference derivation, written as plainly as 4.3 states it (one
    /// `derive_key` over a concatenated buffer), independent of the
    /// incremental hasher used in production.
    fn reference_chain(ns: &Key, tokens: &[i32], images: &[ImageRecord], m: usize) -> Vec<Key> {
        let mut keys = vec![blake3::derive_key(CTX_CHUNK, ns)];
        for k in 0..m {
            let a = (k as u32) * C;
            let mut buf = keys[k].to_vec();
            for t in &tokens[a as usize..(a + C) as usize] {
                buf.extend_from_slice(&t.to_le_bytes());
            }
            for r in images.iter().filter(|r| r.start >= a && r.start < a + C) {
                buf.extend_from_slice(&(r.start - a).to_le_bytes());
                buf.extend_from_slice(&r.len.to_le_bytes());
                buf.extend_from_slice(&r.hash);
            }
            keys.push(blake3::derive_key(CTX_CHUNK, &buf));
        }
        keys
    }

    fn reference_tail(base: &Key, tokens: &[i32], images: &[ImageRecord], t: u32) -> Key {
        let a = t / C * C;
        let mut buf = base.to_vec();
        buf.extend_from_slice(&(t - a).to_le_bytes());
        for x in &tokens[a as usize..t as usize] {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        for r in images.iter().filter(|r| r.start >= a && r.start < t) {
            buf.extend_from_slice(&(r.start - a).to_le_bytes());
            buf.extend_from_slice(&r.len.to_le_bytes());
            buf.extend_from_slice(&r.hash);
        }
        blake3::derive_key(CTX_TAIL, &buf)
    }

    fn golden_ns() -> NamespaceInputs {
        NamespaceInputs {
            model: [1; 32],
            tower: [2; 32],
            engram: [3; 32],
            stores: super::super::format::v41_stores().to_vec(),
            chunk_positions: C,
            swa_window: 128,
            ced_decoder_start: 20,
            ced: true,
            index_k: true,
            kv_epoch: 1,
            format: 1,
        }
    }

    fn golden_images() -> Vec<ImageRecord> {
        // One image starting inside chunk 1 and running into chunk 2 (a block
        // that straddles a chunk boundary), one starting in the open part of
        // the golden tail.
        vec![
            ImageRecord { start: 1500, len: 700, hash: [9; 32] },
            ImageRecord { start: 2100, len: 10, hash: [8; 32] },
        ]
    }

    /// G5 golden vector. These hex strings ARE the on-disk key ABI: if this
    /// test fails, every chunk and tail already stored becomes unreachable.
    /// Change them only together with `KV_EPOCH` or a new derive context.
    #[test]
    fn golden_key_vector() {
        let ns = golden_ns().key();
        let tokens = toks(2 * C as usize + 300, 11);
        let images = golden_images();
        let chain = KeyChain::new(ns);
        let mut cur = chain.cursor();
        let c1 = cur.chain(&tokens, &images, 1);
        let c2 = cur.chain(&tokens, &images, 2);
        let (base, t_key) = chain.tail_key(&mut cur, &tokens, &images, 2 * C + 200);
        assert_eq!(base, c2);
        let (_, w_key) = chain.tail_key(&mut cur, &tokens, &images, 2 * C);

        // Independent derivation agrees.
        let r = reference_chain(&ns, &tokens, &images, 2);
        assert_eq!((r[0], r[1], r[2]), (*chain.root(), c1, c2));
        assert_eq!(t_key, reference_tail(&c2, &tokens, &images, 2 * C + 200));
        assert_eq!(w_key, reference_tail(&c2, &tokens, &images, 2 * C));

        let got = [hex(&ns), hex(&c1), hex(&c2), hex(&t_key), hex(&w_key)];
        let want = [
            "226dc4b3153eb9136e83333dacbb6fe87b1f8744cc05c33ea4a4d089095b060a",
            "17e7943790d51b2d3676f14911b4461f6623d603debfaaf0e73d52fab2b4e05e",
            "e26f8ec96344183d9008dcd73eacd5f7a70497ecd4ba35356e4c01fd8f46d793",
            "1ee14fcc1563032e91f8bc87d48ff2d8747a8e010c1c77333cf0cfbd515aae7d",
            "64f6c89dc36bc0bbb99190f82da650665d750a24e96d3861ec2218082916445a",
        ];
        assert!(got == want, "key derivation changed (ns, chain_1, chain_2, tail(2C+200), tail(2C)):\n{got:#?}");
    }

    #[test]
    fn incremental_matches_reference_for_every_prefix_length() {
        let ns = golden_ns().key();
        let tokens = toks(3 * C as usize + 17, 5);
        let images = golden_images();
        let chain = KeyChain::new(ns);
        let r = reference_chain(&ns, &tokens, &images, 3);
        let mut cur = chain.cursor();
        for t in 0..=tokens.len() as u32 {
            let (base, key) = chain.tail_key(&mut cur, &tokens, &images, t);
            assert_eq!(base, r[(t / C) as usize], "chain_b at t={t}");
            assert_eq!(key, reference_tail(&base, &tokens, &images, t), "tail key at t={t}");
        }
    }

    #[test]
    fn image_identity_reaches_cuts_inside_the_block() {
        let ns = golden_ns().key();
        let tokens = toks(3 * C as usize, 3);
        let a = vec![ImageRecord { start: 900, len: 700, hash: [1; 32] }];
        let b = vec![ImageRecord { start: 900, len: 700, hash: [2; 32] }];
        let chain = KeyChain::new(ns);
        // A cut inside the block (1000 < 900 + 700), in the open part of chunk 0.
        let (_, ka) = chain.tail_key(&mut chain.cursor(), &tokens, &a, 1000);
        let (_, kb) = chain.tail_key(&mut chain.cursor(), &tokens, &b, 1000);
        assert_ne!(ka, kb, "different pixels, same layout: keys must differ inside the block");
        // A cut in the NEXT chunk, still inside the block: carried by chain_1.
        let (ba, ka2) = chain.tail_key(&mut chain.cursor(), &tokens, &a, 1500);
        let (bb, kb2) = chain.tail_key(&mut chain.cursor(), &tokens, &b, 1500);
        assert_ne!(ba, bb);
        assert_ne!(ka2, kb2);
        // A cut before IMAGE_START does not see the image at all.
        let (_, k0a) = chain.tail_key(&mut chain.cursor(), &tokens, &a, 900);
        let (_, k0b) = chain.tail_key(&mut chain.cursor(), &tokens, &b, 900);
        assert_eq!(k0a, k0b);
    }

    #[test]
    fn contexts_and_namespaces_separate() {
        let mut n = golden_ns();
        let k0 = n.key();
        n.ced = false;
        assert_ne!(k0, n.key(), "CED is part of the namespace");
        n.ced = true;
        n.kv_epoch = 2;
        assert_ne!(k0, n.key(), "KV_EPOCH is part of the namespace");
        n.kv_epoch = 1;
        n.format = 2;
        assert_ne!(k0, n.key(), "FORMAT_VERSION is part of the namespace");
        n.format = 1;
        assert_eq!(k0, n.key());
        // The same input under the three contexts gives three keys.
        let x = [7u8; 32];
        let a = blake3::derive_key(CTX_NAMESPACE, &x);
        let b = blake3::derive_key(CTX_CHUNK, &x);
        let c = blake3::derive_key(CTX_TAIL, &x);
        assert!(a != b && b != c && a != c);
    }

    #[test]
    fn cursor_seed_extends_and_checks() {
        let chain = KeyChain::new(golden_ns().key());
        let tokens = toks(3 * C as usize, 9);
        let mut full = chain.cursor();
        let keys: Vec<Key> = full.keys_to(&tokens, &[], 3)[1..].to_vec();
        let mut c = chain.cursor();
        c.chain(&tokens, &[], 1);
        c.seed(&keys); // overlaps chain_1, extends to chain_3
        assert_eq!(c.known(), 3);
        assert_eq!(c.chain(&tokens, &[], 3), keys[2]);
        let other = toks(3 * C as usize, 10);
        let mut d = chain.cursor();
        d.chain(&other, &[], 1);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| d.seed(&keys)));
        assert!(r.is_err(), "seeding a cursor of another request must not pass silently");
    }

    #[test]
    fn key_hasher_spreads_keys() {
        use std::hash::BuildHasher;
        let b = KeyBuild::default();
        let k1: Key = core::array::from_fn(|i| i as u8);
        let mut k2 = k1;
        k2[0] ^= 1;
        assert_ne!(b.hash_one(k1), b.hash_one(k2));
        let mut m: KeyMap<u32> = KeyMap::default();
        m.insert(k1, 1);
        m.insert(k2, 2);
        assert_eq!((m[&k1], m[&k2]), (1, 2));
    }

    #[test]
    fn hex_roundtrip_and_images_in() {
        let k: Key = core::array::from_fn(|i| (i * 37) as u8);
        assert_eq!(parse_hex_key(&hex(&k)), Some(k));
        assert_eq!(parse_hex_key("zz"), None);
        let imgs = vec![
            ImageRecord { start: 5, len: 1, hash: [0; 32] },
            ImageRecord { start: 10, len: 1, hash: [0; 32] },
            ImageRecord { start: 20, len: 1, hash: [0; 32] },
        ];
        assert_eq!(images_in(&imgs, 0, 5).len(), 0);
        assert_eq!(images_in(&imgs, 5, 11).len(), 2);
        assert_eq!(images_in(&imgs, 11, 20).len(), 0);
        assert_eq!(images_in(&imgs, 20, 21)[0].start, 20);
    }
}
