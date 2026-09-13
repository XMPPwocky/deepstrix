//! DeepSeek-V4.1 Engram n-gram hashing (reference: `inference/engram.py`,
//! `NgramHashState`), token-only, reproduced bit-for-bit from the tables
//! `scripts/v41_oracle/export_engram_hash.py` dumps (`engram_hash.json` +
//! `token_map.bin`): the NFKC/NFD/StripAccents/Lowercase-compressed token
//! map, per-layer odd multipliers, the (layer, n-gram order, head) primes and
//! the bucket offsets. Per position and Engram layer:
//!
//! ```text
//! c[s]      = token_map[id[pos - s]]   s = 0..3   (pad_id when pos < s)
//! prod[s]   = c[s] * mult[layer][s]                (i64, bounded: no overflow)
//! rolling   = prod[0]; for n in 1..3: rolling ^= prod[n]
//!             col (n-1)*8 + head = rolling % prime[layer][n-1][head] + offset[layer][col]
//! ```
//!
//! Hashes depend only on the token ids, so rows can be gathered as soon as a
//! token is known (ENGINE_PORT.md M2: prefetch off the critical path).
//!
//! Image spans (`NgramHashState.forward` with `token_mask`, model.py
//! `engram_mask = ~image_mask`): every position inside an image span is a
//! DEAD token. A dead position takes no Engram contribution (the reference
//! zeroes the gate; the engine gets all-zero rows, and `wkv` has no bias, so
//! `value = 0` exactly) and blocks the look-back: once a shift lands on a dead
//! token (or before the sequence start) that and every older slot is `pad_id`,
//! so no n-gram ever straddles an image. [`EngramHash::compress`] returns
//! [`DEAD`] for every id outside the vocabulary — the server's image slots
//! carry synthetic ids `>= vocab` (`v4flash_vision::synthetic_token_id`),
//! which is exactly the reference's `token_types >= 0` mask.

use std::path::Path;

use color_eyre::eyre::{self, eyre};

pub const ENGRAM_LAYERS: usize = 2;
pub const ENGRAM_NGRAM: usize = 4; // max_ngram_size: the current token + 3 look-backs
pub const ENGRAM_HEADS: usize = 8;
pub const ENGRAM_COLS: usize = (ENGRAM_NGRAM - 1) * ENGRAM_HEADS; // 24 rows per position per layer
pub const ENGRAM_ROW_DIM: usize = 256;
/// Compressed id of a token that takes no part in any n-gram (an image-span
/// position). `NgramHashState.DEAD` in the reference.
pub const DEAD: i32 = -1;

pub struct EngramHash {
    pub layer_ids: [usize; ENGRAM_LAYERS],
    pub num_embeddings: [u64; ENGRAM_LAYERS],
    pub pad_id: i32,
    token_map: Vec<i32>,
    multipliers: [[i64; ENGRAM_NGRAM]; ENGRAM_LAYERS],
    primes: [[[i64; ENGRAM_HEADS]; ENGRAM_NGRAM - 1]; ENGRAM_LAYERS],
    offsets: [[i64; ENGRAM_COLS]; ENGRAM_LAYERS],
    /// Self-check from the dump: (prompt ids, hash ids [T][layer][col]).
    pub check: Option<(Vec<i32>, Vec<Vec<Vec<i64>>>)>,
}

fn i64s(v: &serde_json::Value, what: &str) -> eyre::Result<Vec<i64>> {
    v.as_array()
        .ok_or_else(|| eyre!("{what}: not an array"))?
        .iter()
        .map(|x| x.as_i64().ok_or_else(|| eyre!("{what}: not an integer")))
        .collect()
}

impl EngramHash {
    /// Load from the directory holding `engram_hash.json` and `token_map.bin`.
    pub fn load(dir: &Path) -> eyre::Result<Self> {
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("engram_hash.json"))?)?;
        let layer_ids = i64s(&meta["layer_ids"], "layer_ids")?;
        if layer_ids.len() != ENGRAM_LAYERS || meta["max_ngram_size"].as_u64() != Some(ENGRAM_NGRAM as u64)
            || meta["n_heads"].as_u64() != Some(ENGRAM_HEADS as u64) || meta["head_dim"].as_u64() != Some(ENGRAM_ROW_DIM as u64)
        {
            return Err(eyre!("engram_hash.json: unexpected layout {}", meta["layer_ids"]));
        }
        let num_embeddings = i64s(&meta["num_embeddings"], "num_embeddings")?;
        let bufs = &meta["buffers"];
        let tm = &bufs["token_map"];
        let token_map_bytes = std::fs::read(dir.join(tm["file"].as_str().ok_or_else(|| eyre!("token_map.file"))?))?;
        if tm["dtype"] != "i32" {
            return Err(eyre!("token_map dtype {} (want i32)", tm["dtype"]));
        }
        let token_map: Vec<i32> = token_map_bytes.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
        let mut out = Self {
            layer_ids: [layer_ids[0] as usize, layer_ids[1] as usize],
            num_embeddings: [num_embeddings[0] as u64, num_embeddings[1] as u64],
            pad_id: meta["pad_id"].as_i64().ok_or_else(|| eyre!("pad_id"))? as i32,
            token_map,
            multipliers: [[0; ENGRAM_NGRAM]; ENGRAM_LAYERS],
            primes: [[[0; ENGRAM_HEADS]; ENGRAM_NGRAM - 1]; ENGRAM_LAYERS],
            offsets: [[0; ENGRAM_COLS]; ENGRAM_LAYERS],
            check: None,
        };
        for l in 0..ENGRAM_LAYERS {
            let m = i64s(&bufs["multipliers"]["values"][l], "multipliers")?;
            out.multipliers[l].copy_from_slice(&m);
            for n in 0..ENGRAM_NGRAM - 1 {
                let p = i64s(&bufs["primes"]["values"][l][n], "primes")?;
                out.primes[l][n].copy_from_slice(&p);
            }
            let o = i64s(&bufs["offsets"]["values"][l], "offsets")?;
            out.offsets[l].copy_from_slice(&o);
        }
        if let Some(c) = meta.get("check") {
            let ids = i64s(&c["prompt_ids"], "check.prompt_ids")?.iter().map(|&x| x as i32).collect();
            let mut h = Vec::new();
            for t in c["hash_ids"].as_array().ok_or_else(|| eyre!("check.hash_ids"))? {
                let mut per_layer = Vec::new();
                for l in t.as_array().ok_or_else(|| eyre!("check.hash_ids[t]"))? {
                    per_layer.push(i64s(l, "check.hash_ids[t][l]")?);
                }
                h.push(per_layer);
            }
            out.check = Some((ids, h));
        }
        Ok(out)
    }

    /// Compressed id of a token (the n-gram alphabet), or [`DEAD`] for an id
    /// outside the vocabulary.
    ///
    /// Load-bearing for VL: image spans carry synthetic ids past
    /// `token_map.len()` (`v4flash_vision::synthetic_token_id`), which the
    /// reference masks out of Engram entirely (`engram_mask = ~image_mask`
    /// in model.py, `token_mask` → `DEAD` in engram.py). Mapping them to
    /// `DEAD` here reproduces that: [`Self::hash_ids`] blocks the look-back
    /// at a dead slot, and the caller stages all-zero rows for dead
    /// positions (see `EngramCtx` in the server). A negative id is not a
    /// token either and would wrap through `as usize`, so it is dead too.
    #[inline]
    pub fn compress(&self, id: i32) -> i32 {
        if id < 0 {
            return DEAD;
        }
        self.token_map.get(id as usize).copied().unwrap_or(DEAD)
    }

    /// Hash ids for position `pos` given the compressed ids `c[0..=pos]`
    /// (only the last four are read; [`DEAD`] entries mark image tokens).
    /// Rows for Engram layer `l` are `out[l]`, already offset into that
    /// layer's table.
    ///
    /// Look-back rule (`NgramHashState.forward`): walking shifts 0..3, once a
    /// shift falls before the sequence start or on a dead token, that slot
    /// AND every older one read `pad_id` (`blocked` is cumulative). A dead
    /// position itself therefore hashes the all-pad n-gram; its rows are
    /// never used (the caller zeroes them), matching the zeroed gate.
    pub fn hash_ids(&self, c: &[i32], pos: usize) -> [[i64; ENGRAM_COLS]; ENGRAM_LAYERS] {
        let mut toks = [self.pad_id as i64; ENGRAM_NGRAM];
        let mut blocked = false;
        for (s, t) in toks.iter_mut().enumerate() {
            blocked |= pos < s || c[pos - s] == DEAD;
            if !blocked {
                *t = c[pos - s] as i64;
            }
        }
        let mut out = [[0i64; ENGRAM_COLS]; ENGRAM_LAYERS];
        for l in 0..ENGRAM_LAYERS {
            let tok = |s: usize| -> i64 { toks[s] };
            let mut rolling = tok(0).wrapping_mul(self.multipliers[l][0]);
            for n in 1..ENGRAM_NGRAM {
                rolling ^= tok(n).wrapping_mul(self.multipliers[l][n]);
                debug_assert!(rolling >= 0, "engram hash: negative rolling value (multiplier bound violated)");
                for h in 0..ENGRAM_HEADS {
                    let col = (n - 1) * ENGRAM_HEADS + h;
                    out[l][col] = rolling.rem_euclid(self.primes[l][n - 1][h]) + self.offsets[l][col];
                }
            }
        }
        out
    }

    /// Convenience: hash ids for every position of a token-id sequence.
    pub fn hash_sequence(&self, ids: &[i32]) -> Vec<[[i64; ENGRAM_COLS]; ENGRAM_LAYERS]> {
        let c: Vec<i32> = ids.iter().map(|&i| self.compress(i)).collect();
        (0..ids.len()).map(|p| self.hash_ids(&c, p)).collect()
    }

    /// Bit-equality against the dump's own self-check (the 6-token " Paris" prompt).
    pub fn self_check(&self) -> eyre::Result<()> {
        let (ids, want) = self.check.as_ref().ok_or_else(|| eyre!("engram_hash.json has no check block"))?;
        let got = self.hash_sequence(ids);
        for (t, (g, w)) in got.iter().zip(want).enumerate() {
            for l in 0..ENGRAM_LAYERS {
                if g[l][..] != w[l][..] {
                    return Err(eyre!("engram hash self-check: T{t} layer {} differs: got {:?} want {:?}", self.layer_ids[l], &g[l][..4], &w[l][..4]));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The look-back rule with dead tokens, against a hand-run of
    /// `NgramHashState.forward` on a tiny table. Positions 0..2 text,
    /// 3..4 image (DEAD), 5..6 text.
    #[test]
    fn dead_tokens_block_the_lookback() {
        let h = EngramHash {
            layer_ids: [1, 14],
            num_embeddings: [1 << 40, 1 << 40],
            pad_id: 7,
            token_map: vec![10, 11, 12, 13, 14, 15],
            multipliers: [[3, 5, 7, 11]; ENGRAM_LAYERS],
            primes: [[[1 << 30; ENGRAM_HEADS]; ENGRAM_NGRAM - 1]; ENGRAM_LAYERS],
            offsets: [[0; ENGRAM_COLS]; ENGRAM_LAYERS],
            check: None,
        };
        // ids: 0 1 2 | 100 100 (out of vocab → DEAD) | 3 4
        let ids = [0, 1, 2, 100, 100, 3, 4];
        let c: Vec<i32> = ids.iter().map(|&i| h.compress(i)).collect();
        assert_eq!(c, vec![10, 11, 12, DEAD, DEAD, 13, 14]);
        let col0 = |p: usize| h.hash_ids(&c, p)[0][0]; // (1-gram → 2-gram) bucket, head 0, layer 0
        let hash = |t0: i64, t1: i64| (t0 * 3) ^ (t1 * 5);
        assert_eq!(col0(0), hash(10, 7)); // before the start → pad
        assert_eq!(col0(2), hash(12, 11));
        assert_eq!(col0(3), hash(7, 7)); // dead position: all pad
        assert_eq!(col0(5), hash(13, 7)); // look-back hits the dead span → pad
        assert_eq!(col0(6), hash(14, 13));
        // 4-gram column of position 6: [14, 13, DEAD→pad, pad] — blocked is sticky.
        let col16 = h.hash_ids(&c, 6)[0][2 * ENGRAM_HEADS];
        assert_eq!(col16, (14 * 3) ^ (13 * 5) ^ (7 * 7) ^ (7 * 11));
        // Negative ids are dead, never wrap.
        assert_eq!(h.compress(-5), DEAD);
    }

    fn dir() -> Option<std::path::PathBuf> {
        let p = std::path::PathBuf::from(std::env::var("HOME").ok()?).join(".cache/deepstrix/v41/engram");
        p.join("engram_hash.json").exists().then_some(p)
    }

    #[test]
    fn matches_reference_self_check() -> eyre::Result<()> {
        let Some(d) = dir() else { eprintln!("no engram dump; skipping"); return Ok(()); };
        let h = EngramHash::load(&d)?;
        assert_eq!(h.layer_ids, [1, 14]);
        h.self_check()?;
        // Every row id lands inside its (layer, order, head) bucket and the table.
        let ids = h.check.as_ref().unwrap().0.clone();
        for hs in h.hash_sequence(&ids) {
            for l in 0..ENGRAM_LAYERS {
                for col in 0..ENGRAM_COLS {
                    let (n, head) = (col / ENGRAM_HEADS, col % ENGRAM_HEADS);
                    let lo = h.offsets[l][col];
                    let hi = lo + h.primes[l][n][head];
                    assert!(hs[l][col] >= lo && hs[l][col] < hi && (hs[l][col] as u64) < h.num_embeddings[l]);
                }
            }
        }
        Ok(())
    }
}
