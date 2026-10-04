//! Qwen3-Embedding (GGUF `general.architecture = qwen3`): the model's
//! description and the host-side pieces of the hub's embed phase
//! (docs/v41/EMBED_PHASE_DESIGN.md), plus a CPU reference forward that is the
//! oracle for the GPU path.
//!
//! The weights are never resident: a phase streams one layer at a time out of
//! the GGUF ([`Qwen3EmbedModel::read_layer_into`]) into a weight-ring slot laid
//! out by [`SlotLayout`]. Only `output_norm` (10 KB) is kept in host memory.

use std::collections::HashMap;

use color_eyre::eyre::{self, eyre};

use crate::gguf::{Gguf, GgufTensor, GgufType};
use crate::kquants::dequant_to_f32;
use crate::mapped::MappedGguf;

/// The text the reference tokenizer appends to every input; its hidden state
/// is the embedding (last-token pooling).
pub const EOS_TEXT: &str = "<|endoftext|>";

/// Alignment of every tensor inside a ring slot (and of slot sizes).
pub const SLOT_ALIGN: usize = 256;

/// Q8_0: 32 weights in 34 bytes.
const Q8_0_BLOCK_ELEMS: u64 = 32;
const Q8_0_BLOCK_BYTES: u64 = 34;

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen3EmbedConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub n_vocab: usize,
    pub rope_theta: f32,
    pub eps: f32,
    /// `qwen3.context_length` (the trained context; inputs are capped below it).
    pub n_ctx_train: usize,
}

impl Qwen3EmbedConfig {
    /// Rows of the fused q‖k‖v projection.
    pub fn qkv_rows(&self) -> usize {
        (self.n_head + 2 * self.n_kv_head) * self.head_dim
    }
    pub fn q_width(&self) -> usize {
        self.n_head * self.head_dim
    }
    pub fn kv_width(&self) -> usize {
        self.n_kv_head * self.head_dim
    }
}

/// Where one tensor's bytes live in the GGUF.
#[derive(Clone, Debug)]
pub struct TensorLoc {
    pub name: String,
    pub shard: usize,
    pub offset: u64,
    pub bytes: u64,
    pub dtype: GgufType,
    /// GGUF order: `dims[0]` = row length (K).
    pub dims: Vec<u64>,
}

impl TensorLoc {
    fn from_tensor(t: &GgufTensor) -> Self {
        TensorLoc {
            name: t.name.clone(),
            shard: t.shard,
            offset: t.abs_offset,
            bytes: t.byte_size,
            dtype: t.dtype,
            dims: t.dims.clone(),
        }
    }
}

/// One layer's tensors.
#[derive(Clone, Debug)]
pub struct LayerLoc {
    pub q: TensorLoc,
    pub k: TensorLoc,
    pub v: TensorLoc,
    pub o: TensorLoc,
    pub gate: TensorLoc,
    pub up: TensorLoc,
    pub down: TensorLoc,
    pub attn_norm: TensorLoc,
    pub ffn_norm: TensorLoc,
    pub q_norm: TensorLoc,
    pub k_norm: TensorLoc,
}

/// Byte offsets of one layer inside a weight-ring slot (design §5.2). Q8_0
/// rows are independent, so q‖k‖v and gate‖up sit back to back and each pair
/// (triple) is one GEMM: `qkv` = `q`, with `k` and `v` right after it;
/// `gate_up` = `gate`, with `up` right after it.
#[derive(Clone, Debug, PartialEq)]
pub struct SlotLayout {
    pub q: usize,
    pub k: usize,
    pub v: usize,
    pub o: usize,
    pub gate: usize,
    pub up: usize,
    pub down: usize,
    pub attn_norm: usize,
    pub ffn_norm: usize,
    pub q_norm: usize,
    pub k_norm: usize,
    /// Bytes of each projection (Q8_0) and norm (f32).
    pub q_bytes: usize,
    pub kv_bytes: usize,
    pub o_bytes: usize,
    pub ff_bytes: usize,
    pub down_bytes: usize,
    /// The whole slot, a multiple of [`SLOT_ALIGN`].
    pub bytes: usize,
}

fn align_up(x: usize, a: usize) -> usize {
    x.div_ceil(a) * a
}

fn q8_0_bytes(rows: usize, k: usize) -> usize {
    rows * (k / Q8_0_BLOCK_ELEMS as usize) * Q8_0_BLOCK_BYTES as usize
}

impl SlotLayout {
    pub fn new(cfg: &Qwen3EmbedConfig) -> Self {
        let q_bytes = q8_0_bytes(cfg.q_width(), cfg.n_embd);
        let kv_bytes = q8_0_bytes(cfg.kv_width(), cfg.n_embd);
        let o_bytes = q8_0_bytes(cfg.n_embd, cfg.q_width());
        let ff_bytes = q8_0_bytes(cfg.n_ff, cfg.n_embd);
        let down_bytes = q8_0_bytes(cfg.n_embd, cfg.n_ff);
        // q, k, v contiguous (one GEMM), and gate, up contiguous (one GEMM):
        // no padding inside either group.
        let q = 0;
        let k = q + q_bytes;
        let v = k + kv_bytes;
        let o = align_up(v + kv_bytes, SLOT_ALIGN);
        let gate = align_up(o + o_bytes, SLOT_ALIGN);
        let up = gate + ff_bytes;
        let down = align_up(up + ff_bytes, SLOT_ALIGN);
        let attn_norm = align_up(down + down_bytes, SLOT_ALIGN);
        let ffn_norm = align_up(attn_norm + cfg.n_embd * 4, SLOT_ALIGN);
        let q_norm = align_up(ffn_norm + cfg.n_embd * 4, SLOT_ALIGN);
        let k_norm = align_up(q_norm + cfg.head_dim * 4, SLOT_ALIGN);
        let bytes = align_up(k_norm + cfg.head_dim * 4, SLOT_ALIGN);
        SlotLayout { q, k, v, o, gate, up, down, attn_norm, ffn_norm, q_norm, k_norm, q_bytes, kv_bytes, o_bytes, ff_bytes, down_bytes, bytes }
    }

    /// Every tensor of `layer` with its offset in the slot.
    pub fn placements<'a>(&self, layer: &'a LayerLoc) -> [(&'a TensorLoc, usize); 11] {
        [
            (&layer.q, self.q),
            (&layer.k, self.k),
            (&layer.v, self.v),
            (&layer.o, self.o),
            (&layer.gate, self.gate),
            (&layer.up, self.up),
            (&layer.down, self.down),
            (&layer.attn_norm, self.attn_norm),
            (&layer.ffn_norm, self.ffn_norm),
            (&layer.q_norm, self.q_norm),
            (&layer.k_norm, self.k_norm),
        ]
    }
}

/// A Qwen3-Embedding GGUF, validated, with every tensor located.
pub struct Qwen3EmbedModel {
    pub cfg: Qwen3EmbedConfig,
    pub layers: Vec<LayerLoc>,
    pub token_embd: TensorLoc,
    /// Bytes of one `token_embd` row.
    pub token_row_bytes: usize,
    pub output_norm: Vec<f32>,
    pub slot: SlotLayout,
    /// Id of [`EOS_TEXT`].
    pub eos_id: u32,
}

fn meta_u64(g: &Gguf, key: &str) -> eyre::Result<u64> {
    g.metadata(key).and_then(|v| v.as_u64()).ok_or_else(|| eyre!("qwen3 embed GGUF: missing or non-integer `{key}`"))
}

fn meta_f32(g: &Gguf, key: &str) -> eyre::Result<f32> {
    g.metadata(key).and_then(|v| v.as_f32()).ok_or_else(|| eyre!("qwen3 embed GGUF: missing or non-f32 `{key}`"))
}

fn tensor<'a>(g: &'a Gguf, name: &str) -> eyre::Result<&'a GgufTensor> {
    g.tensor(name).ok_or_else(|| eyre!("qwen3 embed GGUF: tensor `{name}` not found"))
}

/// `name` must be a 2-D Q8_0 matrix `[rows, k]` (GGUF dims `[k, rows]`).
fn q8_matrix(g: &Gguf, name: &str, k: usize, rows: usize) -> eyre::Result<TensorLoc> {
    let t = tensor(g, name)?;
    if t.dtype != GgufType::Q8_0 {
        return Err(eyre!("qwen3 embed GGUF: `{name}` is {}, only Q8_0 is supported (use the Q8_0 GGUF)", t.dtype.name()));
    }
    if t.dims != [k as u64, rows as u64] {
        return Err(eyre!("qwen3 embed GGUF: `{name}` dims {:?}, expected [{k}, {rows}]", t.dims));
    }
    if t.byte_size as usize != q8_0_bytes(rows, k) {
        return Err(eyre!("qwen3 embed GGUF: `{name}` is {} B, expected {}", t.byte_size, q8_0_bytes(rows, k)));
    }
    Ok(TensorLoc::from_tensor(t))
}

/// `name` must be an f32 vector of `n`.
fn f32_vector(g: &Gguf, name: &str, n: usize) -> eyre::Result<TensorLoc> {
    let t = tensor(g, name)?;
    if t.dtype != GgufType::F32 || t.dims != [n as u64] {
        return Err(eyre!("qwen3 embed GGUF: `{name}` is {} {:?}, expected F32 [{n}]", t.dtype.name(), t.dims));
    }
    Ok(TensorLoc::from_tensor(t))
}

impl Qwen3EmbedModel {
    /// Validate `file` as a Q8_0 Qwen3-Embedding GGUF and locate every tensor.
    /// Reads `output_norm` (the only weight kept in host memory).
    pub fn from_gguf(file: &MappedGguf) -> eyre::Result<Self> {
        let g = file.gguf();
        match g.architecture() {
            Some("qwen3") => {}
            other => return Err(eyre!("embed GGUF architecture is {other:?}, expected \"qwen3\"")),
        }
        let n_layer = meta_u64(g, "qwen3.block_count")? as usize;
        let n_embd = meta_u64(g, "qwen3.embedding_length")? as usize;
        let n_ff = meta_u64(g, "qwen3.feed_forward_length")? as usize;
        let n_head = meta_u64(g, "qwen3.attention.head_count")? as usize;
        let n_kv_head = meta_u64(g, "qwen3.attention.head_count_kv")? as usize;
        let n_ctx_train = meta_u64(g, "qwen3.context_length")? as usize;
        let head_dim = match g.metadata("qwen3.attention.key_length").and_then(|v| v.as_u64()) {
            Some(d) => d as usize,
            None => n_embd / n_head.max(1),
        };
        if let Some(vd) = g.metadata("qwen3.attention.value_length").and_then(|v| v.as_u64()) {
            if vd as usize != head_dim {
                return Err(eyre!("qwen3 embed GGUF: value_length {vd} != key_length {head_dim}"));
            }
        }
        let rope_theta = meta_f32(g, "qwen3.rope.freq_base")?;
        let eps = meta_f32(g, "qwen3.attention.layer_norm_rms_epsilon")?;
        if n_layer == 0 || n_head == 0 || n_kv_head == 0 || n_head % n_kv_head != 0 {
            return Err(eyre!("qwen3 embed GGUF: bad head counts n_head={n_head} n_kv_head={n_kv_head} n_layer={n_layer}"));
        }
        if head_dim % 2 != 0 || n_embd % Q8_0_BLOCK_ELEMS as usize != 0 || n_ff % Q8_0_BLOCK_ELEMS as usize != 0 {
            return Err(eyre!("qwen3 embed GGUF: head_dim {head_dim}, n_embd {n_embd}, n_ff {n_ff} must be even / multiples of 32"));
        }
        let te = tensor(g, "token_embd.weight")?;
        if te.dims.len() != 2 || te.dims[0] != n_embd as u64 {
            return Err(eyre!("qwen3 embed GGUF: token_embd dims {:?}, expected [{n_embd}, vocab]", te.dims));
        }
        let (block_elems, block_bytes) = te
            .dtype
            .block_shape()
            .ok_or_else(|| eyre!("qwen3 embed GGUF: token_embd dtype {} has no block shape", te.dtype.name()))?;
        if n_embd as u64 % block_elems as u64 != 0 {
            return Err(eyre!("qwen3 embed GGUF: n_embd {n_embd} is not whole {} blocks", te.dtype.name()));
        }
        let n_vocab = te.dims[1] as usize;
        let token_row_bytes = n_embd / block_elems as usize * block_bytes as usize;
        let token_embd = TensorLoc::from_tensor(te);

        let cfg = Qwen3EmbedConfig { n_layer, n_embd, n_ff, n_head, n_kv_head, head_dim, n_vocab, rope_theta, eps, n_ctx_train };
        let mut layers = Vec::with_capacity(n_layer);
        for il in 0..n_layer {
            let n = |s: &str| format!("blk.{il}.{s}.weight");
            layers.push(LayerLoc {
                q: q8_matrix(g, &n("attn_q"), n_embd, cfg.q_width())?,
                k: q8_matrix(g, &n("attn_k"), n_embd, cfg.kv_width())?,
                v: q8_matrix(g, &n("attn_v"), n_embd, cfg.kv_width())?,
                o: q8_matrix(g, &n("attn_output"), cfg.q_width(), n_embd)?,
                gate: q8_matrix(g, &n("ffn_gate"), n_embd, n_ff)?,
                up: q8_matrix(g, &n("ffn_up"), n_embd, n_ff)?,
                down: q8_matrix(g, &n("ffn_down"), n_ff, n_embd)?,
                attn_norm: f32_vector(g, &n("attn_norm"), n_embd)?,
                ffn_norm: f32_vector(g, &n("ffn_norm"), n_embd)?,
                q_norm: f32_vector(g, &n("attn_q_norm"), head_dim)?,
                k_norm: f32_vector(g, &n("attn_k_norm"), head_dim)?,
            });
        }
        let on = f32_vector(g, "output_norm.weight", n_embd)?;
        let mut bytes = vec![0u8; on.bytes as usize];
        file.read_range_into(on.shard, on.offset, &mut bytes)?;
        let output_norm = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

        let vocab = crate::tokenizer::BpeVocab::from_gguf(g)?;
        let eos_id = vocab
            .lookup_token_id(EOS_TEXT)
            .ok_or_else(|| eyre!("qwen3 embed GGUF: vocab has no `{EOS_TEXT}`"))? as u32;
        let slot = SlotLayout::new(&cfg);
        Ok(Qwen3EmbedModel { cfg, layers, token_embd, token_row_bytes, output_norm, slot, eos_id })
    }

    /// Fill `dst` (at least `slot.bytes`) with layer `il`'s tensors at their
    /// [`SlotLayout`] offsets. Each tensor is one pread (page cache dropped
    /// after, `MappedGguf::read_range_into`).
    pub fn read_layer_into(&self, file: &MappedGguf, il: usize, dst: &mut [u8]) -> eyre::Result<()> {
        if dst.len() < self.slot.bytes {
            return Err(eyre!("read_layer_into: slot buffer {} B < {}", dst.len(), self.slot.bytes));
        }
        for (t, off) in self.slot.placements(&self.layers[il]) {
            file.read_range_into(t.shard, t.offset, &mut dst[off..off + t.bytes as usize])?;
        }
        Ok(())
    }

    /// Token-embedding rows of `ids`, dequantized: `[ids.len(), n_embd]` f32.
    /// Each distinct id is read once; runs of adjacent ids share one pread.
    pub fn token_rows(&self, file: &MappedGguf, ids: &[u32]) -> eyre::Result<Vec<f32>> {
        let n_embd = self.cfg.n_embd;
        let mut uniq: Vec<u32> = ids.to_vec();
        uniq.sort_unstable();
        uniq.dedup();
        if let Some(&bad) = uniq.last().filter(|&&id| id as usize >= self.cfg.n_vocab) {
            return Err(eyre!("token id {bad} out of range (vocab {})", self.cfg.n_vocab));
        }
        let rb = self.token_row_bytes;
        let mut rows: HashMap<u32, usize> = HashMap::with_capacity(uniq.len());
        let mut deq: Vec<f32> = Vec::with_capacity(uniq.len() * n_embd);
        let mut buf: Vec<u8> = Vec::new();
        let mut i = 0;
        while i < uniq.len() {
            let mut j = i + 1;
            while j < uniq.len() && uniq[j] == uniq[j - 1] + 1 {
                j += 1;
            }
            buf.resize((j - i) * rb, 0);
            let off = self.token_embd.offset + uniq[i] as u64 * rb as u64;
            file.read_range_into(self.token_embd.shard, off, &mut buf)?;
            for (k, id) in uniq[i..j].iter().enumerate() {
                rows.insert(*id, deq.len() / n_embd);
                dequant_to_f32(self.token_embd.dtype, &buf[k * rb..(k + 1) * rb], &mut deq)?;
            }
            i = j;
        }
        let mut out = Vec::with_capacity(ids.len() * n_embd);
        for id in ids {
            let r = rows[id];
            out.extend_from_slice(&deq[r * n_embd..(r + 1) * n_embd]);
        }
        Ok(out)
    }

    /// The embedding of one input from its last hidden row (before the final
    /// norm): `output_norm`, L2, and MRL truncation (`dims`) + L2 again.
    pub fn finish(&self, last: &[f32], dims: Option<usize>) -> Vec<f32> {
        finish_embedding(last, &self.output_norm, self.cfg.eps, dims)
    }
}

/// RMSNorm ⊙ `w` into `out`.
pub fn rms_norm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let ms = x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len() as f64;
    let s = (1.0 / (ms + eps as f64).sqrt()) as f32;
    for ((o, v), g) in out.iter_mut().zip(x).zip(w) {
        *o = v * s * g;
    }
}

fn l2_normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    if n > 0.0 {
        let inv = (1.0 / n) as f32;
        v.iter_mut().for_each(|x| *x *= inv);
    }
}

/// `output_norm` then L2; with `dims = Some(d)`, truncate to `d` and L2 again
/// (Matryoshka). Truncating a unit vector and renormalizing equals
/// renormalizing the truncated raw vector, so one L2 after truncation would do;
/// two keep the full-dim result exactly the unit vector the reference returns.
pub fn finish_embedding(last: &[f32], output_norm: &[f32], eps: f32, dims: Option<usize>) -> Vec<f32> {
    let mut v = vec![0f32; last.len()];
    rms_norm(last, output_norm, eps, &mut v);
    l2_normalize(&mut v);
    if let Some(d) = dims.filter(|&d| d < v.len()) {
        v.truncate(d);
        l2_normalize(&mut v);
    }
    v
}

/// Cosine similarity (gates).
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        ab += *x as f64 * *y as f64;
        aa += *x as f64 * *x as f64;
        bb += *y as f64 * *y as f64;
    }
    ab / (aa.sqrt() * bb.sqrt()).max(f64::MIN_POSITIVE)
}

// ---- CPU reference forward ------------------------------------------------

/// Dot product with 8 independent accumulators (vectorizes; f32 sums in a
/// fixed order, so the result is deterministic).
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let n8 = a.len() / 8 * 8;
    for (ca, cb) in a[..n8].chunks_exact(8).zip(b[..n8].chunks_exact(8)) {
        for l in 0..8 {
            acc[l] += ca[l] * cb[l];
        }
    }
    let mut s = acc.iter().sum::<f32>();
    for i in n8..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// `y[t, r] = x[t, :] · w[r, :]` for `w` = `[rows, k]` f32, threaded over rows.
fn matmul(x: &[f32], t: usize, w: &[f32], rows: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0f32; t * rows];
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(rows.max(1));
    let per = rows.div_ceil(threads);
    // Each thread owns output columns [r0, r1); write through a per-thread
    // buffer and scatter after, so no two threads share a slice.
    let parts: Vec<(usize, Vec<f32>)> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|ti| {
                let (r0, r1) = (ti * per, ((ti + 1) * per).min(rows));
                s.spawn(move || {
                    let mut part = vec![0f32; t * r1.saturating_sub(r0)];
                    for ti_ in 0..t {
                        let xr = &x[ti_ * k..(ti_ + 1) * k];
                        for r in r0..r1 {
                            part[ti_ * (r1 - r0) + (r - r0)] = dot(xr, &w[r * k..(r + 1) * k]);
                        }
                    }
                    (r0, part)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("matmul thread")).collect()
    });
    for (r0, part) in parts {
        let n = if t == 0 { 0 } else { part.len() / t };
        for ti_ in 0..t {
            y[ti_ * rows + r0..ti_ * rows + r0 + n].copy_from_slice(&part[ti_ * n..(ti_ + 1) * n]);
        }
    }
    y
}

/// NeoX RoPE on one head in place (pairs `i`, `i + d/2`), the HF numerics:
/// `inv_freq[i] = 1 / theta^(2i/d)`, `angle = pos * inv_freq[i]`, in f32.
pub fn rope_neox(x: &mut [f32], pos: usize, theta: f32) {
    let d = x.len();
    let half = d / 2;
    for i in 0..half {
        let inv_freq = 1.0f32 / theta.powf((2 * i) as f32 / d as f32);
        let a = pos as f32 * inv_freq;
        let (s, c) = a.sin_cos();
        let (x0, x1) = (x[i], x[i + half]);
        x[i] = x0 * c - x1 * s;
        x[i + half] = x1 * c + x0 * s;
    }
}

fn read_f32s(file: &MappedGguf, t: &TensorLoc) -> eyre::Result<Vec<f32>> {
    let mut bytes = vec![0u8; t.bytes as usize];
    file.read_range_into(t.shard, t.offset, &mut bytes)?;
    let mut out = Vec::with_capacity(t.bytes as usize / 2);
    dequant_to_f32(t.dtype, &bytes, &mut out)?;
    Ok(out)
}

/// The CPU reference forward: f32 everywhere, Q8_0 weights dequantized one
/// layer at a time. `inputs` are token ids (EOS included). Returns each
/// input's full-dimension embedding (`finish` with `dims = None`).
pub fn cpu_forward(model: &Qwen3EmbedModel, file: &MappedGguf, inputs: &[Vec<u32>]) -> eyre::Result<Vec<Vec<f32>>> {
    let c = &model.cfg;
    let (d, hd) = (c.n_embd, c.head_dim);
    let ids: Vec<u32> = inputs.iter().flatten().copied().collect();
    let t = ids.len();
    let mut starts = Vec::with_capacity(inputs.len());
    let mut acc = 0;
    for inp in inputs {
        if inp.is_empty() {
            return Err(eyre!("cpu_forward: empty input"));
        }
        starts.push(acc);
        acc += inp.len();
    }
    let mut resid = model.token_rows(file, &ids)?;
    let mut xn = vec![0f32; t * d];
    let scale = 1.0 / (hd as f32).sqrt();
    let group = c.n_head / c.n_kv_head;
    for layer in &model.layers {
        let attn_norm = read_f32s(file, &layer.attn_norm)?;
        let ffn_norm = read_f32s(file, &layer.ffn_norm)?;
        let q_norm = read_f32s(file, &layer.q_norm)?;
        let k_norm = read_f32s(file, &layer.k_norm)?;
        for r in 0..t {
            rms_norm(&resid[r * d..(r + 1) * d], &attn_norm, c.eps, &mut xn[r * d..(r + 1) * d]);
        }
        let mut q = matmul(&xn, t, &read_f32s(file, &layer.q)?, c.q_width(), d);
        let mut k = matmul(&xn, t, &read_f32s(file, &layer.k)?, c.kv_width(), d);
        let v = matmul(&xn, t, &read_f32s(file, &layer.v)?, c.kv_width(), d);
        for (s, inp) in starts.iter().zip(inputs) {
            for p in 0..inp.len() {
                let r = s + p;
                for h in 0..c.n_head {
                    let x = &mut q[r * c.q_width() + h * hd..r * c.q_width() + (h + 1) * hd];
                    let tmp = x.to_vec();
                    rms_norm(&tmp, &q_norm, c.eps, x);
                    rope_neox(x, p, c.rope_theta);
                }
                for h in 0..c.n_kv_head {
                    let x = &mut k[r * c.kv_width() + h * hd..r * c.kv_width() + (h + 1) * hd];
                    let tmp = x.to_vec();
                    rms_norm(&tmp, &k_norm, c.eps, x);
                    rope_neox(x, p, c.rope_theta);
                }
            }
        }
        // Causal GQA attention per input.
        let mut att = vec![0f32; t * c.q_width()];
        for (s, inp) in starts.iter().zip(inputs) {
            for p in 0..inp.len() {
                let r = s + p;
                for h in 0..c.n_head {
                    let kh = h / group;
                    let qv = &q[r * c.q_width() + h * hd..r * c.q_width() + (h + 1) * hd];
                    let mut sc: Vec<f32> = (0..=p)
                        .map(|j| dot(qv, &k[(s + j) * c.kv_width() + kh * hd..(s + j) * c.kv_width() + (kh + 1) * hd]) * scale)
                        .collect();
                    let m = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0f32;
                    sc.iter_mut().for_each(|x| {
                        *x = (*x - m).exp();
                        sum += *x;
                    });
                    let out = &mut att[r * c.q_width() + h * hd..r * c.q_width() + (h + 1) * hd];
                    for (j, w) in sc.iter().enumerate() {
                        let vv = &v[(s + j) * c.kv_width() + kh * hd..(s + j) * c.kv_width() + (kh + 1) * hd];
                        for e in 0..hd {
                            out[e] += w / sum * vv[e];
                        }
                    }
                }
            }
        }
        let o = matmul(&att, t, &read_f32s(file, &layer.o)?, d, c.q_width());
        resid.iter_mut().zip(&o).for_each(|(a, b)| *a += b);
        for r in 0..t {
            rms_norm(&resid[r * d..(r + 1) * d], &ffn_norm, c.eps, &mut xn[r * d..(r + 1) * d]);
        }
        let g = matmul(&xn, t, &read_f32s(file, &layer.gate)?, c.n_ff, d);
        let u = matmul(&xn, t, &read_f32s(file, &layer.up)?, c.n_ff, d);
        let h: Vec<f32> = g.iter().zip(&u).map(|(g, u)| g / (1.0 + (-g).exp()) * u).collect();
        let dn = matmul(&h, t, &read_f32s(file, &layer.down)?, d, c.n_ff);
        resid.iter_mut().zip(&dn).for_each(|(a, b)| *a += b);
    }
    Ok(starts
        .iter()
        .zip(inputs)
        .map(|(s, inp)| {
            let r = s + inp.len() - 1;
            model.finish(&resid[r * d..(r + 1) * d], None)
        })
        .collect())
}

// ---- synthetic models (tests) --------------------------------------------

/// Tiny Qwen3-shaped GGUFs with random weights, for host tests and the GPU
/// path's CPU-vs-GPU tests (no real weights needed).
pub mod testing {
    use std::io::BufWriter;
    use std::path::Path;

    use color_eyre::eyre;

    use super::Qwen3EmbedConfig;
    use crate::gguf::{GgufArray, GgufType, GgufValue};
    use crate::gguf_write::{GgufWriter, TensorSpec};
    use crate::kquants::f32_to_f16_bits;

    /// A shape every GPU kernel accepts (GEMM m % 128, k % 32; head_dim 128).
    pub fn tiny_config() -> Qwen3EmbedConfig {
        Qwen3EmbedConfig {
            n_layer: 2,
            n_embd: 256,
            n_ff: 384,
            n_head: 4,
            n_kv_head: 2,
            head_dim: 128,
            n_vocab: 257,
            rope_theta: 1_000_000.0,
            eps: 1e-6,
            n_ctx_train: 4096,
        }
    }

    /// xorshift64*: deterministic, no dependency.
    struct Rng(u64);
    impl Rng {
        fn next_f32(&mut self) -> f32 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            let v = self.0.wrapping_mul(0x2545_f491_4f6c_dd1d);
            ((v >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    /// Quantize `x` (whole 32-blocks) to Q8_0, as llama.cpp's
    /// `quantize_row_q8_0_ref`.
    pub fn quantize_q8_0(x: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(x.len() / 32 * 34);
        for blk in x.chunks_exact(32) {
            let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
            for v in blk {
                out.push((v * id).round() as i8 as u8);
            }
        }
        out
    }

    /// Write a random-weight Qwen3-Embedding GGUF of shape `cfg`. The vocab is
    /// the 256 GPT-2 byte symbols plus `<|endoftext|>` (no merges).
    pub fn write_synthetic(path: &Path, cfg: &Qwen3EmbedConfig, seed: u64) -> eyre::Result<()> {
        let mut rng = Rng(seed | 1);
        let (d, ff, hd) = (cfg.n_embd as u64, cfg.n_ff as u64, cfg.head_dim as u64);
        let qw = (cfg.n_head * cfg.head_dim) as u64;
        let kvw = (cfg.n_kv_head * cfg.head_dim) as u64;
        let mut specs: Vec<TensorSpec> = Vec::new();
        let mut data: Vec<Vec<u8>> = Vec::new();
        let push_q8 = |specs: &mut Vec<TensorSpec>, data: &mut Vec<Vec<u8>>, rng: &mut Rng, name: String, k: u64, rows: u64, amp: f32| {
            let w: Vec<f32> = (0..k * rows).map(|_| rng.next_f32() * amp).collect();
            specs.push(TensorSpec { name, dims: vec![k, rows], dtype: GgufType::Q8_0 });
            data.push(quantize_q8_0(&w));
        };
        let push_f32 = |specs: &mut Vec<TensorSpec>, data: &mut Vec<Vec<u8>>, rng: &mut Rng, name: String, n: u64| {
            let w: Vec<u8> = (0..n).flat_map(|_| (1.0 + 0.1 * rng.next_f32()).to_le_bytes()).collect();
            specs.push(TensorSpec { name, dims: vec![n], dtype: GgufType::F32 });
            data.push(w);
        };
        push_q8(&mut specs, &mut data, &mut rng, "token_embd.weight".into(), d, cfg.n_vocab as u64, 1.0);
        for il in 0..cfg.n_layer {
            let n = |s: &str| format!("blk.{il}.{s}.weight");
            push_f32(&mut specs, &mut data, &mut rng, n("attn_norm"), d);
            push_q8(&mut specs, &mut data, &mut rng, n("attn_q"), d, qw, 0.08);
            push_q8(&mut specs, &mut data, &mut rng, n("attn_k"), d, kvw, 0.08);
            push_q8(&mut specs, &mut data, &mut rng, n("attn_v"), d, kvw, 0.08);
            push_f32(&mut specs, &mut data, &mut rng, n("attn_q_norm"), hd);
            push_f32(&mut specs, &mut data, &mut rng, n("attn_k_norm"), hd);
            push_q8(&mut specs, &mut data, &mut rng, n("attn_output"), qw, d, 0.05);
            push_f32(&mut specs, &mut data, &mut rng, n("ffn_norm"), d);
            push_q8(&mut specs, &mut data, &mut rng, n("ffn_gate"), d, ff, 0.08);
            push_q8(&mut specs, &mut data, &mut rng, n("ffn_up"), d, ff, 0.08);
            push_q8(&mut specs, &mut data, &mut rng, n("ffn_down"), ff, d, 0.05);
        }
        push_f32(&mut specs, &mut data, &mut rng, "output_norm.weight".into(), d);

        let mut tokens: Vec<String> = (0..=255u8).map(|b| String::from_utf8(crate::tokenizer::byte_encode(&[b])).expect("byte symbol")).collect();
        tokens.push(super::EOS_TEXT.to_string());
        assert_eq!(tokens.len(), cfg.n_vocab, "tiny vocab must match n_vocab");
        let kv: Vec<(&str, GgufValue)> = vec![
            ("general.architecture", GgufValue::String("qwen3".into())),
            ("qwen3.block_count", GgufValue::U32(cfg.n_layer as u32)),
            ("qwen3.context_length", GgufValue::U32(cfg.n_ctx_train as u32)),
            ("qwen3.embedding_length", GgufValue::U32(cfg.n_embd as u32)),
            ("qwen3.feed_forward_length", GgufValue::U32(cfg.n_ff as u32)),
            ("qwen3.attention.head_count", GgufValue::U32(cfg.n_head as u32)),
            ("qwen3.attention.head_count_kv", GgufValue::U32(cfg.n_kv_head as u32)),
            ("qwen3.attention.key_length", GgufValue::U32(cfg.head_dim as u32)),
            ("qwen3.attention.value_length", GgufValue::U32(cfg.head_dim as u32)),
            ("qwen3.rope.freq_base", GgufValue::F32(cfg.rope_theta)),
            ("qwen3.attention.layer_norm_rms_epsilon", GgufValue::F32(cfg.eps)),
            ("tokenizer.ggml.model", GgufValue::String("gpt2".into())),
            ("tokenizer.ggml.pre", GgufValue::String("qwen2".into())),
            ("tokenizer.ggml.tokens", GgufValue::Array(GgufArray::String(tokens))),
            ("tokenizer.ggml.merges", GgufValue::Array(GgufArray::String(Vec::new()))),
            ("tokenizer.ggml.eos_token_id", GgufValue::U32(256)),
            ("tokenizer.ggml.add_bos_token", GgufValue::Bool(false)),
        ];
        let kv_refs: Vec<(&str, &GgufValue)> = kv.iter().map(|(k, v)| (*k, v)).collect();
        let f = std::fs::File::create(path)?;
        let mut w = GgufWriter::new(BufWriter::new(f), &kv_refs, &specs, 32)?;
        for chunk in &data {
            w.write_tensor_chunk(chunk)?;
        }
        w.finish()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny(dir: &std::path::Path, seed: u64) -> (MappedGguf, Qwen3EmbedModel) {
        let p = dir.join(format!("tiny-{seed}.gguf"));
        testing::write_synthetic(&p, &testing::tiny_config(), seed).expect("write");
        let f = MappedGguf::open(&p).expect("open");
        let m = Qwen3EmbedModel::from_gguf(&f).expect("model");
        (f, m)
    }

    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("qwen3-embed-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn synthetic_model_parses_and_lays_out() {
        let dir = tmpdir("parse");
        let (_f, m) = tiny(&dir, 7);
        assert_eq!(m.cfg, testing::tiny_config());
        assert_eq!(m.eos_id, 256);
        assert_eq!(m.token_row_bytes, 256 / 32 * 34);
        let s = &m.slot;
        assert_eq!(s.k, s.q + s.q_bytes);
        assert_eq!(s.v, s.k + s.kv_bytes);
        assert_eq!(s.up, s.gate + s.ff_bytes);
        assert_eq!(s.bytes % SLOT_ALIGN, 0);
        // Placements never overlap.
        let mut spans: Vec<(usize, usize)> = s.placements(&m.layers[0]).iter().map(|(t, o)| (*o, o + t.bytes as usize)).collect();
        spans.sort();
        for w in spans.windows(2) {
            assert!(w[0].1 <= w[1].0, "{spans:?}");
        }
        assert!(spans.last().unwrap().1 <= s.bytes);
    }

    #[test]
    fn real_model_slot_size_matches_the_design() {
        let cfg = Qwen3EmbedConfig {
            n_layer: 36, n_embd: 2560, n_ff: 9728, n_head: 32, n_kv_head: 8, head_dim: 128,
            n_vocab: 151_665, rope_theta: 1e6, eps: 1e-6, n_ctx_train: 40960,
        };
        let s = SlotLayout::new(&cfg);
        assert_eq!(s.q_bytes, 11_141_120);
        assert_eq!(s.kv_bytes, 2_785_280);
        assert_eq!(s.o_bytes, 11_141_120);
        assert_eq!(s.ff_bytes, 26_460_160);
        assert_eq!(s.down_bytes, 26_460_160);
        // 107,254,784 B of tensors (design §2) plus alignment padding.
        assert!(s.bytes >= 107_254_784 && s.bytes < 107_254_784 + 8 * SLOT_ALIGN, "{}", s.bytes);
    }

    #[test]
    fn layer_read_places_every_tensor() {
        let dir = tmpdir("read");
        let (f, m) = tiny(&dir, 11);
        let mut slot = vec![0u8; m.slot.bytes];
        m.read_layer_into(&f, 1, &mut slot).unwrap();
        for (t, off) in m.slot.placements(&m.layers[1]) {
            let mut want = vec![0u8; t.bytes as usize];
            f.read_range_into(t.shard, t.offset, &mut want).unwrap();
            assert_eq!(&slot[off..off + want.len()], &want[..], "{}", t.name);
        }
    }

    #[test]
    fn token_rows_match_the_table() {
        let dir = tmpdir("rows");
        let (f, m) = tiny(&dir, 13);
        let ids = [5u32, 6, 7, 5, 256, 0, 7];
        let rows = m.token_rows(&f, &ids).unwrap();
        let mut table = vec![0u8; m.token_embd.bytes as usize];
        f.read_range_into(m.token_embd.shard, m.token_embd.offset, &mut table).unwrap();
        let mut all = Vec::new();
        dequant_to_f32(GgufType::Q8_0, &table, &mut all).unwrap();
        let d = m.cfg.n_embd;
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(&rows[i * d..(i + 1) * d], &all[*id as usize * d..(*id as usize + 1) * d]);
        }
        assert!(m.token_rows(&f, &[257]).is_err());
    }

    #[test]
    fn finish_normalizes_and_truncates() {
        let w = vec![1.0f32; 8];
        let v = finish_embedding(&[3.0, 4.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], &w, 1e-6, None);
        assert!((v.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 1e-6);
        let t = finish_embedding(&[3.0, 4.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], &w, 1e-6, Some(2));
        assert_eq!(t.len(), 2);
        assert!((t[0] - 0.6).abs() < 1e-6 && (t[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn rope_neox_rotates_half_pairs() {
        // pos 0 is the identity; the rotation preserves each pair's norm.
        let mut x: Vec<f32> = (0..8).map(|i| i as f32 + 1.0).collect();
        let x0 = x.clone();
        rope_neox(&mut x, 0, 1e6);
        assert_eq!(x, x0);
        rope_neox(&mut x, 5, 10.0);
        for i in 0..4 {
            let a = x0[i] * x0[i] + x0[i + 4] * x0[i + 4];
            let b = x[i] * x[i] + x[i + 4] * x[i + 4];
            assert!((a - b).abs() < 1e-4);
        }
        // Pair 0 rotates by exactly `pos` radians (inv_freq[0] = 1).
        let mut y = vec![1.0f32, 0.0, 0.0, 0.0];
        rope_neox(&mut y, 1, 10.0);
        assert!((y[0] - 1f32.cos()).abs() < 1e-6 && (y[2] - 1f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn cpu_forward_is_causal_and_packing_invariant() {
        let dir = tmpdir("fwd");
        let (f, m) = tiny(&dir, 17);
        let a: Vec<u32> = vec![10, 20, 30, 40, 256];
        let b: Vec<u32> = vec![99, 98, 256];
        let alone_a = cpu_forward(&m, &f, std::slice::from_ref(&a)).unwrap();
        let alone_b = cpu_forward(&m, &f, std::slice::from_ref(&b)).unwrap();
        let packed = cpu_forward(&m, &f, &[a.clone(), b.clone()]).unwrap();
        assert!(cosine(&alone_a[0], &packed[0]) > 0.999_999);
        assert!(cosine(&alone_b[0], &packed[1]) > 0.999_999);
        assert!((packed[0].iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 1e-5);
        // A different prefix changes the embedding (the model is not degenerate).
        let c: Vec<u32> = vec![11, 20, 30, 40, 256];
        let alone_c = cpu_forward(&m, &f, &[c]).unwrap();
        assert!(cosine(&alone_a[0], &alone_c[0]) < 0.9999);
    }
}
