//! GGUF v3 writer — the inverse of [`crate::gguf`]'s parser, used by
//! `gguf-dequant-dense` to emit single-file "dumper variant" models.
//!
//! Layout written (little-endian throughout, mirroring the parser):
//!   header (magic, version=3, n_tensors, n_kv)
//!   KV section (each: string key, u32 type id, value)
//!   tensor directory (each: string name, u32 ndim, u64 dims, u32 type id,
//!                     u64 rel_offset)
//!   pad to `alignment`
//!   tensor data, each tensor's rel_offset aligned to `alignment`
//!
//! The writer is two-phase: [`GgufWriter::new`] takes the full KV list and
//! tensor directory up front (sizes must be known — they follow from dtype
//! and dims), writes everything up to the data section, and precomputes
//! each tensor's offset. The caller then streams payloads in directory
//! order via [`GgufWriter::write_tensor_chunk`] / finishes with
//! [`GgufWriter::finish`].

use std::io::{self, Seek, Write};

use color_eyre::eyre::{self, eyre};

use crate::gguf::{GGUF_MAGIC, GgufArray, GgufType, GgufValue};

/// One tensor directory entry to be written.
#[derive(Debug, Clone)]
pub struct TensorSpec {
    pub name: String,
    /// Dims in GGUF order (dim[0] = innermost / row length).
    pub dims: Vec<u64>,
    pub dtype: GgufType,
}

impl TensorSpec {
    pub fn elements(&self) -> u64 {
        self.dims.iter().filter(|&&d| d != 0).product()
    }

    pub fn byte_size(&self) -> eyre::Result<u64> {
        self.dtype
            .size_of(self.elements())
            .map_err(|e| eyre!("{}: {e}", self.name))
    }
}

pub struct GgufWriter<W: Write + Seek> {
    w: W,
    alignment: u64,
    /// (byte_size, rel_offset) per tensor, in directory order.
    sizes: Vec<(u64, u64)>,
    /// Index of the tensor currently being streamed.
    cur: usize,
    /// Bytes of the current tensor written so far.
    cur_written: u64,
    /// Absolute file position where tensor data begins.
    data_start: u64,
}

impl<W: Write + Seek> GgufWriter<W> {
    /// Write header + KV section + tensor directory. `kvs` are written in
    /// the given order; tensor rel_offsets are assigned in `tensors` order,
    /// each aligned to `alignment`.
    pub fn new(
        mut w: W,
        kvs: &[(&str, &GgufValue)],
        tensors: &[TensorSpec],
        alignment: u64,
    ) -> eyre::Result<Self> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(eyre!("alignment {alignment} must be a power of two"));
        }

        // Assign offsets.
        let mut sizes = Vec::with_capacity(tensors.len());
        let mut running: u64 = 0;
        for t in tensors {
            running = align_up(running, alignment);
            let bytes = t.byte_size()?;
            sizes.push((bytes, running));
            running = running
                .checked_add(bytes)
                .ok_or_else(|| eyre!("tensor data overflows u64"))?;
        }

        w.write_all(&GGUF_MAGIC.to_le_bytes())?;
        w.write_all(&3u32.to_le_bytes())?;
        w.write_all(&(tensors.len() as u64).to_le_bytes())?;
        w.write_all(&(kvs.len() as u64).to_le_bytes())?;

        for (key, value) in kvs {
            write_string(&mut w, key)?;
            w.write_all(&value.type_id().to_le_bytes())?;
            write_value(&mut w, value)?;
        }

        for (t, (_, rel)) in tensors.iter().zip(&sizes) {
            write_string(&mut w, &t.name)?;
            w.write_all(&(t.dims.len() as u32).to_le_bytes())?;
            for d in &t.dims {
                w.write_all(&d.to_le_bytes())?;
            }
            w.write_all(&t.dtype.id().to_le_bytes())?;
            w.write_all(&rel.to_le_bytes())?;
        }

        // Pad to the data-section alignment boundary.
        let pos = w.stream_position()?;
        let data_start = align_up(pos, alignment);
        write_zeros(&mut w, data_start - pos)?;

        Ok(GgufWriter {
            w,
            alignment,
            sizes,
            cur: 0,
            cur_written: 0,
            data_start,
        })
    }

    /// Stream a chunk of the current tensor's payload. Tensors must be
    /// written in directory order; a tensor is complete once exactly its
    /// byte_size has been streamed, after which the writer pads to the next
    /// tensor's aligned offset and advances.
    pub fn write_tensor_chunk(&mut self, chunk: &[u8]) -> eyre::Result<()> {
        let (bytes, _) = *self
            .sizes
            .get(self.cur)
            .ok_or_else(|| eyre!("write past the last tensor"))?;
        let remaining = bytes - self.cur_written;
        if (chunk.len() as u64) > remaining {
            return Err(eyre!(
                "tensor {} of {}: chunk of {} exceeds remaining {} bytes",
                self.cur,
                self.sizes.len(),
                chunk.len(),
                remaining
            ));
        }
        self.w.write_all(chunk)?;
        self.cur_written += chunk.len() as u64;
        if self.cur_written == bytes {
            self.cur += 1;
            self.cur_written = 0;
            if let Some(&(_, next_rel)) = self.sizes.get(self.cur) {
                let pos = self.w.stream_position()?;
                let want = self.data_start + next_rel;
                debug_assert!(want >= pos);
                write_zeros(&mut self.w, want - pos)?;
            }
        }
        Ok(())
    }

    /// Total padded byte size of tensor `index` (for progress reporting).
    pub fn tensor_bytes(&self, index: usize) -> u64 {
        self.sizes[index].0
    }

    /// Verify every tensor was fully streamed and flush.
    pub fn finish(mut self) -> eyre::Result<W> {
        if self.cur != self.sizes.len() || self.cur_written != 0 {
            return Err(eyre!(
                "finish: only {} of {} tensors fully written",
                self.cur,
                self.sizes.len()
            ));
        }
        self.w.flush()?;
        Ok(self.w)
    }

    pub fn alignment(&self) -> u64 {
        self.alignment
    }
}

fn align_up(n: u64, alignment: u64) -> u64 {
    let rem = n % alignment;
    if rem == 0 { n } else { n + (alignment - rem) }
}

fn write_zeros<W: Write>(w: &mut W, mut n: u64) -> io::Result<()> {
    const Z: [u8; 4096] = [0u8; 4096];
    while n > 0 {
        let take = n.min(Z.len() as u64) as usize;
        w.write_all(&Z[..take])?;
        n -= take as u64;
    }
    Ok(())
}

fn write_string<W: Write>(w: &mut W, s: &str) -> io::Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())
}

fn write_value<W: Write>(w: &mut W, v: &GgufValue) -> io::Result<()> {
    match v {
        GgufValue::U8(x) => w.write_all(&[*x]),
        GgufValue::I8(x) => w.write_all(&[*x as u8]),
        GgufValue::U16(x) => w.write_all(&x.to_le_bytes()),
        GgufValue::I16(x) => w.write_all(&x.to_le_bytes()),
        GgufValue::U32(x) => w.write_all(&x.to_le_bytes()),
        GgufValue::I32(x) => w.write_all(&x.to_le_bytes()),
        GgufValue::F32(x) => w.write_all(&x.to_le_bytes()),
        GgufValue::Bool(x) => w.write_all(&[*x as u8]),
        GgufValue::String(s) => write_string(w, s),
        GgufValue::Array(a) => write_array(w, a),
        GgufValue::U64(x) => w.write_all(&x.to_le_bytes()),
        GgufValue::I64(x) => w.write_all(&x.to_le_bytes()),
        GgufValue::F64(x) => w.write_all(&x.to_le_bytes()),
    }
}

fn write_array<W: Write>(w: &mut W, a: &GgufArray) -> io::Result<()> {
    // item type id, u64 len, items
    let (type_id, len) = match a {
        GgufArray::U8(v) => (0u32, v.len()),
        GgufArray::I8(v) => (1, v.len()),
        GgufArray::U16(v) => (2, v.len()),
        GgufArray::I16(v) => (3, v.len()),
        GgufArray::U32(v) => (4, v.len()),
        GgufArray::I32(v) => (5, v.len()),
        GgufArray::F32(v) => (6, v.len()),
        GgufArray::Bool(v) => (7, v.len()),
        GgufArray::String(v) => (8, v.len()),
        GgufArray::U64(v) => (10, v.len()),
        GgufArray::I64(v) => (11, v.len()),
        GgufArray::F64(v) => (12, v.len()),
        GgufArray::Nested(v) => (9, v.len()),
    };
    w.write_all(&type_id.to_le_bytes())?;
    w.write_all(&(len as u64).to_le_bytes())?;
    match a {
        GgufArray::U8(v) => w.write_all(v)?,
        GgufArray::I8(v) => {
            for x in v {
                w.write_all(&[*x as u8])?;
            }
        }
        GgufArray::U16(v) => {
            for x in v {
                w.write_all(&x.to_le_bytes())?;
            }
        }
        GgufArray::I16(v) => {
            for x in v {
                w.write_all(&x.to_le_bytes())?;
            }
        }
        GgufArray::U32(v) => {
            for x in v {
                w.write_all(&x.to_le_bytes())?;
            }
        }
        GgufArray::I32(v) => {
            for x in v {
                w.write_all(&x.to_le_bytes())?;
            }
        }
        GgufArray::F32(v) => {
            for x in v {
                w.write_all(&x.to_le_bytes())?;
            }
        }
        GgufArray::Bool(v) => {
            for x in v {
                w.write_all(&[*x as u8])?;
            }
        }
        GgufArray::String(v) => {
            for s in v {
                write_string(w, s)?;
            }
        }
        GgufArray::U64(v) => {
            for x in v {
                w.write_all(&x.to_le_bytes())?;
            }
        }
        GgufArray::I64(v) => {
            for x in v {
                w.write_all(&x.to_le_bytes())?;
            }
        }
        GgufArray::F64(v) => {
            for x in v {
                w.write_all(&x.to_le_bytes())?;
            }
        }
        GgufArray::Nested(v) => {
            // Each item is itself GgufValue::Array; the item payload is the
            // nested array's own (type, len, items) — matches the parser's
            // read_value(r, 9, depth+1) recursion.
            for item in v {
                match item {
                    GgufValue::Array(inner) => write_array(w, inner)?,
                    other => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("nested array item is not an array: {other:?}"),
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::Gguf;
    use std::io::Cursor;

    /// Build a synthetic GGUF fully in memory, re-parse it with the real
    /// parser, and assert metadata + tensor directory + payload bytes all
    /// round-trip.
    #[test]
    fn round_trip_synthetic() {
        let kvs_owned: Vec<(String, GgufValue)> = vec![
            ("general.architecture".into(), GgufValue::String("test".into())),
            ("a.u32".into(), GgufValue::U32(42)),
            ("a.f32".into(), GgufValue::F32(1.5)),
            ("a.bool".into(), GgufValue::Bool(true)),
            ("a.i64".into(), GgufValue::I64(-7)),
            (
                "a.strings".into(),
                GgufValue::Array(GgufArray::String(vec!["x".into(), "yy".into()])),
            ),
            (
                "a.i32s".into(),
                GgufValue::Array(GgufArray::I32(vec![1, -2, 3])),
            ),
            (
                "a.f32s".into(),
                GgufValue::Array(GgufArray::F32(vec![0.25, -0.5])),
            ),
        ];
        let kvs: Vec<(&str, &GgufValue)> =
            kvs_owned.iter().map(|(k, v)| (k.as_str(), v)).collect();

        let tensors = vec![
            TensorSpec {
                name: "t0.f32".into(),
                dims: vec![3, 2],
                dtype: GgufType::F32,
            },
            TensorSpec {
                name: "t1.f16".into(),
                dims: vec![5],
                dtype: GgufType::F16,
            },
            TensorSpec {
                name: "t2.q8_0".into(),
                dims: vec![32],
                dtype: GgufType::Q8_0,
            },
        ];

        let mut w =
            GgufWriter::new(Cursor::new(Vec::new()), &kvs, &tensors, 32).unwrap();
        let payloads: Vec<Vec<u8>> = vec![
            (0..24).collect(),
            (100..110).collect(),
            (0..34).collect(),
        ];
        for p in &payloads {
            // stream in two chunks to exercise the chunk path
            let (a, b) = p.split_at(p.len() / 2);
            w.write_tensor_chunk(a).unwrap();
            w.write_tensor_chunk(b).unwrap();
        }
        let cursor = w.finish().unwrap();
        let bytes = cursor.into_inner();

        let mut rd = Cursor::new(&bytes);
        let g = Gguf::parse_reader(&mut rd, bytes.len() as u64).unwrap();

        assert_eq!(g.version, 3);
        assert_eq!(g.n_kv, kvs_owned.len() as u64);
        assert_eq!(g.n_tensors, 3);
        assert_eq!(g.alignment, 32);

        // Metadata round-trips, in order.
        let parsed: Vec<(&str, &GgufValue)> = g.metadata_in_order().collect();
        assert_eq!(parsed.len(), kvs_owned.len());
        for ((wk, wv), (pk, pv)) in kvs.iter().zip(&parsed) {
            assert_eq!(wk, pk);
            assert_eq!(format!("{wv:?}"), format!("{pv:?}"));
        }

        // Tensor directory round-trips with aligned offsets + right bytes.
        for (spec, payload) in tensors.iter().zip(&payloads) {
            let t = g.tensor(&spec.name).unwrap();
            assert_eq!(t.dims, spec.dims);
            assert_eq!(t.dtype, spec.dtype);
            assert_eq!(t.byte_size, payload.len() as u64);
            assert_eq!(t.rel_offset % 32, 0);
            let start = t.abs_offset as usize;
            assert_eq!(&bytes[start..start + payload.len()], &payload[..]);
        }

        // Directory order preserved.
        let names: Vec<&str> = g.tensors().iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["t0.f32", "t1.f16", "t2.q8_0"]);
    }

    #[test]
    fn chunk_overflow_rejected() {
        let tensors = vec![TensorSpec {
            name: "t".into(),
            dims: vec![2],
            dtype: GgufType::F32,
        }];
        let mut w = GgufWriter::new(Cursor::new(Vec::new()), &[], &tensors, 32).unwrap();
        assert!(w.write_tensor_chunk(&[0u8; 9]).is_err());
        w.write_tensor_chunk(&[0u8; 8]).unwrap();
        assert!(w.write_tensor_chunk(&[0u8; 1]).is_err());
        w.finish().unwrap();
    }

    #[test]
    fn unfinished_rejected() {
        let tensors = vec![TensorSpec {
            name: "t".into(),
            dims: vec![2],
            dtype: GgufType::F32,
        }];
        let w = GgufWriter::new(Cursor::new(Vec::new()), &[], &tensors, 32).unwrap();
        assert!(w.finish().is_err());
    }
}
