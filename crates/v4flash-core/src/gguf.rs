//! GGUF v3 parser. No mmap — we only read the header + metadata + tensor
//! directory at parse time; raw tensor data is accessed lazily via
//! [`Gguf::tensor_byte_range`] which the caller resolves with mmap/Read.
//!
//! Reference: `external/ds4/ds4.c:825-1242`. Format spec is little-endian
//! throughout.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek};
use std::path::Path;

use color_eyre::eyre::{self, eyre};

/// GGUF magic: "GGUF" little-endian as a u32.
pub const GGUF_MAGIC: u32 = u32::from_le_bytes(*b"GGUF");

/// Default tensor-data alignment if `general.alignment` isn't in metadata.
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// Errors specific to GGUF parsing. Anything that's not "the file is
/// malformed" bubbles up as eyre.
#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("not a GGUF file (magic = {0:#x}, expected {GGUF_MAGIC:#x})")]
    BadMagic(u32),
    #[error("unsupported GGUF version {0} (only v3 supported)")]
    UnsupportedVersion(u32),
    #[error("file truncated: needed {needed} bytes at offset {offset}, file size is {size}")]
    Truncated {
        offset: u64,
        needed: u64,
        size: u64,
    },
    #[error("unknown metadata value type {0}")]
    UnknownValueType(u32),
    #[error("unknown tensor type {0}")]
    UnknownTensorType(u32),
    #[error("tensor {name:?} has unsupported number of dimensions: {ndim}")]
    BadTensorDims { name: String, ndim: u32 },
    #[error("string at offset {0} is not valid UTF-8: {1}")]
    BadUtf8(u64, std::str::Utf8Error),
    #[error("metadata array nesting too deep (>{0})")]
    ArrayTooDeep(u32),
    #[error("arithmetic overflow during tensor size calculation")]
    Overflow,
    #[error(transparent)]
    Io(#[from] io::Error),
}

/// Top-level GGUF handle. Header + metadata + tensor directory parsed
/// once at construction; tensor bytes are *not* loaded.
pub struct Gguf {
    pub version: u32,
    pub alignment: u64,
    pub n_tensors: u64,
    pub n_kv: u64,
    pub tensor_data_offset: u64,
    pub file_size: u64,
    metadata: HashMap<String, GgufValue>,
    tensors: Vec<GgufTensor>,
    tensor_index: HashMap<String, usize>,
}

impl Gguf {
    /// Parse a GGUF file's header + metadata + tensor directory.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let file = File::open(path.as_ref())?;
        let file_size = file.metadata()?.len();
        // Buffered for the small reads; metadata + tensor dir is at most
        // a few MB even for huge models.
        let mut reader = BufReader::with_capacity(1 << 16, file);
        Self::parse(&mut reader, file_size)
    }

    fn parse<R: Read + Seek>(r: &mut R, file_size: u64) -> Result<Self, GgufError> {
        let magic = read_u32(r)?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::BadMagic(magic));
        }
        let version = read_u32(r)?;
        if version != 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }
        let n_tensors = read_u64(r)?;
        let n_kv = read_u64(r)?;

        // Pass 1: metadata. Look for general.alignment along the way.
        let mut metadata = HashMap::with_capacity(n_kv as usize);
        let mut alignment = DEFAULT_ALIGNMENT;
        for _ in 0..n_kv {
            let key = read_string(r)?;
            let type_id = read_u32(r)?;
            let value = read_value(r, type_id, 0)?;
            if key == "general.alignment" {
                if let GgufValue::U32(v) = &value {
                    if *v > 0 {
                        alignment = *v as u64;
                    }
                }
            }
            metadata.insert(key, value);
        }

        // Pass 2: tensor directory.
        let mut tensors = Vec::with_capacity(n_tensors as usize);
        let mut tensor_index = HashMap::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = read_string(r)?;
            let ndim = read_u32(r)?;
            if ndim == 0 || ndim > 8 {
                return Err(GgufError::BadTensorDims { name, ndim });
            }
            let mut dims = Vec::with_capacity(ndim as usize);
            let mut elements: u64 = 1;
            for _ in 0..ndim {
                let d = read_u64(r)?;
                if d != 0 {
                    elements = elements
                        .checked_mul(d)
                        .ok_or(GgufError::Overflow)?;
                }
                dims.push(d);
            }
            let type_id = read_u32(r)?;
            let rel_offset = read_u64(r)?;
            let dtype = GgufType::from_id(type_id);

            let bytes = match dtype {
                Some(dt) => dt.size_of(elements)?,
                None => 0,
            };

            tensor_index.insert(name.clone(), tensors.len());
            tensors.push(GgufTensor {
                name,
                dims,
                dtype: dtype.unwrap_or(GgufType::Unknown(type_id)),
                rel_offset,
                abs_offset: 0, // filled in after we know tensor_data_offset
                elements,
                byte_size: bytes,
            });
        }

        // Tensor data starts at the next alignment boundary after the
        // end of the tensor directory.
        let cursor_pos = r.stream_position()?;
        let tensor_data_offset = align_up(cursor_pos, alignment);

        // Resolve absolute offsets + range-check.
        for t in &mut tensors {
            let abs = tensor_data_offset
                .checked_add(t.rel_offset)
                .ok_or(GgufError::Overflow)?;
            if t.byte_size != 0 {
                let end = abs.checked_add(t.byte_size).ok_or(GgufError::Overflow)?;
                if end > file_size {
                    return Err(GgufError::Truncated {
                        offset: abs,
                        needed: t.byte_size,
                        size: file_size,
                    });
                }
            }
            t.abs_offset = abs;
        }

        Ok(Gguf {
            version,
            alignment,
            n_tensors,
            n_kv,
            tensor_data_offset,
            file_size,
            metadata,
            tensors,
            tensor_index,
        })
    }

    pub fn metadata(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    pub fn metadata_keys(&self) -> impl Iterator<Item = &str> {
        self.metadata.keys().map(String::as_str)
    }

    pub fn tensors(&self) -> &[GgufTensor] {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Option<&GgufTensor> {
        self.tensor_index.get(name).map(|&i| &self.tensors[i])
    }

    /// Convenience accessors.
    pub fn architecture(&self) -> Option<&str> {
        self.metadata("general.architecture")
            .and_then(GgufValue::as_str)
    }
}

/// One tensor's directory entry.
#[derive(Debug, Clone)]
pub struct GgufTensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: GgufType,
    /// Offset relative to `Gguf::tensor_data_offset`.
    pub rel_offset: u64,
    /// Absolute byte offset in the file. Caller can mmap or pread here.
    pub abs_offset: u64,
    pub elements: u64,
    pub byte_size: u64,
}

/// GGUF metadata value. We store everything by value; the parser owns
/// all strings/arrays (no borrow from the file).
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(GgufArray),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    pub fn as_str(&self) -> Option<&str> {
        if let GgufValue::String(s) = self { Some(s) } else { None }
    }
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufValue::U32(v) => Some(*v),
            GgufValue::U64(v) => (*v).try_into().ok(),
            _ => None,
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::U32(v) => Some(*v as u64),
            GgufValue::U64(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        if let GgufValue::F32(v) = self { Some(*v) } else { None }
    }
    pub fn as_bool(&self) -> Option<bool> {
        if let GgufValue::Bool(v) = self { Some(*v) } else { None }
    }
    pub fn type_id(&self) -> u32 {
        match self {
            GgufValue::U8(_) => 0,
            GgufValue::I8(_) => 1,
            GgufValue::U16(_) => 2,
            GgufValue::I16(_) => 3,
            GgufValue::U32(_) => 4,
            GgufValue::I32(_) => 5,
            GgufValue::F32(_) => 6,
            GgufValue::Bool(_) => 7,
            GgufValue::String(_) => 8,
            GgufValue::Array(_) => 9,
            GgufValue::U64(_) => 10,
            GgufValue::I64(_) => 11,
            GgufValue::F64(_) => 12,
        }
    }
}

/// Strongly-typed array variants. Items are homogeneous per GGUF spec.
#[derive(Debug, Clone)]
pub enum GgufArray {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    Bool(Vec<bool>),
    String(Vec<String>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F64(Vec<f64>),
    /// Array-of-arrays. Items are GgufValue::Array(...) themselves.
    Nested(Vec<GgufValue>),
}

impl GgufArray {
    pub fn len(&self) -> usize {
        match self {
            GgufArray::U8(v) => v.len(),
            GgufArray::I8(v) => v.len(),
            GgufArray::U16(v) => v.len(),
            GgufArray::I16(v) => v.len(),
            GgufArray::U32(v) => v.len(),
            GgufArray::I32(v) => v.len(),
            GgufArray::F32(v) => v.len(),
            GgufArray::Bool(v) => v.len(),
            GgufArray::String(v) => v.len(),
            GgufArray::U64(v) => v.len(),
            GgufArray::I64(v) => v.len(),
            GgufArray::F64(v) => v.len(),
            GgufArray::Nested(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_strings(&self) -> Option<&[String]> {
        if let GgufArray::String(v) = self { Some(v) } else { None }
    }
    pub fn as_i32s(&self) -> Option<&[i32]> {
        if let GgufArray::I32(v) = self { Some(v) } else { None }
    }
    pub fn as_u32s(&self) -> Option<&[u32]> {
        if let GgufArray::U32(v) = self { Some(v) } else { None }
    }
    pub fn as_f32s(&self) -> Option<&[f32]> {
        if let GgufArray::F32(v) = self { Some(v) } else { None }
    }
}

/// GGUF tensor element type. Table from `external/ds4/ds4.c:856-886`.
/// Variant names intentionally mirror the GGUF spec (Q8_0, IQ2_XXS, ...);
/// upper-camel-case would obscure the correspondence.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ1_S,
    IQ4_NL,
    IQ3_S,
    IQ2_S,
    IQ4_XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    IQ1_M,
    BF16,
    Unknown(u32),
}

impl GgufType {
    pub fn from_id(id: u32) -> Option<Self> {
        Some(match id {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            16 => Self::IQ2_XXS,
            17 => Self::IQ2_XS,
            18 => Self::IQ3_XXS,
            19 => Self::IQ1_S,
            20 => Self::IQ4_NL,
            21 => Self::IQ3_S,
            22 => Self::IQ2_S,
            23 => Self::IQ4_XS,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::IQ1_M,
            30 => Self::BF16,
            _ => return None,
        })
    }

    /// (block_elems, block_bytes) for computing tensor sizes.
    pub fn block_shape(&self) -> Option<(u32, u32)> {
        Some(match self {
            Self::F32 => (1, 4),
            Self::F16 => (1, 2),
            Self::Q4_0 => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::Q5_0 => (32, 22),
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),
            Self::Q8_1 => (32, 40),
            Self::Q2_K => (256, 84),
            Self::Q3_K => (256, 110),
            Self::Q4_K => (256, 144),
            Self::Q5_K => (256, 176),
            Self::Q6_K => (256, 210),
            Self::Q8_K => (256, 292),
            Self::IQ2_XXS => (256, 66),
            Self::IQ2_XS => (256, 74),
            Self::IQ3_XXS => (256, 98),
            Self::IQ1_S => (256, 110),
            Self::IQ4_NL => (256, 50),
            Self::IQ3_S => (256, 110),
            Self::IQ2_S => (256, 82),
            Self::IQ4_XS => (256, 136),
            Self::I8 => (1, 1),
            Self::I16 => (1, 2),
            Self::I32 => (1, 4),
            Self::I64 => (1, 8),
            Self::F64 => (1, 8),
            Self::IQ1_M => (256, 56),
            Self::BF16 => (1, 2),
            Self::Unknown(_) => return None,
        })
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Q4_0 => "q4_0",
            Self::Q4_1 => "q4_1",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
            Self::Q8_0 => "q8_0",
            Self::Q8_1 => "q8_1",
            Self::Q2_K => "q2_k",
            Self::Q3_K => "q3_k",
            Self::Q4_K => "q4_k",
            Self::Q5_K => "q5_k",
            Self::Q6_K => "q6_k",
            Self::Q8_K => "q8_k",
            Self::IQ2_XXS => "iq2_xxs",
            Self::IQ2_XS => "iq2_xs",
            Self::IQ3_XXS => "iq3_xxs",
            Self::IQ1_S => "iq1_s",
            Self::IQ4_NL => "iq4_nl",
            Self::IQ3_S => "iq3_s",
            Self::IQ2_S => "iq2_s",
            Self::IQ4_XS => "iq4_xs",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F64 => "f64",
            Self::IQ1_M => "iq1_m",
            Self::BF16 => "bf16",
            Self::Unknown(_) => "<unknown>",
        }
    }

    /// Bytes for `n_elements` of this type, accounting for block packing.
    pub fn size_of(&self, n_elements: u64) -> Result<u64, GgufError> {
        let (block_elems, block_bytes) = self
            .block_shape()
            .ok_or_else(|| GgufError::UnknownTensorType(match self {
                Self::Unknown(id) => *id,
                _ => 0,
            }))?;
        let block_elems = block_elems as u64;
        let block_bytes = block_bytes as u64;
        let blocks = (n_elements + block_elems - 1) / block_elems;
        blocks.checked_mul(block_bytes).ok_or(GgufError::Overflow)
    }
}

// ----- internal readers -----

fn read_u8<R: Read>(r: &mut R) -> Result<u8, GgufError> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}
fn read_u16<R: Read>(r: &mut R) -> Result<u16, GgufError> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}
fn read_u32<R: Read>(r: &mut R) -> Result<u32, GgufError> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_u64<R: Read>(r: &mut R) -> Result<u64, GgufError> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn read_i8<R: Read>(r: &mut R) -> Result<i8, GgufError> {
    Ok(read_u8(r)? as i8)
}
fn read_i16<R: Read>(r: &mut R) -> Result<i16, GgufError> {
    Ok(read_u16(r)? as i16)
}
fn read_i32<R: Read>(r: &mut R) -> Result<i32, GgufError> {
    Ok(read_u32(r)? as i32)
}
fn read_i64<R: Read>(r: &mut R) -> Result<i64, GgufError> {
    Ok(read_u64(r)? as i64)
}
fn read_f32<R: Read>(r: &mut R) -> Result<f32, GgufError> {
    Ok(f32::from_le_bytes(read_u32(r)?.to_le_bytes()))
}
fn read_f64<R: Read>(r: &mut R) -> Result<f64, GgufError> {
    Ok(f64::from_le_bytes(read_u64(r)?.to_le_bytes()))
}
fn read_bool<R: Read>(r: &mut R) -> Result<bool, GgufError> {
    Ok(read_u8(r)? != 0)
}

fn read_string<R: Read + Seek>(r: &mut R) -> Result<String, GgufError> {
    let offset_for_err = r.stream_position()?;
    let len = read_u64(r)? as usize;
    // Reasonable sanity bound — no GGUF string should be > 16 MiB.
    if len > 16 * 1024 * 1024 {
        return Err(GgufError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("implausible string length {len} at offset {offset_for_err}"),
        )));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| GgufError::BadUtf8(offset_for_err, e.utf8_error()))
}

fn read_value<R: Read + Seek>(
    r: &mut R,
    type_id: u32,
    depth: u32,
) -> Result<GgufValue, GgufError> {
    if depth > 8 {
        return Err(GgufError::ArrayTooDeep(depth));
    }
    Ok(match type_id {
        0 => GgufValue::U8(read_u8(r)?),
        1 => GgufValue::I8(read_i8(r)?),
        2 => GgufValue::U16(read_u16(r)?),
        3 => GgufValue::I16(read_i16(r)?),
        4 => GgufValue::U32(read_u32(r)?),
        5 => GgufValue::I32(read_i32(r)?),
        6 => GgufValue::F32(read_f32(r)?),
        7 => GgufValue::Bool(read_bool(r)?),
        8 => GgufValue::String(read_string(r)?),
        9 => GgufValue::Array(read_array(r, depth)?),
        10 => GgufValue::U64(read_u64(r)?),
        11 => GgufValue::I64(read_i64(r)?),
        12 => GgufValue::F64(read_f64(r)?),
        other => return Err(GgufError::UnknownValueType(other)),
    })
}

fn read_array<R: Read + Seek>(r: &mut R, depth: u32) -> Result<GgufArray, GgufError> {
    let item_type = read_u32(r)?;
    let len_u64 = read_u64(r)?;
    let len = len_u64 as usize;
    if (len as u64) != len_u64 {
        return Err(GgufError::Overflow);
    }
    Ok(match item_type {
        0 => GgufArray::U8((0..len).map(|_| read_u8(r)).collect::<Result<_, _>>()?),
        1 => GgufArray::I8((0..len).map(|_| read_i8(r)).collect::<Result<_, _>>()?),
        2 => GgufArray::U16((0..len).map(|_| read_u16(r)).collect::<Result<_, _>>()?),
        3 => GgufArray::I16((0..len).map(|_| read_i16(r)).collect::<Result<_, _>>()?),
        4 => GgufArray::U32((0..len).map(|_| read_u32(r)).collect::<Result<_, _>>()?),
        5 => GgufArray::I32((0..len).map(|_| read_i32(r)).collect::<Result<_, _>>()?),
        6 => GgufArray::F32((0..len).map(|_| read_f32(r)).collect::<Result<_, _>>()?),
        7 => GgufArray::Bool((0..len).map(|_| read_bool(r)).collect::<Result<_, _>>()?),
        8 => GgufArray::String((0..len).map(|_| read_string(r)).collect::<Result<_, _>>()?),
        9 => {
            // Nested array; descend.
            let mut items = Vec::with_capacity(len);
            for _ in 0..len {
                items.push(read_value(r, 9, depth + 1)?);
            }
            GgufArray::Nested(items)
        }
        10 => GgufArray::U64((0..len).map(|_| read_u64(r)).collect::<Result<_, _>>()?),
        11 => GgufArray::I64((0..len).map(|_| read_i64(r)).collect::<Result<_, _>>()?),
        12 => GgufArray::F64((0..len).map(|_| read_f64(r)).collect::<Result<_, _>>()?),
        other => return Err(GgufError::UnknownValueType(other)),
    })
}

fn align_up(n: u64, alignment: u64) -> u64 {
    if alignment <= 1 {
        return n;
    }
    let rem = n % alignment;
    if rem == 0 { n } else { n + (alignment - rem) }
}

/// Lift a `GgufError` into an eyre report. For binaries.
pub fn open_eyre(path: impl AsRef<Path>) -> eyre::Result<Gguf> {
    Gguf::open(path).map_err(|e| eyre!("GGUF parse error: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_up_basic() {
        assert_eq!(align_up(0, 32), 0);
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(31, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
        assert_eq!(align_up(100, 1), 100);
    }

    #[test]
    fn tensor_size_examples() {
        // Q8_0: 32 elements per block, 34 bytes/block
        assert_eq!(GgufType::Q8_0.size_of(32).unwrap(), 34);
        assert_eq!(GgufType::Q8_0.size_of(64).unwrap(), 68);
        // Not block-aligned: rounds up
        assert_eq!(GgufType::Q8_0.size_of(33).unwrap(), 68);
        // IQ2_XXS: 256 elements per block, 66 bytes/block
        assert_eq!(GgufType::IQ2_XXS.size_of(256).unwrap(), 66);
        assert_eq!(GgufType::IQ2_XXS.size_of(512).unwrap(), 132);
        // F16: dense
        assert_eq!(GgufType::F16.size_of(1000).unwrap(), 2000);
    }

    #[test]
    fn magic_constant() {
        assert_eq!(GGUF_MAGIC, 0x46554747);
    }
}
