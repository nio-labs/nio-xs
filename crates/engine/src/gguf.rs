use anyhow::{anyhow, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::io::{Read, Seek};

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
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    pub fn to_u32(&self) -> Result<u32> {
        match self {
            Self::U8(v) => Ok(*v as u32),
            Self::U16(v) => Ok(*v as u32),
            Self::U32(v) => Ok(*v),
            Self::U64(v) => Ok(*v as u32),
            Self::I8(v) => Ok(*v as u32),
            Self::I16(v) => Ok(*v as u32),
            Self::I32(v) => Ok(*v as u32),
            Self::I64(v) => Ok(*v as u32),
            _ => Err(anyhow!("expected integer for u32, got {:?}", self)),
        }
    }

    pub fn to_f32(&self) -> Result<f32> {
        match self {
            Self::F32(v) => Ok(*v),
            Self::F64(v) => Ok(*v as f32),
            _ => Err(anyhow!("expected float for f32, got {:?}", self)),
        }
    }

    pub fn to_string(&self) -> Result<String> {
        match self {
            Self::String(s) => Ok(s.clone()),
            _ => Err(anyhow!("expected string, got {:?}", self)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: u32,
    pub offset: u64,
}

pub struct GgufModel {
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, TensorMeta>,
    pub tensor_data_offset: usize,
}

fn read_str<R: Read>(reader: &mut R) -> Result<String> {
    let len = reader.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| anyhow!("utf8 error: {e}"))
}

fn read_val<R: Read>(reader: &mut R, vtype: u32) -> Result<GgufValue> {
    match vtype {
        0 => Ok(GgufValue::U8(reader.read_u8()?)),
        1 => Ok(GgufValue::I8(reader.read_i8()?)),
        2 => Ok(GgufValue::U16(reader.read_u16::<LittleEndian>()?)),
        3 => Ok(GgufValue::I16(reader.read_i16::<LittleEndian>()?)),
        4 => Ok(GgufValue::U32(reader.read_u32::<LittleEndian>()?)),
        5 => Ok(GgufValue::I32(reader.read_i32::<LittleEndian>()?)),
        6 => Ok(GgufValue::F32(reader.read_f32::<LittleEndian>()?)),
        7 => Ok(GgufValue::Bool(reader.read_u8()? != 0)),
        8 => Ok(GgufValue::String(read_str(reader)?)),
        9 => {
            let sub_type = reader.read_u32::<LittleEndian>()?;
            let count = reader.read_u64::<LittleEndian>()? as usize;
            let mut list = Vec::with_capacity(count);
            for _ in 0..count {
                list.push(read_val(reader, sub_type)?);
            }
            Ok(GgufValue::Array(list))
        }
        10 => Ok(GgufValue::U64(reader.read_u64::<LittleEndian>()?)),
        11 => Ok(GgufValue::I64(reader.read_i64::<LittleEndian>()?)),
        12 => Ok(GgufValue::F64(reader.read_f64::<LittleEndian>()?)),
        _ => Err(anyhow!("unsupported GGUF value type {vtype}")),
    }
}

impl GgufModel {
    pub fn read<R: Read + Seek>(reader: &mut R) -> Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != b"GGUF" {
            return Err(anyhow!("invalid GGUF magic"));
        }
        let version = reader.read_u32::<LittleEndian>()?;
        if version != 2 && version != 3 {
            return Err(anyhow!("unsupported GGUF version {version}"));
        }
        let tensor_count = reader.read_u64::<LittleEndian>()? as usize;
        let metadata_kv_count = reader.read_u64::<LittleEndian>()? as usize;

        let mut metadata = HashMap::with_capacity(metadata_kv_count);
        for _ in 0..metadata_kv_count {
            let key = read_str(reader)?;
            let vtype = reader.read_u32::<LittleEndian>()?;
            let val = read_val(reader, vtype)?;
            metadata.insert(key, val);
        }

        let mut tensors = HashMap::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = read_str(reader)?;
            let n_dims = reader.read_u32::<LittleEndian>()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(reader.read_u64::<LittleEndian>()? as usize);
            }
            dims.reverse(); // Reverse dims to match row-major order: [n_out, n_in]
            let dtype = reader.read_u32::<LittleEndian>()?;
            let offset = reader.read_u64::<LittleEndian>()?;
            tensors.insert(
                name.clone(),
                TensorMeta {
                    name,
                    shape: dims,
                    dtype,
                    offset,
                },
            );
        }

        let cur_pos = reader.stream_position()? as usize;
        let alignment = match metadata.get("general.alignment") {
            Some(GgufValue::U32(a)) => *a as usize,
            Some(GgufValue::U64(a)) => *a as usize,
            _ => 32usize,
        };
        let pad = (alignment - (cur_pos % alignment)) % alignment;
        let tensor_data_offset = cur_pos + pad;

        Ok(Self {
            metadata,
            tensors,
            tensor_data_offset,
        })
    }
}
