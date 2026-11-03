use anyhow::{anyhow, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use memmap2::Mmap;
use nioxs_engine::quants;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{PathBuf};

#[derive(Clone)]
struct TensorEntry {
    name: String,
    dims: Vec<u64>,
    dtype: u32,
    offset: u64,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: quantize_vocab_iq4nl <input_gguf> <output_gguf>");
        std::process::exit(1);
    }
    let input_path = PathBuf::from(&args[1]);
    let output_path = PathBuf::from(&args[2]);

    println!("============================================================");
    println!("💎 NIO-XS GGUF VOCABULARY QUANTIZER (IQ4_NL Option B)");
    println!("   Input:  {}", input_path.display());
    println!("   Output: {}", output_path.display());
    println!("============================================================");

    let start_time = std::time::Instant::now();

    let file = File::open(&input_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let mut cursor = std::io::Cursor::new(&mmap[..]);

    let mut magic = [0u8; 4];
    cursor.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err(anyhow!("Not a GGUF file"));
    }
    let version = cursor.read_u32::<LittleEndian>()?;
    let tensor_count = cursor.read_u64::<LittleEndian>()? as usize;
    let kv_count = cursor.read_u64::<LittleEndian>()? as usize;

    let kv_start = cursor.position() as usize;

    // Skip metadata KVs and record raw slice
    for _ in 0..kv_count {
        let klen = cursor.read_u64::<LittleEndian>()? as usize;
        cursor.seek(SeekFrom::Current(klen as i64))?;
        let vtype = cursor.read_u32::<LittleEndian>()?;
        skip_kv_value(&mut cursor, vtype)?;
    }
    let kv_end = cursor.position() as usize;
    let metadata_bytes = &mmap[kv_start..kv_end];

    // Read tensor metadata
    let mut tensors = Vec::with_capacity(tensor_count);
    let mut token_embd_idx = None;

    for i in 0..tensor_count {
        let nlen = cursor.read_u64::<LittleEndian>()? as usize;
        let mut name_buf = vec![0u8; nlen];
        cursor.read_exact(&mut name_buf)?;
        let name = String::from_utf8(name_buf)?;
        let ndims = cursor.read_u32::<LittleEndian>()? as usize;
        let mut dims = Vec::with_capacity(ndims);
        for _ in 0..ndims {
            dims.push(cursor.read_u64::<LittleEndian>()?);
        }
        let dtype = cursor.read_u32::<LittleEndian>()?;
        let offset = cursor.read_u64::<LittleEndian>()?;

        if name == "token_embd.weight" {
            token_embd_idx = Some(i);
        }

        tensors.push(TensorEntry {
            name,
            dims,
            dtype,
            offset,
        });
    }

    let tensor_header_end = cursor.position() as usize;
    let pad = (32 - (tensor_header_end % 32)) % 32;
    let input_data_start = tensor_header_end + pad;

    let embd_idx = token_embd_idx.ok_or_else(|| anyhow!("token_embd.weight not found"))?;
    let embd_offset = tensors[embd_idx].offset as usize;
    let embd_dtype = tensors[embd_idx].dtype;
    let embd_dims = tensors[embd_idx].dims.clone();
    println!("Found token_embd.weight: dims={:?}, dtype={}, offset={}", embd_dims, embd_dtype, embd_offset);

    if embd_dtype != 8 {
        return Err(anyhow!("token_embd.weight is not Q8_0 (found dtype {})", embd_dtype));
    }

    let dim_d = embd_dims[0] as usize;
    let vocab_v = embd_dims[1] as usize;
    let old_row_bytes = (dim_d / 32) * 34;
    let new_row_bytes = (dim_d / 32) * 18;
    let old_total_bytes = vocab_v * old_row_bytes;
    let new_total_bytes = vocab_v * new_row_bytes;
    let delta = (old_total_bytes - new_total_bytes) as u64;

    println!("   Embedding dimension: {}", dim_d);
    println!("   Vocabulary count:    {}", vocab_v);
    println!("   Current row size:    {} bytes (Total: {:.2} MB)", old_row_bytes, old_total_bytes as f64 / 1_048_576.0);
    println!("   Target IQ4_NL size:  {} bytes (Total: {:.2} MB)", new_row_bytes, new_total_bytes as f64 / 1_048_576.0);
    println!("   Net RAM reduction:   {:.2} MB ({:.1}%)", (old_total_bytes - new_total_bytes) as f64 / 1_048_576.0, (1.0 - new_total_bytes as f64 / old_total_bytes as f64) * 100.0);

    // Multithreaded quantization of token_embd.weight
    println!("\n⚡ Quantizing token_embd.weight to IQ4_NL with Rayon (Lloyd-Max centroid optimization)...");
    let embd_data_start = input_data_start + embd_offset;
    let old_embd_slice = &mmap[embd_data_start..embd_data_start + old_total_bytes];

    let mut new_embd_bytes = vec![0u8; new_total_bytes];

    new_embd_bytes
        .par_chunks_exact_mut(new_row_bytes)
        .enumerate()
        .for_each(|(r, out_row)| {
            let old_row = &old_embd_slice[r * old_row_bytes..(r + 1) * old_row_bytes];
            let mut f32_buf = vec![0.0f32; dim_d];
            quants::dequantize_row_q8_0(old_row, &mut f32_buf);
            quants::quantize_row_iq4_nl(&f32_buf, out_row);
        });

    println!("✅ Quantization completed in {:.2}s!", start_time.elapsed().as_secs_f64());

    // Update tensor list
    tensors[embd_idx].dtype = 20; // IQ4_NL
    for i in (embd_idx + 1)..tensors.len() {
        tensors[i].offset -= delta;
    }

    // Write new GGUF
    println!("\n📦 Writing output GGUF to {}...", output_path.display());
    let out_file = File::create(&output_path)?;
    let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, out_file);

    writer.write_all(b"GGUF")?;
    writer.write_u32::<LittleEndian>(version)?;
    writer.write_u64::<LittleEndian>(tensor_count as u64)?;
    writer.write_u64::<LittleEndian>(kv_count as u64)?;
    writer.write_all(metadata_bytes)?;

    for t in &tensors {
        writer.write_u64::<LittleEndian>(t.name.len() as u64)?;
        writer.write_all(t.name.as_bytes())?;
        writer.write_u32::<LittleEndian>(t.dims.len() as u32)?;
        for &d in &t.dims {
            writer.write_u64::<LittleEndian>(d)?;
        }
        writer.write_u32::<LittleEndian>(t.dtype)?;
        writer.write_u64::<LittleEndian>(t.offset)?;
    }

    let current_pos = writer.stream_position()? as usize;
    let out_pad = (32 - (current_pos % 32)) % 32;
    if out_pad > 0 {
        writer.write_all(&vec![0u8; out_pad])?;
    }

    // Write tensor data
    // 1. Tensor 0 (output_norm.weight): from offset 0 to embd.offset
    let pre_embd_len = embd_offset;
    if pre_embd_len > 0 {
        writer.write_all(&mmap[input_data_start..input_data_start + pre_embd_len])?;
    }

    // 2. Quantized token_embd.weight
    writer.write_all(&new_embd_bytes)?;

    // 3. Post-embd tensors: from tensor[embd_idx + 1].offset in original file to EOF
    let post_start = input_data_start + embd_offset + old_total_bytes;
    let post_slice = &mmap[post_start..];
    writer.write_all(post_slice)?;

    writer.flush()?;

    let final_size = output_path.metadata()?.len();
    println!("🎉 DONE! Output artifact size: {:.2} MB (saved {:.2} MB)",
        final_size as f64 / 1_048_576.0,
        (mmap.len() as i64 - final_size as i64) as f64 / 1_048_576.0
    );

    Ok(())
}

fn skip_kv_value<R: Read + Seek>(reader: &mut R, vtype: u32) -> Result<()> {
    match vtype {
        0 | 1 | 7 => { reader.seek(SeekFrom::Current(1))?; }
        2 | 3 => { reader.seek(SeekFrom::Current(2))?; }
        4 | 5 | 6 => { reader.seek(SeekFrom::Current(4))?; }
        8 => {
            let len = reader.read_u64::<LittleEndian>()?;
            reader.seek(SeekFrom::Current(len as i64))?;
        }
        9 => {
            let subtype = reader.read_u32::<LittleEndian>()?;
            let count = reader.read_u64::<LittleEndian>()?;
            for _ in 0..count {
                skip_kv_value(reader, subtype)?;
            }
        }
        10 | 11 | 12 => { reader.seek(SeekFrom::Current(8))?; }
        other => return Err(anyhow!("unsupported GGUF value type {other}")),
    }
    Ok(())
}
