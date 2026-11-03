use std::fs::File;
use std::io::BufReader;
use nioxs_engine::gguf::GgufModel;
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = if args.len() > 1 { &args[1] } else { "/home/mn/nio-labs/nio-xs-models/beluga.vmicrolm-vector.gguf" };
    let mut file = BufReader::new(File::open(path)?);
    let model = GgufModel::read(&mut file)?;
    
    let mut family_bytes: HashMap<String, usize> = HashMap::new();
    let mut family_count: HashMap<String, usize> = HashMap::new();
    let mut total_bytes = 0;

    for (name, t) in &model.tensors {
        let num_elements: usize = t.shape.iter().product();
        let bytes = match t.dtype {
            0 => num_elements * 4, // F32
            1 => num_elements * 2, // F16
            8 => (num_elements / 32) * 34, // Q8_0
            18 => (num_elements / 256) * 98, // IQ3_XXS
            20 => (num_elements / 32) * 18, // IQ4_NL
            other => panic!("unknown dtype {other}"),
        };
        total_bytes += bytes;

        let family = if name.contains("ffn_gate") {
            "ffn_gate".to_string()
        } else if name.contains("ffn_up") {
            "ffn_up".to_string()
        } else if name.contains("ffn_down") {
            "ffn_down".to_string()
        } else if name.contains("attn_q") {
            "attn_q".to_string()
        } else if name.contains("attn_k") {
            "attn_k".to_string()
        } else if name.contains("attn_v") {
            "attn_v".to_string()
        } else if name.contains("attn_output") {
            "attn_output".to_string()
        } else if name.contains("token_embd") {
            "token_embd".to_string()
        } else {
            "norms_and_biases".to_string()
        };

        *family_bytes.entry(family.clone()).or_insert(0) += bytes;
        *family_count.entry(family).or_insert(0) += 1;
    }

    println!("Total Model Tensors Size: {:.2} MB", total_bytes as f64 / 1_048_576.0);
    let mut fams: Vec<_> = family_bytes.keys().collect();
    fams.sort_by_key(|f| std::cmp::Reverse(family_bytes[*f]));
    for f in fams {
        let b = family_bytes[f];
        println!("{:20} {:3} tensors | {:8.2} MB ({:5.1}%)", f, family_count[f], b as f64 / 1_048_576.0, (b as f64 / total_bytes as f64) * 100.0);
    }

    Ok(())
}
