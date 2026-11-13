use nioxs_engine::gguf::GgufModel;
use std::fs::File;
use std::io::BufReader;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = if args.len() > 1 {
        &args[1]
    } else {
        "/root/.nioxs/models/tiger.3b-q4km.gguf"
    };
    let mut file = BufReader::new(File::open(path)?);
    let model = GgufModel::read(&mut file)?;

    for (name, t) in &model.tensors {
        println!("{name}: dtype={}, shape={:?}", t.dtype, t.shape);
    }
    Ok(())
}
