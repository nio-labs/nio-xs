use nioxs_nano::{Model, parse};
use serde_json::{Value, json};
use std::collections::BTreeMap;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = std::env::args().nth(1).ok_or("provide a JSONL corpus")?;
    let model = Model::embedded()?;
    let mut raw = BTreeMap::<(String, String), usize>::new();
    let mut routed = BTreeMap::<(String, String), usize>::new();
    let (mut n, mut negatives, mut false_accept) = (0, 0, 0);
    let mut mismatches = vec![];
    for line in std::fs::read_to_string(file)?.lines() {
        let row: Value = serde_json::from_str(line)?;
        let prompt = row["prompt"].as_str().ok_or("missing prompt")?;
        let expected = row["intent"].as_str().ok_or("missing intent")?;
        let prediction = model.predict(prompt).label;
        let actual = parse(&model, prompt).label().to_string();
        *raw.entry((expected.into(), prediction)).or_default() += 1;
        *routed.entry((expected.into(), actual.clone())).or_default() += 1;
        n += 1;
        if expected == "unsupported" {
            negatives += 1;
            if !["unsupported", "clarify"].contains(&actual.as_str()) {
                false_accept += 1;
            }
        }
        if actual != expected {
            mismatches.push(json!({"id":row["id"],"expected":expected,"actual":actual}));
        }
    }
    let metrics = |matrix: BTreeMap<(String, String), usize>| {
        let classes:Vec<Value>=model.labels.iter().map(|label|{let tp=*matrix.get(&(label.clone(),label.clone())).unwrap_or(&0);let fp:usize=matrix.iter().filter(|((a,p),_)|p==label&&a!=label).map(|(_,n)|*n).sum();let fn_:usize=matrix.iter().filter(|((a,p),_)|a==label&&p!=label).map(|(_,n)|*n).sum();json!({"label":label,"count":matrix.iter().filter(|((a,_),_)|a==label).map(|(_,n)|*n).sum::<usize>(),"f1":if 2*tp+fp+fn_==0{0.0}else{(2*tp)as f64/(2*tp+fp+fn_)as f64}})}).collect();
        let macro_f1 = classes
            .iter()
            .map(|v| v["f1"].as_f64().unwrap())
            .sum::<f64>()
            / classes.len() as f64;
        json!({"macro_f1":macro_f1,"classes":classes})
    };
    let raw_metrics = metrics(raw);
    let routed_metrics = metrics(routed);
    let accepted = routed_metrics["macro_f1"].as_f64().unwrap() >= 0.9
        && false_accept as f64 / negatives.max(1) as f64 <= 0.02;
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"model_version":model.model_version,"records":n,"unsupported_count":negatives,"unsupported_false_accepts":false_accept,"raw_classifier":raw_metrics,"typed_router":routed_metrics,"mismatches":mismatches,"acceptance_passed":accepted,"corpus_kind":"post-freeze authored synthetic corpus; not real-world accuracy"})
        )?
    );
    if !accepted {
        std::process::exit(1);
    }
    Ok(())
}
