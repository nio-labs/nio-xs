use nioxs_browser::{decode_entities, ResearchReport};
#[cfg(test)]
use nioxs_browser::ResearchPassage;

pub fn sanitize_text(text: &str) -> String {
    let decoded = decode_entities(text);
    let junk_phrases = [
        "infinite loading ? click here reload",
        "infinite loading ? click here",
        "infinite loading",
        "click here reload",
        "click here to reload",
        "feel free to watch!",
        "feel free to watch",
        "log into your account",
        "skip to content",
        "all rights reserved",
    ];

    let mut cleaned = decoded;
    for junk in &junk_phrases {
        let lower = cleaned.to_ascii_lowercase();
        if let Some(pos) = lower.find(junk) {
            let end = pos + junk.len();
            cleaned = format!("{}{}", &cleaned[..pos], &cleaned[end..]);
        }
    }

    cleaned
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim_matches(|c: char| "`\"',;:!? \t\r\n".contains(c))
        .to_string()
}

pub fn extract_subject(query: &str) -> String {
    let prefixes = [
        "what is the definition of ",
        "what is ",
        "what are ",
        "who is ",
        "who are ",
        "what's ",
        "tell me about ",
        "explain to me ",
        "explain ",
        "overview of ",
        "details on ",
        "how does ",
        "how do i ",
        "how to ",
    ];

    let mut current = query.trim();
    let mut changed = true;
    while changed {
        changed = false;
        let lower = current.to_ascii_lowercase();
        for prefix in &prefixes {
            if lower.starts_with(prefix) {
                current = current[prefix.len()..].trim();
                changed = true;
                break;
            }
        }
    }

    let stripped = current.trim_end_matches(|c: char| "?!. \t\r\n".contains(c));
    if stripped.is_empty() {
        return query.trim().to_string();
    }

    let mut chars = stripped.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

fn is_redundant(candidate: &str, existing: &[String]) -> bool {
    let cand_words: Vec<&str> = candidate
        .split_whitespace()
        .filter(|w| w.len() > 3)
        .collect();
    if cand_words.is_empty() {
        return true;
    }

    for ex in existing {
        let ex_words: Vec<&str> = ex.split_whitespace().filter(|w| w.len() > 3).collect();
        let overlap = cand_words.iter().filter(|w| ex_words.contains(w)).count();
        if overlap as f32 / cand_words.len() as f32 > 0.55 {
            return true;
        }
    }
    false
}

#[derive(PartialEq)]
enum QueryIntent {
    Definition,
    HowTo,
    Troubleshoot,
    General,
}

pub fn synthesize_research_answer(query: &str, report: &ResearchReport) -> String {
    if report.passages.is_empty() {
        return format!(
            "I searched for '{query}' across web and documentation sources, but didn't find high-confidence cited passages. Try checking Wikipedia with `wiki {query}`."
        );
    }

    let subject = extract_subject(query);
    let lower_q = query.to_ascii_lowercase();
    
    let intent = if lower_q.contains("how to") || lower_q.contains("how do i") || lower_q.contains("example") {
        QueryIntent::HowTo
    } else if lower_q.contains("error") || lower_q.contains("failed") || lower_q.contains("fix") {
        QueryIntent::Troubleshoot
    } else if lower_q.contains("what is") || lower_q.contains("define") {
        QueryIntent::Definition
    } else {
        QueryIntent::General
    };

    let mut candidate_sentences: Vec<(usize, String)> = Vec::new();

    for p in &report.passages {
        let full_text = format!("{}\n{}", p.excerpt, p.snippet);
        let cleaned = sanitize_text(&full_text);

        for chunk in cleaned.split(|c| c == '.' || c == '\n') {
            let sentence = chunk.trim();
            if sentence.len() < 20 || sentence.len() > 400 {
                continue;
            }
            let lower_s = sentence.to_ascii_lowercase();
            let lower_sub = subject.to_ascii_lowercase();

            let mut score: usize = 0;
            if lower_s.contains(&lower_sub) {
                score += 5;
            }
            
            match intent {
                QueryIntent::Definition => {
                    if lower_s.contains("is a ") || lower_s.contains("is an ") || lower_s.contains("refers to") {
                        score += 5;
                    }
                    if lower_s.contains("used for") || lower_s.contains("designed to") {
                        score += 3;
                    }
                }
                QueryIntent::HowTo => {
                    if lower_s.contains("you can") || lower_s.contains("to do this") || lower_s.contains("example") || lower_s.contains("step") {
                        score += 5;
                    }
                    if sentence.contains("use ") || sentence.contains("run ") {
                        score += 2;
                    }
                }
                QueryIntent::Troubleshoot => {
                    if lower_s.contains("occurs when") || lower_s.contains("caused by") || lower_s.contains("solution") || lower_s.contains("fix") {
                        score += 5;
                    }
                }
                QueryIntent::General => {
                    if lower_s.contains("is ") || lower_s.contains("allows") || lower_s.contains("provides") {
                        score += 3;
                    }
                }
            }

            // Penalties
            if lower_s.contains("add to cart") || lower_s.contains("buy ") || lower_s.contains("sign up") || lower_s.contains("log in") {
                continue;
            }

            if score > 0 {
                candidate_sentences.push((score, sentence.to_string()));
            }
        }
    }

    candidate_sentences.sort_by_key(|a| std::cmp::Reverse(a.0));

    let mut answer = String::new();
    let mut used_sentences: Vec<String> = Vec::new();

    if intent == QueryIntent::Definition {
        let lead_definition = candidate_sentences.iter().find(|(_, s)| {
            let lower = s.to_ascii_lowercase();
            lower.contains(&subject.to_ascii_lowercase())
                && (lower.contains("is a ") || lower.contains("is an ") || lower.contains("refers to"))
        });

        if let Some((_, def)) = lead_definition {
            used_sentences.push(def.clone());
            let mut clean_def = def.clone();
            if let Some(idx) = clean_def.find(&format!("{} is a", subject)) {
                clean_def = clean_def[idx..].to_string();
            } else if let Some(idx) = clean_def.find(&format!("{} is an", subject)) {
                clean_def = clean_def[idx..].to_string();
            }
            
            let junk_markers = [" Retrieved ", " Download ", " 202", " Language Reference "];
            for marker in junk_markers {
                if let Some(idx) = clean_def.find(marker) {
                    clean_def = clean_def[..idx].to_string();
                }
            }
            
            if !clean_def.starts_with(&format!("**{subject}**")) && !clean_def.starts_with(&subject) {
                answer.push_str(&format!("**{subject}**: {clean_def}."));
            } else {
                answer.push_str(&format!("{clean_def}."));
            }
            return answer;
        }
    }

    // Fallback or How-To/Troubleshooting: combine top 2-3 sentences intelligently
    let top_title = &report.passages[0].title;
    let clean_title = if top_title.contains('|') {
        top_title.split('|').nth(0).unwrap_or("").trim()
    } else {
        top_title.as_str()
    };
    
    answer.push_str(&format!("Based on '{clean_title}':\n\n"));
    
    let mut count = 0;
    for (_, s) in &candidate_sentences {
        if used_sentences.contains(s) || is_redundant(s, &used_sentences) {
            continue;
        }
        if s.contains("Retrieved") || s.contains("Download") {
            continue;
        }

        used_sentences.push(s.clone());
        answer.push_str(&format!("- {s}.\n"));
        count += 1;
        if count >= 3 {
            break;
        }
    }
    
    if count == 0 {
        answer = format!("**{subject}**: Found related pages but could not extract a concrete answer. Top match: {clean_title}.");
    }

    answer.trim().to_string()
}
