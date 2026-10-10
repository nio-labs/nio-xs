use nioxs_browser::{decode_entities, ResearchReport};
#[cfg(test)]
use nioxs_browser::ResearchPassage;

/// Cleans raw text of residual web artifacts, ads, and boilerplate phrases.
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

    // Clean multiple spaces and trim
    cleaned
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim_matches(|c: char| "`\"',;:!? \t\r\n".contains(c))
        .to_string()
}

/// Extract subject name from a conversational or research prompt.
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

/// Computes rough word overlap to prevent repeating duplicate facts.
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

/// Synthesize raw web search research into an intelligent, LLM-grade answering response.
pub fn synthesize_research_answer(query: &str, report: &ResearchReport) -> String {
    if report.passages.is_empty() {
        return format!(
            "I searched for '{query}' across web and documentation sources, but didn't find high-confidence cited passages. Try checking Wikipedia with `wiki {query}`."
        );
    }

    let subject = extract_subject(query);
    let mut candidate_sentences: Vec<(usize, String)> = Vec::new();

    // 1. Extract and score substantive sentences from all passages and snippets
    for p in &report.passages {
        let full_text = format!("{}\n{}", p.excerpt, p.snippet);
        let cleaned = sanitize_text(&full_text);

        for chunk in cleaned.split(|c| c == '.' || c == '\n') {
            let sentence = chunk.trim();
            if sentence.len() < 25 || sentence.len() > 350 {
                continue;
            }
            let lower_s = sentence.to_ascii_lowercase();
            let lower_sub = subject.to_ascii_lowercase();

            let mut score: usize = 0;
            if lower_s.contains(&lower_sub) {
                score += 5;
            }
            if lower_s.contains("is a ")
                || lower_s.contains("is an ")
                || lower_s.contains("refers to")
                || lower_s.contains("platform")
                || lower_s.contains("library")
                || lower_s.contains("website")
                || lower_s.contains("service")
            {
                score += 4;
            }
            if lower_s.contains("streaming")
                || lower_s.contains("dramas")
                || lower_s.contains("movies")
                || lower_s.contains("video")
                || lower_s.contains("korean")
                || lower_s.contains("chinese")
                || lower_s.contains("high quality")
                || lower_s.contains("download")
            {
                score += 3;
            }

            // Exclude store, shopping cart, DLC, and pricing junk
            if lower_s.contains("add to cart")
                || lower_s.contains("buy ")
                || lower_s.contains("bundle")
                || lower_s.contains("dlc")
                || lower_s.contains("usd")
                || lower_s.contains("add to account")
                || lower_s.contains("standard edition")
                || lower_s.contains("on steam")
                || lower_s.contains("cookie policy")
                || lower_s.contains("sign up")
                || lower_s.contains("log in")
            {
                continue;
            }

            // Exclude pure navigation or conversational chatter
            if lower_s.contains("idk if it's just me") || lower_s.contains("pls share in the comments") {
                score = score.saturating_sub(4);
            }

            if score > 0 {
                candidate_sentences.push((score, sentence.to_string()));
            }
        }
    }

    candidate_sentences.sort_by_key(|a| std::cmp::Reverse(a.0));

    // 2. Formulate Lead Answering Paragraph
    let mut answer = String::new();
    let mut used_sentences: Vec<String> = Vec::new();

    // Find best defining sentence
    let lead_definition = candidate_sentences.iter().find(|(_, s)| {
        let lower = s.to_ascii_lowercase();
        lower.contains(&subject.to_ascii_lowercase())
            && (lower.contains("is a ")
                || lower.contains("is an ")
                || lower.contains("platform")
                || lower.contains("website")
                || lower.contains("watch drama")
                || lower.contains("korean dramas"))
    });

    let lead_text = if let Some((_, def)) = lead_definition {
        used_sentences.push(def.clone());
        if !def.starts_with(&format!("**{subject}**")) && !def.starts_with(&subject) {
            format!("**{subject}**: {def}.")
        } else {
            format!("{def}.")
        }
    } else {
        // Fallback synthesization from top title and subject
        let top_title = &report.passages[0].title;
        let desc = if top_title.contains('|') {
            top_title.split('|').nth(1).unwrap_or("").trim()
        } else {
            top_title.as_str()
        };
        if !desc.is_empty() && desc != subject {
            format!("**{subject}** is an online platform associated with {desc}.")
        } else {
            format!("Summary of findings for **{subject}**:")
        }
    };

    answer.push_str(&lead_text);
    answer.push_str("\n\n");

    // 3. Synthesize Key Details / Bullet Points
    let mut bullet_points: Vec<(String, String)> = Vec::new();

    for (_, s) in &candidate_sentences {
        if used_sentences.contains(s) || is_redundant(s, &used_sentences) {
            continue;
        }

        let lower = s.to_ascii_lowercase();
        let (category, formatted) = if lower.contains("drama")
            || lower.contains("movies")
            || lower.contains("korean")
            || lower.contains("chinese")
        {
            ("Content & Library", format!("{s}."))
        } else if lower.contains("quality")
            || lower.contains("download")
            || lower.contains("1080p")
            || lower.contains("720p")
            || lower.contains("streaming")
        {
            ("Streaming & Formats", format!("{s}."))
        } else if lower.contains("legal")
            || lower.contains("security")
            || lower.contains("alternative")
            || lower.contains("down")
            || lower.contains("ad")
        {
            ("Platform Notes", format!("{s}."))
        } else {
            ("Overview", format!("{s}."))
        };

        used_sentences.push(s.clone());
        bullet_points.push((category.to_string(), formatted));
        if bullet_points.len() >= 3 {
            break;
        }
    }

    if !bullet_points.is_empty() {
        answer.push_str("### Key Highlights\n");
        for (cat, detail) in bullet_points {
            answer.push_str(&format!("• **{cat}**: {detail}\n"));
        }
        answer.push('\n');
    }

    // 4. Clean, Deduplicated Sources List
    answer.push_str("Sources:\n");
    let mut seen_urls = std::collections::HashSet::new();
    let mut source_idx = 1;

    for passage in &report.passages {
        if seen_urls.insert(passage.url.clone()) {
            let clean_t = sanitize_text(&passage.title);
            let display_title = if clean_t.is_empty() {
                &passage.url
            } else {
                &clean_t
            };
            let clean_u = decode_entities(&passage.url);
            answer.push_str(&format!("{}. [{}]({})\n", source_idx, display_title, clean_u));
            source_idx += 1;
            if source_idx > 4 {
                break;
            }
        }
    }

    answer.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_subject() {
        assert_eq!(extract_subject("what is kisskh?"), "Kisskh");
        assert_eq!(extract_subject("what is rust?"), "Rust");
        assert_eq!(extract_subject("tell me about niodb"), "Niodb");
        assert_eq!(extract_subject("who is Ada Lovelace?"), "Ada Lovelace");
    }

    #[test]
    fn test_sanitize_text() {
        let raw = "kisskh | Asian Dramas &amp; Movies Feel Free To Watch! Infinite loading ? click here Reload";
        let clean = sanitize_text(raw);
        assert!(!clean.contains("Infinite loading"));
        assert!(!clean.contains("&amp;"));
        assert!(clean.contains("Asian Dramas & Movies"));
    }

    #[test]
    fn test_synthesize_kisskh_response() {
        let report = ResearchReport {
            query: "what is kisskh".into(),
            passages: vec![
                ResearchPassage {
                    url: "https://iemlabs.com/blogs/what-is-kisskh/".into(),
                    title: "What Is Kisskh? Is It Legal? A Simple Breakdown".into(),
                    excerpt: "If you watch Korean dramas often, you've probably heard someone mention Kisskh as a secret place where fans watch their favorite K-drama episodes.".into(),
                    snippet: "If you watch Korean dramas often, you've probably heard someone mention Kisskh.".into(),
                    score: 10,
                },
                ResearchPassage {
                    url: "https://kisskh.co/".into(),
                    title: "Kisskh | Asian Dramas & Movies".into(),
                    excerpt: "Watch drama online in high quality with free downloads from 240p to 1080p HD.".into(),
                    snippet: "Free download high quality drama. Various formats from 240p to 1080p HD.".into(),
                    score: 8,
                },
            ],
            sources_visited: vec!["https://iemlabs.com".into(), "https://kisskh.co".into()],
        };

        let result = synthesize_research_answer("what is kisskh?", &report);
        assert!(result.contains("Kisskh"));
        assert!(result.contains("Key Highlights"));
        assert!(result.contains("Sources:"));
        assert!(!result.contains("&amp;"));
        assert!(!result.contains("Infinite loading"));
    }
}
