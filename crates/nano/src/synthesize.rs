use nioxs_browser::{ResearchReport, decode_entities};

pub fn sanitize_text(text: &str) -> String {
    let decoded = decode_entities(text);
    // Strip decorative glyphs (arrows, emoji, bullets) that leak in from
    // navigation chrome and search-result headlines.
    let mut denoised = String::with_capacity(decoded.len());
    let mut prev_was_space = true;
    for c in decoded.chars() {
        let is_junk = matches!(
            c,
            '\u{2191}' | '\u{2193}' | '\u{2190}' | '\u{2192}'   // arrows
            | '\u{2022}' | '\u{25CF}' | '\u{25AA}' | '\u{25B6}' // bullets
            | '\u{2728}' | '\u{26A1}' | '\u{1F525}' | '\u{1F680}'
            | '\u{2705}' | '\u{274C}' | '\u{2764}' | '\u{1F4A1}'
            | '\u{26A0}' | '\u{1F514}' | '\u{1F4E2}'
        ) || (c as u32 >= 0x1F300 && c as u32 <= 0x1FAFF); // emoji blocks
        if is_junk {
            if !prev_was_space {
                denoised.push(' ');
                prev_was_space = true;
            }
            continue;
        }
        denoised.push(c);
        prev_was_space = c.is_whitespace();
    }

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
        "jump to content from wikipedia the free encyclopedia",
        "this article has multiple issues",
        "find sources:",
        "learn how and when to remove this message",
        "some of this article's listed sources may not be reliable",
        "news \u{b7} newspapers \u{b7} books \u{b7} scholar \u{b7} jstor",
    ];

    let mut cleaned = denoised;
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
        "who created ",
        "who wrote ",
        "who made ",
        "who develops ",
        "what's ",
        "tell me about ",
        "explain to me ",
        "explain ",
        "overview of ",
        "details on ",
        "why is ",
        "why are ",
        "why does ",
        "why do ",
        "why did ",
        "why should i ",
        "why use ",
        "why ",
        "when was ",
        "when is ",
        "where is ",
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

/// Detects search-result headlines and navigation labels that leaked into
/// extracted text. Real prose sentences pass; titles like
/// `"Why Zig when There is Already C++, D, and Rust?"` do not.
fn is_headline_like(sentence: &str) -> bool {
    let words: Vec<&str> = sentence.split_whitespace().collect();
    if words.len() < 6 {
        return true;
    }
    if sentence.starts_with('"') || sentence.ends_with('"') || sentence.ends_with('\u{201d}') {
        return true;
    }
    let title_case = words
        .iter()
        .filter(|w| w.chars().next().is_some_and(|c| c.is_ascii_uppercase()))
        .count();
    // Headlines capitalize most words; prose sentences capitalize only the first
    // (plus proper nouns). Reject when > 45% of words are capitalized mid-sentence.
    title_case * 20 > words.len() * 9
}

/// Navigation chrome and maintenance banners that leak into scraped pages.
const NAV_MARKERS: [&str; 22] = [
    "jump to content",
    "from wikipedia",
    "free encyclopedia",
    "main page",
    "current events",
    "random article",
    "about wikipedia",
    "toggle the table of contents",
    "personal tools",
    "create account",
    "log in",
    "retrieved from",
    "privacy policy",
    "disclaimers",
    "mobile view",
    "this article has multiple issues",
    "find sources:",
    "learn how and when to remove this message",
    "listed sources may not be reliable",
    "cookie statement",
    "wikimedia foundation",
    "needs additional citations",
];

/// Text that ends a sentence when a maintenance banner bleeds into it.
const DEFINITION_STOP_MARKERS: [&str; 6] = [
    " this article",
    " find sources",
    " learn how",
    " some of this article",
    " retrieved ",
    " from wikipedia",
];

fn is_navigation_line(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    NAV_MARKERS.iter().any(|marker| lower.contains(marker))
}

/// Finds the sentence that actually defines the subject, even when navigation
/// chrome or maintenance banners precede it in the scraped text. Without this,
/// a flattened page header can swallow the definition entirely.
fn find_definition(text: &str, subject: &str) -> Option<String> {
    let lower_text = text.to_ascii_lowercase();
    let lower_subject = subject.to_ascii_lowercase();
    if lower_subject.is_empty() {
        return None;
    }
    let patterns = [
        format!("{lower_subject} is a"),
        format!("{lower_subject} is an"),
        format!("{lower_subject} is the"),
        format!("{lower_subject} refers to"),
        format!("{lower_subject} is "),
        format!("{lower_subject} was "),
    ];
    for pattern in patterns {
        let Some(start) = lower_text.find(&pattern) else {
            continue;
        };
        // Never start mid-word ("zig" inside "zigzag").
        if lower_text[..start]
            .chars()
            .next_back()
            .is_some_and(|c| c.is_alphanumeric() || c == '-')
        {
            continue;
        }
        let rest = &text[start..];
        let end = rest.find('.').unwrap_or(rest.len());
        let candidate = &rest[..end];
        let lower_candidate = candidate.to_ascii_lowercase();
        let cut = DEFINITION_STOP_MARKERS
            .iter()
            .filter_map(|marker| lower_candidate.find(marker))
            .min()
            .unwrap_or(candidate.len());
        let sentence = candidate[..cut].trim();
        // The pattern already guarantees a subject and a verb, so a short
        // definition like "Zig is a language" is valid here even though generic
        // fragments need a higher bar.
        if sentence.len() >= 15 && !is_navigation_line(sentence) {
            return Some(sentence.to_string());
        }
    }
    None
}

/// Collapses a passage excerpt into at most two concise sentences. The answer
/// body carries no sources, links, or headings beyond the subject.
fn concise_body(candidates: &[(usize, String)], used: &mut Vec<String>) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    for (_, s) in candidates {
        if used.contains(s) || is_redundant(s, used) {
            continue;
        }
        if s.contains("Retrieved") || s.contains("Download") {
            continue;
        }
        used.push(s.clone());
        parts.push(format!("{}.", s.trim_end_matches('.')));
        if parts.len() >= 2 {
            break;
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" "))
    }
}

#[derive(PartialEq)]
enum QueryIntent {
    Definition,
    HowTo,
    Troubleshoot,
    Why,
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

    let intent = if lower_q.starts_with("why ")
        || lower_q.starts_with("why is ")
        || lower_q.starts_with("why does ")
        || lower_q.starts_with("why do ")
        || lower_q.contains("why should")
        || lower_q.contains("reason for")
    {
        QueryIntent::Why
    } else if lower_q.contains("how to")
        || lower_q.contains("how do i")
        || lower_q.contains("example")
    {
        QueryIntent::HowTo
    } else if lower_q.contains("error") || lower_q.contains("failed") || lower_q.contains("fix") {
        QueryIntent::Troubleshoot
    } else if lower_q.contains("what is") || lower_q.contains("define") {
        QueryIntent::Definition
    } else {
        QueryIntent::General
    };

    let mut candidate_sentences: Vec<(usize, String)> = Vec::new();
    let page_text: String = report
        .passages
        .iter()
        .map(|p| format!("{}\n{}\n", p.excerpt, p.snippet))
        .collect();

    for p in &report.passages {
        let full_text = format!("{}\n{}", p.excerpt, p.snippet);

        // Work line by line: collapsing whitespace first glues navigation chrome
        // onto the real sentence and destroys the sentence boundaries.
        let chunks = full_text
            .lines()
            .map(sanitize_text)
            .filter(|line| line.len() >= 20 && !is_navigation_line(line))
            .flat_map(|line| {
                line.split(|c| c == '.' || c == '!' || c == '?')
                    .map(|part| part.to_string())
                    .collect::<Vec<_>>()
            });

        for chunk in chunks {
            let sentence = chunk
                .trim()
                .trim_matches(|c: char| c == '"' || c == '\'' || c == '\u{201c}' || c == '\u{201d}')
                .trim();
            if sentence.len() < 20
                || sentence.len() > 400
                || is_headline_like(sentence)
                || is_navigation_line(sentence)
                || sentence.matches(" \u{b7} ").count() >= 2
                // Prose sentences start with a capital; a lowercase opener is
                // usually the tail of a sentence split on an abbreviation.
                || !sentence.chars().next().is_some_and(|c| c.is_uppercase())
            {
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
                    if lower_s.contains("is a ")
                        || lower_s.contains("is an ")
                        || lower_s.contains("refers to")
                    {
                        score += 5;
                    }
                    if lower_s.contains("used for") || lower_s.contains("designed to") {
                        score += 3;
                    }
                }
                QueryIntent::HowTo => {
                    if lower_s.contains("you can")
                        || lower_s.contains("to do this")
                        || lower_s.contains("example")
                        || lower_s.contains("step")
                    {
                        score += 5;
                    }
                    if sentence.contains("use ") || sentence.contains("run ") {
                        score += 2;
                    }
                }
                QueryIntent::Troubleshoot => {
                    if lower_s.contains("occurs when")
                        || lower_s.contains("caused by")
                        || lower_s.contains("solution")
                        || lower_s.contains("fix")
                    {
                        score += 5;
                    }
                }
                QueryIntent::Why => {
                    if lower_s.contains("because")
                        || lower_s.contains("reason")
                        || lower_s.contains("since ")
                    {
                        score += 5;
                    }
                    if lower_s.contains("designed")
                        || lower_s.contains("aims to")
                        || lower_s.contains("goal")
                        || lower_s.contains("instead of")
                        || lower_s.contains("trade")
                        || lower_s.contains("benefit")
                    {
                        score += 3;
                    }
                }
                QueryIntent::General => {
                    if lower_s.contains("is ")
                        || lower_s.contains("allows")
                        || lower_s.contains("provides")
                    {
                        score += 3;
                    }
                }
            }

            // Penalties
            if lower_s.contains("add to cart")
                || lower_s.contains("buy ")
                || lower_s.contains("sign up")
                || lower_s.contains("log in")
            {
                continue;
            }

            if score > 0 {
                candidate_sentences.push((score, sentence.to_string()));
            }
        }
    }

    candidate_sentences.sort_by_key(|a| std::cmp::Reverse(a.0));

    let mut used_sentences: Vec<String> = Vec::new();

    // A genuine definition beats any scored fragment.
    let definition = find_definition(&page_text, &subject);
    if !matches!(
        intent,
        QueryIntent::Why | QueryIntent::HowTo | QueryIntent::Troubleshoot
    ) && let Some(def) = &definition
    {
        used_sentences.push(def.clone());
        let def_clean = format!("{}.", def.trim_end_matches('.'));
        if let Some(elaboration) = concise_body(&candidate_sentences, &mut used_sentences) {
            return format!("### {subject}\n\n{def_clean} {elaboration}");
        } else {
            return format!("### {subject}\n\n{def_clean}");
        }
    }

    if intent == QueryIntent::Definition {
        let lead_definition = candidate_sentences.iter().find(|(_, s)| {
            let lower = s.to_ascii_lowercase();
            lower.contains(&subject.to_ascii_lowercase())
                && (lower.contains("is a ")
                    || lower.contains("is an ")
                    || lower.contains("refers to"))
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

            let def_text = if clean_def.starts_with(&subject) {
                format!("{clean_def}.")
            } else {
                format!("{subject}: {clean_def}.")
            };
            return format!("### {subject}\n\n{def_text}");
        }
    }

    // Why/How-To/Troubleshooting/General fallback: clean concise paragraph
    let body = concise_body(&candidate_sentences, &mut used_sentences)
        .or_else(|| {
            find_definition(&page_text, &subject)
                .map(|def| format!("{}.", def.trim_end_matches('.')))
        })
        .unwrap_or_else(|| {
            "I found related pages, but they did not contain a concrete answer to this question."
                .to_string()
        });

    format!("### {subject}\n\n{body}")
}

/// Synthesizes a natural, structured NLP response for reading project files.
///
/// Combines file type detection, semantic code overview/synopsis, syntax-highlighted
/// excerpting, and actionable next steps into a polished pair-programmer answer.
pub fn synthesize_read_response(path: &str, raw_content: &str) -> String {
    let clean_content = if raw_content.starts_with("Lines ") {
        raw_content
            .split_once('\n')
            .map_or(raw_content, |(_, rest)| rest)
    } else {
        raw_content
    };
    let lines: Vec<&str> = clean_content.lines().collect();
    let total_lines = lines.len();

    let ext = path.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    let filename = path.rsplit('/').next().unwrap_or(path);

    let (fence_lang, role) = match ext.as_str() {
        "rs" => ("rust", "Rust Source Module"),
        "toml" => ("toml", "Configuration Manifest"),
        "json" => ("json", "JSON Configuration"),
        "yaml" | "yml" => ("yaml", "YAML Specification"),
        "md" => ("markdown", "Markdown Documentation"),
        "ts" => ("typescript", "TypeScript Module"),
        "tsx" => ("tsx", "TypeScript React Component"),
        "js" => ("javascript", "JavaScript Module"),
        "jsx" => ("jsx", "JavaScript React Component"),
        "py" => ("python", "Python Script / Module"),
        "go" => ("go", "Go Source File"),
        "c" | "h" => ("c", "C Source File"),
        "cpp" | "hpp" | "cc" => ("cpp", "C++ Source File"),
        "sh" | "bash" => ("bash", "Shell Script"),
        "sql" => ("sql", "SQL Database Script"),
        "html" => ("html", "HTML Document"),
        "css" => ("css", "Stylesheet"),
        _ => match filename.to_ascii_lowercase().as_str() {
            "dockerfile" => ("dockerfile", "Container Specification"),
            "makefile" => ("makefile", "Build Automation Script"),
            "license" | "license.md" => ("markdown", "Open-Source License"),
            _ => ("text", "Project File"),
        },
    };

    let synopsis = generate_file_synopsis(filename, &ext, &lines);

    let limit = 8000;
    let excerpt_text: String = clean_content.chars().take(limit).collect();
    let is_truncated = clean_content.chars().count() > limit;

    let follow_up = match ext.as_str() {
        "rs" => format!("💡 **Suggested Actions**: Ask `analyze file {path}` for a safety and code health audit, or `where is <symbol> defined?` to trace implementations."),
        "toml" | "json" => "💡 **Suggested Actions**: Ask `inspect project` to view workspace architecture, or `audit dependencies` to check versions.".to_string(),
        "md" => format!("💡 **Suggested Actions**: Ask `search code for <topic>` to find matching code references across the project."),
        "py" | "ts" | "js" | "go" => format!("💡 **Suggested Actions**: Ask `analyze file {path}` to audit function complexity, error handling, and tests."),
        _ => format!("💡 **Suggested Actions**: Ask `analyze file {path}` for a structural review."),
    };

    let mut out = String::new();
    out.push_str(&format!("### 📄 `{path}` ({role})\n\n"));
    if !synopsis.is_empty() {
        out.push_str(&format!("{synopsis}\n\n"));
    }
    out.push_str(&format!("```{fence_lang}\n{excerpt_text}\n```\n\n"));
    if is_truncated {
        out.push_str(&format!(
            "*(Output truncated at 8,000 characters. Total ~{} lines)*\n\n",
            total_lines
        ));
    }
    out.push_str(&follow_up);
    out
}

fn generate_file_synopsis(filename: &str, ext: &str, lines: &[&str]) -> String {
    if lines.is_empty() {
        return "**Overview**: File is empty (0 lines).".to_string();
    }

    match ext {
        "toml" => {
            let mut pkg_name = None;
            let mut pkg_ver = None;
            let mut dep_count = 0;
            let mut in_deps = false;
            for line in lines {
                let trimmed = line.trim();
                if trimmed.starts_with("name =") && pkg_name.is_none() {
                    pkg_name = trimmed
                        .split('=')
                        .nth(1)
                        .map(|s| s.trim().trim_matches('"'));
                } else if trimmed.starts_with("version =") && pkg_ver.is_none() {
                    pkg_ver = trimmed
                        .split('=')
                        .nth(1)
                        .map(|s| s.trim().trim_matches('"'));
                } else if trimmed.starts_with('[') {
                    in_deps = trimmed.contains("dependencies");
                } else if in_deps && trimmed.contains('=') && !trimmed.starts_with('#') {
                    dep_count += 1;
                }
            }
            if let Some(name) = pkg_name {
                let ver_str = pkg_ver.map_or("".into(), |v| format!(" (`v{v}`)"));
                if dep_count > 0 {
                    format!(
                        "**Overview**: Manifest for crate **`{name}`**{ver_str} with {dep_count} declared dependenc{}.",
                        if dep_count == 1 { "y" } else { "ies" }
                    )
                } else {
                    format!("**Overview**: Manifest for crate **`{name}`**{ver_str}.")
                }
            } else if lines.iter().any(|l| l.contains("[workspace]")) {
                "**Overview**: Cargo workspace manifest configuring workspace members and shared configurations.".to_string()
            } else {
                format!(
                    "**Overview**: TOML configuration containing {} lines.",
                    lines.len()
                )
            }
        }
        "json" => {
            if filename == "package.json" {
                let mut name = None;
                let mut ver = None;
                for line in lines {
                    let trimmed = line.trim();
                    if trimmed.starts_with("\"name\":") && name.is_none() {
                        name = trimmed
                            .split(':')
                            .nth(1)
                            .map(|s| s.trim().trim_matches(|c: char| "\", ".contains(c)));
                    } else if trimmed.starts_with("\"version\":") && ver.is_none() {
                        ver = trimmed
                            .split(':')
                            .nth(1)
                            .map(|s| s.trim().trim_matches(|c: char| "\", ".contains(c)));
                    }
                }
                if let Some(n) = name {
                    let v_str = ver.map_or("".into(), |v| format!(" (`v{v}`)"));
                    return format!("**Overview**: Node.js package manifest for **`{n}`**{v_str}.");
                }
            }
            format!(
                "**Overview**: JSON configuration file containing {} lines.",
                lines.len()
            )
        }
        "md" => {
            let title = lines
                .iter()
                .find(|l| l.trim().starts_with("# "))
                .map(|l| l.trim().trim_start_matches('#').trim());
            let desc = lines
                .iter()
                .find(|l| !l.trim().is_empty() && !l.trim().starts_with('#'))
                .map(|l| l.trim());
            match (title, desc) {
                (Some(t), Some(d)) => format!("**Overview**: {t} — {d}"),
                (Some(t), None) => format!("**Overview**: Documentation for {t}."),
                _ => format!(
                    "**Overview**: Markdown documentation containing {} lines.",
                    lines.len()
                ),
            }
        }
        "rs" => {
            let mut fns = Vec::new();
            let mut types = Vec::new();
            for line in lines {
                let trimmed = line.trim();
                if let Some(rest) = trimmed
                    .strip_prefix("pub fn ")
                    .or_else(|| trimmed.strip_prefix("fn "))
                {
                    if let Some(name) = rest.split('(').next() {
                        let clean = name.trim();
                        if !clean.is_empty() && fns.len() < 3 && !fns.contains(&clean) {
                            fns.push(clean);
                        }
                    }
                } else if let Some(rest) = trimmed
                    .strip_prefix("pub struct ")
                    .or_else(|| trimmed.strip_prefix("struct "))
                {
                    if let Some(name) = rest.split_whitespace().next() {
                        let clean = name.trim().trim_end_matches('{').trim_end_matches(';');
                        if !clean.is_empty() && types.len() < 2 && !types.contains(&clean) {
                            types.push(clean);
                        }
                    }
                } else if let Some(rest) = trimmed
                    .strip_prefix("pub enum ")
                    .or_else(|| trimmed.strip_prefix("enum "))
                {
                    if let Some(name) = rest.split_whitespace().next() {
                        let clean = name.trim().trim_end_matches('{');
                        if !clean.is_empty() && types.len() < 2 && !types.contains(&clean) {
                            types.push(clean);
                        }
                    }
                }
            }
            let mut parts = Vec::new();
            if !types.is_empty() {
                let formatted = types
                    .iter()
                    .map(|t| format!("`{t}`"))
                    .collect::<Vec<_>>()
                    .join(", ");
                parts.push(format!("defines types {formatted}"));
            }
            if !fns.is_empty() {
                let formatted = fns
                    .iter()
                    .map(|f| format!("`{f}()`"))
                    .collect::<Vec<_>>()
                    .join(", ");
                parts.push(format!("implements functions {formatted}"));
            }
            if !parts.is_empty() {
                format!(
                    "**Overview**: Rust source (~{} lines) that {}.",
                    lines.len(),
                    parts.join(" and ")
                )
            } else {
                format!(
                    "**Overview**: Rust source containing {} lines.",
                    lines.len()
                )
            }
        }
        "py" => {
            let mut funcs = 0;
            let mut classes = 0;
            for line in lines {
                let trimmed = line.trim();
                if trimmed.starts_with("def ") {
                    funcs += 1;
                }
                if trimmed.starts_with("class ") {
                    classes += 1;
                }
            }
            if classes > 0 || funcs > 0 {
                format!(
                    "**Overview**: Python script containing {classes} class{} and {funcs} function{} (~{} lines).",
                    if classes == 1 { "" } else { "es" },
                    if funcs == 1 { "" } else { "s" },
                    lines.len()
                )
            } else {
                format!(
                    "**Overview**: Python script containing {} lines.",
                    lines.len()
                )
            }
        }
        "ts" | "tsx" | "js" | "jsx" => {
            let mut exports = 0;
            for line in lines {
                if line.trim().starts_with("export ") {
                    exports += 1;
                }
            }
            if exports > 0 {
                format!(
                    "**Overview**: JavaScript/TypeScript module declaring {exports} export{} (~{} lines).",
                    if exports == 1 { "" } else { "s" },
                    lines.len()
                )
            } else {
                format!(
                    "**Overview**: JavaScript/TypeScript module containing {} lines.",
                    lines.len()
                )
            }
        }
        _ => {
            format!(
                "**Overview**: Project file containing {} lines.",
                lines.len()
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nioxs_browser::ResearchPassage;

    fn passage(title: &str, url: &str, excerpt: &str) -> ResearchPassage {
        ResearchPassage {
            url: url.into(),
            title: title.into(),
            excerpt: excerpt.into(),
            snippet: String::new(),
            score: 10,
        }
    }

    /// Reproduces the `why zig?` response from the bug report: search-result
    /// headlines leaking into the answer as quoted bullets with arrows/emoji.
    #[test]
    fn research_answer_is_professional_and_link_free() {
        let report = ResearchReport {
            query: "zig".into(),
            passages: vec![
                passage(
                    "Zig (programming language) - Wikipedia",
                    "https://en.wikipedia.org/wiki/Zig_(programming_language)",
                    "Zig is a general-purpose programming language designed by Andrew Kelley. \n\n\u{2191} \"Why Zig when There is Already C++, D, and Rust? \u{26A1} Zig Programming Language\".",
                ),
                passage(
                    "No surprises on any system: Q&A with Loris Cro of Zig - Stack Overflow",
                    "https://stackoverflow.com/questions/zig-qa",
                    "\u{2191} \"No surprises on any system: Q&A with Loris Cro of Zig - Stack Overflow\".",
                ),
            ],
            sources_visited: vec![],
        };

        let answer = synthesize_research_answer("zig", &report);

        // No raw markdown link syntax may survive: terminal clients print it literally.
        assert!(
            !answer.contains("]("),
            "raw markdown link leaked:\n{answer}"
        );
        // Decorative glyphs and quoted headlines must be stripped.
        assert!(
            !answer.contains('\u{2191}'),
            "arrow glyph leaked:\n{answer}"
        );
        assert!(!answer.contains('\u{26A1}'), "emoji leaked:\n{answer}");
        assert!(
            !answer.contains("\"Why Zig"),
            "quoted headline leaked:\n{answer}"
        );
        // The answer carries no sources or URLs.
        assert!(!answer.contains("http"), "source URL leaked:\n{answer}");
        assert!(
            !answer.contains("Sources:"),
            "source section leaked:\n{answer}"
        );
        // The real definition should lead the answer, and stay concise.
        assert!(answer.starts_with("### Zig"), "missing heading:\n{answer}");
        assert!(
            answer.contains("general-purpose programming language"),
            "definition sentence missing:\n{answer}"
        );
        assert!(
            answer.lines().count() <= 4,
            "answer is not concise:\n{answer}"
        );
    }

    #[test]
    fn why_intent_prefers_reason_sentences() {
        let report = ResearchReport {
            query: "why zig".into(),
            passages: vec![passage(
                "Zig overview",
                "https://example.org/zig",
                "Zig aims to simplify systems programming. It was designed to avoid hidden control flow because implicit behavior makes code harder to reason about in large codebases.",
            )],
            sources_visited: vec![],
        };
        let answer = synthesize_research_answer("why zig", &report);
        assert!(
            answer.contains("designed") || answer.contains("because"),
            "{answer}"
        );
        assert!(!answer.contains("]("), "{answer}");
        assert!(!answer.contains("http"), "{answer}");
        assert!(
            answer.lines().count() <= 4,
            "answer is not concise:\n{answer}"
        );
    }

    /// Real captured text: Wikipedia's skip-link, site notice, and maintenance
    /// banners precede the definition. Previously these were flattened into one
    /// giant "sentence" that became the whole answer.
    #[test]
    fn wikipedia_chrome_does_not_swallow_the_definition() {
        let report = ResearchReport {
            query: "What is zig?".into(),
            passages: vec![passage(
                "Zig (programming language) - Wikipedia",
                "https://en.wikipedia.org/wiki/Zig_(programming_language)",
                "Zig (programming language) \u{2013} Wikipedia Jump to content From Wikipedia, the free encyclopedia General-purpose programming language This article has multiple issues. Find sources: \"Zig\" programming language \u{2013} news \u{b7} newspapers \u{b7} books \u{b7} scholar \u{b7} JSTOR (October 2025) (Learn how and when to remove this message) Some of this article's listed sources may not be reliable.\nZig is a general-purpose programming language designed by Andrew Kelley and first released in 2016.",
            )],
            sources_visited: vec![],
        };

        let answer = synthesize_research_answer("What is zig?", &report);

        assert_eq!(
            answer,
            "### Zig\n\nZig is a general-purpose programming language designed by Andrew Kelley and first released in 2016.",
            "definition was not isolated:\n{answer}"
        );
        for junk in [
            "Jump to content",
            "free encyclopedia",
            "multiple issues",
            "Learn how",
            "\u{b7}",
            "http",
        ] {
            assert!(!answer.contains(junk), "'{junk}' leaked into:\n{answer}");
        }
    }

    #[test]
    fn definition_finder_ignores_mid_word_matches() {
        assert_eq!(
            find_definition("A zigzag pattern is decorative.", "zig"),
            None
        );
        assert_eq!(
            find_definition("Intro text. Zig is a language.", "Zig").as_deref(),
            Some("Zig is a language")
        );
    }

    #[test]
    fn headline_like_fragments_are_rejected() {
        assert!(is_headline_like(
            "Why Zig When There Is Already C Plus Plus"
        ));
        assert!(!is_headline_like(
            "Zig is a general-purpose programming language designed for robustness."
        ));
    }
}
