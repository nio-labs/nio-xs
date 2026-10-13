use serde::{Deserialize, Serialize};
use std::{path::Path, time::Duration};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "snake_case")]
pub enum PageStatus {
    Readable,
    Challenge,
    AccessDenied,
    Empty,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ResearchPassage {
    pub url: String,
    pub title: String,
    pub excerpt: String,
    pub snippet: String,
    pub score: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ResearchReport {
    pub query: String,
    pub passages: Vec<ResearchPassage>,
    pub sources_visited: Vec<String>,
}

pub fn classify(text: &str) -> PageStatus {
    if text.trim().is_empty() {
        return PageStatus::Empty;
    }
    let visible = extract_text(text);
    let lower = visible.to_lowercase();
    let raw_lower = text.to_lowercase();

    let is_challenge = [
        "verify you are human",
        "unusual traffic from your computer",
        "our systems have detected unusual traffic",
        "unfortunately, bots use duckduckgo",
        "select all squares containing",
        "confirm this search was made by a human",
        "security check to access",
        "cf-browser-verification",
    ]
    .iter()
    .any(|s| lower.contains(s))
        || raw_lower.contains("class='g-recaptcha'")
        || raw_lower.contains("class=\"g-recaptcha\"")
        || raw_lower.contains("data-sitekey=")
        || raw_lower.contains("cf-chl-")
        || (lower.contains("captcha")
            && (lower.contains("verify") || lower.contains("enter the characters")));

    if is_challenge {
        PageStatus::Challenge
    } else if [
        "access denied",
        "403 forbidden",
        "sign in to continue",
        "401 unauthorized",
    ]
    .iter()
    .any(|s| lower.contains(s))
        && visible.len() < 1000
    {
        PageStatus::AccessDenied
    } else {
        PageStatus::Readable
    }
}

pub fn decode_entities(text: &str) -> String {
    let mut s = text.to_string();
    for _ in 0..3 {
        let prev = s.clone();
        s = s
            .replace("&quot;", "\"")
            .replace("&#34;", "\"")
            .replace("&apos;", "'")
            .replace("&#39;", "'")
            .replace("&#x27;", "'")
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&nbsp;", " ")
            .replace("&#8217;", "'")
            .replace("&#8216;", "'")
            .replace("&#8220;", "\"")
            .replace("&#8221;", "\"")
            .replace("&#8211;", "-")
            .replace("&#8212;", "—");
        if s == prev {
            break;
        }
    }
    s
}

pub fn clean_title(raw: &str) -> String {
    let decoded = decode_entities(raw);
    let trimmed = decoded.trim();
    if let Some((head, _tail)) = trimmed.rsplit_once(" - ") {
        let tail_lower = _tail.trim().to_ascii_lowercase();
        if tail_lower.contains(".com")
            || tail_lower.contains(".org")
            || tail_lower.contains(".net")
            || tail_lower == "reddit"
            || tail_lower == "wikipedia"
            || tail_lower == "github"
        {
            return head.trim().to_string();
        }
    }
    trimmed.to_string()
}

pub fn is_boilerplate_or_spam(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    let spam_indicators = [
        "infinite loading",
        "click here reload",
        "enable javascript",
        "log into your account",
        "privacy policy",
        "terms and conditions",
        "terms of service",
        "all rights reserved",
        "sign in",
        "editorial policies",
        "skip to content",
        "cookie preferences",
        "feel free to watch! infinite loading",
    ];
    let count = spam_indicators
        .iter()
        .filter(|s| lower.contains(*s))
        .count();
    if count >= 2 || lower.contains("infinite loading") {
        return true;
    }
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    if lines.len() > 6 {
        let short_lines = lines
            .iter()
            .filter(|l| l.split_whitespace().count() <= 3)
            .count();
        if (short_lines as f32) / (lines.len() as f32) > 0.65 {
            return true;
        }
    }
    false
}

pub fn extract_title(html: &str) -> Option<String> {
    let lower = html.to_ascii_lowercase();
    let start = lower.find("<title")?;
    let tag_end = lower[start..].find('>')? + start;
    let end = lower[tag_end..].find("</title>")? + tag_end;
    let title = &html[tag_end + 1..end];
    let cleaned = decode_entities(title).trim().to_string();
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

pub fn extract_text(html: &str) -> String {
    let mut out = String::with_capacity(html.len() / 2);
    let mut in_tag = false;
    let mut tag_buf = String::with_capacity(32);
    let mut skip_content = false;
    let mut skip_tag = String::new();
    let mut last_was_space = true;

    let chars: Vec<char> = html.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if in_tag {
            if c == '>' {
                in_tag = false;
                let tag_lower = tag_buf.trim().to_ascii_lowercase();
                let tag_name = tag_lower
                    .split_whitespace()
                    .next()
                    .unwrap_or("")
                    .trim_start_matches('/');

                if !skip_content
                    && (tag_name == "script"
                        || tag_name == "style"
                        || tag_name == "noscript"
                        || tag_name == "svg"
                        || tag_name == "nav"
                        || tag_name == "footer"
                        || tag_name == "header"
                        || tag_name == "aside")
                {
                    if !tag_buf.starts_with('/') && !tag_buf.ends_with('/') {
                        skip_content = true;
                        skip_tag = tag_name.to_string();
                    }
                } else if skip_content && tag_buf.starts_with('/') && tag_name == skip_tag {
                    skip_content = false;
                    skip_tag.clear();
                } else if !skip_content
                    && matches!(
                        tag_name,
                        "p" | "article" | "section" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6"
                    )
                    && !last_was_space
                {
                    out.push_str("\n\n");
                    last_was_space = true;
                } else if !skip_content
                    && matches!(tag_name, "div" | "li" | "tr" | "br")
                    && !last_was_space
                {
                    out.push('\n');
                    last_was_space = true;
                }
                tag_buf.clear();
            } else {
                tag_buf.push(c);
            }
        } else if c == '<' {
            in_tag = true;
            tag_buf.clear();
        } else if !skip_content {
            if c.is_whitespace() {
                if !last_was_space {
                    out.push(' ');
                    last_was_space = true;
                }
            } else {
                out.push(c);
                last_was_space = false;
            }
        }
        i += 1;
    }

    let decoded = decode_entities(&out);
    let mut cleaned_lines = Vec::new();
    let mut was_blank = true;
    for line in decoded.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if !was_blank {
                cleaned_lines.push("");
                was_blank = true;
            }
        } else {
            cleaned_lines.push(trimmed);
            was_blank = false;
        }
    }
    cleaned_lines.join("\n")
}

pub fn parse_ddg_lite_results(html: &str) -> Vec<SearchResult> {
    let mut results = Vec::new();
    let mut cursor = 0;

    while let Some(link_pos) = html[cursor..].find("class='result-link'") {
        let abs_pos = cursor + link_pos;
        let href_marker = "href=\"";
        let Some(rel_href_start) = html[..abs_pos].rfind(href_marker) else {
            cursor = abs_pos + 18;
            continue;
        };
        let href_start = rel_href_start + href_marker.len();
        let Some(href_len) = html[href_start..].find('"') else {
            cursor = abs_pos + 18;
            continue;
        };
        let url = &html[href_start..href_start + href_len];

        let Some(tag_end) = html[abs_pos..].find('>') else {
            cursor = abs_pos + 18;
            continue;
        };
        let text_start = abs_pos + tag_end + 1;
        let Some(text_end) = html[text_start..].find("</a>") else {
            cursor = abs_pos + 18;
            continue;
        };
        let title_raw = &html[text_start..text_start + text_end];
        let title = extract_text(title_raw);

        let snippet = if let Some(snip_pos) = html[abs_pos..].find("class='result-snippet'") {
            let snip_abs = abs_pos + snip_pos;
            if let Some(snip_tag_end) = html[snip_abs..].find('>') {
                let snip_text_start = snip_abs + snip_tag_end + 1;
                if let Some(snip_text_end) = html[snip_text_start..].find("</td>") {
                    extract_text(&html[snip_text_start..snip_text_start + snip_text_end])
                } else {
                    String::new()
                }
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        if url.starts_with("http") && !url.contains("duckduckgo.com") && !title.is_empty() {
            results.push(SearchResult {
                title,
                url: url.to_string(),
                snippet,
            });
        }

        cursor = abs_pos + 18;
        if results.len() >= 10 {
            break;
        }
    }

    results
}

pub fn parse_ddg_html_results(html: &str) -> Vec<SearchResult> {
    let mut results = Vec::new();
    let blocks = html.split("<div class=\"result results_links results_links_deep");
    for block in blocks.skip(1) {
        let title = if let Some(a_pos) = block.find("class=\"result__a\"") {
            let after_a = &block[a_pos..];
            if let Some(tag_end) = after_a.find('>') {
                let text_start = a_pos + tag_end + 1;
                if let Some(text_end) = block[text_start..].find("</a>") {
                    extract_text(&block[text_start..text_start + text_end])
                } else {
                    String::new()
                }
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        let raw_url = if let Some(u_pos) = block.find("class=\"result__url\"") {
            let after_u = &block[u_pos..];
            if let Some(href_pos) = after_u.find("href=\"") {
                let start = u_pos + href_pos + 6;
                if let Some(end) = block[start..].find('"') {
                    &block[start..start + end]
                } else {
                    ""
                }
            } else {
                ""
            }
        } else {
            ""
        };

        let final_url = if raw_url.contains("uddg=") {
            let encoded = raw_url
                .split("uddg=")
                .nth(1)
                .unwrap_or("")
                .split('&')
                .next()
                .unwrap_or("");
            // Basic URL percent-decoding
            let mut decoded = String::new();
            let bytes = encoded.as_bytes();
            let mut i = 0;
            while i < bytes.len() {
                if bytes[i] == b'%' && i + 2 < bytes.len() {
                    if let Ok(b) = u8::from_str_radix(
                        std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or(""),
                        16,
                    ) {
                        decoded.push(b as char);
                        i += 3;
                        continue;
                    }
                }
                decoded.push(bytes[i] as char);
                i += 1;
            }
            decoded
        } else {
            raw_url.to_string()
        };

        let snippet = if let Some(s_pos) = block.find("class=\"result__snippet\"") {
            let after_s = &block[s_pos..];
            if let Some(tag_end) = after_s.find('>') {
                let text_start = s_pos + tag_end + 1;
                if let Some(text_end) = block[text_start..].find("</a>") {
                    extract_text(&block[text_start..text_start + text_end])
                } else {
                    String::new()
                }
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        if final_url.starts_with("http") && !title.is_empty() {
            results.push(SearchResult {
                title,
                url: final_url,
                snippet,
            });
        }
        if results.len() >= 10 {
            break;
        }
    }
    results
}

pub fn parse_wikipedia_results(html: &str) -> Vec<SearchResult> {
    let mut results = Vec::new();
    let mut cursor = 0;

    while let Some(head_pos) = html[cursor..].find("class=\"mw-search-result-heading\"") {
        let abs_pos = cursor + head_pos;
        let href_marker = "href=\"";
        let Some(rel_href_start) = html[abs_pos..].find(href_marker) else {
            cursor = abs_pos + 30;
            continue;
        };
        let href_start = abs_pos + rel_href_start + href_marker.len();
        let Some(href_len) = html[href_start..].find('"') else {
            cursor = abs_pos + 30;
            continue;
        };
        let rel_url = &html[href_start..href_start + href_len];

        let Some(tag_end) = html[abs_pos + rel_href_start..].find('>') else {
            cursor = abs_pos + 30;
            continue;
        };
        let text_start = abs_pos + rel_href_start + tag_end + 1;
        let Some(text_end) = html[text_start..].find("</a>") else {
            cursor = abs_pos + 30;
            continue;
        };
        let title = extract_text(&html[text_start..text_start + text_end]);

        let snippet = if let Some(snip_pos) = html[text_start..].find("class=\"searchresult\"") {
            let snip_abs = text_start + snip_pos;
            if let Some(snip_tag_end) = html[snip_abs..].find('>') {
                let snip_text_start = snip_abs + snip_tag_end + 1;
                if let Some(snip_text_end) = html[snip_text_start..].find("</div>") {
                    extract_text(&html[snip_text_start..snip_text_start + snip_text_end])
                } else {
                    String::new()
                }
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        let full_url = if rel_url.starts_with('/') {
            format!("https://en.wikipedia.org{rel_url}")
        } else {
            rel_url.to_string()
        };

        if !title.is_empty() {
            results.push(SearchResult {
                title,
                url: full_url,
                snippet,
            });
        }

        cursor = text_start + 10;
        if results.len() >= 10 {
            break;
        }
    }

    results
}

pub const USER_AGENT: &str = "NioNano/0.1.0 (https://github.com/nio-labs/nio-xs; contact@nio.ai)";

pub async fn search_wikipedia(query: &str, limit: usize) -> Result<Vec<SearchResult>, String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(8))
        .user_agent(USER_AGENT)
        .build()
        .map_err(|e| format!("cannot build HTTP client: {e}"))?;

    let response = client
        .get("https://en.wikipedia.org/w/index.php")
        .query(&[("search", query)])
        .send()
        .await
        .map_err(|e| format!("wikipedia search request failed: {e}"))?;

    let html = response
        .text()
        .await
        .map_err(|e| format!("failed reading wikipedia response: {e}"))?;

    let mut results = parse_wikipedia_results(&html);
    results.truncate(limit);
    Ok(results)
}

pub async fn fetch_github_repo(repo: &str) -> Result<String, String> {
    let clean_repo = repo
        .trim_matches(|c: char| "`\"',;!?/".contains(c))
        .trim_start_matches("github.com/")
        .trim_start_matches("https://github.com/");

    let raw_readme_url = format!("https://raw.githubusercontent.com/{clean_repo}/HEAD/README.md");
    if let Ok(text) = fetch_url(&raw_readme_url).await {
        return Ok(format!(
            "GitHub repository README ({clean_repo}):\n\n{text}"
        ));
    }
    let repo_url = format!("https://github.com/{clean_repo}");
    let html = fetch_url(&repo_url).await?;
    let text = extract_text(&html);
    Ok(format!("GitHub repository ({clean_repo}):\n\n{text}"))
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PackageMetadata {
    pub ecosystem: String,
    pub name: String,
    pub latest_version: String,
    pub description: String,
    pub license: Option<String>,
    pub homepage: Option<String>,
    pub repository: Option<String>,
    pub documentation: Option<String>,
}

pub async fn fetch_crate_info(crate_name: &str) -> Result<PackageMetadata, String> {
    let clean = crate_name
        .trim()
        .trim_matches(|c: char| "`\"',;!?".contains(c));
    let url = format!("https://crates.io/api/v1/crates/{clean}");
    let json_text = fetch_url(&url).await?;
    let value: serde_json::Value =
        serde_json::from_str(&json_text).map_err(|e| format!("invalid crates.io response: {e}"))?;

    let krate = &value["crate"];
    if krate.is_null() {
        return Err(format!("crate '{clean}' not found on crates.io"));
    }

    Ok(PackageMetadata {
        ecosystem: "cargo".into(),
        name: krate["name"].as_str().unwrap_or(clean).to_string(),
        latest_version: krate["max_stable_version"]
            .as_str()
            .or_else(|| krate["max_version"].as_str())
            .unwrap_or("unknown")
            .to_string(),
        description: krate["description"].as_str().unwrap_or("").to_string(),
        license: None,
        homepage: krate["homepage"].as_str().map(str::to_string),
        repository: krate["repository"].as_str().map(str::to_string),
        documentation: krate["documentation"].as_str().map(str::to_string),
    })
}

pub async fn fetch_npm_info(pkg_name: &str) -> Result<PackageMetadata, String> {
    let clean = pkg_name
        .trim()
        .trim_matches(|c: char| "`\"',;!?".contains(c));
    let url = format!("https://registry.npmjs.org/{clean}");
    let json_text = fetch_url(&url).await?;
    let value: serde_json::Value =
        serde_json::from_str(&json_text).map_err(|e| format!("invalid npm response: {e}"))?;

    let name = value["name"].as_str().unwrap_or(clean).to_string();
    let latest_version = value["dist-tags"]["latest"]
        .as_str()
        .unwrap_or("unknown")
        .to_string();
    let description = value["description"].as_str().unwrap_or("").to_string();
    let license = value["license"].as_str().map(str::to_string);
    let homepage = value["homepage"].as_str().map(str::to_string);

    Ok(PackageMetadata {
        ecosystem: "npm".into(),
        name,
        latest_version,
        description,
        license,
        homepage,
        repository: None,
        documentation: None,
    })
}

pub async fn fetch_url(url: &str) -> Result<String, String> {
    if !url.starts_with("https://") && !url.starts_with("http://") {
        return Err("fetch only accepts HTTP/HTTPS URLs".into());
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .redirect(reqwest::redirect::Policy::limited(5))
        .user_agent(USER_AGENT)
        .build()
        .map_err(|e| format!("cannot build HTTP client: {e}"))?;

    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;

    if !response.status().is_success() {
        return Err(format!("HTTP error {}", response.status()));
    }

    let text = response
        .text()
        .await
        .map_err(|e| format!("failed reading response: {e}"))?;

    if text.len() > 2 * 1024 * 1024 {
        return Err("page exceeds 2 MiB limit".into());
    }

    Ok(text)
}

/// Backward compatible fetch wrapper that replaces the external Obscura binary with native HTTP.
pub async fn fetch(_legacy_binary: &Path, url: &str) -> Result<String, String> {
    fetch_url(url).await
}

pub async fn search(query: &str, limit: usize) -> Result<Vec<SearchResult>, String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(8))
        .build()
        .map_err(|e| format!("cannot build HTTP client: {e}"))?;

    // Primary: Query DuckDuckGo HTML endpoint using standard browser navigation headers
    let ddg_html_res = client
        .get("https://html.duckduckgo.com/html/")
        .query(&[("q", query)])
        .header("user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
        .header("accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8")
        .header("accept-language", "en-US,en;q=0.9")
        .header("sec-ch-ua", "\"Not/A)Brand\";v=\"8\", \"Chromium\";v=\"126\", \"Google Chrome\";v=\"126\"")
        .header("sec-ch-ua-mobile", "?0")
        .header("sec-ch-ua-platform", "\"Windows\"")
        .header("sec-fetch-dest", "document")
        .header("sec-fetch-mode", "navigate")
        .header("sec-fetch-site", "none")
        .header("sec-fetch-user", "?1")
        .header("upgrade-insecure-requests", "1")
        .send()
        .await;

    if let Ok(response) = ddg_html_res {
        if let Ok(html) = response.text().await {
            let status = classify(&html);
            if status != PageStatus::Challenge && status != PageStatus::AccessDenied {
                let mut results = parse_ddg_html_results(&html);
                if !results.is_empty() {
                    results.truncate(limit);
                    return Ok(results);
                }
            }
        }
    }

    // Secondary: Query DDG Lite
    let ddg_lite_res = client
        .post("https://lite.duckduckgo.com/lite/")
        .form(&[("q", query)])
        .header("user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
        .send()
        .await;

    if let Ok(response) = ddg_lite_res {
        if let Ok(html) = response.text().await {
            let status = classify(&html);
            if status != PageStatus::Challenge && status != PageStatus::AccessDenied {
                let mut results = parse_ddg_lite_results(&html);
                if !results.is_empty() {
                    results.truncate(limit);
                    return Ok(results);
                }
            }
        }
    }

    // Tertiary Fallback: Search Wikipedia directly
    let wiki_res = search_wikipedia(query, limit).await;
    if let Ok(results) = wiki_res
        && !results.is_empty()
    {
        return Ok(results);
    }

    Err("search engine challenge encountered".into())
}

pub fn extract_passages(text: &str, query: &str) -> Vec<(usize, String)> {
    let terms: Vec<String> = query
        .to_ascii_lowercase()
        .split_whitespace()
        .filter(|t| t.len() > 2)
        .map(|t| t.to_string())
        .collect();

    let paragraphs: Vec<&str> = text.split("\n\n").collect();
    let mut scored: Vec<(usize, String)> = Vec::new();

    for p in paragraphs {
        let cleaned = p.trim();
        if cleaned.len() < 40 || is_boilerplate_or_spam(cleaned) {
            continue;
        }
        let lower = cleaned.to_ascii_lowercase();
        let mut score = 0;
        for t in &terms {
            score += lower.matches(t).count();
        }
        if score > 0 {
            let excerpt: String = cleaned.chars().take(800).collect();
            scored.push((score, excerpt));
        }
    }

    scored.sort_by_key(|a| std::cmp::Reverse(a.0));
    scored
}

pub async fn research(query: &str, max_pages: usize) -> Result<ResearchReport, String> {
    use futures_util::stream::{self, StreamExt};

    let candidates = search(query, max_pages.max(1)).await?;
    let targets: Vec<SearchResult> = candidates.into_iter().take(max_pages).collect();
    let sources_visited: Vec<String> = targets.iter().map(|t| t.url.clone()).collect();

    // Concurrently fetch candidate pages with a concurrency limit of 2 parallel requests
    let fetched_pages: Vec<(SearchResult, Option<String>)> = stream::iter(targets)
        .map(|item| async move {
            let page_text = match fetch_url(&item.url).await {
                Ok(raw_html) if classify(&raw_html) == PageStatus::Readable => {
                    Some(extract_text(&raw_html))
                }
                _ => None,
            };
            (item, page_text)
        })
        .buffer_unordered(2)
        .collect()
        .await;

    let mut passages = Vec::new();
    let terms: Vec<String> = query
        .to_ascii_lowercase()
        .split_whitespace()
        .filter(|t| t.len() > 2)
        .map(|t| t.to_string())
        .collect();

    for (item, maybe_text) in fetched_pages {
        let mut added_body_passage = false;
        let cleaned_title = clean_title(&item.title);
        let cleaned_snippet = decode_entities(&item.snippet);

        if let Some(text) = maybe_text {
            let top_passages = extract_passages(&text, query);
            for (score, excerpt) in top_passages.into_iter().take(2) {
                if !is_boilerplate_or_spam(&excerpt) {
                    passages.push(ResearchPassage {
                        url: item.url.clone(),
                        title: cleaned_title.clone(),
                        snippet: cleaned_snippet.clone(),
                        excerpt,
                        score: score * 2,
                    });
                    added_body_passage = true;
                }
            }
        }
        // If the fetched page was SPA, blocked, or pure boilerplate, use the search snippet!
        if !added_body_passage
            && !cleaned_snippet.trim().is_empty()
            && !is_boilerplate_or_spam(&cleaned_snippet)
        {
            let snip_lower = cleaned_snippet.to_ascii_lowercase();
            let mut score = 1;
            for t in &terms {
                score += snip_lower.matches(t).count();
            }
            passages.push(ResearchPassage {
                url: item.url.clone(),
                title: cleaned_title,
                snippet: cleaned_snippet.clone(),
                excerpt: cleaned_snippet,
                score,
            });
        }
    }

    passages.sort_by_key(|a| std::cmp::Reverse(a.score));

    Ok(ResearchReport {
        query: query.to_string(),
        passages,
        sources_visited,
    })
}

/// Run 2 research queries in parallel (e.g. comparing two packages or topics concurrently).
pub async fn research_dual(
    query_a: &str,
    query_b: &str,
    pages_per_query: usize,
) -> (
    Result<ResearchReport, String>,
    Result<ResearchReport, String>,
) {
    tokio::join!(
        research(query_a, pages_per_query),
        research(query_b, pages_per_query)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocked_pages_are_not_evidence() {
        assert_eq!(
            classify("<form class='g-recaptcha'>Verify you are human</form>"),
            PageStatus::Challenge
        );
        assert_eq!(classify("403 Forbidden"), PageStatus::AccessDenied);
        assert_eq!(classify("   "), PageStatus::Empty);
        assert_eq!(
            classify("<h1>Project documentation</h1>"),
            PageStatus::Readable
        );
    }

    #[test]
    fn html_extraction_strips_scripts_and_tags() {
        let html = r#"
            <html>
                <head><title>Test Title</title></head>
                <body>
                    <script>var x = 10;</script>
                    <h1>Header One</h1>
                    <p>This is a <b>paragraph</b> with &amp; entity.</p>
                </body>
            </html>
        "#;
        assert_eq!(extract_title(html), Some("Test Title".to_string()));
        let text = extract_text(html);
        assert!(text.contains("Header One"));
        assert!(text.contains("This is a paragraph with & entity."));
        assert!(!text.contains("var x = 10"));
    }

    #[test]
    fn ddg_lite_parser_finds_results() {
        let sample = r#"
            <table>
                <tr>
                    <td>
                        <a rel="nofollow" href="https://example.org/rust" class='result-link'>Rust Language</a>
                    </td>
                </tr>
                <tr>
                    <td class='result-snippet'>
                        Modern systems language focusing on safety.
                    </td>
                </tr>
            </table>
        "#;
        let res = parse_ddg_lite_results(sample);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].title, "Rust Language");
        assert_eq!(res[0].url, "https://example.org/rust");
        assert!(res[0].snippet.contains("Modern systems language"));
    }

    #[test]
    fn test_clean_title_and_entities() {
        assert_eq!(
            clean_title("What Is Kisskh? Is It Legal? A Simple Breakdown - iemlabs.com"),
            "What Is Kisskh? Is It Legal? A Simple Breakdown"
        );
        assert_eq!(
            decode_entities("kisskh | Asian Dramas &amp; Movies"),
            "kisskh | Asian Dramas & Movies"
        );
    }

    #[test]
    fn test_offline_doc_cache() {
        let temp_dir = std::env::temp_dir().join(format!("nioxs-doc-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp_dir);
        let entry = DocCacheEntry {
            package: "tokio".into(),
            ecosystem: "cargo".into(),
            version: "1.38.0".into(),
            summary: "Asynchronous runtime for Rust".into(),
            documentation_md: "## Tokio Runtime\nProvides multi-threaded event loop.".into(),
            timestamp_unix: 1700000000,
        };
        assert!(save_doc_cache(&temp_dir, &entry).is_ok());

        let loaded = load_doc_cache(&temp_dir, "tokio", "cargo").expect("loaded doc");
        assert_eq!(loaded.version, "1.38.0");
        assert!(loaded.documentation_md.contains("Tokio Runtime"));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}

/// Structured documentation entry stored in local cache.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DocCacheEntry {
    pub package: String,
    pub ecosystem: String, // cargo, npm, pypi, go
    pub version: String,
    pub summary: String,
    pub documentation_md: String,
    pub timestamp_unix: u64,
}

/// Saves documentation to `.nioxs/docs/<ecosystem>/<package>.json`.
pub fn save_doc_cache(root: &Path, entry: &DocCacheEntry) -> Result<(), String> {
    let dir = root.join(".nioxs/docs").join(&entry.ecosystem);
    let _ = std::fs::create_dir_all(&dir);
    let file = dir.join(format!("{}.json", entry.package.to_ascii_lowercase()));
    let data = serde_json::to_string_pretty(entry)
        .map_err(|e| format!("cannot serialize doc entry: {e}"))?;
    std::fs::write(&file, data)
        .map_err(|e| format!("cannot write doc entry to {}: {e}", file.display()))
}

/// Loads documentation from `.nioxs/docs/<ecosystem>/<package>.json` if available.
pub fn load_doc_cache(root: &Path, package: &str, ecosystem: &str) -> Option<DocCacheEntry> {
    let file = root
        .join(".nioxs/docs")
        .join(ecosystem)
        .join(format!("{}.json", package.to_ascii_lowercase()));
    let data = std::fs::read_to_string(&file).ok()?;
    serde_json::from_str(&data).ok()
}
