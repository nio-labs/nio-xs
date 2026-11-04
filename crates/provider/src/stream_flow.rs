//! Stream Output Flow Cadence Engine
//!
//! Provides intelligent buffering and rhythm for companion generation:
//! - `Token` (Default): Classic raw token-by-token typewriter stream.
//! - `Burst`: Instant first-token feedback, followed by natural sentence/clause cadence.
//! - `Paragraph`: Buffers until full paragraph boundaries (`\n\n`).
//! - Formats chain-of-thought `<think>` blocks in dim styling and filters stray tags.

use std::io::{IsTerminal, Write};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FlowMode {
    Burst,
    #[default]
    Token,
    Paragraph,
}

impl FlowMode {
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "burst" | "sentence" | "chunk" => Self::Burst,
            "paragraph" | "block" => Self::Paragraph,
            _ => Self::Token,
        }
    }
}

const THINK_OPEN: &[&str] = &["<think>", "<thought>", "<thinking>"];
const THINK_CLOSE: &[&str] = &["</think>", "</thought>", "</thinking>"];
const STRAY_TAGS: &[&str] = &[
    "think />",
    "<think/>",
    "</think>",
    "<think>",
    "</thought>",
    "<thought>",
    "</thinking>",
    "<thinking>",
];

pub struct StreamFlow {
    mode: FlowMode,
    mode_buffer: String,
    stream_buffer: String,
    has_emitted_first: bool,
    in_think: bool,
    think_completed: bool,
    is_tty: bool,
}

impl StreamFlow {
    pub fn new(mode: FlowMode) -> Self {
        Self {
            mode,
            mode_buffer: String::with_capacity(512),
            stream_buffer: String::with_capacity(512),
            has_emitted_first: false,
            in_think: false,
            think_completed: false,
            is_tty: std::io::stdout().is_terminal(),
        }
    }

    pub fn push(&mut self, chunk: &str) {
        self.stream_buffer.push_str(chunk);
        self.drain_stream();
    }

    fn drain_stream(&mut self) {
        loop {
            if self.stream_buffer.is_empty() {
                break;
            }

            if self.in_think {
                // Look for earliest closing think tag
                let mut earliest_close = None;
                for tag in THINK_CLOSE {
                    if let Some(pos) = self.stream_buffer.find(tag) {
                        match earliest_close {
                            None => earliest_close = Some((pos, tag.len())),
                            Some((min_pos, _)) if pos < min_pos => {
                                earliest_close = Some((pos, tag.len()))
                            }
                            _ => {}
                        }
                    }
                }

                if let Some((pos, tag_len)) = earliest_close {
                    let thought_text = &self.stream_buffer[..pos];
                    if !thought_text.is_empty() {
                        self.emit_raw(thought_text);
                    }
                    if self.is_tty {
                        self.emit_raw("\x1b[0m\r\n\r\n");
                    } else {
                        self.emit_raw("\n\n");
                    }
                    self.stream_buffer.drain(..pos + tag_len);
                    self.in_think = false;
                    self.think_completed = true;
                    continue;
                }

                // Check if buffer ends with a partial prefix of any closing tag
                let hold_back = Self::longest_matching_prefix(&self.stream_buffer, THINK_CLOSE);
                let safe_len = self.stream_buffer.len().saturating_sub(hold_back);
                if safe_len > 0 {
                    let to_emit: String = self.stream_buffer.drain(..safe_len).collect();
                    self.emit_raw(&to_emit);
                }
                break;
            }

            if !self.think_completed {
                let trimmed = self.stream_buffer.trim_start();
                if trimmed.is_empty() {
                    break;
                }

                let mut matched_open = None;
                for tag in THINK_OPEN {
                    if trimmed.starts_with(tag) {
                        matched_open = Some(tag.len());
                        break;
                    }
                }

                if let Some(tag_len) = matched_open {
                    let tag_start = self.stream_buffer.find('<').unwrap_or(0);
                    self.stream_buffer.drain(..tag_start + tag_len);
                    self.in_think = true;
                    if self.is_tty {
                        self.emit_raw("\x1b[2;38;5;244m💭 ");
                    }
                    continue;
                }

                // Check if trimmed might be a prefix of any opening think tag
                let is_prefix = THINK_OPEN.iter().any(|tag| tag.starts_with(trimmed));
                if is_prefix && trimmed.len() < 12 {
                    break;
                }

                // Not starting with a think tag
                self.think_completed = true;
            }

            // Regular post-thought response streaming
            let mut earliest_stray = None;
            for tag in STRAY_TAGS {
                if let Some(pos) = self.stream_buffer.find(tag) {
                    match earliest_stray {
                        None => earliest_stray = Some((pos, tag.len())),
                        Some((min_pos, _)) if pos < min_pos => {
                            earliest_stray = Some((pos, tag.len()))
                        }
                        _ => {}
                    }
                }
            }

            if let Some((pos, tag_len)) = earliest_stray {
                let clean_chunk: String = self.stream_buffer.drain(..pos).collect();
                if !clean_chunk.is_empty() {
                    self.emit_answer_chunk(&clean_chunk);
                }
                self.stream_buffer.drain(..tag_len);
                continue;
            }

            let hold_back = Self::longest_matching_prefix(&self.stream_buffer, STRAY_TAGS);
            let safe_len = self.stream_buffer.len().saturating_sub(hold_back);
            if safe_len > 0 {
                let to_emit: String = self.stream_buffer.drain(..safe_len).collect();
                self.emit_answer_chunk(&to_emit);
            }
            break;
        }
    }

    fn longest_matching_prefix(buffer: &str, candidates: &[&str]) -> usize {
        // Walk backward char-by-char to avoid slicing inside multi-byte UTF-8 sequences.
        let chars: Vec<(usize, char)> = buffer.char_indices().collect();
        let max_chars = chars.len().min(12);
        for n in (1..=max_chars).rev() {
            let byte_start = chars[chars.len() - n].0;
            let suffix = &buffer[byte_start..];
            if candidates.iter().any(|c| c.starts_with(suffix)) {
                return buffer.len() - byte_start;
            }
        }
        0
    }

    fn emit_raw(&self, text: &str) {
        print!("{text}");
        let _ = std::io::stdout().flush();
    }

    fn emit_answer_chunk(&mut self, chunk: &str) {
        match self.mode {
            FlowMode::Token => {
                self.emit_raw(chunk);
            }
            FlowMode::Burst => {
                self.mode_buffer.push_str(chunk);

                if !self.has_emitted_first {
                    if self.mode_buffer.contains(' ')
                        || self.mode_buffer.contains('\n')
                        || self.mode_buffer.len() >= 12
                    {
                        self.emit_raw(&self.mode_buffer);
                        self.mode_buffer.clear();
                        self.has_emitted_first = true;
                    }
                    return;
                }

                let should_flush = if self.mode_buffer.contains("\n\n")
                    || self.mode_buffer.contains("\n- ")
                    || self.mode_buffer.contains("\n* ")
                    || self.mode_buffer.contains("```\n")
                {
                    true
                } else if self.mode_buffer.contains(". ")
                    || self.mode_buffer.contains(".\n")
                    || self.mode_buffer.contains("? ")
                    || self.mode_buffer.contains("?\n")
                    || self.mode_buffer.contains("!\n")
                    || self.mode_buffer.contains("! ")
                {
                    true
                } else if self.mode_buffer.contains(", ")
                    || self.mode_buffer.contains("; ")
                    || self.mode_buffer.contains(": ")
                {
                    self.mode_buffer.split_whitespace().count() >= 4
                } else {
                    self.mode_buffer.split_whitespace().count() >= 6 || self.mode_buffer.len() >= 48
                };

                if should_flush {
                    self.emit_raw(&self.mode_buffer);
                    self.mode_buffer.clear();
                }
            }
            FlowMode::Paragraph => {
                self.mode_buffer.push_str(chunk);
                if self.mode_buffer.contains("\n\n") {
                    self.emit_raw(&self.mode_buffer);
                    self.mode_buffer.clear();
                }
            }
        }
    }

    pub fn finish(&mut self) {
        if self.in_think {
            if self.is_tty {
                self.emit_raw("\x1b[0m\r\n\r\n");
            } else {
                self.emit_raw("\n\n");
            }
            self.in_think = false;
        }

        // Emit any remaining stream buffer as answer
        if !self.stream_buffer.is_empty() {
            let mut remaining = std::mem::take(&mut self.stream_buffer);
            for tag in STRAY_TAGS {
                remaining = remaining.replace(tag, "");
            }
            if !remaining.is_empty() {
                self.emit_answer_chunk(&remaining);
            }
        }

        if !self.mode_buffer.is_empty() {
            self.emit_raw(&self.mode_buffer);
            self.mode_buffer.clear();
        }
    }
}
