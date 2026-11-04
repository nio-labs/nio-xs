//! Stream Output Flow Cadence Engine
//!
//! Provides intelligent buffering and rhythm for companion generation:
//! - `Token` (Default): Classic raw token-by-token typewriter stream.
//! - `Burst`: Instant first-token feedback, followed by natural sentence/clause cadence.
//! - `Paragraph`: Buffers until full paragraph boundaries (`\n\n`).

use std::io::Write;

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

pub struct StreamFlow {
    mode: FlowMode,
    buffer: String,
    has_emitted_first: bool,
}

impl StreamFlow {
    pub fn new(mode: FlowMode) -> Self {
        Self {
            mode,
            buffer: String::with_capacity(512),
            has_emitted_first: false,
        }
    }

    pub fn push(&mut self, chunk: &str) {
        match self.mode {
            FlowMode::Token => {
                print!("{chunk}");
                let _ = std::io::stdout().flush();
            }
            FlowMode::Burst => {
                self.buffer.push_str(chunk);

                // Instant first touch: Flush the first 1-2 words immediately to eliminate TTFT wait
                if !self.has_emitted_first {
                    if self.buffer.contains(' ') || self.buffer.contains('\n') || self.buffer.len() >= 12 {
                        print!("{}", self.buffer);
                        let _ = std::io::stdout().flush();
                        self.buffer.clear();
                        self.has_emitted_first = true;
                    }
                    return;
                }

                // Check for natural sentence/clause cadence
                let should_flush = if self.buffer.contains("\n\n")
                    || self.buffer.contains("\n- ")
                    || self.buffer.contains("\n* ")
                    || self.buffer.contains("```\n")
                {
                    true
                } else if self.buffer.contains(". ")
                    || self.buffer.contains(".\n")
                    || self.buffer.contains("? ")
                    || self.buffer.contains("?\n")
                    || self.buffer.contains("!\n")
                    || self.buffer.contains("! ")
                {
                    true
                } else if self.buffer.contains(", ") || self.buffer.contains("; ") || self.buffer.contains(": ") {
                    self.buffer.split_whitespace().count() >= 4
                } else {
                    self.buffer.split_whitespace().count() >= 6 || self.buffer.len() >= 48
                };

                if should_flush {
                    print!("{}", self.buffer);
                    let _ = std::io::stdout().flush();
                    self.buffer.clear();
                }
            }
            FlowMode::Paragraph => {
                self.buffer.push_str(chunk);
                if self.buffer.contains("\n\n") {
                    print!("{}", self.buffer);
                    let _ = std::io::stdout().flush();
                    self.buffer.clear();
                }
            }
        }
    }

    pub fn finish(&mut self) {
        if !self.buffer.is_empty() {
            print!("{}", self.buffer);
            let _ = std::io::stdout().flush();
            self.buffer.clear();
        }
    }
}
