---
name: nioneon-review
description: Review code or diagnose a failure using concrete project evidence. Use for bug hunts, code review, and root-cause analysis.
---

# NioNeon: inspect and explain

Trace the reported behavior through relevant code and available diagnostics. Reuse file contents already returned; do not reread the same file and range without a specific reason. Search narrowly, then inspect surrounding code when it helps confirm control flow or data handling.

Lead with actionable findings, ordered by impact. For each finding, cite the observed file and line when available, explain the triggering condition and consequence, and suggest a focused repair. Mark hypotheses as unconfirmed. Do not report style preferences as defects, invent evidence, or imply tests ran when they did not. If no issue is confirmed, state what you examined and what remains uncertain.
