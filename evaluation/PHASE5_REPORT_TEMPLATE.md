# Phase 5 report template — independent paired AXIS vs baseline

> **Status: not run.** This file defines the *structure* and required evidence for
> the Phase 5 independent evaluation. It intentionally contains no results.
> Replace every `<TBD>` only after the run exists; keep the denominators. An empty
> placeholder is not a passing result. Until this report is filled with a
> reproducible run, [§4.5](../docs/PLAN.md#4-five-advancement-priorities) and
> [§10](../docs/PLAN.md#10-acceptance-and-measurement) remain open.

This report measures one thing: whether AXIS improves real coding outcomes over
the **same upstream model, prompt, tools, context, and budget without AXIS**. A
classifier or router macro-F1 score is *not* coding-agent evidence and must be
reported separately from task success.

## 0. Run identity

| Field | Value |
| --- | --- |
| Date (UTC) | `<TBD>` |
| Operator | `<TBD>` |
| AXIS version / commit | `<TBD>` |
| Workspace manifest version | `<TBD>` |
| Upstream model ID(s) | `<TBD>` (exact Kilo `/models` IDs) |
| Nano artifact SHA-256 | `<TBD>` (from `models/nano.json`) |
| Harness version | `<TBD>` |
| Task-set freeze hash | `<TBD>` |

Reproducibility requires the exact version, model ID, artifact digest, and
harness settings above. A run missing any of them is not citable.

## 1. Goal and hypothesis

- **Claim under test:** `<TBD>` (one sentence, falsifiable).
- **Null:** with everything else held equal, AXIS does not improve hidden-check
  completion over the baseline.
- **Falsified by:** `<TBD>` (name the metric and threshold that would refute the claim).

## 2. Task set

Requirements ([PLAN §5](../docs/PLAN.md#5-project-context-and-web-research), [§10](../docs/PLAN.md#10-acceptance-and-measurement)):

- [ ] Tasks come from independent, real repositories — not authored by the same
      synthetic generator that produced classifier fixtures.
- [ ] Family/project-separated splits; no repository appears in both tuning and
      acceptance.
- [ ] Each task has an **independent hidden completion check** (test, diff
      invariant, or oracle) that the assistant cannot see or edit.
- [ ] Nio-owned fresh tasks reduce benchmark familiarity; SWE-bench-style tasks
      are reported as one signal, not the headline.
- [ ] The set was frozen (hash recorded in §0) **before** tuning.

| Task family | Repos | Tasks | Source | Hidden check kind | Frozen? |
| --- | --- | --- | --- | --- | --- |
| `<TBD>` | `<TBD>` | `<TBD>` | `<TBD>` | `<TBD>` | `<TBD>` |

**Exclusions and why:** `<TBD>` (e.g. tasks dropped for missing or flaky checks;
dropping is a result, not a skip).

## 3. Paired design

Each task is run twice with the **only** difference being AXIS on/off.

- [ ] Same task instance, repository revision, and starting state.
- [ ] Same upstream model ID and provider at run time.
- [ ] Same system/user prompt and same offered tool set.
- [ ] Same context budget, tool budget, and wall-clock/token/cost ceiling.
- [ ] Same caller approval policy; approvals pre-authorized identically.
- [ ] Runs paired per task (not aggregate means only); report both members.
- [ ] Run order and any warm-up/cache effects recorded.
- [ ] Each task run on a fresh checkout; no cross-run state leakage.

| Control | Value |
| --- | --- |
| Randomization / ordering | `<TBD>` |
| Retries allowed | `<TBD>` (retries must be counted in the denominator) |
| Cancellation policy | `<TBD>` |
| Judge/checker version | `<TBD>` |

## 4. Metrics and denominators

Report every metric **by task type**, with denominators (n) and failures listed,
not averaged away.

### 4.1 Outcome
- Hidden-check pass rate: AXIS `<TBD>` (n=`<TBD>`) vs baseline `<TBD>` (n=`<TBD>`) → delta `<TBD>`, paired CI `<TBD>`.
- Regression rate after edits (previously passing checks that broke): `<TBD>`.
- Failed checks remaining at turn end: `<TBD>`.
- Unsupported / false completion claims (claimed done, hidden check failed): `<TBD>`.
- False **Go** verdicts at the final gate: `<TBD>`.
- Unsupported requests incorrectly accepted: `<TBD>`.
- Tasks requiring human intervention: `<TBD>`.

### 4.2 Process
- Evidence coverage (claims traceable to returned evidence) / total claims: `<TBD>`.
- Repeated tool calls and unnecessary file reads: `<TBD>` vs `<TBD>`.
- Handoffs per task and wasted handoffs: `<TBD>`.

### 4.3 Cost and resources
- Latency p50 / p90 / p99: `<TBD>`.
- Upstream tokens in/out: `<TBD>`.
- Cost per task family: `<TBD>`.
- Peak memory, startup time, installed size: `<TBD>`.

### 4.4 Calibration and abstention (NioHelm / MicroLM)

Measure confidence quality separately from task success.

- Abstention rate, and accuracy **on non-abstained** decisions (coverage–accuracy
  trade-off): `<TBD>`.
- Probabilities binned (e.g. 10 bins) with observed accuracy per bin — reliability
  table/diagram: `<TBD>`.
- Expected Calibration Error (ECE) and Brier score, vs the rules/score baseline: `<TBD>`.
- Stage-selection and abstention quality **improve over the fixed-weight rules
  baseline on held-out tasks**: `<TBD>`.
- Unauthorized actions did **not** increase: `<TBD>`.

## 5. Security and lifecycle gates

Each is pass/fail with the observed evidence; a green build is not evidence.

- [ ] Denied approvals never execute.
- [ ] Paths stay inside the caller's project.
- [ ] Tool output is treated as untrusted data and cannot change the allowlist.
- [ ] No side effect is automatically replayed after an uncertain outcome.
- [ ] Expired handoff state cannot authorize a new action.
- [ ] Cancellation stops pending work; partial results are labelled.
- [ ] Tool/round limits terminate cleanly with an accurate answer.
- [ ] A provider restart resumes continuation **without replaying tools** (or
      refuses it) — record which, per task.

## 6. Acceptance criteria

The report is accepted only when all of the following hold.

**For the evaluation itself**
- [ ] Every §0 field is filled and the run is reproducible from the recorded inputs.
- [ ] Every §4 metric has a denominator; failures are listed, not omitted.
- [ ] Paired per-task rows are published, not only aggregate means.
- [ ] The claim in §1 is either supported with the recorded evidence or explicitly withdrawn.
- [ ] Limitations in §8 are published alongside any positive result.

**For the NioHelm calibrated-confidence + abstention change**
- [ ] Confidence is **calibrated on held-out data**: report ECE and a reliability
      table against the current fixed-weight heuristic as the baseline.
- [ ] An **explicit abstention path** exists and is exercised: the decision layer
      returns abstain/needs-input rather than guessing when evidence is
      insufficient; the abstention rate is reported.
- [ ] On held-out tasks, stage selection and abstention **improve over the rules
      baseline**, with no increase in unauthorized actions.
- [ ] Abstention and confidence are **not** used as permission: an abstain/Go
      decision cannot grant a tool or path the caller withheld.
- [ ] The heuristic's hand-set confidence weights are **removed or clearly
      labelled as uncalibrated**; no shipped claim describes them as probabilities.
- [ ] The measured change is reproducible from the version/artifact hashes in §0.

## 7. Results

- Paired per-task table: `<TBD>`
- Aggregate by family: `<TBD>`
- Notable regressions and their root cause: `<TBD>`
- Raw artifacts (logs, JSON, diffs) committed under `evaluation/reports/`: `<TBD>`

## 8. Threats to validity and limitations

- Corpus and repository selection bias: `<TBD>`.
- Benchmark familiarity / contamination: `<TBD>`.
- Flaky or under-powered hidden checks: `<TBD>`.
- Judge/checker error (false pass/fail): `<TBD>`.
- Provider/model drift during the run (Kilo catalog changes): `<TBD>`.
- What this report explicitly does **not** show (e.g. parity with large
  generative models, arbitrary-language understanding): `<TBD>`.

## 9. Verdict

`<TBD>` — Go / Conditional Go / No-Go, with the decisive evidence, unmet criteria,
and residual risks. A Go here describes readiness against these criteria only; it
does not replace caller approval or claim unrun checks passed.

---

**Maintenance:** results belong here; goals and acceptance gates belong in
[`docs/PLAN.md`](../docs/PLAN.md). Do not restate the roadmap in this file.
