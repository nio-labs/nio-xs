# Intent dataset and Phase 1 evaluation plan

Schema version: 1. The seed corpus is `seed.jsonl`; it is illustrative, not enough to train or calibrate a release classifier. Each record has id, family_id, split, prompt, intent, expected_route, and required_fields. Keep intent labels distinct from current implementation support.

Supported future labels: inspect_project, find_code, run_checks, research_sources, compare_packages. Additional labels: clarify, unsupported. Phase 1's read_readme integration fixture is separate; do not claim general project workflows or a learned classifier are delivered.

Stable split: every family belongs wholly to train, validation, or test. No paraphrases from one family may cross splits. Reserve test families before collecting more examples. Include ambiguous references, unavailable tools, multilingual/unknown input, requests for forbidden actions, errors, and unseen project types. Do not put customer/private code into the corpus automatically.

Expand before Phase 2 model acceptance to at least 100 held-out supported requests and 100 held-out unsupported requests. Provisional thresholds: supported intent macro F1 >= 0.90; unsupported requests incorrectly accepted <= 2%; missing required inputs must not dispatch actions. Report denominators, class breakdown, abstentions, and the rule baseline. Revise targets only with recorded evidence, not by excluding difficult examples. These are acceptance targets, not measured accuracy claims.

Phase 1 uses meaningful protocol and workflow tests: actual client tool round trip, matched action IDs, invalid/absent tools, unsupported requests, scoped durable events, idempotency conflict, restart persistence, explicit cancellation, storage outage, and browser challenge fixtures. Live search results are recorded separately and cannot prove permanent search-engine reliability.
