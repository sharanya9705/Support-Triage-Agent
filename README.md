# Agent

## Setup

No install needed — pure standard library. Optionally set an API key:

```bash
export ANTHROPIC_API_KEY=your_key_here
```

Without a key, the agent falls back to rule-based responses grounded in
the retrieved corpus docs (see `fallback_response` in `main.py`).

## Run

```bash
python3 main.py --data ../data --input ../sample_tickets.csv --output ../output.csv
```

Single ticket:

```bash
python3 main.py --data ../data --ticket "I can't login" --company Claude
```

## Approach

1. Load all markdown docs from `data/hackerrank`, `data/claude`, `data/visa`
2. Build a from-scratch BM25 index over the corpus
3. For each ticket: detect injections, out-of-scope requests, and
   high-risk escalation triggers
4. Retrieve the top-5 relevant docs via BM25
5. Generate a grounded response via the Claude API (falls back to
   rule-based if no key is set)
6. Output a structured CSV: `status`, `product_area`, `response`,
   `justification`, `request_type`
