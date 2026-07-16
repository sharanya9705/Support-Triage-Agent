# Multi-Domain Support Triage Agent

A terminal-based agent that triages support tickets across three completely
different product domains (HackerRank, Claude/Anthropic, Visa) using
**BM25 retrieval + an LLM reasoning layer**, with explicit escalation and
injection detection built in from the start — not bolted on after.

Built for HackerRank's Orchestrate hackathon (May 2026). This repo contains
the agent, a representative slice of the support-doc corpus it retrieves
from, and sample tickets to run it against.

## Why this design

Most "AI agent" hackathon submissions wire an LLM straight to an API and
call it done. That falls over the moment a user asks something the system
shouldn't answer on its own — a fraud claim, a legal threat, a request to
override a policy. This agent treats **not answering** as a first-class
outcome, not an afterthought:

1. **Safety layer first.** Every ticket is checked for prompt injection and
   out-of-scope requests *before* retrieval even runs — patterns cover
   several languages, not just English.
2. **BM25 retrieval, not embeddings.** A from-scratch BM25 index (no
   external libraries) over the markdown corpus. Cheaper, more
   interpretable, and a defensible choice for this kind of structured
   support-doc corpus over jumping straight to a vector DB.
3. **Escalation rules as data, not vibes.** A table of (pattern → reason)
   pairs — identity theft, unauthorized charges, security disclosures,
   legal threats — routes straight to a human instead of letting the LLM
   guess. This is the single most important property of a real support
   agent and the thing most hackathon projects skip.
4. **Grounded generation with a hard fallback.** The LLM is instructed to
   answer *only* from retrieved corpus excerpts — no invented policies,
   URLs, or phone numbers. If the API is unavailable, a rule-based fallback
   still produces a structured, sourced answer instead of failing silently.

## Architecture

```
ticket → [injection check] → [out-of-scope check] → [escalation rules]
                                                            │
                                                    BM25 retrieval (top-5)
                                                            │
                                          Claude API (grounded) ──fails──▶ rule-based fallback
                                                            │
                                          structured CSV: status, product_area,
                                          response, justification, request_type
```

## Repo layout

```
agent/main.py       # the agent — corpus loader, BM25, safety checks, LLM call, CLI
data/               # representative sample of the support corpus (hackerrank/claude/visa)
sample_tickets.csv  # example tickets with expected outputs
CHALLENGE.md        # original problem statement
```

> Note: `data/` here is a small representative sample, not the full corpus
> the agent was originally evaluated against — enough to run the demo
> end-to-end without shipping someone else's full scraped dataset.

## Running it

No dependencies beyond the Python standard library — the Anthropic API is
called directly over `urllib`, no SDK required.

```bash
# batch mode — processes every row in sample_tickets.csv
python3 agent/main.py --data data --input sample_tickets.csv --output output.csv

# single ticket
python3 agent/main.py --data data --ticket "I can't log in" --company Claude

# with an API key (falls back to rule-based responses without one)
ANTHROPIC_API_KEY=sk-... python3 agent/main.py --data data --input sample_tickets.csv --output output.csv
```

Output columns: `status` (replied/escalated), `product_area`, `response`,
`justification`, `request_type` (product_issue/feature_request/bug/invalid).

## What I'd build next

- Swap the hand-written escalation pattern list for a small classifier
  trained on labeled escalation examples
- Add a confidence score alongside `status` so borderline "replied" cases
  can still get a light-touch human review
- Multilingual retrieval (BM25 currently tokenizes on a simple regex; a
  language-aware tokenizer would help non-English tickets)
