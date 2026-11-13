# Reproducing annotation/scoring components

scripts/score.py accepts references with `entities` objects or `label` triples,
and predictions with `entities`, `label` or `predicted_label`. IDs must align
exactly; row order alone cannot silently substitute for identity. Strict
matching is exact start/end/category. Overlap matching maximizes total positive
character intersection, then match count, one-to-one within category/sentence.
General matching is used for nonflat oracle output; compiler outputs are flat.

```bash
python scripts/score.py --reference recovered/test.jsonl --predictions predictions.jsonl --output work/scores.json
```

The exact E1 compiler, fixed-K voting/gating, oracle replacement and paired
document bootstrap are preserved under llm_study/. scripts/assemble_prompts.py
reconstructs P2 byte-for-byte from its ten validation IDs/offsets and checks its
original SHA-256. No Dev article sentences are embedded in the public recipe.

Qwen3-8B ran GGUF Q4_K_M on CPU/12 threads, context 16384, greedy,
seed 20261112, official nonthinking template. Luna used gpt-6-luna, reasoning
none, temperature 0/top_p 1, no exposed seed. Both cap output at 1024 tokens,
one attempt per sentence; no fuzzy matching/JSON repair. The archived runners
show actual accounting algorithms, not current endpoint/pricing guarantees.
Fresh API use requires new configuration, credentials and cost limits; none
are enabled in this repository. No requests are made by the offline tools.

To rerun full reported analyses requires the matching reference labels,
exact source inputs and original prediction/generation records. Raw text-bearing records and trained weights are not public. Text-free
predictions are included for four LLM runs and twelve Gold/E6 recurrent
evaluations. scripts/reproduce_scores.py recomputes their observed scores
and optional oracle routing;
do not present a synthetic self-check as reproduction of all paper numbers.
