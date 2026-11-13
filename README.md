# Vietnamese Disaster Span Corpus: paper-v9 research code

Companion source and text-free predictions for **A Ten-Category Vietnamese Disaster Span Corpus with
LLM-Assisted Annotation**. This is a public-inspection preparation; software
licensing remains undecided (LICENSE_STATUS.md). It makes no sole-author claim.

Dataset: https://huggingface.co/datasets/Ahzy181/vietnamese-disaster-span-corpus-standoff

The dataset distributes stand-off annotations/metadata, not article sentences.
Frozen core splits are 9,686 train / 803 validation / 1,000 Gold; E6 is a separate
300-row, two-episode reference. Exact historical source documents are a real
dependency: complete live-URL text recovery has not been demonstrated. Do not
describe this release as a fully downloadable text training corpus.

## Start with offline verification

Python 3.10+; these commands make no API calls and perform no training:

```bash
python scripts/validate_dataset.py --dataset /path/to/dataset_repo
python -m unittest discover -s tests -v
```

## Reproduce reported scores without article text

```bash
python scripts/reproduce_scores.py --dataset /path/to/dataset_repo --output work/reproduced_scores.json
# Optional: the full oracle/random routing calculation, still offline:
python scripts/reproduce_scores.py --dataset /path/to/dataset_repo --oracle --output work/reproduced_oracle.json
```

The included 16 prediction files contain IDs, hashes and character-offset labels,
without sentences or raw LLM responses. They allow score recomputation on the
preserved outputs. This is not rerunning training or LLM generation.

## Recover exact input text

Provide lawfully obtained source records with document_id and the exact stored
text field listed in dataset metadata. A changed article or a URL alone is not
sufficient. Hash mismatches fail; no approximate text replacement is allowed.

```bash
python scripts/rehydrate_standoff.py --dataset /path/to/dataset_repo --documents /path/to/source_documents.jsonl --output work/reconstructed --report work/recovery.json
python scripts/assemble_prompts.py --validation work/reconstructed/validation.jsonl --output work/prompts
```

For E6 provide document records with `document_id` and the exact saved `body`
field, then add `--extra` to the reconstruction command and use a new output
folder. Neither source file is distributed by this repository.

## Code map

| Folder | Purpose |
|---|---|
| scripts/ | Validation, exact reconstruction, prompt assembly and scoring CLI |
| original_server/ | Original returned training/model/alignment source and historical configuration |
| inference/ | Maintained source-unit inference, exact weight hashes/configs; weights not bundled |
| llm_study/ | Actual no-repair compiler, strict/overlap scorer, general oracle matcher, gating/bootstrap algorithms |
| archived_runners/ | Executed Qwen/Luna runner source for inspection; no live launch configuration |
| prompts/ | P1 guideline and a text-free recipe reconstructing exact P2 from ten Dev examples |
| results/ | Aggregate measured tables and common error taxonomy; no raw article/prediction text |
| configs/ | Current paper protocol, distinguishing dev-best selection from old epoch-20 history |

See docs/TRAINING_AND_INFERENCE.md, docs/EXPERIMENTS.md and docs/PIPELINE.md.
The original server CLI's `evaluate` targets an older nine-model epoch-20
protocol; it is **not** the v9 primary evaluation. Do not run it as a substitute
for development-best checkpoint evaluation.

## Scientific conditions

Original A1/A2 agreement is 49.77% strict / 70.14% overlap entity F1. Gold has
960 normalized unique texts and prior evaluation/convention exposure. E4 is
assisted-reference consistency (98.65%), not independent accuracy. E4/E6 merges
were assistant-curated after assisted A and repaired B, then curator-accepted;
independent pre-merge IAA is unavailable. E6 separation is curator-confirmed
for two episodes. E2 is a reference-oracle simulation, not measured human gain.
LLM generations are single runs, one quantized CPU model and a dated API alias.
Measured token-rate costs are not billing receipts or human labor costs.

No install command downloads weights, runs training or starts API generation.
Fine-tuned weights remain private. No raw review sessions, API keys, source
snapshots, logs, caches or deferred 1,563 rows are included.
