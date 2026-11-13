# Training source and maintained inference

`original_server/` preserves the executed server algorithms. Training uses
Linux (fcntl file locks), Python 3.12, GPU/FP32, PhoBERT-base and VnCoreNLP-1.2.
It requires exact reconstructed text, the preserved PhoBERT assets/conversion,
the declared Java/VnCoreNLP files and a fresh project manifest/cache. These
large/private resources are not bundled. No from-scratch training was rerun
during publication preparation; exact numeric equivalence is not promised.

Current paper settings: seeds 42/999/2026, BiLSTM and BiLSTM-CRF, 20 complete
epochs, batch 8, 128 hidden units per direction, dropout .2; embeddings/lower
8 encoder layers frozen, upper 4 trained. AdamW encoder/head LR 3e-5/5e-4,
weight decay .01, 10% warmup/linear decay, gradient clip 1.0. Select maximum
original Dev strict F1 (earliest tie), not Gold. Original selected epochs:
BiLSTM 9/6/11, CRF 11/6/11. The archived `protocol.json`/`evaluate` retain
historical epoch-20/nine-run wording; consult configs/paper_protocol.json for
the v9 primary reporting rule. Training already saves best_checkpoint.pt.

The repository includes the full original core training/preprocessing/model
source for inspection and adaptation. It is not the private server's source
snapshots, model cache or complete run directory. Original manifests are not
forged to claim that a different machine has the historical files.

## Existing private weight inference

Install requirements-model.txt using the correct PyTorch wheel for your device.
For raw text also install requirements-segmentation.txt and Java/VnCoreNLP.
Weights must be obtained separately; inference/checkpoint_map.json specifies
their exact hashes. Obtain the matching tokenizer files and verify against
inference/assets/tokenizer_requirements.json, placing them in
inference/assets/tokenizer/. VnCoreNLP file hashes/official URLs are recorded
in inference/assets/vncorenlp_assets.json; upstream terms apply.

```bash
python inference/infer.py --run bilstm_crf_seed42 --weights-archive /path/to/essential_checks_20261111_weights.zip --prepared /path/to/prepared_cache.jsonl --output work/predictions.jsonl
python inference/infer.py --run bilstm_crf_seed42 --weights /path/to/bilstm_crf_seed42_dev_best.safetensors --input /path/to/exact_text.jsonl --vncorenlp /path/to/vncorenlp --output work/new_predictions.jsonl
```

These commands are instructions, not work performed during release. Prepared
cache input avoids Java but must follow the preserved source-unit schema.
Raw text uses exact VnCoreNLP and original-character alignment without rewriting.
Gold inference was CPU; kernels/devices can differ. Preserve original results.
