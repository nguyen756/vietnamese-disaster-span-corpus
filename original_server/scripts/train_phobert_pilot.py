"""PhoBERT + source-unit BiLSTM NER on the frozen CRF pilot article split.

Offline CPU experiment. Fine-tune upper encoder layers and a sequence head;
retain all source offsets, all labels and every held-out sentence. Evaluate a
fixed final epoch only. Gold Test never enters the model or checkpoint selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

from phobert_ner_data import official_piece_ids, prepare
from train_crf_pilot import ROOT, decode, error_analysis, normalize, read_jsonl, score, sha, write_json, write_jsonl


def model_class():
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    class PhoBertNER(nn.Module):
        def __init__(self, encoder, labels, hidden=128, dropout=0.2):
            super().__init__()
            self.encoder = encoder
            self.dropout = nn.Dropout(dropout)
            self.sequence = nn.LSTM(encoder.config.hidden_size, hidden, batch_first=True, bidirectional=True)
            self.classifier = nn.Linear(hidden * 2, len(labels))

        def forward(self, batch):
            hidden = self.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
            flat = hidden.reshape(-1, hidden.shape[-1])
            selected = flat.index_select(0, batch["pool_src"]) * batch["pool_weight"].unsqueeze(1)
            units = hidden.new_zeros((batch["batch_size"] * batch["max_units"], hidden.shape[-1]))
            units.index_add_(0, batch["pool_dst"], selected)
            units = units.reshape(batch["batch_size"], batch["max_units"], hidden.shape[-1])
            packed = pack_padded_sequence(self.dropout(units), batch["lengths"], batch_first=True, enforce_sorted=False)
            output, _ = self.sequence(packed)
            output, _ = pad_packed_sequence(output, batch_first=True, total_length=batch["max_units"])
            return self.classifier(self.dropout(output))

    return PhoBertNER


def collate(items, label_to_id):
    import torch
    max_pieces = max(len(chunk["input_ids"]) for item in items for chunk in item["chunks"])
    max_units = max(len(item["units"]) for item in items)
    chunks = [chunk for item in items for chunk in item["chunks"]]
    input_ids = torch.ones((len(chunks), max_pieces), dtype=torch.long)
    attention = torch.zeros_like(input_ids)
    targets = torch.full((len(items), max_units), -100, dtype=torch.long)
    sources, destinations, weights, chunk_base = [], [], [], 0
    for ci, chunk in enumerate(chunks):
        length = len(chunk["input_ids"])
        input_ids[ci, :length] = torch.tensor(chunk["input_ids"])
        attention[ci, :length] = 1
    for bi, item in enumerate(items):
        if item["tags"]:
            targets[bi, :len(item["tags"])] = torch.tensor([label_to_id[tag] for tag in item["tags"]])
        for ci, pi, ui, weight in item["pooling_pairs"]:
            sources.append((chunk_base + ci) * max_pieces + pi)
            destinations.append(bi * max_units + ui)
            weights.append(weight)
        chunk_base += len(item["chunks"])
    return {"input_ids": input_ids, "attention_mask": attention, "targets": targets,
            "pool_src": torch.tensor(sources, dtype=torch.long),
            "pool_dst": torch.tensor(destinations, dtype=torch.long),
            "pool_weight": torch.tensor(weights, dtype=torch.float32),
            "lengths": torch.tensor([len(item["units"]) for item in items], dtype=torch.long),
            "batch_size": len(items), "max_units": max_units}


def bucket_batches(items, batch_size, seed):
    """Shuffle text-length buckets; no labels or validation metrics are consulted."""
    rng = random.Random(seed)
    order = list(range(len(items)))
    rng.shuffle(order)
    batches = []
    for first in range(0, len(order), batch_size * 16):
        bucket = sorted(order[first:first + batch_size * 16], key=lambda i: items[i]["piece_count"])
        batches.extend(bucket[j:j + batch_size] for j in range(0, len(bucket), batch_size))
    rng.shuffle(batches)
    assert sorted(i for batch in batches for i in batch) == list(range(len(items)))
    return batches


def predict(model, prepared, labels, batch_size=8):
    import torch
    mapping = {tag: i for i, tag in enumerate(labels)}
    predictions = [None] * len(prepared)
    model.eval()
    start = time.perf_counter()
    batches = bucket_batches(prepared, batch_size, 0)
    with torch.inference_mode():
        for step, indices in enumerate(batches, 1):
            items = [prepared[i] for i in indices]
            batch = collate(items, mapping)
            tags = model(batch).argmax(-1).tolist()
            for index, item, sequence in zip(indices, items, tags):
                predictions[index] = decode(item["units"], [labels[tag] for tag in sequence[:len(item["units"])]])
            if step % 25 == 0 or step == len(batches):
                print(json.dumps({"phase": "inference", "batch": step, "batches": len(batches),
                                  "elapsed_seconds": round(time.perf_counter() - start, 1)}), flush=True)
    assert all(item is not None for item in predictions)
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, default=ROOT / "experiments/crf_pilot_0001")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/phobert_pilot_0001")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--train-layers", type=int, default=4)
    parser.add_argument("--encoder-lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from transformers import PhobertTokenizer, RobertaConfig, RobertaModel
    from underthesea import word_tokenize

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output, split = args.output.resolve(), args.split.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "experiment_config.json").exists():
        raise ValueError("Use a new experiment directory to preserve the saved run")
    protected = [ROOT / "data/train_agent_annotated.jsonl", ROOT / "data/gold_test_final.jsonl",
                 ROOT / "annotation_work/provenance.jsonl", ROOT / "documentation/annotation_guideline.md",
                 split / "train.jsonl", split / "validation.jsonl", split / "split_manifest.json"]
    hashes = {str(path): sha(path) for path in protected}
    prior = json.loads((split / "results.json").read_text(encoding="utf-8"))
    assert sha(protected[0]) == prior["input_sha256"]["data/train_agent_annotated.jsonl"]
    train, validation = read_jsonl(split / "train.jsonl"), read_jsonl(split / "validation.jsonl")
    assert len(train) == prior["split"]["training_rows"] and len(validation) == prior["split"]["validation_rows"]
    assert not {r["document_id"] for r in train} & {r["document_id"] for r in validation}
    assert not {normalize(r["text"]) for r in train} & {normalize(r["text"]) for r in validation}
    entity_labels = list(prior["learning_curve"][-1]["validation"]["per_label"])
    labels = ["O"] + [prefix + label for label in entity_labels for prefix in ["B-", "I-"]]
    label_to_id = {tag: i for i, tag in enumerate(labels)}
    cache = Path.home() / ".cache/huggingface/hub/models--vinai--phobert-base/snapshots"
    tokenizer_source = cache / "c1e37c5c86f918761049cef6fa216b4779d0d01d"
    weights_source = cache / "92f45c73e86450797c850a26ba9410a95dd7d897"
    tokenizer = Tokenizer.from_file(str(tokenizer_source / "tokenizer.json"))
    official = PhobertTokenizer(vocab_file=str(tokenizer_source / "vocab.txt"), merges_file=str(tokenizer_source / "bpe.codes"))
    for text in ["Mưa lớn gây lũ_quét tại Nghệ_An .", "ngập dưới 1 m", "nhiều tuyến giao_thông , cơ_sở hạ_tầng bị hư_hại"]:
        assert official.encode(text) == tokenizer.encode(text).ids, "Fast/official PhoBERT token IDs disagree"
    config = {"architecture": "PhoBERT-base + source-unit bidirectional LSTM (hidden 128) + BIO classifier",
              "encoder": "vinai/phobert-base", "weights_snapshot": weights_source.name,
              "tokenizer_snapshot": tokenizer_source.name, "device": "cpu", "threads": args.threads,
              "word_segmenter": "underthesea.word_tokenize", "source_offsets_preserved": True,
              "upper_encoder_layers_trained": args.train_layers, "epochs": args.epochs,
              "batch_size": args.batch_size, "encoder_learning_rate": args.encoder_lr,
              "head_learning_rate": args.head_lr, "weight_decay": 0.01, "warmup_fraction": 0.1,
              "seed": args.seed, "dropout": 0.2, "lstm_hidden_per_direction": 128,
              "max_encoder_subwords": 254, "overlap_stride": 192,
              "checkpoint_selection": "Fixed final epoch; no validation-based selection or tuning",
              "labels": labels, "source_split": str(split), "protected_input_sha256": hashes,
              "base_weights_sha256": sha(weights_source / "model.safetensors"),
              "versions": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "tokenizers", "underthesea", "safetensors"]}}
    write_json(output / "experiment_config.json", config)
    shutil.copyfile(split / "split_manifest.json", output / "split_manifest.json")
    tokenizer.save(str(output / "tokenizer.json"))
    preparation_start = time.perf_counter()
    prepared = []
    for i, row in enumerate(train + validation, 1):
        item = prepare(row, word_tokenize, tokenizer)
        assert decode(item["units"], item["tags"]) == row["label"]
        assert official.encode(item["segmented_text"], add_special_tokens=False) == official_piece_ids(
            tokenizer.encode(item["segmented_text"], add_special_tokens=False).ids), "Official token ID parity failed"
        assert all(0 <= token_id < 64001 for chunk in item["chunks"] for token_id in chunk["input_ids"])
        prepared.append(item)
        if i % 200 == 0:
            print(json.dumps({"phase": "preprocessing", "rows": i, "total": len(train) + len(validation)}), flush=True)
    prepared_train, prepared_validation = prepared[:len(train)], prepared[len(train):]
    stats = {"training_rows": len(train), "validation_rows": len(validation),
             "segmentation_fallback_rows": [r["source_row"] for r in prepared if r["segmentation_fallback"]],
             "total_source_units": sum(len(r["units"]) for r in prepared),
             "total_encoder_pieces": sum(r["piece_count"] for r in prepared),
             "segmentation_marker_only_pieces": sum(r["marker_only_pieces"] for r in prepared),
             "converted_merge_entries_mapped_to_unk": sum(r["converted_merge_entries_mapped_to_unk"] for r in prepared),
             "official_phobert_token_ids_match_on_all_rows": True,
             "max_encoder_pieces_per_sentence": max(r["piece_count"] for r in prepared),
             "multi_chunk_rows": sum(len(r["chunks"]) > 1 for r in prepared),
             "all_source_units_covered": True, "all_gold_spans_round_trip_exactly": True,
             "preprocessing_seconds": time.perf_counter() - preparation_start}
    write_json(output / "preprocessing_report.json", stats)
    write_jsonl(output / "prepared_inputs.jsonl", prepared)
    print(json.dumps(stats), flush=True)
    encoder_config = RobertaConfig.from_pretrained(str(tokenizer_source), local_files_only=True)
    encoder_config.save_pretrained(output / "encoder_config")
    encoder = RobertaModel.from_pretrained(str(weights_source), config=encoder_config,
                                          add_pooling_layer=False, local_files_only=True, attn_implementation="sdpa")
    assert 0 < args.train_layers <= len(encoder.encoder.layer)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    for layer in encoder.encoder.layer[-args.train_layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
    model = model_class()(encoder, labels)
    encoder_params = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
    head_params = list(model.sequence.parameters()) + list(model.classifier.parameters())
    config["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config["total_parameters"] = sum(p.numel() for p in model.parameters())
    write_json(output / "experiment_config.json", config)
    optimizer = torch.optim.AdamW([{"params": encoder_params, "lr": args.encoder_lr},
                                   {"params": head_params, "lr": args.head_lr}], weight_decay=0.01)
    steps_per_epoch = len(bucket_batches(prepared_train, args.batch_size, args.seed))
    total_steps, warmup = steps_per_epoch * args.epochs, math.ceil(steps_per_epoch * args.epochs * 0.1)
    def schedule(step):
        return step / max(1, warmup) if step < warmup else max(0.0, (total_steps - step) / max(1, total_steps - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    history, training_start, global_step = [], time.perf_counter(), 0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    for epoch in range(args.epochs):
        model.train()
        batches = bucket_batches(prepared_train, args.batch_size, args.seed + epoch)
        total_loss, total_units, epoch_start = 0.0, 0, time.perf_counter()
        for step, indices in enumerate(batches, 1):
            batch = collate([prepared_train[i] for i in indices], label_to_id)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits.reshape(-1, len(labels)), batch["targets"].reshape(-1))
            assert torch.isfinite(loss), "Non-finite training loss"
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder_params + head_params, 1.0)
            optimizer.step()
            scheduler.step()
            units = int((batch["targets"] != -100).sum())
            total_loss += float(loss.detach()) * units
            total_units += units
            global_step += 1
            if step % 10 == 0 or step == len(batches):
                event = {"phase": "training", "epoch": epoch + 1, "batch": step, "batches": len(batches),
                         "global_step": global_step, "mean_loss": total_loss / total_units,
                         "epoch_seconds": round(time.perf_counter() - epoch_start, 1),
                         "total_seconds": round(time.perf_counter() - training_start, 1)}
                print(json.dumps(event), flush=True)
                with (output / "training_log.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event) + "\n")
        history.append({"epoch": epoch + 1, "mean_training_loss": total_loss / total_units,
                        "seconds": time.perf_counter() - epoch_start})
        write_json(output / "history.json", history)
    training_seconds = time.perf_counter() - training_start
    print("Saving fixed-final-epoch model", flush=True)
    save_file({name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()},
              str(output / "model.safetensors"))
    print("Evaluating the frozen validation split", flush=True)
    predictions = predict(model, prepared_validation, labels, args.batch_size)
    write_jsonl(output / "predictions.jsonl", [dict(row, predicted_label=pred) for row, pred in zip(validation, predictions)])
    errors, categories = error_analysis(validation, predictions)
    write_jsonl(output / "errors.jsonl", errors)
    validation_metrics = score(validation, predictions, entity_labels)
    print(json.dumps({"validation": validation_metrics["micro"]}), flush=True)
    print("Evaluating training fit", flush=True)
    training_predictions = predict(model, prepared_train, labels, args.batch_size)
    training_metrics = score(train, training_predictions, entity_labels)
    result = {"configuration": config, "preprocessing": stats, "history": history,
              "training_seconds": training_seconds, "training": training_metrics,
              "validation": validation_metrics, "error_categories": categories,
              "crf_validation": prior["learning_curve"][-1]["validation"],
              "entity_micro_f1_change": validation_metrics["micro"]["f1"] - prior["learning_curve"][-1]["validation"]["micro"]["f1"],
              "gold_test_used": False, "model_sha256": sha(output / "model.safetensors")}
    for status, key in [("user_reported_reviewed_accepted", "reviewed_validation"),
                        ("agent_annotated_not_human_reviewed", "unreviewed_validation")]:
        pairs = [(row, pred) for row, pred in zip(validation, predictions) if row["review_status"] == status]
        result[key] = score([row for row, _ in pairs], [pred for _, pred in pairs], entity_labels)
    assert hashes == {str(path): sha(path) for path in protected}, "A protected input changed"
    result["protected_inputs_unchanged"] = True
    write_json(output / "results.json", result)
    print(json.dumps({"complete": True, "validation_f1": validation_metrics["micro"]["f1"],
                      "crf_f1": prior["learning_curve"][-1]["validation"]["micro"]["f1"],
                      "training_seconds": training_seconds, "results": str(output / "results.json")}), flush=True)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
