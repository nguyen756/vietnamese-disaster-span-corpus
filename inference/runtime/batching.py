"""Original numerical functions; imports isolated for portability. PRIVATE."""
import random

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

