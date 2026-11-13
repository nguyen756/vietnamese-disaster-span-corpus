"""Explicit single-run training: twenty complete epochs; never called by setup/tests."""
import fcntl
import csv
import math
import os
from pathlib import Path
import random
import time
import traceback
import numpy as np
import torch
from safetensors.torch import save_file
from transformers import RobertaConfig, RobertaModel
from common import (EXP, PACKAGE, read_json, save_json, sha, protocol, versions, source_hashes,
                    load_cache, run_path, log_event, read_jsonl, write_jsonl, score, decode,
                    collate, bucket_batches, error_analysis)
from model import NERModel


def move_batch(batch, device):
    return {key: value.to(device) if hasattr(value, 'to') and key != 'lengths' else value
            for key, value in batch.items()}


def predict(model, prepared, labels, device, batch_size=8):
    mapping = {label: i for i, label in enumerate(labels)}
    predictions = [None] * len(prepared)
    model.eval()
    with torch.inference_mode():
        for indices in bucket_batches(prepared, batch_size, 0):
            batch = move_batch(collate([prepared[i] for i in indices], mapping), device)
            tags = model.tag_sequences(model(batch), batch)
            for index, sequence in zip(indices, tags):
                predictions[index] = decode(prepared[index]['units'], [labels[tag] for tag in sequence])
    assert all(item is not None for item in predictions)
    return predictions


def build_model(architecture, bio, pretrained=True, encoder_config=None):
    base = PACKAGE / 'assets/phobert-base'
    config = encoder_config or RobertaConfig.from_pretrained(str(base), local_files_only=True)
    if pretrained:
        encoder = RobertaModel.from_pretrained(str(base), config=config, local_files_only=True,
                                              add_pooling_layer=False, attn_implementation='sdpa')
    else:
        config._attn_implementation = 'sdpa'
        encoder = RobertaModel(config, add_pooling_layer=False)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    for layer in encoder.encoder.layer[-protocol()['train_layers']:]:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
    return NERModel(encoder, bio, architecture, hidden=protocol()['lstm_hidden'], dropout=protocol()['dropout'])


def atomic_checkpoint(path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def link_checkpoint(run, name, checkpoint):
    path = run / name
    temporary = run / (name + '.tmp')
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(checkpoint.relative_to(run))
    temporary.replace(path)


def write_history(run, history):
    save_json(run / 'history.json', history)
    with (run / 'history.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch', 'mean_training_loss', 'training_seconds', 'evaluation_seconds', 'encoder_lr', 'head_lr', 'precision', 'recall', 'f1', 'macro_f1'])
        writer.writeheader()
        for h in history:
            writer.writerow({**{k: h[k] for k in ['epoch', 'mean_training_loss', 'training_seconds', 'evaluation_seconds', 'encoder_lr', 'head_lr']}, **{k: h['development']['micro'][k] for k in ['precision', 'recall', 'f1']}, 'macro_f1': h['development']['macro_f1']})


def train_one(architecture, seed, device_name='cpu', resume=False, restart=False):
    plan = protocol()
    assert plan['epochs'] == 20 and plan['early_stopping'] is False
    if (EXP / 'evaluation/model_lock.json').exists():
        raise RuntimeError('Final evaluation is locked; training changes require a separate experiment')
    run = run_path(architecture, seed)
    if (run / 'results.json').exists():
        raise FileExistsError(f'Completed run exists: {run}; no automatic overwrite or extra epochs')
    run.mkdir(parents=True, exist_ok=True)
    with (run / 'training.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checkpoints = run / 'checkpoints'
        checkpoints.mkdir(exist_ok=True)
        existing = sorted(checkpoints.glob('epoch_*.pt'))
        if existing and not resume:
            raise RuntimeError('Epoch checkpoints exist; use --resume explicitly')
        if resume and not existing:
            raise FileNotFoundError('No completed epoch checkpoint exists; start without --resume')
        if restart and (resume or existing):
            raise RuntimeError('Restart-before-first-checkpoint only applies when there is no completed epoch')
        if not resume and not restart and (run / 'experiment_config.json').exists():
            raise RuntimeError('Partial run without checkpoint; preserve its logs and choose --restart-before-first-checkpoint')
        manifest, prepared_train, prepared_dev = load_cache()
        train = read_jsonl(PACKAGE / 'data/train.jsonl')
        dev = read_jsonl(PACKAGE / 'data/validation.jsonl')
        assert len(train) == len(prepared_train) == 9686
        assert len(dev) == len(prepared_dev) == 803
        labels = read_json(PACKAGE / 'reference/annotation_schema.json')['labels']
        bio = ['O'] + [prefix + label for label in labels for prefix in ['B-', 'I-']]
        mapping = {label: i for i, label in enumerate(bio)}
        torch.set_num_threads(plan['threads'])
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        device = torch.device(device_name)
        if device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable; no automatic CPU fallback between runs')
            torch.cuda.manual_seed_all(seed)
        spec = {**plan, 'architecture': architecture, 'seed': seed, 'device': device_name,
                'bio_labels': bio, 'versions': versions(), 'code_sha256': source_hashes(),
                'cache_manifest_sha256': sha(EXP / 'cache/manifest.json')}
        if (run / 'experiment_config.json').exists():
            assert read_json(run / 'experiment_config.json') == spec, 'Existing run configuration differs'
        model = build_model(architecture, bio).to(device)
        encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
        head_params = [p for name, p in model.named_parameters() if not name.startswith('encoder.') and p.requires_grad]
        optimizer = torch.optim.AdamW([{'params': encoder_params, 'lr': plan['encoder_lr']},
                                       {'params': head_params, 'lr': plan['head_lr']}], weight_decay=plan['weight_decay'])
        batches_per_epoch = len(bucket_batches(prepared_train, plan['batch_size'], seed))
        total_steps = batches_per_epoch * plan['epochs']
        warmup = math.ceil(total_steps * plan['warmup_fraction'])
        def schedule(step):
            return step / max(1, warmup) if step < warmup else max(0., (total_steps - step) / max(1, total_steps - warmup))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        history, start_epoch, global_step = [], 0, 0
        if resume:
            state = torch.load(existing[-1], map_location='cpu', weights_only=False)
            assert state['training_spec'] == spec, 'Cannot resume different settings, device, versions, code, or cache'
            model.load_state_dict(state['model'], strict=True)
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            history, start_epoch, global_step = state['history'], state['completed_epochs'], state['global_step']
            assert [h['epoch'] for h in history] == list(range(1, start_epoch + 1))
            random.setstate(state['python_rng'])
            np.random.set_state(state['numpy_rng'])
            torch.set_rng_state(state['torch_rng'])
            if device.type == 'cuda' and state['cuda_rng'] is not None:
                torch.cuda.set_rng_state_all(state['cuda_rng'])
            del state
        save_json(run / 'experiment_config.json', spec)
        model.encoder.config.save_pretrained(run / 'encoder_config')
        save_json(run / 'STATUS.json', {'status': 'training', 'completed_epochs': start_epoch, 'planned_epochs': 20})
        log_event(run, 'training_started', architecture=architecture, seed=seed, device=device_name,
                  start_epoch=start_epoch, planned_epochs=20, early_stopping=False,
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        try:
            for epoch in range(start_epoch + 1, 21):
                model.train()
                batches = bucket_batches(prepared_train, plan['batch_size'], seed + epoch - 1)
                total_loss, total_units = 0., 0
                started = time.perf_counter()
                for step, indices in enumerate(batches, 1):
                    batch = move_batch(collate([prepared_train[i] for i in indices], mapping), device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = model.loss(model(batch), batch)
                    assert torch.isfinite(loss), 'Non-finite loss'
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(encoder_params + head_params, plan['gradient_clip'])
                    optimizer.step()
                    scheduler.step()
                    units = int((batch['targets'] != -100).sum())
                    total_loss += float(loss.detach()) * units
                    total_units += units
                    global_step += 1
                    if step % 25 == 0 or step == len(batches):
                        log_event(run, 'training_batch', epoch=epoch, batch=step, batches=len(batches),
                                  global_step=global_step, mean_loss=total_loss / total_units,
                                  seconds=time.perf_counter() - started)
                training_seconds = time.perf_counter() - started
                evaluation_start = time.perf_counter()
                predictions = predict(model, prepared_dev, bio, device, plan['batch_size'])
                metrics = score(dev, predictions, labels)
                record = {'epoch': epoch, 'mean_training_loss': total_loss / total_units,
                          'training_seconds': training_seconds, 'development': metrics,
                          'encoder_lr': optimizer.param_groups[0]['lr'], 'head_lr': optimizer.param_groups[1]['lr'],
                          'evaluation_seconds': time.perf_counter() - evaluation_start}
                history.append(record)
                write_jsonl(run / f'development_epoch_{epoch:02d}.jsonl',
                            [dict(row, predicted_label=p) for row, p in zip(dev, predictions)])
                save_json(run / f'metrics_epoch_{epoch:02d}.json', record)
                path = checkpoints / f'epoch_{epoch:02d}.pt'
                atomic_checkpoint(path, {'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                                        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                                        'completed_epochs': epoch, 'history': history, 'global_step': global_step,
                                        'training_spec': spec, 'python_rng': random.getstate(),
                                        'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
                                        'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None})
                best = max(history, key=lambda h: h['development']['micro']['f1'])  # earliest tie
                link_checkpoint(run, 'last_checkpoint.pt', path)
                link_checkpoint(run, 'best_development_checkpoint.pt', checkpoints / f"epoch_{best['epoch']:02d}.pt")
                write_history(run, history)
                save_json(run / 'STATUS.json', {'status': 'training' if epoch < 20 else 'finalizing',
                                               'completed_epochs': epoch, 'planned_epochs': 20,
                                               'best_development_epoch': best['epoch']})
                log_event(run, 'epoch_completed', epoch=epoch, mean_training_loss=record['mean_training_loss'],
                          development=metrics['micro'], checkpoint=str(path), best_development_epoch=best['epoch'])
            assert [h['epoch'] for h in history] == list(range(1, 21))
            save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(run / 'model_epoch20.safetensors'))
            best = max(history, key=lambda h: h['development']['micro']['f1'])
            write_history(run, history)
            link_checkpoint(run, 'last_checkpoint.pt', checkpoints / 'epoch_20.pt')
            link_checkpoint(run, 'best_development_checkpoint.pt', checkpoints / f"epoch_{best['epoch']:02d}.pt")
            result = {'architecture': architecture, 'seed': seed, 'completed_epochs': 20,
                      'primary_checkpoint_epoch': 20, 'primary_model_sha256': sha(run / 'model_epoch20.safetensors'),
                      'development_epoch20': history[-1]['development'],
                      'best_development_epoch': best['epoch'], 'best_development': best['development'],
                      'training_seconds': sum(h['training_seconds'] for h in history),
                      'history': history, 'configuration': spec, 'gold_test_used_during_training': False,
                      'all_epoch_checkpoints_retained': True}
            save_json(run / 'results.json', result)
            save_json(run / 'STATUS.json', {'status': 'complete', 'completed_epochs': 20, 'planned_epochs': 20,
                                           'best_development_epoch': best['epoch'], 'gold_test_scored': False})
            log_event(run, 'training_complete', epoch20_development=history[-1]['development']['micro'])
        except BaseException as exc:
            completed = len(list(checkpoints.glob('epoch_*.pt')))
            save_json(run / 'STATUS.json', {'status': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                                           'completed_epochs': completed, 'planned_epochs': 20, 'error': str(exc)})
            with (run / 'failure.log').open('a') as stream:
                stream.write(traceback.format_exc() + '\n')
            raise
