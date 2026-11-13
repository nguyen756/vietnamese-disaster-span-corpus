"""Manual entry points. No command automatically starts the next training run."""
import argparse
import json
import os
from common import EXP, ARCHITECTURES, SEEDS, verify_sources, run_path, read_json, versions, protocol


def main():
    parser = argparse.ArgumentParser(description='VnCoreNLP, three architectures, three seeds, twenty full epochs')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('prepare', help='Prepare/verify train+dev cache; no training and no Gold Test preprocessing')
    trainer = commands.add_parser('train', help='Train only the explicitly selected architecture and seed')
    trainer.add_argument('--architecture', choices=ARCHITECTURES, required=True)
    trainer.add_argument('--seed', choices=SEEDS, type=int, required=True)
    trainer.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    trainer.add_argument('--resume', action='store_true')
    trainer.add_argument('--restart-before-first-checkpoint', action='store_true', help='Restart the same configuration if interrupted before epoch 1 completed; preserve logs')
    commands.add_parser('status', help='Read the status of the nine runs')
    commands.add_parser('evaluate', help='Evaluate all nine fixed epoch-20 models after all runs complete')
    commands.add_parser('package', help='Package completed metrics, predictions, errors, logs and nine final models')
    args = parser.parse_args()
    if args.command == 'status':
        for architecture in ARCHITECTURES:
            for seed in SEEDS:
                run = run_path(architecture, seed)
                state = read_json(run / 'STATUS.json') if (run / 'STATUS.json').exists() else {'status': 'not_started', 'completed_epochs': 0}
                print(json.dumps({'run': run.name, **state}, ensure_ascii=False))
        return
    verify_sources()
    if args.command == 'prepare':
        from preprocessing import prepare_training_cache
        prepare_training_cache()
    elif args.command == 'train':
        config = read_json(EXP / 'configs' / f'{args.architecture}_seed{args.seed}.json')
        assert all(config[k] == v for k, v in protocol().items()), 'Run configuration differs from the fixed protocol'
        assert config['architecture'] == args.architecture and config['seed'] == args.seed
        # These environment flags take effect before importing PyTorch.
        if args.device == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        from training import train_one
        train_one(args.architecture, args.seed, args.device, args.resume, args.restart_before_first_checkpoint)
    elif args.command == 'evaluate':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        from evaluation import evaluate_all
        evaluate_all()
    elif args.command == 'package':
        from evaluation import package_results
        package_results()


if __name__ == '__main__':
    main()
