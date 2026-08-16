#!/usr/bin/env python3
"""Invoke public SESNet five-fold fine-tuning from released CV tables.

Run ``finetune.py`` first to create work/cv_files and a shared scaler. This
wrapper writes the archived hyperparameter YAML for each fold and calls the
public ``python -m lit.train --yaml ...`` entrypoint. An optional pretraining
checkpoint can be supplied for initialization.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sesnet-root", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, default=Path("reference/SiAT1.fasta"))
    parser.add_argument("--cv-dir", type=Path, default=Path("work/cv_files"))
    parser.add_argument("--output-dir", type=Path, default=Path("work/finetune"))
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(1, 6):
        config = {
            "name": f"SiAT1_finetune_fold{fold}", "dir": str(args.feature_dir.resolve()),
            "fasta": str(args.fasta.resolve()),
            "train_tsv": [str((args.cv_dir / f"train_fold{fold}.tsv").resolve())],
            "valid_tsv": [str((args.cv_dir / f"test_fold{fold}.tsv").resolve())],
            "test_tsv": [str((args.cv_dir / f"test_fold{fold}.tsv").resolve())],
            "target_name": "score", "local_name": None, "local_static": False, "local_freeze": False,
            "global_name": "esm", "global_static": False, "global_freeze": False,
            "agg": "attn", "agg_seq_score": True, "decoder": "mlp", "decoder_seq_score": True,
            "batch_size": 2, "hidden": 256, "dropout": 0.2, "optim": "adam", "warmup": -1,
            "rt": False, "seed": 42, "epoch": 100, "patience": 30, "lr": 1e-4,
            "checkpoint_dir": str((args.output_dir / f"fold{fold}_checkpoints").resolve()),
            "result_file": str((args.output_dir / f"fold{fold}_result.pkl").resolve()),
            "test_out": None, "monitor": "val_corr", "monitor_mode": "max",
            "ckpt": str(args.init_checkpoint.resolve()) if args.init_checkpoint else None,
        }
        yaml_path = args.output_dir / f"fold{fold}.yaml"
        with yaml_path.open("w") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        subprocess.run([sys.executable, "-m", "lit.train", "--yaml", str(yaml_path.resolve())],
                       cwd=args.sesnet_root.resolve(), check=True)


if __name__ == "__main__":
    main()
