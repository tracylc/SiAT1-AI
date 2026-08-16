#!/usr/bin/env python3
"""Five-fold SESNet ensemble prediction using released SiAT1 checkpoints.

This script is intentionally written against the public SESNet components.
It returns individual fold predictions and mean/median ensembles on both the
normalized model scale and original %WT activity scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from db.dataset import MutantDataset
from df.feature_manager import FeatureManager
from lit.lit_module import SESNetLitModule


def load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def predict_one(checkpoint: Path, fasta: Path, feature_dir: Path, input_tsv: Path) -> tuple[list[str], np.ndarray]:
    module = SESNetLitModule.load_from_checkpoint(str(checkpoint), map_location="cpu")
    module.eval()
    manager = FeatureManager(name="SiAT1", fasta=str(fasta), data_dir=str(feature_dir), rt=False)
    manager.init_from_str(list({module.target_name} | module.features))
    manager.load()
    dataset = MutantDataset.from_files(fasta=str(fasta), tsv=str(input_tsv), feature_manager=manager)
    loader = dataset.gen_loader(batch_size=min(len(dataset), 16), workers=0)
    mutants, scores = [], []
    with torch.inference_mode():
        for batch in loader:
            _, _, prediction = module.forward(**batch)
            mutants.extend(batch["mutants"])
            scores.extend(prediction.flatten().cpu().tolist())
    return mutants, np.asarray(scores, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="TSV with mutant and a placeholder score column")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, default=Path("reference/SiAT1.fasta"))
    parser.add_argument("--feature-dir", type=Path, required=True, help="SESNet feature cache for --input")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--scaler", type=Path, default=Path("work/finetune_scaler.pkl"))
    args = parser.parse_args()

    input_table = pd.read_csv(args.input, sep="\t")
    if "mutant" not in input_table:
        raise KeyError("Input must have a mutant column")
    scaler = load_torch(args.scaler)
    normalized = []
    for fold in range(1, 6):
        mutants, pred = predict_one(args.checkpoint_dir / f"fold{fold}.pt", args.fasta, args.feature_dir, args.input)
        if mutants != input_table.mutant.tolist():
            raise RuntimeError(f"Prediction order mismatch in fold {fold}")
        normalized.append(pred)
    normalized = np.asarray(normalized)
    raw = scaler.inverse_transform(normalized.reshape(-1, 1)).reshape(normalized.shape)
    result = pd.DataFrame({"mutant": input_table.mutant})
    for fold in range(1, 6):
        result[f"fold{fold}_normalized"] = normalized[fold - 1]
        result[f"fold{fold}_prediction_percent_wt"] = raw[fold - 1]
    result["ensemble_mean_prediction_percent_wt"] = raw.mean(axis=0)
    result["ensemble_median_prediction_percent_wt"] = np.median(raw, axis=0)
    result["ensemble_std_percent_wt"] = raw.std(axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
