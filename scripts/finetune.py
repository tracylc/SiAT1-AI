#!/usr/bin/env python3
"""Prepare released five-fold fine-tuning tables for SESNet.

The release contains the final five checkpoint weights for inference.  This
script makes reproducible train/test TSV files from the released 63-point
experimental data and the released fold membership. It also writes the single
global MinMaxScaler used for the released five-fold tables.

Train a fold with a compatible public SESNet checkout by passing the emitted
TSV paths and the SiAT1 reference files to that checkout's training entrypoint.
The architecture/hyperparameters are recorded in README.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("work/cv_files"))
    args = parser.parse_args()
    data = pd.read_csv(args.data_dir / "experimental_single_63.csv")
    split = pd.read_csv(args.data_dir / "cv_split.csv")
    merged = data.merge(split, on="mutant", validate="one_to_one")
    if len(merged) != 63 or set(merged.outer_test_fold) != {1, 2, 3, 4, 5}:
        raise RuntimeError("experimental_single_63.csv and cv_split.csv are inconsistent")

    scaler = MinMaxScaler().fit(merged[["relative_activity_percent_wt"]])
    merged["score"] = scaler.transform(merged[["relative_activity_percent_wt"]])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(scaler, args.output_dir.parent / "finetune_scaler.pkl")
    for fold in range(1, 6):
        test = merged.loc[merged.outer_test_fold.eq(fold), ["mutant", "score"]]
        train = merged.loc[~merged.outer_test_fold.eq(fold), ["mutant", "score"]]
        train.to_csv(args.output_dir / f"train_fold{fold}.tsv", sep="\t", index=False)
        test.to_csv(args.output_dir / f"test_fold{fold}.tsv", sep="\t", index=False)
    merged.to_csv(args.output_dir.parent / "experimental_single_63_normalized.csv", index=False)
    print(f"Prepared five CV folds in {args.output_dir}")


if __name__ == "__main__":
    main()
