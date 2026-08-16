#!/usr/bin/env python3
"""Generate or score specified SiAT1 single/double mutants with ESM-1v.

Default ``--models 1`` reproduces the historical SiAT1 virtual-label scorer.
Pass ``--models 1 2 3 4 5`` to write the 5-model ensemble mean as well.
Double mutants use the ordered joint-probability difference defined in README.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from esm import pretrained
from tqdm import tqdm


AA = "ACDEFGHIKLMNPQRSTVWY"
PATTERN = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")


def fasta_sequence(path: Path) -> str:
    return "".join(line.strip() for line in path.read_text().splitlines() if not line.startswith(">")).upper()


def parse_mutant(text: str, wt: str) -> list[tuple[str, int, str]]:
    sites = []
    for item in str(text).upper().split(";"):
        hit = PATTERN.fullmatch(item.strip())
        if not hit:
            raise ValueError(f"Invalid mutation {item!r}; expected A123V")
        old, pos_text, new = hit.groups()
        pos = int(pos_text) - 1
        if not 0 <= pos < len(wt) or wt[pos] != old or old == new:
            raise ValueError(f"Invalid WT residue or position in {item!r}")
        sites.append((old, pos, new))
    if len({site[1] for site in sites}) != len(sites):
        raise ValueError(f"A position is repeated in {text!r}")
    return sorted(sites, key=lambda x: x[1])


def apply(wt: str, sites: list[tuple[str, int, str]]) -> str:
    seq = list(wt)
    for _, position, new in sites:
        seq[position] = new
    return "".join(seq)


@torch.inference_mode()
def probability(model, alphabet, device, sequences, positions):
    """Masked-marginal single score or historical ordered double joint score."""
    _, _, tokens = alphabet.get_batch_converter()([(str(i), seq) for i, seq in enumerate(sequences)])
    tokens = tokens.to(device)
    rows = torch.arange(len(sequences), device=device)
    first_pos = torch.tensor([1 + pos[0] for pos in positions], device=device)
    first_aa = torch.tensor([alphabet.get_idx(seq[pos[0]]) for seq, pos in zip(sequences, positions)], device=device)
    first = tokens.clone(); first[rows, first_pos] = alphabet.mask_idx
    scores = torch.log_softmax(model(first)["logits"], dim=-1)[rows, first_pos, first_aa]
    if len(positions[0]) == 1:
        return scores
    second_pos = torch.tensor([1 + pos[1] for pos in positions], device=device)
    second_aa = torch.tensor([alphabet.get_idx(seq[pos[1]]) for seq, pos in zip(sequences, positions)], device=device)
    double = tokens.clone(); double[rows, first_pos] = alphabet.mask_idx; double[rows, second_pos] = alphabet.mask_idx
    scores += torch.log_softmax(model(double)["logits"], dim=-1)[rows, second_pos, second_aa]
    return scores


def score_table(table: pd.DataFrame, wt: str, models: list[int], batch_size: int) -> pd.DataFrame:
    parsed = [parse_mutant(value, wt) for value in table.mutant]
    if any(len(sites) not in (1, 2) for sites in parsed):
        raise ValueError("Only single and double mutants are supported")
    order = {len(sites) for sites in parsed}
    if len(order) != 1:
        raise ValueError("Provide single and double mutants in separate runs")
    sequences = [apply(wt, sites) for sites in parsed]
    positions = [[site[1] for site in sites] for sites in parsed]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = table.copy()
    all_scores = []
    for number in models:
        name = f"esm1v_t33_650M_UR90S_{number}"
        model, alphabet = pretrained.load_model_and_alphabet(name)
        model = model.to(device).eval()
        values = []
        for start in tqdm(range(0, len(table), batch_size), desc=name):
            end = min(start + batch_size, len(table))
            mutant = probability(model, alphabet, device, sequences[start:end], positions[start:end])
            wildtype = probability(model, alphabet, device, [wt] * (end - start), positions[start:end])
            values.extend((mutant - wildtype).cpu().tolist())
        output[name] = values
        all_scores.append(values)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output["score"] = np.asarray(all_scores).mean(axis=0)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="CSV/TSV with a mutant column")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--models", type=int, nargs="+", choices=range(1, 6), default=[1])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    table = pd.read_csv(args.input, sep="\t" if args.input.suffix.lower() in {".tsv", ".tab"} else ",")
    table = table.rename(columns=lambda column: str(column).lstrip("\ufeff"))
    if "mutant" not in table:
        raise KeyError("Input must have a mutant column")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    score_table(table, fasta_sequence(args.fasta), args.models, args.batch_size).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
