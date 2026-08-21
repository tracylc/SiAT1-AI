# SiAT1-AI reproducibility package

This archive contains the virtual-label data, experimental single-mutant data,
five-fold splits, scoring/fine-tuning/prediction scripts, reference inputs and
the five final fine-tuned model weights used for SiAT1 combinatorial prediction.

## Release layout

```text
SiAT1-AI/
  README.md
  requirements.txt
  data/
    experimental_single_63.csv
    virtual_single_8493.csv
    virtual_double_30000.csv
    cv_split.csv
  reference/
    SiAT1.fasta
    SiAT1.pdb
    SiAT1.braw
  scripts/
    score_virtual_mutants.py
    finetune.py
    finetune_sesnet.py
    predict_ensemble.py
  config/
    pretrain.yaml
  checkpoints/
    fold1.pt ... fold5.pt   # available at Zenodo: DOI: 10.5281/zenodo.21962712
```

## Released data

| File | Rows | Content |
|---|---:|---|
| `experimental_single_63.csv` | 63 | Experimental non-WT single-mutant activity, in `%WT` and `/WT` |
| `virtual_single_8493.csv` | 8,493 | Normalized ESM-1v virtual labels for all single substitutions |
| `virtual_double_30000.csv` | 30,000 | Normalized ESM-1v virtual labels for random double substitutions |
| `cv_split.csv` | 63 | Five-fold membership of the 63 experimentally tested non-WT mutants |

The virtual labels have the exact normalized scale used in SiAT1 pretraining.
Their `pretrain_split` column records the original fixed 90%/10% train/validation
membership, so the two released CSV files reproduce the archived split when
their `train` and `valid` subsets are combined, respectively.
`experimental_single_63.csv` excludes the WT reference `M1M=100` and retains
both `relative_activity_percent_wt` and `relative_activity_wt`.

## Dependencies and public components

The released scripts were tested with Python 3.9. Install the packages listed
in `requirements.txt` and the public fusion-model implementation source
tree, which supplies `db/`, `df/`, `lit/`, and `model/` for fine-tuning and
ensemble inference. The SiAT1 sequence–structure fusion model was implemented using the published SESNet framework. The third-party components are:

- [ESM / ESM-1v](https://github.com/facebookresearch/esm):
  `esm1v_t33_650M_UR90S_1` to `_5`;
- [LigandMPNN](https://github.com/dauparas/LigandMPNN): published LigandMPNN
  implementation and pretrained weights;
- [SESNet](https://github.com/SESNet/SESNet-release): published SESNet
  implementation used for pretraining, fine-tuning and inference;
- [PyTorch](https://pytorch.org/): 2.3.0;
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/): 2.4.0;
- [scikit-learn](https://scikit-learn.org/): scaling and metrics;
- [CCMpred](https://github.com/soedinglab/CCMpred): Potts feature input.

The supplied `reference/SiAT1.fasta`, `.pdb`, and `.braw` are required to
construct the same SESNet features.

## Single-mutant candidate generation

Single-mutant candidates were generated using the published LigandMPNN and
ESM-1v models. LigandMPNN was applied to the NAC-bound structural context for
proximal-site design, whereas ESM-1v masked-marginal scoring was used for
evolutionary prioritisation. The corresponding third-party repositories and
model identifiers are provided above; their source code and pretrained weights
are not redistributed in this archive. The 63 experimentally tested single
mutants are provided in `experimental_single_63.csv`.

## Virtual scoring

Input CSV/TSV needs a one-based `mutant` column, for example `K30A;E312R`.

```bash
python scripts/score_virtual_mutants.py \
  --fasta reference/SiAT1.fasta --input mutants.tsv \
  --output virtual_scores.csv --models 1
```

For a double mutant, the model-1 historical joint label is the difference
between ordered masked joint log probabilities of mutant and WT sequences.
Historical SiAT1 virtual labels used ESM-1v model 1 only. The optional command
`--models 1 2 3 4 5` produces the five scores and their arithmetic mean for
new virtual-data generation; it does not redefine the archived model-1 labels.

## Virtual pretraining

`config/pretrain.yaml` records the archived pretraining architecture and
hyperparameters: 256 hidden units, dropout 0.1, batch size 256, Adam learning
rate 1e-5, 300 maximum epochs, early-stopping patience 100, seed 42, and a
frozen ESM global feature. First filter `virtual_single_8493.csv` and
`virtual_double_30000.csv` by `pretrain_split`, concatenate the two `train`
subsets and the two `valid` subsets, and write tab-delimited `mutant` and
`score` columns to `work/pretrain_train.tsv` and `work/pretrain_valid.tsv`.

With a public SESNet checkout placed alongside this package, construct the
feature cache using those two TSV files and the released FASTA/PDB/braw inputs:

```bash
python -m df.preprocess --name SiAT1_pretrain \
  --fasta ../SiAT1-AI/reference/SiAT1.fasta --pdb ../SiAT1-AI/reference/SiAT1.pdb \
  --tsv ../SiAT1-AI/work/pretrain_train.tsv --tsv ../SiAT1-AI/work/pretrain_valid.tsv \
  --dir ../SiAT1-AI/work/pretrain_features --feature esmif1_seq_score \
  --feature esmif1_seq_score_full --feature score --feature potts \
  --braw ../SiAT1-AI/reference/SiAT1.braw
python -m lit.train --yaml ../SiAT1-AI/config/pretrain.yaml
```

## Fine-tuning and prediction

```bash
python scripts/finetune.py --data-dir data --output-dir work/cv_files
```

The script recreates the released five-fold CV tables for the 63 experimental
mutants and fits one global MinMax scaler to their raw `%WT` values. 
Fine-tune the five fold-specific models with the archived architecture and
hyperparameters using:

```bash
python scripts/finetune_sesnet.py --sesnet-root path/to/SESNet \
  --feature-dir path/to/feature_cache --init-checkpoint path/to/pretrained.pt
```

Use the public SESNet preprocessing workflow to create a feature cache for a
new input TSV, then predict with the final 5 released weights:

```bash
python scripts/predict_ensemble.py --input new_mutants.tsv \
  --feature-dir path/to/feature_cache --output ensemble_predictions.csv
```

The prediction file contains the outputs of all five fold models in normalized
and inverse-scaled `%WT` units, together with their mean, median and standard
deviation. Combinatorial candidates in the original study were selected based
on consensus among the fold-wise rankings. Inverse scaling is linear, so
averaging before or after inverse scaling is equivalent.

