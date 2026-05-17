# Cold-Start Embedding Alignment

Reproducible research code for cold-start item recommendation experiments.
The repository reproduces Let-It-Go style baselines and evaluates pairwise
alignment methods for transferring content embeddings into the collaborative
embedding space.

The repository intentionally contains only experiment code. It does not include
datasets, generated artifacts, notebooks, thesis text, slides, or environment
specific files.

## Repository Structure

```text
cold-start-embedding-alignment/
├── src/
│   ├── experiment/      # data loading, models, training, pairwise pipeline
│   ├── paper/           # paper baseline reproduction and comparison runners
│   ├── preparation/     # dataset preparation scripts
│   ├── reports/         # reusable plotting helpers
│   └── paths.py         # canonical artifact paths
├── scripts/             # CLI entrypoints
├── data/README.md       # data placement notes
├── pyproject.toml       # Python dependencies
├── poetry.lock
└── .gitignore
```

## Experiments

The main experiment families are:

| ID | Description |
|---|---|
| `E00` | Content initialization baseline |
| `E0` | Let-It-Go trainable delta baseline |
| `E3` / `E3S` | Transformer mapper with InfoNCE-style alignment |
| `E16` | Norm-aware residual fusion |
| `E16_*` | E16 variants: warm samplers, all-item inference, delta targets, frequency-aware blending |

The recommended runner is `scripts/run_e16_variants.sh`. It trains `E00` and
`E0` once per seed, then runs all selected `E16_*` variants in a single
pipeline pass per seed.

## Data

The code expects datasets in Let-It-Go compatible format:

```text
<dataset-root>/
├── processed/
│   ├── train_interactions.parquet
│   ├── val_interactions.parquet
│   ├── test_interactions.parquet
│   ├── ground_truth.parquet
│   ├── item2index_warm.pkl
│   └── item2index_cold.pkl
└── item_embeddings/
    ├── embeddings_warm.npy
    └── embeddings_cold.npy
```

Amazon M2 uses the original Let-It-Go CSV names:

```text
processed/train_data.csv
processed/val_data.csv
processed/test_inputs.csv
processed/test_target.csv
```

Supported dataset keys in the runners are `zvuk`, `amazon_m2`, and `yambda`.

Yambda-50M can be prepared directly from HuggingFace:

```bash
python scripts/prepare_yambda.py
```

Zvuk can be prepared from a local raw copy:

```bash
python scripts/prepare_letitgo_zvuk_splits.py \
  --zvuk-data-path /path/to/zvuk/raw \
  --output-dir ~/let-it-go/data/zvuk
```

## Quick Start

Install dependencies:

```bash
poetry install
```

Run the E16 variant sweep:

```bash
bash scripts/run_e16_variants.sh amazon_m2
bash scripts/run_e16_variants.sh zvuk
bash scripts/run_e16_variants.sh yambda
```

Run only selected variants:

```bash
bash scripts/run_e16_variants.sh yambda "E16_CLEAR,E16_ALL_LOW,E16_ALL_FREQ"
```

Run unified paper comparison:

```bash
python scripts/run_paper_comparison.py \
  --letitgo-dataset amazon_m2 \
  --pairwise-experiments E3,E16 \
  --seeds 42,221,451,934,1984
```

## Reproduction Commands

Comparable sweep on Zvuk:

```bash
python scripts/run_paper_comparable_sweep.py \
  --letitgo-dataset zvuk \
  --output-dir artifacts/paper_comparison/zvuk_comparable_sweep \
  --pairwise-experiments E3,E16 \
  --seeds 42,221,451,934,1984 \
  --topk 10
```

Unified comparison on Amazon M2:

```bash
python scripts/run_paper_comparison.py \
  --letitgo-dataset amazon_m2 \
  --output-dir artifacts/paper_comparison/amazon_m2_paper_comparison \
  --pairwise-experiments E3,E3S,E14,E16 \
  --seeds 42,221,451,934,1984 \
  --topk 10
```

Unified comparison on Yambda-50M:

```bash
python scripts/run_paper_comparison.py \
  --letitgo-dataset yambda \
  --output-dir artifacts/paper_comparison/yambda_paper_comparison \
  --pairwise-experiments E3,E3S,E14,E16 \
  --seeds 42,221,451,934,1984 \
  --topk 10
```

Prepare Yambda-50M and run E16 variant sweeps:

```bash
python scripts/prepare_yambda.py \
  --output-dir "$HOME/let-it-go/data/yambda"

bash scripts/run_e16_variants.sh
bash scripts/run_e16_variants.sh yambda
```

## Outputs

Generated results are written under `artifacts/` and are ignored by git:

```text
artifacts/paper_comparison/
├── zvuk_comparable_sweep/
├── amazon_m2_paper_comparison/
├── yambda_paper_comparison/
├── zvuk_e16_variants/
├── amazon_m2_e16_variants/
└── yambda_e16_variants/
```

Each run directory contains the same core files when applicable:

- `experiment_results.csv` / `experiment_results.json`: per-seed records.
- `results_summary.csv` / `results_summary.json`: aggregated metrics.
- `experiment_significance.csv` / `experiment_significance.json`: pairwise tests against `E0`.
- `run_metadata.json`: seeds, dataset statistics, device, and evaluation mode.
- `runtime_details.csv` or `paper_runtime_details.csv`: runtime diagnostics.
- `variant_manifest.json`: variant definitions for comparable sweeps.

## Clean Repository Policy

This repository should contain only reproducible research code. Keep out:

- notebooks and notebook outputs;
- generated artifacts, checkpoints, logs, and datasets;
- private paths, local machine paths, credentials, tokens, and environment files;
- thesis text, slides, and presentation materials.
