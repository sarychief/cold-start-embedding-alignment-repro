#!/usr/bin/env bash
# Single-pass E16 variant sweep: SASRec trains once per seed, then all
# E16_* mappers run sequentially inside a single experiment grid.
#
# Usage:
#   bash scripts/run_e16_variants.sh                   # zvuk + amazon_m2, all variants
#   bash scripts/run_e16_variants.sh zvuk              # Zvuk only
#   bash scripts/run_e16_variants.sh amazon_m2         # Amazon M2 only
#   bash scripts/run_e16_variants.sh yambda            # Yambda only
#   bash scripts/run_e16_variants.sh amazon_m2 "E16_CLEAR,E16_ALL_LOW,E16_ALL_FREQ"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SEEDS="42,221,451,934,1984"
DATASET_FILTER="${1:-all}"
VARIANTS="${2:-}"

run_dataset() {
    local dataset="$1"
    local output_dir="$REPO_ROOT/artifacts/paper_comparison/${dataset}_e16_variants"

    echo "========================================================================"
    echo "  E16 SINGLE-PASS VARIANT SWEEP | dataset=$dataset"
    echo "  output: $output_dir"
    echo "  seeds: $SEEDS"
    echo "  variants: ${VARIANTS:-all ($(python -u -c "from pathlib import Path; import sys; sys.path.insert(0,str(Path('$REPO_ROOT')/'src')); from paper.e16_variants import VARIANT_REGISTRY; print(len(VARIANT_REGISTRY))" 2>/dev/null || echo '?'))}"
    echo "========================================================================"

    local variant_arg=""
    if [[ -n "$VARIANTS" ]]; then
        variant_arg="--variants $VARIANTS"
    fi

    python -u "$REPO_ROOT/scripts/run_e16_variants.py" \
        --letitgo-dataset "$dataset" \
        --output-dir "$output_dir" \
        --seeds "$SEEDS" \
        --pairwise-mode full \
        $variant_arg
}

if [[ "$DATASET_FILTER" == "all" ]]; then
    run_dataset "zvuk"
    run_dataset "amazon_m2"
elif [[ "$DATASET_FILTER" == "zvuk" || "$DATASET_FILTER" == "amazon_m2" || "$DATASET_FILTER" == "yambda" ]]; then
    run_dataset "$DATASET_FILTER"
else
    echo "Unknown dataset: $DATASET_FILTER (expected: zvuk, amazon_m2, yambda, or all)" >&2
    exit 1
fi
