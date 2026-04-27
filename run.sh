#!/bin/bash

set -e  # Stop on first error

UV=$(command -v uv || echo "uv")

cd "$(dirname "$0")"

echo "============================================================"
echo "Running models_naive_training.py ..."
echo "============================================================"
PYTHONPATH=src $UV run python -m src.models_naive_training

echo ""
echo "============================================================"
echo "Running models_naive_no_race_training.py ..."
echo "============================================================"
PYTHONPATH=src $UV run python -m src.models_naive_no_race_training

echo ""
echo "============================================================"
echo "Running bias_mitigation.py ..."
echo "============================================================"
PYTHONPATH=src $UV run python -m src.bias_mitigation

echo ""
echo "============================================================"
echo "Running models_comparison.py ..."
echo "============================================================"
PYTHONPATH=src $UV run python -m src.models_comparison

echo ""
echo "All scripts completed successfully."

