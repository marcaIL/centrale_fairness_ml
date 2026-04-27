#!/bin/bash

set -e  # Stop on first error

UV=/Users/marceau_bouilly/.asdf/installs/uv/0.10.6/bin/uv

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
echo "============================================================"
echo "Running privacy_attacks/attribute_inference_race.py ..."
echo "============================================================"
PYTHONPATH=src $UV run python -m src.privacy_attacks.attribute_inference_race

echo ""
echo "============================================================"
echo "Running privacy_attacks/defense_output_perturbation.py ..."
echo "============================================================"
PYTHONPATH=src $UV run python -m src.privacy_attacks.defense_output_perturbation

echo ""
echo "============================================================"
echo "Running adversarial_attacks/evasion_attack.py ..."
echo "============================================================"
PYTHONPATH=src $UV run python -m src.adversarial_attacks.evasion_attack

echo ""
echo "============================================================"
echo "Running adversarial_attacks/evasion_defenses.py ..."
echo "============================================================"
PYTHONPATH=src $UV run python -m src.adversarial_attacks.evasion_defenses

echo ""
echo "All scripts completed successfully."

