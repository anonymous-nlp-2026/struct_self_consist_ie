#!/bin/bash
set -e
source activate
cd .

echo "Merging SciERC 10-epoch LoRA adapter..."
echo "  Base model: ./models/Qwen/Qwen3-8B"
echo "  Adapter: checkpoints/qwen3-8b-scierc-10epoch-lora/"
echo "  Output: checkpoints/qwen3-8b-scierc-10epoch-merged/"

llamafactory-cli export export_config_scierc_10epoch.yaml

echo "Merge complete."
ls -la checkpoints/qwen3-8b-scierc-10epoch-merged/config.json
