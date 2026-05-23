# When Self-Consistency Fails for NER: Degeneracy, Signal Collapse, and the Correlation-Selection Gap

Code for the paper "When Self-Consistency Fails for NER: Degeneracy, Signal Collapse, and the Correlation-Selection Gap" (anonymous submission).

## Overview

This repository provides the implementation for studying test-time compute scaling (self-consistency and best-of-N) for Named Entity Recognition. We characterize when scaling succeeds and fails using NER as the primary testbed (SciERC, CoNLL-2003, Few-NERD, WNUT-17), with cross-task pilots on relation extraction and SQL parsing.

Key components:
- **Sampling**: Multi-sample inference with vLLM + XGrammar constrained decoding (`code/sampling.py`)
- **Consistency metrics**: Fleiss' Kappa, structural soft Jaccard, log-probability signals (`code/consistency.py`, `code/unified_metrics.py`)
- **Evaluation**: Strict-match NER F1, per-instance correlation analysis (`code/evaluation.py`)
- **Selection & Construction**: Entity-level consensus construction, MBR selection, adaptive budget allocation (`code/entity_construction*.py`, `code/ccs_selection.py`)
- **Diagnostic framework**: Degeneracy detection, signal collapse analysis, correlation-selection gap diagnostics (`code/analysis_dgs_*.py`)

## Requirements

- Python 3.10+
- PyTorch 2.1+
- vLLM (for multi-sample inference with constrained decoding)
- Transformers, Datasets
- LLaMA-Factory (for LoRA fine-tuning)
- NumPy, SciPy, scikit-learn, matplotlib

Install dependencies:
```bash
pip install torch vllm transformers datasets accelerate scipy scikit-learn matplotlib seaborn tqdm
```

## Data Preparation

### Datasets
- **SciERC**: `code/data_utils.py` handles loading and conversion
- **CoNLL-2003**: `code/convert_conll2003.py`
- **Few-NERD**: `code/convert_fewnerd.py`
- **WNUT-17**: `code/convert_wnut17.py`

Convert datasets to the unified UIE JSON format:
```bash
python code/convert_conll2003.py
python code/convert_fewnerd.py
python code/convert_wnut17.py
```

The conversion scripts download datasets from HuggingFace and produce LLaMA-Factory compatible JSON files in `data/`.

### Models

We use LoRA fine-tuned models:
- **Qwen3-8B** (primary): fine-tuned with configs in `configs/train/`
- **Qwen3-4B**: scale ablation
- **LLaMA-3.1-8B / LLaMA-3.2-3B**: cross-architecture check
- **Qwen2.5-32B**: larger-scale experiment

Fine-tune with LLaMA-Factory:
```bash
llamafactory-cli train configs/train/train_config_scierc_3ep.yaml
```

Export (merge LoRA weights):
```bash
llamafactory-cli export configs/export/export_config_scierc_3epoch.yaml
```

## Running Experiments

### 1. Multi-Sample Inference

The sampling pipeline generates N constrained samples per instance:
```bash
python code/sampling.py \
    --model_path ./models/merged_model \
    --dataset scierc \
    --n_samples 16 \
    --temperature 1.0
```

### 2. Signal Analysis (5-Signal Pipeline)

Compute all quality estimation signals (MV, SJ, LP, VC, EM) for sampled outputs:
```bash
python code/eval_n16_5signal.py        # SciERC N=16
python code/exp021_fewnerd_full_analysis.py  # Few-NERD
python code/analyze_exp002_conll2003.py      # CoNLL-2003
python code/wnut17_full_signal_analysis.py   # WNUT-17
```

### 3. Selection & Construction

Entity-level consensus construction and sample selection:
```bash
python code/entity_construction_fair.py  # Fair entity construction
python code/ccs_selection.py            # Cascaded consistency selection
python code/compute_selection_f1.py     # Selection F1 evaluation
```

### 4. Diagnostic Analysis

Degeneracy-gated selection and signal collapse analysis:
```bash
python code/analysis_dgs_full_validation.py  # Full DGS validation
python code/analysis_dgs_multiseed.py        # Multi-seed DGS analysis
python code/analysis_dgs_cross_model.py      # Cross-model analysis
```

### 5. Adaptive Budget

Adaptive compute allocation experiments:
```bash
python b7_adaptive_budget.py
python b7_adaptive_budget_v2.py
```

### 6. Bootstrap Significance Tests

```bash
python run_bootstrap_loo.py       # Leave-one-out bootstrap
python bootstrap_analysis.py      # Bootstrap significance analysis
```

## Project Structure

```
.
├── code/                    # Core modules and analysis scripts
│   ├── sampling.py          # vLLM multi-sample inference
│   ├── consistency.py       # Consistency metrics (FK, SJ, LP)
│   ├── evaluation.py        # NER/RE evaluation
│   ├── unified_metrics.py   # Unified F1 and degeneracy detection
│   ├── data_utils.py        # Data loading and conversion
│   ├── entity_construction*.py  # Entity consensus construction
│   ├── analysis_dgs_*.py    # Degeneracy-gated selection analysis
│   ├── probe/               # Hidden-state probing experiments
│   └── pipeline/            # Multi-stage pipeline scripts
├── scripts/                 # Experiment post-processing and analysis
├── configs/                 # Training and export configurations
│   ├── train/               # LLaMA-Factory training configs
│   ├── export/              # LoRA merge/export configs
│   └── lora/                # LoRA adapter configs
├── experiments/             # Cross-task pilot experiments
│   ├── fewrel_re_sc/        # Relation extraction pilot
│   └── humaneval_sc/        # Code generation pilot
├── examples/
│   └── deepspeed/           # DeepSpeed ZeRO-2 config
└── *.py, *.sh               # Top-level experiment scripts
```

## Reproducing Key Results

### Table 2: Selection F1 (5-Signal Comparison)
```bash
python code/table2_selection_f1.py
```

### Correlation-Selection Gap Analysis
```bash
python code/exp010_selection_curves.py
```

### Degeneracy Analysis
```bash
python code/analysis_degen_gated_selection.py
python code/batch_recalc_degeneracy.py
```

### Temperature Ablation
```bash
python code/adaptive_temperature_analysis.py
python code/analyze_exp026_temp_5signal.py
```

### N-Scaling Curves
```bash
python code/lp_best_n_scaling.py
python scripts/n_scaling_gf_maxlp.py
```

## License

This code is released for research purposes under the MIT License.
