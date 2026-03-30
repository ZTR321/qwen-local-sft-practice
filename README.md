# Qwen2.5-7B Local SFT Practice

A learning-oriented project that explores local instruction fine-tuning with Qwen2.5-7B under resource-constrained settings.

## Highlights

- local instruction fine-tuning workflow with `Qwen2.5-7B-Instruct`
- finance-related sample filtering from a public Chinese instruction dataset
- LoRA adapter injection with 4-bit model loading
- artifact saving for train/eval splits, stats, outputs, and quick preview
- interview-friendly project scope with explicit boundaries

## What This Project Does

- Filters a public Chinese instruction dataset for finance-related samples
- Converts samples into chat-template formatted training text
- Loads `Qwen/Qwen2.5-7B-Instruct`
- Applies LoRA adapters with 4-bit loading via Unsloth
- Runs a minimal SFT training loop
- Saves artifacts and performs a quick inference preview

## Why I Built It

I wanted to understand the smallest honest fine-tuning loop I could explain in interviews:

`data filtering -> model loading -> LoRA injection -> training config -> artifact saving -> quick validation`

The goal is not to claim a production system, but to show a real training workflow that can be defended at code level.

## Workflow

```mermaid
flowchart LR
    A[Raw instruction dataset] --> B[Keyword-based filtering]
    B --> C[Train / eval split]
    C --> D[Chat template formatting]
    D --> E[Load Qwen2.5-7B-Instruct]
    E --> F[Inject LoRA adapters]
    F --> G[Run SFT training]
    G --> H[Save adapters and artifacts]
    H --> I[Quick inference preview]
```

## Tech Stack

- Python
- PyTorch
- Hugging Face Datasets
- TRL
- Unsloth
- LoRA / QLoRA-style workflow

## Repository Structure

```text
.
├── train_sft.py
├── README.md
├── requirements.txt
├── .gitignore
└── project_notes.md
```

## Training Choices In This Version

| Item | Current Choice |
| --- | --- |
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Data source | `shibing624/alpaca-zh` |
| Filtering idea | finance-related keyword filtering + minimum output length |
| Sequence length | `2048` |
| Quantization | `4-bit` loading |
| LoRA rank | `r=16` |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Batch size | `per_device_train_batch_size=2` |
| Gradient accumulation | `4` |
| Epochs | `1` |
| Learning rate | `2e-4` |

## Output Artifacts

The script is designed to produce several useful artifacts under the project directory, including:

- filtered training data
- filtered evaluation data
- dataset statistics
- training outputs
- saved LoRA adapters
- a quick inference preview for sanity checking

This is useful for explaining not only "how training was launched", but also "what was kept for later inspection".

## How To Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_sft.py
```

## What I Can Defend In Interviews

- why a local SFT loop is useful for learning model adaptation
- why LoRA is a parameter-efficient alternative to full fine-tuning
- why 4-bit loading and gradient accumulation matter under limited hardware
- why chat templates are important in instruction-style training
- why lightweight validation is still different from a full benchmark or product evaluation

## Key Points I Can Explain

- Why use LoRA instead of full fine-tuning
- Why 4-bit loading and gradient accumulation are useful in limited hardware settings
- Why chat templates matter in instruction tuning
- Why loss decrease is not the same thing as full business validation

## Current Boundaries

- Uses filtered public instruction data instead of a carefully labeled proprietary dataset
- Validation is still lightweight and not a full benchmark suite
- Focuses on a reproducible training loop rather than production deployment

## Future Improvements

- add stronger data quality checks beyond keyword filtering
- add structured evaluation prompts and metrics
- compare LoRA and QLoRA variants more explicitly
- separate training, evaluation, and artifact management into clearer modules

## Notes

This repository is intended as a learning and interview portfolio project rather than a production-ready training framework.
