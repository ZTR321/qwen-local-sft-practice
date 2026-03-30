# Qwen2.5-7B Local SFT Practice

A learning-oriented project that explores local instruction fine-tuning with Qwen2.5-7B under resource-constrained settings.

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
├── requirements.txt
├── .gitignore
└── project_notes.md
```

## How To Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_sft.py
```

## Key Points I Can Explain

- Why use LoRA instead of full fine-tuning
- Why 4-bit loading and gradient accumulation are useful in limited hardware settings
- Why chat templates matter in instruction tuning
- Why loss decrease is not the same thing as full business validation

## Current Boundaries

- Uses filtered public instruction data instead of a carefully labeled proprietary dataset
- Validation is still lightweight and not a full benchmark suite
- Focuses on a reproducible training loop rather than production deployment

## Notes

This repository is intended as a learning and interview portfolio project rather than a production-ready training framework.
