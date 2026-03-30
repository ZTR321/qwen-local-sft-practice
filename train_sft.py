from __future__ import annotations

import json
from pathlib import Path

import unsloth
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer


DATASET_NAME = "shibing624/alpaca-zh"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_DIR / "project_1_artifacts"
TRAIN_JSON_PATH = ARTIFACT_DIR / "financial_sft_train.json"
EVAL_JSON_PATH = ARTIFACT_DIR / "financial_sft_eval.json"
STATS_JSON_PATH = ARTIFACT_DIR / "financial_sft_stats.json"
PREVIEW_JSON_PATH = ARTIFACT_DIR / "quick_inference_preview.json"
TRAIN_OUTPUT_DIR = ARTIFACT_DIR / "outputs"
ADAPTER_OUTPUT_DIR = ARTIFACT_DIR / "qwen2_5_7b_fin_lora"
MAX_SEQ_LENGTH = 2048
SEED = 42
EVAL_RATIO = 0.1
MAX_TRAIN_SAMPLES = None

FINANCIAL_KEYWORDS = [
    "金融",
    "经济",
    "投资",
    "财务",
    "股票",
    "基金",
    "财报",
    "营收",
    "利润",
    "市盈率",
    "量化宽松",
    "美联储",
    "降息",
    "资产负债表",
    "现金流",
    "审计",
    "宏观",
    "微观",
    "证券",
    "债券",
    "私募",
    "风控",
    "合规",
]


def normalize_text(value: object) -> str:
    return str(value or "").strip()


def build_user_prompt(instruction: str, input_text: str) -> str:
    parts = [f"任务指令：{instruction}"]
    if input_text:
        parts.append(f"补充信息：{input_text}")
    return "\n\n".join(parts)


def is_financial_candidate(example: dict) -> bool:
    instruction = normalize_text(example.get("instruction"))
    input_text = normalize_text(example.get("input"))
    output_text = normalize_text(example.get("output"))
    source_text = f"{instruction}\n{input_text}"
    has_keyword = any(keyword in source_text for keyword in FINANCIAL_KEYWORDS)
    output_length_ok = len(output_text) > 50
    return has_keyword and output_length_ok


def ensure_artifact_dir() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def prepare_datasets():
    print("正在下载并筛选中文 instruction 数据集，请稍候...")
    raw_dataset = load_dataset(DATASET_NAME, split="train")
    filtered_dataset = raw_dataset.filter(is_financial_candidate)

    if len(filtered_dataset) == 0:
        raise ValueError("筛选后没有可训练样本，请调整关键词或过滤规则。")

    filtered_dataset = filtered_dataset.shuffle(seed=SEED)

    if MAX_TRAIN_SAMPLES is not None:
        filtered_dataset = filtered_dataset.select(
            range(min(MAX_TRAIN_SAMPLES, len(filtered_dataset)))
        )

    eval_size = max(1, int(len(filtered_dataset) * EVAL_RATIO))
    if len(filtered_dataset) <= 5:
        train_dataset = filtered_dataset
        eval_dataset = filtered_dataset.select([])
    else:
        split_dataset = filtered_dataset.train_test_split(test_size=eval_size, seed=SEED)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

    train_dataset.to_json(str(TRAIN_JSON_PATH), force_ascii=False)
    eval_dataset.to_json(str(EVAL_JSON_PATH), force_ascii=False)

    stats = {
        "data_source": DATASET_NAME,
        "seed": SEED,
        "raw_sample_count": len(raw_dataset),
        "filtered_sample_count": len(filtered_dataset),
        "train_sample_count": len(train_dataset),
        "eval_sample_count": len(eval_dataset),
        "max_train_samples": MAX_TRAIN_SAMPLES,
        "keyword_count": len(FINANCIAL_KEYWORDS),
    }
    STATS_JSON_PATH.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        "数据准备完成："
        f"原始样本 {stats['raw_sample_count']} 条，"
        f"筛选后 {stats['filtered_sample_count']} 条，"
        f"训练集 {stats['train_sample_count']} 条，"
        f"验证集 {stats['eval_sample_count']} 条。"
    )
    print(f"训练集已保存到：{TRAIN_JSON_PATH.name}")
    print(f"验证集已保存到：{EVAL_JSON_PATH.name}")
    return train_dataset, eval_dataset


def format_for_sft(dataset, tokenizer):
    def apply_chat_template(example: dict) -> dict:
        instruction = normalize_text(example.get("instruction"))
        input_text = normalize_text(example.get("input"))
        output_text = normalize_text(example.get("output"))
        messages = [
            {"role": "user", "content": build_user_prompt(instruction, input_text)},
            {"role": "assistant", "content": output_text},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    return dataset.map(
        apply_chat_template,
        remove_columns=dataset.column_names,
    )


def load_model_and_tokenizer():
    print("正在加载 Qwen2.5-7B-Instruct 基座模型...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def attach_lora(model):
    print("正在注入 LoRA 适配器...")
    return FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )


def build_training_args(enable_eval: bool) -> SFTConfig:
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args_kwargs = {
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 1,
        "learning_rate": 2e-4,
        "logging_steps": 5,
        "output_dir": str(TRAIN_OUTPUT_DIR),
        "optim": "adamw_8bit",
        "fp16": torch.cuda.is_available() and not use_bf16,
        "bf16": use_bf16,
        "report_to": "none",
        "seed": SEED,
        "save_total_limit": 2,
        "dataset_text_field": "text",
        "dataset_num_proc": 2,
        "max_length": MAX_SEQ_LENGTH,
        "eos_token": "<|im_end|>",
    }
    if enable_eval:
        args_kwargs.update(
            {
                "per_device_eval_batch_size": 2,
                "eval_strategy": "steps",
                "eval_steps": 20,
                "save_strategy": "steps",
                "save_steps": 20,
            }
        )
    else:
        args_kwargs.update(
            {
                "eval_strategy": "no",
                "save_strategy": "epoch",
            }
        )
    return SFTConfig(**args_kwargs)


def run_quick_inference(model, tokenizer, eval_dataset) -> None:
    if len(eval_dataset) == 0:
        print("没有验证集样本，跳过快速推理预览。")
        return

    sample = eval_dataset[0]
    user_prompt = build_user_prompt(
        normalize_text(sample.get("instruction")),
        normalize_text(sample.get("input")),
    )
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    FastLanguageModel.for_inference(model)
    device = next(model.parameters()).device
    inputs = tokenizer([prompt_text], return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        temperature=0.0,
        use_cache=True,
    )
    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][prompt_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    preview = {
        "instruction": normalize_text(sample.get("instruction")),
        "input": normalize_text(sample.get("input")),
        "reference_output": normalize_text(sample.get("output")),
        "generated_output": generated_text,
    }
    PREVIEW_JSON_PATH.write_text(
        json.dumps(preview, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"快速推理预览已保存到：{PREVIEW_JSON_PATH.name}")


def main():
    ensure_artifact_dir()
    train_source_dataset, eval_source_dataset = prepare_datasets()

    model, tokenizer = load_model_and_tokenizer()
    model = attach_lora(model)

    train_dataset = format_for_sft(train_source_dataset, tokenizer)
    eval_dataset = format_for_sft(eval_source_dataset, tokenizer)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
        args=build_training_args(enable_eval=len(eval_dataset) > 0),
    )

    print("开始进行金融场景 SFT 微调...")
    train_result = trainer.train()
    eval_metrics = trainer.evaluate() if len(eval_dataset) > 0 else {}

    model.save_pretrained(str(ADAPTER_OUTPUT_DIR))
    tokenizer.save_pretrained(str(ADAPTER_OUTPUT_DIR))
    print(f"LoRA 适配器已保存到：{ADAPTER_OUTPUT_DIR.name}")

    run_summary = {
        "train_loss": getattr(train_result, "training_loss", None),
        "eval_metrics": eval_metrics,
        "adapter_output_dir": str(ADAPTER_OUTPUT_DIR),
        "train_json_path": str(TRAIN_JSON_PATH),
        "eval_json_path": str(EVAL_JSON_PATH),
    }
    (ARTIFACT_DIR / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    try:
        run_quick_inference(model, tokenizer, eval_source_dataset)
    except Exception as exc:
        print(f"快速推理预览失败，但不影响训练结果保存：{exc}")


if __name__ == "__main__":
    main()
