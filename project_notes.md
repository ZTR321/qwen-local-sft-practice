# project_1_qwen_sft.md

## 项目名称
Qwen2.5-7B 本地化指令微调实践

## 一句话介绍
基于本地环境和开源模型 Qwen2.5-7B，完成垂直场景 instruction 数据处理、SFT 训练与推理验证，探索资源受限条件下的本地化微调方案。

## 背景
- 目标是让通用模型在特定场景下输出更稳定、更贴近目标领域风格
- 不依赖外部 API，强调本地化实验与资源受限场景可行性

## 我的职责
- 加载模型与 tokenizer
- 配置 LoRA / QLoRA 训练
- 准备 instruction 数据
- 运行训练并观察 loss / 样例输出
- 保存 adapter 并做基础推理验证

## 技术栈
- Python
- PyTorch
- Hugging Face
- TRL / SFTTrainer
- Unsloth
- LoRA / QLoRA
- 4-bit quantization

## 高频面试点
1. 为什么选 Qwen2.5-7B
2. 为什么做本地化微调
3. 为什么使用 LoRA / QLoRA
4. 为什么 target_modules 会包含 q_proj / v_proj 等层
5. 为什么要用 4-bit + gradient accumulation
6. 数据从哪里来，怎么清洗
7. 训练后保存的是完整模型还是 adapter
8. 怎么判断微调有效
9. loss 下降说明什么，不说明什么
10. 如果重做，会怎么补评测和数据质量控制

## 你要能讲出的最小闭环
背景问题 → 数据处理 → 模型加载 → LoRA 注入 → 训练配置 → loss / 样例观察 → adapter 保存 → 推理验证

## 易被追问的漏洞
- 把微调说成“灌新知识”
- 把 LoRA 工具说成算法创新
- 把 loss 改善直接等价成业务成功
- 对 q_proj / v_proj、4-bit、梯度累积讲不清
