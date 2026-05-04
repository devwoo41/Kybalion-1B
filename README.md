# Kybalion-1B: Experiment Code

This repository contains the training and evaluation code for **Kybalion-1B**, as described in:

> **Kybalion-1B: A Compute-Efficient Continued Pretraining and Instruction Tuning Pipeline for 1B-Scale Language Models**  
> Wooju Lee, Hankuk University of Foreign Studies, 2026.

Model weights are available at: [devwoo/Kybalion-1B](https://huggingface.co/devwoo/Kybalion-1B)  
GGUF (quantized) version: [devwoo/Kybalion-1B-GGUF](https://huggingface.co/devwoo/Kybalion-1B-GGUF)

---

## Overview

Kybalion-1B is trained through a two-stage pipeline starting from `meta-llama/Llama-3.2-1B`:

1. **Continued Pre-Training (CPT)** — ~3.5B tokens, 6-domain balanced mixture
2. **LoRA-based Supervised Fine-Tuning (SFT)** — instruction, math, code datasets

All experiments were run on **Google Colab A100 80GB** using the Jupyter notebooks in this directory.

---

## Notebooks

All notebooks are designed to run sequentially on **Google Colab (A100, High-RAM)**. Each notebook is self-contained and resumes from Google Drive checkpoints.

| Notebook | Description | Est. Time (A100) |
|----------|-------------|-----------------|
| `01_prepare_data.ipynb` | Download and tokenize multi-domain CPT data | 3–5 hours |
| `02_train_cpt.ipynb` | Continued pre-training with HuggingFace Trainer | 2.5–3 days |
| `03_train_sft.ipynb` | LoRA-based SFT, merge and save final model | 3–5 hours |
| `04_evaluate.ipynb` | Benchmark evaluation with lm-evaluation-harness | 2–3 hours |

---

## Requirements

```bash
pip install -r requirements.txt
```

> **Note:** These notebooks are designed for Google Colab. Each notebook installs its own dependencies via `!pip install` cells at the top.

---

## Data Mixture (CPT)

| Domain | Dataset | Ratio |
|--------|---------|-------|
| Education | [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 35% |
| Mathematics | [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math) | 20% |
| Code | [StarCoderData (Python)](https://huggingface.co/datasets/bigcode/starcoderdata) | 15% |
| Textbook | [Cosmopedia web_samples_v2](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) | 15% |
| Science | [Cosmopedia stanford](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) | 10% |
| Story | [Cosmopedia stories](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) | 5% |

---

## Training Configuration

### CPT Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Base model | `meta-llama/Llama-3.2-1B` |
| Learning rate | 2e-5 |
| LR schedule | Cosine decay |
| Warmup steps | 1,000 |
| Weight decay | 0.1 |
| Effective batch size | 32 (4 × 8 grad accum) |
| Max sequence length | 4,096 |
| Precision | BF16 |
| Optimizer | AdamW |
| Max steps | ~25,000 |
| Total tokens | ~3.5B |

### SFT Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Method | LoRA |
| Rank (r) | 64 |
| Alpha | 128 |
| Dropout | 0.05 |
| Target modules | q/k/v/o/gate/up/down proj |
| Learning rate | 1e-4 |
| LR schedule | Cosine decay |
| Epochs | 3 |

---

## Benchmark Results

All scores measured with `lm-evaluation-harness` under identical conditions (bfloat16, batch_size=8, A100).

| Benchmark | Few-shot | TinyLlama-1.1B | Llama-3.2-1B-Instruct | **Kybalion-1B** |
|-----------|:--------:|:--------------:|:---------------------:|:---------------:|
| MMLU | 5 | 25.0% | 46.1% | **32.0%** |
| ARC-Challenge | 25 | 37.2% | 41.5% | **37.6%** |
| GSM8K | 5 | 2.4% | 33.5% | **10.8%** |
| HellaSwag | 10 | 61.2% | 61.1% | **63.8%** ★ |
| WinoGrande | 5 | 61.8% | 62.4% | **62.4%** ★ |
| TruthfulQA | 0 | 37.4% | 43.3% | **40.0%** |

★ Outperforms or matches Llama-3.2-1B-Instruct

---

## Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("devwoo/Kybalion-1B")
model = AutoModelForCausalLM.from_pretrained(
    "devwoo/Kybalion-1B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

def chat(user_message, system="You are a helpful and knowledgeable AI assistant."):
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_message}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print(chat("Explain the Pythagorean theorem with an example."))
```

---

## Setup for Colab

Each notebook mounts Google Drive automatically. Before running:

1. Upload your HuggingFace token as a Colab Secret named `HF_TOKEN`
2. Run notebooks in order: `01` → `02` → `03` → `04`
3. Checkpoints are saved to Google Drive and resumed automatically across sessions

---

## License

This code is released under the **MIT License**.  
Model weights follow the [Llama 3.2 Community License](https://ai.meta.com/llama/license/).
