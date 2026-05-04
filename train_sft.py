"""
Phase 3: LoRA-based Supervised Fine-Tuning (SFT).

Loads the CPT checkpoint produced by train_cpt.py, applies LoRA adapters,
trains on OpenHermes 2.5 + MetaMathQA + CodeAlpaca, then merges the adapters
back into the full weights and saves the final model.
"""

import argparse
import os

import torch
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# ──────────────────────────────────────────────
# Chat template (Llama 3.2 format)
# ──────────────────────────────────────────────

CHAT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are Prometheus, a helpful and knowledgeable AI assistant.<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{instruction}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{response}<|eot_id|>"
)


# ──────────────────────────────────────────────
# Dataset formatters (mirrors notebook exactly)
# ──────────────────────────────────────────────

def format_openhermes(example):
    convos = example.get("conversations", [])
    instruction = ""
    response = ""
    for msg in convos:
        if msg["from"] == "human":
            instruction = msg["value"]
        elif msg["from"] == "gpt":
            response = msg["value"]
    if instruction and response:
        return {"text": CHAT_TEMPLATE.format(instruction=instruction, response=response)}
    return {"text": ""}


def format_metamath(example):
    return {"text": CHAT_TEMPLATE.format(
        instruction=example.get("query", ""),
        response=example.get("response", ""),
    )}


def format_codealpaca(example):
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    if inp:
        instruction = f"{instruction}\n\nInput:\n{inp}"
    return {"text": CHAT_TEMPLATE.format(
        instruction=instruction,
        response=example.get("output", ""),
    )}


# ──────────────────────────────────────────────
# Checkpoint helper
# ──────────────────────────────────────────────

def find_latest_checkpoint(directory):
    if not os.path.isdir(directory):
        return None
    checkpoints = [
        os.path.join(directory, d)
        for d in os.listdir(directory)
        if d.startswith("checkpoint-")
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 3: LoRA SFT")
    parser.add_argument("--cpt_model_dir", type=str, required=True,
                        help="Path to the final CPT model (output of train_cpt.py)")
    parser.add_argument("--sft_output_dir", type=str, required=True,
                        help="Directory for SFT checkpoints")
    parser.add_argument("--final_merged_dir", type=str, required=True,
                        help="Directory for the merged (LoRA-free) final model")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--hermes_samples", type=int, default=100_000)
    parser.add_argument("--math_samples", type=int, default=50_000)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    args = parser.parse_args()

    hf_token = args.hf_token
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    assert os.path.exists(args.cpt_model_dir), (
        f"CPT model not found at {args.cpt_model_dir}. Run train_cpt.py first!"
    )

    # ── Model + LoRA ─────────────────────────
    print(f"Loading CPT model from {args.cpt_model_dir}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.cpt_model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.cpt_model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.1f}%)")

    # ── SFT Data ──────────────────────────────
    print("Loading OpenHermes 2.5...")
    hermes = load_dataset("teknium/OpenHermes-2.5", split="train", token=hf_token)
    hermes = hermes.shuffle(seed=42).select(range(min(args.hermes_samples, len(hermes))))
    hermes = hermes.map(format_openhermes, remove_columns=hermes.column_names)
    hermes = hermes.filter(lambda x: len(x["text"]) > 100)

    print("Loading MetaMathQA...")
    math_data = load_dataset("meta-math/MetaMathQA", split="train", token=hf_token)
    math_data = math_data.shuffle(seed=42).select(range(min(args.math_samples, len(math_data))))
    math_data = math_data.map(format_metamath, remove_columns=math_data.column_names)

    print("Loading CodeAlpaca-20k...")
    code_data = load_dataset("sahil2801/CodeAlpaca-20k", split="train", token=hf_token)
    code_data = code_data.map(format_codealpaca, remove_columns=code_data.column_names)

    sft_dataset = concatenate_datasets([hermes, math_data, code_data]).shuffle(seed=42)
    sft_split = sft_dataset.train_test_split(test_size=0.02, seed=42)
    sft_train = sft_split["train"]
    sft_eval = sft_split["test"]

    print(f"\nSFT Dataset:")
    print(f"  Train: {len(sft_train):,} examples")
    print(f"  Eval:  {len(sft_eval):,} examples")

    # ── Tokenize ──────────────────────────────
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )

    print("Tokenizing...")
    train_tokenized = sft_train.map(
        tokenize_fn, batched=True, remove_columns=["text"],
        num_proc=2, desc="Tokenizing train",
    )
    eval_tokenized = sft_eval.map(
        tokenize_fn, batched=True, remove_columns=["text"],
        num_proc=2, desc="Tokenizing eval",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    # ── Training ──────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.sft_output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0.01,
        bf16=True,
        num_train_epochs=args.num_train_epochs,
        save_steps=500,
        save_total_limit=3,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=500,
        report_to="none",
        gradient_checkpointing=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=data_collator,
    )

    sft_checkpoint = find_latest_checkpoint(args.sft_output_dir)
    if sft_checkpoint:
        print(f"Resuming SFT from checkpoint: {sft_checkpoint}")
    else:
        print("No checkpoint found. Starting fresh SFT training.")

    print("Starting SFT training...")
    trainer.train(resume_from_checkpoint=sft_checkpoint)

    # ── Merge LoRA → full weights ─────────────
    checkpoints = [
        os.path.join(args.sft_output_dir, d)
        for d in os.listdir(args.sft_output_dir)
        if d.startswith("checkpoint-")
    ]
    assert checkpoints, f"No checkpoints found in {args.sft_output_dir}"
    latest_ckpt = max(checkpoints, key=os.path.getmtime)
    print(f"\nMerging LoRA weights from {latest_ckpt}...")

    base_model = AutoModelForCausalLM.from_pretrained(
        args.cpt_model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    merged_model = PeftModel.from_pretrained(base_model, latest_ckpt)
    merged_model = merged_model.merge_and_unload()

    os.makedirs(args.final_merged_dir, exist_ok=True)
    merged_model.save_pretrained(args.final_merged_dir)
    tokenizer.save_pretrained(args.final_merged_dir)

    print(f"Final merged model saved to: {args.final_merged_dir}")
    print(f"Model size: {sum(p.numel() for p in merged_model.parameters()) / 1e9:.2f}B parameters")

    # ── Quick inference test ──────────────────
    test_prompts = [
        "Explain the theory of relativity in simple terms.",
        "Write a Python function that checks if a number is prime.",
        "What is 15% of 240?",
    ]
    for prompt in test_prompts:
        formatted = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful and knowledgeable AI assistant.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = merged_model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        print(f"\n{'='*60}\nQ: {prompt}\nA: {response[:500]}")

    print(f"\n{'='*60}")
    print(f"SFT complete! Proceed to evaluation: python evaluate.py --model_path {args.final_merged_dir}")


if __name__ == "__main__":
    main()
