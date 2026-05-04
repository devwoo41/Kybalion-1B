"""
Phase 2: Continued Pre-Training (CPT) on the multi-domain dataset.

Loads the tokenized data produced by prepare_data.py and trains
meta-llama/Llama-3.2-1B (base) with a causal language modelling objective.
Resumes automatically from the latest checkpoint in output_dir.
"""

import argparse
import os

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def find_latest_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    checkpoints = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-")
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Continued Pre-Training")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with train/ and eval/ Arrow datasets (output of prepare_data.py)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for checkpoints and final model")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B",
                        help="HuggingFace model ID of the base model")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="If set, report metrics to this wandb project")
    args = parser.parse_args()

    hf_token = args.hf_token
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    # ── Model ────────────────────────────────
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
        attn_implementation="eager",  # required for gradient checkpointing
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    print(f"Dtype: {next(model.parameters()).dtype}")

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # ── Data ─────────────────────────────────
    train_dataset = load_from_disk(os.path.join(args.data_dir, "train"))
    eval_dataset = load_from_disk(os.path.join(args.data_dir, "eval"))

    print(f"Train: {len(train_dataset):,} sequences ({len(train_dataset) * args.max_seq_length / 1e9:.2f}B tokens)")
    print(f"Eval:  {len(eval_dataset):,} sequences")

    effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
    total_steps = len(train_dataset) // effective_batch
    print(f"Effective batch size: {effective_batch}")
    print(f"Total training steps: {total_steps:,}")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ── Training arguments ────────────────────
    use_wandb = args.wandb_project is not None
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        bf16=True,
        save_steps=args.save_steps,
        save_total_limit=5,
        num_train_epochs=1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        report_to="wandb" if use_wandb else "none",
        run_name="kybalion-1b-cpt" if use_wandb else None,
    )

    if use_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    # ── Trainer ──────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    checkpoint = find_latest_checkpoint(args.output_dir)
    if checkpoint:
        print(f"Resuming from checkpoint: {checkpoint}")
    else:
        print("No checkpoint found. Starting fresh training.")

    print(f"\n{'='*60}\nTRAINING START\n{'='*60}")
    trainer.train(resume_from_checkpoint=checkpoint)

    # ── Save final model ──────────────────────
    final_dir = os.path.join(args.output_dir, "final")
    print(f"\nSaving final CPT model to {final_dir}...")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    if trainer.state.log_history:
        train_losses = [x["loss"] for x in trainer.state.log_history if "loss" in x]
        eval_losses = [x["eval_loss"] for x in trainer.state.log_history if "eval_loss" in x]
        if train_losses:
            print(f"Train loss: {train_losses[0]:.4f} -> {train_losses[-1]:.4f}")
        if eval_losses:
            print(f"Eval loss:  {eval_losses[0]:.4f} -> {eval_losses[-1]:.4f}")

    print(f"\nCPT complete! Proceed to Phase 3 with: python train_sft.py --cpt_model_dir {final_dir}")


if __name__ == "__main__":
    main()
