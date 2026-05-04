"""
Phase 1: Multi-domain data preparation for CPT.

Streams data from 6 domains, tokenizes with the Llama 3.2 tokenizer,
packs into fixed-length sequences, and saves to disk with chunk checkpointing.
Every 10,000 sequences are written to a chunk directory so the job can be
resumed after interruption without re-processing earlier documents.
"""

import argparse
import gc
import glob as glob_module
import json
import os
import random

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from tqdm.auto import tqdm
from transformers import AutoTokenizer


# ──────────────────────────────────────────────
# Domain configuration (mirrors notebook exactly)
# ──────────────────────────────────────────────

TOTAL_TOKENS_TARGET = 3_500_000_000  # 3.5B tokens

DOMAIN_CONFIG = {
    "fineweb_edu": {
        "ratio": 0.35,
        "tokens": int(TOTAL_TOKENS_TARGET * 0.35),
        "dataset_id": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-100BT",
        "data_dir": None,
        "text_field": "text",
        "filter_fn": lambda x: x.get("score", 0) >= 3.0,
    },
    "openwebmath": {
        "ratio": 0.20,
        "tokens": int(TOTAL_TOKENS_TARGET * 0.20),
        "dataset_id": "open-web-math/open-web-math",
        "subset": None,
        "data_dir": None,
        "text_field": "text",
        "filter_fn": None,
    },
    "stack_python": {
        "ratio": 0.15,
        "tokens": int(TOTAL_TOKENS_TARGET * 0.15),
        "dataset_id": "bigcode/starcoderdata",
        "subset": None,
        "data_dir": "python",
        "text_field": "content",
        "filter_fn": None,
    },
    "cosmopedia": {
        "ratio": 0.15,
        "tokens": int(TOTAL_TOKENS_TARGET * 0.15),
        "dataset_id": "HuggingFaceTB/cosmopedia",
        "subset": "web_samples_v2",
        "data_dir": None,
        "text_field": "text",
        "filter_fn": None,
    },
    "cosmopedia_science": {
        "ratio": 0.10,
        "tokens": int(TOTAL_TOKENS_TARGET * 0.10),
        "dataset_id": "HuggingFaceTB/cosmopedia",
        "subset": "stanford",
        "data_dir": None,
        "text_field": "text",
        "filter_fn": None,
    },
    "cosmopedia_stories": {
        "ratio": 0.05,
        "tokens": int(TOTAL_TOKENS_TARGET * 0.05),
        "dataset_id": "HuggingFaceTB/cosmopedia",
        "subset": "stories",
        "data_dir": None,
        "text_field": "text",
        "filter_fn": None,
    },
}


# ──────────────────────────────────────────────
# Chunk checkpoint helpers
# ──────────────────────────────────────────────

def get_existing_chunks(domain_dir):
    """Return (list_of_valid_chunk_paths, total_sequences_saved)."""
    if not os.path.exists(domain_dir):
        return [], 0

    chunk_dirs = sorted(glob_module.glob(os.path.join(domain_dir, "chunk_*")))
    total_seqs = 0
    valid_chunks = []

    for chunk_dir in chunk_dirs:
        try:
            ds = Dataset.load_from_disk(chunk_dir)
            total_seqs += len(ds)
            valid_chunks.append(chunk_dir)
            del ds
        except Exception:
            import shutil
            shutil.rmtree(chunk_dir, ignore_errors=True)

    return valid_chunks, total_seqs


def save_chunk(domain_dir, chunk_idx, input_ids_list):
    """Write a list of token sequences as an Arrow dataset chunk."""
    chunk_path = os.path.join(domain_dir, f"chunk_{chunk_idx:04d}")
    ds = Dataset.from_dict({
        "input_ids": input_ids_list,
        "labels": [ids[:] for ids in input_ids_list],
    })
    ds.save_to_disk(chunk_path)
    return chunk_path


def load_domain_from_chunks(domain_dir):
    """Concatenate all saved chunks into a single Dataset."""
    chunk_dirs = sorted(glob_module.glob(os.path.join(domain_dir, "chunk_*")))
    if not chunk_dirs:
        return None

    datasets = []
    for chunk_dir in chunk_dirs:
        try:
            ds = Dataset.load_from_disk(chunk_dir)
            datasets.append(ds)
        except Exception as e:
            print(f"  Warning: failed to load {chunk_dir}: {e}")

    if not datasets:
        return None

    return concatenate_datasets(datasets)


# ──────────────────────────────────────────────
# Core collection function
# ──────────────────────────────────────────────

def collect_and_tokenize_domain(domain_name, config, tokenizer, max_seq_length,
                                chunk_size, domain_dir, hf_token):
    """
    Stream a single domain, tokenize, pack into max_seq_length chunks,
    and save to disk every chunk_size sequences.
    Resumes automatically from any previously saved chunks.
    """
    target_tokens = config["tokens"]
    target_sequences = target_tokens // max_seq_length

    os.makedirs(domain_dir, exist_ok=True)
    existing_chunks, existing_seqs = get_existing_chunks(domain_dir)

    print(f"\n{'='*60}")
    print(f"Processing: {domain_name}")
    print(f"  Target: {target_tokens/1e6:.0f}M tokens ({target_sequences:,} sequences)")
    print(f"  Dataset: {config['dataset_id']}")
    if existing_seqs > 0:
        print(f"  Resuming: {existing_seqs:,} sequences already saved ({len(existing_chunks)} chunks)")
        print(f"  Remaining: {target_sequences - existing_seqs:,} sequences")
    print(f"{'='*60}")

    if existing_seqs >= target_sequences:
        print(f"  Already complete! ({existing_seqs:,} >= {target_sequences:,})")
        return

    remaining_sequences = target_sequences - existing_seqs
    next_chunk_idx = len(existing_chunks)

    load_kwargs = {
        "path": config["dataset_id"],
        "split": "train",
        "streaming": True,
        "token": hf_token,
        "trust_remote_code": True,
    }
    if config.get("subset"):
        load_kwargs["name"] = config["subset"]
    if config.get("data_dir"):
        load_kwargs["data_dir"] = config["data_dir"]

    ds_stream = load_dataset(**load_kwargs)

    if config["filter_fn"]:
        ds_stream = ds_stream.filter(config["filter_fn"])

    # Skip documents already processed in a previous run
    if existing_seqs > 0:
        estimated_tokens_done = existing_seqs * max_seq_length
        print(f"  Skipping ~{estimated_tokens_done/1e6:.0f}M tokens worth of documents...")
        tokens_skipped = 0
        text_field = config["text_field"]
        skip_pbar = tqdm(total=int(estimated_tokens_done), desc="  Skipping", unit="tok")
        for example in ds_stream:
            text = example.get(text_field, "")
            if not text or len(text.strip()) < 50:
                continue
            est_tokens = len(text) // 4
            tokens_skipped += est_tokens
            skip_pbar.update(est_tokens)
            if tokens_skipped >= estimated_tokens_done:
                break
        skip_pbar.close()

    text_field = config["text_field"]
    token_buffer = []
    chunk_buffer = []
    total_new_seqs = 0

    pbar = tqdm(total=remaining_sequences, desc=f"  Packing {domain_name}", unit="seq")

    for example in ds_stream:
        text = example.get(text_field, "")
        if not text or len(text.strip()) < 50:
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_buffer.extend(tokens)
        token_buffer.append(tokenizer.eos_token_id)

        while len(token_buffer) >= max_seq_length:
            chunk_buffer.append(token_buffer[:max_seq_length])
            token_buffer = token_buffer[max_seq_length:]
            total_new_seqs += 1
            pbar.update(1)

            if len(chunk_buffer) >= chunk_size:
                save_path = save_chunk(domain_dir, next_chunk_idx, chunk_buffer)
                print(f"\n  Chunk {next_chunk_idx} saved: {len(chunk_buffer):,} seqs -> {save_path}")
                next_chunk_idx += 1
                chunk_buffer = []
                gc.collect()

            if total_new_seqs >= remaining_sequences:
                break

        if total_new_seqs >= remaining_sequences:
            break

    pbar.close()

    if chunk_buffer:
        save_path = save_chunk(domain_dir, next_chunk_idx, chunk_buffer)
        print(f"  Final chunk {next_chunk_idx} saved: {len(chunk_buffer):,} seqs -> {save_path}")
        del chunk_buffer

    complete_marker = os.path.join(domain_dir, "_COMPLETE")
    total_seqs = existing_seqs + total_new_seqs
    with open(complete_marker, "w") as f:
        f.write(str(total_seqs))

    actual_tokens = total_seqs * max_seq_length
    print(f"\n  {domain_name} COMPLETE: {total_seqs:,} sequences ({actual_tokens/1e6:.0f}M tokens)")
    del token_buffer
    gc.collect()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Prepare CPT data")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write tokenized data (train/ and eval/ subdirs)")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"),
                        help="HuggingFace access token (or set HF_TOKEN env var)")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--chunk_size", type=int, default=10_000,
                        help="Save a checkpoint every N sequences per domain")
    args = parser.parse_args()

    hf_token = args.hf_token
    if not hf_token:
        raise ValueError("HuggingFace token required. Pass --hf_token or set HF_TOKEN env var.")
    os.environ["HF_TOKEN"] = hf_token

    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        token=hf_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("=" * 60)
    print("Kybalion-1B Data Mix Configuration")
    print("=" * 60)
    for name, cfg in DOMAIN_CONFIG.items():
        print(f"  {name:22s} | {cfg['ratio']*100:5.1f}% | {cfg['tokens']/1e9:.3f}B tokens")
    print(f"  {'TOTAL':22s} | 100.0% | {TOTAL_TOKENS_TARGET/1e9:.3f}B tokens")
    print(f"  Sequence length: {args.max_seq_length}")
    print(f"  Chunk checkpoint: every {args.chunk_size:,} sequences")

    # Collect each domain
    domain_datasets = {}
    for domain_name, config in DOMAIN_CONFIG.items():
        domain_dir = os.path.join(args.output_dir, f"domain_{domain_name}")
        complete_marker = os.path.join(domain_dir, "_COMPLETE")

        if os.path.exists(complete_marker):
            print(f"\n[SKIP] {domain_name} - already complete")
            ds = load_domain_from_chunks(domain_dir)
            if ds is not None:
                domain_datasets[domain_name] = ds
                print(f"  Loaded: {len(ds):,} sequences")
            continue

        collect_and_tokenize_domain(
            domain_name, config, tokenizer, args.max_seq_length,
            args.chunk_size, domain_dir, hf_token,
        )

        ds = load_domain_from_chunks(domain_dir)
        if ds is not None:
            domain_datasets[domain_name] = ds

        gc.collect()

    print(f"\n{'='*60}")
    print("All domains collected!")
    for name, ds in domain_datasets.items():
        print(f"  {name:22s}: {len(ds):,} sequences")
    total_seqs = sum(len(ds) for ds in domain_datasets.values())
    print(f"  {'TOTAL':22s}: {total_seqs:,} sequences ({total_seqs * args.max_seq_length / 1e9:.2f}B tokens)")

    # Shuffle + train/eval split
    print("\nConcatenating all domains...")
    combined = concatenate_datasets(list(domain_datasets.values()))
    print(f"  Combined: {len(combined):,} sequences")

    print("Shuffling...")
    combined = combined.shuffle(seed=42)

    split = combined.train_test_split(test_size=0.01, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"\n  Train: {len(train_dataset):,} sequences ({len(train_dataset) * args.max_seq_length / 1e9:.2f}B tokens)")
    print(f"  Eval:  {len(eval_dataset):,} sequences ({len(eval_dataset) * args.max_seq_length / 1e6:.0f}M tokens)")

    train_path = os.path.join(args.output_dir, "train")
    eval_path = os.path.join(args.output_dir, "eval")

    print(f"\nSaving train dataset to {train_path}...")
    train_dataset.save_to_disk(train_path)

    print(f"Saving eval dataset to {eval_path}...")
    eval_dataset.save_to_disk(eval_path)

    # Verify
    train_check = load_from_disk(train_path)
    eval_check = load_from_disk(eval_path)
    print("\n=== Verification ===")
    print(f"Train: {len(train_check):,} sequences")
    print(f"Eval:  {len(eval_check):,} sequences")
    print(f"Features: {train_check.features}")
    sample = train_check[0]["input_ids"]
    print(f"Sample sequence length: {len(sample)}")
    print(f"First 100 tokens decoded: {tokenizer.decode(sample[:100])}")
    print(f"\nTotal train tokens: {len(train_check) * args.max_seq_length / 1e9:.2f}B")
    print(f"Data preparation complete!")


if __name__ == "__main__":
    main()
