"""
Phase 4: Benchmark evaluation with lm-evaluation-harness.

Evaluates Kybalion-1B (and optionally baseline models) on 6 standard benchmarks
and produces a bar chart and radar chart for the paper.

Usage:
    python evaluate.py \
        --model_path /path/to/Kybalion-1B \
        --results_dir ./results \
        [--eval_baselines]          # also evaluate TinyLlama and Llama-3.2-1B-Instruct
        [--hf_token YOUR_TOKEN]
"""

import argparse
import json
import os

import lm_eval
import matplotlib.pyplot as plt
import numpy as np


# ──────────────────────────────────────────────
# Benchmark configuration (mirrors notebook)
# ──────────────────────────────────────────────

BENCHMARKS = [
    ("MMLU",          "mmlu",           5),
    ("ARC-Challenge",  "arc_challenge",  25),
    ("GSM8K",          "gsm8k",          5),
    ("HellaSwag",      "hellaswag",      10),
    ("WinoGrande",     "winogrande",     5),
    ("TruthfulQA",     "truthfulqa_mc2", 0),
]

# Key used to extract the primary metric from lm_eval results
BENCHMARK_KEYS = {
    "MMLU":          ("mmlu",           "acc,none"),
    "ARC-Challenge":  ("arc_challenge",  "acc_norm,none"),
    "GSM8K":          ("gsm8k",          "exact_match,strict-match"),
    "HellaSwag":      ("hellaswag",      "acc_norm,none"),
    "WinoGrande":     ("winogrande",     "acc,none"),
    "TruthfulQA":     ("truthfulqa_mc2", "acc,none"),
}

BASELINE_MODELS = {
    "TinyLlama-1.1B":        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    # Phi-1.5 excluded: PhiConfig incompatibility with modern transformers versions
}


# ──────────────────────────────────────────────
# Score extraction
# ──────────────────────────────────────────────

def extract_score(result_dict, task_name, metric_key):
    """Pull a scalar score from an lm_eval result dict; returns None on failure."""
    try:
        results = result_dict.get("results", {})
        for key, vals in results.items():
            if task_name in key:
                score = vals.get(metric_key)
                if score is not None:
                    return round(score * 100, 2)
        return None
    except Exception:
        return None


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

def evaluate_model(model_display_name, model_path, results_dir, hf_token):
    """Run all benchmarks for one model; skip benchmarks already on disk."""
    model_results = {}

    for bench_name, task, fewshot in BENCHMARKS:
        safe_name = model_display_name.replace("/", "_")
        output_file = os.path.join(results_dir, f"{safe_name}_{task}.json")

        if os.path.exists(output_file):
            print(f"  [SKIP] {bench_name} already evaluated", flush=True)
            with open(output_file) as f:
                model_results[bench_name] = json.load(f)
            continue

        print(f"  Running {bench_name} ({fewshot}-shot)...", flush=True)
        try:
            model_args = f"pretrained={model_path},dtype=bfloat16,trust_remote_code=True"
            if hf_token:
                model_args += f",token={hf_token}"

            results = lm_eval.simple_evaluate(
                model="hf",
                model_args=model_args,
                tasks=[task],
                num_fewshot=fewshot,
                batch_size=8,
            )
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            model_results[bench_name] = results
            print(f"  Done: {bench_name}", flush=True)

        except Exception as e:
            print(f"  Error on {bench_name}: {e}", flush=True)
            model_results[bench_name] = {"error": str(e)}

    return model_results


# ──────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────

def parse_scores(all_results):
    """Convert raw lm_eval dicts into {model: {benchmark: score}} structure."""
    parsed = {}
    bench_names = list(BENCHMARK_KEYS.keys())

    for model_name, bench_results in all_results.items():
        parsed[model_name] = {}
        for display_name in bench_names:
            # The key in bench_results matches the first element of BENCHMARKS tuple
            bench_data = bench_results.get(display_name, {})
            task_name, metric_key = BENCHMARK_KEYS[display_name]
            score = extract_score(bench_data, task_name, metric_key)
            parsed[model_name][display_name] = score if score is not None else 0.0

    return parsed


def plot_bar(parsed_scores, results_dir):
    benchmarks = list(BENCHMARK_KEYS.keys())
    models = list(parsed_scores.keys())
    colors = ["#96CEB4", "#FF6B6B", "#4ECDC4"]

    x = np.arange(len(benchmarks))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(14, 8))
    for i, (model_name, color) in enumerate(zip(models, colors)):
        scores = [parsed_scores[model_name].get(b, 0) for b in benchmarks]
        bars = ax.bar(x + i * width, scores, width, label=model_name,
                      color=color, edgecolor="white")
        for bar, score in zip(bars, scores):
            if score > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                        f"{score:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Benchmark", fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title(
        "Kybalion-1B vs Competitive 1B Models\n"
        "(All scores measured under identical conditions with lm-evaluation-harness)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(benchmarks, fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    out = os.path.join(results_dir, "benchmark_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Bar chart saved to {out}")


def plot_radar(parsed_scores, results_dir):
    benchmarks = list(BENCHMARK_KEYS.keys())
    N = len(benchmarks)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    colors = ["#96CEB4", "#FF6B6B", "#4ECDC4", "#45B7D1"]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    for (model_name, scores), color in zip(parsed_scores.items(), colors):
        values = [scores.get(b, 0) for b in benchmarks] + [scores.get(benchmarks[0], 0)]
        ax.plot(angles, values, "o-", linewidth=2, label=model_name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(benchmarks, fontsize=11)
    ax.set_ylim(0, 80)
    ax.set_title("Kybalion-1B: Multi-Benchmark Radar", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()
    out = os.path.join(results_dir, "radar_chart.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Radar chart saved to {out}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Benchmark Evaluation")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the final merged model (output of train_sft.py)")
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="Directory for JSON results and chart images")
    parser.add_argument("--eval_baselines", action="store_true",
                        help="Also evaluate TinyLlama-1.1B and Llama-3.2-1B-Instruct")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    hf_token = args.hf_token
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    os.makedirs(args.results_dir, exist_ok=True)

    # Build model list: Kybalion-1B always first
    eval_models = {"Kybalion-1B": args.model_path}
    if args.eval_baselines:
        eval_models.update(BASELINE_MODELS)

    all_results = {}
    for model_display_name, model_path in eval_models.items():
        print(f"\n{'='*60}\nEvaluating: {model_display_name}\n{'='*60}", flush=True)
        all_results[model_display_name] = evaluate_model(
            model_display_name, model_path, args.results_dir, hf_token
        )

    print(f"\n{'='*60}\nAll evaluations complete!\n{'='*60}")

    # Parse and display scores
    parsed_scores = parse_scores(all_results)

    print("\nBenchmark Results (directly measured, identical conditions)")
    print("=" * 70)
    header = f"{'Benchmark':14s}" + "".join(f"  {m:25s}" for m in parsed_scores)
    print(header)
    print("-" * len(header))
    for bench in BENCHMARK_KEYS:
        row = f"{bench:14s}"
        for model_name in parsed_scores:
            score = parsed_scores[model_name].get(bench, 0.0)
            row += f"  {score:5.1f}%{'':20s}"
        print(row)
    print("=" * 70)

    comparison_path = os.path.join(args.results_dir, "final_comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(parsed_scores, f, indent=2)
    print(f"\nFull results saved to {comparison_path}")

    # Plots
    plot_bar(parsed_scores, args.results_dir)
    plot_radar(parsed_scores, args.results_dir)

    print(f"\nEvaluation complete!")


if __name__ == "__main__":
    main()
