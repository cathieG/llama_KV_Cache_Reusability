import json
import math
import argparse
import statistics
from collections import defaultdict

import evaluate
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------
# METRIC SETUP
# ---------------------------
bleu_metric = evaluate.load("sacrebleu")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ---------------------------
# IO
# ---------------------------
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ---------------------------
# BASIC HELPERS
# ---------------------------
def mean_or_none(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def token_accuracy(a, b, horizon=20):
    a = (a or [])[:horizon]
    b = (b or [])[:horizon]
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    matches = sum(1 for x, y in zip(a, b) if x == y)
    return matches / n


def tokens_until_first_deviation(a, b, horizon=20):
    a = (a or [])[:horizon]
    b = (b or [])[:horizon]
    n = min(len(a), len(b))
    count = 0
    for i in range(n):
        if a[i] == b[i]:
            count += 1
        else:
            break
    return count


# ---------------------------
# KL DIVERGENCE
# ---------------------------
def softmax_np(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits)
    exps = np.exp(logits)
    denom = np.sum(exps)
    if denom == 0:
        return None
    return exps / denom


def kl_divergence_from_logits(p_logits, q_logits, eps=1e-12):
    """
    Computes KL(P || Q) from raw logits.
    P = native distribution
    Q = transfer distribution
    """
    if p_logits is None or q_logits is None:
        return None

    p = softmax_np(p_logits)
    q = softmax_np(q_logits)

    if p is None or q is None:
        return None
    if p.shape != q.shape:
        return None

    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)

    return float(np.sum(p * np.log(p / q)))


def avg_kl_first5(native_step_logits, transfer_step_logits, max_steps=5):
    if native_step_logits is None or transfer_step_logits is None:
        return None

    n = min(len(native_step_logits), len(transfer_step_logits), max_steps)
    if n == 0:
        return None

    kls = []
    for i in range(n):
        kl = kl_divergence_from_logits(native_step_logits[i], transfer_step_logits[i])
        if kl is not None:
            kls.append(kl)

    return mean_or_none(kls)


# ---------------------------
# GROUPED SUMMARY
# ---------------------------
def build_group_summary(per_example, group_key):
    grouped = defaultdict(list)
    for e in per_example:
        grouped[e[group_key]].append(e)

    summary = {}
    for group_value, rows in grouped.items():
        ok_rows = [r for r in rows if not r["failure"]]

        summary[group_value] = {
            "num_examples": len(rows),
            "num_successes": sum(1 for r in rows if not r["failure"]),
            "num_failures": sum(1 for r in rows if r["failure"]),
            "failure_rate": (
                sum(1 for r in rows if r["failure"]) / len(rows) if rows else None
            ),
            "avg_cosine_similarity": mean_or_none(
                [r["cosine_similarity"] for r in ok_rows]
            ),
            "avg_token_accuracy_h20": mean_or_none(
                [r["token_accuracy"] for r in ok_rows]
            ),
            "avg_tokens_until_first_deviation_h20": mean_or_none(
                [r["tokens_until_first_deviation"] for r in ok_rows]
            ),
            "avg_kl_divergence_first5": mean_or_none(
                [r["avg_kl_divergence_first5"] for r in ok_rows]
            ),
        }

    return summary


# ---------------------------
# MAIN
# ---------------------------
def main(native_file, transfer_file, output_file):
    native = load_jsonl(native_file)
    transfer = load_jsonl(transfer_file)

    native_by_id = {x["prompt_id"]: x for x in native}
    per_example = []

    success_count = 0
    failure_count = 0

    bleu_predictions = []
    bleu_references = []

    # batch embedding
    pred_texts = []
    ref_texts = []
    successful_per_example_indices = []

    # scalar metric accumulators
    cosine_scores = []
    token_acc_scores = []
    deviation_scores = []
    kl_scores = []

    # timing accumulators
    prefill_1b_scores = []
    transform_scores = []
    resume_ttft_scores = []
    transfer_ttft_total_scores = []
    native_ttft_total_scores = []

    for t in transfer:
        pid = t["prompt_id"]
        n = native_by_id.get(pid)

        if n is None:
            failure_count += 1
            per_example.append({
                "prompt_id": pid,
                "category": t.get("category"),
                "length": t.get("length"),
                "method": t.get("method"),
                "bleu": None,
                "cosine_similarity": None,
                "token_accuracy": None,
                "tokens_until_first_deviation": None,
                "avg_kl_divergence_first5": None,
                "failure": True,
                "error": f"No matching native output for prompt_id={pid}"
            })
            continue

        if (not t.get("transfer_success", False)) or (t.get("generated_text") is None):
            failure_count += 1
            per_example.append({
                "prompt_id": pid,
                "category": t.get("category"),
                "length": t.get("length"),
                "method": t.get("method"),
                "bleu": None,
                "cosine_similarity": None,
                "token_accuracy": None,
                "tokens_until_first_deviation": None,
                "avg_kl_divergence_first5": None,
                "failure": True,
                "error": t.get("error")
            })
            continue

        success_count += 1

        pred_text = t.get("generated_text", "")
        ref_text = n.get("generated_text", "")

        pred_tokens = t.get("generated_token_ids") or []
        ref_tokens = n.get("generated_token_ids") or []

        # corpus BLEU inputs
        bleu_predictions.append(pred_text)
        bleu_references.append([ref_text])

        # batch embedding inputs
        pred_texts.append(pred_text)
        ref_texts.append(ref_text)
        successful_per_example_indices.append(len(per_example))

        # token-level metrics
        acc = token_accuracy(pred_tokens, ref_tokens, horizon=20)
        deviation = tokens_until_first_deviation(pred_tokens, ref_tokens, horizon=20)

        token_acc_scores.append(acc)
        deviation_scores.append(deviation)

        # KL divergence over first 5 steps
        avg_kl = avg_kl_first5(
            n.get("step_logits_first5"),
            t.get("step_logits_first5"),
            max_steps=5
        )
        if avg_kl is not None:
            kl_scores.append(avg_kl)

        # timing metrics
        prefill_1b = t.get("prefill_1b_ms")
        transform_ms = t.get("transform_ms")
        resume_ttft = t.get("resume_ttft_ms")
        transfer_ttft_total = t.get("ttft_total_ms")
        native_ttft_total = n.get("ttft_total_ms")

        if prefill_1b is not None:
            prefill_1b_scores.append(prefill_1b)
        if transform_ms is not None:
            transform_scores.append(transform_ms)
        if resume_ttft is not None:
            resume_ttft_scores.append(resume_ttft)
        if transfer_ttft_total is not None:
            transfer_ttft_total_scores.append(transfer_ttft_total)
        if native_ttft_total is not None:
            native_ttft_total_scores.append(native_ttft_total)

        per_example.append({
            "prompt_id": pid,
            "category": t.get("category"),
            "length": t.get("length"),
            "method": t.get("method"),
            "bleu": None,  # corpus BLEU only
            "cosine_similarity": None,  # filled later
            "token_accuracy": acc,
            "tokens_until_first_deviation": deviation,
            "avg_kl_divergence_first5": avg_kl,
            "failure": False,
            "error": None
        })

    # ---------------------------
    # BATCH COSINE SIMILARITY
    # ---------------------------
    if pred_texts:
        pred_embs = embedder.encode(pred_texts, convert_to_numpy=True)
        ref_embs = embedder.encode(ref_texts, convert_to_numpy=True)
        cos_matrix = cosine_similarity(pred_embs, ref_embs)

        for local_i, per_example_idx in enumerate(successful_per_example_indices):
            cos = float(cos_matrix[local_i, local_i])
            per_example[per_example_idx]["cosine_similarity"] = cos
            cosine_scores.append(cos)

    # ---------------------------
    # CORPUS BLEU
    # ---------------------------
    bleu_score = None
    if bleu_predictions:
        bleu_score = bleu_metric.compute(
            predictions=bleu_predictions,
            references=bleu_references
        )["score"]

    # ---------------------------
    # LATENCY SUMMARY
    # ---------------------------
    avg_prefill_1b_ms = mean_or_none(prefill_1b_scores)
    avg_transform_ms = mean_or_none(transform_scores)
    avg_resume_ttft_ms = mean_or_none(resume_ttft_scores)
    avg_ttft_total_ms = mean_or_none(transfer_ttft_total_scores)
    native_8b_baseline_ttft_ms = mean_or_none(native_ttft_total_scores)

    if (
        avg_ttft_total_ms is not None
        and avg_ttft_total_ms > 0
        and native_8b_baseline_ttft_ms is not None
    ):
        system_speedup_x = native_8b_baseline_ttft_ms / avg_ttft_total_ms
    else:
        system_speedup_x = None

    # ---------------------------
    # FINAL SUMMARY
    # ---------------------------
    summary = {
        "method": transfer[0]["method"] if transfer else "unknown",

        "num_examples": len(transfer),
        "num_successes": success_count,
        "num_failures": failure_count,
        "failure_rate": failure_count / len(transfer) if transfer else None,

        # quality metrics
        "corpus_bleu": bleu_score,
        "avg_cosine_similarity": mean_or_none(cosine_scores),
        "avg_token_accuracy_h20": mean_or_none(token_acc_scores),
        "avg_tokens_until_first_deviation_h20": mean_or_none(deviation_scores),
        "avg_kl_divergence_first5": mean_or_none(kl_scores),

        # latency metrics
        "avg_prefill_1b_ms": avg_prefill_1b_ms,
        "avg_transform_ms": avg_transform_ms,
        "avg_resume_ttft_ms": avg_resume_ttft_ms,
        "avg_ttft_total_ms": avg_ttft_total_ms,
        "native_8b_baseline_ttft_ms": native_8b_baseline_ttft_ms,
        "system_speedup_x": system_speedup_x,

        # grouped analysis
        "by_length": build_group_summary(per_example, "length"),
        "by_category": build_group_summary(per_example, "category"),

        "per_example": per_example
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(
        {k: v for k, v in summary.items() if k != "per_example"},
        indent=2
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--native_file", required=True)
    parser.add_argument("--transfer_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    main(args.native_file, args.transfer_file, args.output_file)
