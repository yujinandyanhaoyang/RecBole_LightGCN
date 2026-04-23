from __future__ import annotations

import json
import math
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "recbole_outputs_smoke.jsonl"
GROUND_TRUTH_FILE = BASE_DIR / "ground_truth_smoke.jsonl"
SUMMARY_FILE = BASE_DIR / "eval_results_smoke.json"
TOPK = 10


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as file_handle:
        return [json.loads(line) for line in file_handle if line.strip()]


def extract_ground_truth(record: dict[str, object]) -> list[str]:
    if "ground_truth" in record:
        value = record["ground_truth"]
        if value and isinstance(value[0], dict):
            return [entry["book_id"] for entry in value]  # type: ignore[index]
        return list(value)  # type: ignore[arg-type]
    if "relevant_books" in record:
        return [entry["book_id"] for entry in record["relevant_books"]]  # type: ignore[index]
    raise KeyError(f"Unrecognized ground-truth schema: {record.keys()}")


def ndcg_at_k(predicted: list[str], ground_truth: set[str]) -> float:
    hits = [1 if item_id in ground_truth else 0 for item_id in predicted[:TOPK]]
    dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits) if hit)
    ideal_hits = min(len(ground_truth), TOPK)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1 / math.log2(index + 2) for index in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(predicted: list[str], ground_truth: set[str]) -> float:
    for index, item_id in enumerate(predicted[:TOPK], start=1):
        if item_id in ground_truth:
            return 1.0 / index
    return 0.0


def hit_at_k(predicted: list[str], ground_truth: set[str]) -> float:
    return 1.0 if any(item_id in ground_truth for item_id in predicted[:TOPK]) else 0.0


def main() -> None:
    output_rows = load_jsonl(OUTPUT_FILE)
    ground_truth_rows = load_jsonl(GROUND_TRUTH_FILE)

    predictions = {row["user_id"]: row["top10"] for row in output_rows}
    ground_truth = {row["user_id"]: extract_ground_truth(row) for row in ground_truth_rows}

    missing_users = sorted(set(ground_truth) - set(predictions))
    extra_users = sorted(set(predictions) - set(ground_truth))
    if missing_users or extra_users:
        raise ValueError(
            f"Prediction/ground-truth user mismatch. missing={missing_users[:5]}, extra={extra_users[:5]}"
        )

    per_user_results: list[dict[str, object]] = []
    ndcg_values: list[float] = []
    mrr_values: list[float] = []
    hit_values: list[float] = []

    for user_id in sorted(ground_truth):
        predicted = predictions[user_id]
        user_ground_truth = set(ground_truth[user_id])
        ndcg = ndcg_at_k(predicted, user_ground_truth)
        mrr = mrr_at_k(predicted, user_ground_truth)
        hit = hit_at_k(predicted, user_ground_truth)
        ndcg_values.append(ndcg)
        mrr_values.append(mrr)
        hit_values.append(hit)
        per_user_results.append(
            {
                "user_id": user_id,
                "NDCG@10": ndcg,
                "MRR@10": mrr,
                "HitRate@10": hit,
                "ground_truth": sorted(user_ground_truth),
                "top10": predicted,
            }
        )
        print(
            f"{user_id} NDCG@10={ndcg:.4f} MRR@10={mrr:.4f} HitRate@10={hit:.4f} "
            f"ground_truth={sorted(user_ground_truth)} top10={predicted}"
        )

    summary = {
        "NDCG@10": sum(ndcg_values) / len(ndcg_values) if ndcg_values else 0.0,
        "MRR@10": sum(mrr_values) / len(mrr_values) if mrr_values else 0.0,
        "HitRate@10": sum(hit_values) / len(hit_values) if hit_values else 0.0,
        "evaluated_users": len(per_user_results),
    }

    print(
        f"Macro NDCG@10={summary['NDCG@10']:.4f} MRR@10={summary['MRR@10']:.4f} "
        f"HitRate@10={summary['HitRate@10']:.4f} evaluated_users={summary['evaluated_users']}"
    )

    with SUMMARY_FILE.open("w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, ensure_ascii=False, indent=2)
        file_handle.write("\n")


if __name__ == "__main__":
    main()
