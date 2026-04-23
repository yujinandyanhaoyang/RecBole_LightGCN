# Smoke Run Summary

Date: 2026-04-23

## Run Scope
- Model: `LightGCN`
- Dataset: `amazon_books_smoke`
- Users: `200`
- Inference: full-sort top-10 over the full item catalog

## Validation
- Output shape check: passed
- Item vocabulary check: passed
- User coverage check: passed
- External evaluation input: `ground_truth_smoke.jsonl`

## Results
- `NDCG@10`: `0.47274446559836547`
- `MRR@10`: `0.433186507936508`
- `HitRate@10`: `0.945`
- `evaluated_users`: `200`

## Artifacts
- Predicted rankings: `recbole_outputs_smoke.jsonl`
- Summary JSON: `eval_results_smoke.json`
- Training checkpoint: `saved/LightGCN-Apr-23-2026_15-07-53.pth`
