from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "recbole_outputs_smoke.jsonl"
TEST_USERS_FILE = BASE_DIR / "recbole_test_users_smoke.jsonl"
ITEM_FILE = BASE_DIR / "amazon_books_smoke.item"
EXPECTED_USERS = 200
EXPECTED_TOPK = 10


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as file_handle:
        return [json.loads(line) for line in file_handle if line.strip()]


def main() -> None:
    output_rows = load_jsonl(OUTPUT_FILE)
    test_users_rows = load_jsonl(TEST_USERS_FILE)
    with ITEM_FILE.open("r", encoding="utf-8") as file_handle:
        item_tokens = {
            line.rstrip("\n").split("\t", 1)[0]
            for line in file_handle
            if line.strip() and not line.startswith("item_id:token")
        }

    test_user_ids = [row["user_id"] for row in test_users_rows]
    output_users = [row["user_id"] for row in output_rows]
    output_by_user = {row["user_id"]: row["top10"] for row in output_rows}

    overall_pass = True

    unique_output_users = set(output_users)
    if len(unique_output_users) == EXPECTED_USERS and len(output_rows) == EXPECTED_USERS:
        print(f"PASS user count: {len(output_rows)} rows, {len(unique_output_users)} unique users")
    else:
        print(
            f"FAIL user count: {len(output_rows)} rows, {len(unique_output_users)} unique users, expected {EXPECTED_USERS}"
        )
        overall_pass = False

    wrong_lengths = {user_id: len(items) for user_id, items in output_by_user.items() if len(items) != EXPECTED_TOPK}
    if not wrong_lengths:
        print(f"PASS top-k count: every user has exactly {EXPECTED_TOPK} recommendations")
    else:
        sample = list(wrong_lengths.items())[:5]
        print(f"FAIL top-k count: {sample}")
        overall_pass = False

    invalid_items: dict[str, list[str]] = {}
    for user_id, top10 in output_by_user.items():
        bad_items = [item_id for item_id in top10 if item_id not in item_tokens]
        if bad_items:
            invalid_items[user_id] = bad_items
    if not invalid_items:
        print("PASS item vocabulary: all recommended book_id values exist in amazon_books_smoke.item")
    else:
        sample = list(invalid_items.items())[:5]
        print(f"FAIL item vocabulary: {sample}")
        overall_pass = False

    missing_users = sorted(set(test_user_ids) - set(output_users))
    if not missing_users:
        print("PASS coverage: every test user has an output row")
    else:
        print(f"FAIL coverage: missing users {missing_users[:10]}")
        overall_pass = False

    if overall_pass:
        print("✅ All checks passed")
    else:
        print("❌ Check failed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
