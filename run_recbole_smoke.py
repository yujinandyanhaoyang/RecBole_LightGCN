from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import numpy as np
import scipy.sparse as sp

np.bool = getattr(np, "bool_", bool)
np.int = getattr(np, "int_", int)
np.float = float
np.float_ = float
np.complex = complex
np.complex_ = complex
np.object = object
np.str = str
np.long = int
np.unicode = str
np.unicode_ = str

if not hasattr(sp.dok_matrix, "_update"):
    sp.dok_matrix._update = sp.dok_matrix.update

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.utils import get_model, get_trainer, init_logger, init_seed

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "recbole_smoke_config.yaml"
TEST_USERS_FILE = BASE_DIR / "recbole_test_users_smoke.jsonl"
OUTPUT_FILE = BASE_DIR / "recbole_outputs_smoke.jsonl"
CHECKPOINT_DIR = BASE_DIR / "saved"
BATCH_SIZE = 100
TOPK = 10
MODEL_NAME = "LightGCN"
DATASET_NAME = "amazon_books_smoke"


def read_test_user_ids() -> list[str]:
    with TEST_USERS_FILE.open("r", encoding="utf-8") as file_handle:
        return [json.loads(line)["user_id"] for line in file_handle if line.strip()]


def main() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    test_user_ids = read_test_user_ids()
    if len(test_user_ids) != 200:
        raise ValueError(f"Expected 200 smoke users, got {len(test_user_ids)}")

    start_time = time.time()
    print(f"[1/3] Building config for {MODEL_NAME}")
    config = Config(
        model=MODEL_NAME,
        dataset=DATASET_NAME,
        config_file_list=[str(CONFIG_FILE)],
    )
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    print("[1/3] Creating dataset and dataloaders")
    dataset = create_dataset(config)
    train_data, valid_data, _ = data_preparation(config, dataset)

    model_class = get_model(config["model"])
    model = model_class(config, train_data.dataset).to(config["device"])
    trainer_class = get_trainer(config["MODEL_TYPE"], config["model"])
    trainer = trainer_class(config, model)

    print(f"[2/3] Training {MODEL_NAME}")
    trainer.fit(train_data, valid_data, verbose=True, saved=True, show_progress=False)
    original_torch_load = torch.load

    def legacy_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = legacy_torch_load
    trainer.resume_checkpoint(trainer.saved_model_file)
    model.eval()

    output_rows: list[dict[str, object]] = []
    with torch.no_grad():
        for start_index in range(0, len(test_user_ids), BATCH_SIZE):
            batch_tokens = test_user_ids[start_index:start_index + BATCH_SIZE]
            uid_series = dataset.token2id(dataset.uid_field, batch_tokens)
            uid_tensor = torch.as_tensor(uid_series, dtype=torch.long, device=config["device"])
            interaction = Interaction({dataset.uid_field: uid_tensor})
            scores = model.full_sort_predict(interaction).view(len(batch_tokens), dataset.item_num)
            scores[:, 0] = float("-inf")
            _, topk_iid_list = torch.topk(scores, TOPK, dim=-1)
            raw_tokens = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())
            topk_item_tokens = raw_tokens.tolist() if hasattr(raw_tokens, "tolist") else list(raw_tokens)
            for user_id, item_row in zip(batch_tokens, topk_item_tokens):
                output_rows.append({"user_id": user_id, "top10": list(item_row)})

    with OUTPUT_FILE.open("w", encoding="utf-8") as file_handle:
        for row in output_rows:
            file_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time
    print(f"[3/3] 已处理用户数: {len(output_rows)}")
    print(f"输出文件路径: {OUTPUT_FILE}")
    print(f"耗时: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
