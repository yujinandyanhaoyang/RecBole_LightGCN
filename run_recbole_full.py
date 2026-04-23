"""
run_recbole_full.py
实验一正式推理脚本 — LightGCN implemented in RecBole
数据集：Amazon Books 5-core（432,515本书，787,585测试用户）
硬件：NVIDIA A10 (24GB) + 28GB RAM

后台运行：
    nohup python3 run_recbole_full.py > recbole_full_run.log 2>&1 &
    tail -f recbole_full_run.log
"""

from __future__ import annotations

import json
import time
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import scipy.sparse as sp

# ── numpy / scipy 兼容性补丁 ──────────────────────────────────────
np.bool    = getattr(np, "bool_",    bool)
np.int     = getattr(np, "int_",     int)
np.float   = float
np.float_  = float
np.complex = getattr(np, "complex128", complex)
np.object  = object
np.str     = str
np.long    = int
np.unicode = str

if not hasattr(sp.dok_matrix, "_update"):
    sp.dok_matrix._update = sp.dok_matrix.update

# ── 必须在 numpy 补丁之后 import RecBole ──────────────────────────
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.utils import get_model, get_trainer, init_logger, init_seed

# ══════════════════════════════════════════════════════════════════
# 路径配置（按 GPU 服务器实际路径修改）
# ══════════════════════════════════════════════════════════════════
BASE_DIR          = Path(__file__).resolve().parent
CONFIG_FILE       = BASE_DIR / "recbole_full_config.yaml"
TEST_USERS_FILE   = Path("/mnt/workspace/experiment_1/amazon_books_5core/recbole_test_users_5core.jsonl")
OUTPUT_FILE       = BASE_DIR / "recbole_outputs.jsonl"
OOV_FILE          = BASE_DIR / "recbole_oov_users.jsonl"
LOG_FILE          = BASE_DIR / "recbole_full_run.log"
CHECKPOINT_DIR    = BASE_DIR / "saved"

MODEL_NAME        = "LightGCN"
DATASET_NAME      = "amazon_books_5core"
BATCH_SIZE        = 1024          # A10 24GB，432k书 × 1024用户 ≈ 1.7GB float32，安全
TOPK              = 10
PROTOCOL_REF      = "book_rec_eval_v1"
METHOD_TAG        = "LightGCN"
EXPECTED_USERS    = 787585

# ══════════════════════════════════════════════════════════════════
# 日志（同时写 stdout 和文件，nohup 可 tail）
# ══════════════════════════════════════════════════════════════════
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("recbole_full")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for h in [logging.StreamHandler(sys.stdout),
               logging.FileHandler(LOG_FILE, encoding="utf-8")]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger

# ══════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════
def read_test_user_ids() -> list[str]:
    users = []
    with TEST_USERS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                users.append(json.loads(line)["user_id"])
    return users


def format_output_row(user_id: str, book_ids: list[str]) -> dict:
    """统一论文输出格式"""
    return {
        "user_id":        user_id,
        "method":         METHOD_TAG,
        "protocol_ref":   PROTOCOL_REF,
        "recommendations": [
            {"rank": rank + 1, "book_id": bid}
            for rank, bid in enumerate(book_ids)
        ],
    }

# ══════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════
def main() -> None:
    logger = setup_logging()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    run_start = time.time()

    logger.info("=" * 60)
    logger.info("实验一正式推理 — LightGCN implemented in RecBole")
    logger.info(f"数据集: Amazon Books 5-core")
    logger.info(f"启动:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"GPU:    {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU-only'}")
    logger.info("=" * 60)

    # ── Step 1: 读取测试用户 ────────────────────────────────────
    logger.info("[Step 1/4] 读取测试用户...")
    test_user_ids = read_test_user_ids()
    logger.info(f"  测试用户数: {len(test_user_ids):,}")
    if len(test_user_ids) != EXPECTED_USERS:
        logger.warning(f"  ⚠️  预期 {EXPECTED_USERS:,}，实际 {len(test_user_ids):,}，请核查")

    # ── Step 2: 初始化 RecBole，训练 LightGCN ──────────────────
    logger.info("[Step 2/4] 初始化 RecBole...")
    config = Config(
        model=MODEL_NAME,
        dataset=DATASET_NAME,
        config_file_list=[str(CONFIG_FILE)],
    )
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    logger.info("  构建数据集（首次运行约 5~15 分钟）...")
    dataset    = create_dataset(config)
    train_data, valid_data, _ = data_preparation(config, dataset)

    logger.info(f"  用户数: {dataset.user_num:,}")
    logger.info(f"  书目数: {dataset.item_num:,}  (含 padding，实际 {dataset.item_num - 1:,})")
    logger.info(f"  交互数: {dataset.inter_num:,}")

    model         = get_model(config["model"])(config, train_data.dataset).to(config["device"])
    trainer       = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    logger.info(f"[Step 2/4] 训练 LightGCN（最多 {config['epochs']} epochs，"
                f"early stopping={config['stopping_step']}）...")
    train_start = time.time()
    trainer.fit(train_data, valid_data, verbose=True, saved=True, show_progress=True)
    logger.info(f"  训练完成，耗时 {(time.time()-train_start)/3600:.2f} 小时")
    logger.info(f"  最佳 checkpoint: {trainer.saved_model_file}")

    # ── 加载最佳 checkpoint ────────────────────────────────────
    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
    trainer.resume_checkpoint(trainer.saved_model_file)
    torch.load = _orig_load
    model.eval()
    logger.info("  最佳 checkpoint 已加载")

    # ── Step 3: 分批 full-sort Top-10 推理 ─────────────────────
    logger.info(f"[Step 3/4] 分批推理（BATCH_SIZE={BATCH_SIZE}，"
                f"共 {(len(test_user_ids)+BATCH_SIZE-1)//BATCH_SIZE:,} 批）...")

    output_rows:   list[dict] = []
    skipped_users: list[str]  = []
    infer_start   = time.time()
    total_batches = (len(test_user_ids) + BATCH_SIZE - 1) // BATCH_SIZE

    with torch.no_grad():
        for batch_idx, start in enumerate(range(0, len(test_user_ids), BATCH_SIZE)):
            batch_tokens = test_user_ids[start: start + BATCH_SIZE]

            # OOV 过滤
            valid_tokens = []
            for uid in batch_tokens:
                uid_id = dataset.token2id(dataset.uid_field, [uid])[0]
                if uid_id != 0:
                    valid_tokens.append(uid)
                else:
                    skipped_users.append(uid)

            if not valid_tokens:
                continue

            uid_tensor  = torch.as_tensor(
                dataset.token2id(dataset.uid_field, valid_tokens),
                dtype=torch.long, device=config["device"]
            )
            interaction = Interaction({dataset.uid_field: uid_tensor})
            scores      = model.full_sort_predict(interaction).view(len(valid_tokens), dataset.item_num)
            scores[:, 0] = float("-inf")          # 屏蔽 padding item

            _, topk_iids = torch.topk(scores, TOPK, dim=-1)
            raw_tokens   = dataset.id2token(dataset.iid_field, topk_iids.cpu())
            topk_list    = raw_tokens.tolist() if hasattr(raw_tokens, "tolist") else list(raw_tokens)

            for uid, item_row in zip(valid_tokens, topk_list):
                output_rows.append(format_output_row(uid, list(item_row)))

            # 进度日志（每 100 批一次）
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == total_batches:
                done    = len(output_rows) + len(skipped_users)
                elapsed = time.time() - infer_start
                eta     = elapsed / max(done, 1) * (len(test_user_ids) - done)
                logger.info(
                    f"  批次 {batch_idx+1:,}/{total_batches:,} | "
                    f"{done:,}/{len(test_user_ids):,} 用户 "
                    f"({done/len(test_user_ids)*100:.1f}%) | "
                    f"已用 {elapsed/60:.1f}min | ETA {eta/60:.1f}min"
                )

    infer_elapsed = time.time() - infer_start

    # ── Step 4: 写出结果 ────────────────────────────────────────
    logger.info(f"[Step 4/4] 写出结果文件...")
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info(f"  → {OUTPUT_FILE}  ({len(output_rows):,} 用户)")

    if skipped_users:
        with OOV_FILE.open("w", encoding="utf-8") as f:
            for uid in skipped_users:
                f.write(json.dumps({"user_id": uid}) + "\n")
        logger.warning(f"  ⚠️  OOV 用户: {len(skipped_users):,} 人 → {OOV_FILE}")
        logger.warning(f"      论文中需说明 OOV 用户排除在外部评估之外")

    # ── 最终汇总 ───────────────────────────────────────────────
    total_elapsed = time.time() - run_start
    logger.info("=" * 60)
    logger.info("✅ 实验一 RecBole-LightGCN 正式推理完成")
    logger.info(f"  成功推理用户: {len(output_rows):,}")
    logger.info(f"  OOV 跳过:     {len(skipped_users):,}")
    logger.info(f"  推理耗时:     {infer_elapsed/3600:.2f} 小时")
    logger.info(f"  总耗时:       {total_elapsed/3600:.2f} 小时")
    logger.info(f"  完成时间:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    logger.info("下一步: recbole_outputs.jsonl + acps_outputs.jsonl → eval_experiment1.py")


if __name__ == "__main__":
    main()
