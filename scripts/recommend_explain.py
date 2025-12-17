# recsys_synth/scripts/recommend_explain.py
"""
Инференс + объяснение:
- top-5 рекомендаций на пользователя
- logits + conf (sigmoid(logit))
- сколько просмотрено (watch_stats)
- профиль по истории: топ-жанры по просмотренному
- (для отладки) "true" предпочтения из users.jsonl (так в реале нет, но тут полезно)

Запуск:
  python scripts/recommend_explain.py --data_dir data --model_path data/model_bpr_personalized.pt --top_k 5
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch
import torch.nn as nn


class BPRModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, emb_dim: int) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def score(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_ids)
        v = self.item_emb(item_ids)
        dot = (u * v).sum(dim=-1)
        ub = self.user_bias(user_ids).squeeze(-1)
        ib = self.item_bias(item_ids).squeeze(-1)
        return dot + ub + ib


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def load_items(items_path: Path) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    with items_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[int(row["item_id"])] = row
    return out


def load_users(users_path: Path) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    with users_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[int(row["user_id"])] = row
    return out


def load_watch_stats(watch_path: Path) -> Dict[int, List[dict]]:
    out: Dict[int, List[dict]] = {}
    with watch_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out.setdefault(int(row["user_id"]), []).append(row)
    return out


def history_profile_genres(
    watch_rows: List[dict],
    items_map: Dict[int, dict],
    min_completion: float = 0.2,
    top_n: int = 5,
) -> List[dict]:
    c: Counter[str] = Counter()
    for r in watch_rows:
        comp = float(r.get("completion", 0.0))
        if comp < min_completion:
            continue
        item_id = int(r["item_id"])
        genres = items_map.get(item_id, {}).get("genres") or []
        for g in genres:
            c[g] += 1
    return [{"genre": g, "count": int(cnt)} for g, cnt in c.most_common(top_n)]


def true_profile_genres(user_row: dict, top_n: int = 5) -> List[dict]:
    pref = user_row.get("genre_pref") or {}
    top = sorted(pref.items(), key=lambda x: float(x[1]), reverse=True)[:top_n]
    return [{"genre": g, "weight": round(float(w), 6)} for g, w in top]


@torch.no_grad()
def recommend_topk(
    model: BPRModel,
    user_id: int,
    num_items: int,
    exclude_items: Set[int],
    device: torch.device,
    top_k: int,
) -> List[Tuple[int, float]]:
    model.eval()
    all_items = torch.arange(num_items, dtype=torch.long, device=device)
    user_ids = torch.full((num_items,), user_id, dtype=torch.long, device=device)
    logits = model.score(user_ids, all_items).detach().cpu()

    # исключаем просмотренные
    for item_id in exclude_items:
        if 0 <= item_id < num_items:
            logits[item_id] = -1e9

    k = min(top_k, num_items)
    values, indices = torch.topk(logits, k=k)
    return [(int(i), float(v)) for i, v in zip(indices.tolist(), values.tolist())]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")  # auto|cpu|cuda
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    num_users = int(meta["num_users"])
    num_items = int(meta["num_items"])

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    payload = torch.load(Path(args.model_path), map_location="cpu")
    emb_dim = int(payload["emb_dim"])

    model = BPRModel(num_users=num_users, num_items=num_items, emb_dim=emb_dim)
    model.load_state_dict(payload["state_dict"])
    model.to(device)

    items_map = load_items(data_dir / "items.jsonl")
    users_map = load_users(data_dir / "users.jsonl")
    watch_map = load_watch_stats(data_dir / "watch_stats.jsonl")

    results: List[dict] = []
    for u in range(num_users):
        watch_rows = watch_map.get(u, [])
        watched_items = {int(r["item_id"]) for r in watch_rows if float(r.get("completion", 0.0)) >= 0.05}

        top = recommend_topk(
            model=model,
            user_id=u,
            num_items=num_items,
            exclude_items=watched_items,
            device=device,
            top_k=args.top_k,
        )

        recs = []
        for rank, (item_id, logit) in enumerate(top, start=1):
            item = items_map.get(item_id, {})
            recs.append(
                {
                    "rank": rank,
                    "item_id": item_id,
                    "title": item.get("title"),
                    "genres": item.get("genres"),
                    "logit": round(float(logit), 6),
                    "conf": round(float(sigmoid(float(logit))), 6),
                }
            )

        user_row = users_map.get(u, {})
        results.append(
            {
                "user_id": u,
                "watched_count": len(watched_items),
                "history_top_genres": history_profile_genres(watch_rows, items_map, min_completion=0.2, top_n=5),
                "true_top_genres_debug": true_profile_genres(user_row, top_n=5),
                "recommendations": recs,
            }
        )

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
