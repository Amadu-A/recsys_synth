# recsys_synth/scripts/train_bpr_gpu.py
"""
BPR (pairwise) обучение эмбеддингов + negative sampling, GPU (AMP без warnings).

Идея:
  score(u, i) = dot(U[u], V[i]) + bu[u] + bi[i]
  loss_BPR = -log(sigmoid(score(u, pos) - score(u, neg)))

Плюс:
- time-based split на train/val по watch_stats (по started_at)
- метрика: Recall@K по val-позитивам (а не val_loss BCE, который легко переобучается)
- опциональная персонализация: после глобального обучения донастраиваем user-вектор
  отдельно для каждого пользователя (item-эмбеддинги фиксируем).

Запуск:
  python scripts/train_bpr_gpu.py --data_dir data --epochs 5 --steps_per_epoch 3000 --batch_size 2048 --emb_dim 64 --lr 0.02 --k_eval 50 --personalize_steps 300
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch
import torch.nn as nn


# -------------------------
# Model
# -------------------------
class BPRModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, emb_dim: int) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        nn.init.normal_(self.user_emb.weight, std=0.02)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def score(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_ids)
        v = self.item_emb(item_ids)
        dot = (u * v).sum(dim=-1)
        ub = self.user_bias(user_ids).squeeze(-1)
        ib = self.item_bias(item_ids).squeeze(-1)
        return dot + ub + ib


# -------------------------
# Data loading
# -------------------------
def _parse_ts(ts: str) -> float:
    # ISO 8601 with timezone
    return datetime.fromisoformat(ts).timestamp()


@dataclass
class UserInteractions:
    train_pos: List[int]
    val_pos: List[int]
    train_pos_set: Set[int]
    val_pos_set: Set[int]
    watched_all_set: Set[int]


def load_user_interactions(
    watch_stats_path: Path,
    num_users: int,
    min_pos_completion: float = 0.6,
    min_watch_for_seen: float = 0.05,
    val_ratio: float = 0.2,
) -> Dict[int, UserInteractions]:
    """
    Строим:
    - watched_all_set: всё, что пользователь стартовал (seen)
    - train_pos / val_pos: позитивы по completion>=min_pos_completion, разбитые по времени
    """
    per_user: Dict[int, List[Tuple[float, int, float]]] = {u: [] for u in range(num_users)}
    watched_all: Dict[int, Set[int]] = {u: set() for u in range(num_users)}

    with watch_stats_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            user_id = int(row["user_id"])
            item_id = int(row["item_id"])
            completion = float(row["completion"])
            started_at = row["started_at"]

            if completion >= min_watch_for_seen:
                watched_all[user_id].add(item_id)

            per_user[user_id].append((_parse_ts(started_at), item_id, completion))

    out: Dict[int, UserInteractions] = {}
    for user_id in range(num_users):
        rows = sorted(per_user[user_id], key=lambda x: x[0])
        positives = [(ts, item_id) for ts, item_id, comp in rows if comp >= min_pos_completion]

        if len(positives) < 2:
            # слишком мало позитивов — всё в train, val пустой
            train_pos = [item_id for _, item_id in positives]
            val_pos: List[int] = []
        else:
            cut = max(1, int(len(positives) * (1.0 - val_ratio)))
            train_pos = [item_id for _, item_id in positives[:cut]]
            val_pos = [item_id for _, item_id in positives[cut:]]

        out[user_id] = UserInteractions(
            train_pos=train_pos,
            val_pos=val_pos,
            train_pos_set=set(train_pos),
            val_pos_set=set(val_pos),
            watched_all_set=watched_all[user_id],
        )

    return out


def build_sampling_pools(ui: Dict[int, UserInteractions]) -> List[int]:
    # список пользователей, у которых есть хотя бы 1 train positive
    return [u for u, x in ui.items() if len(x.train_pos) > 0]


# -------------------------
# Training helpers
# -------------------------
def bpr_loss(pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
    # -log(sigmoid(pos-neg)) = softplus(neg-pos)
    return torch.nn.functional.softplus(neg_score - pos_score)


def sample_batch(
    rng: random.Random,
    users_with_pos: List[int],
    ui: Dict[int, UserInteractions],
    num_items: int,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Возвращает CPU тензоры user_ids, pos_ids, neg_ids (потом .to(device))
    """
    user_ids: List[int] = []
    pos_ids: List[int] = []
    neg_ids: List[int] = []

    for _ in range(batch_size):
        u = rng.choice(users_with_pos)
        pos = rng.choice(ui[u].train_pos)

        # negative sampling: случайный item, которого нет в позитиве
        # (можно делать жёстче: исключать watched_all_set; для демо достаточно исключить train_pos_set)
        attempts = 0
        while True:
            neg = rng.randrange(num_items)
            if neg not in ui[u].train_pos_set:
                break
            attempts += 1
            if attempts > 50:
                # крайне маловероятно, но чтобы не зависнуть
                neg = (pos + 1) % num_items
                break

        user_ids.append(u)
        pos_ids.append(pos)
        neg_ids.append(neg)

    return (
        torch.tensor(user_ids, dtype=torch.long),
        torch.tensor(pos_ids, dtype=torch.long),
        torch.tensor(neg_ids, dtype=torch.long),
    )


@torch.no_grad()
def recall_at_k(
    model: BPRModel,
    ui: Dict[int, UserInteractions],
    num_users: int,
    num_items: int,
    k: int,
    device: torch.device,
) -> float:
    """
    Для каждого пользователя:
      - кандидаты: все items
      - маскируем train_pos (как уже "известные")
      - смотрим, сколько val_pos попало в topK
    """
    model.eval()
    recalls: List[float] = []

    all_items = torch.arange(num_items, dtype=torch.long, device=device)

    for u in range(num_users):
        val_set = ui[u].val_pos_set
        if not val_set:
            continue

        user_ids = torch.full((num_items,), u, dtype=torch.long, device=device)
        scores = model.score(user_ids, all_items).detach()

        # маска train_pos
        for item_id in ui[u].train_pos_set:
            if 0 <= item_id < num_items:
                scores[item_id] = -1e9

        topk = torch.topk(scores, k=min(k, num_items)).indices.tolist()
        hits = sum(1 for item_id in topk if item_id in val_set)
        recalls.append(hits / max(1, len(val_set)))

    if not recalls:
        return 0.0
    return float(sum(recalls) / len(recalls))


def personalize_user(
    model: BPRModel,
    user_id: int,
    ui: UserInteractions,
    num_items: int,
    device: torch.device,
    steps: int,
    lr: float,
    l2: float,
    rng: random.Random,
) -> None:
    """
    Персональная донастройка: фиксируем item-эмбеддинги, оптимизируем только embedding/bias пользователя.
    """
    if steps <= 0 or len(ui.train_pos) == 0:
        return

    # копия параметров пользователя (будем обучать отдельно)
    u_vec = nn.Parameter(model.user_emb.weight[user_id].detach().clone().to(device))
    u_bias = nn.Parameter(model.user_bias.weight[user_id].detach().clone().to(device))

    opt = torch.optim.AdamW([u_vec, u_bias], lr=lr, weight_decay=0.0)

    item_emb = model.item_emb.weight.detach().to(device)
    item_bias = model.item_bias.weight.detach().to(device)

    for _ in range(steps):
        pos = rng.choice(ui.train_pos)

        # negative sampling (исключаем train_pos)
        attempts = 0
        while True:
            neg = rng.randrange(num_items)
            if neg not in ui.train_pos_set:
                break
            attempts += 1
            if attempts > 50:
                neg = (pos + 1) % num_items
                break

        pos_v = item_emb[pos]
        neg_v = item_emb[neg]
        pos_b = item_bias[pos].squeeze(-1)
        neg_b = item_bias[neg].squeeze(-1)

        pos_score = (u_vec * pos_v).sum() + u_bias.squeeze(-1) + pos_b
        neg_score = (u_vec * neg_v).sum() + u_bias.squeeze(-1) + neg_b

        loss = torch.nn.functional.softplus(neg_score - pos_score)
        if l2 > 0:
            loss = loss + l2 * (u_vec.pow(2).sum() + u_bias.pow(2).sum())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # записываем обратно в модель
    model.user_emb.weight.data[user_id].copy_(u_vec.detach().cpu())
    model.user_bias.weight.data[user_id].copy_(u_bias.detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps_per_epoch", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--l2", type=float, default=1e-6)
    parser.add_argument("--k_eval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")  # auto|cpu|cuda
    parser.add_argument("--save_path", type=str, default="model_bpr.pt")

    # персонализация
    parser.add_argument("--personalize_steps", type=int, default=0)
    parser.add_argument("--personalize_lr", type=float, default=5e-2)

    # определение позитивов
    parser.add_argument("--min_pos_completion", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)

    args = parser.parse_args()

    rng = random.Random(args.seed)
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

    ui = load_user_interactions(
        watch_stats_path=data_dir / "watch_stats.jsonl",
        num_users=num_users,
        min_pos_completion=args.min_pos_completion,
        val_ratio=args.val_ratio,
    )
    users_with_pos = build_sampling_pools(ui)

    print(f"[info] device={device} users={num_users} items={num_items} users_with_pos={len(users_with_pos)}")
    if len(users_with_pos) == 0:
        raise SystemExit("No positive interactions found (increase num_impressions or lower min_pos_completion).")

    model = BPRModel(num_users=num_users, num_items=num_items, emb_dim=args.emb_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_recall = -1.0
    save_path = Path(args.save_path)
    if not save_path.is_absolute():
        save_path = data_dir / save_path

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for step in range(args.steps_per_epoch):
            u_cpu, pos_cpu, neg_cpu = sample_batch(
                rng=rng,
                users_with_pos=users_with_pos,
                ui=ui,
                num_items=num_items,
                batch_size=args.batch_size,
            )
            u = u_cpu.to(device, non_blocking=True)
            pos = pos_cpu.to(device, non_blocking=True)
            neg = neg_cpu.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                pos_score = model.score(u, pos)
                neg_score = model.score(u, neg)
                loss_vec = bpr_loss(pos_score, neg_score)
                loss = loss_vec.mean()

                if args.l2 > 0:
                    # L2 по участвующим эмбеддингам (лёгкая регуляризация)
                    reg = (
                        model.user_emb(u).pow(2).sum(dim=-1)
                        + model.item_emb(pos).pow(2).sum(dim=-1)
                        + model.item_emb(neg).pow(2).sum(dim=-1)
                    ).mean()
                    loss = loss + args.l2 * reg

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total_loss += float(loss.item())

        r = recall_at_k(model, ui, num_users, num_items, k=args.k_eval, device=device)
        avg_loss = total_loss / max(1, args.steps_per_epoch)

        print(f"[epoch {epoch}] bpr_loss={avg_loss:.6f} recall@{args.k_eval}={r:.4f}")

        if r > best_recall:
            best_recall = r
            payload = {
                "state_dict": model.state_dict(),
                "num_users": num_users,
                "num_items": num_items,
                "emb_dim": args.emb_dim,
                "min_pos_completion": args.min_pos_completion,
                "val_ratio": args.val_ratio,
            }
            torch.save(payload, save_path)
            print(f"[ok] saved best to {save_path} (best recall@{args.k_eval}={best_recall:.4f})")

    # Персональная донастройка user-векторов (по желанию)
    if args.personalize_steps > 0:
        print(f"[info] personalization: steps={args.personalize_steps} lr={args.personalize_lr}")
        # грузим лучший чекпоинт, чтобы персонализация шла от best
        payload = torch.load(save_path, map_location="cpu")
        model.load_state_dict(payload["state_dict"])
        model.to(device)

        for u in range(num_users):
            personalize_user(
                model=model,
                user_id=u,
                ui=ui[u],
                num_items=num_items,
                device=device,
                steps=args.personalize_steps,
                lr=args.personalize_lr,
                l2=args.l2,
                rng=rng,
            )

        # сохраняем уже персонализированную версию отдельно
        personalized_path = save_path.with_name(save_path.stem + "_personalized.pt")
        payload2 = {
            "state_dict": model.state_dict(),
            "num_users": num_users,
            "num_items": num_items,
            "emb_dim": args.emb_dim,
            "personalized": True,
        }
        torch.save(payload2, personalized_path)
        print(f"[ok] saved personalized model to {personalized_path}")

    print("[done]")


if __name__ == "__main__":
    main()
