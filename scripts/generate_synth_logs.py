# recsys_synth/scripts/generate_synth_logs.py
"""
Генератор синтетических логов для рекомендательной системы.

Что генерируем:
- 10k фильмов (items.jsonl)
- 10 пользователей (users.jsonl) с жанровыми предпочтениями
- много логов:
  - impressions.jsonl (показы)
  - events.jsonl (клики/старты/лайки)
  - watch_stats.jsonl (время просмотра/досмотр)
- плюс готовый датасет для обучения "весов":
  - train_samples.jsonl: одна строка на impression с label/weight

Запуск пример:
  python scripts/generate_synth_logs.py --out_dir data --num_items 10000 --num_users 10 --num_impressions 200000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple


GENRES = [
    "action", "comedy", "drama", "thriller", "crime", "romance", "sci-fi", "fantasy",
    "horror", "mystery", "animation", "family", "adventure", "war", "history",
    "music", "sport", "documentary", "western", "biography",
]

SURFACES = ["home", "search", "similar", "trending", "collection"]


@dataclass(frozen=True)
class Item:
    item_id: int
    title: str
    release_year: int
    duration_sec: int
    genres: List[str]
    tags: List[str]


@dataclass(frozen=True)
class User:
    user_id: int
    country: str
    ui_language: str
    genre_pref: Dict[str, float]  # веса предпочтений по жанрам


def sigmoid(x: float) -> float:
    # устойчиво к большим значениям
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def make_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def random_dirichlet(k: int, alpha: float, rng: random.Random) -> List[float]:
    # очень простой Dirichlet через гамма-распределение
    samples = [rng.gammavariate(alpha, 1.0) for _ in range(k)]
    s = sum(samples) or 1.0
    return [v / s for v in samples]


def gen_items(num_items: int, rng: random.Random) -> List[Item]:
    items: List[Item] = []
    for item_id in range(num_items):
        release_year = rng.randint(1970, 2025)
        duration_sec = rng.randint(70, 160) * 60  # 70..160 минут
        num_genres = rng.randint(1, 3)
        genres = rng.sample(GENRES, k=num_genres)
        num_tags = rng.randint(0, 6)
        tags = [f"tag_{rng.randint(1, 250)}" for _ in range(num_tags)]
        title = f"Movie #{item_id:05d}"
        items.append(
            Item(
                item_id=item_id,
                title=title,
                release_year=release_year,
                duration_sec=duration_sec,
                genres=genres,
                tags=tags,
            )
        )
    return items


def gen_users(num_users: int, rng: random.Random) -> List[User]:
    countries = ["NL", "DE", "FR", "ES", "IT"]
    languages = ["en", "ru", "nl", "de", "fr"]

    users: List[User] = []
    for user_id in range(num_users):
        probs = random_dirichlet(len(GENRES), alpha=0.35, rng=rng)  # более "острые" вкусы
        genre_pref = {g: p for g, p in zip(GENRES, probs)}
        users.append(
            User(
                user_id=user_id,
                country=rng.choice(countries),
                ui_language=rng.choice(languages),
                genre_pref=genre_pref,
            )
        )
    return users


def preference_score(user: User, item: Item) -> float:
    # сумма предпочтений по жанрам фильма + лёгкая поправка на год/длину
    g_sum = sum(user.genre_pref.get(g, 0.0) for g in item.genres)
    year_bonus = 0.02 * ((item.release_year - 2000) / 25.0)  # слегка "любовь к новому"
    duration_penalty = -0.01 * ((item.duration_sec / 60.0 - 110.0) / 50.0)
    return g_sum + year_bonus + duration_penalty


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--num_items", type=int, default=10_000)
    parser.add_argument("--num_users", type=int, default=10)
    parser.add_argument("--num_impressions", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    make_dir(out_dir)

    items = gen_items(args.num_items, rng=rng)
    users = gen_users(args.num_users, rng=rng)

    # Быстрый доступ
    items_by_id: List[Item] = items

    # Пишем справочники
    items_path = out_dir / "items.jsonl"
    users_path = out_dir / "users.jsonl"
    write_jsonl(items_path, [asdict(i) for i in items])
    write_jsonl(users_path, [asdict(u) for u in users])

    impressions: List[dict] = []
    events: List[dict] = []
    watch_stats: List[dict] = []
    train_samples: List[dict] = []

    base_time = datetime.now(timezone.utc) - timedelta(days=7)
    request_counter = 0

    # Чтобы “похоже на реальность”: часть показов случайная, часть "в вкусе"
    prefer_mix = 0.35  # доля показов из предпочтительных жанров
    max_position = 30

    for imp_id in range(args.num_impressions):
        user = users[rng.randrange(args.num_users)]

        # выбор item: микс random и "pref"
        if rng.random() < prefer_mix:
            # грубо: выбираем жанр из top-N, затем item с этим жанром
            top_genres = sorted(user.genre_pref.items(), key=lambda x: x[1], reverse=True)[:4]
            chosen_genre = rng.choice([g for g, _ in top_genres])
            # попробуем несколько раз найти item с жанром, иначе fallback на random
            item_id = None
            for _ in range(10):
                cand = rng.randrange(args.num_items)
                if chosen_genre in items_by_id[cand].genres:
                    item_id = cand
                    break
            if item_id is None:
                item_id = rng.randrange(args.num_items)
        else:
            item_id = rng.randrange(args.num_items)

        item = items_by_id[item_id]

        shown_at = base_time + timedelta(seconds=imp_id * 2)  # монотонное время
        surface = rng.choice(SURFACES)
        position = rng.randint(1, max_position)

        request_counter += 1
        request_id = f"req-{request_counter:012d}"

        impressions.append(
            {
                "impression_id": imp_id,
                "user_id": user.user_id,
                "item_id": item.item_id,
                "surface": surface,
                "position": position,
                "shown_at": shown_at.isoformat(),
                "request_id": request_id,
            }
        )

        # вероятности действий зависят от preference_score
        pref = preference_score(user, item)
        # клик: зависит от pref и позиции (ниже позиция => меньше шанс)
        pos_pen = -0.04 * (position - 1)
        p_click = sigmoid(-2.2 + 10.0 * pref + pos_pen)
        clicked = 1 if rng.random() < p_click else 0

        started = 0
        completion = 0.0
        watch_time_sec = 0

        if clicked:
            events.append(
                {
                    "event_id": f"ev-{imp_id}-click",
                    "user_id": user.user_id,
                    "item_id": item.item_id,
                    "type": "click",
                    "value": None,
                    "occurred_at": (shown_at + timedelta(seconds=rng.randint(1, 40))).isoformat(),
                    "request_id": request_id,
                }
            )
            # старт: условно зависит от pref
            p_start = sigmoid(-1.2 + 8.0 * pref)
            started = 1 if rng.random() < p_start else 0

        if started:
            events.append(
                {
                    "event_id": f"ev-{imp_id}-play_start",
                    "user_id": user.user_id,
                    "item_id": item.item_id,
                    "type": "play_start",
                    "value": None,
                    "occurred_at": (shown_at + timedelta(seconds=rng.randint(5, 90))).isoformat(),
                    "request_id": request_id,
                }
            )

            # досмотр: зависит от pref и случайности
            p_complete = sigmoid(-0.3 + 9.0 * pref)
            # completion распределим более "гладко"
            if rng.random() < p_complete:
                completion = min(1.0, max(0.6, rng.random() * 0.4 + 0.6))
            else:
                completion = rng.random() * 0.6

            watch_time_sec = int(item.duration_sec * completion)

            # иногда лайк/дизлайк
            p_like = sigmoid(-1.0 + 10.0 * pref)
            if rng.random() < p_like and completion >= 0.5:
                events.append(
                    {
                        "event_id": f"ev-{imp_id}-like",
                        "user_id": user.user_id,
                        "item_id": item.item_id,
                        "type": "like",
                        "value": 1.0,
                        "occurred_at": (shown_at + timedelta(seconds=rng.randint(60, 600))).isoformat(),
                        "request_id": request_id,
                    }
                )
            elif rng.random() < 0.05 and completion < 0.2:
                events.append(
                    {
                        "event_id": f"ev-{imp_id}-dislike",
                        "user_id": user.user_id,
                        "item_id": item.item_id,
                        "type": "dislike",
                        "value": -1.0,
                        "occurred_at": (shown_at + timedelta(seconds=rng.randint(60, 600))).isoformat(),
                        "request_id": request_id,
                    }
                )

            watch_stats.append(
                {
                    "watch_id": f"w-{imp_id}",
                    "user_id": user.user_id,
                    "item_id": item.item_id,
                    "started_at": (shown_at + timedelta(seconds=rng.randint(1, 120))).isoformat(),
                    "ended_at": (shown_at + timedelta(seconds=rng.randint(300, 7200))).isoformat(),
                    "watch_time_sec": watch_time_sec,
                    "completion": round(completion, 4),
                    "device_type": rng.choice(["tv", "mobile", "web"]),
                    "session_id": f"sess-{user.user_id}-{imp_id // 50}",
                }
            )

        # label для обучения: считаем позитивом "досмотр >= 0.6"
        label = 1 if (started == 1 and completion >= 0.6) else 0

        # weight: хотим, чтобы "сильные" сигналы сильнее влияли
        # (досмотр — сильнее, клик — слабее, без действия — ещё слабее)
        if label == 1:
            weight = 2.5 + 2.0 * completion
        elif started == 1:
            weight = 1.2 + 0.8 * completion
        elif clicked == 1:
            weight = 0.6
        else:
            weight = 0.2

        train_samples.append(
            {
                "user_id": user.user_id,
                "item_id": item.item_id,
                "shown_at": shown_at.isoformat(),
                "surface": surface,
                "position": position,
                "clicked": clicked,
                "started": started,
                "watch_time_sec": watch_time_sec,
                "completion": round(completion, 4),
                "label": label,
                "weight": round(float(weight), 4),
            }
        )

    # Пишем логи
    write_jsonl(out_dir / "impressions.jsonl", impressions)
    write_jsonl(out_dir / "events.jsonl", events)
    write_jsonl(out_dir / "watch_stats.jsonl", watch_stats)
    write_jsonl(out_dir / "train_samples.jsonl", train_samples)

    meta = {
        "num_items": args.num_items,
        "num_users": args.num_users,
        "num_impressions": args.num_impressions,
        "seed": args.seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] wrote to: {out_dir.resolve()}")
    print(f"     items.jsonl:        {args.num_items} lines")
    print(f"     users.jsonl:        {args.num_users} lines")
    print(f"     impressions.jsonl:  {len(impressions)} lines")
    print(f"     events.jsonl:       {len(events)} lines")
    print(f"     watch_stats.jsonl:  {len(watch_stats)} lines")
    print(f"     train_samples.jsonl:{len(train_samples)} lines")


if __name__ == "__main__":
    main()
