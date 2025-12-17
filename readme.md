# Система рекомендаций для просмотра тайтлов. CLI (Генерация логов, Обучение BPR, Выборка по весам BPR)

## BCE (Binary Cross-Entropy)
Идея: учим модель предсказывать вероятность “позитивного события” для пары (user, item).
## BPR (Bayesian Personalized Ranking)
Идея: учим модель ранжировать, а не предсказывать абсолютную вероятность.

## Точные структуры файлов и что означает каждый ключ
Все файлы — JSONL: 1 строка = 1 JSON-объект.

### Что обязательно логировать (минимальный контракт)
- Impressions (показы) — иначе у тебя нет корректных негативов
- user_id
- item_id
- shown_at
- surface (экран/лента/похожее)
- position (позиция в ленте)
- request_id (идентификатор выдачи)
- Events (клики/старты/лайки)
- user_id, item_id, type, occurred_at, request_id
- Watch stats (просмотр)
- user_id, item_id
- watch_time_sec
- completion (или хотя бы watch_time + duration)
- started_at (важно для time-split)

Это минимально, чтобы:
- учить retrieval (BPR/ALS/two-tower)
- позже учить ranking 


items.jsonl (каталог)
```
{
  "item_id": 2891,
  "title": "Movie #02891",
  "release_year": 2012,
  "duration_sec": 6780,
  "genres": ["history", "animation", "crime"],
  "tags": ["tag_12", "tag_77"]
}
```
item_id:int — идентификатор тайтла

genres:list[str] — жанры (контентная разметка; в BPR напрямую не участвует, но помогает объяснять)

остальное — мета

- users.jsonl (пользователи)
```
{
  "user_id": 0,
  "country": "NL",
  "ui_language": "ru",
  "genre_pref": {"documentary":0.3139, "drama":0.2520, "...":0.0001}
}
```
genre_pref — только для синтетики/отладки (в реальных данных у тебя этого нет).

В обучении BPR это не используется; модель сама “вытаскивает” предпочтения из поведения.

- impressions.jsonl (показы)
```
{
  "impression_id": 123,
  "user_id": 0,
  "item_id": 2891,
  "surface": "home",
  "position": 4,
  "shown_at": "2025-12-10T12:00:01+00:00",
  "request_id": "req-000000000123"
}
```
Это главный лог, чтобы делать корректные негативы: “показали, но не выбрал”.

- events.jsonl (события)
```
{
  "event_id": "ev-123-click",
  "user_id": 0,
  "item_id": 2891,
  "type": "click",
  "value": null,
  "occurred_at": "2025-12-10T12:00:12+00:00",
  "request_id": "req-000000000123"
}
```
type ∈ click | play_start | like | dislike | rate | ...

request_id связывает событие с конкретным показом.

- watch_stats.jsonl (просмотр)
```
{
  "watch_id": "w-123",
  "user_id": 0,
  "item_id": 2891,
  "started_at": "2025-12-10T12:01:00+00:00",
  "ended_at": "2025-12-10T13:20:00+00:00",
  "watch_time_sec": 3600,
  "completion": 0.72,
  "device_type": "tv",
  "session_id": "sess-0-2"
}
```
completion 0..1 — доля досмотра (ключевой сигнал “понравилось”)

В BPR-обучении позитив берётся отсюда: completion >= min_pos_completion (по умолчанию 0.6).

- train_samples.jsonl
Это датасет для BCE-версии (которую ты запускал раньше). Для BPR-скрипта он не нужен.

## Что означает “предпочтение” в этой модели
Это латентные факторы:
- U[u] = “скрытый вкус пользователя” (вектор)
- V[i] = “скрытый профиль тайтла” (вектор)

если U[u] хорошо “со-направлен” с V[i], тайтл попадает в top

В синтетике скрытая причина — жанры, но модель не знает жанры, она видит только факты просмотра и учится по ним.



### генерация
* каталог 10 000 тайтлов (items.jsonl)
* 10 пользователей (users.jsonl)
* логи показа/клика/просмотра (как в онлайн-кинотеатре)
* и производные обучающие записи (для BCE-версии), но BPR-тренер их не использует
```commandline
python scripts/generate_synth_logs.py --out_dir data --num_items 10000 --num_users 10 --num_impressions 200000 --seed 42
```


### BPR обучение (GPU/AMP без warnings)
Учит retrieval-модель типа “пользователь ↔ тайтл” на pairwise/BPR:
* позитив = тайтл, который пользователь посмотрел достаточно (по watch_stats.jsonl)
* негатив = случайный тайтл, которого нет в позитивах (negative sampling)
```commandline
python scripts/train_bpr_gpu.py --data_dir data --epochs 5 --steps_per_epoch 3000 --batch_size 2048 --emb_dim 64 --lr 0.02 --k_eval 50
```


### BPR + персональная донастройка user-векторов (то, что ты просишь "персонально")
После общего обучения делает “доточечную” персонализацию:
* фиксирует item-эмбеддинги
* под каждого пользователя подкручивает только его user-вектор/базу на его истории
```commandline
python scripts/train_bpr_gpu.py --data_dir data --epochs 5 --steps_per_epoch 3000 --batch_size 2048 --emb_dim 64 --lr 0.02 --k_eval 50 --personalize_steps 300 --personalize_lr 0.05
```


### объяснение рекомендаций (top-5, logits/conf, watched_count, профили)
Считает для каждого user top-K:
* score(u,i) = dot(U[u], V[i]) + bu[u] + bi[i]
* conf = sigmoid(score) (это скор/уверенность, а не строго откалиброванная вероятность)
```commandline
python scripts/recommend_explain.py --data_dir data --model_path data/model_bpr_personalized.pt --top_k 5
```

### После запуска получаем 2 модели 
1) model_bpr.pt — “глобальная” модель

Это результат обычного BPR-обучения:

- учатся все параметры одновременно:
  * U[user_id] (эмбеддинги пользователей)
  * V[item_id] (эмбеддинги тайтлов)
  * bu[user_id], bi[item_id] (bias’ы)

смысл: модель находит общие латентные факторы “кто что любит”, используя данные всех пользователей.

Это базовая retrieval-модель.

2) model_bpr_personalized.pt — та же модель, но после “персонализации”

На этом шаге происходит другое:

- фиксируем параметры тайтлов (обычно это главное):
    * V[item_id], bi[item_id] НЕ меняются
    * для каждого пользователя отдельно “подкручиваем” только его параметры:
    * U[user_id], bu[user_id] обновляются градиентом на его истории

И после этого мы сохраняем новый чекпоинт — чтобы не потерять исходный “глобальный” вариант.

model_bpr.pt — хорошая база (и подходит новым/редким пользователям)
model_bpr_personalized.pt — лучше для пользователей с историей (и особенно если вкусы “нестандартные”)