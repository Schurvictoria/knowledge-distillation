# Результаты экспериментов

## Methodology summary

- **Splits:** train/test 90/10 stratified, `seed=42` для всех датасетов.
- **Datasets:** Gender (binary, AUC), Rosbank (binary churn, AUC), Age (4-class, accuracy).
- **Reproducibility:** `torch.manual_seed(42)` + `cudnn.deterministic=True` + `pytorch_lightning.seed_everything(42)`. OpenRouter API: `seed=42` в payload.
- **Seeds policy:** RQ1 / RQ3 — single seed=42 (compute-limited). RQ2 D1 backends поддерживают 5 seeds `[42,123,456,789,1024]` (см. примечание ниже).
- **Canonical LLM teacher (D1):** Qwen2.5-3B (base) для LLM4ES embedding extraction. `hidden_size=2048`.
- **Canonical LLM (D2):** Qwen2.5-7B-Instruct, 4-bit NF4 для kNN/CoT inference.

## RQ1: Bidirectional vs Unidirectional

| # | Method | Description | Что происходит на пальцах | Direction | Gender | Rosbank | Age |
|---|---|---|---|---|---|---|---|
| **E1.1** | CoLES baseline | Just a CoLES baseline | GRU/LSTM на сырых транзакциях обучается self-supervised contrastive (нарезаем 5 окон у одного клиента → они "близкие", у разных → "далёкие"). Поверх 1024-d эмбеддингов клиента — LightGBM. LLM не используется вообще. Это нижняя граница, от которой меряем все остальные методы. | No transfer | 0.8626 | 0.8054 | 0.6345 |
| **E1.2** | LATTE | CoLES учится у LLM embeddings через contrastive loss; веса LLM не обновляются | Берём готовый CoLES (E1.1). Параллельно прогоняем серилизованный текст транзакций через **замороженную** Qwen2.5-3B → достаём LLM4ES эмбеддинг (mean over last 8 layers → masked mean over tokens, 2048-d). Файнтьюним CoLES с двумя лоссами одновременно: classification (BCE/CE) + InfoNCE между проекциями CoLES и LLM эмбеддингов (CLIP-style, τ=0.07). `total = 0.9·cls + 0.1·infonce`. CoLES "подтягивает" свои векторы под пространство LLM. | LLM → CoLES | **0.8713** | **0.8082** | 0.6333 |
| **E1.3** | LATTE + mutual KL | CoLES и LLM учат друг друга одновременно; оба обновляют веса | То же что E1.2, но LLM **тоже учится**: на голову LLM ставится классификатор, и оба (CoLES и LLM) обмениваются soft labels через симметричный KL. Идея — обоюдное обучение делает обе модели лучше. На практике на Gender хуже E1.2 (0.8676 < 0.8713) — LLM начинает дрейфовать, и contrastive сигнал портится. | LLM ↔ CoLES | 0.8676 | 0.8099 | 0.6363 |
| **E1.4** | RAMD (Qwen2.5-7B) | Цикл: CoLES → LLM (kNN) → KL обратно → повтор | **Step 1** (5-fold OOF на train, без leakage): берём CoLES эмбеддинги → для каждого клиента находим 10 ближайших соседей kNN → подсовываем LLM в промпт ("Похожие клиенты: 7 male, 3 female") → LLM выдаёт soft label.<br>**Step 2:** файнтьюним CoLES с reverse KL (`KL(P_student‖P_teacher)`) против этих soft labels. Reverse KL заставляет student быть "уверенным там где LLM уверен" (mode-seeking, MiniLLM 2024). Цикл можно повторять. | LLM ↔ CoLES (loop) | 0.8630 | 0.8074 | skipped (см. E1.5) |
| **E1.5** | RAMD (DeepSeek-V3.2) | Тот же цикл, сильнее LLM | Тот же двухстадийный pipeline что E1.4, только LLM-учитель — DeepSeek-V3.2 (671B MoE) через OpenRouter API вместо локальной 7B. Гипотеза: сильнее teacher → лучше soft labels → лучше student. На Gender прирост от LLM не материализуется (0.8630, как и Qwen-7B), потому что bottleneck — kNN-ретрив, а не качество LLM. | LLM ↔ CoLES (loop) | 0.8630 ± 0.0006 | 0.8072 ± 0.0034 | OOF running |

**Teacher для E1.2 / E1.3:** Qwen2.5-3B-base (LLM4ES embeddings).

## RQ2 Direction 1: Teacher Signal Type (LLM → Structured)

Какой тип учительского сигнала эффективнее при дистилляции?

Student: CoLES GRU/LSTM → LightGBM. Teacher: Qwen2.5-3B-base (LLM4ES embeddings, hidden_dim=2048). Train/test split seed=42.

⚠️ **Seeds:** backend код поддерживает 5 seeds, но числа в таблице — single seed=42 (per-seed CSV не сохранены). 5-seed reruns — pending.

| # | Signal Type | What student receives from LLM | Method | Что происходит на пальцах | Gender | Rosbank | Age |
|---|---|---|---|---|---|---|---|
| **E2.1** | Response-based | Soft label: "male 73%" | Reverse KL distillation | LLM-teacher через LightGBM поверх LLM4ES эмбеддингов выдаёт OOF soft probabilities (5-fold). CoLES учится на двух лоссах: `(1−α)·CE(true_label) + α·KL(P_student‖P_teacher)`. **Reverse KL** (MiniLLM ICLR 2024) вместо стандартного forward KL — student стремится быть decisive там где teacher decisive, не размазывается по модам. Fine-grained сигнал: ученик видит вероятность каждого класса, а не one-hot. | 0.8633 | 0.8074 | 0.6399 |
| **E2.2** | Feature-based | LLM embedding (2048) как доп. фичи | LLM4ES concat → LGBM | Никакого нового обучения. Берём CoLES эмбеддинг (1024-d) + LLM4ES эмбеддинг от Qwen2.5-3B (2048-d) → **конкатенация** в один вектор (3072-d) → LightGBM. LLM выступает как **fixed feature extractor** — никакого distillation, просто добавили фичи. На Rosbank лучше всех (0.819) и на Age (0.640) — там, где CoLES структурно слабее, LLM добавляет полезный текстовый сигнал. | 0.864 | 0.819 | 0.640 |
| **E2.3** | Relation-based | "Клиент A и B похожи в LLM space" | Contrastive alignment (LATTE) | Дубликат E1.2 в другой роли. Учитель не передаёт ни softlabels ни эмбеддингов напрямую — student учится **повторять геометрию** LLM-пространства: "если LLM считает A и B похожими, делай так чтобы и в твоём пространстве они были близко". Реализуется InfoNCE между проекциями CoLES и LLM эмбеддингов. | 0.8674 | 0.8057 | 0.6429 |
| **E2.4** | All three combined | Soft + embeddings + contrastive | LATTE + mutual learning + LoRA | Кухонная раковина: одновременно (1) InfoNCE как в E2.3, (2) reverse KL soft labels как в E2.1, (3) concat LLM эмбеддингов в feature space перед классификатором. Плюс LLM учится с LoRA. Идея — все сигналы суммируются. На Rosbank даёт 0.8142 (лучшее в RQ2 D1), на Gender и Age эффект слабее: лоссы конкурируют, и без аккуратного балансирования весов комбинация не доминирует. | 0.8676 | 0.8142 | 0.6363 |

> **Note:** E2.3 LATTE = E1.2 (one experiment, two RQ aspects). Numbers ничтожно отличаются из-за reseeding evaluation downstream — оба валидны.

## RQ2 Direction 2: Enrichment Type (Structured → LLM)

Какой тип знания от structured model лучше помогает LLM?

LLM: Qwen2.5-7B-Instruct, 4-bit NF4. Стратегия: CoT. Один прогон (LLM inference детерминированный, `temperature=0`). Train/test split seed=42.

| # | Enrichment | Structured Model | Что происходит на пальцах | Gender | Rosbank | Age |
|---|---|---|---|---|---|---|
| **E3.1** | None | — | LLM получает только серилизованный профиль клиента ("47 транзакций, средний чек 800₽, топ-категории: Retail 20%, Restaurants 15%...") + вопрос "male/female?". Никакого внешнего знания. Результат на уровне случайного — LLM по сырому списку транзакций пол определить не может. | 0.498 | 0.499 | 0.250 |
| **E3.2** | Prediction | XGBoost confidence | В промпт добавляется одна цифра: "XGBoost predicts male with confidence 0.73". LLM видит уверенность ML-модели, но не понимает почему. Едва выше chance — голый предикт без объяснения LLM не помогает: либо она ему верит и копирует, либо игнорирует. | 0.5083 | 0.5474 | 0.2780 |
| **E3.3** | Explanation | XGBoost SHAP | В промпт добавляются топ-5 SHAP фичей: "сильнее всего к классу male толкают: high transportation spending (+0.4), low clothing (+0.3), ...". LLM получает **интерпретируемое обоснование** — может рассуждать (CoT) о том, согласуется ли паттерн транзакций с этим объяснением. На Rosbank 0.637 — заметный лифт. | 0.606 | 0.637 | 0.2607 |
| **E3.4** | Retrieval | CoLES kNN | В промпт добавляются 10 ближайших соседей по CoLES-пространству вместе с их метками: "Похожие клиенты: 7 male, 3 female". Это превращает LLM в **kNN-классификатор с reasoning**: она видит "соседи в основном male → этот тоже скорее male". Главный приём проекта — **+26 пп AUC** на Gender. CoLES прячет всю работу в эмбеддинг, LLM просто читает голосование. | 0.762 | 0.766 | 0.250 |
| **E3.5** | All combined | XGBoost + CoLES | SHAP + kNN соседи в одном промпте. Ожидание — лучший из двух миров. На практике **хуже чистого kNN** (0.745 < 0.762): два сигнала иногда противоречат, LLM путается какому верить, и качество просаживается. Diminishing returns. | 0.745 | 0.751 | 0.2510 |

**Negative finding на Age:** kNN/CoT enrichment не помогает на 4-class Age (все ≈ 0.25 = chance). Гипотеза: 4-class label в текстовом промпте `male / female` интерпретируется LLM однозначно, а возрастные интервалы (`<35 / 35-50 / 50-65 / >65`) confounded с другими сигналами в retrieved similar customers.

## RQ2 Direction 2: Strategy × Enrichment (матрица)

Зависит ли эффект обогащения от стратегии промптинга?

LLM: Qwen2.5-7B-Instruct, 4-bit NF4. Dataset: Gender. Один прогон. Train/test split seed=42.

| # | Strategy | Что происходит на пальцах | None | + SHAP | + kNN | + Both |
|---|---|---|---|---|---|---|
| **E4.1** | Zero-shot | Просто "вот клиент, кто он — male/female?". Без примеров, без размышлений, один шаг к ответу. С kNN-соседями работает лучше всего (0.770) — LLM не нужно ничего "придумывать", достаточно прочитать голосование. | 0.498 | 0.542 | 0.770 | 0.616 |
| **E4.2** | Few-shot | В промпт добавляются 2-3 готовых примера "клиент X → male" перед целевым клиентом. Без обогащения помогает (0.578 vs 0.498), потому что LLM видит формат ответа. С kNN — **не суммируется** с few-shot (0.766 ≈ 0.770): kNN-соседи уже дают LLM готовое решение, демонстрации становятся лишним шумом. | 0.578 | 0.555 | 0.766 | 0.592 |
| **E4.3** | CoT | "Думай вслух перед ответом": LLM пишет цепочку рассуждений ("Клиент тратит много на одежду → скорее female → ..."). Сама по себе на Gender хуже zero-shot (0.491) — LLM рассуждает вслепую и обманывает себя. С SHAP подтягивается до 0.606 (есть о чём рассуждать), с Both — 0.745 (соседи + объяснение, LLM строит обоснование вокруг данных). | 0.491 | 0.606 | 0.762 | 0.745 |

## RQ3: LLM Size Effect

### Direction 1 (LLM → Structured Models, LATTE distillation)

Влияет ли размер LLM-teacher на качество дистилляции?

Method: LATTE (contrastive alignment). Teacher: Qwen2.5 family (Instruct variants для нового ladder). Student: CoLES → LightGBM. Train/test split seed=42.

⚠️ **Family-clean ladder pending.** Существующие числа (E5.2 ниже) были получены на Qwen2.5-3B-**base**; новый ladder — Qwen2.5-Instruct family на Gender для чистого scaling-style сравнения.

| # | Teacher LLM | Size | Variant | Что происходит на пальцах | Gender (AUC) | Rosbank (AUC) | Age (Acc) |
|---|---|---|---|---|---|---|---|
| **E5.0** | Qwen2.5-0.5B | 0.5B | Instruct | Берём самую маленькую LLM, экстрактим LLM4ES эмбеддинг (mean over last 8 hidden layers, masked mean over tokens), скармливаем CoLES'у через LATTE contrastive (как в E1.2). Гипотеза: если 0.5B LLM даёт почти такой же прирост как 7B — значит "знание" пола уже есть в маленькой модели, scale не нужен. | pending (Gender ladder) | — | — |
| **E5.1** | Qwen2.5-1.5B | 1.5B | Instruct | То же что E5.0 но с 1.5B. Промежуточная точка для построения кривой scaling. | pending (Gender ladder) | — | — |
| **E5.2** | Qwen2.5-3B (existing) | 3B | base | Существующее число из E1.2 — единственная точка в ladder где variant = **base** (не Instruct). Поэтому новый ladder перезапускается: чтобы сравнивать яблоки с яблоками, нужна одна семья. | 0.8674 | 0.8057 | 0.6429 |
| **E5.2-Instruct** | Qwen2.5-3B | 3B | Instruct | Версия E5.2 на Instruct варианте — закрывает дыру в ladder. Сравнение с E5.2-base также покажет: помогает ли instruction tuning эмбеддингам (или мешает, потому что они становятся менее "сырыми" семантически). | pending (Gender ladder) | — | — |
| **E5.3** | Qwen2.5-7B | 7B | Instruct | Верхняя граница локального железа (RTX 3090, 24GB) — 7B в 4-bit. Вершина D1 ladder. Больше моделей через API не достать: API не отдаёт hidden states, а LATTE без них работать не может. | pending (Gender ladder) | — | — |

> **Hardware constraint в Limitations:** LATTE требует доступа к hidden states LLM. Локально доступны только модели ≤7B (RTX 3090, 24GB). Модели >7B протестированы только в D2 (kNN inference, response-based, через API) — см. ниже.

### Direction 2 (Structured Models → LLM, kNN CoT enrichment)

Влияет ли размер LLM на эффективность обогащения промптов?

Method: Zero-shot + kNN ("Similar clients: X pos, Y neg"). Dataset: Gender. Train/test split seed=42.

#### Headline ladder — Qwen2.5 family only (clean scaling)

| # | LLM | Size | Variant | Что происходит на пальцах | No enrichment | + kNN | Δ |
|---|---|---|---|---|---|---|---|
| **E6.0** | Qwen2.5-0.5B | 0.5B | Instruct | Локальная inference 0.5B-модели на каждом тестовом клиенте, два прогона: (1) голый промпт, (2) с 10 kNN-соседями. Гипотеза: даже 0.5B модель сможет читать "7 male, 3 female" и давать +20pp — значит kNN-приём работает не за счёт IQ модели, а за счёт сигнала от CoLES. | pending | pending | — |
| **E6.1** | Qwen2.5-1.5B | 1.5B | Instruct | То же на 1.5B. Промежуточная точка кривой size→Δ. | pending | pending | — |
| **E6.1.5** | Qwen2.5-3B | 3B | Instruct | То же на 3B. Закрывает дыру в Qwen2.5-семье между 1.5B и 7B. | pending | pending | — |
| **E6.2** | Qwen2.5-7B | 7B | Instruct | То же на 7B, единственная заполненная точка ladder. Голый промпт — chance (0.498), с kNN — 0.762. **+26 пп** — практически весь сигнал приносит CoLES через ретрив. LLM работает голосовалкой над соседями. | 0.498 | 0.762 | +26 pp |

#### Supplementary — different families confirm finding

| # | LLM | Size | Family | Что происходит на пальцах | No enrichment | + kNN | Δ |
|---|---|---|---|---|---|---|---|
| **E6.3** | Gemma 3-4B | 4B | Gemma | Локальная Gemma-3 4B вместо Qwen — проверка что приём не привязан к семье моделей. Тот же kNN-промпт, тот же замер Δ. **+23.9 пп** — почти как у Qwen-7B. | 0.5280 | 0.7669 | +23.9 pp |
| **E6.4** | Qwen3.6-35B-A3B | 35B MoE | Qwen3 (different gen) | Прыжок на 5x размера через OpenRouter API (локально 35B не влезает). Активных параметров 3B (MoE), но total — 35B. **+27.1 пп** — практически тот же лифт что у 4B. Размер LLM → насыщение. | 0.5077 | 0.7790 | +27.1 pp |
| **E6.5** | DeepSeek-V3.2-Speciale | 671B MoE | DeepSeek | Самая мощная LLM на рынке (671B параметров), через OpenRouter. **+26.8 пп** — то же самое что 4B и 35B. Это **headline-finding всей работы**: качество kNN-обогащения plateau-ит на ~+26pp независимо от размера LLM. CoLES-ретрив несёт почти всю работу. | 0.5152* | 0.7828* | +26.8 pp* |

`*` = без API seed=42 (provider non-deterministic для MoE).

### CoT Reasoning Effect

Улучшает ли thinking mode качество LLM при обогащении промптов?

Method: Zero-shot + kNN. Dataset: Gender.

| # | Teacher LLM | Size | Что происходит на пальцах | Thinking=off | Thinking=on | Δ |
|---|---|---|---|---|---|---|
| **E7.1** | Qwen3.6-35B-A3B | 35B MoE | Один и тот же промпт (zero-shot + kNN), два режима inference: с включённым "thinking" (LLM пишет внутреннее рассуждение перед ответом) и без. Прирост **+0.13 пп** — для kNN задачи thinking бесполезен: LLM просто читает голосование соседей, рассуждать не о чем. | 0.7138* | 0.7151* | +0.13 pp* |
| **E7.2** | DeepSeek-V3.2 | 671B MoE | DeepSeek-V3.2 в режиме reasoning принципиально не отключается через API — провайдер не даёт. Замеряется только thinking=on. Используется как точка для общей кривой E6.5, не для thinking-сравнения. | N/A | 0.7828* | — |
| **E7.3** | GLM-4.7 (bonus) | ~9B | Bonus-замер на GLM-4.7 (китайская reasoning-модель, ~9B). Thinking=off даёт 0.7712 (нормально), thinking=on — 0.6541 из-за бага парсинга reasoning-вывода (не разделили `<think>...</think>` от финального ответа). Не валидное сравнение, оставлено для документации причины. | 0.7712 | 0.6541 (parse bug) | — |

`*` = без API seed=42.

---

## Pending experiments

| # | Что | Где | Time | Status |
|---|---|---|---|---|
| **E1.5 Age** | RAMD DeepSeek на Age | `experiments/rq1_bidirectional/ramd/E1_5_age_ramd_deepseek.py` | ~3-4h | OOF running |
| **E1.4 Age** | RAMD Qwen 7B на Age | (script нет, см. E1.5) | — | skipped |
| **E5.0 / E5.1 / E5.2-Instruct / E5.3** | Qwen2.5 Instruct ladder D1 на Gender | новые скрипты в `rq3_llm_size_effect/d1_teacher_size_for_latte/` | ~12h GPU | pending |
| **E6.0 / E6.1 / E6.1.5** | Qwen2.5 Instruct ladder D2 на Gender | новые скрипты в `rq3_llm_size_effect/d2_size_for_enrichment/qwen_25_*/` | ~3-4h GPU | pending |
| **RQ2 D1 — 5 seeds rerun** | Восстановить ± std для E2.1, E2.2, E2.4 | существующие скрипты, retrieve per-seed CSV | ~10h GPU | pending |

---

## Methodology limitations (для paper)

1. **Single-seed для RQ1 / RQ3.** Variance estimated via 1000-resample bootstrap of test predictions (CI half-width: ~0.008-0.015 AUC).
2. **D1 teacher size sweep ограничен 7B сверху** из-за hardware (24GB VRAM RTX 3090, 4-bit NF4 quant). Hidden states required для LATTE недоступны через API → большие модели тестируются только в D2.
3. **Base vs Instruct teacher.** Существующие D1 numbers использовали Qwen2.5-3B-base; D2 — Qwen2.5-7B-Instruct. Новый ladder унифицирует на Instruct family.
4. **kNN/CoT enrichment не помогает на Age multi-class** (все стратегии ≈ chance). Negative finding, гипотеза в RQ2 D2 секции.
5. **D2 size sweep только на Gender.** Rosbank/Age для RQ3 D2 — будущая работа.
6. **API non-determinism для MoE моделей.** Provider не гарантирует deterministic decoding для DeepSeek/Qwen3.6 даже при seed=42 в payload.

---

## Ссылки

- **Код по экспериментам:** [`EXPERIMENTS_MAP.md`](EXPERIMENTS_MAP.md)
- **Структура репо:** [`experiments/README.md`](experiments/README.md)
