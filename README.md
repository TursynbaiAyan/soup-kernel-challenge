# Soup Kernel Challenge: слитый `lm_head` + cross-entropy на Triton

Решение кейса [Soup](https://github.com/MakazhanAlpamys/Soup) на Decentrathon 6.0. Тот же
`soup train` (Qwen2.5-1.5B-Instruct, NF4, LoRA r=16 на q/v, batch 4, 512 токенов, 64 шага) на
Kaggle T4, но голова модели и функция потерь считаются тремя нашими Triton-кернелами: без
тензора логитов `[batch, seq, 151 936]` и только по реальным (не pad) позициям.

- **Kaggle-ноутбук (публичная сохранённая версия):** `<ссылка на Kaggle-ноутбук>`
- **Репозиторий:** https://github.com/TursynbaiAyan/soup-kernel-challenge
- **Финальный коммит:** `<ссылка на коммит>`
- **Презентация:** [docs/Soup-Kernel-Brightly.pptx](docs/Soup-Kernel-Brightly.pptx)
  (генератор `docs/build_deck.cjs`, цифры из `docs/results.json`)
- **Демо-видео (2:01):** [docs/video/Soup-Kernel-Brightly-demo.mp4](docs/video/Soup-Kernel-Brightly-demo.mp4)
  (исходники в `docs/video/`: canvas-анимация, озвучка, рендер)
- **Ноутбук для Kaggle с этим репозиторием:** [kaggle/soup_kernel_run.ipynb](kaggle/soup_kernel_run.ipynb)

## Что сделано и почему

### Где было узкое место

Профиль базового Soup (README кейса): около 7% времени шага — softmax и cross-entropy по словарю
в 151 936 токенов, а логиты занимают гигабайты памяти. Разбор кода transformers 5.16.1,
TRL 0.29.1 и Soup 0.77.0 показал, что на каждом шаге базовый прогон делает следующее:

1. приводит fp32-вес `lm_head` (он общий с эмбеддингами, `prepare_model_for_kbit_training`
   держит его в fp32) к fp16: копия на 467 МБ, которая живёт до backward;
2. строит fp16-логиты `[4, T, 151 936]` для **всех** позиций, включая pad;
3. `ForCausalLMLoss` приводит их к fp32 (ещё 1,2 ГБ при T=512) и держит буферы softmax и
   градиента той же формы;
4. TRL ещё раз читает полные логиты для `entropy` и `mean_token_accuracy`, а Soup — для
   loss каждой строки в `rewind.jsonl`.

Весь «хвост» модели занимает около 3,8 ГБ сверх весов и активаций (замер ниже). При этом голова
заморожена: никому, кроме loss и метрик, логиты не нужны.

### Наши кернелы (`solution/fused_ce.py`)

Позиции с `attention_mask = 1` собираются в один список (pad отбрасывается — это разрешено
правилами: строки разных батчей не смешиваются). Сначала идут позиции с меткой, потом
остальные. Дальше чанками по 1 024 позиции:

| Кернел | Что делает |
|---|---|
| `lm_head_logits_kernel` | логиты чанка = h @ Wᵀ. W читается **в fp32** и округляется до fp16 в регистрах (то же округление, что у autocast), поэтому копии весов на 467 МБ нет. Аккумуляция в fp32, результат в fp16, как у cuBLAS под autocast. Порядок тайлов — сначала по строкам, чтобы блоки с одним куском W шли подряд и брали его из L2. |
| `ce_stats_grad_kernel` | один program на позицию, online-softmax по словарю в fp32: running max, Σe^(x−m), Σe^(x−m)·x. Отсюда log-sum-exp, **энтропия** (lse − E[x]), **argmax** (первый индекс максимума, как `torch.argmax`) и **cross-entropy метки**. Для позиций с меткой второй проход пишет на место логитов градиент (softmax − onehot)·256 в fp16. Множитель 256 поднимает малые вероятности над порогом fp16 и снимается в fp32 позже. |
| `lm_head_dgrad_kernel` | dh = G @ W с fp32-весом, округлённым в регистрах. Свёртка идёт по словарю (K = 151 936), поэтому split-K: частичные суммы складываются `atomic_add` в fp32. Считается только для позиций с меткой. |

Градиент по скрытым состояниям вычисляется прямо в forward: голова заморожена, ничего больше не
нужно. В backward остаётся только умножить его на градиент loss (в нём сидит множитель
GradScaler). В памяти одновременно живёт лишь буфер чанка `1 024 × 151 936` fp16 (около 311 МБ).

Размеры тайлов фиксированы, без autotune: время компиляции входит в замер. Для аргументов, которые
меняются от шага к шагу, стоит `do_not_specialize`, поэтому каждый кернел компилируется **ровно
один раз** за прогон. Разделяемая память — 24–32 КБ на блок при пределе 64 КБ на T4.

### Как это подключено (`solution/patch_ce.py`)

- `Qwen2ForCausalLM.forward`: в обучающем forward с метками (только когда его вызвал наш
  `compute_loss`) декодер работает как обычно, затем вместо `lm_head` и `ForCausalLMLoss`
  вызывается `fused_lm_head_loss`. Возвращается `loss` и `logits=None`. Нормировка — как в
  `ForCausalLMLoss`: сумма / `num_items_in_batch`, если его передали, иначе среднее.
- `trl.SFTTrainer.compute_loss`: делает то же, что TRL перед делегированием, вызывает
  `transformers.Trainer.compute_loss`, затем пишет `entropy`, `num_tokens` и
  `mean_token_accuracy` **по формулам TRL 0.29.1** из значений, посчитанных кернелами. Loss
  каждой позиции кладётся туда, откуда recorder Soup его читает, когда у шага нет логитов
  (`trainer._soup_chunked_step.token_loss`, см. `soup_cli/trainer/rewind_hf.py`). Встроенный
  режим Soup `stream_loss_chunk` **не включается**: используется только вход его recorder.
- Всё, что под условия не подходит (eval, нет меток, другая модель, Liger, label smoothing,
  свой `compute_loss_func`, prompt tuning, context parallel, нет fp16/bf16 autocast), идёт
  исходным путём без изменений. Патчи защищены так, что 79 тестов Soup проходят.
- Пример `solution/swiglu.py` оставлен включённым (снижает пик ещё примерно на 2 ГБ). Своими
  кернелами мы его не считаем.

### Источники идей (код написан нами)

- Chunked fused linear + cross-entropy: Liger-Kernel, `fused_linear_cross_entropy`
  (https://github.com/linkedin/Liger-Kernel); Cut Your Losses in Large-Vocabulary Language
  Models, Wijmans et al., 2024 (https://arxiv.org/abs/2411.09009).
- Online softmax: Milakov & Gimelshein, *Online normalizer calculation for softmax*, 2018
  (https://arxiv.org/abs/1805.02867).
- Порядок тайлов для переиспользования L2 и split-K: туториал Triton «Matrix Multiplication»
  (https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html).
- Интерфейс шага без логитов для rewind-лога: `soup_cli/utils/stream_loss_chunk.py` и
  `soup_cli/trainer/rewind_hf.py` из Soup 0.77.0.

## Как запустить

### На Kaggle (так проверяет жюри)

1. Kaggle → **Create → New Notebook → File → Import Notebook** → `templates/kaggle_notebook.ipynb`.
2. Справа: **Accelerator: GPU T4 x2**, **Internet: on**.
3. В первой ячейке: `REPO_URL = "https://github.com/TursynbaiAyan/soup-kernel-challenge.git"`, `COMMIT = "<финальный коммит>"`.
4. **Run All** (около 10 минут). В конце `harness/run_task.py` печатает `speedup`,
   `memory base/solution` и `eval loss vs base`, а `harness/check_rules.py` — 10 проверок.

### Локально (Linux, CUDA GPU)

```bash
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python -m pytest solution -q                              # наши кернелы против PyTorch
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q           # 79 тестов Soup с нашими патчами
python task/prepare_data.py && python harness/run_task.py && python harness/check_rules.py
```

`solution/test_kernels.py` сравнивает loss, энтропию, loss позиций, accuracy и градиент по
hidden с эталоном, который повторяет путь autocast (fp16-логиты, fp32 loss, формулы TRL). Есть
кейсы с несколькими чанками, нечётным словарём, `num_items_in_batch` и реальным размером
головы 1 536 × 151 936.

## Результаты

### Kaggle T4 (из `runs/result.json` нашего ноутбука)

| | База Soup | Наше решение | Отношение |
|---|---|---|---|
| Время `soup train`, с | `<…>` | `<…>` | speedup `<…>`x |
| Пик памяти GPU (nvidia-smi), МиБ | `<…>` | `<…>` | `<…>`x |
| Eval loss (публичные строки) | `<…>` | `<…>` | `<…>` % |
| Доля наших кернелов (проверка 4) | — | `<…>` % | ≥ 5% |

### Локальные замеры, проверенные до Kaggle (RTX 3050 6 GB, torch 2.14.1, triton-windows 3.7.1)

Голова и loss одного шага, 4 строки длиной до 400 токенов (1 160 реальных позиций), реальный
размер 1 536 × 151 936, forward + backward:

| | Эталон (autocast `lm_head` + `ForCausalLMLoss`) | Наши кернелы |
|---|---|---|
| Время | 218 мс | **93 мс** (в 2,3 раза быстрее) |
| Дополнительный пик памяти | 3 800 МиБ | **295 МиБ** (в 13 раз меньше) |

По кернелам на 1 024 позициях: логиты 35,0 мс против 36,5 у cuBLAS fp16 (при том что мы читаем
fp32-веса), градиент 34,9 мс против 32,2, cross-entropy 7,6 мс.

Сквозной прогон: TRL `SFTTrainer` с `RewindSFTTrainer` из Soup, PEFT LoRA r=16 на q/v, fp16,
2-слойная Qwen2 с настоящим словарём и токенизатором, 12 шагов. База против решения:

| Величина | Отличие от базы |
|---|---|
| loss по шагам | первый шаг 0,0001%, в среднем 0,0001% (порог 1% / 3%) |
| entropy, mean_token_accuracy, num_tokens | 0,0000% |
| loss строк в rewind | в среднем 0,0002%, максимум 0,0005%; id строк и число токенов совпадают |
| grad_norm | в среднем 0,014% |
| пик памяти PyTorch | 2 030 → 473 МиБ |

Тесты Soup с патчами: 79 passed. Статический разбор `check_rules.py`: все три кернела
засчитаны как наши, сходство с примером 0,19–0,32 (порог ручной проверки — 0,8), запрещённых
импортов нет.

## Ограничения и что дальше

- Числа выше сняты не на T4. Тайлы подбирались на Ampere с запасом по разделяемой памяти для
  Turing, но итог на T4 может отличаться. Таблица Kaggle — источник правды.
- Ускоряется только голова. Остальные ~44% шага (RMSNorm, RoPE, приведения типов, LoRA) и
  распаковка NF4 остаются как в базе. Следующие шаги: слитый RMSNorm с приведением к fp16,
  слитая LoRA-добавка в q/v, NF4-dequant внутри matmul.
- Pad убирается только в голове. Декодер по-прежнему считает внимание и MLP по паддингу.
- `atomic_add` в split-K делает градиент недетерминированным на уровне округления (разброс
  grad_norm около 0,05%, намного меньше допусков).
- fp32-вес эмбеддингов (0,93 ГБ) остаётся в памяти: он нужен для поиска эмбеддингов, а
  хранить его в fp16 значило бы менять модель.
- Буфер чанка (311 МБ) — компромисс скорость/память: при `CHUNK = 512` памяти меньше, но
  fp32-веса читаются вдвое чаще.

## Лицензия

Apache-2.0, см. `LICENSE` и `NOTICE`. Пять тестовых файлов взяты из Soup v0.77.0 (Apache-2.0).
Данные: [AmanMussa/kazakh-instruction-v2](https://huggingface.co/datasets/AmanMussa/kazakh-instruction-v2)
(MIT). Модель: [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
(Apache-2.0).
