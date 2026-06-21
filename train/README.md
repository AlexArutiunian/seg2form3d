# Training pipeline

Цель: обучить быструю Mask R-CNN повторять качественные маски медленного SAM3, не смешивая повторные кадры одного SKU между train и test.

## 1. Подготовка teacher masks

Исходная run-папка должна содержать совпадающие имена:

```text
run_.../
  camera/rgb/*.png
  sam3/mask_index/*.png
  sam3/meta/*.json
```

Собрать полный COCO dataset:

```bash
python train/prepare_coco.py data/bag_runs_tight_roi/run_* \
  --backend sam3 \
  --selection all \
  --output data/training/sam3_teacher_coco
```

`--selection center` воспроизводит правило measurement-разметки: выбирается ближайший к центру сегмент; если несколько находятся в пределах нормализованной дистанции `0.03`, выбирается самый большой. Для обучения instance segmentation обычно нужен `all` после ручной проверки teacher masks.

## 2. Уникальность

```bash
python train/deduplicate_coco.py \
  --images data/training/sam3_teacher_coco/train \
  --annotations data/training/sam3_teacher_coco/annotations_train.json \
  --output data/training/sam3_teacher_coco/unique_groups_train.json
```

Метод объединяет HSV-гистограмму masked RGB crop и нормализованный силуэт. Это near-duplicate heuristic, а не распознавание SKU. Порог необходимо проверить вручную по случайной выборке групп. Для итогового benchmark предпочтительно разбивать данные по физическому SKU/run, а не по отдельным кадрам.

## 3. Fine-tuning

```bash
pip install -r train/requirements.txt
cp .env.example .env
./train/run_train.sh
```

По умолчанию используется `maskrcnn_resnet50_fpn_v2` с COCO weights, заменёнными heads для классов `background + sku`. Каждый epoch сохраняется отдельно; `history.json` содержит loss и ETA. Возобновление:

```bash
./train/run_train.sh --resume runs/.../maskrcnn_epoch_006.pt
```

## 4. Evaluation

```bash
python train/evaluate_maskrcnn.py \
  --dataset data/training/sam3_teacher_coco \
  --split test \
  --checkpoint runs/.../maskrcnn_epoch_012.pt
```

Сохраняются COCO mask/bbox `AP`, `AP50`, `AP75`, `AR100`. Выбирать checkpoint только по validation; test запускать один раз для итогового отчёта. Размерный benchmark по 40 SKU является отдельной downstream-проверкой и не заменяет mask AP.

## 5. Serving

```bash
PYTHONPATH=src:train python train/serve_maskrcnn.py \
  --checkpoint runs/.../maskrcnn_epoch_012.pt \
  --bind 127.0.0.1 --port 5557
```

Порт нельзя публиковать наружу. Подключение с клиента выполняется через `scripts/start_tunnels.sh`.

## Что не хранится в Git

RGB/depth, COCO datasets, `.bag`, checkpoints и `.env`. Для воспроизводимости сохраняются `manifest.json`, commit SHA, конфигурация запуска, `history.json` и metrics JSON рядом с артефактами эксперимента.
