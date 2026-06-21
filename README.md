# seg2form3d

RGB-D pipeline для складских SKU: камера -> instance segmentation -> 3D-размеры и форма -> решение `ACCEPT / REJECT / ROBOT / UNKNOWN`.

## Что входит

- RealSense и RGB-D камера робота;
- параллельные удалённые backend-ы SAM3 и RTDM/Mask R-CNN;
- сохранение RGB, depth, mask-index, overlay и metadata;
- offline-пересчёт геометрии без повторной сегментации;
- строгая 3D-проверка цилиндров и экспериментальная оценка normals/planes;
- полный teacher-student train pipeline: run-папки -> COCO -> dedup -> fine-tune -> COCO evaluation -> inference server.

GPU-веса, датасеты, адреса лаборатории и ключи в Git не входят.

## Быстрый старт

```bash
git clone https://github.com/AlexArutiunian/seg2form3d.git
cd seg2form3d
cp .env.example .env
./scripts/setup.sh
```

Заполните локальный `.env`. Для inference-серверов используйте SSH-туннели:

```bash
./scripts/start_tunnels.sh
./scripts/run_realsense.sh
# или
./scripts/run_robot.sh
```

`s` сохраняет кадр, `q`/`Esc` завершает работу. Каждый запуск создаёт `data/run_YYYYMMDD_HHMMSS` со структурой `camera/`, `sam3/`, `rtdm/`, `grid/`.

## Пересчёт

Сохранённые маски можно пересчитать после изменения геометрии:

```bash
source .venv/bin/activate
sku-rgbd-recompute --run-dir data/run_... --backend sam3 --out-dir data/recomputed
```

## Обучение

```bash
source .venv/bin/activate
pip install -r train/requirements.txt

python train/prepare_coco.py data/run_* \
  --backend sam3 \
  --output data/training/sam3_teacher_coco

./train/run_train.sh

python train/evaluate_maskrcnn.py \
  --dataset data/training/sam3_teacher_coco \
  --checkpoint runs/maskrcnn_sam3_teacher/maskrcnn_epoch_012.pt
```

Полная процедура и правила разделения данных: [train/README.md](train/README.md).

## Документы

- [Архитектура](ARCHITECTURE.md)
- [Критерии решения](docs/DECISION_RULES.md)
- [Сравнение с рабочими прогонами](docs/EXPERIMENTS.md)
- [Безопасность и секреты](SECURITY.md)

## Проверка

```bash
pip install -e ".[dev]"
pytest -q
python scripts/security_check.py
```

На полностью offline-машине без `pytest`: `PYTHONPATH=src python3 scripts/run_tests_without_pytest.py`.

Проект исследовательский: пороги формы и размеров должны проходить отдельную валидацию на SKU заказчика перед production-решениями.
