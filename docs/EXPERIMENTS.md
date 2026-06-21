# Состояние относительно рабочих прогонов

## Что было в исходном GitHub

Исходный репозиторий содержал capture/runtime, remote clients, PCA/OBB, три shape backend-а, UI, сохранение и offline recompute. Это рабочий клиент, но не полный исследовательский pipeline.

Не были сохранены:

- сбор объединённого `all_pool` и rescue-переразметка;
- выбор только центрального сегмента для shape/size overlay;
- статистика и экспорт unique masks;
- COCO-конвертация SAM3 teacher labels;
- обучение и evaluation fine-tuned Mask R-CNN;
- сервер дообученной модели и webcam comparison;
- benchmark Locate Anything;
- конфиги/манифесты конкретных экспериментов.

## Матрица полноты

| Часть | Исходный GitHub | После этого обновления | Осталось сделать |
|---|---|---|---|
| RealSense/robot RGB-D capture | да | да, без адресов | hardware integration test |
| SAM3/RTDM remote client | да | да | новый безопасный wire protocol |
| RGB-D размеры + offline recompute | да | да | калибровочный benchmark |
| Strict 3D cylinder/box | частично | документирован | перенести финальные tuned thresholds |
| Normals/planes roll risk | ранняя v1 | v2: cross-section, axis, partial-axis, compact/semantic | проверить на сохранённом all_pool |
| all-pool/rescue overlays | нет | только формат run и COCO import | восстановить QC/rescue tool по исходным данным |
| Unique-mask analysis | нет | heuristic grouping tool | embedding + physical SKU IDs |
| SAM3 -> Mask R-CNN training | нет | полный train/eval/serve contour | GPU regression run |
| RGB+depth reject classifier | нет | нет | отдельная архитектура и labels accept/reject |
| Locate Anything backend | нет | нет | adapter + benchmark |

### История расхождения критериев

Первый публичный snapshot содержал только раннее решение по plane coverage, normal clusters и weighted score. В `surface_normals_v2` добавлены зафиксированные поздние правила cross-section rotational support, axis-supported/partial-axis цилиндров, compact-round и semantic roll-risk. Коэффициент `0.45 * curved_surface_ratio` остаётся диагностикой, но больше не является самостоятельной причиной отказа. Синтетические regression-тесты добавлены; повторный benchmark на исходном `all_pool` всё ещё обязателен, поскольку сами RGB-D артефакты не хранятся в публичном Git.

## Зафиксированные результаты прошлых прогонов

Набор SAM3 teacher COCO: `4233 train / 899 val / 257 test`. Лучшим из проверенных был full-data checkpoint после epoch 2. На downstream-наборе 40 SKU:

| Segmentation | Axis MAE | Sorted MAE |
|---|---:|---:|
| SAM3 teacher | 38.7 mm | 36.0 mm |
| исходный RTDM | 85.5 mm | 74.7 mm |
| RTDM fine-tune unique | 68.5 mm | 66.3 mm |
| RTDM fine-tune full data | 52.5 mm | 47.3 mm |

Full-data fine-tuning улучшил исходный RTDM примерно на `38.6%` по axis MAE и `36.6%` по sorted MAE, но не достиг teacher SAM3. Эти числа перенесены из журналов рабочего прогона; веса и исходные данные в публичный Git не включены.

## Практический вывод

Для обучения использовать полный очищенный набор с ограничением повторов sampler-ом или group-aware split. Unique subset полезен для контроля разнообразия и честной оценки, но сам по себе оказался слишком мал и хуже передал вариативность масок. Следующий обязательный шаг качества: физические SKU ID и отдельный test по объектам, которых модель не видела в соседних кадрах.
