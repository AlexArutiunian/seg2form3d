# Architecture

## Поток данных

```text
RealSenseSource / RobotCameraSource
                |
             RGBDFrame
                |
      +---------+---------+
      |                   |
 Remote SAM3       Remote RTDM/Mask R-CNN
      |                   |
      +---------+---------+
                |
      SegmentationInstance[]
                |
       MeasurementEngine
  mask + aligned depth + intrinsics
        |                  |
 robust PCA/OBB       shape backends
        |           legacy / strict / normals
        +---------+--------+
                  |
       ACCEPT / REJECT / UNKNOWN
                  |
            UI + RunWriter
```

Камера, сегментация и геометрия связаны типами из `src/sku_rgbd/models.py`. Замена сегментатора не должна менять measurement-код; изменение критериев не требует повторного GPU-инференса по уже сохранённым mask-index.

## Train-контур

```text
saved runs -> prepare_coco.py -> COCO all
                              -> deduplicate_coco.py -> unique groups/report
COCO all -> train_maskrcnn.py -> checkpoints -> evaluate_maskrcnn.py
                                      |
                               serve_maskrcnn.py
                                      |
                           RemoteSegmentationClient
```

Основное обучение использует все качественные teacher masks. Группы near-duplicates применяются для статистики, балансировки и group-aware split. Это сохраняет вариативность поз/освещения, но не позволяет соседним кадрам одного объекта завысить test-метрики.

## Контракты

- `CameraSource.read() -> RGBDFrame`.
- `RemoteSegmentationClient.segment() -> SegmentationResult`.
- `MeasurementEngine.measure_all(frame, instances) -> MeasurementResult[]`.
- Run-папка содержит исходные данные, достаточные для offline-пересчёта.
- Inference-сервер слушает только loopback; внешний доступ идёт через SSH.

## Ограничения

- Текущий wire protocol использует legacy `pickle` для совместимости с существующими SAM3/RTDM-серверами. Он допустим только внутри доверенного loopback/SSH-канала.
- Pipeline не содержит production tracking и глобальной SKU identity.
- Depth обязателен для размеров и 3D-формы; RGB-only режим даёт только segmentation.
- Locate Anything пока не интегрирован: API и benchmark должны быть добавлены отдельным backend adapter, не в measurement engine.

