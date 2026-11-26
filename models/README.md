# Моделі

| Файл | Що робить | Звідки | Ліцензія |
|---|---|---|---|
| `yolo11n.pt` | знаходить машини, автобуси, вантажівки, мотоцикли (COCO) | [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) | AGPL-3.0 |
| `plates/yolo-v9-t-384-license-plates-end2end.onnx` | знаходить табличку номера на фото машини | [ankandrew/open-image-models](https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-t-384-license-plates-end2end.onnx) | MIT, [`plates/LICENSE-open-image-models`](plates/LICENSE-open-image-models) |
| `plates/cct_s_v2_global.onnx` + `plates/cct_s_v2_global_plate_config.yaml` | читає текст таблички (латиниця й цифри, номери 60+ країн) | [ankandrew/fast-plate-ocr](https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/cct_s_v2_global.onnx) | MIT, [`plates/LICENSE-fast-plate-ocr`](plates/LICENSE-fast-plate-ocr) |

Файли покладено в репо, щоб усе запускалось без додаткових завантажень. Посилання вище — оригінали.
