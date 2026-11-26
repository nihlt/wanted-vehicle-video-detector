"""Номери нейромережами: детектор табличок (YOLOv9-t) + читання тексту (fast-plate-ocr).

  python tools/plates_nn.py dataset

Моделі лежать у models/plates/ (обидві з GitHub ankandrew, ліцензія MIT):
  yolo-v9-t-384-license-plates-end2end.onnx  — знаходить табличку номера на фото машини (7.8 МБ);
  cct_s_v2_global.onnx                       — читає текст таблички: латинські літери й цифри до 10 символів,
                                               навчена на номерах 60+ країн, серед них Україна (5.3 МБ).

На відміну від класичного пошуку (find_plates.py), детектор уміє відповісти «номера немає»,
а OCR дає текст і впевненість для кожного символу — це знадобиться для голосування по треку.

Що робить main(): для кожної вирізки машини з dataset/car_crops шукає номер і читає його,
порівнює з ручною розміткою (dataset/labels_draft.json) так само, як find_plates.py,
і малює аркуші dataset/preview/nn_plates_NN.jpg (фото машини з рамкою + вирізка номера + текст).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Номери на відеокарті: змінна оточення PLATES_DEVICE=gpu. Глобально стоїть CPU-версія onnxruntime 1.20.1
# (її потребують інші програми), тому GPU-версію 1.26.0 (остання під CUDA 12, як у torch) поставлено окремо
# в tools/.ort_gpu і підставляємо її першою. CUDA і cuDNN беруться з torch: на Windows `import torch`
# завантажує всі бібліотеки з torch/lib, тож torch імпортуємо раніше за onnxruntime.
GPU = os.environ.get('PLATES_DEVICE') == 'gpu'
if GPU:
    assert 'onnxruntime' not in sys.modules, 'onnxruntime уже завантажено: GPU-версію підставити не вийде'
    import torch  # noqa: F401
    sys.path.insert(0, str(Path(__file__).resolve().parent / '.ort_gpu'))
    import onnxruntime
    assert onnxruntime.__version__ == '1.26.0' and '.ort_gpu' in onnxruntime.__file__, onnxruntime.__file__

import cv2
import numpy as np
from fast_plate_ocr import LicensePlateRecognizer
from open_image_models.detection.core.yolo_v9.inference import YoloV9Detector

from find_plates import evaluate

MODELS = Path(__file__).resolve().parent.parent / 'models' / 'plates'


def full_speed():
    """Просить Windows не пригальмовувати цей процес (EcoQoS).

    Скрипт без вікна Windows вважає фоновим і через кілька секунд переносить на енергоощадні ядра
    гібридного процесора (i5-13500H: 4 швидких + 8 ощадних). Номери тоді читаються в ~10 разів повільніше.
    Діє лише на поточний процес, системні налаштування не змінює.
    """
    import ctypes
    import sys
    from ctypes import wintypes
    if sys.platform != 'win32':
        return False

    class State(ctypes.Structure):
        _fields_ = [('Version', wintypes.ULONG), ('ControlMask', wintypes.ULONG), ('StateMask', wintypes.ULONG)]

    k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.SetProcessInformation.restype = wintypes.BOOL
    execution_speed, process_power_throttling = 0x1, 4
    st = State(1, execution_speed, 0)  # ControlMask: керуємо швидкістю; StateMask 0: не пригальмовувати
    ok = k32.SetProcessInformation(k32.GetCurrentProcess(), process_power_throttling,
                                   ctypes.byref(st), ctypes.sizeof(st))
    if not ok:
        print('full_speed: Windows відмовив, код помилки', ctypes.get_last_error(), flush=True)
    return bool(ok)


class PlateReader:
    def __init__(self, models=MODELS, conf=0.25):
        # use_tf32=0: на відеокарті рахуємо в повній точності FP32, як на процесорі (TF32 змінює розмиті читання)
        prov = ([('CUDAExecutionProvider', {'use_tf32': 0})] if GPU else []) + ['CPUExecutionProvider']
        self.det = YoloV9Detector(models / 'yolo-v9-t-384-license-plates-end2end.onnx', ['License Plate'],
                                  conf_thresh=conf, providers=prov)
        self.ocr = LicensePlateRecognizer(onnx_model_path=models / 'cct_s_v2_global.onnx',
                                          plate_config_path=models / 'cct_s_v2_global_plate_config.yaml',
                                          providers=prov)
        if GPU:  # якщо CUDA не підхопилась, onnxruntime мовчки працює на процесорі — тоді краще впасти одразу
            used = {name: m.model.get_providers() for name, m in (('детектор', self.det), ('OCR', self.ocr))}
            if any(v[0] != 'CUDAExecutionProvider' for v in used.values()):
                raise RuntimeError(f'номери не на відеокарті: {used}')

    def detect(self, img):
        """Фото машини (BGR) -> [(x1, y1, x2, y2, conf), ...], найвпевненіші спершу."""
        out = [(d.bounding_box.x1, d.bounding_box.y1, d.bounding_box.x2, d.bounding_box.y2, float(d.confidence))
               for d in self.det.predict(img)]
        return sorted(out, key=lambda b: -b[4])

    def read(self, plates):
        """Вирізки номерів (BGR) -> [(текст, [впевненість кожного символу]), ...]."""
        if not plates:
            return []
        preds = self.ocr.run([cv2.cvtColor(p, cv2.COLOR_BGR2RGB) for p in plates], return_confidence=True)
        return [(p.plate, [float(c) for c in np.ravel(p.char_probs)[:len(p.plate)]]) for p in preds]


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def put(img, s, org, scale=0.5, col=(255, 255, 255)):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ds')
    a = p.parse_args()
    ds = Path(a.ds)
    crops = json.loads((ds / 'car_crops' / 'crops.json').read_text(encoding='utf-8'))['crops']
    labels = json.loads((ds / 'labels_draft.json').read_text(encoding='utf-8'))
    reader = PlateReader()

    results, tiles, t_det, t_ocr = [], [], 0.0, 0.0
    for c in crops:
        img = cv2.imread(str(ds / 'car_crops' / c['file']))
        t0 = time.perf_counter()
        found = reader.detect(img)
        t_det += time.perf_counter() - t0
        rec = dict(c, plate=None)
        text, probs, plate_img = None, [], None
        if found:
            x1, y1, x2, y2, conf = found[0]
            x1, y1 = max(0, x1), max(0, y1)
            plate_img = img[y1:y2, x1:x2]
            t0 = time.perf_counter()
            text, probs = reader.read([plate_img])[0]
            t_ocr += time.perf_counter() - t0
            ox, oy = c['origin']
            rec['plate'] = {'bbox': [x1, y1, x2 - x1, y2 - y1], 'frame_bbox': [ox + x1, oy + y1, x2 - x1, y2 - y1],
                            'conf': round(conf, 3), 'text': text, 'char_probs': [round(v, 3) for v in probs]}
        results.append(rec)

        # плитка для аркуша: машина з рамкою номера, під нею вирізка номера і прочитаний текст
        tile = np.full((250, 240, 3), 40, np.uint8)
        k = min(236 / img.shape[1], 150 / img.shape[0])
        th = cv2.resize(img, (round(img.shape[1] * k), round(img.shape[0] * k)), interpolation=cv2.INTER_AREA)
        if found:
            x1, y1, x2, y2, _ = found[0]
            cv2.rectangle(th, (round(x1 * k), round(y1 * k)), (round(x2 * k), round(y2 * k)), (0, 255, 0), 2)
        tile[20:20 + th.shape[0], 2:2 + th.shape[1]] = th
        put(tile, c['file'][:-4], (4, 14), 0.42)
        if plate_img is not None and plate_img.size:
            pl = cv2.resize(plate_img, (180, 45), interpolation=cv2.INTER_CUBIC)
            tile[174:219, 2:182] = pl
            put(tile, f'{text}  {min(probs) if probs else 0:.2f}', (4, 240), 0.55, (120, 255, 120))
        else:
            put(tile, 'nomera nemae', (4, 200), 0.55, (160, 160, 255))
        tiles.append(tile)

    reachable, ok, wrong, neutral = evaluate(results, labels)
    # текст: для влучних номерів порівнюємо прочитане з розміченим
    by_id = {(stem, q['id']): q for stem, ps in labels.items() for q in ps}
    from find_plates import iou
    exact, dist = 0, []
    for r in results:
        stem = Path(r['frame']).stem
        if stem not in labels or not r['plate']:
            continue
        best = max(labels[stem], key=lambda q: iou(q['bbox'], r['plate']['frame_bbox']))
        if best['text'] and iou(best['bbox'], r['plate']['frame_bbox']) >= 0.3:
            d = edit_distance(r['plate']['text'], best['text'])
            dist.append((best['text'], r['plate']['text'], d))
            exact += d == 0
    n = len(crops)
    none = sum(r['plate'] is None for r in results)
    print(f'вирізок машин {n}, номер знайдено на {n - none}, «номера немає» на {none}')
    print(f'розмічені номери: знайдено {len(ok)} з {len(reachable)}, хибних рамок {len(wrong)}, '
          f'влучань у нечитабельні {neutral}')
    print(f'текст: прочитано точно {exact} з {len(dist)} знайдених; помилки (еталон -> прочитано, символів):')
    for gt, txt, d in sorted(dist, key=lambda x: -x[2]):
        print(f'   {gt:<10} -> {txt:<10} {d}')
    print(f'швидкість на процесорі: детектор {t_det / n * 1000:.0f} мс/машину, OCR {t_ocr / max(1, n - none) * 1000:.1f} мс/номер')
    print('хибні рамки:', wrong)

    prev = ds / 'preview'
    prev.mkdir(exist_ok=True)
    for old in prev.glob('nn_plates_*.jpg'):
        old.unlink()
    cols, per = 7, 42
    for s in range(0, len(tiles), per):
        chunk = tiles[s:s + per]
        rows = [np.hstack(chunk[i:i + cols] + [np.full_like(tiles[0], 40)] * (cols - len(chunk[i:i + cols])))
                for i in range(0, len(chunk), cols)]
        cv2.imwrite(str(prev / f'nn_plates_{s // per:02d}.jpg'), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    (ds / 'plates_nn.json').write_text(json.dumps({'results': results}, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
