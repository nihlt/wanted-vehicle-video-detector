"""Знаходить машини на кадрах і зберігає ці кадри з намальованими рамками.

  python tools/research/detect_cars.py dataset/images dataset/cars

Модель: YOLO11n ("nano"), вже натренована на наборі фото COCO (80 типів об'єктів).
З них беремо лише транспорт: car, motorcycle, bus, truck. Нічого не тренуємо.

Лишаємо тільки близькі машини: рамка має бути не меншою за --min-area пікселів (ширина × висота).
Поріг 25 000 (≈ 160×160, 1.2% кадру) підібрано за розміткою номерів: найменша машина,
у якої номер ще читається, мала рамку 176×171 = 30 096. Дальші машини номера все одно не дадуть.

Створює в <out>/:
  t0070.0.jpg ...  ті самі кадри з рамками машин (колір = клас, підпис = клас і впевненість 0..1);
  cars.json        рамки у пікселях [x, y, w, h], клас, впевненість.
Наприкінці друкує, скільки в середньому йде на кадр: читання файлу, детекція, малювання й запис.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

VEHICLES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}  # номери цих класів у COCO
COLORS = {'car': (60, 220, 60), 'motorcycle': (0, 200, 255), 'bus': (255, 160, 0), 'truck': (200, 80, 255)}  # BGR


def draw(img, x1, y1, x2, y2, name, conf):
    col = COLORS[name]
    cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
    label = f'{name} {conf:.2f}'
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    top = y1 - th - 10 if y1 > th + 10 else y1  # підпис над рамкою, а якщо не влазить — всередині
    cv2.rectangle(img, (x1, top), (x1 + tw + 8, top + th + 10), col, -1)
    cv2.putText(img, label, (x1 + 4, top + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('src')
    p.add_argument('out')
    p.add_argument('--model', default='models/yolo11n.pt')
    p.add_argument('--imgsz', type=int, default=640, help='до якого розміру зменшувати кадр перед детекцією')
    p.add_argument('--conf', type=float, default=0.25, help='мінімальна впевненість, нижче — відкидаємо')
    p.add_argument('--min-area', type=int, default=25000, help='мінімальна площа рамки в пікселях, менші — далеко')
    a = p.parse_args()

    files = sorted(Path(a.src).glob('*.jpg'))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(a.model)
    opts = dict(imgsz=a.imgsz, conf=a.conf, classes=list(VEHICLES), device=0, quantize=16, verbose=False,
                agnostic_nms=True)  # одна машина = одна рамка, навіть якщо модель вагається між car і bus

    # Перший прогін на відеокарті повільний (запуск CUDA), тому робимо його окремо, щоб не псував заміри.
    t = time.perf_counter()
    model.predict(cv2.imread(str(files[0])), **opts)
    print(f'розігрів моделі: {time.perf_counter() - t:.1f} с')

    t_read = t_det = t_draw = 0.0
    records, counts, n_far = [], {}, 0
    for i, f in enumerate(files, 1):
        t0 = time.perf_counter()
        img = cv2.imread(str(f))
        t1 = time.perf_counter()
        r = model.predict(img, **opts)[0]
        t2 = time.perf_counter()

        cars = []
        for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            name = VEHICLES[int(c)]
            x1, y1, x2, y2 = (round(v) for v in (x1, y1, x2, y2))
            if (x2 - x1) * (y2 - y1) < a.min_area:  # далека машина — пропускаємо
                n_far += 1
                continue
            cars.append({'cls': name, 'conf': round(s, 3), 'bbox': [x1, y1, x2 - x1, y2 - y1]})
            counts[name] = counts.get(name, 0) + 1
            draw(img, x1, y1, x2, y2, name, s)
        cv2.imwrite(str(out / f.name), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        t3 = time.perf_counter()

        t_read, t_det, t_draw = t_read + t1 - t0, t_det + t2 - t1, t_draw + t3 - t2
        records.append({'file': f'{Path(a.src).name}/{f.name}', 'cars': cars})
        print(f'{i}/{len(files)}  {f.stem}: {len(cars)} шт.  детекція {(t2 - t1) * 1000:.0f} мс')

    (out / 'cars.json').write_text(json.dumps(
        {'model': Path(a.model).name, 'imgsz': a.imgsz, 'conf': a.conf, 'min_area': a.min_area, 'images': records},
        ensure_ascii=False, indent=1), encoding='utf-8')

    n = len(files)
    print(f'\nБлизьких: {sum(counts.values())} ({", ".join(f"{k} {v}" for k, v in sorted(counts.items()))})'
          f' | відкинуто далеких: {n_far}')
    print(f'Середньо на кадр: читання {t_read / n * 1000:.0f} мс | детекція {t_det / n * 1000:.0f} мс '
          f'| малювання+запис {t_draw / n * 1000:.0f} мс')
    print(f'Лише детекція: {n / t_det:.0f} кадрів/с')
    print(f'-> {out}')


if __name__ == '__main__':
    main()
