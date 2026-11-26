"""Шукає номерний знак на кожному фото машини. Класичний комп'ютерний зір, без нейромереж.

  python tools/find_plates.py dataset [--prep none|clahe|blur|sharpen]

За замовчуванням --prep sharpen: на розмітці він дав найкращий результат (16 з 18 номерів, 6 хибних рамок;
без обробки — 15 і 8, clahe — 16 і 9, blur — 13 і 10). Вибірка мала, тож різниця в 1–2 номери — орієнтир, не доказ.

Ідея: номер — світлий прямокутник, на якому щільно, в один ряд, стоять темні символи.
  1. Фото машини зменшуємо до ширини 400 px, щоб розміри в пікселях були однакові для великих і малих машин.
  2. Попередня обробка (--prep): clahe — вирівняти контраст; blur — прибрати шум; sharpen — підняти різкість.
  3. Blackhat: залишає тільки дрібні темні деталі на світлішому фоні, тобто символи.
  4. Sobel по x: вертикальні краї. У номері їх багато (кожен символ), на кузові мало.
  5. Закриття витягнутим прямокутником: сусідні символи зливаються в одну пляму.
  6. Поріг (Otsu) і маска світлих ділянок: лишаються плями, що лежать на світлому.
  7. Фільтр плям: пропорції як у номера, розмір відносно машини, не у верхній чверті (не дах, не вікна).
  8. Найкраща пляма: найбільше вертикальних країв × найсвітліша.

Створює:
  dataset/plates_found/<фото>.jpg   фото машини з рамкою знайденого номера (якщо знайдено);
  dataset/plates_found/plates.json  рамки в координатах фото машини і в координатах кадру;
  dataset/plate_crops/<фото>.jpg    вирізаний номер з повної роздільності.
Якщо є dataset/labels_draft.json — порівнює з ручною розміткою і друкує, скільки номерів знайдено.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

W = 400  # до якої ширини зменшуємо фото машини


def preprocess(gray, mode):
    if mode == 'clahe':
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    if mode == 'blur':
        return cv2.GaussianBlur(gray, (3, 3), 0)
    if mode == 'sharpen':  # unsharp mask: оригінал + (оригінал − розмитий)
        return cv2.addWeighted(gray, 1.6, cv2.GaussianBlur(gray, (0, 0), 2), -0.6, 0)
    return gray


def find_plate(gray):
    """Повертає (x, y, w, h, score) найкращого кандидата в зменшеному фото або None."""
    H, Wd = gray.shape
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5)))
    edges = np.abs(cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3))
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    blob = cv2.GaussianBlur(edges, (5, 5), 0)
    blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5)))
    blob = cv2.threshold(blob, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    blob = cv2.dilate(cv2.erode(blob, None, iterations=2), None, iterations=2)
    light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    blob = cv2.bitwise_and(blob, light)
    blob = cv2.erode(cv2.dilate(blob, None, iterations=2), None, iterations=1)

    best = None
    for c in cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        (_, cy), (rw, rh), _ = cv2.minAreaRect(c)  # повернутий прямокутник — для пропорцій нахиленого номера
        long, short = max(rw, rh), min(rw, rh)
        if short < 6 or not 2 <= long / short <= 8:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 1.5 * h or not 0.08 * Wd <= w <= 0.6 * Wd or cy < 0.25 * H:  # номер лежить горизонтально
            continue
        score = edges[y:y + h, x:x + w].mean() / 255 * light[y:y + h, x:x + w].mean() / 255
        if best is None or score > best[4]:
            best = (x, y, w, h, score)
    return best


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter)


def evaluate(results, labels):
    """Порівнює знайдене з ручною розміткою. Рахуються тільки кадри, що є в розмітці."""
    reachable, found, wrong, neutral = set(), set(), [], 0
    for r in results:
        stem = Path(r['frame']).stem
        if stem not in labels:
            continue
        ox, oy = r['origin']
        cw, ch = r['size']
        for p in labels[stem]:  # номер «досяжний», якщо його центр лежить у фото якоїсь машини
            x, y, w, h = p['bbox']
            if p['text'] and ox <= x + w / 2 <= ox + cw and oy <= y + h / 2 <= oy + ch:
                reachable.add((stem, p['id']))
        if r['plate'] is None:
            continue
        best = max(labels[stem], key=lambda p: iou(p['bbox'], r['plate']['frame_bbox']), default=None)
        if best is None or iou(best['bbox'], r['plate']['frame_bbox']) < 0.3:
            wrong.append(r['file'])
        elif best['text']:
            found.add((stem, best['id']))
        else:
            neutral += 1  # знайшли номер, який людина теж не прочитала, — не плюс і не мінус
    return reachable, found, wrong, neutral


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ds')
    p.add_argument('--prep', default='sharpen', choices=['none', 'clahe', 'blur', 'sharpen'])
    a = p.parse_args()

    ds = Path(a.ds)
    crops = json.loads((ds / 'car_crops' / 'crops.json').read_text(encoding='utf-8'))['crops']
    out_found, out_cut = ds / 'plates_found', ds / 'plate_crops'
    for d in (out_found, out_cut):
        d.mkdir(exist_ok=True)
        for old in d.glob('*.jpg'):
            old.unlink()

    results, t_find = [], 0.0
    for c in crops:
        img = cv2.imread(str(ds / 'car_crops' / c['file']))
        k = img.shape[1] / W
        t0 = time.perf_counter()
        small = cv2.resize(img, (W, round(img.shape[0] / k)), interpolation=cv2.INTER_AREA)
        gray = preprocess(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), a.prep)
        best = find_plate(gray)
        t_find += time.perf_counter() - t0

        rec = dict(c, plate=None)
        if best:
            x, y, w, h = (round(v * k) for v in best[:4])  # назад у пікселі повного фото машини
            ox, oy = c['origin']
            rec['plate'] = {'bbox': [x, y, w, h], 'frame_bbox': [ox + x, oy + y, w, h], 'score': round(best[4], 3)}
            m = max(4, h // 3)
            cv2.imwrite(str(out_cut / c['file']), img[max(0, y - m):y + h + m, max(0, x - m):x + w + m])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), max(2, round(k * 2)))
        cv2.imwrite(str(out_found / c['file']), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        results.append(rec)

    (out_found / 'plates.json').write_text(json.dumps({'prep': a.prep, 'results': results}, ensure_ascii=False,
                                                      indent=1), encoding='utf-8')
    n_found = sum(r['plate'] is not None for r in results)
    print(f'prep={a.prep}: кандидат знайдено на {n_found} з {len(results)} фото машин, '
          f'{t_find / len(results) * 1000:.1f} мс на фото')

    lab_path = ds / 'labels_draft.json'
    if lab_path.exists():
        labels = json.loads(lab_path.read_text(encoding='utf-8'))
        reachable, found, wrong, neutral = evaluate(results, labels)
        print(f'  розмітка: знайдено {len(found & reachable)} з {len(reachable)} читабельних номерів | '
              f'хибних рамок: {len(wrong)} | на нечитабельних номерах: {neutral}')
        print(f'  пропущені: {sorted(reachable - found)}')
        print(f'  хибні: {wrong}')


if __name__ == '__main__':
    main()
