"""Один раз проганяє YOLO по відео і зберігає всі рамки машин у JSON (кеш детекцій).

  python tools/research/detect_clip.py player/clip.mp4 dataset/clip/dets.json

Навіщо: різні трекери (і еталонну розмітку) порівнюємо на ОДНИХ І ТИХ САМИХ детекціях,
тоді різниця в результаті залежить лише від трекера. І не треба щоразу ганяти нейромережу.

Поріг впевненості низький (0.1, а не 0.25): ByteTrack/BoT-SORT уміють використовувати слабкі
рамки, щоб не губити машину, коли детектор на мить у ній сумнівається.

Формат: {video, fps, width, height, step, conf, frames: [[idx, t, [[x1,y1,x2,y2,conf,cls], ...]], ...]}
Прогрес пишеться в <out>.log.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

VEHICLES = {2: 'car', 3: 'moto', 5: 'bus', 7: 'truck'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('video')
    p.add_argument('out')
    p.add_argument('--step', type=int, default=2)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--conf', type=float, default=0.1)
    p.add_argument('--model', default='models/yolo11n.pt')
    a = p.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = open(out.with_suffix('.log'), 'w', encoding='utf-8', buffering=1)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = (n_frames + a.step - 1) // a.step
    model = YOLO(a.model)
    opts = dict(imgsz=640, conf=a.conf, classes=list(VEHICLES), device=0, quantize=16, agnostic_nms=True,
                verbose=False)
    frames, t0 = [], time.perf_counter()

    def run(batch):
        for (idx, _), r in zip(batch, model.predict([f for _, f in batch], **opts)):
            boxes = [[round(v) for v in b] + [round(c, 3), int(k)]
                     for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
            frames.append([idx, round(idx / fps, 3), boxes])
        el = time.perf_counter() - t0
        log.write(f'{len(frames)}/{total} кадрів | {len(frames) / el:.1f} кадр/с | минуло {el:.0f} с | '
                  f'залишилось ~{(total - len(frames)) / max(len(frames) / el, 1e-9):.0f} с\n')

    batch, idx = [], 0
    while True:
        if idx % a.step == 0:
            ok, frame = cap.read()
            if not ok:
                break
            batch.append((idx, frame))
            if len(batch) == a.batch:
                run(batch)
                batch = []
        elif not cap.grab():
            break
        idx += 1
    if batch:
        run(batch)

    data = {'video': a.video, 'fps': round(fps, 3), 'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 'step': a.step, 'conf': a.conf, 'model': a.model,
            'frames': frames}
    out.write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')
    n = sum(len(f[2]) for f in frames)
    msg = f'ГОТОВО за {time.perf_counter() - t0:.0f} с: {len(frames)} кадрів, {n} рамок'
    log.write(msg + '\n')
    print(msg)


if __name__ == '__main__':
    main()
