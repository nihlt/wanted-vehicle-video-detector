"""Чому номери читаються повільно поруч із YOLO: міряємо швидкість детектора номерів і OCR у різних умовах.

  python tools/research/bench_plates.py noyolo   # лише onnxruntime, без torch
  python tools/research/bench_plates.py yolo     # як у process_clip: між вимірами YOLO працює на відеокарті
  python tools/research/bench_plates.py torch1   # те саме, але torch обмежено одним потоком процесора
  ... fast                              # другим аргументом: попросити Windows не пригальмовувати процес

Змінні середовища (напр. OMP_WAIT_POLICY=PASSIVE) задаються ззовні, до запуску.
"""
import glob
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))  # основні модулі лежать у tools/
mode = sys.argv[1]
if mode != 'noyolo':
    import torch
    from ultralytics import YOLO
    if mode == 'torch1':
        torch.set_num_threads(1)
    m = YOLO(str(ROOT / 'models' / 'yolo11n.pt'))
    fr = cv2.VideoCapture(str(ROOT / 'player' / 'clip.mp4')).read()[1]
    m.predict(fr, device=0, verbose=False)

from plates_nn import PlateReader, full_speed  # noqa: E402

if 'fast' in sys.argv[2:]:
    print('full_speed:', full_speed(), flush=True)

r = PlateReader()
crops = [cv2.imread(f) for f in sorted(glob.glob(str(ROOT / 'dataset' / 'car_crops' / '*.jpg')))[:20]]
plates = [cv2.resize(c, (128, 64)) for c in crops[:3]]
for rep in range(5):
    if mode != 'noyolo':
        m.predict([fr] * 8, device=0, verbose=False)  # робота YOLO між вимірами, як у справжньому циклі
    t = time.perf_counter()
    for c in crops:
        r.detect(c)
    td = (time.perf_counter() - t) / len(crops)
    t = time.perf_counter()
    for _ in range(10):
        r.read(plates)
    to = (time.perf_counter() - t) / 10
    print(f'{mode:<8} прохід {rep}: детектор {td * 1000:.1f} мс/машину, OCR {to * 1000:.1f} мс на 3 номери', flush=True)
