"""Порівнює трекери з еталоном на однакових, заздалегідь збережених детекціях.

  python tools/research/compare_trackers.py dataset/clip/dets.json player/clip.mp4 dataset/clip/gt/tracks_gt.json dataset/clip/compare

Трекери:
  iou            — наш простий трекер із плеєра (IoU ≥ 0.3, пам'ять 10 кадрів, рамки ≥ 0.25). Це точка відліку.
  bytetrack      — ByteTrack: передбачає рух машини (фільтр Калмана) і другим проходом підхоплює слабкі
                   рамки (0.1–0.25), щоб не губити машину, коли детектор на мить сумнівається.
  botsort        — BoT-SORT: те саме + компенсація руху камери (ми їдемо, і весь кадр зсувається).
  *_b60          — те саме, але машину пам'ятаємо 2 с замість 1 с (довше перекриття зустрічною машиною).

Що міряємо (як у плеєрі: картка = трек, що був близько ≥ 0.5 с, близько = рамка ≥ 25 000 px²):
  склейки  — картка, де змішані дві справжні машини (найгірше: змішає номери двох машин);
  розпад   — зайві картки, коли одна машина дала кілька карток;
  сміття   — картка, якій не відповідає жодна машина з еталону;
  пропуски — машина з еталону (близько ≥ 0.5 с), для якої немає картки.

Створює в <out>/: <трекер>.json (усі рамки треків), report.txt, progress.log.
"""
import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # основні модулі лежать у tools/
from process_clip import BASE, BOT, Dets, IoUTracker, iou

MIN_AREA, MIN_SEEN, SHOW_CONF = 25000, 0.5, 0.25


def make_trackers():
    return {
        'iou': IoUTracker(),
        'bytetrack': BYTETracker(SimpleNamespace(**BASE)),
        'bytetrack_b60': BYTETracker(SimpleNamespace(**{**BASE, 'track_buffer': 60})),
        'botsort': BOTSORT(SimpleNamespace(**BASE, **BOT)),
        'botsort_b60': BOTSORT(SimpleNamespace(**{**BASE, 'track_buffer': 60}, **BOT)),
    }


def run(dets, video, log):
    trackers = make_trackers()
    rows = {name: [] for name in trackers}  # [t, track_id, x1, y1, x2, y2, conf]
    frames = dets['frames']
    wanted = {f[0]: k for k, f in enumerate(frames)}
    cap, idx, t0 = cv2.VideoCapture(video), 0, time.perf_counter()
    while idx <= frames[-1][0]:
        if idx not in wanted:
            cap.grab()
            idx += 1
            continue
        ok, img = cap.read()
        if not ok:
            break
        k = wanted[idx]
        _, t, boxes = frames[k]
        a = np.array(boxes, dtype=np.float32).reshape(-1, 6)
        for name, tr in trackers.items():
            if name == 'iou':
                strong = a[a[:, 4] >= SHOW_CONF]
                ids = tr.update([tuple(b[:4]) for b in strong], k)
                rows[name] += [[t, i] + b[:5].tolist() for i, b in zip(ids, strong)]
            else:
                out = tr.update(Dets(a), img)
                # беремо рамку детектора (а не згладжену трекером), щоб чесно порівнювати з еталоном
                rows[name] += [[t, int(r[4])] + a[int(r[7]), :5].tolist() for r in out]
        idx += 1
        if (k + 1) % 60 == 0 or k + 1 == len(frames):
            el = time.perf_counter() - t0
            log.write(f'{k + 1}/{len(frames)} кадрів | {(k + 1) / el:.1f} кадр/с | минуло {el:.0f} с | '
                      f'залишилось ~{(len(frames) - k - 1) / ((k + 1) / el):.0f} с\n')
    return rows


def evaluate(rows, gt):
    """Повертає підсумок і подробиці для одного трекера."""
    gt_at = defaultdict(list)
    for b in gt['boxes']:
        gt_at[round(b[0], 3)].append((b[1:5], b[5]))
    long_cars = {c['id'] for c in gt['cars'] if c['last'] - c['first'] >= MIN_SEEN}

    tracks = defaultdict(list)
    for t, tid, x1, y1, x2, y2, conf in rows:
        if (x2 - x1) * (y2 - y1) >= MIN_AREA:
            tracks[tid].append((t, [x1, y1, x2, y2], conf))
    cards = {tid: r for tid, r in tracks.items() if r[-1][0] - r[0][0] >= MIN_SEEN}

    owner, mixed, junk = {}, [], []
    for tid, r in cards.items():
        votes, strong = Counter(), 0
        for t, box, conf in r:
            if conf < SHOW_CONF:
                continue  # слабкі рамки в еталоні не розмічені — не рахуємо ні «за», ні «проти»
            strong += 1
            s, cid = max(((iou(box, gb), c) for gb, c in gt_at.get(round(t, 3), [])), default=(0, None))
            if s >= 0.5:
                votes[cid] += 1
        matched = sum(votes.values())
        if not strong or matched / strong < 0.5:
            junk.append(tid)
            continue
        (top, n), *rest = votes.most_common()
        owner[tid] = top
        others = matched - n
        if others >= 3 and others / matched >= 0.1:
            mixed.append((tid, dict(votes.most_common(3))))

    per_car = defaultdict(list)
    for tid, cid in owner.items():
        per_car[cid].append(tid)
    splits = {cid: len(v) for cid, v in per_car.items() if len(v) > 1}
    missed = sorted(long_cars - set(per_car))
    return {
        'cards': len(cards), 'cars': len(per_car), 'extra': sum(n - 1 for n in splits.values()),
        'mixed': len(mixed), 'junk': len(junk), 'missed': len(missed),
        'splits': splits, 'mixed_detail': mixed, 'missed_detail': missed, 'long_cars': len(long_cars),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dets')
    p.add_argument('video')
    p.add_argument('gt')
    p.add_argument('out')
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / 'progress.log', 'w', encoding='utf-8', buffering=1)
    dets = json.loads(Path(a.dets).read_text(encoding='utf-8'))
    gt = json.loads(Path(a.gt).read_text(encoding='utf-8'))

    rows = run(dets, a.video, log)
    lines = [f'Еталон: {len(gt["cars"])} машин, з них близько ≥ {MIN_SEEN} с: '
             f'{sum(c["last"] - c["first"] >= MIN_SEEN for c in gt["cars"])}', '',
             f'{"трекер":<15}{"карток":>7}{"машин":>7}{"розпад":>8}{"склейки":>9}{"сміття":>8}{"пропуски":>10}']
    details = []
    for name, r in rows.items():
        (out / f'{name}.json').write_text(json.dumps({'tracker': name, 'rows': r}, separators=(',', ':')),
                                          encoding='utf-8')
        e = evaluate(r, gt)
        lines.append(f'{name:<15}{e["cards"]:>7}{e["cars"]:>7}{e["extra"]:>8}{e["mixed"]:>9}{e["junk"]:>8}'
                     f'{e["missed"]:>10}')
        details += [f'', f'== {name}', f'  розпад (машина: карток): {e["splits"]}',
                    f'  склейки (картка: машини): {e["mixed_detail"]}', f'  пропущені машини: {e["missed_detail"]}']
    report = '\n'.join(lines + [''] + ['розпад = зайві картки через те, що одна машина дала кілька карток'] + details)
    (out / 'report.txt').write_text(report + '\n', encoding='utf-8')
    log.write('ГОТОВО\n')
    print(report)


if __name__ == '__main__':
    main()
