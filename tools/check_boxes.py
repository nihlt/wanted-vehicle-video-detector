"""Перевірка рамок у плеєрі: чи плавні і чи не відійшли від того, що бачив детектор.

  python tools/check_boxes.py dataset/clip/pass.json player/data.js [dataset/clip/data_before_step11.js]

Для кожного data.js (новий і, якщо дано, старий для порівняння):
1. Тремтіння: друга різниця центру і розміру рамки за три сусідні пораховані кадри, px кадру
   (рівномірний рух і рівномірне прискорення дають 0, лишаються смикання).
2. Вірність: наскільки показана рамка відходить від сирої рамки детектора на тому самому кадрі
   (найбільший зсув краю, px, і IoU). Для номера сира рамка — знахідка свого номера (plate_track без пропусків).
3. Скільки кадрів машини з рамкою номера, і скільки разів центр номера поза рамкою машини.
4. Дірки: машина зникає на 1–3 пораховані кадри посеред проїзду (видно як блимання).
"""
import json
import sys
from pathlib import Path

import numpy as np

from find_plates import iou
from track_plates import decide, plate_track


def load_js(path):
    raw = Path(path).read_text(encoding='utf-8')
    return json.loads(raw[raw.index('{'):raw.rstrip().rindex(';')])


def series(d):
    """{n: {t: car}}, {n: {t: plate}} у пікселях кадру з data.js."""
    W, H = d['source']['width'], d['source']['height']
    px = lambda x, y, w, h: [x * W, y * H, (x + w) * W, (y + h) * H]
    cars, plates = {}, {}
    for t, rows in d['samples']:
        for r in rows:
            cars.setdefault(r[0], {})[t] = px(*r[1:5])
            if len(r) >= 9:
                plates.setdefault(r[0], {})[t] = px(*r[5:9])
    return cars, plates


def jitter(seqs, idx):
    c, s = [], []
    for seq in seqs.values():
        ts = sorted(seq)
        for a, b, e in zip(ts, ts[1:], ts[2:]):
            if idx[e] - idx[a] != 2:
                continue
            f = lambda x: np.array([(x[0] + x[2]) / 2, (x[1] + x[3]) / 2, x[2] - x[0], x[3] - x[1]])
            D = f(seq[a]) - 2 * f(seq[b]) + f(seq[e])
            c.append(np.hypot(D[0], D[1]))
            s.append(max(abs(D[2]), abs(D[3])))
    q = lambda v: f'медіана {np.median(v):4.1f}, 90% {np.percentile(v, 90):5.1f}, 99% {np.percentile(v, 99):5.1f}'
    return f'центр {q(c)} | розмір {q(s)}'


def fidelity(raw, shown):
    dev, io = [], []
    for n, seq in raw.items():
        for t, b in seq.items():
            if t in shown.get(n, {}):
                s = shown[n][t]
                dev.append(max(abs(p - q) for p, q in zip(b, s)))
                io.append(iou(b, s))
    return (f'зсув краю медіана {np.median(dev):4.1f} px, 95% {np.percentile(dev, 95):5.1f}, max {max(dev):4.0f} | '
            f'IoU медіана {np.median(io):.3f}, 1% {np.percentile(io, 1):.3f}, min {min(io):.3f}')


def holes(cars, idx):
    k = 0
    for seq in cars.values():
        ii = sorted(idx[t] for t in seq)
        k += sum(1 <= b - a - 1 <= 3 for a, b in zip(ii, ii[1:]))
    return k


def main():
    cache = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    idx = {t: i for i, t in enumerate(cache['times'])}
    report = json.loads(Path(sys.argv[1]).with_name('decisions.json').read_text(encoding='utf-8'))
    raw_cars, raw_plates = {}, {}
    for r in report:
        tid = r['track']
        cars = {b[0]: b[1:] for b in cache['tracks'][tid]['boxes']}
        obs = cache['obs'].get(tid, [])
        g = decide(obs)
        raw_cars[r['n']] = cars
        raw_plates[r['n']] = plate_track(obs, cars, g if g and g['status'] != 'guess' else None, max_gap=0)
    print(f'сирі рамки детектора:\n  машина: {jitter(raw_cars, idx)}\n  номер:  {jitter(raw_plates, idx)}')
    for path in sys.argv[2:]:
        cars, plates = series(load_js(path))
        out = sum(1 for n, seq in plates.items() for t, p in seq.items()
                  if not (cars[n][t][0] <= (p[0] + p[2]) / 2 <= cars[n][t][2]
                          and cars[n][t][1] <= (p[1] + p[3]) / 2 <= cars[n][t][3]))
        n_car = sum(map(len, cars.values()))
        n_pl = sum(map(len, plates.values()))
        print(f'\n{path}')
        print(f'  тремтіння машина: {jitter(cars, idx)}')
        print(f'  тремтіння номер:  {jitter(plates, idx)}')
        print(f'  вірність машина:  {fidelity(raw_cars, cars)}')
        print(f'  вірність номер:   {fidelity(raw_plates, plates)}')
        print(f'  кадрів машин {n_car}, з рамкою номера {n_pl} ({n_pl / n_car:.0%}); '
              f'центр номера поза машиною: {out}; дірок 1–3 кадри посеред проїзду: {holes(cars, idx)}')


if __name__ == '__main__':
    main()
