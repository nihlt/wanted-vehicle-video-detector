"""Еталон для трекінгу: яка рамка на якому кадрі належить якій справжній машині.

Крок 1 — заготовка для ручної перевірки:
  python tools/research/gt_tracks.py review dataset/clip/dets.json player/clip.mp4 dataset/clip/gt
    1. Беремо рамки з кешу детекцій (впевненість ≥ 0.25, як у плеєрі).
    2. Дуже обережний трекер (IoU ≥ 0.5, пропуск не більше 2 кадрів) ріже відео на «шматки» —
       такий скоріше розірве одну машину на кілька шматків, ніж склеїть дві різні машини.
    3. Лишаємо лише близькі рамки (≥ 25 000 px²) — саме ці машини нас цікавлять.
       Шматки з 1–3 рамок відкидаємо (це ~2% рамок, судити по них неможливо).
    4. Малюємо для людини (або для мене) аркуші:
         pairs_NN.jpg  — пари: зліва кінець шматка A (зелена рамка), справа початок шматка B (червона),
                         та сама ділянка кадру з сусідами (сірі рамки з підписами). Кандидати
                         «та сама машина, трекер її загубив» (SPLIT) або «дві рамки на одній машині» (DUP);
         strips_NN.jpg — довгі шматки, 6 кадрів уздовж шматка: перевірити, що там одна машина.
    fragments.json — шматки з усіма рамками.

Крок 2 — після перевірки записуємо рішення в labels.json:
  {"same": [["f003", "f007"], ...],          # шматки, які є однією машиною
   "cut": {"f019": [25.5]},                  # розрізати шматок у цей момент (якщо там дві машини)
   "drop": ["f040"]}                         # не машина / сміття
  python tools/research/gt_tracks.py build dataset/clip/gt
    -> tracks_gt.json: {cars: [{id, fragments, first, last}], boxes: [[t, x1, y1, x2, y2, car_id], ...]}
    -> cars.jpg: аркуш «одна плитка = одна справжня машина» для звірки з відео.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

MIN_AREA, SHOW_CONF = 25000, 0.25
MIN_BOXES = 4  # шматки з 1–3 рамок (~2% рамок) у еталон не беремо: надто короткі, щоб судити


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def centre(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def fragments_from(dets, thr=0.5, max_gap=2):
    """Обережний IoU-трекер по всіх рамках; повертає шматки з близькими рамками."""
    tracks, next_id, out = {}, 1, {}
    for k, (idx, t, boxes) in enumerate(dets['frames']):
        boxes = [b[:4] for b in boxes if b[4] >= SHOW_CONF]
        alive = {i: tr for i, tr in tracks.items() if k - tr['k'] <= max_gap}
        pairs = sorted(((iou(tr['box'], b), i, j) for i, tr in alive.items() for j, b in enumerate(boxes)),
                       reverse=True)
        ids, used = [None] * len(boxes), set()
        for s, i, j in pairs:
            if s < thr:
                break
            if i not in used and ids[j] is None:
                ids[j] = i
                used.add(i)
        for j, b in enumerate(boxes):
            if ids[j] is None:
                ids[j], next_id = next_id, next_id + 1
            tracks[ids[j]] = {'box': b, 'k': k}
            if area(b) >= MIN_AREA:
                out.setdefault(ids[j], []).append([idx, t] + b)
    frags = sorted(out.values(), key=lambda r: r[0][1])
    return {f'f{n:03d}': rows for n, rows in enumerate(frags, 1)}


def read_crops(video, wanted):
    """wanted: {frame_idx: [(key, box), ...]} -> {key: crop}. Читаємо відео один раз підряд."""
    cap, crops, idx, last = cv2.VideoCapture(video), {}, 0, max(wanted)
    while idx <= last:
        if idx in wanted:
            ok, frame = cap.read()
            if not ok:
                break
            for key, (x1, y1, x2, y2) in wanted[idx]:
                crops[key] = frame[max(0, y1):y2, max(0, x1):x2].copy()
        elif not cap.grab():
            break
        idx += 1
    return crops


def fit(img, h):
    return cv2.resize(img, (max(1, round(img.shape[1] * h / img.shape[0])), h), interpolation=cv2.INTER_AREA)


def text(img, s, org, scale=0.55):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def save_sheets(rows, out, prefix, per_sheet, width=1800):
    """rows: список картинок однакової ширини ≤ width; клеїмо вертикально по per_sheet штук."""
    for n in range(0, len(rows), per_sheet):
        chunk = rows[n:n + per_sheet]
        sheet = np.full((sum(r.shape[0] for r in chunk), width, 3), 40, np.uint8)
        y = 0
        for r in chunk:
            sheet[y:y + r.shape[0], :r.shape[1]] = r
            y += r.shape[0]
        cv2.imwrite(str(out / f'{prefix}_{n // per_sheet:02d}.jpg'), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


def candidate_pairs(frags, max_gap_s=1.5):
    """Кандидати «та сама машина»: B почався після кінця A (до 1.5 с) поруч, або A і B одночасно на одній машині."""
    pairs = []
    items = list(frags.items())
    for a, ra in items:
        for b, rb in items:
            if a == b:
                continue
            gap = rb[0][1] - ra[-1][1]
            if 0 < gap <= max_gap_s:
                (ax, ay), (bx, by) = centre(ra[-1][2:]), centre(rb[0][2:])
                w = max(ra[-1][4] - ra[-1][2], rb[0][4] - rb[0][2])
                if abs(ax - bx) <= 1.2 * w and abs(ay - by) <= 0.5 * w:
                    pairs.append(('split', a, b, gap))
            elif a < b and gap <= 0 and ra[0][1] <= rb[-1][1]:  # одночасно: чи накладаються рамки?
                fa = {r[0]: r[2:] for r in ra}
                best = max((iou(fa[r[0]], r[2:]) for r in rb if r[0] in fa), default=0)
                if best >= 0.3:
                    pairs.append(('dup', a, b, best))
    return pairs


def review(a):
    dets = json.loads(Path(a.dets).read_text(encoding='utf-8'))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob('*_??.jpg'):
        old.unlink()
    frags = fragments_from(dets)
    tiny = {f: r for f, r in frags.items() if len(r) < MIN_BOXES}
    frags = {f: r for f, r in frags.items() if len(r) >= MIN_BOXES}
    pairs = candidate_pairs(frags)
    longs = [f for f, r in frags.items() if r[-1][1] - r[0][1] >= 1.5]

    # для кожної пари: однакова ділянка кадру навколо обох рамок (з запасом), щоб було видно сусідів
    at = {}  # кадр -> [(шматок, рамка)] для підписів усіх машин на ділянці
    for f, rows in frags.items():
        for r in rows:
            at.setdefault(r[0], []).append((f, r[2:]))
    views = []  # (ключ, кадр, ділянка, [шматки, які підсвітити])
    for n, (kind, x, y, _) in enumerate(pairs):
        if kind == 'split':
            ra, rb = frags[x][-1], frags[y][0]
        else:  # dup: перший і середній спільні кадри
            common = sorted(set(r[0] for r in frags[x]) & set(r[0] for r in frags[y]))
            fa = {r[0]: r for r in frags[x]}
            ra, rb = fa[common[0]], fa[common[len(common) // 2]]
        u = [min(ra[2], rb[2]), min(ra[3], rb[3]), max(ra[4], rb[4]), max(ra[5], rb[5])]
        mx, my = (u[2] - u[0]) * 0.35, (u[3] - u[1]) * 0.35
        reg = [max(0, round(u[0] - mx)), max(0, round(u[1] - my)), round(u[2] + mx), round(u[3] + my)]
        views.append(((n, 'a'), ra[0], reg, (x, y)))
        views.append(((n, 'b'), rb[0], reg, (x, y)))
    wanted = {}
    for key, idx, reg, _ in views:
        wanted.setdefault(idx, []).append((key, reg))
    for f in longs:
        rows = frags[f]
        for i in np.linspace(0, len(rows) - 1, 6).round().astype(int):
            wanted.setdefault(rows[i][0], []).append(((f, int(i)), rows[i][2:]))
    crops = read_crops(a.video, wanted)

    def view(key, idx, reg, hl, h=260, max_w=880):
        img = crops[key]
        k = min(h / img.shape[0], max_w / img.shape[1])
        img = cv2.resize(img, (round(img.shape[1] * k), round(img.shape[0] * k)), interpolation=cv2.INTER_AREA)
        for f, b in at.get(idx, []):
            p1 = (round((b[0] - reg[0]) * k), round((b[1] - reg[1]) * k))
            p2 = (round((b[2] - reg[0]) * k), round((b[3] - reg[1]) * k))
            col = (0, 220, 0) if f == hl[0] else (0, 0, 255) if f == hl[1] else (200, 200, 200)
            cv2.rectangle(img, p1, p2, col, 3 if f in hl else 1)
            text(img, f, (max(2, p1[0] + 3), max(14, p1[1] + 16)), 0.5)
        return img

    H, pair_rows = 260, []
    for n, (kind, x, y, v) in enumerate(pairs):
        va, vb = view(*views[2 * n]), view(*views[2 * n + 1])
        row = np.full((H + 30, 1800, 3), 40, np.uint8)
        row[30:30 + va.shape[0], :va.shape[1]] = va
        row[30:30 + vb.shape[0], 900:900 + vb.shape[1]] = vb
        ta, tb = frags[x], frags[y]
        if kind == 'split':
            tx = (f'p{n + 1:02d} SPLIT?  {x} ({ta[0][1]:.2f}-{ta[-1][1]:.2f}) kinec t={ta[-1][1]:.2f}   ->   '
                  f'{y} ({tb[0][1]:.2f}-{tb[-1][1]:.2f}) pochatok t={tb[0][1]:.2f}   gap {v:.2f}s')
        else:
            tx = (f'p{n + 1:02d} DUP?  {x} ({ta[0][1]:.2f}-{ta[-1][1]:.2f}) i {y} ({tb[0][1]:.2f}-{tb[-1][1]:.2f}) '
                  f'odnochasno, iou {v:.2f}')
        text(row, tx, (6, 21), 0.62)
        pair_rows.append(row)
    save_sheets(pair_rows, out, 'pairs', 7)

    strip_rows = []
    for f in longs:
        rows = frags[f]
        r = np.full((156, 1800, 3), 40, np.uint8)
        x = 0
        for i in np.linspace(0, len(rows) - 1, 6).round().astype(int):
            c = fit(crops[(f, int(i))], 130)[:, :295]
            r[26:156, x:x + c.shape[1]] = c
            text(r, f'{rows[i][1]:.2f}', (x + 4, 150), 0.45)
            x += 300
        text(r, f'{f}  {rows[0][1]:.2f}-{rows[-1][1]:.2f} s', (4, 18))
        strip_rows.append(r)
    save_sheets(strip_rows, out, 'strips', 12)

    (out / 'fragments.json').write_text(json.dumps(
        {'source': a.dets, 'video': a.video, 'fragments': frags, 'tiny': tiny,
         'pairs': [[k, x, y, round(v, 3)] for k, x, y, v in pairs]}, separators=(',', ':')), encoding='utf-8')
    print(f'шматків {len(frags)} (+{len(tiny)} коротких відкинуто), кандидатів-пар {len(pairs)} '
          f'(split {sum(p[0] == "split" for p in pairs)}, dup {sum(p[0] == "dup" for p in pairs)}), '
          f'довгих шматків {len(longs)}')


def build(a):
    out = Path(a.out)
    fr = json.loads((out / 'fragments.json').read_text(encoding='utf-8'))
    lab = json.loads((out / 'labels.json').read_text(encoding='utf-8'))
    frags = fr['fragments']
    # розрізаємо шматки, де виявились дві машини
    for f, cuts in lab.get('cut', {}).items():
        rows, parts = frags.pop(f), []
        for c in sorted(cuts) + [1e9]:
            parts.append([r for r in rows if r[1] < c])
            rows = [r for r in rows if r[1] >= c]
        for n, p in enumerate(parts):
            if p:
                frags[f'{f}{"abcdefgh"[n]}'] = p
    for f in lab.get('drop', []):
        frags.pop(f, None)
    # об'єднуємо «та сама машина» (union-find)
    parent = {f: f for f in frags}
    def root(f):
        while parent[f] != f:
            parent[f] = parent[parent[f]]
            f = parent[f]
        return f
    for x, y in lab.get('same', []):
        if x in parent and y in parent:
            parent[root(y)] = root(x)
    groups = {}
    for f in frags:
        groups.setdefault(root(f), []).append(f)
    groups = sorted(groups.values(), key=lambda g: min(frags[f][0][1] for f in g))

    cars, boxes = [], []
    for n, g in enumerate(groups, 1):
        rows = sorted((r for f in g for r in frags[f]), key=lambda r: r[1])
        cid = f'c{n:03d}'
        cars.append({'id': cid, 'fragments': sorted(g), 'first': rows[0][1], 'last': rows[-1][1], 'n_boxes': len(rows)})
        boxes += [[r[1]] + r[2:] + [cid] for r in rows]
    boxes.sort(key=lambda b: b[0])
    (out / 'tracks_gt.json').write_text(json.dumps({'note': 'еталон: рамка (t, x1, y1, x2, y2) -> справжня машина',
                                                    'cars': cars, 'boxes': boxes}, ensure_ascii=False,
                                                   separators=(',', ':')), encoding='utf-8')

    # аркуш: найбільша рамка кожної машини
    best = {}
    for b in boxes:
        if b[5] not in best or area(b[1:5]) > area(best[b[5]][1:5]):
            best[b[5]] = b
    idx_of = {r[1]: r[0] for f in frags.values() for r in f}
    wanted = {}
    for cid, b in best.items():
        wanted.setdefault(idx_of[b[0]], []).append((cid, b[1:5]))
    crops = read_crops(fr['video'], wanted)
    tiles = []
    for c in cars:
        img = cv2.resize(crops[c['id']], (280, 190), interpolation=cv2.INTER_AREA)
        tile = np.full((216, 280, 3), 40, np.uint8)
        tile[26:] = img
        text(tile, f"{c['id']} {c['first']:.1f}-{c['last']:.1f}s", (4, 18))
        tiles.append(tile)
    cols = 6
    rows_img = [np.hstack(tiles[i:i + cols] + [np.full((216, 280, 3), 40, np.uint8)] * (cols - len(tiles[i:i + cols])))
                for i in range(0, len(tiles), cols)]
    save_sheets(rows_img, out, 'cars', 6, width=280 * cols)
    print(f'справжніх машин {len(cars)} (з {len(fr["fragments"])} шматків), рамок {len(boxes)}')


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest='cmd', required=True)
    r = sp.add_parser('review')
    r.add_argument('dets')
    r.add_argument('video')
    r.add_argument('out')
    b = sp.add_parser('build')
    b.add_argument('out')
    a = p.parse_args()
    review(a) if a.cmd == 'review' else build(a)


if __name__ == '__main__':
    main()
