"""Перевірка рішень по номерах (після tools/process_clip.py).

  python tools/research/check_plates_clip.py dataset/clip/pass.json dataset/labels_draft.json --offset 290

1. Еталон: розмічені кадри dataset/images, що потрапляють у кліп (t0302.0 = 12.0 с кліпу при --offset 290).
   Для кожного розміченого номера шукаємо трек, якому він належить на тому кадрі, і порівнюємо рішення треку
   з розміченим текстом.
2. Відомі з перегляду номери (LS988 — Lexus, AA8921YB — помаранчева машина): чи знайдені взагалі.
4. (з --player) Чи скаче рамка номера в плеєрі: друга різниця положення центру рамки за три сусідні кадри,
   у пікселях кадру (рівномірний рух машини так не рахується, лишаються лише смикання).
3. Один і той самий номер на кількох треках:
   - треки одночасно в кадрі — номер сусідньої машини «протік» не в свій трек (помилка);
   - треки один за одним — це одна машина, яку трекер розбив на шматки (підказка для склеювання).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # основні модулі лежать у tools/
from plates_nn import edit_distance  # noqa: E402

KNOWN = {'LS988': 'Lexus', 'AA8921YB': 'помаранчева машина, 0:43'}


def iou(a, b):  # рамки [x1, y1, x2, y2]
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def jitter(player):
    raw = (Path(player) / 'data.js').read_text(encoding='utf-8')
    d = json.loads(raw[raw.index('{'):raw.rstrip().rindex(';')])
    W, H = d['source']['width'], d['source']['height']
    hist = defaultdict(dict)
    for t, rows in d['samples']:
        for r in rows:
            if len(r) >= 9:
                hist[r[0]][t] = ((r[5] + r[7] / 2) * W, (r[6] + r[8] / 2) * H, r[3] * W)
    moves, jumps = [], 0
    for seq in hist.values():
        ts = sorted(seq)
        for a, b, c in zip(ts, ts[1:], ts[2:]):
            if c - a > 0.1:
                continue
            v = ((seq[a][0] - 2 * seq[b][0] + seq[c][0]) ** 2 + (seq[a][1] - 2 * seq[b][1] + seq[c][1]) ** 2) ** 0.5
            moves.append(v)
            jumps += v > 0.15 * seq[b][2]
    moves.sort()
    print(f'\n== 4. Тремтіння рамки номера в плеєрі: медіана {moves[len(moves) // 2]:.1f} px, '
          f'95% {moves[int(len(moves) * 0.95)]:.1f} px, стрибків > 15% ширини машини {jumps} з {len(moves)}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cache')
    p.add_argument('labels')
    p.add_argument('--offset', type=float, required=True)
    p.add_argument('--player', help='тека плеєра з data.js — тоді ще й міряємо тремтіння рамки номера')
    a = p.parse_args()
    cache = json.loads(Path(a.cache).read_text(encoding='utf-8'))
    report = json.loads(Path(a.cache).with_name('decisions.json').read_text(encoding='utf-8'))
    labels = json.loads(Path(a.labels).read_text(encoding='utf-8'))
    by_track = {r['track']: r for r in report}
    times = cache['times']
    obs_at, car_at = defaultdict(list), defaultdict(list)
    for tid, lst in cache['obs'].items():
        for o in lst:
            obs_at[o['t']].append((tid, o))
    for tid, tr in cache['tracks'].items():
        for t, *box in tr['boxes']:
            car_at[t].append((tid, box))

    def who(tid):
        r = by_track.get(tid)
        if not r:
            return 'трек не в плеєрі (коротший за 0.5 с)', None
        d = r['decision']
        return f'№{r["n"]}', d

    print('== 1. Розмічені номери в кліпі')
    ok = total = 0
    for stem, plates in sorted(labels.items()):
        tc = float(stem[1:]) - a.offset
        if not 0 <= tc <= cache['source']['duration']:
            continue
        tn = min(times, key=lambda t: abs(t - tc))
        for q in plates:
            if not q['text']:
                continue
            total += 1
            x, y, w, h = q['bbox']
            lb = [x, y, x + w, y + h]
            # чия табличка на цьому кадрі (±0.1 с): спершу за рамкою номера, інакше — за рамкою машини
            near = [t for t in times if abs(t - tn) <= 0.1]
            cand = [(iou(lb, o['box']), tid) for t in near for tid, o in obs_at[t]]
            s, tid = max(cand, default=(0, None))
            how = 'рамка номера'
            if s < 0.3:
                cx, cy = x + w / 2, y + h / 2
                inside = [(-(b[2] - b[0]) * (b[3] - b[1]), tid) for tid, b in car_at[tn]
                          if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]]
                tid = min(inside)[1] if inside else None  # найменша машина, що містить номер
                how = 'лише рамка машини'
            if tid is None:
                print(f'  {q["text"]:<10} {stem}: машину не трекали як близьку')
                continue
            name, d = who(tid)
            got = d['text'] if d else '—'
            good = got == q['text']
            ok += good
            st = d['status'] if d else 'не прочитано'
            print(f'  {q["text"]:<10} {stem} -> {name:<8} рішення {got:<10} ({st}, {how}) {"OK" if good else "ПОМИЛКА"}')
    print(f'  збіглося {ok} з {total}')

    print('\n== 2. Відомі з перегляду номери')
    for text, what in KNOWN.items():
        hits = [(edit_distance(text, r['decision']['text']), r) for r in report if r['decision']]
        hits = sorted((d, r['n'], r['decision']['text'], r['decision']['status']) for d, r in hits if d <= 2)
        print(f'  {text:<10} ({what}): ' + (', '.join(f'№{n} {t} ({s}, різниця {d})' for d, n, t, s in hits) or 'не знайдено'))

    print('\n== 3. Однаковий номер на кількох треках')
    groups = defaultdict(list)
    for r in report:
        if r['decision'] and r['decision']['status'] != 'guess':
            groups[r['decision']['text']].append(r)
    any_dup = False
    for text, rs in groups.items():
        if len(rs) < 2:
            continue
        any_dup = True
        rs.sort(key=lambda r: r['first'])
        overlap = any(x['last'] > y['first'] for i, x in enumerate(rs) for y in rs[i + 1:])
        kind = 'ОДНОЧАСНО — номер протік у чужий трек' if overlap else 'один за одним — ймовірно, одна машина розбита на шматки'
        parts = ', '.join(f'№{r["n"]} {r["first"]:.1f}–{r["last"]:.1f} с ({r["decision"]["status"]})' for r in rs)
        print(f'  {text:<10} {parts}: {kind}')
    if not any_dup:
        print('  немає')

    if a.player:
        jitter(a.player)
    st = [r['decision']['status'] if r['decision'] else None for r in report]
    print(f'\n== Підсумок: машин {len(report)}, номер прочитано {st.count("read")}, непевно {st.count("uncertain")}, '
          f'лише розмиті кадри {st.count("guess")}, жодного читання {st.count(None)}')


if __name__ == '__main__':
    main()
