"""Номер машини з усього треку: спостереження з кожного кадру -> одне рішення.

1. observe()  — на кадрі для кожної близької машини: детектор знаходить до 3 кандидатів номера,
                OCR читає кожного. Табличка, що потрапила у вирізки двох сусідніх машин, дістається тій,
                де вона «на своєму місці» (ближче до низу посередині машини).
2. decide()   — голосування по всіх кадрах машини:
                - кожне читання важить стільки, скільки його найслабший символ (чіткий кадр ~1.0, розмитий ~0.1);
                - перемагає текст із найбільшою підтримкою (схожі читання теж підтримують, частково);
                - потім кожен символ уточнюємо голосуванням серед читань тієї самої довжини;
                - впевненість = частка ваги читань, що збіглися з рішенням точно.
3. plate_track() — де на кожному кадрі рамка СВОГО номера: лише кадри, де прочитано цей номер
                (± 2 символи), плюс нечитабельні кадри на тому самому місці машини. Положення таблички
                на кадрі зі своїм номером — точна знахідка детектора; пропуски до 0.5 с заповнюємо
                прямою інтерполяцією. Тому рамка не скаче на фари чи сусідні машини.
"""
from collections import Counter

from find_plates import iou
from plates_nn import edit_distance

MIN_W = 0.2       # читання слабші за це не голосують (розмитий номер = сміттєвий текст)
STRONG = 0.8      # читання з усіма символами ≥ 0.8 майже завжди правильні (крок 8 у docs/PIPELINE.md)
MEDIUM = 0.5      # без жодного читання ≥ 0.5 текст на кліпі був сміттям (1111, 44, 111) — крок 9


def _misplace(b, car):
    """Наскільки табличка далека від звичного місця номера на машині: низ посередині."""
    x1, y1, x2, y2 = car
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return abs(cx - (x1 + x2) / 2) / (x2 - x1) + abs(cy - (y1 + 0.7 * (y2 - y1))) / (y2 - y1)


def observe(frame, cars, reader, top_k=3, margin=0.03):
    """cars: [(tid, (x1, y1, x2, y2)), ...] -> {tid: [obs]}, obs = {box, det, text, probs, w}."""
    H, W = frame.shape[:2]
    cands = []
    for tid, (x1, y1, x2, y2) in cars:
        mx, my = round((x2 - x1) * margin), round((y2 - y1) * margin)
        ox, oy = max(0, x1 - mx), max(0, y1 - my)
        crop = frame[oy:min(H, y2 + my), ox:min(W, x2 + mx)]
        for px1, py1, px2, py2, c in reader.detect(crop)[:top_k]:
            b = [ox + max(0, px1), oy + max(0, py1), ox + min(px2, crop.shape[1]), oy + min(py2, crop.shape[0])]
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if b[2] - b[0] >= 4 and b[3] - b[1] >= 2 and x1 <= cx <= x2 and y1 <= cy <= y2:
                cands.append((tid, b, c, (x1, y1, x2, y2)))
    keep = [a for a in cands
            if all(_misplace(a[1], a[3]) <= _misplace(b[1], b[3])
                   for b in cands if b[0] != a[0] and iou(a[1], b[1]) >= 0.5)]
    reads = reader.read([frame[b[1]:b[3], b[0]:b[2]] for _, b, _, _ in keep])
    out = {}
    for (tid, b, c, _), (text, probs) in zip(keep, reads):
        out.setdefault(tid, []).append({'box': b, 'det': round(c, 3), 'text': text,
                                        'probs': [round(p, 3) for p in probs],
                                        'w': round(min(probs), 3) if probs else 0.0})
    return out


def decide(obs):
    """obs: усі спостереження машини (з полем t). -> рішення або None, якщо номер не прочитано жодного разу.

    status: 'read' — є чітке читання (усі символи ≥ 0.8), що збіглося з рішенням;
            'uncertain' — чіткого немає, але найкраще збіжне читання ≥ 0.5: показуємо з «?»;
            'guess' — усі кадри розмиті (найкраще < 0.5): текст майже завжди сміття (1111, 44…), не показуємо.
    """
    # '_' — порожня позиція OCR: «_C», «__4» — це артефакти розмитих кадрів, а не номер
    reads = [o for o in obs if o['text'] and '_' not in o['text'] and o['w'] >= MIN_W]
    if not reads:
        return None

    def support(c):
        return sum(o['w'] * max(0.0, 1 - edit_distance(o['text'], c) / max(len(c), len(o['text'])))
                   for o in reads)

    best = max({o['text'] for o in reads}, key=support)
    same = [o for o in reads if len(o['text']) == len(best) and edit_distance(o['text'], best) <= 2]
    chars = []
    for i in range(len(best)):
        votes = Counter()
        for o in same:
            votes[o['text'][i]] += o['probs'][i]
        chars.append(votes.most_common(1)[0][0])
    text = ''.join(chars)
    agree = [o for o in reads if o['text'] == text]
    if not agree:  # голосування склало рядок, якого не прочитали жодного разу — не вигадуємо, беремо реальне читання
        text = best
        agree = [o for o in reads if o['text'] == text]
    total = sum(o['w'] for o in reads)
    n_strong = sum(o['w'] >= STRONG for o in agree)
    conf = sum(o['w'] for o in agree) / total
    alts = [[t, n] for t, n in Counter(o['text'] for o in reads if o['text'] != text).most_common(3)]
    max_w = max(o['w'] for o in agree)
    if n_strong >= 1 or (len(agree) >= 3 and conf >= 0.5 and max_w >= MEDIUM):
        status = 'read'
    else:
        status = 'uncertain' if max_w >= MEDIUM else 'guess'
    return {'text': text, 'conf': round(conf, 3), 'status': status, 'max_w': round(max_w, 3),
            'n_agree': len(agree), 'n_reads': len(reads), 'n_strong': n_strong, 'alts': alts}


def _rel(o, car):
    x1, y1, x2, y2 = car
    return (((o['box'][0] + o['box'][2]) / 2 - x1) / (x2 - x1), ((o['box'][1] + o['box'][3]) / 2 - y1) / (y2 - y1))


def _near_reads(o, reads):
    """Чи нечитабельна табличка o там, де свій номер прочитано на найближчих кадрах (у пікселях кадру).

    Місце відносно машини (near_median) обманює, коли машину обрізає край кадру: рамка машини стоїть,
    а номер їде. Тому ще порівнюємо з прочитаним номером: між читаннями з обох боків (≤ 0.5 с) — пряма
    між ними, з одного боку — лише якщо читання не далі 0.1 с. Центр має бути не далі ширини таблички.
    """
    before = [p for p in reads if o['t'] - 0.5 <= p['t'] <= o['t']]
    after = [p for p in reads if o['t'] <= p['t'] <= o['t'] + 0.5]
    a, b = (before[-1] if before else None), (after[0] if after else None)
    if a and b and b['t'] > a['t']:
        k = (o['t'] - a['t']) / (b['t'] - a['t'])
        pred = [p + (q - p) * k for p, q in zip(a['box'], b['box'])]
    else:
        near = [p for p in (a, b) if p and abs(p['t'] - o['t']) <= 0.1]
        if not near:
            return True  # поруч немає читань — перевіряти нема з чим, лишається перевірка місця на машині
        pred = near[0]['box']
    dx = (o['box'][0] + o['box'][2] - pred[0] - pred[2]) / 2
    dy = (o['box'][1] + o['box'][3] - pred[1] - pred[3]) / 2
    return (dx * dx + dy * dy) ** 0.5 <= max(pred[2] - pred[0], o['box'][2] - o['box'][0])


def plate_track(obs, cars, decision, max_gap=0.5):
    """obs: спостереження з полем t; cars: {t: рамка машини}. -> {t: рамка свого номера}.

    - На кадрі, де свій номер знайдено, показуємо саме цю знахідку детектора: вона точна і майже не тремтить
      (медіана ~1 px між кадрами), а стрибки на чужі таблички вже відсіяно вибором «свого» номера.
    - Пропуски до max_gap с заповнюємо прямою інтерполяцією між знахідками з обох боків. Перевірено:
      це точніше (IoU 0.92), ніж переносити рамку разом з рамкою машини (0.85), бо рамка машини сама тремтить,
      а згладжування відносно машини зсувало номер, коли машину обрізав край кадру (крок 10 у docs/PIPELINE.md).
    """
    if decision:
        own = [o for o in obs if o['text'] and edit_distance(o['text'], decision['text']) <= 2]
    else:
        own = [o for o in obs if o['det'] >= 0.5]
    if not own:
        return {}
    rel = {id(o): _rel(o, cars[o['t']]) for o in obs}

    def near_median(o, pool, tol):
        pool = [p for p in pool if p is not o]
        if not pool:
            return True
        mx = sorted(rel[id(p)][0] for p in pool)[len(pool) // 2]
        my = sorted(rel[id(p)][1] for p in pool)[len(pool) // 2]
        return abs(rel[id(o)][0] - mx) <= tol and abs(rel[id(o)][1] - my) <= tol

    if decision:  # нечитабельні кадри беремо, якщо табличка там само, де свій номер на сусідніх кадрах
        own_ids = {id(o) for o in own}
        reads = sorted(own, key=lambda p: p['t'])
        for o in obs:
            if id(o) in own_ids or (o['text'] and o['w'] >= MIN_W):
                continue
            pool = [p for p in own if abs(p['t'] - o['t']) <= 0.5]
            if pool and near_median(o, pool, 0.1) and _near_reads(o, reads):
                own.append(o)
    else:
        own = [o for o in own if near_median(o, own, 0.12)]

    by_t = {}
    for o in own:  # якщо на кадрі кілька кандидатів — той, що ближче до звичного місця
        cur = by_t.get(o['t'])
        if cur is None or _misplace(o['box'], cars[o['t']]) < _misplace(cur['box'], cars[cur['t']]):
            by_t[o['t']] = o
    ts = sorted(by_t)
    out = {t: list(by_t[t]['box']) for t in ts}
    car_ts = sorted(cars)
    for a, b in zip(ts, ts[1:]):
        if b - a > max_gap:
            continue
        for t in car_ts:
            if a < t < b:
                k = (t - a) / (b - a)
                out[t] = [round(p + (q - p) * k) for p, q in zip(out[a], out[b])]
    return out
