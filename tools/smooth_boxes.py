"""Плавні рамки для показу: прибирає «дихання» детектора, не відстаючи від машини.

Для кожного кадру беремо сусідні кадри цієї ж машини (± half секунд, з обох боків) і проводимо через
кожен край рамки (x1, y1, x2, y2 окремо, у пікселях кадру) плавну криву — параболу. Парабола враховує
швидкість і прискорення: напрямок руху береться з кадрів до і після, тому рамка не відстає і не
забігає вперед. Ближчі кадри важать більше (вага tricube). Якщо сусіди лише з одного боку (перший чи
останній кадр машини) — пряма з тим самим напрямком руху.
- Викиди: край стрибнув на кадр-два і повернувся. Такий кадр не голосує, а його рамку беремо з кривої,
  інакше одна погана рамка розмазалась би горбом по всьому вікну. Прибираємо по одному, від найгіршого,
  не більше двох кадрів поспіль, і лише якщо значення «повернулось»: рух до і після лежить на одній лінії.
  Якщо ні — це справжня зміна рамки (машину частково закрила інша, низ рамки «відпав»), її не викидаємо,
  а плавно проводимо. На першому/останньому кадрах проїзду викидів не шукаємо: там не перевірити, чи
  значення повернулось, а номер біля краю кадру справді різко прискорюється (ширококутна камера).
- Край, що вперся в межу кадру (машина чи номер виїжджають за кадр), лишаємо на межі і не беремо в криву:
  справжній край там уже за кадром, а різкий злам руху на межі парабола не пройде.
- Короткі пропуски (≤ max_gap) заповнюємо прямою між згладженими краями з обох боків.
Згладжені рамки — лише для показу в плеєрі; рішення, вирізки й найкращий момент рахуються з сирих.
"""
import numpy as np

MAX_RUN = 2  # скільки кадрів поспіль можна вважати викидом


def _window(ts, t, half):
    return int(np.searchsorted(ts, t - half, 'right')), int(np.searchsorted(ts, t + half, 'left'))


def _fit(ts, xs, ok, t, half, skip=None, min_n=1):
    """Значення в момент t: зважена парабола по сусідах ± half с (пряма, якщо сусіди лише з одного боку).

    None, якщо сусідів менше за min_n.
    """
    lo, hi = _window(ts, t, half)
    m = ok[lo:hi].copy()
    if skip is not None and lo <= skip < hi:
        m[skip - lo] = False
    n = int(m.sum())
    if n < max(1, min_n):
        return None
    d = ts[lo:hi][m] - t
    w = (1 - (np.abs(d) / half) ** 3) ** 3
    deg = 2 if (d < 0).sum() >= 2 and (d > 0).sum() >= 2 else 1
    deg = min(deg, n - 1)
    sw = np.sqrt(w)
    coef = np.linalg.lstsq(np.vander(d, deg + 1) * sw[:, None], xs[lo:hi][m] * sw, rcond=None)[0]
    return float(coef[-1])


def _clean(ts, xs, size, half, ok):
    """Прибирає викиди по одному: кадр, найдальший від прогнозу по сусідах (поріг max(10 px, 8% розміру)).

    Прогноз — щонайменше з 3 сусідів («прогноз» з однієї точки при швидкому русі хибно бачить викид).
    Потім кожну серію прибраних кадрів перевіряємо (_returns): якщо рух до і після неї не сходиться —
    це справжня зміна рамки, а не викид, і кадри повертаємо.
    """
    ok, usable = ok.copy(), ok.copy()
    thr = np.maximum(10.0, 0.08 * size)

    def resid(i):
        if not ok[i]:
            return 0.0
        v = _fit(ts, xs, ok, ts[i], half, skip=i, min_n=3)
        return 0.0 if v is None else abs(xs[i] - v) / thr[i]

    def run_if_removed(i):  # довжина серії прибраних кадрів поспіль, якщо прибрати ще й i
        a = i
        while a > 0 and usable[a - 1] and not ok[a - 1]:
            a -= 1
        b = i
        while b + 1 < len(ok) and usable[b + 1] and not ok[b + 1]:
            b += 1
        return b - a + 1

    r = np.array([resid(i) for i in range(len(xs))])
    while True:
        i = int(r.argmax())
        if r[i] <= 1:
            break
        r[i] = 0.0
        if run_if_removed(i) > MAX_RUN:
            continue
        ok[i] = False
        lo, hi = _window(ts, ts[i], half)
        for j in range(lo, hi):
            if ok[j]:
                r[j] = resid(j)
    i = 0
    while i < len(ok):
        if ok[i] or not usable[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(ok) and usable[j + 1] and not ok[j + 1]:
            j += 1
        if not _returns(ts, xs, ok, i, j, 2 * half, thr[i:j + 1].max()):
            ok[i:j + 1] = True
        i = j + 1
    return ok


def _returns(ts, xs, ok, i, j, reach, thr):
    """Чи кадри i..j — справжній викид: рух до і після них лежить на одній лінії (значення «повернулось»).

    Продовження руху з одного боку (пряма по 2–4 найближчих кадрах) має влучити в найближчий кадр
    з другого боку. Немає кадрів з якогось боку (початок чи кінець проїзду) — не викид.
    """
    def near(m, t):
        idx = np.flatnonzero(m)
        return idx[np.argsort(np.abs(ts[idx] - t))[:4]]

    L = near(ok & (ts < ts[i]) & (ts >= ts[i] - reach), ts[i])
    R = near(ok & (ts > ts[j]) & (ts <= ts[j] + reach), ts[j])
    line = lambda idx: np.polyfit(ts[idx], xs[idx], 1) if len(idx) >= 2 else None
    lineL, lineR = line(L), line(R)
    if lineL is not None and len(R):
        k = R[np.argmin(ts[R])]
        return abs(np.polyval(lineL, ts[k]) - xs[k]) <= thr
    if lineR is not None and len(L):
        k = L[np.argmax(ts[L])]
        return abs(np.polyval(lineR, ts[k]) - xs[k]) <= thr
    return False


def _at_border(B, k, W, H):
    """Край k (0..3 = x1, y1, x2, y2) уперся в межу кадру."""
    return B[:, k] <= 4 if k < 2 else B[:, k] >= (W, H)[k - 2] - 4


def _arrays(series):
    keys = sorted(series)
    B = np.array([series[t] for t in keys], float)
    return keys, np.array(keys, float), B, np.maximum(B[:, 2] - B[:, 0], B[:, 3] - B[:, 1])


def smooth(series, out_times, W, H, half=0.13, max_gap=0.5):
    """series: {t: [x1, y1, x2, y2]} сирі рамки; out_times: моменти, де може бути рамка (усі кадри машини).

    -> {t: [x1, y1, x2, y2]} на кожному кадрі з series і в пропусках ≤ max_gap.
    """
    if not series:
        return {}
    keys, ts, B, size = _arrays(series)
    sm = B.copy()
    for k in range(4):
        edge = _at_border(B, k, W, H)
        ok = _clean(ts, B[:, k], size, half, ~edge)
        for i, t in enumerate(ts):
            v = None if edge[i] else _fit(ts, B[:, k], ok, t, half, min_n=2)
            if v is not None:
                sm[i, k] = v
    out = {t: sm[i] for i, t in enumerate(keys)}
    for i in range(len(keys) - 1):
        a, b = keys[i], keys[i + 1]
        if b - a > max_gap:
            continue
        for t in out_times:
            if a < t < b:
                out[t] = sm[i] + (sm[i + 1] - sm[i]) * ((t - a) / (b - a))
    lim = np.array([W, H, W, H], float)
    return {t: [int(round(v)) for v in np.clip(b, 0, lim)] for t, b in out.items()}

