"""Перевірка прочитаних номерів у базі розшуку МВС через cardetect.in.ua.

  python tools/wanted_check.py runs/bus-driving/data.js              # усі прочитані номери -> runs/bus-driving/wanted.js
  python tools/wanted_check.py runs/bus-driving/data.js --limit 10   # лише перші 10
  python tools/wanted_check.py --plates BC9349TA BC8535IB             # конкретні номери (латиницею, як дає OCR)

Результат для плеєра — wanted.js поруч з data.js (window.WANTED: номер машини -> результат). Плеєр показує
його на картці («не в розшуку» / «у розшуку»), у подробицях (марка й колір з бази) і на відео (червона табличка).

Сайт має GET-ендпоінт /api/wanted/check?plate=... (без входу і капчі). Це внутрішній API сайту, не
документований: може змінитись або почати блокувати. Тому запити йдуть повільно (пауза між ними), а
відповіді зберігаються в кеш (--cache) і повторно не запитуються. Кеш не застаріває сам: розшук змінюється,
тож щоб перевірити заново — видаліть файл кешу (дата перевірки є в кожному записі й у підказці в плеєрі).

Важливо: база шукає лише за КИРИЛИЦЕЮ. OCR видає латиницю, і «BC9349TA» латиницею повертає «чисто», хоча
«ВС9349ТА» кирилицею в розшуку. Тому літери перекладаються в кирилицю (I -> українська І). Перевіряються
лише номери стандартного українського формату (АА 1234 АА); решта позначаються «не перевірено», а не «чисто».

Збіг у базі ще не означає, що це та сама машина: OCR міг помилитись на символ і влучити в чужий номер.
Тому для збігу виводиться марка й колір з бази, щоб порівняти з вирізкою машини.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = 'https://cardetect.in.ua/api/wanted/check'
LAT2CYR = str.maketrans('ABCEHIKMOPTX', 'АВСЕНІКМОРТХ')  # латинські літери, що є на українських номерах
UA_PLATE = re.compile(r'^[ABCEHIKMOPTX]{2}\d{4}[ABCEHIKMOPTX]{2}$')


def save_json(obj, path):
    tmp = f'{path}.tmp'
    Path(tmp).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding='utf-8')
    os.replace(tmp, path)


def query(plate_cyr, timeout=30):
    url = f'{API}?{urllib.parse.urlencode({"plate": plate_cyr})}'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json',
                                               'Referer': 'https://cardetect.in.ua/wanted-check'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))


def check(plate, cache, delay):
    """plate латиницею з OCR -> {'result': wanted|was_wanted|clean|skipped|error, ...}. Кешує все, крім помилок."""
    if not UA_PLATE.match(plate):
        return {'result': 'skipped', 'why': 'не стандартний український формат'}
    cyr = plate.translate(LAT2CYR)
    if cyr in cache:
        return {**cache[cyr], 'cached': True}
    time.sleep(delay)
    t0 = time.time()
    try:
        resp = query(cyr)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return {'result': 'error', 'why': str(e)}
    d = resp.get('data') or {}
    if resp.get('type') != 'success' or 'status' not in d:
        return {'result': 'error', 'why': (d.get('error') if isinstance(d, dict) else None) or resp.get('message') or str(resp)[:200]}
    ep = next(iter((d.get('episodes') or {}).values()), {})
    out = {
        'result': 'wanted' if d.get('is_wanted_now') else 'was_wanted' if d.get('was_wanted_before') else 'clean',
        'status': d['status'], 'plate_cyr': cyr, 'checked_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'seconds': round(time.time() - t0, 2),
    }
    if ep:
        out.update({k: ep.get(k) for k in ('brand', 'color', 'theft_date', 'insert_date', 'organ', 'is_active')})
    cache[cyr] = out
    return out


def load_vehicles(path):
    raw = Path(path).read_text(encoding='utf-8')
    data = json.loads(raw[raw.index('{'):raw.rstrip().rindex(';')])
    return data['vehicles'], (data.get('source') or {}).get('offset', 0)


def hms(s):
    s = int(s)
    return f'{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}' if s >= 3600 else f'{s // 60}:{s % 60:02d}'


LABEL = {'wanted': 'ЗБІГ: у розшуку зараз', 'was_wanted': 'був у розшуку раніше', 'clean': 'чисто',
         'skipped': 'не перевірено', 'error': 'помилка'}


def player_js(read, cache):
    """window.WANTED для плеєра: номер машини (n) -> результат. Машини з однаковим номером отримують один результат."""
    out = {}
    for v in read:
        t = v['plate']['text']
        if not UA_PLATE.match(t):
            out[v['n']] = {'result': 'skipped'}
            continue
        r = cache.get(t.translate(LAT2CYR))
        if r:
            out[v['n']] = {k: r[k] for k in ('result', 'brand', 'color', 'theft_date', 'organ', 'checked_at') if r.get(k)}
    return ('// Згенеровано tools/wanted_check.py: перевірка номерів у базі розшуку (cardetect.in.ua)\n'
            f'window.WANTED = {json.dumps(out, ensure_ascii=False)};\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('data', nargs='?', help='data.js плеєра')
    p.add_argument('--plates', nargs='+', help='перевірити ці номери замість data.js')
    p.add_argument('--limit', type=int, default=0, help='скільки прочитаних номерів перевірити (0 — усі)')
    p.add_argument('--cache', default='dataset/full/wanted.json')
    p.add_argument('--out', help='файл для плеєра (типово wanted.js поруч з data.js)')
    p.add_argument('--delay', type=float, default=1.5, help='пауза перед кожним запитом, с')
    a = p.parse_args()
    if not a.data and not a.plates:
        p.error('потрібен data.js або --plates')

    cache = json.loads(Path(a.cache).read_text(encoding='utf-8')) if Path(a.cache).exists() else {}
    Path(a.cache).parent.mkdir(parents=True, exist_ok=True)

    read = []
    if a.plates:
        rows = [(None, None, None, pl.upper()) for pl in a.plates]
    else:
        out = a.out or str(Path(a.data).parent / 'wanted.js')
        vehicles, offset = load_vehicles(a.data)
        read = sorted((v for v in vehicles if v['plate']['status'] == 'read'), key=lambda v: v['first_seen'])
        first = {}
        for v in read:
            if UA_PLATE.match(v['plate']['text']):
                first.setdefault(v['plate']['text'], v)
        rows = [(v['n'], offset + v['best_time'], v['vehicle_type'], t) for t, v in first.items()]
        print(f'прочитаних номерів {len(read)}: стандартного формату {sum(UA_PLATE.match(v["plate"]["text"]) is not None for v in read)} '
              f'({len(first)} різних, перевіряються), інших {len(read) - sum(UA_PLATE.match(v["plate"]["text"]) is not None for v in read)} — пропущено', flush=True)
        if a.limit:
            rows = rows[:a.limit]

    counts, t0 = {}, time.time()
    for i, (n, t, vtype, plate) in enumerate(rows, 1):
        r = check(plate, cache, a.delay)
        save_json(cache, a.cache)
        counts[r['result']] = counts.get(r['result'], 0) + 1
        el = time.time() - t0
        prog = f'[{i:>3}/{len(rows)} {hms(el)}, ще ~{hms(el / i * (len(rows) - i))}] ' if len(rows) > 1 else ''
        where = f'№{n:<4} {hms(t):>7}  ' if n is not None else ''
        extra = ''
        if r['result'] in ('wanted', 'was_wanted'):
            extra = f"  | у базі: {r.get('brand')}, {r.get('color')}, викрадено {str(r.get('theft_date'))[:10]}"
            if vtype:
                extra += f' | на відео: {vtype}'
        elif r['result'] in ('skipped', 'error'):
            extra = f"  ({r['why']})"
        took = 'кеш' if r.get('cached') else f"{r['seconds']:.1f} с" if 'seconds' in r else ''
        print(f'{prog}{where}{plate:<9} {LABEL[r["result"]]:<22} {took:>6}{extra}', flush=True)
        if read and (i % 20 == 0 or i == len(rows)):
            Path(out).write_text(player_js(read, cache), encoding='utf-8')
    print('разом: ' + ', '.join(f'{LABEL[k]} {v}' for k, v in counts.items()) + f'  (кеш: {a.cache})')
    if read:
        Path(out).write_text(player_js(read, cache), encoding='utf-8')
        print(f'для плеєра: {out}')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
