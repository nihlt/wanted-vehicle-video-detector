"""Проганяє весь ланцюжок по відео і готує дані для плеєра player/.

  python tools/process_clip.py player/clip.mp4 player --offset 290            # повний прохід (~8 хв) + фінал
  python tools/process_clip.py player/clip.mp4 player --offset 290 --reuse    # лише фінал з кешу (~20 с)

Етап 1, прохід по відео (кожен другий кадр, --step 2, тобто ~30 кадрів/с з 60):
  1. YOLO11n знаходить машини (пачками по 8 кадрів, так швидше).
  2. Трекер склеює рамки сусідніх кадрів в одну машину. За замовчуванням BoT-SORT (--tracker botsort):
     передбачає, куди рухається машина, враховує рух самої камери і підхоплює слабкі рамки (0.1–0.25).
     На еталоні (tools/research/compare_trackers.py) він не склеїв жодної пари різних машин, на відміну від
     простого IoU-трекера (--tracker iou: та сама машина, якщо рамка перекривається з попередньою на IoU ≥ 0.3).
  3. На кожній близькій машині (рамка ≥ 25 000 px²) нейромережа шукає номер, OCR читає текст
     (tools/track_plates.py observe()). Усе це зберігається в кеш --cache без жодних рішень.
Етап 2, фінал (tools/track_plates.py):
  4. Номер машини — одне рішення з усіх кадрів треку: голосування читань з вагою = впевненість OCR.
  5. Рамка номера — лише там, де знайдено свій номер (точна знахідка детектора), пропуски до 0.5 с інтерпольовано.
  6. Найкращий момент для картки: машина не обрізана краєм кадру, номер прочитано найчіткіше, рамка найбільша.
  7. Рамки для показу в плеєрі згладжені в часі (tools/smooth_boxes.py): крива по сусідніх кадрах з обох боків,
     поодинокі стрибки детектора прибрано, пропуски машини до 0.3 с і номера до 0.5 с заповнено.
     Рішення, вирізки й найкращий момент рахуються з сирих рамок.
Машини, які були близько менше ніж 0.5 с, відкидаємо: здебільшого це сміття або обірвані треки.

Створює в <out>/:
  data.js        window.LIVE = {source, samples, vehicles}; рамки нормалізовані до 0–1;
  crops/         vNNN_car.jpg і vNNN_plate.jpg для карток;
  progress.log   прогрес обробки.
А поруч із кешем — decisions.json: рішення по кожній машині (для перевірки).
Довгий прохід (усе відео — ~2 год) раз на --checkpoint секунд зберігає зроблене в --cache (з позначкою
partial): якщо процес упаде, `--reuse` збере плеєр хоча б з обробленої частини.
"""
import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.trackers.bot_sort import BOTSORT

from plates_nn import PlateReader, full_speed
from smooth_boxes import smooth
from track_plates import decide, observe, plate_track

VEHICLES = {2: 'car', 3: 'moto', 5: 'bus', 7: 'truck'}
# Налаштування трекерів ultralytics (як у cfg/trackers/botsort.yaml), щоб не залежати від yaml-файлу
BASE = dict(track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.25, track_buffer=30,
            match_thresh=0.8, fuse_score=True)
BOT = dict(gmc_method='sparseOptFlow', proximity_thresh=0.5, appearance_thresh=0.8, with_reid=False, model='auto')


class Dets:
    """Рамки одного кадру ([x1, y1, x2, y2, conf, cls]) у вигляді, який чекають трекери ultralytics."""

    def __init__(self, a):
        self.a = a

    conf = property(lambda s: s.a[:, 4])
    cls = property(lambda s: s.a[:, 5])
    xyxy = property(lambda s: s.a[:, :4])

    @property
    def xywh(self):
        a = self.a
        return np.stack([(a[:, 0] + a[:, 2]) / 2, (a[:, 1] + a[:, 3]) / 2, a[:, 2] - a[:, 0], a[:, 3] - a[:, 1]], 1)

    def __len__(self):
        return len(self.a)

    def __getitem__(self, k):
        return Dets(self.a[k])


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


class IoUTracker:
    """Найпростіший трекер: жадібно зіставляє рамки з треками за перекриттям."""

    def __init__(self, thr=0.3, max_gap=10):
        self.thr, self.max_gap = thr, max_gap  # max_gap: скільки оброблених кадрів трек може «не бачити» машину
        self.tracks, self.next_id = {}, 1

    def update(self, boxes, k):
        alive = {tid: t for tid, t in self.tracks.items() if k - t['k'] <= self.max_gap}
        pairs = sorted(((iou(t['box'], b), tid, j) for tid, t in alive.items() for j, b in enumerate(boxes)),
                       reverse=True)
        ids, used = [None] * len(boxes), set()
        for s, tid, j in pairs:
            if s < self.thr:
                break
            if tid in used or ids[j] is not None:
                continue
            ids[j] = tid
            used.add(tid)
        for j in range(len(boxes)):
            if ids[j] is None:
                ids[j] = self.next_id
                self.next_id += 1
            self.tracks[ids[j]] = {'box': boxes[j], 'k': k}
        return ids


def jpg(img, max_w=None):
    if max_w and img.shape[1] > max_w:
        img = cv2.resize(img, (max_w, round(img.shape[0] * max_w / img.shape[1])), interpolation=cv2.INTER_AREA)
    return cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


def hms(s):
    s = int(s)
    return f'{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}'


def save_json(obj, path):
    """Запис через тимчасовий файл: якщо процес упаде посеред запису, старий файл лишиться цілим."""
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, separators=(',', ':')), encoding='utf-8')
    os.replace(tmp, path)


def run_pass(a, log):
    """Етап 1: детекція, трекінг, номери. Повертає кеш: рамки машин і всі спостереження номерів по кадрах."""
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    Wf, Hf = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = (n_frames + a.step - 1) // a.step
    if a.limit:  # інакше прогрес і «залишилось» рахуються від усього відео
        total = min(total, int(a.limit * fps) // a.step + 1)
    model = YOLO(a.model)
    # BoT-SORT уміє користуватися слабкими рамками (0.1–0.25), тож детектор просимо віддавати і їх
    gpu = torch.cuda.device_count() > 0  # без відеокарти NVIDIA теж працює, але в рази повільніше
    opts = dict(imgsz=640, conf=0.1 if a.tracker == 'botsort' else 0.25, classes=list(VEHICLES),
                device=0 if gpu else 'cpu', agnostic_nms=True, verbose=False)
    if gpu:
        opts['quantize'] = 16  # FP16 на відеокарті
    tracker = BOTSORT(SimpleNamespace(**BASE, **BOT)) if a.tracker == 'botsort' else IoUTracker()
    reader = PlateReader()

    times, tracks, obs = [], {}, {}
    t_start, done, t_plates, n_cars = time.perf_counter(), 0, 0.0, 0
    t_log = t_save = t_start

    def cache(partial):
        return {'tracker': a.tracker, 'partial': partial,
                'source': {'file': Path(a.video).name, 'duration': round(n_frames / fps, 3), 'width': Wf,
                           'height': Hf, 'fps': round(fps, 3), 'step': a.step},
                'times': times, 'tracks': tracks, 'obs': obs}

    def process(batch):
        nonlocal done, t_plates, n_cars, t_log, t_save
        results = model.predict([f for _, f in batch], **opts)
        for (idx, frame), r in zip(batch, results):
            t = round(idx / fps, 3)
            det = np.concatenate([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()[:, None],
                                  r.boxes.cls.cpu().numpy()[:, None]], 1).astype(np.float32)
            if a.tracker == 'botsort':
                out = tracker.update(Dets(det), frame)
                # рамка самого детектора (а не згладжена трекером) + номер треку
                found = [(int(o[4]), det[int(o[7])]) for o in out]
            else:
                found = list(zip(tracker.update([tuple(d[:4]) for d in det], done), det))
            close = []
            for tid, d in found:
                box, c = tuple(round(float(v)) for v in d[:4]), d[5]
                if (box[2] - box[0]) * (box[3] - box[1]) < a.min_area:
                    continue  # трекаємо всі машини, а показуємо і читаємо номери лише близьких
                close.append((tid, box))
                tr = tracks.setdefault(str(tid), {'cls': {}, 'boxes': []})
                tr['boxes'].append([t, *box])
                tr['cls'][VEHICLES[int(c)]] = tr['cls'].get(VEHICLES[int(c)], 0) + 1
            t0 = time.perf_counter()
            for tid, lst in observe(frame, close, reader).items():
                for o in lst:
                    o['t'] = t
                obs.setdefault(str(tid), []).extend(lst)
            t_plates += time.perf_counter() - t0
            n_cars += len(close)
            times.append(t)
            done += 1
        now = time.perf_counter()
        el = now - t_start
        if now - t_log >= 5 or done >= total:  # рядок у лог раз на 5 с
            t_log = now
            log.write(f'{done}/{total} кадрів ({done / total:.1%}) | {done / el:.1f} кадр/с | минуло {hms(el)} | '
                      f'залишилось ~{hms((total - done) / max(done / el, 1e-9))} | відео до {times[-1]:.0f} с | '
                      f'номери: {t_plates / done * 1000:.0f} мс/кадр, {t_plates / max(n_cars, 1) * 1000:.0f} мс/машину\n')
        if a.checkpoint and now - t_save >= a.checkpoint:
            t_save = now
            save_json(cache(True), a.cache)
            log.write(f'збережено проміжний результат: {done} кадрів, відео до {times[-1]:.0f} с -> {a.cache}\n')

    batch, idx = [], 0
    while not (a.limit and idx > a.limit * fps):
        if idx % a.step == 0:
            ok, frame = cap.read()
            if not ok:
                break
            batch.append((idx, frame))
            if len(batch) == a.batch:
                process(batch)
                batch = []
        elif not cap.grab():  # кадр, який пропускаємо: лише прочитати, без перетворення в картинку
            break
        idx += 1
    if batch:
        process(batch)
    log.write(f'прохід завершено за {hms(time.perf_counter() - t_start)}\n')
    return cache(False)


def read_frames(video, fps, wanted_t, log):
    """Другий прохід: дістає кадри лише в потрібні моменти (для вирізок на картки).

    Відео стиснене, тож щоб дійти до кадру, декодуємо всі попередні: на всьому відео це кілька хвилин.
    """
    want = {round(t * fps): t for t in wanted_t}
    cap, frames, idx = cv2.VideoCapture(video), {}, 0
    last = max(want)
    t_start = t_log = time.perf_counter()
    while idx <= last:
        if idx in want:
            ok, f = cap.read()
            if not ok:
                break
            frames[want[idx]] = f
        elif not cap.grab():
            break
        idx += 1
        now = time.perf_counter()
        if now - t_log >= 5:
            t_log, el = now, now - t_start
            log.write(f'вирізки для карток: кадр {idx}/{last} ({idx / last:.1%}) | {idx / el:.0f} кадр/с | '
                      f'минуло {hms(el)} | залишилось ~{hms((last - idx) / (idx / el))}\n')
    return frames


def finalize(cache, a, log):
    """Етап 2: рішення по треку, стабільні рамки номерів, картки, data.js."""
    out = Path(a.out)
    (out / 'crops').mkdir(parents=True, exist_ok=True)
    for old in (out / 'crops').glob('*.jpg'):
        old.unlink()
    src = cache['source']
    Wf, Hf = src['width'], src['height']
    nb = lambda b: [round(b[0] / Wf, 4), round(b[1] / Hf, 4), round(b[2] / Wf, 4), round(b[3] / Hf, 4)]
    xywh = lambda b: [b[0], b[1], b[2] - b[0], b[3] - b[1]]
    whole = lambda c: c[0] > 8 and c[2] < Wf - 8  # машина не обрізана лівим/правим краєм кадру
    area = lambda c: (c[2] - c[0]) * (c[3] - c[1])

    # Лишаємо машини, які були близько хоча б min_seen секунд; нумеруємо за часом появи
    tracks = cache['tracks']
    keep = sorted((tid for tid, tr in tracks.items() if tr['boxes'][-1][0] - tr['boxes'][0][0] >= a.min_seen),
                  key=lambda tid: tracks[tid]['boxes'][0][0])
    info, report = {}, []
    for n, tid in enumerate(keep, 1):
        cars = {b[0]: b[1:] for b in tracks[tid]['boxes']}
        o = cache['obs'].get(tid, [])
        guess = decide(o)
        dec = guess if guess and guess['status'] != 'guess' else None  # сміття з розмитих кадрів не показуємо
        pt = plate_track(o, cars, dec)
        if dec:  # найкращий момент — де «свій» номер прочитано найчіткіше
            best = max((x for x in o if x['text'] == dec['text']),
                       key=lambda x: (whole(cars[x['t']]), x['w'], area(cars[x['t']])))
            bt, crop_box = best['t'], best['box']
        else:
            bt = max(cars, key=lambda t: (whole(cars[t]), t in pt, area(cars[t])))
            crop_box = pt.get(bt)
        # для показу: плавні рамки машини і номера (той самий розмір вікна, щоб номер не «їздив» по машині)
        span = [t for t in cache['times'] if min(cars) <= t <= max(cars)]
        cars_s = smooth(cars, span, Wf, Hf, max_gap=a.car_gap)
        pt_s = smooth(plate_track(o, cars, dec, max_gap=0), sorted(cars_s), Wf, Hf, max_gap=0.5)
        info[tid] = dict(n=n, cars=cars, pt=pt, dec=dec, best_t=bt, crop_box=crop_box, cars_s=cars_s, pt_s=pt_s)
        report.append({'n': n, 'track': tid, 'first': min(cars), 'last': max(cars), 'n_obs': len(o),
                       'n_plate_frames': len(pt), 'best_t': bt, 'decision': guess})

    frames = read_frames(a.video, src['fps'], {i['best_t'] for i in info.values()}, log)
    vehicles = []
    for tid in keep:
        i = info[tid]
        vid, car, dec, f = f'v{i["n"]:03d}', i['cars'][i['best_t']], i['dec'], frames[i['best_t']]
        x1, y1, x2, y2 = car
        (out / 'crops' / f'{vid}_car.jpg').write_bytes(jpg(f[y1:y2, x1:x2], 480))
        plate_at = i['pt'].get(i['best_t']) or i['crop_box']
        if i['crop_box']:
            px1, py1, px2, py2 = i['crop_box']
            m = max(4, (py2 - py1) // 3)
            (out / 'crops' / f'{vid}_plate.jpg').write_bytes(jpg(f[max(0, py1 - m):py2 + m, max(0, px1 - m):px2 + m]))
        cls = tracks[tid]['cls']
        vehicles.append({
            'id': vid, 'n': i['n'], 'first_seen': min(i['cars']), 'last_seen': max(i['cars']),
            'best_time': i['best_t'], 'bbox': nb(xywh(car)),
            'vehicle_type': max(cls, key=cls.get),
            'plate': {'found': bool(i['pt']), 'bbox': nb(xywh(plate_at)) if plate_at else None,
                      'text': dec['text'] if dec else None, 'status': dec['status'] if dec else None,
                      'confidence': dec['conf'] if dec else None,
                      'n_agree': dec['n_agree'] if dec else 0, 'n_reads': dec['n_reads'] if dec else 0,
                      'alts': dec['alts'] if dec else []},
            'thumb': f'crops/{vid}_car.jpg', 'plate_crop': f'crops/{vid}_plate.jpg' if i['crop_box'] else None,
        })

    at_t = {}
    for tid in keep:
        i = info[tid]
        for t, car in i['cars_s'].items():
            pb = i['pt_s'].get(t)
            at_t.setdefault(t, []).append([i['n']] + nb(xywh(car)) + (nb(xywh(pb)) if pb else []))
    samples = [[t, at_t.get(t, [])] for t in cache['times']]

    data = {
        'tracker': cache['tracker'],
        'source': {**src, 'offset_in_original': a.offset, 'original_file': a.original, 'min_area': a.min_area},
        'samples': samples,
        'vehicles': vehicles,
    }
    (out / 'data.js').write_text('// Згенеровано tools/process_clip.py — автоматична розмітка, не редагувати вручну\n'
                                 'window.LIVE = ' + json.dumps(data, ensure_ascii=False, separators=(',', ':')) + ';\n',
                                 encoding='utf-8')
    Path(a.cache).with_name('decisions.json').write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                         encoding='utf-8')
    st = [v['plate']['status'] for v in vehicles]
    msg = (f'машин у плеєрі {len(vehicles)}: номер прочитано {st.count("read")}, '
           f'непевно {st.count("uncertain")}, не прочитано {st.count(None)}')
    log.write(msg + '\n')
    print(msg)


def stay_awake():
    """Поки процес працює, просимо Windows не засинати через бездіяльність (діє лише до кінця процесу,
    налаштувань не змінює). Закрита кришка чи ручна гібернація все одно присплять ноутбук."""
    import ctypes
    import sys
    if sys.platform == 'win32':
        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('video')
    p.add_argument('out')
    p.add_argument('--offset', type=float, default=0, help='з якої секунди оригінального відео вирізано кліп')
    p.add_argument('--original', default='bus-driving.webm')
    p.add_argument('--step', type=int, default=2)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--min-area', type=int, default=25000)
    p.add_argument('--min-seen', type=float, default=0.5)
    p.add_argument('--car-gap', type=float, default=0.3, help='пропуски рамки машини до стількох секунд заповнюємо')
    p.add_argument('--model', default='models/yolo11n.pt')
    p.add_argument('--tracker', choices=['botsort', 'iou'], default='botsort')
    p.add_argument('--cache', default='dataset/clip/pass.json', help='куди зберегти / звідки взяти результат етапу 1')
    p.add_argument('--limit', type=float, default=0, help='обробити лише перші N секунд (для перевірки)')
    p.add_argument('--checkpoint', type=float, default=300, help='раз на стільки секунд зберігати зроблене в --cache')
    p.add_argument('--reuse', action='store_true', help='не проганяти відео, взяти етап 1 з --cache')
    a = p.parse_args()

    full_speed()  # інакше Windows через кілька секунд пригальмовує процес у ~10 разів (див. plates_nn.full_speed)
    stay_awake()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    Path(a.cache).parent.mkdir(parents=True, exist_ok=True)
    log = open(Path(a.out) / 'progress.log', 'a' if a.reuse else 'w', encoding='utf-8', buffering=1)
    t0 = time.perf_counter()
    if a.reuse:
        cache = json.loads(Path(a.cache).read_text(encoding='utf-8'))
    else:
        cache = run_pass(a, log)
        save_json(cache, a.cache)
    if cache.get('partial'):
        log.write(f'увага: у кеші лише частина відео (до {cache["times"][-1]:.0f} с) — прохід не дійшов до кінця\n')
    finalize(cache, a, log)
    log.write(f'ГОТОВО за {hms(time.perf_counter() - t0)}\n')


if __name__ == '__main__':
    main()
