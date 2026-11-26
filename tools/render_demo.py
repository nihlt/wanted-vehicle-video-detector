"""Коротке демо для README: відрізок відео з рамками машин і прочитаними номерами, як у плеєрі.

  python tools/render_demo.py player/data.js player/clip.mp4 docs/demo.gif --start 8 --dur 7
  python tools/render_demo.py player/data.js player/clip.mp4 docs/demo.gif --pick 6   # сам знайде 6 с з найбільшою кількістю номерів (так зроблено docs/demo.gif)

Рамки беруться з data.js (той самий файл, що читає плеєр) і так само, як у плеєрі, ставляться пропорційно
часу між двома порахованими кадрами. Синя рамка — машина, зелена — номер, біла табличка — прочитаний номер
(помаранчева рамка таблички — прочитано непевно). GIF кодує ffmpeg з власною палітрою (менший файл, чисті кольори).
"""
import argparse
import bisect
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BLUE, GREEN, CONE, INK = (47, 111, 224), (74, 222, 128), (240, 138, 75), (22, 24, 27)


def load_font(name, size):
    try:
        return ImageFont.truetype(f'C:/Windows/Fonts/{name}', size)
    except OSError:
        return ImageFont.load_default(size)


def load_data(path):
    raw = Path(path).read_text(encoding='utf-8')
    return json.loads(raw[raw.index('{'):raw.rstrip().rindex(';')])


def plate_text(t):
    if len(t) == 8 and t[:2].isalpha() and t[2:6].isdigit() and t[6:].isalpha():
        return f'{t[:2]} {t[2:6]} {t[6:]}'
    return t


def boxes_at(times, samples, t):
    """Як boxesAt() у player/app.js: між двома порахованими кадрами — пропорційно часу."""
    hi = bisect.bisect_left(times, t)
    if hi < len(times) and times[hi] - t < 1e-3:
        return samples[hi][1]
    lo = hi - 1
    if lo < 0 or hi >= len(times) or times[hi] - times[lo] > 0.05:
        k = min(range(max(0, hi - 1), min(len(times), hi + 1)), key=lambda i: abs(times[i] - t))
        return samples[k][1] if abs(times[k] - t) <= 0.05 else []
    k = (t - times[lo]) / (times[hi] - times[lo])
    A, B = {r[0]: r for r in samples[lo][1]}, {r[0]: r for r in samples[hi][1]}
    out = []
    for n in A.keys() | B.keys():
        a, b = A.get(n), B.get(n)
        if not a or not b:
            out.append(a if (a and k < 0.5) or not b else b)
            continue
        mix = lambda i, j: [a[q] + (b[q] - a[q]) * k for q in range(i, j)]
        plate = mix(5, 9) if len(a) > 5 and len(b) > 5 else []
        out.append([n, *mix(1, 5), *plate])
    return out


def pick_window(data, dur):
    """Початок відрізка, де найдовше видно машини з прочитаним номером (сума секунд по всіх машинах)."""
    read = {v['n'] for v in data['vehicles'] if v['plate']['status'] == 'read'}
    times = [s[0] for s in data['samples']]
    score = [sum(r[0] in read for r in s[1]) for s in data['samples']]
    cum = np.concatenate([[0], np.cumsum(score)])
    best, best_t = -1, 0.0
    for i, t in enumerate(times):
        j = bisect.bisect_left(times, t + dur)
        if j >= len(times):
            break
        if cum[j] - cum[i] > best:
            best, best_t = cum[j] - cum[i], t
    return best_t


def draw(frame, rows, vehicles, W, H, scale):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    w_out, h_out = img.size
    big, small = load_font('arialbd.ttf', round(34 * scale)), load_font('arialbd.ttf', round(28 * scale))
    lw = max(2, round(3 * scale))
    for r in rows:
        n, x, y, w, h = r[:5]
        x1, y1, x2, y2 = x * w_out, y * h_out, (x + w) * w_out, (y + h) * h_out
        d.rectangle([x1, y1, x2, y2], outline=BLUE, width=lw)
        d.text((x1 + 6 * scale, y1 + 4 * scale), str(n), font=small, fill='white', stroke_width=max(2, lw), stroke_fill='black')
        if len(r) >= 9:
            px, py, pw, ph = r[5:9]
            d.rectangle([px * w_out, py * h_out, (px + pw) * w_out, (py + ph) * h_out], outline=GREEN, width=max(1, lw - 1))
        p = vehicles.get(n, {}).get('plate') or {}
        if p.get('text'):
            s = plate_text(p['text']) + ('' if p['status'] == 'read' else '?')
            tw = d.textlength(s, font=big)
            bw, bh = tw + 20 * scale, 46 * scale
            ly, lx = y1 - bh - 6 * scale, x1
            if ly < 4:
                ly, lx = y1 + 6 * scale, x1 + 70 * scale
            lx = min(max(4, lx), w_out - bw - 4)
            d.rounded_rectangle([lx, ly, lx + bw, ly + bh], radius=6 * scale, fill=(251, 251, 248),
                                outline=INK if p['status'] == 'read' else CONE, width=lw)
            d.text((lx + 10 * scale, ly + bh / 2), s, font=big, fill=INK, anchor='lm')
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('data')
    p.add_argument('video')
    p.add_argument('out', help='.gif (палітра ffmpeg) або .mp4')
    p.add_argument('--start', type=float, default=0, help='з якої секунди відео')
    p.add_argument('--dur', type=float, default=7)
    p.add_argument('--pick', type=float, default=0, help='знайти найкращий відрізок такої довжини замість --start/--dur')
    p.add_argument('--width', type=int, default=720)
    p.add_argument('--fps', type=float, default=10)
    a = p.parse_args()

    data = load_data(a.data)
    W, H = data['source']['width'], data['source']['height']
    if a.pick:
        a.start, a.dur = pick_window(data, a.pick), a.pick
        print(f'відрізок: {a.start:.1f}–{a.start + a.dur:.1f} с')
    times = [s[0] for s in data['samples']]
    vehicles = {v['n']: v for v in data['vehicles']}
    w_out = a.width
    h_out = round(H * w_out / W / 2) * 2
    scale = w_out / W * 1.6  # текст і лінії трохи товщі, ніж пропорційно: у маленькому GIF вони інакше губляться

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.out.endswith('.gif'):
        vf = 'split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle'
        enc = ['-vf', vf, '-loop', '0']
    else:
        enc = ['-c:v', 'libx264', '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart']
    ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w_out}x{h_out}',
                           '-r', str(a.fps), '-i', '-', *enc, a.out], stdin=subprocess.PIPE)
    # Читаємо з початку і рахуємо час за номером кадру, як process_clip.py (перемотування OpenCV неточне)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    next_t, n, idx = a.start, 0, -1
    while next_t <= a.start + a.dur:
        idx += 1
        t = idx / fps
        if t + 0.5 / fps < next_t:
            if not cap.grab():
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (w_out, h_out), interpolation=cv2.INTER_AREA)
        ff.stdin.write(draw(small, boxes_at(times, data['samples'], t), vehicles, W, H, scale).tobytes())
        next_t += 1 / a.fps
        n += 1
    ff.stdin.close()
    ff.wait()
    print(f'{n} кадрів -> {a.out} ({Path(a.out).stat().st_size / 1e6:.1f} МБ)')


if __name__ == '__main__':
    main()
