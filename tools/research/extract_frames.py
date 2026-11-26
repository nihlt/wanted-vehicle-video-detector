"""Витягує кадри з відео в повній якості за заданими моментами часу.

Приклади:
  # кожні 2 с на ділянках 0–12 с і 56–92 с
  python tools/research/extract_frames.py bus-driving.webm out/ --windows 0-12,56-92 --step 2
  # конкретні моменти
  python tools/research/extract_frames.py bus-driving.webm out/ --times 70,190.5

Кожен кадр витягується окремим запуском ffmpeg з `-ss` перед `-i`:
ffmpeg перемотує до найближчого ключового кадру і декодує до точного часу.
Файл називається за часом у відео: t0070.0.jpg = кадр на 70.0 с.
Прогрес пишеться в <out>/progress.log.
"""
import argparse
import subprocess
import time
from pathlib import Path


def frame_times(args):
    if args.times:
        return [float(t) for t in args.times.split(',')]
    times = []
    for w in args.windows.split(','):
        a, b = (float(x) for x in w.split('-'))
        t = a
        while t <= b + 1e-6:
            times.append(round(t, 2))
            t += args.step
    return sorted(set(times))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('video')
    p.add_argument('out')
    p.add_argument('--windows', help='ділянки у секундах: 0-12,56-92')
    p.add_argument('--step', type=float, default=2.0, help='крок у секундах всередині ділянки')
    p.add_argument('--times', help='конкретні моменти у секундах: 70,190.5')
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    times = frame_times(args)
    log = open(out / 'progress.log', 'w', encoding='utf-8')
    start = time.time()

    for i, t in enumerate(times, 1):
        dst = out / f't{t:06.1f}.jpg'
        if not dst.exists():
            # -q:v 2 — висока якість JPEG (майже без втрат), щоб дрібний текст номерів не розмився
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', str(t), '-i', args.video,
                            '-frames:v', '1', '-q:v', '2', str(dst)], check=True)
        el = time.time() - start
        rate = i / el if el > 0 else 0
        eta = (len(times) - i) / rate if rate > 0 else 0
        line = f'{i}/{len(times)} кадрів | {rate:.2f} кадр/с | минуло {el:.0f} с | залишилось ~{eta:.0f} с'
        print(line, file=log, flush=True)
    print('ГОТОВО', file=log, flush=True)


if __name__ == '__main__':
    main()
