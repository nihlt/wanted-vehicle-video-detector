"""Аркуші для перевірки на око: кожна машина з плеєра — фото, вирізка номера і рішення по треку.

  python tools/research/sheet_vehicles.py player dataset/preview

Читає <player>/data.js і <player>/crops/, пише <out>/vehicles_NN.jpg (по 28 машин на аркуш).
Колір підпису: зелений — прочитано, помаранчевий — непевно, сірий — не прочитано.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # основні модулі лежать у tools/
from plates_nn import put  # noqa: E402

GREEN, ORANGE, GREY = (120, 255, 120), (80, 160, 255), (170, 170, 170)


def main():
    player, out = Path(sys.argv[1]), Path(sys.argv[2])
    raw = (player / 'data.js').read_text(encoding='utf-8')
    data = json.loads(raw[raw.index('{'):raw.rstrip().rindex(';')])
    tiles = []
    for v in data['vehicles']:
        tile = np.full((240, 250, 3), 40, np.uint8)
        car = cv2.imread(str(player / v['thumb']))
        k = min(246 / car.shape[1], 140 / car.shape[0])
        car = cv2.resize(car, (round(car.shape[1] * k), round(car.shape[0] * k)), interpolation=cv2.INTER_AREA)
        tile[20:20 + car.shape[0], 2:2 + car.shape[1]] = car
        p = v['plate']
        put(tile, f'#{v["n"]}  {v["best_time"]:.1f} s', (4, 14), 0.42)
        if v['plate_crop']:
            pl = cv2.resize(cv2.imread(str(player / v['plate_crop'])), (180, 45), interpolation=cv2.INTER_CUBIC)
            tile[165:210, 2:182] = pl
        if p['text']:
            col = GREEN if p['status'] == 'read' else ORANGE
            put(tile, f'{p["text"]}{"" if p["status"] == "read" else "?"}  {p["n_agree"]}/{p["n_reads"]}',
                (4, 232), 0.55, col)
        else:
            put(tile, 'ne prochytano', (4, 232), 0.55, GREY)
        tiles.append(tile)

    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob('vehicles_*.jpg'):
        old.unlink()
    cols, per = 7, 28
    blank = np.full_like(tiles[0], 40)
    for s in range(0, len(tiles), per):
        chunk = tiles[s:s + per]
        rows = [np.hstack(chunk[i:i + cols] + [blank] * (cols - len(chunk[i:i + cols])))
                for i in range(0, len(chunk), cols)]
        cv2.imwrite(str(out / f'vehicles_{s // per:02d}.jpg'), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f'{len(tiles)} машин -> {out}/vehicles_*.jpg')


if __name__ == '__main__':
    main()
