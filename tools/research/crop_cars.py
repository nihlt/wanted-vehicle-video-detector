"""Вирізає кожну знайдену машину з кадру в окреме фото — на них далі шукаємо номер.

  python tools/research/crop_cars.py dataset

Бере рамки з dataset/cars/cars.json, ріже з оригінальних кадрів dataset/images/ (без намальованих рамок),
у повній роздільності. Створює в dataset/car_crops/:
  t0070.0_c0.jpg ...  машина №0 з кадру t0070.0 і т. д.;
  crops.json          для кожного фото: з якого кадру і де саме в кадрі воно було вирізане
                      (щоб знайдений на фото номер можна було перевести назад у координати кадру).
"""
import json
import sys
from pathlib import Path

from PIL import Image

MARGIN = 0.03  # трохи поля навколо рамки машини, щоб номер біля краю не обрізався


def main():
    ds = Path(sys.argv[1])
    cars = json.loads((ds / 'cars' / 'cars.json').read_text(encoding='utf-8'))
    out = ds / 'car_crops'
    out.mkdir(exist_ok=True)
    for old in out.glob('*.jpg'):
        old.unlink()

    crops = []
    for rec in cars['images']:
        im = Image.open(ds / rec['file'])
        stem = Path(rec['file']).stem
        for k, c in enumerate(rec['cars']):
            x, y, w, h = c['bbox']
            mx, my = round(w * MARGIN), round(h * MARGIN)
            x0, y0 = max(0, x - mx), max(0, y - my)
            x1, y1 = min(im.width, x + w + mx), min(im.height, y + h + my)
            name = f'{stem}_c{k}.jpg'
            im.crop((x0, y0, x1, y1)).save(out / name, quality=95)
            crops.append({'file': name, 'frame': rec['file'], 'origin': [x0, y0], 'size': [x1 - x0, y1 - y0],
                          'cls': c['cls'], 'conf': c['conf']})

    (out / 'crops.json').write_text(json.dumps({'crops': crops}, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'{len(crops)} машин -> {out}')


if __name__ == '__main__':
    main()
