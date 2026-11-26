"""Малює аркуші для перевірки розмітки датасету.

  python tools/research/make_preview.py dataset

Створює в dataset/preview/:
  frames_NN.jpg — кадри з усіма рамками номерів (зелена — прочитаний, помаранчева — text: null);
  plates_NN.jpg — кожен номер вирізано й збільшено, поруч розмічений текст. Тут зручно звіряти,
                  чи текст правильний і чи рамка щільно обходить номер.
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_font(name, size):
    """Шрифт Windows; на інших системах — вбудований шрифт Pillow."""
    try:
        return ImageFont.truetype(f'C:/Windows/Fonts/{name}', size)
    except OSError:
        return ImageFont.load_default(size)

GREEN, ORANGE = (40, 220, 90), (255, 140, 0)
FONT = load_font('arialbd.ttf', 26)
SMALL = load_font('arial.ttf', 20)


def frames_sheets(ds, labels, out):
    cols, rows, w, h = 3, 3, 640, 360
    per = cols * rows
    imgs = labels['images']
    for s in range(0, len(imgs), per):
        sheet = Image.new('RGB', (cols * w, rows * h), 'black')
        for i, rec in enumerate(imgs[s:s + per]):
            im = Image.open(ds / rec['file']).convert('RGB')
            d = ImageDraw.Draw(im)
            for p in rec['plates']:
                x, y, bw, bh = p['bbox']
                d.rectangle([x - 3, y - 3, x + bw + 3, y + bh + 3], outline=GREEN if p['text'] else ORANGE, width=6)
            im = im.resize((w, h), Image.LANCZOS)
            d = ImageDraw.Draw(im)
            n_read = sum(1 for p in rec['plates'] if p['text'])
            d.text((8, 6), f"{Path(rec['file']).stem}  ·  номерів: {n_read} + {len(rec['plates']) - n_read} нечитаних",
                   fill='white', font=SMALL, stroke_width=2, stroke_fill='black')
            sheet.paste(im, ((i % cols) * w, (i // cols) * h))
        sheet.save(out / f'frames_{s // per:02d}.jpg', quality=88)


def plates_sheets(ds, labels, out):
    tiles = []
    for rec in labels['images']:
        im = Image.open(ds / rec['file']).convert('RGB')
        for p in rec['plates']:
            x, y, bw, bh = p['bbox']
            mx, my = max(12, bw // 3), max(10, bh)  # поле навколо номера
            crop = im.crop((x - mx, y - my, x + bw + mx, y + bh + my))
            k = 300 / crop.width
            crop = crop.resize((300, round(crop.height * k)), Image.LANCZOS)
            d = ImageDraw.Draw(crop)
            d.rectangle([mx * k, my * k, (mx + bw) * k, (my + bh) * k], outline=GREEN if p['text'] else ORANGE, width=2)
            tiles.append((crop, p, Path(rec['file']).stem))

    tw, th, cols, rows = 640, 150, 3, 7
    per = cols * rows
    for s in range(0, len(tiles), per):
        sheet = Image.new('RGB', (cols * tw, rows * th), (30, 30, 30))
        d = ImageDraw.Draw(sheet)
        for i, (crop, p, stem) in enumerate(tiles[s:s + per]):
            x, y = (i % cols) * tw, (i // cols) * th
            c = crop.crop((0, 0, 300, min(crop.height, th - 8)))
            sheet.paste(c, (x + 6, y + 4))
            d.text((x + 316, y + 18), p['text'] or 'не читається', fill=GREEN if p['text'] else ORANGE, font=FONT)
            d.text((x + 316, y + 60), f"{p['id']}  ·  {stem}", fill=(200, 200, 200), font=SMALL)
            d.text((x + 316, y + 88), f"{p['bbox'][2]}×{p['bbox'][3]} px", fill=(150, 150, 150), font=SMALL)
        sheet.save(out / f'plates_{s // per:02d}.jpg', quality=90)
    return len(tiles)


def main():
    ds = Path(sys.argv[1])
    labels = json.loads((ds / 'labels.json').read_text(encoding='utf-8'))
    out = ds / 'preview'
    out.mkdir(exist_ok=True)
    for old in out.glob('*.jpg'):
        old.unlink()
    frames_sheets(ds, labels, out)
    n = plates_sheets(ds, labels, out)
    print(f"{len(labels['images'])} кадрів, {n} номерів -> {out}")


if __name__ == '__main__':
    main()
