"""Складає кадри з папки в аркуші-сітки, щоб швидко переглянути, де є машини з номерами.

  python tools/research/contact_sheet.py <папка з кадрами> <папка для аркушів> [--cols 4 --rows 4 --width 480]

На кожній мініатюрі підписано час кадру у відео (береться з імені файлу t0070.0.jpg).
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_font(name, size):
    """Шрифт Windows; на інших системах — вбудований шрифт Pillow."""
    try:
        return ImageFont.truetype(f'C:/Windows/Fonts/{name}', size)
    except OSError:
        return ImageFont.load_default(size)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('src')
    p.add_argument('out')
    p.add_argument('--cols', type=int, default=4)
    p.add_argument('--rows', type=int, default=4)
    p.add_argument('--width', type=int, default=480)
    p.add_argument('--fit', action='store_true', help='зберігати пропорції замість розтягування до 16:9')
    p.add_argument('--name', default='sheet', help='початок імені файлів аркушів')
    a = p.parse_args()

    files = sorted(Path(a.src).glob('t*.jpg'))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    w, h = a.width, a.width * 9 // 16
    per = a.cols * a.rows
    font = load_font('arialbd.ttf', 24)

    for s in range(0, len(files), per):
        sheet = Image.new('RGB', (w * a.cols, h * a.rows), 'black')
        d = ImageDraw.Draw(sheet)
        for i, f in enumerate(files[s:s + per]):
            x, y = (i % a.cols) * w, (i // a.cols) * h
            im = Image.open(f)
            if a.fit:  # не розтягувати: вписати зі збереженням пропорцій (для фото машин і номерів)
                im.thumbnail((w - 4, h - 34), Image.LANCZOS)
                sheet.paste(im, (x + (w - im.width) // 2, y + 32))
            else:
                sheet.paste(im.resize((w, h), Image.LANCZOS), (x, y))
            d.text((x + 6, y + 4), f.stem[1:], fill='yellow', font=font,
                   stroke_width=2, stroke_fill='black')
        sheet.save(out / f'{a.name}_{s // per:02d}.jpg', quality=88)
    print(f'{len(files)} кадрів -> {(len(files) + per - 1) // per} аркушів у {out}')


if __name__ == '__main__':
    main()
