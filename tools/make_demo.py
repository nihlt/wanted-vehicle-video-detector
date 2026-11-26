"""Онлайн-демо для GitHub Pages: оброблене відео -> demo/ (кліп 720p, дані, вирізки, сторінка плеєра).

  python tools/make_demo.py player demo                       # хвилина 4:50–5:50, як у README
  python tools/make_demo.py runs/bus-driving_290-350 demo     # або будь-який результат run.py

Кліп перекодовується в 720p і 30 кадрів/с зі стисненням сильнішим, ніж у run.py (~15 МБ на хвилину замість ~80),
і з -movflags +faststart, щоб браузер почав відтворення, не завантаживши файл цілком.
Рамки в data.js задані в частках кадру й прив'язані до часу, тож від роздільності й частоти кадрів кліпу не залежать.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from run import write_page  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('src', type=Path, help='папка з clip.mp4, data.js і crops/ (наприклад, player або runs/<назва>)')
    p.add_argument('out', type=Path, help='куди скласти демо (наприклад, demo)')
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--crf', type=int, default=30, help='якість H.264: більше число — менший файл')
    a = p.parse_args()

    for need in ('clip.mp4', 'data.js', 'crops'):
        if not (a.src / need).exists():
            sys.exit(f'у {a.src} немає {need}: спершу обробіть відео (python run.py ...)')
    a.out.mkdir(parents=True, exist_ok=True)

    print(f'ffmpeg: {a.src / "clip.mp4"} -> {a.out / "clip.mp4"} ({a.height}p, {a.fps} кадр/с, crf {a.crf})', flush=True)
    subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-stats', '-i', str(a.src / 'clip.mp4'),
                    '-an', '-vf', f'scale=-2:{a.height},fps={a.fps}', '-c:v', 'libx264', '-crf', str(a.crf), '-preset', 'slow',
                    '-g', '30', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(a.out / 'clip.mp4')], check=True)

    shutil.copy2(a.src / 'data.js', a.out / 'data.js')
    wanted = a.src / 'wanted.js'
    if wanted.exists():
        shutil.copy2(wanted, a.out / 'wanted.js')
    shutil.rmtree(a.out / 'crops', ignore_errors=True)
    shutil.copytree(a.src / 'crops', a.out / 'crops')
    write_page(a.out)

    size = sum(f.stat().st_size for f in a.out.rglob('*') if f.is_file())
    print(f'готово: {a.out} ({size / 1e6:.1f} МБ). Локально: python tools/serve.py . 8781 -> /{a.out.as_posix()}/')


if __name__ == '__main__':
    main()
