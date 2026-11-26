"""Одна команда: відео -> машини й номери -> плеєр у браузері.

  python run.py my_video.mp4                              # усе відео
  python run.py bus-driving.webm --start 4:50 --duration 60   # лише хвилину
  python run.py my_video.mp4 --limit 10                   # перші 10 с: швидко перевірити, що все працює
  python run.py my_video.mp4 --view                       # лише відкрити вже оброблене

Що робить:
  1. ffmpeg вирізає потрібний шматок і перекодовує в H.264 (так браузер відтворює і швидко перемотує);
  2. tools/process_clip.py знаходить машини, читає номери, готує дані для плеєра (прогрес видно в консолі);
  3. (--wanted) tools/wanted_check.py перевіряє прочитані номери в базі розшуку МВС;
  4. запускає локальний сервер і відкриває плеєр у браузері.

Усе зберігається в runs/<назва>/: clip.mp4, data.js, crops/, pass.json (кеш проходу), progress.log, index.html.
Невідомі цьому скрипту параметри передаються в tools/process_clip.py як є (наприклад, --min-area 12000).
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tools'))


def seconds(s):
    """'290', '290.5', '4:50' або '1:04:50' -> секунди."""
    sec = 0.0
    for part in s.split(':'):
        sec = sec * 60 + float(part)
    return sec


def run_name(video, start, duration):
    name = re.sub(r'[^\w.-]+', '-', Path(video).stem).strip('-') or 'video'
    if start or duration:
        end = f'{start + duration:g}' if duration else 'end'
        name += f'_{start:g}-{end}'
    return name


def cut(video, out, start, duration):
    """Шматок відео -> H.264 з частими ключовими кадрами (клік по машині перемотує швидко)."""
    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-stats']
    if start:
        cmd += ['-ss', f'{start:g}']
    cmd += ['-i', str(video)]
    if duration:
        cmd += ['-t', f'{duration:g}']
    cmd += ['-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-g', '60', '-pix_fmt', 'yuv420p', str(out)]
    print(f'[1/3] ffmpeg: {video} -> {out}', flush=True)
    subprocess.run(cmd, check=True)


def write_page(out_dir):
    """Сторінка плеєра для папки з даними: розмітка з player/index.html, стилі й скрипт спільні (з player/)."""
    html = (ROOT / 'player' / 'index.html').read_text(encoding='utf-8')
    rel = Path(os.path.relpath(ROOT / 'player', out_dir)).as_posix()  # runs/<назва> -> ../../player
    for old, new in (('href="style.css"', f'href="{rel}/style.css"'),
                     ('src="app.js"', f'src="{rel}/app.js"')):
        assert html.count(old) == 1, f'у player/index.html не знайдено {old}'
        html = html.replace(old, new)
    (out_dir / 'index.html').write_text(html, encoding='utf-8')


def tail(path, proc, start_at=0):
    """Поки процес працює, друкує нові рядки progress.log."""
    pos = start_at
    while True:
        done = proc.poll() is not None
        if path.exists():
            with open(path, encoding='utf-8', errors='replace') as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
            for line in chunk.splitlines():
                print('      ' + line, flush=True)
        if done:
            return proc.returncode
        time.sleep(1)


def process(a, out_dir, clip, extra):
    log = out_dir / 'progress.log'
    cmd = [sys.executable, str(ROOT / 'tools' / 'process_clip.py'), str(clip), str(out_dir),
           '--offset', f'{a.start:g}', '--original', Path(a.video).name, '--cache', str(out_dir / 'pass.json')]
    if a.limit:
        cmd += ['--limit', f'{a.limit:g}']
    if a.reuse:
        cmd.append('--reuse')
    cmd += extra
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    if a.gpu_plates:
        env['PLATES_DEVICE'] = 'gpu'
    start_at = log.stat().st_size if a.reuse and log.exists() else 0
    if not a.reuse:
        log.unlink(missing_ok=True)
    print(f'[2/3] машини й номери (прогрес також у {log})', flush=True)
    # усе потрібне process_clip.py пише в progress.log, його й показуємо; помилки йдуть у stderr як є
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.DEVNULL)
    try:
        code = tail(log, proc, start_at)
    except KeyboardInterrupt:
        proc.terminate()
        raise
    if code:
        sys.exit(f'tools/process_clip.py завершився з помилкою (код {code}), подробиці вище і в {log}')


def free_port(port):
    import socket
    for p in range(port, port + 20):
        with socket.socket() as s:
            if s.connect_ex(('127.0.0.1', p)):
                return p
    sys.exit(f'порти {port}–{port + 19} зайняті, задайте інший через --port')


def serve(out_dir, port, open_browser):
    from serve import RangeHandler  # tools/serve.py: статичний сервер з HTTP Range (без нього відео не перемотується)
    port = free_port(port)
    url = f'http://127.0.0.1:{port}/' + quote(out_dir.relative_to(ROOT).as_posix()) + '/'
    httpd = ThreadingHTTPServer(('127.0.0.1', port), partial(RangeHandler, directory=str(ROOT)))
    print(f'[3/3] плеєр: {url}   (Ctrl+C — зупинити сервер)', flush=True)
    if open_browser:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('сервер зупинено')


def main():
    for stream in (sys.stdout, sys.stderr):  # інакше на Windows українські повідомлення можуть зламатись
        stream.reconfigure(encoding='utf-8', errors='replace')
    p = argparse.ArgumentParser(description='Відео -> машини й номери -> плеєр у браузері.',
                                epilog='Невідомі параметри передаються в tools/process_clip.py.')
    p.add_argument('video', help='будь-яке відео, яке читає ffmpeg (mp4, webm, mkv, mov…)')
    p.add_argument('--start', type=seconds, default=0, help='з якого моменту, напр. 4:50 або 290')
    p.add_argument('--duration', type=seconds, default=0, help='скільки секунд (за замовчуванням до кінця)')
    p.add_argument('--name', help='назва папки в runs/ (за замовчуванням з назви відео)')
    p.add_argument('--limit', type=float, default=0, help='обробити лише перші N секунд шматка')
    p.add_argument('--gpu-plates', action='store_true',
                   help='читати номери на відеокарті (потрібен onnxruntime-gpu у tools/.ort_gpu, див. README)')
    p.add_argument('--wanted', action='store_true', help='перевірити прочитані номери в базі розшуку МВС')
    p.add_argument('--reuse', action='store_true', help='не проганяти відео заново, взяти прохід з кешу')
    p.add_argument('--view', action='store_true', help='нічого не обробляти, лише відкрити плеєр')
    p.add_argument('--port', type=int, default=8781)
    p.add_argument('--no-serve', action='store_true', help='не запускати сервер після обробки')
    p.add_argument('--no-open', action='store_true', help='не відкривати браузер')
    a, extra = p.parse_known_args()

    out_dir = ROOT / 'runs' / (a.name or run_name(a.video, a.start, a.duration))
    clip = out_dir / 'clip.mp4'
    if a.view:
        if not (out_dir / 'data.js').exists():
            done = sorted(d.name for d in (ROOT / 'runs').glob('*') if (d / 'data.js').exists())
            sys.exit(f'{out_dir} ще не оброблено: запустіть без --view (з тими самими --start/--duration)'
                     + (f'. Уже оброблено: {", ".join(done)} (вибрати: --name)' if done else ''))
        return serve(out_dir, a.port, not a.no_open)

    if not Path(a.video).is_file():
        sys.exit(f'відео не знайдено: {a.video}')
    if not shutil.which('ffmpeg'):
        sys.exit('потрібен ffmpeg у PATH: https://ffmpeg.org/download.html (Windows: winget install ffmpeg)')
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'результат буде в {out_dir}', flush=True)

    if a.reuse and clip.exists():
        print('[1/3] кліп уже є, ffmpeg пропущено')
    else:
        # з --limit вирізаємо лише те, що буде оброблено: швидка перевірка не перекодовує все відео
        cut(a.video, clip, a.start, min(a.duration or a.limit, a.limit or a.duration))
    write_page(out_dir)
    process(a, out_dir, clip, extra)
    if a.wanted:
        print('      перевірка номерів у базі розшуку (запити йдуть повільно, з паузою)', flush=True)
        subprocess.run([sys.executable, str(ROOT / 'tools' / 'wanted_check.py'), str(out_dir / 'data.js')],
                       cwd=ROOT, env=dict(os.environ, PYTHONIOENCODING='utf-8'), check=True)
    if not a.no_serve:
        serve(out_dir, a.port, not a.no_open)


if __name__ == '__main__':
    main()
