"""Локальний сервер для плеєрів з підтримкою HTTP Range (без нього відео не перемотується).

  python tools/serve.py . 8781   -> http://127.0.0.1:8781/demo/ (демо), /runs/<назва>/ (результати run.py)
"""
import os
import re
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class RangeHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        rng = self.headers.get('Range')
        path = self.translate_path(self.path)
        if not rng or os.path.isdir(path) or not os.path.exists(path):
            return super().send_head()
        m = re.match(r'bytes=(\d*)-(\d*)', rng)
        size = os.path.getsize(path)
        start = int(m.group(1)) if m.group(1) else 0
        end = int(m.group(2)) if m.group(2) else size - 1
        end = min(end, size - 1)
        f = open(path, 'rb')
        f.seek(start)
        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(path))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length', str(end - start + 1))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self._remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        rem = getattr(self, '_remaining', None)
        if rem is None:
            return super().copyfile(source, outputfile)
        while rem > 0:
            buf = source.read(min(1 << 16, rem))
            if not buf:
                break
            try:
                outputfile.write(buf)
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                break
            rem -= len(buf)

    def end_headers(self):
        if not self.headers.get('Range'):
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def log_message(self, *a):
        pass


if __name__ == '__main__':
    d, port = sys.argv[1], int(sys.argv[2])
    ThreadingHTTPServer(('127.0.0.1', port), partial(RangeHandler, directory=d)).serve_forever()
