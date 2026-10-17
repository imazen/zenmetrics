#!/usr/bin/env python3
"""Measure the viewer's first load: bytes fetched and time to the demo's first score.

Each run uses a fresh browser context (empty HTTP cache). Times are from
navigation start to (a) the worker's ready message (as polled) and (b) the
demo's first score, as the page timestamps it (`data-first-score-ms`; the
older viewer has no stamp, so its polled time is used). Bytes are the
transfer sizes of every response whose request started before the first
score. With --mbps the page runs under Chromium network emulation at
that downlink rate (and 20 ms latency), so module size shows up in the time.
"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import statistics
import threading
from playwright.sync_api import sync_playwright


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        for name, value in (("Cross-Origin-Opener-Policy", "same-origin"),
                            ("Cross-Origin-Embedder-Policy", "require-corp"),
                            ("Cross-Origin-Resource-Policy", "cross-origin")):
            self.send_header(name, value)
        super().end_headers()

    def log_message(self, *_):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('site', type=Path)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--mbps', type=float, help='Emulated downlink in Mbit/s')
    parser.add_argument('--port', type=int, default=3341)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, directory=str(args.site.resolve())))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{args.port}/'
    runs = []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            for _ in range(args.runs):
                context = browser.new_context()
                page = context.new_page()
                fetched = {}

                def finished(request):
                    sizes = request.sizes()
                    fetched[request.url] = (request.timing['startTime'],
                                            sizes['responseBodySize'] + sizes['responseHeadersSize'])

                page.on('requestfinished', finished)
                if args.mbps:
                    cdp = context.new_cdp_session(page)
                    cdp.send('Network.enable')
                    rate = args.mbps * 1e6 / 8
                    cdp.send('Network.emulateNetworkConditions', {
                        'offline': False, 'latency': 20,
                        'downloadThroughput': rate, 'uploadThroughput': rate})
                page.goto(url)
                page.wait_for_function("document.getElementById('startup-timing').textContent.includes('Rayon pool')", timeout=300000)
                ready = page.evaluate("performance.now()")
                page.wait_for_function("/^-?[0-9]/.test(document.getElementById('score').textContent)", timeout=300000)
                polled = page.evaluate("performance.now()")
                stamp = page.evaluate("document.body.dataset.firstScoreMs")
                scored = float(stamp) if stamp else polled
                origin = page.evaluate("performance.timeOrigin")
                page.wait_for_timeout(500)
                before = {u: n for u, (start, n) in fetched.items() if start <= origin + scored}
                runs.append({'ready_ms': ready, 'first_score_ms': scored, 'polled_first_score_ms': polled, 'stamped': bool(stamp),
                             'bytes_before_first_score': sum(before.values()),
                             'files': {u.split('/')[-1]: n for u, n in sorted(before.items())}})
                context.close()
            browser.close()
    finally:
        server.shutdown()
    summary = {'site': str(args.site), 'mbps': args.mbps, 'runs': runs,
               'median_first_score_ms': statistics.median(r['first_score_ms'] for r in runs),
               'median_polled_first_score_ms': statistics.median(r['polled_first_score_ms'] for r in runs),
               'median_ready_ms': statistics.median(r['ready_ms'] for r in runs)}
    text = json.dumps(summary, indent=2)
    if args.out:
        args.out.write_text(text + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'runs'}))
    print('bytes before first score:', [r['bytes_before_first_score'] for r in runs])
    print('first score ms:', [round(r['first_score_ms'], 1) for r in runs])


if __name__ == '__main__':
    main()
