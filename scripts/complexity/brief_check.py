#!/usr/bin/env python3
"""Browser check for the metric-cost dashboard (brief_page.py output).

Loads every tab at two window sizes, fails on any console error or page error, checks that the page
itself does not scroll on desktop sizes, clicks every control button on every tab (each click must
change the drawn output, except buttons that are already selected or disabled), hovers each chart and
opens the detail drawer. Writes one screenshot per tab and size to --shots.

    uv run --with playwright==1.58.0 scripts/complexity/brief_check.py SITE_DIR --shots DIR [--chrome PATH]
"""
import argparse, sys
from pathlib import Path
from playwright.sync_api import sync_playwright

TABS = ['overview', 'speed', 'scaling', 'cost', 'repro', 'models', 'data', 'method']
SIZES = [(1920, 1080), (1280, 720)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('site', type=Path)
    ap.add_argument('--shots', type=Path, required=True)
    ap.add_argument('--chrome', help='Chromium executable (default: Playwright\'s own)')
    a = ap.parse_args()
    a.shots.mkdir(parents=True, exist_ok=True)
    url = (a.site / 'index.html').resolve().as_uri()
    problems = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=a.chrome) if a.chrome else pw.chromium.launch()
        for w, hgt in SIZES:
            page = browser.new_page(viewport={'width': w, 'height': hgt})
            page.on('console', lambda m: m.type == 'error' and problems.append(f'console {w}x{hgt}: {m.text}'))
            page.on('pageerror', lambda e: problems.append(f'pageerror {w}x{hgt}: {e}'))
            for tab in TABS:
                page.goto(f'{url}#{tab}')
                page.wait_for_timeout(150)
                scroll = page.evaluate('document.scrollingElement.scrollHeight - innerHeight')
                if scroll > 2:
                    problems.append(f'{tab} {w}x{hgt}: page scrolls by {scroll}px')
                page.screenshot(path=str(a.shots / f'{tab}_{w}x{hgt}.png'))
                if w != SIZES[0][0]:
                    continue
                svg = page.locator('.plot svg')
                if svg.count():
                    box = svg.bounding_box()
                    for fx, fy in ((0.3, 0.3), (0.6, 0.5), (0.25, 0.7)):
                        page.mouse.move(box['x'] + box['width'] * fx, box['y'] + box['height'] * fy)
                        page.wait_for_timeout(30)
                buttons = page.locator('.rail button[data-k]')
                for i in range(buttons.count()):
                    b = buttons.nth(i)
                    if b.get_attribute('aria-pressed') == 'true' or b.is_disabled():
                        continue
                    before = page.evaluate("document.querySelector('.stage') ? document.querySelector('.stage').innerHTML.length + ':' + location.hash : ''")
                    label = b.inner_text()
                    b.click()
                    page.wait_for_timeout(40)
                    after = page.evaluate("document.querySelector('.stage') ? document.querySelector('.stage').innerHTML.length + ':' + location.hash : ''")
                    if before == after:
                        problems.append(f'{tab}: button "{label}" changed nothing')
                page.goto(f'{url}#{tab}')
                page.wait_for_timeout(100)
                if tab in ('cost', 'scaling'):  # wheel zoom and drag pan must each redraw the chart
                    plot = page.locator('.plot'); box = plot.bounding_box()
                    cx, cy = box['x'] + box['width'] * 0.5, box['y'] + box['height'] * 0.5
                    html = lambda: page.evaluate("document.querySelector('.plot').innerHTML")
                    before = html(); page.mouse.move(cx, cy); page.mouse.wheel(0, -300); page.wait_for_timeout(80)
                    if html() == before:
                        problems.append(f'{tab}: wheel zoom changed nothing')
                    before = html(); page.mouse.down(); page.mouse.move(cx + 60, cy + 40, steps=4); page.mouse.up(); page.wait_for_timeout(80)
                    if html() == before:
                        problems.append(f'{tab}: drag pan changed nothing')
                    page.mouse.dblclick(cx, cy); page.wait_for_timeout(60)
                    page.screenshot(path=str(a.shots / f'{tab}_zoomed_reset.png'))
                if tab in ('speed', 'cost'):
                    target = page.locator('.plot .g-pt' if tab == 'cost' else '.plot svg').first
                    box = target.bounding_box()
                    fx, fy = (0.5, 0.5) if tab == 'cost' else (0.3, 0.2)
                    page.mouse.click(box['x'] + box['width'] * fx, box['y'] + box['height'] * fy)
                    page.wait_for_timeout(80)
                    if page.locator('#drawer').is_hidden():
                        problems.append(f'{tab}: clicking the chart did not open the detail drawer')
                    else:
                        page.screenshot(path=str(a.shots / f'{tab}_drawer.png'))
                        page.keyboard.press('Escape')
            page.close()
        page = browser.new_page(viewport={'width': 1536, 'height': 960})
        page.on('pageerror', lambda e: problems.append(f'pageerror print: {e}'))
        page.goto(url + '?print=1')
        page.wait_for_timeout(300)
        if page.locator('section.page').count() != len(TABS):
            problems.append('print mode did not lay out every tab')
        browser.close()
    for p in problems:
        print('FAIL', p)
    print(f'{len(problems)} problem(s); screenshots in {a.shots}')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
