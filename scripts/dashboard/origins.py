"""Reject third-party resource loads; citation links remain optional navigation."""
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import urlsplit


def local(url):
    url=url.strip()
    return not url or url.startswith(('data:', 'blob:', '#')) or (not urlsplit(url).scheme and not url.startswith('//'))


class Resources(HTMLParser):
    def __init__(self):super().__init__();self.external=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs);urls=[]
        for k in ('src','poster','data','action','formaction'):
            if k in a:urls.append(a[k])
        if tag=='link' and a.get('rel') not in ('canonical','alternate'):urls.append(a.get('href',''))
        if tag=='base':urls.append(a.get('href',''))
        if 'srcset' in a:urls.extend(part.strip().split()[0] for part in a['srcset'].split(',') if part.strip())
        if tag=='meta' and a.get('http-equiv','').lower()=='refresh':
            urls.extend(re.findall(r'(?i)url\s*=\s*(.+)',a.get('content','')))
        for url in urls:
            if not local(url):self.external.append(url)


def check(directory):
    for path in sorted(directory.rglob('*')):
        if path.suffix not in ('.html','.css','.js','.mjs','.svg'):continue
        text=path.read_text()
        if path.suffix in ('.html','.svg'):
            parser=Resources();parser.feed(text)
            if parser.external:raise ValueError(f'external origin resource in {path.name}: {parser.external}')
        for url in re.findall(r'''(?:url\(\s*["']?|@import\s+["'])([^\s\)"']+)''',text):
            if not local(url):raise ValueError(f'external origin CSS in {path.name}: {url}')
        if path.suffix in ('.js','.mjs'):
            for url in re.findall(r'''(?:fetch\s*\(|import\s*\(|from\s+|importScripts\s*\()\s*["']([^"']+)''',text):
                if not local(url):raise ValueError(f'external origin script in {path.name}: {url}')
