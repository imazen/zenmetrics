#!/usr/bin/env python3
"""Aggregate local coding-agent session timing per metric family (sanitized).

Runs on one host and reads that host's own agent records: Claude Code
(~/.claude/projects/*/*.jsonl), Codex (~/.codex/sessions/**/*.jsonl) and Devin
(~/.local/share/devin/cli/sessions.db, metadata and message timestamps only).
It prints aggregate counts only; no prompt, message, command or path text
leaves the host. Method and definitions: docs/IQA_COMPLEXITY.md, section
"Optimization effort".

  agent_effort.py --host <label> --families scripts/complexity/effort_families.json [--out FILE]

Definitions:
- Session in scope: its working directory is a zenmetrics checkout (any
  `zenmetrics` or `zenmetrics--*` directory).
- Event: one timestamped record (message or tool call) in the session.
- Active time: sum of gaps between consecutive events that are at most
  --gap seconds (default 300). Longer gaps count as idle. This is a stated
  definition of agent activity, not a measurement of human labor.
- Elapsed time: last event minus first event.
- A session touches a family when an edit-capable tool call (Bash, Write,
  Edit, MultiEdit, NotebookEdit, apply_patch, Devin tool calls) names one of
  the family's source paths. Session times are then reported per family but
  are not additive across families: a session that touches two families is
  counted under both and in `shared_sessions`.
- Concurrency: active minutes are also reported as the union over all
  in-scope sessions of this host, so overlapping lanes are not double counted.
"""
import argparse, datetime as dt, glob, json, os, re, sqlite3, sys
from pathlib import Path

EDIT_TOOLS = {'Bash', 'Write', 'Edit', 'MultiEdit', 'NotebookEdit', 'apply_patch', 'shell', 'exec_command'}
SCOPE = re.compile(r'/zenmetrics(?:--[\w.-]+)?(?:/|$)')


def ts(value):
    if isinstance(value, (int, float)):
        return float(value) / (1000.0 if value > 1e11 else 1.0)
    return dt.datetime.fromisoformat(str(value).replace('Z', '+00:00')).timestamp()


def claude_sessions(home):
    for f in glob.glob(f'{home}/.claude/projects/*/*.jsonl'):
        events, touch, cwd = [], [], None
        for line in open(f, errors='replace'):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            cwd = d.get('cwd') or cwd
            if 'timestamp' in d:
                events.append(ts(d['timestamp']))
            m = d.get('message')
            if isinstance(m, dict) and isinstance(m.get('content'), list):
                for c in m['content']:
                    if isinstance(c, dict) and c.get('type') == 'tool_use' and c.get('name') in EDIT_TOOLS:
                        touch.append(json.dumps(c.get('input', {})))
        yield 'claude', cwd, events, touch


def codex_sessions(home):
    for f in glob.glob(f'{home}/.codex/sessions/**/*.jsonl', recursive=True):
        events, touch, cwd = [], [], None
        for line in open(f, errors='replace'):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            p = d.get('payload') if isinstance(d.get('payload'), dict) else {}
            if d.get('type') == 'session_meta':
                cwd = p.get('cwd')
            if 'timestamp' in d:
                events.append(ts(d['timestamp']))
            if p.get('type') in ('custom_tool_call', 'function_call'):
                touch.append(str(p.get('input', '')) + str(p.get('arguments', '')))
        yield 'codex', cwd, events, touch


def devin_sessions(home):
    path = f'{home}/.local/share/devin/cli/sessions.db'
    if not os.path.exists(path):
        return
    con = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    for sid, cwd in con.execute('select id, working_directory from sessions'):
        if not SCOPE.search(cwd + '/'):
            continue
        events = [ts(t) for (t,) in con.execute('select created_at from message_nodes where session_id=?', (sid,))]
        touch = [j or '' for (j,) in con.execute('select tool_call_json from tool_call_state where session_id=?', (sid,))]
        yield 'devin', cwd, sorted(events), touch


def active(events, gap):
    events = sorted(events)
    return sum(b - a for a, b in zip(events, events[1:]) if b - a <= gap)


def intervals(events, gap):
    events = sorted(events)
    return [(a, b) for a, b in zip(events, events[1:]) if b - a <= gap]


def union_seconds(spans):
    total, end = 0.0, None
    for a, b in sorted(spans):
        if end is None or a > end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', required=True)
    ap.add_argument('--families', required=True)
    ap.add_argument('--home', default=str(Path.home()))
    ap.add_argument('--gap', type=float, default=300.0)
    ap.add_argument('--out')
    a = ap.parse_args()
    families = json.loads(Path(a.families).read_text())['families']
    patterns = {f['family']: re.compile('|'.join(f['path_regex'])) for f in families}
    per = {f: dict(sessions=0, shared_sessions=0, edit_calls=0, active_s=0.0, elapsed_s=0.0, first=None, last=None,
                   kinds={}) for f in patterns}
    host = dict(sessions=0, sessions_touching_any=0, spans=[], kinds={})
    for source in (claude_sessions, codex_sessions, devin_sessions):
        for kind, cwd, events, touch in source(a.home):
            if not cwd or not SCOPE.search(cwd + '/') or len(events) < 2:
                continue
            host['sessions'] += 1
            host['kinds'][kind] = host['kinds'].get(kind, 0) + 1
            host['spans'] += intervals(events, a.gap)
            hits = {f: sum(bool(p.search(t)) for t in touch) for f, p in patterns.items()}
            hits = {f: n for f, n in hits.items() if n}
            if hits:
                host['sessions_touching_any'] += 1
            for f, n in hits.items():
                r = per[f]
                r['sessions'] += 1
                r['shared_sessions'] += len(hits) > 1
                r['edit_calls'] += n
                r['active_s'] += active(events, a.gap)
                r['elapsed_s'] += max(events) - min(events)
                r['kinds'][kind] = r['kinds'].get(kind, 0) + 1
                day = lambda t: dt.datetime.fromtimestamp(t, dt.timezone.utc).date().isoformat()
                r['first'] = min(filter(None, [r['first'], day(min(events))]))
                r['last'] = max(filter(None, [r['last'], day(max(events))]))
    out = dict(schema=1, host=a.host, gap_seconds=a.gap,
               generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
               host_sessions_in_scope=host['sessions'], host_sessions_touching_a_family=host['sessions_touching_any'],
               host_sessions_by_agent=host['kinds'],
               host_union_active_hours=round(union_seconds(host['spans']) / 3600, 3),
               families=[dict(family=f, sessions=r['sessions'], shared_sessions=r['shared_sessions'],
                              edit_calls=r['edit_calls'], active_hours_nonadditive=round(r['active_s'] / 3600, 3),
                              elapsed_hours_nonadditive=round(r['elapsed_s'] / 3600, 3), first_day=r['first'],
                              last_day=r['last'], sessions_by_agent=r['kinds']) for f, r in per.items()])
    text = json.dumps(out, indent=1) + '\n'
    if a.out:
        Path(a.out).write_text(text)
    sys.stdout.write(text)


if __name__ == '__main__':
    main()
