# Origin tables for PYIQA_UPSTREAM_LICENSES_2026-10-15.md

- `deep_origins.tsv`: one row per pyiqa neural model and shared backbone: upstream repo,
  code licence as read, licence URL pinned to the commit read, weights host and stated
  terms, where pyiqa's weights come from, verdict for a zenmetrics port.
- `classical_origins.tsv`: one row per classical metric or helper: cited origins, the
  operative licence sentences, data files (matched by sha256) and their terms, the
  zenmetrics crate if one exists, verdict.

Collected 2026-10-15 against pyiqa `09796cfe80b0`. Licence texts were read from the
pinned URLs; MATLAB release zips were downloaded and their headers read.
