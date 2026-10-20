# Frozen RTEN configured-thread review

Source/metadata review, 2026-10-20. No inference, build, tests or library
introspection was executed. Existing binaries, plans, results and bounds are
unchanged. [Exact inspected identities](source-evidence.json) accompany this
review.

The unresolved items are actual active utilization, total process thread counts,
canonical runtime cleanliness and exhaustive historical compiler-input identity.
The NN dependency rlibs were reclaimed: their recorded first-probe hashes remain
evidence, but cannot be independently rehashed now. These limits do not establish
a cap bug.

## Conclusion

The resolved dependency is **RTEN 0.27.0**, using **Rayon 1.12.0 / rayon-core
1.13.0**. `Model.run(..., None)` uses RTEN's process-wide pool, separate from
Rayon's default global pool. RTEN reads **RTEN_NUM_THREADS**, not
RAYON_NUM_THREADS, when that pool is first initialized. The retained warm wrapper
and first-evaluation comparator set RTEN_NUM_THREADS before model construction.
No concrete configured-cap mismatch was found at requested one/eight workers.

| Path | Requested 1 | Requested 8 |
| --- | --- | --- |
| PaQ2PiQ RTEN image body/head, `Model.run(..., None)` | RTEN pool configured for 1 | RTEN pool configured for `min(8, num_cpus::get())` |
| DeepDC / LPIPS paired trunks | Sequential image calls through RTEN default pool configured for 1 | Two explicit pools configured for 4 each; paired graph calls run concurrently |
| Native Rust preprocessing/head passes | Work-sized cap of 1 | Work-sized cap at most 8; sequential phases may use fewer |

The selected DeepDC/VGG confirmation cases request one; the eight-worker paired
entry describes the same source branch, not a newly executed control. The PaQ
confirmation case requests eight. This is a configured pool-size conclusion,
not a measured census or an eight-CPU saturation claim. It does not add a
process-wide hard thread limit.

## Source chain

The retained NN target's binary fingerprint links the stock CLI fingerprint,
which links `zenmetrics-nn-fdbcb6ab56d2bc9b`, which links
`rten-002e9e8193d1cdad`. That RTEN fingerprint and `rten-parallel-5d8065b2dcb9619e`
both select `rayon-ea07e479bc5dfaca`, which selects
`rayon-core-3be2740412d73375`. Selected dependency fingerprint values match
the retained child fingerprint bytes. The first-probe build receipt names those
same NN/CLI/Rayon rlib filenames and records their hashes.

Cargo.lock, retained metadata, NN build log and retained dep-info agree on
RTEN/rten-parallel 0.27.0, Rayon 1.12.0 and rayon-core 1.13.0. Only one package
version for each Rayon package occurs in the retained metadata/lock. The three
rayon-core fingerprint directories include build-script and run-build-script
records, not three library versions. No registry source directory was read.

The NN-target fingerprints are the relevant feature evidence: RTEN enables
`onnx_format`; rten-parallel, Rayon and rayon-core have empty feature lists;
zenmetrics-nn enables `default,ort,rten`. The earlier build/metadata.json resolver
lists zenmetrics-nn `default,rten` and Rayon `web_spin_lock`; it is not the NN
target's exact feature census. The retained commands/settings select engine
`rten` even though ORT is compiled into the NN library.

Exact-version source was obtained using `cargo read NAME==VERSION --path-only`.
The four retained registry archives match their Cargo.lock checksums. Each
dependency source file identified in source-evidence.json was compared byte for
byte with its archive member. This binds the inspected release source; it does
not retroactively prove the compiler consumed unmodified source in its original
build directory. Cargo fingerprints/dep-info are provenance witnesses, not
cryptographic source attestations.

## Execution and environment evidence

Line references below use the archive-matched release files, or the frozen
`build/work/zenmetrics` snapshot for native engine code.

- RTEN [model.rs](https://docs.rs/crate/rten/0.27.0/source/src/model.rs),
  lines480–492: missing options become defaults and are passed to Graph.run.
  [graph.rs](https://docs.rs/crate/rten/0.27.0/source/src/graph.rs),
  lines132–145,793–819: a missing custom pool selects RTEN's pool, and graph
  execution runs inside it.
- RTEN [threading.rs](https://docs.rs/crate/rten/0.27.0/source/src/threading.rs),
  lines76–92: OnceLock initializes from RTEN_NUM_THREADS, parsed as u32 and
  clamped to1..logical-CPU count. Missing/invalid values choose the physical-core
  default. Changing the environment after first initialization does not resize
  this pool. Lines21–36 create an explicit Rayon pool and install graph work in
  it. The retained wrappers set valid1/8 values before initialization.
- Rayon-core [lib.rs](https://docs.rs/crate/rayon-core/1.13.0/source/src/lib.rs),
  lines458–485,504–530: a positive explicit num_threads takes precedence over
  RAYON_NUM_THREADS and is documented as an upper bound on pool threads.
  [thread_pool/mod.rs](https://docs.rs/crate/rayon-core/1.13.0/source/src/thread_pool/mod.rs),
  lines19–36,137–142: install selects this pool's execution context, including
  nested Rayon parallel work. rten-parallel `src/par_iter.rs:206–219` delegates
  its parallel branch to Rayon; it creates no independent pool there.
- Frozen `crates/zenmetrics-nn/src/engine.rs:126–133,183–219` reads the native
  worker cap from RTEN_NUM_THREADS and constructs paired half pools.
  Lines239–246,315–334 pass those pools through RunOptions for paired graphs.
  Lines382–409 run PaQ image graphs with default options. Native host worker
  sizing is in `src/par.rs:13–18` and preprocessing dispatch in
  `src/preprocess.rs:9–54`; those phases are separate from the graph pool.
- `compare_first_evaluation.py:18–25` puts the requested count into
  RTEN/RAYON/OMP/MKL/OPENBLAS before dispatching the first probe. The probe
  `resident_nn_first.rs:27–49` constructs its global Rayon pool and prepares the
  model after process startup. Its case-specific environment overrides contain
  engine/model paths, not a different RTEN budget.
- `compare_canonical.py:64–82` supplies PaQ's case count in the outer environment.
  The selected PaQ shell worker calls `canonical_worker_env.py:18–20,28`, which
  copies RAYON_NUM_THREADS into RTEN_NUM_THREADS before exec of the original
  NN worker. The settings choose RTEN and the frozen PaQ weights.

## Important limits and fallback behavior

RTEN suppresses explicit pool construction failure (`threading.rs:30–36`);
`ThreadPool.run` then calls the closure directly (`:21–26`). Nested parallel work
can consequently use its ambient Rayon context. Pool creation success and
effective scheduling were not measured here. Do not describe this source result
as an independently observed exact worker census.

`RunOptions.thread_pool()` uses eager `unwrap_or(threading::thread_pool())`.
Even an explicit paired half-pool run can initialize RTEN's default pool, while
executing the graph in the selected half pool. Idle RTEN/default-Rayon/half-pool
threads can therefore coexist. Counting created threads is not evidence of
simultaneously active parallelism, and summing all pool sizes overstates the
configured graph-execution width in the successful-pool path.

The source resolves the earlier uncertainty about whether valid RTEN_NUM_THREADS
controls default RTEN execution. No additional scoring is needed for that
configured-cap conclusion. If actual utilization becomes a separate acceptance
question, root can admit a scheduler/CPU-stack trace around an unchanged worker
at each budget, retaining TID identities and scheduling intervals and separating
graph kernels from host/dispatch work. Thread creation or task-clock/elapsed
alone cannot establish an exact active-worker census. This review neither
launches that experiment nor relabels any existing timing.
