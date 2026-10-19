# Fresh Butteraugli-max cohort compilation — no scoring

Plan724f7175 binds Rust base73c04e413fddc0e95e66b4147785910253772192,
Butteraugli791d08d725f27b5331ef7b96e73aad89ed895f45, and landed native
adapter62aaf7e94acaee2ccbe9b4f6c16f0ec3416a3559. Reviewed three Git blobs
and actual source/input/overlay hashes passed preflight. Locked offline
metadata and compilation retained Cargo.lock unchanged. The benchmark-only
Rust overlay selects butteraugli_max explicitly and declares rgb-input scope;
metric arithmetic is unchanged. Scope declaration is not measured equivalence.

Fresh binaries:

- Rust iqa-compete SHA256:
  677899211912b6c20eb13b57e29c65910cb581b15e38aeb680cb14a3d2cd8ba2
- Native iqa-jxl SHA256:
  de04cd3180d7ac40e2eac84464fe191ac5ee713201a0a96b393888fe691fb4e2

The native binary matches the prior control-build bytes, but was compiled
fresh (all181 Ninja steps) into a previously absent output directory. It was
not copied from the prior build. Neither old score rows nor old executables
are retroactively qualified by this new build receipt.

Rust command: cargo build --locked --offline --release -p cpu-profile
--features compete,metrics-parallel --bin iqa-compete. Native CMake uses the
landed adapter/CMakeLists, owned libjxl0.12 source, Release and Ninja; build
target iqa-jxl. Four build jobs under both existing ordered admission locks
and run-heavy --mem12G. All compiler/version commands, resolved features,
CMake compile/link commands, dependency files and lock lifecycle are retained.
No NR/NN feature, PyIQA or SSIM builder/inventory was added or consumed.

First preflight stopped before compilation because Ninja was absent from
PATH. Its failed receipt remains. The retained build identified the existing
Ninja under the local tool directory; adding that directory to PATH resolved
it without installing or modifying tools. Second attempt built successfully:
rc0,123s,peak-RSS0.87GiB,min-avail11796MiB,peak-load4.76. These are build
resources, not metric execution time or memory.

Collection retains881 native compiler dependency files,690 Rust build
artifacts, link commands/archive inventory, loader configuration and recursive
ELF closure for both binaries, controller interpreter and loaded standard
library extensions. Resolved owned source hashes, Python source/bytecode,
119 registry package archives matching the frozen Cargo.lock and327 Git
source inputs supplement the source/build binding. Registry package binding
uses archive checksums and actual dependency-file membership; global registry
source checkout bytes were not inspected. Dependency/library inventory is
not a claim that every listed archive contributed symbols to the final binary.
Collection rc0,7s,peak-RSS0.08GiB,min-avail13376MiB,peak-load3.06.

No scorer, warmup, parity, timing sample or acquisition ran. Protocol/scope
fields are source-bound but not observed from metric execution. Dynamic
loads beyond declared ELF roots remain unqualified. Independent exact cohort,
compiler/header/library/source and protocol/scope review is pending, followed
by unchanged numerical parity and actual scorer gate/ownership evidence.
Generic59/60red and diagnostic9d69 pending remain; statistical order,
selection,dependence and strict precision gates are unchanged.0/19 and
attribution_ready=false. Prior bounded known-handle/single-interruption guard
acceptance retains its unknown-handle and late-evidence limitations.
