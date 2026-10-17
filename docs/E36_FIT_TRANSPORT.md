# E36 fit transport

The `e36` profile of `scripts/jobsys/pack_fit_program.py` accepts an independently
hashed public Python/JSON source inventory and one E36 qualification package.
It includes the existing executor, checkpoint inspector and harvest owner, plus
the private `e36_fit_contract.py` adapter. It does not declare or run fits.

E36 uses three matched plain-N arms. Its package binds each arm to the same
feature-only TV pool, actual trainer executable, protocol source and base-family
mask. Each route retains its original seeds, numeric group weights, sampling,
120 epochs, 50,000 pairs per epoch and final epoch 119. Explicit smoke budgets
are refused. Wall caps are read from the matching arm's package; they are limits,
not measured runtime predictions.

Only the registered family mask may derive a base table. Before fitting, its
selected row indices, ordered row keys, key-file hash and complete inherited
manifest are projected without target reads. Inside fit-cell-exec the existing
`write_curated` owner copies unchanged targets and features. The harvest adapter
checks that exact derivation and zero changed targets, then the existing generic
checkpoint gate checks the resulting table hashes, row counts, unchanged weights,
feature declarations, seeds and final epoch. It also checks the checkpoint's
actual E36 TV configuration and executable/protocol identities. Other fit schemas
retain their existing admission rules.

`just e36-fit-contract-check` exercises invented metadata, malformed derivations
and unchanged legacy admission. It performs no training or real label reads.
