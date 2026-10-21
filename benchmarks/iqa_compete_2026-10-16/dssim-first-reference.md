# DSSIM first-call lifetime adapter

This benchmark-only C adapter links the same retained author archive and header
as the full-call reference. Its first clock includes context/image construction
and comparison. Context and images remain alive until that clock stops, matching
the native CUDA scorer lifetime. Subsequent `SAMPLE` observations keep the full
context/image creation, comparison and release boundary. Scoring expressions and
the author library are unchanged; existing warm source contracts stay separate.

Build against a successful, explicitly hashed full-call build receipt:

```sh
just bench-iqa-canonical-dssim-first-build PARENT_RECEIPT NEW_OUTPUT PARENT_SHA256
```

The command serializes compilation under shared admission, verifies the retained
compiler/archive/header/source identities before and after linking, and records
its exact command and ELF hash. A build alone supplies no numerical acceptance.
Original/candidate scalar controls are required before admitting a new first-call
contract; `first_boundary_control.py` retains both COLD and subsequent samples.

```sh
just bench-iqa-canonical-dssim-first-boundary-check NEW_OUTPUT /absolute/compiler
just bench-iqa-canonical-first-boundary-control PLAN NEW_OUTPUT
```

The first check uses a synthetic C API and clock to verify object lifetime at
each clock boundary. It also checks that the unchanged full-call adapter fails
this distinct first-call contract. It executes no real metric and supplies no
accuracy or performance observation. The second command executes independently
bound original/candidate controls; it does not run a native comparison.
