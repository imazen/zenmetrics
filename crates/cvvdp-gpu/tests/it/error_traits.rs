//! Pin the trait-side contracts on `cvvdp_gpu::Error`:
//!
//! - `#[derive(Clone)]` — error values are cloneable, so callers
//!   can stash one in a sticky state across retries / log it
//!   without consuming the original.
//! - `impl std::error::Error` — required so callers can `?`-bubble
//!   through their own error type's `From<dyn Error>` and
//!   `anyhow::Error::from(cvvdp_gpu::Error)` works.
//! - `source()` returns `None` for every variant — these are leaf
//!   errors, no nested cause chain to walk.
//!
//! The existing `error_display_messages_are_actionable` test
//! (tick 282) covers the `Display` impl content; this file
//! covers the trait *implementations*. A refactor that swaps the
//! `Clone` derive for a manual impl that forgets a field, or
//! introduces a `Box<dyn Error>` inner cause that source() should
//! expose, surfaces here.

#![allow(clippy::excessive_precision)]

use cvvdp_gpu::Error;

#[test]
fn error_implements_std_error_trait() {
    // Compile-time check: Error must implement std::error::Error.
    // The `_e: &dyn std::error::Error` coercion fails to compile
    // if the trait isn't implemented; the runtime body is just
    // there to suppress unused-var lints.
    fn requires_std_error(_e: &dyn std::error::Error) {}
    let e = Error::NoCachedReference;
    requires_std_error(&e);
}

#[test]
fn error_clone_preserves_variant_and_payload() {
    // Exercise Clone on every variant. The DimensionMismatch
    // variant carries fields — a refactor that swaps the derive
    // for a manual impl could forget to copy the `expected` or
    // `got` field and silently round-trip with default values.
    let dm = Error::DimensionMismatch {
        expected: 12_288,
        got: 3_072,
    };
    let dm_clone = dm.clone();
    match dm_clone {
        Error::DimensionMismatch { expected, got } => {
            assert_eq!(expected, 12_288, "DimensionMismatch.expected lost in clone");
            assert_eq!(got, 3_072, "DimensionMismatch.got lost in clone");
        }
        other => panic!("clone changed variant: DimensionMismatch -> {other:?}"),
    }

    // Zero-payload variants — clone is a copy-by-value, just
    // confirm the variant survives.
    for original in [
        Error::NoCachedReference,
        Error::NoWarmReference,
        Error::InvalidImageSize,
    ] {
        let cloned = original.clone();
        assert_eq!(
            format!("{original:?}"),
            format!("{cloned:?}"),
            "clone changed variant Debug",
        );
    }
}

#[test]
fn error_source_returns_none_for_all_variants() {
    // cvvdp_gpu::Error variants are leaf errors — they don't wrap
    // a deeper cause. `source()` MUST return None for all of
    // them so a caller walking `Error::source()` chains doesn't
    // accidentally get into an infinite loop or skip past the
    // real failure point.
    //
    // If a future variant wraps a backend error (e.g. cubecl read
    // failure), source() should return Some(&inner). When that
    // happens this test fails loudly + the maintainer documents
    // the new wrap-and-expose contract.
    use std::error::Error as StdError;
    let variants = [
        Error::DimensionMismatch {
            expected: 0,
            got: 0,
        },
        Error::NoCachedReference,
        Error::NoWarmReference,
        Error::InvalidImageSize,
    ];
    for e in variants {
        assert!(
            e.source().is_none(),
            "Error::{e:?}.source() must return None (leaf error); got Some",
        );
    }
}

#[test]
fn error_debug_format_includes_variant_name() {
    // Pin that the derived Debug impl produces the variant name
    // verbatim — callers (`{err:?}` in panic messages, log lines)
    // need it to localise the failure mode. A refactor to a manual
    // Debug impl that prints something other than the variant name
    // would silently degrade error reports.
    assert!(
        format!(
            "{:?}",
            Error::DimensionMismatch {
                expected: 12_288,
                got: 3_072
            }
        )
        .contains("DimensionMismatch"),
        "DimensionMismatch Debug must include the variant name",
    );
    assert!(
        format!("{:?}", Error::NoCachedReference).contains("NoCachedReference"),
        "NoCachedReference Debug must include the variant name",
    );
    assert!(
        format!("{:?}", Error::NoWarmReference).contains("NoWarmReference"),
        "NoWarmReference Debug must include the variant name",
    );
    assert!(
        format!("{:?}", Error::InvalidImageSize).contains("InvalidImageSize"),
        "InvalidImageSize Debug must include the variant name",
    );
}

#[test]
fn error_boxed_dyn_propagation_compiles_and_runs() {
    // Sanity test for the `?`-bubble path: `cvvdp_gpu::Error`
    // converts into `Box<dyn std::error::Error>` automatically.
    // This is the exact path callers like `zenmetrics-cli` use
    // when they propagate the error through their own
    // `Result<_, Box<dyn Error>>` return types.
    fn produces_boxed() -> Result<(), Box<dyn std::error::Error>> {
        Err(Error::NoWarmReference)?;
        Ok(())
    }
    let e = produces_boxed().expect_err("must produce error");
    let msg = e.to_string();
    assert!(
        msg.contains("warm_reference"),
        "boxed error Display lost the actionable hint: {msg:?}",
    );
}
