//! Unit tests for the `common::const_str` byte-loop helpers
//! introduced in tick 584. The helpers replace inline byte-loops
//! across `column_name.rs`, `goldens_metadata.rs`, and
//! `lib_reexports.rs` — pinning their behavior here keeps the
//! shared module honest as call-site count grows.

#[path = "common/mod.rs"]
mod common;
use common::const_str;

// Compile-time pins on the helpers themselves. Each helper is
// `const fn` so we can exercise representative cases at compile
// time — failure aborts the build with the assert! message.

// starts_with: positive cases
const _: () = assert!(const_str::starts_with(b"hello world", b"hello"));
const _: () = assert!(const_str::starts_with(b"hello", b"hello"));
const _: () = assert!(const_str::starts_with(b"any", b"")); // empty prefix trivially matches
// starts_with: negative cases
const _: () = assert!(!const_str::starts_with(b"hello world", b"world"));
const _: () = assert!(!const_str::starts_with(b"hi", b"hello")); // prefix longer than haystack

// ends_with: positive cases
const _: () = assert!(const_str::ends_with(b"hello world", b"world"));
const _: () = assert!(const_str::ends_with(b"hello", b"hello"));
const _: () = assert!(const_str::ends_with(b"any", b"")); // empty suffix trivially matches
// ends_with: negative cases
const _: () = assert!(!const_str::ends_with(b"hello world", b"hello"));
const _: () = assert!(!const_str::ends_with(b"hi", b"hello"));

// contains: positive cases
const _: () = assert!(const_str::contains(b"hello world", b"lo wo"));
const _: () = assert!(const_str::contains(b"hello", b"hello"));
const _: () = assert!(const_str::contains(b"hello", b"")); // empty needle trivially found
const _: () = assert!(const_str::contains(b"abcdef", b"a")); // at start
const _: () = assert!(const_str::contains(b"abcdef", b"f")); // at end
// contains: negative cases
const _: () = assert!(!const_str::contains(b"hello world", b"xyz"));
const _: () = assert!(!const_str::contains(b"hi", b"hello"));

// Runtime test fns. Compile-time asserts above already guarantee
// correctness, but the runtime fns let `cargo test` runners see
// the test names and surface them in coverage reports / diffs.

#[test]
fn starts_with_positive() {
    assert!(const_str::starts_with(b"hello world", b"hello"));
    assert!(const_str::starts_with(b"hello", b"hello"));
    assert!(const_str::starts_with(b"any", b""));
}

#[test]
fn starts_with_negative() {
    assert!(!const_str::starts_with(b"hello world", b"world"));
    assert!(!const_str::starts_with(b"hi", b"hello"));
}

#[test]
fn ends_with_positive() {
    assert!(const_str::ends_with(b"hello world", b"world"));
    assert!(const_str::ends_with(b"hello", b"hello"));
    assert!(const_str::ends_with(b"any", b""));
}

#[test]
fn ends_with_negative() {
    assert!(!const_str::ends_with(b"hello world", b"hello"));
    assert!(!const_str::ends_with(b"hi", b"hello"));
}

#[test]
fn contains_positive() {
    assert!(const_str::contains(b"hello world", b"lo wo"));
    assert!(const_str::contains(b"hello", b"hello"));
    assert!(const_str::contains(b"hello", b""));
    assert!(const_str::contains(b"abcdef", b"a"));
    assert!(const_str::contains(b"abcdef", b"f"));
}

#[test]
fn contains_negative() {
    assert!(!const_str::contains(b"hello world", b"xyz"));
    assert!(!const_str::contains(b"hi", b"hello"));
}
