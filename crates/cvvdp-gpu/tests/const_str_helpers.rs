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

// bytes_eq: positive cases
const _: () = assert!(const_str::bytes_eq(b"hello", b"hello"));
const _: () = assert!(const_str::bytes_eq(b"", b"")); // empty == empty
// bytes_eq: negative cases
const _: () = assert!(!const_str::bytes_eq(b"hello", b"helloo")); // different length
const _: () = assert!(!const_str::bytes_eq(b"hello", b"hellx")); // same length, different content
const _: () = assert!(!const_str::bytes_eq(b"hello", b"")); // non-empty vs empty

// count (tick 600): exact match cases
const _: () = assert!(const_str::count(b"abcabc", b"ab") == 2);
const _: () = assert!(const_str::count(b"aaaa", b"aa") == 2); // non-overlapping
const _: () = assert!(const_str::count(b"hello world", b"o") == 2);
const _: () = assert!(const_str::count(b"hello", b"hello") == 1); // exact
// count: edge cases
const _: () = assert!(const_str::count(b"hello", b"xyz") == 0); // not found
const _: () = assert!(const_str::count(b"hi", b"hello") == 0); // needle longer than haystack
const _: () = assert!(const_str::count(b"hello", b"") == 0); // empty needle → 0 (avoids infinite loop)
const _: () = assert!(const_str::count(b"", b"hello") == 0); // empty haystack

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

#[test]
fn bytes_eq_positive() {
    assert!(const_str::bytes_eq(b"hello", b"hello"));
    assert!(const_str::bytes_eq(b"", b""));
}

#[test]
fn bytes_eq_negative() {
    assert!(!const_str::bytes_eq(b"hello", b"helloo"));
    assert!(!const_str::bytes_eq(b"hello", b"hellx"));
    assert!(!const_str::bytes_eq(b"hello", b""));
}

#[test]
fn count_positive() {
    assert_eq!(const_str::count(b"abcabc", b"ab"), 2);
    assert_eq!(const_str::count(b"aaaa", b"aa"), 2); // non-overlapping
    assert_eq!(const_str::count(b"hello world", b"o"), 2);
    assert_eq!(const_str::count(b"hello", b"hello"), 1);
}

#[test]
fn count_edge_cases() {
    assert_eq!(const_str::count(b"hello", b"xyz"), 0);
    assert_eq!(const_str::count(b"hi", b"hello"), 0);
    assert_eq!(const_str::count(b"hello", b""), 0); // empty needle → 0
    assert_eq!(const_str::count(b"", b"hello"), 0);
}
