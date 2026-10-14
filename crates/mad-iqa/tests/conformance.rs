//! The generic surface honors the zenmetrics-abstractions contract.

#[test]
fn mad_conforms() {
    zenmetrics_abstractions::conformance::assert_conforms_rgb8(&mut mad_iqa::Mad, 64, 64);
}
