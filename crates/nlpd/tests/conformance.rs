//! The generic surface honors the zenmetrics-abstractions contract.

#[test]
fn nlpd_conforms() {
    zenmetrics_abstractions::conformance::assert_conforms_rgb8(&mut nlpd::Nlpd, 112, 100);
}
