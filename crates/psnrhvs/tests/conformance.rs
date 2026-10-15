//! The generic surface honors the zenmetrics-abstractions contract.

#[test]
fn psnrhvs_conforms() {
    zenmetrics_abstractions::conformance::assert_conforms_rgb8(&mut psnrhvs::Psnrhvs, 64, 56);
}
