//! The generic surface honors the zenmetrics-abstractions contract.

#[test]
fn vif_conforms() {
    zenmetrics_abstractions::conformance::assert_conforms_rgb8(&mut vif::Vif, 128, 112);
}

#[test]
fn vifvec_conforms() {
    zenmetrics_abstractions::conformance::assert_conforms_rgb8(&mut vif::Vifvec, 128, 112);
}
