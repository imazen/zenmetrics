//! The generic surface honors the zenmetrics-abstractions contract.

#[test]
fn mdctpsnr_conforms() {
    zenmetrics_abstractions::conformance::assert_conforms_rgb8(&mut mdctpsnr::MdctPsnr, 64, 48);
}
