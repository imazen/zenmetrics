fn main() {
    let models = std::env::var("CARGO_MANIFEST_DIR").unwrap() + "/../models";
    burn_onnx::ModelGen::new()
        .input(&format!("{models}/resnet50_224.onnx"))
        .input(&format!("{models}/vgg16lpips_256.onnx"))
        .out_dir("model/")
        .run_from_script();
}
