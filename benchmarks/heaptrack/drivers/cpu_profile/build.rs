fn main() {
    println!("cargo::rustc-check-cfg=cfg(iqa_compete_cached)");
    println!("cargo::rustc-check-cfg=cfg(iqa_compete_ingress)");
}
