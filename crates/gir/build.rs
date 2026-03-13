fn main() {
    #[cfg(feature = "onnx")]
    {
        let mut config = prost_build::Config::new();
        config.btree_map(["."]);
        config
            .compile_protos(&["proto/onnx.proto3"], &["proto/"])
            .expect("failed to compile ONNX proto");
    }
}
