import Foundation
import MLX
import MLXLMCommon
import XCTest

/// mlx-lm writes `"group_size": null` for per-layer entries left at MLX's default
/// (Kimi K2.6's DQ3_K_M mixed quant does this for all 731 of its entries). The value
/// must resolve to MLX's per-mode default, not fail the whole config decode.
public class QuantizationDefaultsTests: XCTestCase {

    func testNullGroupSizeResolvesToModeDefault() throws {
        let json =
            """
            {
                "model_type": "kimi_k25",
                "quantization": {
                    "group_size": 32,
                    "bits": 4,
                    "mode": "affine",
                    "language_model.model.layers.3.mlp.switch_mlp.gate_proj": {
                        "group_size": null, "bits": 3, "mode": "affine"
                    },
                    "language_model.model.layers.3.mlp.switch_mlp.down_proj": {
                        "group_size": null, "bits": 5, "mode": "affine"
                    },
                    "language_model.model.layers.3.self_attn.embed_q": {
                        "group_size": null, "bits": null
                    },
                    "mx_layer": {"group_size": null, "bits": null, "mode": "mxfp4"},
                    "nv_layer": {"group_size": null, "mode": "nvfp4"},
                    "mx8_layer": {"group_size": null, "bits": null, "mode": "mxfp8"}
                }
            }
            """
        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: json.data(using: .utf8)!)
        let q = try XCTUnwrap(config.perLayerQuantization)

        // Explicit top-level values are untouched.
        let unlisted = try XCTUnwrap(q.quantization(layer: "unlisted"))
        XCTAssertEqual(unlisted.groupSize, 32)
        XCTAssertEqual(unlisted.bits, 4)
        XCTAssertEqual(unlisted.mode, .affine)

        // affine: group 64, bits 4 when absent.
        let gate = try XCTUnwrap(
            q.quantization(layer: "language_model.model.layers.3.mlp.switch_mlp.gate_proj"))
        XCTAssertEqual(gate.groupSize, 64)
        XCTAssertEqual(gate.bits, 3)
        XCTAssertEqual(gate.mode, .affine)
        let down = try XCTUnwrap(
            q.quantization(layer: "language_model.model.layers.3.mlp.switch_mlp.down_proj"))
        XCTAssertEqual(down.groupSize, 64)
        XCTAssertEqual(down.bits, 5)
        // No mode at all: affine defaults, both values absent.
        XCTAssertEqual(
            q.quantization(layer: "language_model.model.layers.3.self_attn.embed_q"),
            .init(groupSize: 64, bits: 4))

        // The other modes follow quantization_params_from_mode in mlx/ops.cpp.
        XCTAssertEqual(q.quantization(layer: "mx_layer")?.asTuple.0, 32)
        XCTAssertEqual(q.quantization(layer: "mx_layer")?.asTuple.1, 4)
        XCTAssertEqual(q.quantization(layer: "mx_layer")?.mode, .mxfp4)
        XCTAssertEqual(q.quantization(layer: "nv_layer")?.asTuple.0, 16)
        XCTAssertEqual(q.quantization(layer: "mx8_layer")?.asTuple.0, 32)
        XCTAssertEqual(q.quantization(layer: "mx8_layer")?.asTuple.1, 8)
    }

    /// The shipped Kimi K2.6 config, when it is on disk: 731 per-layer entries, all
    /// with null group sizes, must decode and resolve the routed experts to 3-bit/64.
    func testRealKimiK26ConfigDecodes() throws {
        let path = NSString(
            string: "~/dev/stream/models/Kimi-K2.6-DQ3/config.json"
        ).expandingTildeInPath
        try XCTSkipUnless(FileManager.default.fileExists(atPath: path), "K2.6 config not present")
        let config = try JSONDecoder().decode(
            BaseConfiguration.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        XCTAssertEqual(config.modelType, "kimi_k25")
        let q = try XCTUnwrap(config.perLayerQuantization)
        XCTAssertEqual(q.perLayerQuantization.count, 731)
        func resolved(_ layer: String) -> (Int, Int, MLX.QuantizationMode)? {
            q.quantization(layer: layer)?.asTuple
        }
        XCTAssertEqual(resolved("language_model.model.layers.3.mlp.switch_mlp.gate_proj")?.0, 64)
        XCTAssertEqual(resolved("language_model.model.layers.3.mlp.switch_mlp.gate_proj")?.1, 3)
        XCTAssertEqual(resolved("language_model.model.layers.3.mlp.switch_mlp.up_proj")?.1, 3)
        XCTAssertEqual(resolved("language_model.model.layers.3.mlp.switch_mlp.down_proj")?.1, 5)
        XCTAssertEqual(resolved("language_model.model.layers.3.self_attn.embed_q")?.0, 64)
        XCTAssertEqual(resolved("language_model.model.layers.3.self_attn.embed_q")?.1, 8)
        XCTAssertEqual(resolved("language_model.model.layers.3.self_attn.embed_q")?.2, .affine)
    }
}
