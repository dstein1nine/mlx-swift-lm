// SharpAI fork overlay tests for deepseek_v41 checkpoint key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXVLM
import XCTest

final class SharpAIDeepseekV41SanitizeTests: XCTestCase {

    private func vector(_ value: Float) -> MLXArray {
        MLXArray([value, value + 1])
    }

    func testDropsRotaryBuffersAndMTPHeads() {
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.rotary_emb.inv_freq": vector(0),
            "mtp.0.eh_proj.weight": vector(1),
            "model.layers.0.self_attn.q_proj.weight": vector(2),
        ]

        let result = SharpAIDeepseekV41Sanitizer.sanitize(weights)

        XCTAssertNil(result["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNil(result["mtp.0.eh_proj.weight"])
        XCTAssertNotNil(result["model.layers.0.self_attn.q_proj.weight"])
    }

    func testKeepsCompressorAndIndexerKeys() {
        let names = [
            "model.layers.2.self_attn.compressor.wkv.weight",
            "model.layers.2.self_attn.indexer.wq_b.weight",
            "model.layers.2.ffn.gate.weight",
        ]
        var weights: [String: MLXArray] = [:]
        for name in names {
            weights[name] = vector(3)
        }

        let result = SharpAIDeepseekV41Sanitizer.sanitize(weights)

        XCTAssertEqual(Set(result.keys), Set(names))
    }

    func testEmptyInputStaysEmpty() {
        XCTAssertTrue(SharpAIDeepseekV41Sanitizer.sanitize([:]).isEmpty)
    }

    /// The Engram tables are read from disk, and DSpark drafter weights are not loaded; the
    /// Engram projection and gate weights load under their own names.
    func testEngramTablesAndDSparkKeysDropped() {
        let dropped = [
            "layers.1.engram.embed.weight",
            "layers.14.engram.embed.scales",
            "mtp.0.dspark.block.weight",
        ]
        var weights: [String: MLXArray] = [
            "layers.1.ffn.gate.weight": vector(4),
            "layers.1.engram.wkv.weight": vector(6),
            "layers.1.engram.q_weight": vector(7),
        ]
        for name in dropped {
            weights[name] = vector(5)
        }

        let result = SharpAIDeepseekV41Sanitizer.toModuleKeys(weights, numExperts: 2)

        XCTAssertEqual(
            Set(result.keys),
            [
                "model.layers.1.mlp.gate.weight", "model.layers.1.engram.wkv.weight",
                "model.layers.1.engram.q_weight",
            ])
    }

    /// An MLX-quantized checkpoint stores each expert's `scales` and `biases` next to its packed
    /// `weight`; all three stack into `switch_mlp`, so the quantized layer finds them.
    func testStacksQuantizedExpertScalesAndBiases() {
        var weights: [String: MLXArray] = [:]
        for expert in 0 ..< 2 {
            for kind in ["weight", "scales", "biases"] {
                weights["layers.3.ffn.experts.\(expert).w2.\(kind)"] = vector(Float(expert))
            }
        }

        let result = SharpAIDeepseekV41Sanitizer.toModuleKeys(weights, numExperts: 2)

        for kind in ["weight", "scales", "biases"] {
            let stackedTensor = result["model.layers.3.mlp.experts.switch_mlp.down_proj.\(kind)"]
            XCTAssertEqual(stackedTensor?.shape, [2, 2], kind)
            XCTAssertEqual(stackedTensor?[1, 0].item(Float.self), 1, kind)
        }
        XCTAssertEqual(result.count, 3)
    }
}
