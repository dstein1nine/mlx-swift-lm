// SharpAI fork overlay tests for minimax_m3_text key remapping. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIMinimaxM3TextSanitizeTests: XCTestCase {

    private func configuration(tied: Bool = false) throws -> SharpAIMinimaxM3TextConfiguration {
        let json = """
            {
              "hidden_size": 4, "intermediate_size": 6, "num_hidden_layers": 4,
              "num_attention_heads": 2, "num_key_value_heads": 2, "vocab_size": 10,
              "num_local_experts": 2, "num_experts_per_tok": 1,
              "tie_word_embeddings": \(tied ? "true" : "false")
            }
            """
        return try JSONDecoder.json5().decode(
            SharpAIMinimaxM3TextConfiguration.self, from: Data(json.utf8))
    }

    private func remap(_ keys: [String], tied: Bool = false) throws -> Set<String> {
        let weights = Dictionary(uniqueKeysWithValues: keys.map { ($0, MLXArray([Float(1)])) })
        return Set(
            SharpAIMinimaxM3TextSanitizer.remapCheckpoint(
                weights, configuration: try configuration(tied: tied)
            ).keys)
    }

    func testVLWrapperPrefixIsStrippedAndVisionDropped() throws {
        let keys = try remap([
            "language_model.model.layers.3.block_sparse_moe.switch_mlp.gate_proj.global_scale",
            "language_model.model.layers.3.self_attn.index_q_proj.scales",
            "language_model.lm_head.weight",
            "vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight",
            "multi_modal_projector.linear_1.weight",
            "patch_merge_mlp.0.weight",
        ])
        XCTAssertEqual(
            keys,
            [
                "model.layers.3.block_sparse_moe.switch_mlp.gate_proj.global_scale",
                "model.layers.3.self_attn.index_q_proj.scales",
                "lm_head.weight",
            ])
    }

    func testMTPRotaryExtraLayersAndTiedHeadAreDropped() throws {
        let keys = try remap(
            [
                "model.layers.0.self_attn.rotary_emb.inv_freq",
                "model.mtp.0.norm.weight",
                "model.layers.4.input_layernorm.weight",
                "model.layers.3.input_layernorm.weight",
                "lm_head.weight",
            ], tied: true)
        XCTAssertEqual(keys, ["model.layers.3.input_layernorm.weight"])
    }

    func testPerExpertTensorsAreStackedIntoSwitchMLP() throws {
        let weights: [String: MLXArray] = [
            "model.layers.3.block_sparse_moe.experts.1.w1.weight": MLXArray([Float(2), 2]),
            "model.layers.3.block_sparse_moe.experts.0.w1.weight": MLXArray([Float(1), 1]),
            "model.layers.3.block_sparse_moe.experts.0.w2.scales": MLXArray([Float(3)]),
            "model.layers.3.block_sparse_moe.experts.1.w2.scales": MLXArray([Float(4)]),
        ]
        let out = SharpAIMinimaxM3TextSanitizer.remapCheckpoint(
            weights, configuration: try configuration())
        XCTAssertEqual(
            Set(out.keys),
            [
                "model.layers.3.block_sparse_moe.switch_mlp.gate_proj.weight",
                "model.layers.3.block_sparse_moe.switch_mlp.down_proj.scales",
            ])
        let gate = try XCTUnwrap(out["model.layers.3.block_sparse_moe.switch_mlp.gate_proj.weight"])
        XCTAssertEqual(gate.shape, [2, 2])
        XCTAssertEqual(gate.asArray(Float.self), [1, 1, 2, 2])
    }
}
