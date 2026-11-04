import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Routing tests for the DeepSeek-V3 gate, which GLM-5.x and Kimi K2.x also use.
final class DeepseekV3RoutingTests: XCTestCase {

    private func config(nRouted: Int, topK: Int, hidden: Int = 4, scaling: Float = 2.5)
        throws -> DeepseekV3Configuration
    {
        let json = """
            {
                "vocab_size": 32, "hidden_size": \(hidden), "intermediate_size": 16,
                "moe_intermediate_size": 8, "num_hidden_layers": 1,
                "num_attention_heads": 2, "num_key_value_heads": 2,
                "n_shared_experts": null, "n_routed_experts": \(nRouted),
                "routed_scaling_factor": \(scaling), "kv_lora_rank": 4, "q_lora_rank": 4,
                "qk_rope_head_dim": 2, "v_head_dim": 2, "qk_nope_head_dim": 2,
                "norm_topk_prob": true, "n_group": 1, "topk_group": 1,
                "num_experts_per_tok": \(topK), "moe_layer_freq": 1,
                "first_k_dense_replace": 0, "max_position_embeddings": 128,
                "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "attention_bias": false
            }
            """
        return try JSONDecoder().decode(DeepseekV3Configuration.self, from: Data(json.utf8))
    }

    /// Expert e responds only to hidden dim e, so the input sets each expert's logit.
    private func oneHot(nRouted: Int, hidden: Int) -> MLXArray {
        var w = [Float](repeating: 0, count: nRouted * hidden)
        for e in 0 ..< nRouted { w[e * hidden + e] = 1 }
        return MLXArray(w).reshaped(nRouted, hidden)
    }

    /// The correction bias picks the experts; the weights come from the unbiased scores.
    /// A large uniform bias (GLM-5.3 ships ~8-34 per layer) must not flatten the weights.
    func testCorrectionBiasSelectsButDoesNotWeight() throws {
        let gate = MoEGate(config: try config(nRouted: 4, topK: 2))
        try gate.update(
            parameters: ModuleParameters.unflattened([
                "weight": oneHot(nRouted: 4, hidden: 4),
                // Uniform +30, and +1.5 more on expert 3 so it is selected over expert 1.
                "e_score_correction_bias": MLXArray([30, 30, 30, 31.5] as [Float]),
            ]),
            verify: [])

        let x = MLXArray([2.0, 1.0, 0.0, -1.0] as [Float]).reshaped(1, 1, 4)
        let (inds, weights) = gate(x)
        eval(inds, weights)

        let chosen = inds.reshaped(-1).asArray(Int32.self).map(Int.init)
        XCTAssertEqual(Set(chosen), [0, 3])

        func sigmoid(_ v: Float) -> Float { 1 / (1 + exp(-v)) }
        let s0 = sigmoid(2), s3 = sigmoid(-1)
        let expected: [Int: Float] = [0: 2.5 * s0 / (s0 + s3), 3: 2.5 * s3 / (s0 + s3)]
        for (e, w) in zip(chosen, weights.reshaped(-1).asArray(Float.self)) {
            XCTAssertEqual(w, expected[e]!, accuracy: 1e-4, "expert \(e)")
        }
    }
}
