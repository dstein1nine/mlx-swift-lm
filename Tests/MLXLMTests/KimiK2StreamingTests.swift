// SharpAI fork tests for Kimi K2 (DeepseekV32) SSD expert streaming. Not part of upstream.

import Foundation
import MLX
@testable import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class KimiK2StreamingTests: XCTestCase {

    /// A tiny `kimi_k25`: layer 0 dense, layers 1-3 MoE with 8 experts, top-2, sized so
    /// the experts quantize at group size 64.
    private func makeModel() throws -> DeepseekV32Model {
        let json = """
            {
                "model_type": "kimi_k25",
                "text_config": {
                    "model_type": "kimi_k2", "vocab_size": 128, "hidden_size": 64,
                    "intermediate_size": 128, "moe_intermediate_size": 64,
                    "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 4,
                    "n_shared_experts": 1, "n_routed_experts": 8, "routed_scaling_factor": 2.827,
                    "kv_lora_rank": 16, "q_lora_rank": 32, "qk_rope_head_dim": 8,
                    "v_head_dim": 16, "qk_nope_head_dim": 16, "norm_topk_prob": true,
                    "n_group": 1, "topk_group": 1, "num_experts_per_tok": 2,
                    "moe_layer_freq": 1, "first_k_dense_replace": 1,
                    "max_position_embeddings": 4096, "rms_norm_eps": 1e-05, "rope_theta": 50000.0,
                    "attention_bias": false, "scoring_func": "sigmoid", "topk_method": "noaux_tc",
                    "num_nextn_predict_layers": 0
                }
            }
            """
        return DeepseekV32Model(
            try JSONDecoder().decode(DeepseekV32Configuration.self, from: Data(json.utf8)))
    }

    /// Like K2.6, one layer's routed down_proj has a different bit width from the rest.
    private let mixedLayer = "language_model.model.layers.3.mlp.switch_mlp.down_proj"

    /// Random weights in the shipped K2.6 layout: everything under `language_model.`, the
    /// routed experts stacked per projection, and every MLP projection quantized.
    private func writeCheckpoint(to directory: URL) throws {
        MLXRandom.seed(26)
        var arrays: [String: MLXArray] = [:]
        for (key, value) in try makeModel().parameters().flattened() {
            let random: MLXArray =
                key.hasSuffix("norm.weight")
                ? 1 + 0.1 * MLXRandom.normal(value.shape)
                : 0.2 * MLXRandom.normal(value.shape)
            let name = "language_model." + key
            if key.contains(".mlp."), key.hasSuffix("_proj.weight") {
                let bits = name.hasPrefix(mixedLayer) ? 8 : 4
                let (q, scales, biases) = quantized(random, groupSize: 64, bits: bits)
                let prefix = String(name.dropLast(".weight".count))
                arrays[name] = q
                arrays[prefix + ".scales"] = scales
                arrays[prefix + ".biases"] = biases!
            } else {
                arrays[name] = random
            }
        }
        let shard = "model.safetensors"
        try save(arrays: arrays, url: directory.appendingPathComponent(shard))
        let index = [
            "weight_map": Dictionary(uniqueKeysWithValues: arrays.keys.map { ($0, shard) })
        ]
        try JSONSerialization.data(withJSONObject: index)
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
    }

    private func load(_ directory: URL, streaming: Bool) throws -> DeepseekV32Model {
        let model = try makeModel()
        if streaming {
            ExpertStreamingConfig.shared.activate(modelDirectory: directory, useDirectIO: true)
        }
        try loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: .init(
                quantization: .init(groupSize: 64, bits: 4),
                perLayerQuantization: [mixedLayer: .quantize(.init(groupSize: 64, bits: 8))]),
            lazyLoad: streaming)
        return model
    }

    /// Logits of a 20-token prefill (40 routed pairs per MoE layer: the prompt path, with
    /// layers 2-3 prefetched by the prompt pipeline) and of six decode steps after it.
    private func run(_ model: DeepseekV32Model) -> (MLXArray, MLXArray) {
        let tokens = MLXArray((0 ..< 26).map { Int32(($0 * 7 + 3) % 128) }).reshaped(1, 26)
        let cache = model.newCache(parameters: nil)
        let prefill = model(tokens[0..., ..<20], cache: cache)
        eval(prefill)
        var steps: [MLXArray] = []
        for t in 20 ..< 26 {
            let step = model(tokens[0..., t ..< (t + 1)], cache: cache)
            eval(step)
            steps.append(step)
        }
        return (prefill, concatenated(steps, axis: 1))
    }

    func testStreamedExpertsMatchResidentExperts() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("kimi-stream-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeCheckpoint(to: directory)

        let (residentPrefill, residentDecode) = run(try load(directory, streaming: false))

        defer { ExpertStreamingConfig.shared.deactivate() }
        // The test prompt is far shorter than real chunks; run the prompt pipeline anyway.
        let minTokens = SwitchGLU.promptPipelineMinTokens
        SwitchGLU.promptPipelineMinTokens = 1
        defer { SwitchGLU.promptPipelineMinTokens = minTokens }
        let streamed = try load(directory, streaming: true)
        XCTAssertEqual(streamed.model.layers.map { $0.promptLink != nil }, [false, true, true, false])
        let switches = streamed.leafModules().flattened().compactMap { $0.1 as? SwitchLinear }
        XCTAssertEqual(switches.count, 9)
        XCTAssertTrue(switches.allSatisfy { $0 is QuantizedSwitchLinear })
        let before = StreamEngine.shared.snapshot()
        let (streamedPrefill, streamedDecode) = run(streamed)
        let after = StreamEngine.shared.snapshot()
        XCTAssertGreaterThan(after.bytesRead, before.bytesRead)
        // Decode (the stacked slot path) frees the prompt pipeline's buffers.
        XCTAssertFalse(PromptExpertPool.isHoldingBuffers)

        XCTAssertEqual(streamedPrefill.shape, residentPrefill.shape)
        XCTAssertEqual(
            abs(streamedPrefill - residentPrefill).max().item(Float.self), 0, accuracy: 1e-4)
        XCTAssertEqual(
            abs(streamedDecode - residentDecode).max().item(Float.self), 0, accuracy: 1e-4)
        XCTAssertGreaterThan(abs(residentDecode).max().item(Float.self), 0)
    }
}
