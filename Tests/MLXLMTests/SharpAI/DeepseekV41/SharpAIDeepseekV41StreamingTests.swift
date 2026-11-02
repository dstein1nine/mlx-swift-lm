// SharpAI fork overlay tests for deepseek_v41 SSD expert streaming. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import XCTest

final class SharpAIDeepseekV41StreamingTests: XCTestCase {

    /// The tiny fixture widened so the experts quantize at group size 64.
    private func makeModel() throws -> SharpAIDeepseekV41Model {
        let json = DeepseekV41TinyFixture.json
            .replacingOccurrences(of: "\"hidden_size\": 32", with: "\"hidden_size\": 64")
            .replacingOccurrences(
                of: "\"moe_intermediate_size\": 16", with: "\"moe_intermediate_size\": 64")
        let config = try JSONDecoder.json5().decode(
            SharpAIDeepseekV41Configuration.self, from: Data(json.utf8))
        return SharpAIDeepseekV41Model(try XCTUnwrap(config.text))
    }

    /// Writes random weights in the MLX checkpoint's layout: module names without `model.`, except the routed
    /// experts, which are 4-bit and stored one tensor per expert as `layers.N.ffn.experts.E.wK`.
    private func writeCheckpoint(to directory: URL) throws {
        MLXRandom.seed(41)
        let projections = ["gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"]
        var arrays: [String: MLXArray] = [:]
        for (key, value) in try makeModel().parameters().flattened() {
            let random: MLXArray =
                key.hasSuffix("norm.weight")
                ? 1 + 0.1 * MLXRandom.normal(value.shape)
                : 0.2 * MLXRandom.normal(value.shape)
            let parts = key.split(separator: ".").map(String.init)
            guard key.contains(".mlp.experts.switch_mlp."), let w = projections[parts[6]] else {
                // Checkpoint names lack the `model.` prefix; the sanitizer adds it back.
                let name =
                    key == "lm_head.weight"
                    ? "head.weight"
                    : (key == "model.embed_tokens.weight"
                        ? "embed.weight" : String(key.dropFirst(6)))
                arrays[name] = random
                continue
            }
            for expert in 0 ..< value.dim(0) {
                let (q, scales, biases) = quantized(random[expert], groupSize: 64, bits: 4)
                let prefix = "layers.\(parts[2]).ffn.experts.\(expert).\(w)"
                arrays[prefix + ".weight"] = q
                arrays[prefix + ".scales"] = scales
                arrays[prefix + ".biases"] = biases!
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

    private func load(_ directory: URL, streaming: Bool) throws -> SharpAIDeepseekV41Model {
        let model = try makeModel()
        if streaming {
            ExpertStreamingConfig.shared.activate(modelDirectory: directory, useDirectIO: true)
        }
        try loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: .init(
                quantization: .init(groupSize: 64, bits: 4), perLayerQuantization: [:]),
            lazyLoad: streaming)
        return model
    }

    /// Logits of a 20-token prefill (40 routed pairs: the per-projection read path) and of
    /// six single-token decode steps after it (the stacked slot path).
    private func run(_ model: SharpAIDeepseekV41Model) -> (MLXArray, MLXArray) {
        let tokens = MLXArray((0 ..< 26).map { Int32(($0 * 7 + 3) % 64) }).reshaped(1, 26)
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
            .appendingPathComponent("v41-stream-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeCheckpoint(to: directory)

        let (residentPrefill, residentDecode) = run(try load(directory, streaming: false))

        defer { ExpertStreamingConfig.shared.deactivate() }
        let streamed = try load(directory, streaming: true)
        let switches = streamed.leafModules().flattened().compactMap { $0.1 as? SwitchLinear }
        XCTAssertEqual(switches.count, 9)
        for sl in switches {
            XCTAssertEqual(sl.unstackedSSDMap?.count, 4)
            XCTAssertTrue(sl is QuantizedSwitchLinear)
        }
        let (streamedPrefill, streamedDecode) = run(streamed)

        XCTAssertEqual(streamedPrefill.shape, residentPrefill.shape)
        XCTAssertEqual(
            abs(streamedPrefill - residentPrefill).max().item(Float.self), 0, accuracy: 1e-4)
        XCTAssertEqual(
            abs(streamedDecode - residentDecode).max().item(Float.self), 0, accuracy: 1e-4)
        XCTAssertGreaterThan(abs(residentDecode).max().item(Float.self), 0)
    }

    func testExpertKeysFollowTheCheckpointNames() throws {
        let model = try makeModel()
        XCTAssertEqual(
            model.unstackedExpertWeightKey(
                switchLinearPath: "model.layers.12.mlp.experts.switch_mlp.up_proj", expert: 383),
            "layers.12.ffn.experts.383.w3.weight")
        XCTAssertEqual(
            model.unstackedExpertWeightKey(
                switchLinearPath: "model.layers.0.mlp.experts.switch_mlp.down_proj", expert: 0),
            "layers.0.ffn.experts.0.w2.weight")
        XCTAssertNil(
            model.unstackedExpertWeightKey(
                switchLinearPath: "model.layers.0.mlp.shared_experts.gate_proj", expert: 0))
    }

    /// The clamped SwiGLU clips gate from above and up from both sides before silu(gate) * up.
    func testSwitchGLUUsesTheClampedActivation() {
        let gate = MLXArray([Float(-3), 4, 25, 12])
        let up = MLXArray([Float(2), -15, 11, 1])
        let product = deepseekV41ClampedSwiGLUProduct(limit: 10)(gate, up)
        let expected = MLXArray([Float(-3), 4, 10, 10]).asType(.float32)
        let want = silu(expected) * MLXArray([Float(2), -10, 10, 1])
        XCTAssertTrue(allClose(product, want, atol: 1e-5).item(Bool.self))
    }
}
