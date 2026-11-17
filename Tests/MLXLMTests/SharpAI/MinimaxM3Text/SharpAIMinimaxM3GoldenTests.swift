// SharpAI fork tests: MiniMax M3 against mlx-lm golden logits. Not part of upstream.
//
// Needs a truncated real checkpoint with golden data, made by ~/dev/stream/scripts:
//   truncate-checkpoint.py <MiniMax-M3.1 dir> <out dir> 5
//   minimax-golden.py <out dir>        (mlx-lm 0.32.2)
// Set MINIMAX_M3_GOLDEN_DIR to the out dir; the tests skip without it.
// The long prompt (> 16 blocks x 128 tokens) is the one that runs the sparse indexer.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class SharpAIMinimaxM3GoldenTests: XCTestCase {

    private func goldenDirectory() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["MINIMAX_M3_GOLDEN_DIR"],
            FileManager.default.fileExists(atPath: path + "/golden.safetensors")
        else {
            throw XCTSkip("MINIMAX_M3_GOLDEN_DIR is not set to a golden test model")
        }
        return URL(fileURLWithPath: path)
    }

    private func load(_ directory: URL, streaming: Bool) throws -> SharpAIMinimaxM3TextModel {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let creator = try XCTUnwrap(SharpAIMinimaxM3TextOverlay.creators["minimax_m3_vl"])
        let model = try XCTUnwrap(try creator(data) as? SharpAIMinimaxM3TextModel)
        let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: data)
        if streaming {
            ExpertStreamingConfig.shared.activate(modelDirectory: directory, useDirectIO: true)
        }
        try loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization, lazyLoad: streaming)
        return model
    }

    /// Last-position logits after the prompt, and 8 greedy tokens through the cache.
    private func run(_ model: SharpAIMinimaxM3TextModel, ids: MLXArray) -> (MLXArray, [Int32]) {
        let cache = model.newCache(parameters: nil)
        let logits = model(ids.reshaped(1, -1), cache: cache)[0, -1].asType(.float32)
        eval(logits)
        var greedy: [Int32] = []
        var next = argMax(logits, axis: -1)
        for _ in 0 ..< 8 {
            let token = next.item(Int32.self)
            greedy.append(token)
            let step = model(MLXArray([token]).reshaped(1, 1), cache: cache)[0, -1]
            next = argMax(step, axis: -1)
        }
        return (logits, greedy)
    }

    private func check(
        _ model: SharpAIMinimaxM3TextModel, golden: [String: MLXArray], label: String
    ) throws {
        for prompt in ["short", "long"] {
            let ids = try XCTUnwrap(golden["\(prompt).ids"])
            let expected = try XCTUnwrap(golden["\(prompt).logits"])
            let expectedGreedy = try XCTUnwrap(golden["\(prompt).greedy"]).asArray(Int32.self)
            let (logits, greedy) = run(model, ids: ids)
            let maxDiff = abs(logits - expected).max().item(Float.self)
            let scale = abs(expected).max().item(Float.self)
            print(
                "[golden] \(label) \(prompt): \(ids.size) tokens, max |dlogit| \(maxDiff) of \(scale), greedy \(greedy)"
            )
            XCTAssertEqual(
                argMax(logits).item(Int32.self), argMax(expected).item(Int32.self),
                "\(label) \(prompt) argmax")
            // Measured 0.031 of 9.06 on the long prompt; the old 128-channel index RoPE gave 0.31.
            XCTAssertLessThan(maxDiff, 0.01 * scale, "\(label) \(prompt) logits")
            XCTAssertEqual(greedy, expectedGreedy, "\(label) \(prompt) greedy")
        }
    }

    func testResidentMatchesMLXLM() throws {
        let directory = try goldenDirectory()
        let golden = try loadArrays(url: directory.appendingPathComponent("golden.safetensors"))
        try check(try load(directory, streaming: false), golden: golden, label: "resident")
    }

    func testStreamedMatchesMLXLM() throws {
        let directory = try goldenDirectory()
        let golden = try loadArrays(url: directory.appendingPathComponent("golden.safetensors"))
        defer { ExpertStreamingConfig.shared.deactivate() }
        let model = try load(directory, streaming: true)
        let switches = model.leafModules().flattened().compactMap {
            $0.1 as? QuantizedSwitchLinear
        }
        XCTAssertEqual(switches.count, 6)
        XCTAssertTrue(switches.allSatisfy { $0.mode == .nvfp4 && $0.globalScale != nil })
        let before = StreamEngine.shared.snapshot()
        try check(model, golden: golden, label: "streamed")
        XCTAssertGreaterThan(StreamEngine.shared.snapshot().bytesRead, before.bytesRead)
    }
}
