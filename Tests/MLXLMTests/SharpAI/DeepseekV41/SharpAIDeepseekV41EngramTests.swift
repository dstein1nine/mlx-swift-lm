// SharpAI fork overlay tests for deepseek_v41 Engram. Not part of upstream.
//
// Expected values come from a NumPy port of deepseek-ai/DeepSeek-V4.1-Flash
// inference/engram.py and the Engram module of inference/model.py.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import XCTest

final class SharpAIDeepseekV41EngramTests: XCTestCase {

    /// The published V4.1 Flash Engram layout with a stand-in token map (id % compressed vocab).
    private func publishedHasher() throws -> DeepseekV41EngramHasher {
        try DeepseekV41EngramHasher(
            layerIds: [1, 14], maxNgramSize: 4, nHeads: 8, vocabSize: 16_000_000,
            compressedVocabSize: 99_092, numEmbeddings: [384_006_168, 384_016_682],
            tokenMap: (0 ..< 129_280).map { $0 % 99_092 }, padTokenId: 2)
    }

    private let tokens = [0, 1, 5, 129_279, 77, 99_092, 3]

    func testHashesMatchReference() throws {
        let hasher = try publishedHasher()
        let ids = hasher.rowIds(compressed: tokens.map { hasher.compress($0) }, history: [])
        XCTAssertEqual(ids.count, tokens.count * 2 * 24)

        // [position][layer][column]
        func row(_ position: Int, _ layer: Int) -> [Int] {
            let start = (position * 2 + layer) * 24
            return ids[start ..< (start + 24)].map(Int.init)
        }
        let layer1Sum = (0 ..< tokens.count).reduce(0) { $0 + row($1, 0).reduce(0, +) }
        let layer14Sum = (0 ..< tokens.count).reduce(0) { $0 + row($1, 1).reduce(0, +) }
        XCTAssertEqual(layer1Sum, 32_287_411_444)
        XCTAssertEqual(layer14Sum, 32_317_831_839)
        XCTAssertEqual(Array(row(0, 0).prefix(4)), [5_702_652, 24_393_184, 39_183_301, 61_503_783])
        XCTAssertEqual(
            Array(row(6, 1).suffix(4)), [321_170_198, 349_945_820, 362_951_361, 381_144_289])
        XCTAssertEqual(
            Array(row(3, 0)[8 ..< 12]), [135_049_873, 147_931_757, 174_700_968, 177_230_996])
    }

    /// Decode hashes one token at a time from the carried history; prefill hashes them together.
    func testIncrementalHashingMatchesOnePass() throws {
        let hasher = try publishedHasher()
        let compressed = tokens.map { hasher.compress($0) }
        let onePass = hasher.rowIds(compressed: compressed, history: [])

        var history: [Int64] = []
        var pieces: [Int32] = []
        for chunk in [Array(compressed[0 ..< 4]), [compressed[4]], Array(compressed[5...])] {
            pieces += hasher.rowIds(compressed: chunk, history: history)
            history = Array((history + chunk).suffix(3))
        }
        XCTAssertEqual(pieces, onePass)
    }

    func testTableSizesMustMatchThePrimes() {
        XCTAssertThrowsError(
            try DeepseekV41EngramHasher(
                layerIds: [1, 14], maxNgramSize: 4, nHeads: 8, vocabSize: 16_000_000,
                compressedVocabSize: 99_092, numEmbeddings: [384_006_168, 384_016_683],
                tokenMap: [0, 1, 2], padTokenId: 2)
        ) { error in
            guard case DeepseekV41EngramError.tableSizeMismatch(let layer, _, _) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(layer, 14)
        }
    }

    private func pattern(_ shape: [Int], _ f: (MLXArray) -> MLXArray) -> MLXArray {
        f(MLXArray(0 ..< shape.reduce(1, *)).asType(.float32).reshaped(shape))
    }

    func testEngramModuleMatchesReference() throws {
        let (hc, dim, inputs) = (4, 8, 6)
        let engram = DeepseekV41Engram(dim: dim, hcMult: hc, inputDims: inputs, eps: 1e-20)
        try engram.update(
            parameters: ModuleParameters.unflattened([
                "wkv.weight": pattern([dim * (hc + 1), inputs]) { 0.5 * sin(0.05 * $0 + 0.2) },
                "q_weight": pattern([hc, dim]) { 1 + 0.1 * cos(0.3 * $0) },
                "k_weight": pattern([hc, dim]) { 1 - 0.05 * sin(0.2 * $0) },
            ]), verify: .all)
        let x = pattern([1, 2, hc, dim]) { sin(0.1 * $0 + 0.3) }
        let embedded = pattern([1, 2, inputs]) { cos(0.07 * $0) }

        let out = engram(x, embedded: embedded)

        XCTAssertEqual(out.sum().item(Float.self), -67.60110, accuracy: 1e-3)
        XCTAssertEqual(abs(out).sum().item(Float.self), 77.93373, accuracy: 1e-3)
        let a = out[0, 1, 2, ..<4].asArray(Float.self)
        for (got, want) in zip(a, [-1.7737355, -2.1595910, -2.4226258, -2.5352831]) {
            XCTAssertEqual(got, Float(want), accuracy: 1e-4)
        }
        let b = out[0, 0, 3, 5...].asArray(Float.self)
        for (got, want) in zip(b, [-2.2565589, -1.9617207, -1.5041634]) {
            XCTAssertEqual(got, Float(want), accuracy: 1e-4)
        }
    }

    /// Rows read with pread from a safetensors shard equal the dequantized table's rows.
    func testTableReadsQuantizedRowsFromSafetensors() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("engram-table-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let headDim = 256
        let full = pattern([10, headDim]) { sin(0.013 * $0) }
        let (w, scales, biases) = quantized(full, groupSize: 64, bits: 4)
        let shard = "model-00001-of-00001.safetensors"
        let names = ["weight": w, "scales": scales, "biases": biases!]
        var arrays: [String: MLXArray] = [:]
        for (kind, array) in names { arrays["layers.1.engram.embed.\(kind)"] = array }
        try save(arrays: arrays, url: directory.appendingPathComponent(shard))
        let index = ["weight_map": Dictionary(uniqueKeysWithValues: arrays.keys.map { ($0, shard) })]
        try JSONSerialization.data(withJSONObject: index)
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))

        let table = try DeepseekV41EngramTable(
            modelDirectory: directory, layerId: 1, headDim: headDim)
        XCTAssertEqual(table.rowCount, 10)
        XCTAssertEqual(table.bits, 4)
        XCTAssertEqual(table.groupSize, 64)

        let rows = try table.rows([7, 0, 7, 9])
        let expected = dequantized(
            w, scales: scales, biases: biases, groupSize: 64, bits: 4, dtype: .float32
        )[MLXArray([7, 0, 7, 9] as [Int32])]
        XCTAssertEqual(rows.shape, [4, headDim])
        XCTAssertEqual(abs(rows - expected).max().item(Float.self), 0)
        XCTAssertThrowsError(try table.rows([10]))
    }

    /// MLX conversions state one bit width; layers packed at another are inferred from shapes.
    func testInfersPerLayerBitsFromPackedWidths() {
        let base = BaseConfiguration.Quantization(groupSize: 64, bits: 4)
        let weights: [String: MLXArray] = [
            // 5120 inputs at 4 bits: 640 packed words, 80 groups
            "a.weight": MLXArray.zeros([4, 640], type: UInt32.self),
            "a.scales": MLXArray.zeros([4, 80]),
            // 5120 inputs at 8 bits: 1280 packed words
            "b.weight": MLXArray.zeros([4, 1280], type: UInt32.self),
            "b.scales": MLXArray.zeros([4, 80]),
        ]

        let result = SharpAIDeepseekV41Model.inferredBits(weights, base: base, overrides: [:])

        XCTAssertNil(result["a"])
        guard case .quantize(let q) = result["b"] else { return XCTFail("b not overridden") }
        XCTAssertEqual(q.bits, 8)
        XCTAssertEqual(q.groupSize, 64)
    }
}
