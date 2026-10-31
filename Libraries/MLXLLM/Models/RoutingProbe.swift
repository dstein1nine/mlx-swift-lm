import Foundation
import MLX
import MLXNN

/// Measures how well each MoE layer's router predicts its experts from earlier hidden
/// states, to size an expert prefetcher. Enabled by `MLX_MOE_ROUTE_PROBE=<path>`; writes
/// one JSON line per decoded token with recall at lookahead 1-3 and guess sizes 8/16/32.
/// Measurement only: it adds GPU syncs. Exact for `n_group == 1` routers (Kimi K2, GLM-5).
final class RoutingProbe: @unchecked Sendable {
    static let shared: RoutingProbe? = {
        guard let path = ProcessInfo.processInfo.environment["MLX_MOE_ROUTE_PROBE"],
            !path.isEmpty
        else { return nil }
        return RoutingProbe(path: path)
    }()

    static let lookaheads = [1, 2, 3]
    static let guessSizes = [8, 16, 32]

    private let handle: FileHandle?
    private var previous: [Int: Set<Int>] = [:]
    private var tokenIndex = 0
    private let lock = NSLock()

    private init(path: String) {
        if !FileManager.default.fileExists(atPath: path) {
            FileManager.default.createFile(atPath: path, contents: nil)
        }
        handle = FileHandle(forWritingAtPath: path)
        handle?.seekToEndOfFile()
        print("[RoutingProbe] writing expert-routing predictability to \(path)")
    }

    /// Rank all experts for `layer` from hidden state `h` (shape [1, 1, hidden]).
    private func ranking(_ moe: DeepseekV3MoE, _ norm: RMSNorm, _ h: MLXArray, top: Int) -> MLXArray {
        let x = norm(h)
        let scores = sigmoid(x.matmul(moe.gate.weight.T)) + moe.gate.e_score_correction_bias
        return argSort(-scores, axis: -1)[.ellipsis, ..<top]
    }

    /// Called after a full single-token forward. `inputs[j]` is the hidden state that
    /// entered layer `j`; `actual[i]` is the expert indices layer `i` routed to.
    func record(
        layers: [DeepseekV32DecoderLayer], inputs: [MLXArray], actual: [Int: MLXArray]
    ) {
        let maxGuess = Self.guessSizes.max()!
        var guesses: [(layer: Int, d: Int, arr: MLXArray)] = []
        for (i, layer) in layers.enumerated() {
            guard let moe = layer.mlp as? DeepseekV3MoE, actual[i] != nil else { continue }
            for d in Self.lookaheads {
                let j = i - d + 1
                guard j >= 0, j < inputs.count else { continue }
                guesses.append((i, d, ranking(moe, layer.postAttentionLayerNorm, inputs[j], top: maxGuess)))
            }
        }
        let actualArrays = actual.sorted { $0.key < $1.key }
        eval(guesses.map(\.arr) + actualArrays.map(\.value))

        var actualSets: [Int: Set<Int>] = [:]
        for (i, a) in actualArrays {
            actualSets[i] = Set(a.asArray(Int32.self).map(Int.init))
        }

        // recall[d][k] summed over layers, plus per-layer d=2/k=16 detail.
        var sums: [String: Double] = [:]
        var counts: [String: Int] = [:]
        var perLayer: [String: [Int: Double]] = [:]
        for g in guesses {
            guard let truth = actualSets[g.layer], !truth.isEmpty else { continue }
            let ranked = g.arr.asArray(Int32.self).map(Int.init)
            for k in Self.guessSizes {
                let hit = Set(ranked.prefix(k)).intersection(truth).count
                let r = Double(hit) / Double(truth.count)
                let key = "d\(g.d)_top\(k)"
                sums[key, default: 0] += r
                counts[key, default: 0] += 1
                perLayer[key, default: [:]][g.layer] = r
            }
        }

        lock.lock()
        defer { lock.unlock() }
        var prevSum = 0.0
        var prevCount = 0
        var prevPerLayer: [Int: Double] = [:]
        for (i, truth) in actualSets where !truth.isEmpty {
            if let p = previous[i] {
                let r = Double(p.intersection(truth).count) / Double(truth.count)
                prevSum += r
                prevCount += 1
                prevPerLayer[i] = r
            }
            previous[i] = truth
        }

        var row: [String: Any] = ["token": tokenIndex]
        for (key, total) in sums {
            row[key] = total / Double(counts[key]!)
        }
        if prevCount > 0 { row["prev_token_top8"] = prevSum / Double(prevCount) }
        // Per-layer detail for the most useful configurations.
        for key in ["d1_top8", "d2_top8", "d2_top16"] {
            if let m = perLayer[key] {
                row["layers_\(key)"] = m.sorted { $0.key < $1.key }.map { [$0.key, $0.value] }
            }
        }
        if !prevPerLayer.isEmpty {
            row["layers_prev_token_top8"] = prevPerLayer.sorted { $0.key < $1.key }.map { [$0.key, $0.value] }
        }
        tokenIndex += 1
        if let data = try? JSONSerialization.data(withJSONObject: row, options: [.sortedKeys]) {
            handle?.write(data)
            handle?.write("\n".data(using: .utf8)!)
        }
    }
}
