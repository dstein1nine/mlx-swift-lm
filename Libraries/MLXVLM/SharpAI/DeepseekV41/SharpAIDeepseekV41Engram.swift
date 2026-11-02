// SharpAI fork overlay: deepseek_v41 Engram memory. Not part of upstream mlx-swift-lm.
//
// Port of deepseek-ai/DeepSeek-V4.1-Flash inference/engram.py (n-gram hashing) and the Engram
// module of inference/model.py. mlx-lm's deepseek_v41.py rejects Engram, so the reference is the
// DeepSeek PyTorch code.
//
// At each Engram layer, every position hashes the 2-, 3- and 4-grams of (compressed) token ids
// ending there, `n_heads` ways each, into a ~384M-row table; the fetched rows go through `wkv` and
// a sigmoid gate adds the result to every hyper-connection stream. The tables are ~61 GB each in
// the MLX 4-bit checkpoint, and a token needs 24 rows of each, so the rows are read from the
// safetensors files with pread instead of loading the tables.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum DeepseekV41EngramError: LocalizedError {
    case missingConfiguration(String)
    case unknownHashMultipliers(layerId: Int, compressedVocabSize: Int, maxNgramSize: Int)
    case tableSizeMismatch(layerId: Int, expected: Int, fromPrimes: Int)
    case tokenMap(String)
    case weightFile(String)

    public var errorDescription: String? {
        switch self {
        case .missingConfiguration(let field):
            return "deepseek_v41 Engram: config has engram_layer_ids but no \(field)."
        case .unknownHashMultipliers(let layerId, let vocab, let ngram):
            return "deepseek_v41 Engram: no hash multipliers for layer \(layerId) with compressed "
                + "vocab \(vocab) and max n-gram \(ngram); they come from numpy's default_rng "
                + "and are tabulated only for the published config."
        case .tableSizeMismatch(let layerId, let expected, let fromPrimes):
            return "deepseek_v41 Engram: layer \(layerId) table has \(expected) rows in the config "
                + "but the bucket primes add up to \(fromPrimes)."
        case .tokenMap(let detail):
            return "deepseek_v41 Engram token map: \(detail)"
        case .weightFile(let detail):
            return "deepseek_v41 Engram table: \(detail)"
        }
    }
}

// MARK: - Hashing

/// The n-gram hash of engram.py: compressed token ids, per-layer multipliers, XOR-rolled over
/// the look-back and reduced modulo a distinct prime per (n-gram size, head).
public struct DeepseekV41EngramHasher: Sendable {
    public let layerIds: [Int]
    public let maxNgramSize: Int
    public let nHeads: Int
    /// [layer][(n - 2) * nHeads + head] bucket modulus for the n-gram of size n.
    public let primes: [[Int64]]
    /// [layer][column] start of that column's bucket range in the layer's table.
    public let offsets: [[Int64]]
    /// [layer][look-back] odd multipliers.
    public let multipliers: [[Int64]]
    /// Token id -> compressed id (`engram_token_map.json`).
    let tokenMap: [Int64]
    /// The compressed id that fills n-gram slots before the start of the sequence.
    public let padId: Int64

    /// Row ids per position and layer: (maxNgramSize - 1) n-gram sizes times nHeads.
    public var columns: Int { (maxNgramSize - 1) * nHeads }

    /// `compute_hash_multipliers` draws these from `np.random.default_rng(10007 * layer_id)`,
    /// bounded by `(int64 max // compressed_vocab_size) // 2`, then `* 2 + 1`. numpy's generator
    /// is not reproduced here; these are its output for the published V4.1 Flash config.
    static let knownMultipliers: [String: [Int64]] = [
        "1|99092|4": [76_632_096_046_245, 4_839_876_093_313, 35_959_672_319_349, 73_987_337_458_391],
        "14|99092|4": [67_716_810_739_261, 51_510_806_800_915, 30_921_347_202_721, 82_619_226_485_591],
    ]

    public init(
        layerIds: [Int], maxNgramSize: Int, nHeads: Int, vocabSize: Int,
        compressedVocabSize: Int, numEmbeddings: [Int], tokenMap: [Int], padTokenId: Int
    ) throws {
        guard tokenMap.indices.contains(padTokenId) else {
            throw DeepseekV41EngramError.tokenMap("pad token \(padTokenId) is outside the map")
        }
        guard let largest = tokenMap.max(), largest < compressedVocabSize, tokenMap.min() ?? 0 >= 0
        else {
            throw DeepseekV41EngramError.tokenMap(
                "ids must lie in 0..<\(compressedVocabSize) (engram_compressed_vocab_size)")
        }
        guard numEmbeddings.count == layerIds.count else {
            throw DeepseekV41EngramError.missingConfiguration(
                "engram_num_embeddings entry for every engram layer")
        }
        self.layerIds = layerIds
        self.maxNgramSize = maxNgramSize
        self.nHeads = nHeads
        self.tokenMap = tokenMap.map(Int64.init)
        self.padId = Int64(tokenMap[padTokenId])

        // EngramLayout.from_args: primes are drawn in order and never reused, so every
        // (layer, n-gram size, head) owns a disjoint bucket range.
        var seen = Set<Int64>()
        var primes: [[Int64]] = []
        for _ in layerIds {
            var layer: [Int64] = []
            for _ in 0 ..< maxNgramSize - 1 {
                var current = Int64(vocabSize - 1)
                for _ in 0 ..< nHeads {
                    current = Self.nextPrime(after: current, excluding: seen)
                    seen.insert(current)
                    layer.append(current)
                }
            }
            primes.append(layer)
        }
        for (i, layerId) in layerIds.enumerated() {
            let total = primes[i].reduce(0, +)
            guard total == Int64(numEmbeddings[i]) else {
                throw DeepseekV41EngramError.tableSizeMismatch(
                    layerId: layerId, expected: numEmbeddings[i], fromPrimes: Int(total))
            }
        }
        self.primes = primes
        self.offsets = primes.map { layer in
            var running: Int64 = 0
            return layer.map { prime in
                defer { running += prime }
                return running
            }
        }
        self.multipliers = try layerIds.map { layerId in
            let key = "\(layerId)|\(compressedVocabSize)|\(maxNgramSize)"
            guard let values = Self.knownMultipliers[key] else {
                throw DeepseekV41EngramError.unknownHashMultipliers(
                    layerId: layerId, compressedVocabSize: compressedVocabSize,
                    maxNgramSize: maxNgramSize)
            }
            return values
        }
    }

    static func isPrime(_ n: Int64) -> Bool {
        if n < 2 { return false }
        if n % 2 == 0 { return n == 2 }
        var d: Int64 = 3
        while d * d <= n {
            if n % d == 0 { return false }
            d += 2
        }
        return true
    }

    static func nextPrime(after start: Int64, excluding seen: Set<Int64>) -> Int64 {
        var candidate = start + 1
        while !isPrime(candidate) || seen.contains(candidate) {
            candidate += 1
        }
        return candidate
    }

    /// The compressed id of `token`; ids past the map (none in a real vocabulary) count as pad.
    public func compress(_ token: Int) -> Int64 {
        tokenMap.indices.contains(token) ? tokenMap[token] : padId
    }

    /// Row ids for the positions of `tokens` (compressed ids), which follow `history` (the
    /// compressed ids of earlier tokens in the sequence; its last `maxNgramSize - 1` suffice).
    /// Laid out [position][layer][column], ready to add to nothing: offsets are applied.
    public func rowIds(compressed tokens: [Int64], history: [Int64]) -> [Int32] {
        let lookBack = maxNgramSize - 1
        let past = Array(history.suffix(lookBack))
        let sequence = past + tokens
        var out = [Int32]()
        out.reserveCapacity(tokens.count * layerIds.count * columns)
        var window = [Int64](repeating: 0, count: maxNgramSize)
        for position in past.count ..< sequence.count {
            for shift in 0 ..< maxNgramSize {
                window[shift] = position >= shift ? sequence[position - shift] : padId
            }
            for layer in layerIds.indices {
                let mult = multipliers[layer]
                // XOR the multiplied ids one look-back at a time; after step i the running
                // value hashes the (i + 1)-gram.
                var rolling = window[0] &* mult[0]
                for i in 1 ..< maxNgramSize {
                    rolling ^= window[i] &* mult[i]
                    for head in 0 ..< nHeads {
                        let column = (i - 1) * nHeads + head
                        let prime = primes[layer][column]
                        let bucket = ((rolling % prime) + prime) % prime
                        out.append(Int32(bucket + offsets[layer][column]))
                    }
                }
            }
        }
        return out
    }
}

// MARK: - Table rows from disk

/// One Engram table (`layers.L.engram.embed.{weight,scales,biases}`, MLX affine-quantized rows)
/// read row by row from its safetensors files.
public final class DeepseekV41EngramTable: @unchecked Sendable {
    struct Tensor {
        let fd: Int32
        let dataStart: Int64
        let rowBytes: Int
        let rows: Int
        let columns: Int
        let dtype: DType
    }

    let weight: Tensor
    let scales: Tensor
    let biases: Tensor
    public let headDim: Int
    public var rowCount: Int { weight.rows }
    public let bits: Int
    public let groupSize: Int

    public init(modelDirectory: URL, layerId: Int, headDim: Int) throws {
        let index = try Self.weightIndex(modelDirectory)
        let base = "layers.\(layerId).engram.embed"
        weight = try Self.tensor("\(base).weight", in: modelDirectory, index: index)
        scales = try Self.tensor("\(base).scales", in: modelDirectory, index: index)
        biases = try Self.tensor("\(base).biases", in: modelDirectory, index: index)
        self.headDim = headDim
        guard weight.dtype == .uint32, weight.rows == scales.rows, weight.rows == biases.rows,
            scales.columns == biases.columns, scales.columns > 0, headDim % scales.columns == 0
        else {
            throw DeepseekV41EngramError.weightFile(
                "\(base) is not an affine-quantized [rows, \(headDim)] table")
        }
        groupSize = headDim / scales.columns
        bits = weight.columns * 32 / headDim
        guard bits * headDim == weight.columns * 32 else {
            throw DeepseekV41EngramError.weightFile("\(base).weight width does not fit \(headDim)")
        }
    }

    deinit {
        for fd in Set([weight.fd, scales.fd, biases.fd]) { close(fd) }
    }

    /// Dequantized rows for `ids`, [ids.count, headDim] float32. Each distinct row is read once.
    public func rows(_ ids: [Int32]) throws -> MLXArray {
        var slot = [Int32: Int32]()
        var unique = [Int]()
        let gather = ids.map { id -> Int32 in
            if let s = slot[id] { return s }
            let s = Int32(unique.count)
            slot[id] = s
            unique.append(Int(id))
            return s
        }
        for id in unique where id < 0 || id >= weight.rows {
            throw DeepseekV41EngramError.weightFile("row \(id) outside 0..<\(weight.rows)")
        }
        let w = try read(weight, rows: unique)
        let s = try read(scales, rows: unique)
        let b = try read(biases, rows: unique)
        let table = dequantized(
            MLXArray(w, [unique.count, weight.columns], dtype: weight.dtype),
            scales: MLXArray(s, [unique.count, scales.columns], dtype: scales.dtype),
            biases: MLXArray(b, [unique.count, biases.columns], dtype: biases.dtype),
            groupSize: groupSize, bits: bits, dtype: .float32)
        return take(table, MLXArray(gather), axis: 0)
    }

    private func read(_ tensor: Tensor, rows: [Int]) throws -> Data {
        var data = Data(count: rows.count * tensor.rowBytes)
        var failure: String?
        let lock = NSLock()
        data.withUnsafeMutableBytes { raw in
            let base = raw.baseAddress!
            let chunk = 64
            let chunks = (rows.count + chunk - 1) / chunk
            DispatchQueue.concurrentPerform(iterations: chunks) { c in
                for i in (c * chunk) ..< min(rows.count, (c + 1) * chunk) {
                    let offset = tensor.dataStart + Int64(rows[i]) * Int64(tensor.rowBytes)
                    let got = pread(
                        tensor.fd, base + i * tensor.rowBytes, tensor.rowBytes, off_t(offset))
                    if got != tensor.rowBytes {
                        lock.lock()
                        failure = "short read of row \(rows[i]) (\(got) of \(tensor.rowBytes) bytes)"
                        lock.unlock()
                    }
                }
            }
        }
        if let failure { throw DeepseekV41EngramError.weightFile(failure) }
        return data
    }

    static func weightIndex(_ directory: URL) throws -> [String: String] {
        let url = directory.appendingPathComponent("model.safetensors.index.json")
        guard let data = try? Data(contentsOf: url),
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let map = json["weight_map"] as? [String: String]
        else {
            throw DeepseekV41EngramError.weightFile("cannot read \(url.path)")
        }
        return map
    }

    static func tensor(_ name: String, in directory: URL, index: [String: String]) throws -> Tensor {
        guard let file = index[name] else {
            throw DeepseekV41EngramError.weightFile("\(name) is not in the weight index")
        }
        let path = directory.appendingPathComponent(file).path
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw DeepseekV41EngramError.weightFile("cannot open \(path)") }
        var length: UInt64 = 0
        guard pread(fd, &length, 8, 0) == 8, length < 1 << 30 else {
            close(fd)
            throw DeepseekV41EngramError.weightFile("bad safetensors header in \(path)")
        }
        var header = Data(count: Int(length))
        let got = header.withUnsafeMutableBytes { pread(fd, $0.baseAddress!, Int(length), 8) }
        guard got == Int(length),
            let json = try? JSONSerialization.jsonObject(with: header) as? [String: Any],
            let entry = json[name] as? [String: Any],
            let shape = entry["shape"] as? [Int], shape.count == 2,
            let span = entry["data_offsets"] as? [Int], span.count == 2,
            let dtypeName = entry["dtype"] as? String
        else {
            close(fd)
            throw DeepseekV41EngramError.weightFile("no 2-D entry for \(name) in \(path)")
        }
        let dtypes: [String: DType] = [
            "U32": .uint32, "F32": .float32, "F16": .float16, "BF16": .bfloat16,
        ]
        guard let dtype = dtypes[dtypeName], shape[0] > 0,
            (span[1] - span[0]) == shape[0] * shape[1] * dtype.size
        else {
            close(fd)
            throw DeepseekV41EngramError.weightFile("unexpected dtype or size for \(name)")
        }
        return Tensor(
            fd: fd, dataStart: Int64(8 + Int(length) + span[0]),
            rowBytes: shape[1] * dtype.size, rows: shape[0], columns: shape[1], dtype: dtype)
    }
}

/// Hasher and tables for a loaded model, attached by `SharpAIDeepseekV41Model.attach`.
public final class DeepseekV41EngramRuntime: @unchecked Sendable {
    public let hasher: DeepseekV41EngramHasher
    /// Keyed by layer id.
    public let tables: [Int: DeepseekV41EngramTable]

    public init(hasher: DeepseekV41EngramHasher, tables: [Int: DeepseekV41EngramTable]) {
        self.hasher = hasher
        self.tables = tables
    }

    /// Reads `engram_token_map.json` (a JSON array: token id -> compressed id) from the model
    /// directory. The reference derives it from the tokenizer; MLX conversions ship it.
    public static func loadTokenMap(modelDirectory: URL) throws -> [Int] {
        let url = modelDirectory.appendingPathComponent("engram_token_map.json")
        guard let data = try? Data(contentsOf: url) else {
            throw DeepseekV41EngramError.tokenMap("\(url.path) is missing")
        }
        guard let map = try JSONSerialization.jsonObject(with: data) as? [Int] else {
            throw DeepseekV41EngramError.tokenMap("\(url.path) is not an array of ints")
        }
        return map
    }
}

// MARK: - Module

/// Writes an n-gram lookup into the residual streams, gated by how well it matches each one
/// (`Engram` in model.py).
public final class DeepseekV41Engram: Module {
    let dim: Int
    let hcMult: Int
    let eps: Float
    /// `clamp_value` of the reference: the floor under |dot| before the signed square root.
    let clampValue: Float = 1e-6

    @ModuleInfo(key: "wkv") var wkv: Linear
    @ParameterInfo(key: "q_weight") var qWeight: MLXArray
    @ParameterInfo(key: "k_weight") var kWeight: MLXArray

    public init(dim: Int, hcMult: Int, inputDims: Int, eps: Float) {
        self.dim = dim
        self.hcMult = hcMult
        self.eps = eps
        self._wkv.wrappedValue = Linear(inputDims, dim * (hcMult + 1), bias: false)
        self._qWeight.wrappedValue = MLXArray.ones([hcMult, dim])
        self._kWeight.wrappedValue = MLXArray.ones([hcMult, dim])
        super.init()
    }

    /// x [B, L, hcMult, dim]; embedded [B, L, columns * headDim] (the fetched rows, flattened).
    public func callAsFunction(_ x: MLXArray, embedded: MLXArray) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        // The reference casts the dequantized rows to bf16 before wkv.
        let kv = wkv(embedded.asType(.bfloat16))
        let key = kv[.ellipsis, ..<(hcMult * dim)].asType(.float32).reshaped(b, l, hcMult, dim)
        let value = kv[.ellipsis, (hcMult * dim)...].asType(.float32)
        let weight = qWeight.asType(.float32) * kWeight.asType(.float32)
        let h = x.asType(.float32)
        // Normalized per (token, stream) over dim, not jointly over the streams.
        let rstd =
            rsqrt(mean(h * h, axis: -1) + eps) * rsqrt(mean(key * key, axis: -1) + eps)
        let dot = sum(h * weight * key, axis: -1) * rstd * Float(pow(Double(dim), -0.5))
        // Signed square root before the sigmoid, as in the training kernel.
        let magnitude = sqrt(maximum(abs(dot), clampValue))
        let gate = sigmoid(which(dot .< 0, -magnitude, magnitude))
        let out = h + gate.expandedDimensions(axis: -1) * value.expandedDimensions(axis: -2)
        return out.asType(x.dtype)
    }
}
