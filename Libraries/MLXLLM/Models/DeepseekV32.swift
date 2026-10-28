// Copyright © 2025 Apple Inc.

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/deepseek_v32.py
// and https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/glm_moe_dsa.py
//
// DeepSeek V3.2 is V3 plus DeepSeek Sparse Attention (DSA): a "lightning indexer"
// scores every cached key against the query and attention is restricted to the
// top `index_topk` of them. GLM-5.2 (`glm_moe_dsa`) is the same architecture — in
// mlx-lm its model class is a bare `class Model(DSV32Model)` over a differently
// shaped config — so both land here.
//
// STAGE 2 (this file): the indexer is implemented. Below `index_topk` cached
// positions, the reference indexer itself returns no selection —
//
//     if k.shape[2] <= self.index_topk:
//         return None
//
// — so attention stays dense and byte-identical to stage 1 in that regime; this
// is the "inertness" property the stage-2 tests check without needing real
// weights. Past `index_topk` (2048 for GLM-5.2), attention is restricted to the
// indexer's top-k selection.
//
// Absorbed-MLA: K/V are cached in latent form (`kv_lora_rank` + `qk_rope_head_dim`
// per token, one head). `embed_q` / `unembed_out` hold the per-head halves of
// `kv_b_proj`; decode lifts the query into latent space and gathers only the
// indexer's selected positions, prefill expands the latents back into per-head
// K/V and masks. GLM-5.3 ships the weights in this layout and skips the indexer
// on its "shared" layers, which reuse the previous layer's top-k selection.
// Checkpoints that still carry `kv_b_proj` (GLM-5.2, DeepSeek V3.2) are converted
// in `sanitize`. The blocked sparse-prefill path of the reference is not ported.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct DeepseekV32Configuration: Codable, Sendable {
    var vocabSize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var moeIntermediateSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var nSharedExperts: Int?
    var nRoutedExperts: Int?
    var routedScalingFactor: Float
    var kvLoraRank: Int
    var qLoraRank: Int
    var qkRopeHeadDim: Int
    var vHeadDim: Int
    var qkNopeHeadDim: Int
    var normTopkProb: Bool
    var nGroup: Int?
    var topkGroup: Int?
    var numExpertsPerTok: Int?
    var moeLayerFreq: Int
    var firstKDenseReplace: Int
    var maxPositionEmbeddings: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    /// Kept as a stored property so the synthesised `Encodable` conformance has
    /// somewhere to put the key; `ropeTheta`/`ropeScaling` above are what the model
    /// actually reads.
    var ropeParameters: [String: StringOrNumber]?
    var attentionBias: Bool

    // DSA-specific.
    var indexTopk: Int
    var indexNHeads: Int
    var indexHeadDim: Int
    /// Whether the indexer's own RoPE is "traditional" (rotates adjacent pairs) vs
    /// the default "half-split" scheme. Not necessarily the same as the main
    /// attention's rope, which this port (matching the mlx-lm reference) always
    /// builds with `traditional: true` regardless of this field.
    var indexerRopeInterleave: Bool
    var numNextnPredictLayers: Int
    /// Per-layer "full" (own indexer) or "shared" (reuse the previous layer's top-k).
    var indexerTypes: [String]
    var indexTopkFreq: Int
    var indexSkipTopkOffset: Int

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case normTopkProb = "norm_topk_prob"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeLayerFreq = "moe_layer_freq"
        case firstKDenseReplace = "first_k_dense_replace"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case ropeParameters = "rope_parameters"
        case attentionBias = "attention_bias"
        case indexTopk = "index_topk"
        case indexNHeads = "index_n_heads"
        case indexHeadDim = "index_head_dim"
        case indexerRopeInterleave = "indexer_rope_interleave"
        case numNextnPredictLayers = "num_nextn_predict_layers"
        case indexerTypes = "indexer_types"
        case indexTopkFreq = "index_topk_freq"
        case indexSkipTopkOffset = "index_skip_topk_offset"
    }

    /// `index_topk_pattern` is read-only input: it is folded into `indexerTypes`.
    private enum PatternKeys: String, CodingKey {
        case indexTopkPattern = "index_topk_pattern"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try c.decode(Int.self, forKey: .numKeyValueHeads)
        nSharedExperts = try c.decodeIfPresent(Int.self, forKey: .nSharedExperts)
        nRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .nRoutedExperts)
        routedScalingFactor = try c.decode(Float.self, forKey: .routedScalingFactor)
        kvLoraRank = try c.decode(Int.self, forKey: .kvLoraRank)
        qLoraRank = try c.decode(Int.self, forKey: .qLoraRank)
        qkRopeHeadDim = try c.decode(Int.self, forKey: .qkRopeHeadDim)
        vHeadDim = try c.decode(Int.self, forKey: .vHeadDim)
        qkNopeHeadDim = try c.decode(Int.self, forKey: .qkNopeHeadDim)
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        nGroup = try c.decodeIfPresent(Int.self, forKey: .nGroup)
        topkGroup = try c.decodeIfPresent(Int.self, forKey: .topkGroup)
        numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok)
        moeLayerFreq = try c.decodeIfPresent(Int.self, forKey: .moeLayerFreq) ?? 1
        firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 0
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 2048
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false

        indexTopk = try c.decodeIfPresent(Int.self, forKey: .indexTopk) ?? 2048
        indexNHeads = try c.decodeIfPresent(Int.self, forKey: .indexNHeads) ?? 0
        indexHeadDim = try c.decodeIfPresent(Int.self, forKey: .indexHeadDim) ?? 0
        // The reference Indexer hardcodes traditional=True; treat absence the same way.
        indexerRopeInterleave =
            try c.decodeIfPresent(Bool.self, forKey: .indexerRopeInterleave) ?? true
        numNextnPredictLayers =
            try c.decodeIfPresent(Int.self, forKey: .numNextnPredictLayers) ?? 0

        // Mirrors glm_moe_dsa.ModelArgs.__post_init__: explicit list, then pattern
        // string ("F"/"S") or list, then the freq/offset formula. DeepSeek V3.2
        // configs have none of these and resolve to all "full".
        indexTopkFreq = max(try c.decodeIfPresent(Int.self, forKey: .indexTopkFreq) ?? 1, 1)
        indexSkipTopkOffset = try c.decodeIfPresent(Int.self, forKey: .indexSkipTopkOffset) ?? 2
        let patterns = try decoder.container(keyedBy: PatternKeys.self)
        if let explicit = try c.decodeIfPresent([String].self, forKey: .indexerTypes) {
            indexerTypes = explicit
        } else if let pattern = try? patterns.decodeIfPresent(
            String.self, forKey: .indexTopkPattern)
        {
            indexerTypes = pattern.map { $0 == "S" ? "shared" : "full" }
        } else if let pattern = try? patterns.decodeIfPresent(
            [String].self, forKey: .indexTopkPattern)
        {
            indexerTypes = pattern
        } else {
            let freq = indexTopkFreq
            let offset = indexSkipTopkOffset
            indexerTypes = (0 ..< numHiddenLayers).map {
                max($0 - offset + 1, 0) % freq == 0 ? "full" : "shared"
            }
        }

        // GLM-5.2 carries no top-level `rope_theta`/`rope_scaling`; both live inside a
        // `rope_parameters` dict, which mlx-lm unpacks in ModelArgs.__post_init__.
        // DeepSeek's own V3.2 configs keep the flat layout, so accept either.
        let ropeParameters = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)
        self.ropeParameters = ropeParameters

        if let flat = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) {
            ropeTheta = flat
        } else if case .float(let theta) = ropeParameters?["rope_theta"] {
            ropeTheta = theta
        } else if case .int(let theta) = ropeParameters?["rope_theta"] {
            ropeTheta = Float(theta)
        } else {
            throw DecodingError.keyNotFound(
                CodingKeys.ropeTheta,
                .init(
                    codingPath: decoder.codingPath,
                    debugDescription:
                        "no rope_theta at the top level or inside rope_parameters"))
        }

        if let flat = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling) {
            ropeScaling = flat
        } else if let ropeParameters,
            case .string(let type)? = ropeParameters["rope_type"], type != "default"
        {
            // Only forward rope_parameters as scaling when it actually describes a
            // scheme; "default" means plain RoPE and a non-nil dict here would make
            // the V3 stack look for yarn fields that are not there.
            ropeScaling = ropeParameters
        } else {
            ropeScaling = nil
        }
    }

    /// The V3.2 base stack is V3: same MLA projections, same grouped `noaux_tc` gate,
    /// same MoE block. Only the indexer is new, and stage 1 does not build it — so the
    /// model is constructed straight from the equivalent V3 configuration.
    var v3Configuration: DeepseekV3Configuration {
        DeepseekV3Configuration(
            vocabSize: vocabSize,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            moeIntermediateSize: moeIntermediateSize,
            numHiddenLayers: numHiddenLayers,
            numAttentionHeads: numAttentionHeads,
            numKeyValueHeads: numKeyValueHeads,
            nSharedExperts: nSharedExperts,
            nRoutedExperts: nRoutedExperts,
            routedScalingFactor: routedScalingFactor,
            kvLoraRank: kvLoraRank,
            qLoraRank: qLoraRank,
            qkRopeHeadDim: qkRopeHeadDim,
            vHeadDim: vHeadDim,
            qkNopeHeadDim: qkNopeHeadDim,
            normTopkProb: normTopkProb,
            nGroup: nGroup,
            topkGroup: topkGroup,
            numExpertsPerTok: numExpertsPerTok,
            moeLayerFreq: moeLayerFreq,
            firstKDenseReplace: firstKDenseReplace,
            maxPositionEmbeddings: maxPositionEmbeddings,
            rmsNormEps: rmsNormEps,
            ropeTheta: ropeTheta,
            ropeScaling: ropeScaling,
            attentionBias: attentionBias
        )
    }
}

/// The DSA "lightning indexer". Scores every cached key against the query with a
/// cheap dedicated head set (`index_n_heads` heads of `index_head_dim`, typically
/// much smaller than the main attention heads) and returns the `index_topk`
/// highest-scoring cached positions — or `nil` when the cache isn't yet longer
/// than `index_topk`, matching the reference's "nothing to select yet" regime.
///
/// RoPE here rotates only the first `qk_rope_head_dim` of each `index_head_dim`
/// vector — a partial rotary embedding, same trick `DeepseekV3Attention` doesn't
/// need (its rope-carrying half is a separate tensor) but the indexer's combined
/// per-head vector does. `RoPE(dimensions:)` in mlx-swift already leaves any
/// dimensions past `dimensions` untouched, so this is a plain `initializeRope`
/// call with `dims: qkRopeHeadDim` against a `headDim`-wide tensor — `traditional`
/// comes from `indexer_rope_interleave` (GLM-5.2 ships `true`), independently of
/// the main attention's rope, which this port always builds `traditional: true`
/// regardless (matching the mlx-lm reference's hardcoded value there).
class DeepseekV32Indexer: Module {
    let nHeads: Int
    let headDim: Int
    let indexTopk: Int
    let softmaxScale: Float

    let rope: RoPELayer
    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "k_norm") var kNorm: LayerNorm
    @ModuleInfo(key: "weights_proj") var weightsProj: Linear

    init(_ args: DeepseekV32Configuration) {
        self.nHeads = args.indexNHeads
        self.headDim = args.indexHeadDim
        self.indexTopk = args.indexTopk
        self.softmaxScale = pow(Float(headDim), -0.5)

        self._wqB.wrappedValue = Linear(args.qLoraRank, nHeads * headDim, bias: false)
        self._wk.wrappedValue = Linear(args.hiddenSize, headDim, bias: false)
        self._kNorm.wrappedValue = LayerNorm(dimensions: headDim)
        self._weightsProj.wrappedValue = Linear(args.hiddenSize, nHeads, bias: false)

        self.rope = initializeRope(
            dims: args.qkRopeHeadDim, base: args.ropeTheta,
            traditional: args.indexerRopeInterleave,
            scalingConfig: args.ropeScaling, maxPositionEmbeddings: args.maxPositionEmbeddings)
    }

    /// - Parameters:
    ///   - x: hidden states, `[B, L, hiddenSize]`.
    ///   - qr: the low-rank query after `q_a_layernorm(q_a_proj(x))` — the same
    ///     intermediate the main attention's `q_b_proj` reads, reused here rather
    ///     than recomputed.
    ///   - causalMask: a materialised boolean causal mask sized to the indexer's
    ///     own cache (`[B, 1, L, cachedLength]`), or `nil` when masking is a no-op
    ///     (decode: a single new query has no future cached position to exclude).
    ///   - cache: the indexer's own key cache (`CacheList`'s second slot) — holds
    ///     only `k`, never queried for `v`.
    /// - Returns: `[B, 1, L, indexTopk]` top-k indices into the cached axis, or
    ///   `nil` if the cache is not yet longer than `indexTopk`.
    func callAsFunction(
        _ x: MLXArray, qr: MLXArray, causalMask: MLXArray?, cache: KVCache?
    ) -> MLXArray? {
        let (b, s, _) = (x.dim(0), x.dim(1), x.dim(2))

        var q = wqB(qr).reshaped(b, s, nHeads, headDim).transposed(0, 2, 1, 3)
        var k = wk(x)
        k = kNorm(k)
        k = k.reshaped(b, 1, s, headDim)

        q = applyRotaryPosition(rope, to: q, cache: cache)
        k = applyRotaryPosition(rope, to: k, cache: cache)

        if let cache {
            (k, _) = cache.update(keys: k, values: MLXArray.zeros([b, 1, s, 0]))
        }

        guard k.dim(2) > indexTopk else { return nil }

        var scores = matmul(q, k.transposed(0, 1, 3, 2))
        scores = maximum(scores, 0)
        var weights = weightsProj(x) * (pow(Float(nHeads), -0.5) * softmaxScale)
        weights = weights.transposed(0, 2, 1)[.ellipsis, .newAxis]
        scores = scores * weights
        scores = scores.sum(axis: 1, keepDims: true)

        if let causalMask {
            scores = which(causalMask, scores, MLXArray(-Float.infinity))
        }

        return argPartition(-scores, kth: indexTopk - 1, axis: -1)[.ellipsis, ..<indexTopk]
    }
}

/// Absorbed-MLA attention with the DSA indexer. Port of
/// `glm_moe_dsa.GlmMoeDsaAttention` over `deepseek_v32.DeepseekV32Attention`.
class DeepseekV32Attention: Module {
    let qkRopeHeadDim: Int
    let kvLoraRank: Int
    let vHeadDim: Int
    let qkNopeHeadDim: Int
    let qHeadDim: Int
    let numHeads: Int
    var scale: Float
    /// "shared" layers have no indexer and reuse the previous layer's selection.
    let skipTopk: Bool

    let rope: RoPELayer
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "embed_q") var embedQ: MultiLinear
    @ModuleInfo(key: "unembed_out") var unembedOut: MultiLinear
    @ModuleInfo(key: "indexer") var indexer: DeepseekV32Indexer?

    init(_ args: DeepseekV32Configuration, layerIdx: Int) {
        let hiddenSize = args.hiddenSize
        self.numHeads = args.numAttentionHeads
        self.qkRopeHeadDim = args.qkRopeHeadDim
        self.kvLoraRank = args.kvLoraRank
        self.vHeadDim = args.vHeadDim
        self.qkNopeHeadDim = args.qkNopeHeadDim
        self.qHeadDim = args.qkNopeHeadDim + args.qkRopeHeadDim
        self.scale = pow(Float(qHeadDim), -0.5)
        self.skipTopk =
            layerIdx < args.indexerTypes.count && args.indexerTypes[layerIdx] == "shared"

        let qLoraRank = args.qLoraRank
        self._qAProj.wrappedValue = Linear(hiddenSize, qLoraRank, bias: args.attentionBias)
        self._qALayerNorm.wrappedValue = RMSNorm(dimensions: qLoraRank)
        self._qBProj.wrappedValue = Linear(qLoraRank, numHeads * qHeadDim, bias: false)

        self._kvAProjWithMqa.wrappedValue = Linear(
            hiddenSize, kvLoraRank + qkRopeHeadDim, bias: args.attentionBias)
        self._kvALayerNorm.wrappedValue = RMSNorm(dimensions: kvLoraRank)
        self._embedQ.wrappedValue = MultiLinear(
            inputDims: qkNopeHeadDim, outputDims: kvLoraRank, numHeads: numHeads)
        self._unembedOut.wrappedValue = MultiLinear(
            inputDims: kvLoraRank, outputDims: vHeadDim, numHeads: numHeads)
        self._oProj.wrappedValue = Linear(numHeads * vHeadDim, hiddenSize, bias: args.attentionBias)

        if let ropeScaling = args.ropeScaling {
            let mScaleAllDim = ropeScaling["mscale_all_dim"]?.asFloat() ?? 0.0
            if mScaleAllDim != 0 {
                let scalingFactor = ropeScaling["factor"]?.asFloat() ?? 1.0
                if scalingFactor > 1 {
                    let s = 0.1 * mScaleAllDim * log(scalingFactor) + 1.0
                    self.scale = self.scale * s * s
                }
            }
        }

        self.rope = initializeRope(
            dims: qkRopeHeadDim, base: args.ropeTheta, traditional: true,
            scalingConfig: args.ropeScaling, maxPositionEmbeddings: args.maxPositionEmbeddings)
        self._indexer.wrappedValue = skipTopk ? nil : DeepseekV32Indexer(args)
    }

    /// - Parameters:
    ///   - mask: boolean causal mask `[B, 1, L, S]` for prefill, `nil` for decode.
    ///   - cache: `CacheList(main)` for shared layers, `CacheList(main, indexer)` otherwise.
    ///   - prevTopkIndices: the previous layer's selection, used when `skipTopk`.
    /// - Returns: the attention output and this layer's top-k selection (or `nil`).
    func callAsFunction(
        _ x: MLXArray, mask: MLXArray?, cache: CacheList?, prevTopkIndices: MLXArray?
    ) -> (MLXArray, MLXArray?) {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))
        let mainCache = cache?[0]
        let indexerCache: KVCache? = (cache?.count ?? 0) > 1 ? cache?[1] : nil

        let qr = qALayerNorm(qAProj(x))
        let q = qBProj(qr).reshaped(B, L, numHeads, qHeadDim).transposed(0, 2, 1, 3)
        let splitQ = split(q, indices: [qkNopeHeadDim], axis: -1)
        var qNope = splitQ[0]
        var qPe = splitQ[1]

        let compressedKv = kvAProjWithMqa(x)
        let splitKv = split(compressedKv, indices: [kvLoraRank], axis: -1)
        var kvLatent = kvALayerNorm(splitKv[0]).expandedDimensions(axis: 1)
        var kPe = splitKv[1].reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)

        qPe = applyRotaryPosition(rope, to: qPe, cache: mainCache)
        kPe = applyRotaryPosition(rope, to: kPe, cache: mainCache)

        if let mainCache {
            (kvLatent, kPe) = mainCache.update(keys: kvLatent, values: kPe)
        }

        let topkIndices: MLXArray?
        if let indexer {
            topkIndices = indexer(x, qr: qr, causalMask: mask, cache: indexerCache)
        } else {
            topkIndices = prevTopkIndices
        }

        var finalMask = mask
        if let topkIndices {
            if L == 1 {
                let idx = topkIndices[0..., 0..., 0, 0..., .newAxis]
                kvLatent = takeAlong(
                    kvLatent, broadcast(idx, to: idx.shape.dropLast() + [kvLatent.dim(-1)]),
                    axis: 2)
                kPe = takeAlong(
                    kPe, broadcast(idx, to: idx.shape.dropLast() + [kPe.dim(-1)]), axis: 2)
                if let m = finalMask {
                    finalMask = takeAlong(m, topkIndices, axis: -1)
                }
            } else {
                var shape = topkIndices.shape
                shape[shape.count - 1] = kvLatent.dim(2)
                var sparseMask = MLXArray.zeros(shape, dtype: .bool)
                sparseMask = putAlong(sparseMask, topkIndices, values: MLXArray(true), axis: -1)
                if let m = finalMask {
                    sparseMask = sparseMask & m
                }
                finalMask = sparseMask
            }
        }

        // Keep the indexer cache in the graph even when its selection is unused,
        // so the graph does not grow across decode steps below index_topk.
        keepIndexerCacheEvaluated(main: mainCache, indexer: indexerCache)

        var peScores = matmul(qPe * scale, kPe.swappedAxes(-1, -2))
        if let finalMask {
            peScores = which(finalMask, peScores, finiteMinimum(for: peScores.dtype))
        }

        var output: MLXArray
        if L == 1 {
            qNope = embedQ(qNope)
            output = MLXFast.scaledDotProductAttention(
                queries: qNope, keys: kvLatent, values: kvLatent, scale: scale, mask: peScores)
            output = unembedOut(output)
        } else {
            let keys = embedQ(kvLatent, transpose: false)
            let values = unembedOut(kvLatent)
            output = MLXFast.scaledDotProductAttention(
                queries: qNope, keys: keys, values: values, scale: scale, mask: peScores)
        }

        output = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return (oProj(output), topkIndices)
    }

    private func keepIndexerCacheEvaluated(main: KVCache?, indexer: KVCache?) {
        guard
            let indexerKeys = (indexer as? KVCacheSimple)?.keys
                ?? (indexer as? RotatingKVCache)?.keys
        else { return }
        if let main = main as? KVCacheSimple, let keys = main.keys {
            main.keys = depends(input: keys, dependencies: [indexerKeys])
        } else if let main = main as? RotatingKVCache, let keys = main.keys {
            main.keys = depends(input: keys, dependencies: [indexerKeys])
        }
    }

    /// The most negative finite value of `dtype`, as the reference's `mx.finfo(dtype).min`.
    private func finiteMinimum(for dtype: DType) -> MLXArray {
        switch dtype {
        case .float16: return MLXArray(Float(-65504)).asType(.float16)
        case .bfloat16: return MLXArray(Float(-3.3895313892515355e38)).asType(.bfloat16)
        default: return MLXArray(-Float.greatestFiniteMagnitude).asType(dtype)
        }
    }
}

class DeepseekV32DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DeepseekV32Attention
    var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: DeepseekV32Configuration, layerIdx: Int) {
        self._selfAttn.wrappedValue = DeepseekV32Attention(args, layerIdx: layerIdx)

        let v3Config = args.v3Configuration
        if v3Config.nRoutedExperts != nil,
            layerIdx >= v3Config.firstKDenseReplace,
            layerIdx % v3Config.moeLayerFreq == 0
        {
            self.mlp = DeepseekV3MoE(config: v3Config)
        } else {
            self.mlp = DeepseekV3MLP(config: v3Config)
        }

        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXArray?, cache: CacheList?, prevTopkIndices: MLXArray?
    ) -> (MLXArray, MLXArray?) {
        let (r, topkIndices) = selfAttn(
            inputLayerNorm(x), mask: mask, cache: cache, prevTopkIndices: prevTopkIndices)
        let h = x + r
        let r2 = mlp(postAttentionLayerNorm(h))
        return (h + r2, topkIndices)
    }
}

public class DeepseekV32ModelInner: Module, LayerPartitionable, StreamableMoE {
    var args: DeepseekV32Configuration
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    var layers: [DeepseekV32DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    public var gpuLayerCount: Int? = nil
    public var streamExperts: Bool = false
    public var totalLayerCount: Int { layers.count }

    init(_ args: DeepseekV32Configuration) {
        self.args = args
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabSize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.numHiddenLayers).map {
            DeepseekV32DecoderLayer(args, layerIdx: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, cache: [CacheList]?) -> MLXArray {
        var h = embedTokens(x)
        let L = h.dim(1)

        // One boolean causal mask for every layer; each layer ANDs in its own top-k.
        let offset = cache?.first?[0].offset ?? 0
        let mask: MLXArray? = L > 1 ? createCausalMask(n: L, offset: offset) : nil

        var prevTopkIndices: MLXArray? = nil
        for (i, layer) in layers.enumerated() {
            let (out, topkIndices) = partitionedLayerCall(
                index: i, gpuLayerCount: gpuLayerCount, stream: streamExperts
            ) {
                layer(h, mask: mask, cache: cache?[i], prevTopkIndices: prevTopkIndices)
            }
            h = out
            prevTopkIndices = topkIndices
        }

        return norm(h)
    }
}

public class DeepseekV32Model: Module, LLMModel, LoRAModel {
    public var kvHeads: [Int] {
        Array(repeating: args.numAttentionHeads, count: args.numHiddenLayers)
    }

    var args: DeepseekV32Configuration
    public var model: DeepseekV32ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(_ args: DeepseekV32Configuration) {
        self.args = args
        self.model = DeepseekV32ModelInner(args)
        self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let typedCache: [CacheList]?
        if let cache {
            guard let cast = cache as? [CacheList] else {
                // A non-CacheList cache here means the caller didn't get it from
                // this model's own newCache(parameters:) — silently falling back
                // to `nil` (as `cache as? [CacheList]` alone would) would run a
                // full fresh prefill with no visible error, which reads as "the
                // model forgot everything" rather than "wrong cache type was
                // passed" and is far harder to debug. Fail loudly instead.
                fatalError(
                    "DeepseekV32Model requires CacheList-typed caches (from its own "
                        + "newCache(parameters:)); got \(type(of: cache))"
                )
            }
            typedCache = cast
        } else {
            typedCache = nil
        }
        let out = model(inputs, cache: typedCache)
        return lmHead(out)
    }

    /// `CacheList(main, indexer)` per layer, or `CacheList(main)` for shared layers,
    /// which run no indexer. Both children use the same concrete type so trimming
    /// and the indexer's offset bookkeeping stay in lockstep.
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< args.numHiddenLayers).map { layerIdx in
            let shared =
                layerIdx < args.indexerTypes.count && args.indexerTypes[layerIdx] == "shared"
            func make() -> KVCache {
                if let maxKVSize = parameters?.maxKVSize {
                    return RotatingKVCache(maxSize: maxKVSize, keep: 4)
                }
                return KVCacheSimple()
            }
            return shared ? CacheList(make()) : CacheList(make(), make())
        }
    }

    /// Multi-token-prediction layers sit past the main stack and have no module to
    /// load into — same reasoning as `DeepseekV3Model.isMultiTokenPredictionLayer`.
    func isMultiTokenPredictionLayer(_ key: String) -> Bool {
        guard key.starts(with: "model.layers.") else { return false }
        let parts = key.split(separator: ".")
        guard parts.count >= 3, let layerIdx = Int(parts[2]) else { return false }
        return layerIdx >= args.numHiddenLayers
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights = weights

        func dequant(weight: MLXArray, scaleInv: MLXArray) -> MLXArray {
            let bs = 128
            let (m, n) = (weight.dim(0), weight.dim(1))
            let padBottom = (bs - m % bs) % bs
            let padSide = (bs - n % bs) % bs

            var padded = padded(weight, widths: [.init((0, padBottom)), .init((0, padSide))])
            padded = padded.reshaped([(m + padBottom) / bs, bs, (n + padSide) / bs, bs])
            let scaled = padded * scaleInv[0..., .newAxis, 0..., .newAxis]
            return scaled.reshaped([m + padBottom, n + padSide])[0 ..< m, 0 ..< n]
        }

        for (key, value) in weights {
            if key.contains("weight_scale_inv") {
                let weightKey = key.replacingOccurrences(of: "_scale_inv", with: "")
                if let weight = weights[weightKey] {
                    let dequantized = dequant(weight: weight, scaleInv: value)
                    newWeights[weightKey] = dequantized
                }
            } else if newWeights[key] == nil {
                newWeights[key] = value
            }
        }

        for l in 0 ..< args.numHiddenLayers {
            let prefix = "model.layers.\(l)"
            for (_, projName) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
                for key in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).mlp.experts.0.\(projName).\(key)"
                    if weights[firstKey] != nil {
                        let joined = (0 ..< (args.nRoutedExperts ?? 1)).map {
                            weights["\(prefix).mlp.experts.\($0).\(projName).\(key)"]!
                        }
                        newWeights["\(prefix).mlp.switch_mlp.\(projName).\(key)"] = stacked(joined)
                    }
                }
            }
        }

        // Checkpoints with `kv_b_proj` (GLM-5.2, DeepSeek V3.2) are split into the
        // absorbed per-head `embed_q` / `unembed_out` that GLM-5.3 ships directly.
        for l in 0 ..< args.numHiddenLayers {
            let prefix = "model.layers.\(l).self_attn"
            guard let packed = newWeights.removeValue(forKey: "\(prefix).kv_b_proj.weight") else {
                continue
            }
            let scales = newWeights.removeValue(forKey: "\(prefix).kv_b_proj.scales")
            let biases = newWeights.removeValue(forKey: "\(prefix).kv_b_proj.biases")
            var bits = 0
            var groupSize = 0
            var full = packed
            if let scales {
                bits = packed.dim(-1) * 32 / args.kvLoraRank
                groupSize = args.kvLoraRank / scales.dim(-1)
                full = dequantized(
                    packed, scales: scales, biases: biases, groupSize: groupSize, bits: bits)
            }
            full = full.reshaped(args.numAttentionHeads, args.qkNopeHeadDim + args.vHeadDim, -1)
            let wk = full[0..., ..<args.qkNopeHeadDim, 0...].swappedAxes(-1, -2)
            let wv = full[0..., args.qkNopeHeadDim..., 0...]
            if scales != nil {
                let (qk, sk, bk) = quantized(wk, groupSize: groupSize, bits: bits)
                let (qv, sv, bv) = quantized(wv, groupSize: groupSize, bits: bits)
                newWeights["\(prefix).embed_q.weight"] = qk
                newWeights["\(prefix).embed_q.scales"] = sk
                newWeights["\(prefix).embed_q.biases"] = bk
                newWeights["\(prefix).unembed_out.weight"] = qv
                newWeights["\(prefix).unembed_out.scales"] = sv
                newWeights["\(prefix).unembed_out.biases"] = bv
            } else {
                newWeights["\(prefix).embed_q.weight"] = wk
                newWeights["\(prefix).unembed_out.weight"] = wv
            }
        }

        // Stage 1 dropped `.indexer.` weights entirely (no module to load them
        // into). Stage 2 has `DeepseekV32Indexer`, so they're kept — only
        // `.attn.compressor.` (an unrelated, still-unimplemented long-range
        // compression feature some checkpoints in this family also carry) is
        // still dropped.
        return newWeights.filter { key, _ in
            !isMultiTokenPredictionLayer(key)
                && !key.contains("rotary_emb.inv_freq")
                && !key.contains(".attn.compressor.")
        }
    }

    public var loraLayers: [Module] {
        model.layers
    }
}
