//
//  SharpAIMinimaxM3TextModel.swift
//  MLXLLM
//
//  SharpAI fork overlay: text decoder for MiniMax M3 (model_type minimax_m3_text, and the
//  language model of minimax_m3_vl). Not part of upstream mlx-swift-lm.
//
//  Ported from mlx-lm 0.32.2 `models/minimax_m3_vl.py`. The module tree uses the same names,
//  so MLX checkpoints and their per-layer quantization entries map one to one.
//
//  Pre-norm decoder. Dense layers (moe_layer_freq 0) use a SwiGLU-OAI MLP. MoE layers use a
//  sigmoid router: top-k on score + bias, weights from the unbiased scores renormalized over
//  the chosen experts and scaled by routed_scaling_factor, plus one shared expert.
//  Attention is GQA with Gemma-style per-head qk RMSNorm and partial RoPE. A sparse layer also
//  runs a block indexer that keeps the top-k key blocks per indexer head; it changes nothing
//  until the context holds more than topk blocks.
//
//  Each layer keeps a CacheList(main KV, indexer keys). The indexer keys carry RoPE, so they
//  cannot be rebuilt from the main cache.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Gemma-style RMSNorm: scales by (1 + weight). The weight starts at 0.
class SharpAIMinimaxM3TextRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        _weight.wrappedValue = MLXArray.zeros([dimensions])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: 1 + weight, eps: eps)
    }
}

/// SwiGLU-OAI: g = min(gate, limit), out = (clip(up, -limit, limit) + 1) * (g * sigmoid(alpha * g)).
func sharpAIMinimaxM3SwigluOAI(gate: MLXArray, up: MLXArray, alpha: Float, limit: Float)
    -> MLXArray
{
    let g = minimum(gate, limit)
    let u = clip(up, min: -limit, max: limit)
    return (u + 1) * (g * sigmoid(alpha * g))
}

/// Dense SwiGLU-OAI MLP. Used by the dense layers and by the shared expert.
class SharpAIMinimaxM3TextMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    let alpha: Float
    let limit: Float

    init(hiddenSize: Int, intermediateSize: Int, alpha: Float, limit: Float) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        self.alpha = alpha
        self.limit = limit
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(
            sharpAIMinimaxM3SwigluOAI(
                gate: gateProj(x), up: upProj(x), alpha: alpha, limit: limit))
    }
}

/// Routed experts plus one shared expert. Checkpoint keys: `block_sparse_moe.{gate,
/// e_score_correction_bias, switch_mlp, shared_experts}`.
class SharpAIMinimaxM3TextSparseMoE: Module, UnaryLayer {
    @ModuleInfo(key: "gate") var gate: Linear
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: SharpAIMinimaxM3TextMLP
    let topK: Int
    let routedScalingFactor: Float

    init(_ config: SharpAIMinimaxM3TextConfiguration) {
        let alpha = config.swigluAlpha
        let limit = config.swigluLimit
        let numExperts = config.numLocalExperts ?? 0
        self.topK = config.numExpertsPerTok ?? 1
        self.routedScalingFactor = config.routedScalingFactor

        _gate.wrappedValue = Linear(config.hiddenSize, numExperts, bias: false)
        _eScoreCorrectionBias.wrappedValue = MLXArray.zeros([numExperts])
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.intermediateSize ?? 0,
            numExperts: numExperts,
            activation: { g in
                let clamped = minimum(g, limit)
                return clamped * sigmoid(alpha * clamped)
            },
            activationProduct: { g, u in
                sharpAIMinimaxM3SwigluOAI(gate: g, up: u, alpha: alpha, limit: limit)
            },
            bias: false)
        _sharedExperts.wrappedValue = SharpAIMinimaxM3TextMLP(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.numSharedExperts
                * (config.sharedIntermediateSize ?? config.intermediateSize ?? 0),
            alpha: alpha, limit: limit)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shared = sharedExperts(x)

        // Route in the weight dtype; upcast only the logits.
        let scores = sigmoid(gate(x).asType(.float32))
        let selection = scores + eScoreCorrectionBias
        let inds = argPartition(-selection, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = takeAlong(scores, inds, axis: -1)
        weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
        weights = (weights * routedScalingFactor).asType(x.dtype)

        let routed = (switchMLP(x, inds) * weights[.ellipsis, .newAxis]).sum(axis: -2)
        return routed + shared
    }
}

/// GQA attention with per-head qk RMSNorm, partial RoPE, and, on sparse layers, a block
/// indexer whose weights sit on this module (`index_{q,k}_proj`, `index_{q,k}_norm`).
class SharpAIMinimaxM3TextAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPE
    let sparse: SharpAIMinimaxM3TextSparseAttentionConfiguration?
    let indexRope: RoPE?

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: SharpAIMinimaxM3TextRMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: SharpAIMinimaxM3TextRMSNorm?
    @ModuleInfo(key: "index_q_proj") var indexQProj: Linear?
    @ModuleInfo(key: "index_k_proj") var indexKProj: Linear?
    @ModuleInfo(key: "index_q_norm") var indexQNorm: SharpAIMinimaxM3TextRMSNorm?
    @ModuleInfo(key: "index_k_norm") var indexKNorm: SharpAIMinimaxM3TextRMSNorm?

    init(_ config: SharpAIMinimaxM3TextConfiguration, layer: Int) {
        self.numHeads = config.attentionHeads
        self.numKVHeads = config.kvHeads
        self.headDim = config.resolvedHeadDim
        self.scale = pow(Float(config.resolvedHeadDim), -0.5)
        self.rope = RoPE(
            dimensions: config.rotaryDimensions, traditional: false, base: config.ropeTheta)

        _qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)
        if config.useQkNorm {
            _qNorm.wrappedValue = SharpAIMinimaxM3TextRMSNorm(
                dimensions: headDim, eps: config.rmsNormEps)
            _kNorm.wrappedValue = SharpAIMinimaxM3TextRMSNorm(
                dimensions: headDim, eps: config.rmsNormEps)
        }

        if config.isSparseAttentionLayer(layer), let sparse = config.sparseAttention {
            self.sparse = sparse
            // The reference truncates the index RoPE tables to min(rotary_dim, index_dim).
            self.indexRope = RoPE(
                dimensions: min(config.rotaryDimensions, sparse.indexDim), traditional: false,
                base: config.ropeTheta)
            _indexQProj.wrappedValue = Linear(
                config.hiddenSize, sparse.indexHeads * sparse.indexDim, bias: false)
            _indexKProj.wrappedValue = Linear(config.hiddenSize, sparse.indexDim, bias: false)
            _indexQNorm.wrappedValue = SharpAIMinimaxM3TextRMSNorm(
                dimensions: sparse.indexDim, eps: config.rmsNormEps)
            _indexKNorm.wrappedValue = SharpAIMinimaxM3TextRMSNorm(
                dimensions: sparse.indexDim, eps: config.rmsNormEps)
        } else {
            self.sparse = nil
            self.indexRope = nil
        }
        super.init()
    }

    /// Keeps only the top-k key blocks per query and indexer head, widened to every query
    /// head of the group. Returns nil when every block fits the budget, so nothing is dropped.
    private func blockMask(
        _ x: MLXArray, sparse: SharpAIMinimaxM3TextSparseAttentionConfiguration,
        offset: RoPEOffset?, cache: KVCache?
    ) -> MLXArray? {
        let (B, L) = (x.dim(0), x.dim(1))
        let (H, D) = (sparse.indexHeads, sparse.indexDim)

        // Cache the index keys even when no block is dropped: later steps need them.
        var k = indexKNorm!(indexKProj!(x).reshaped(B, L, 1, D)).transposed(0, 2, 1, 3)
        k = applyRotaryPosition(indexRope!, to: k, offset: offset)
        if let cache {
            (k, _) = cache.update(keys: k, values: MLXArray.zeros([B, 1, L, 0], dtype: k.dtype))
        }

        let blockSize = sparse.blockSize
        let topk = sparse.topkBlocks
        let keyCount = k.dim(2)
        let fullBlocks = keyCount / blockSize
        let tailKeys = keyCount % blockSize
        let blockCount = fullBlocks + (tailKeys > 0 ? 1 : 0)
        if topk >= blockCount { return nil }

        var q = indexQNorm!(indexQProj!(x).reshaped(B, L, H, D)).transposed(0, 2, 1, 3)
        q = applyRotaryPosition(indexRope!, to: q, offset: offset)

        // Blocks are cut over cache slots, so queries are placed by slot.
        let qpos = MLXArray((keyCount - L ..< keyCount).map { Int32($0) })
        let causal =
            MLXArray((0 ..< keyCount).map { Int32($0) }).reshaped(1, keyCount)
            .<= qpos.reshaped(L, 1)
        var scores = matmul(q.asType(.float32), k.asType(.float32).transposed(0, 1, 3, 2))
        // A future key must not win a block, so mask before pooling.
        scores = which(causal, scores, MLXArray(-Float.infinity))
        let split = fullBlocks * blockSize
        var blockScores = scores[.ellipsis, ..<split]
            .reshaped(B, H, L, fullBlocks, blockSize).max(axis: -1)
        if tailKeys > 0 {
            let tail = scores[.ellipsis, split...].max(axis: -1, keepDims: true)
            blockScores = concatenated([blockScores, tail], axis: -1)
        }
        if sparse.localBlocks > 0 {
            let qBlock = floorDivide(qpos, MLXArray(Int32(blockSize))).reshaped(L, 1)
            let ids = MLXArray((0 ..< blockCount).map { Int32($0) }).reshaped(1, blockCount)
            let isLocal = (ids .<= qBlock) & (ids .> qBlock - MLXArray(Int32(sparse.localBlocks)))
            blockScores = which(isLocal, MLXArray(Float.infinity), blockScores)
        }

        let inds = argPartition(-blockScores, kth: topk - 1, axis: -1)[.ellipsis, ..<topk]
        var keep = putAlong(
            MLXArray.zeros(blockScores.shape, dtype: .bool), inds, values: MLXArray(true),
            axis: -1)
        // A kept block still holds future keys, so mask again.
        keep = repeated(keep, count: blockSize, axis: -1)[.ellipsis, ..<keyCount] & causal
        return repeated(keep, count: numHeads / H, axis: 1)
    }

    /// Query rows per attention pass on long sparse prompt chunks. The block mask is
    /// `[64 heads, rows, keys]`, so whole chunks cost O(L²) memory (8K tokens: tens of GB).
    /// Tiles append to the cache in order, so each row sees the same keys as in one pass.
    /// `MINIMAX_ATTN_TILE=N`; 0 disables.
    static let queryTile =
        Int(ProcessInfo.processInfo.environment["MINIMAX_ATTN_TILE"] ?? "") ?? 1024

    /// - Parameter cache: `CacheList(main KV, indexer keys)` for this layer, or nil.
    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: CacheList?
    ) -> MLXArray {
        let (L, tile) = (x.dim(1), Self.queryTile)
        guard sparse != nil, cache != nil, tile > 0, L > tile, case .causal = mask else {
            return attend(x, mask: mask, cache: cache)
        }
        let tiles = stride(from: 0, to: L, by: tile).map {
            attend(x[0..., $0 ..< min($0 + tile, L)], mask: .causal, cache: cache)
        }
        return concatenated(tiles, axis: 1)
    }

    private func attend(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: CacheList?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let mainCache = cache?[0]
        let offset = mainCache?.ropeOffset

        var queries = qProj(x).reshaped(B, L, numHeads, headDim)
        var keys = kProj(x).reshaped(B, L, numKVHeads, headDim)
        let values = vProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)
        if let qNorm, let kNorm {
            queries = qNorm(queries)
            keys = kNorm(keys)
        }
        queries = applyRotaryPosition(rope, to: queries.transposed(0, 2, 1, 3), offset: offset)
        keys = applyRotaryPosition(rope, to: keys.transposed(0, 2, 1, 3), offset: offset)

        var attentionMask = mask
        if let sparse, let block = blockMask(x, sparse: sparse, offset: offset, cache: cache?[1]) {
            attentionMask = .array(block)
        }

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: mainCache, scale: scale,
            mask: attentionMask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)
        return oProj(output)
    }
}

class SharpAIMinimaxM3TextDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIMinimaxM3TextAttention
    @ModuleInfo(key: "block_sparse_moe") var moe: SharpAIMinimaxM3TextSparseMoE?
    @ModuleInfo(key: "mlp") var mlp: SharpAIMinimaxM3TextMLP?
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: SharpAIMinimaxM3TextRMSNorm
    @ModuleInfo(key: "post_attention_layernorm")
    var postAttentionLayerNorm: SharpAIMinimaxM3TextRMSNorm
    /// The next MoE layer, for expert prefetch during streamed prompt chunks.
    var promptLink: SharpAIMinimaxM3TextPromptLink?

    init(_ config: SharpAIMinimaxM3TextConfiguration, layer: Int) {
        _selfAttn.wrappedValue = SharpAIMinimaxM3TextAttention(config, layer: layer)
        if config.isMoELayer(layer) {
            _moe.wrappedValue = SharpAIMinimaxM3TextSparseMoE(config)
        } else {
            _mlp.wrappedValue = SharpAIMinimaxM3TextMLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.denseWidth ?? 0,
                alpha: config.swigluAlpha,
                limit: config.swigluLimit)
        }
        _inputLayerNorm.wrappedValue = SharpAIMinimaxM3TextRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = SharpAIMinimaxM3TextRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: CacheList?
    ) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let normed = postAttentionLayerNorm(h)
        if let moe, let promptLink, SwitchGLU.usesPromptPipeline(tokens: h.dim(1)) {
            moe.switchMLP.promptLookahead = ExpertLookahead(
                predicted: promptLink.predict(h), target: promptLink.next.switchMLP)
        }
        if let moe {
            return h + moe(normed)
        }
        return h + mlp!(normed)
    }
}

/// Guesses the next MoE layer's experts by running its router on this layer's
/// post-attention hidden state, so their reads overlap this layer's compute.
final class SharpAIMinimaxM3TextPromptLink {
    let norm: SharpAIMinimaxM3TextRMSNorm
    let next: SharpAIMinimaxM3TextSparseMoE
    let topK: Int

    init(norm: SharpAIMinimaxM3TextRMSNorm, next: SharpAIMinimaxM3TextSparseMoE, topK: Int) {
        self.norm = norm
        self.next = next
        self.topK = topK
    }

    func predict(_ h: MLXArray) -> MLXArray {
        let scores = sigmoid(next.gate(norm(h)).asType(.float32)) + next.eScoreCorrectionBias
        return argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK].flattened()
    }
}

class SharpAIMinimaxM3TextInner: Module, StreamableMoE {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    /// SSD streaming happens inside SwitchGLU; this flag only tells the server it applies.
    var streamExperts = false
    let layers: [SharpAIMinimaxM3TextDecoderLayer]
    let norm: SharpAIMinimaxM3TextRMSNorm

    init(_ config: SharpAIMinimaxM3TextConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map {
            SharpAIMinimaxM3TextDecoderLayer(config, layer: $0)
        }
        self.norm = SharpAIMinimaxM3TextRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
        for i in layers.indices.dropLast() where layers[i].moe != nil {
            guard let next = layers[i + 1].moe else { continue }
            layers[i].promptLink = SharpAIMinimaxM3TextPromptLink(
                norm: layers[i + 1].postAttentionLayerNorm, next: next,
                topK: SwitchGLU.promptPredictTopK ?? next.topK)
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [CacheList]?) -> MLXArray {
        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first?[0])
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

class SharpAIMinimaxM3TextModel: Module, LLMModel {
    let vocabularySize: Int
    let kvHeads: [Int]
    let configuration: SharpAIMinimaxM3TextConfiguration
    let model: SharpAIMinimaxM3TextInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ configuration: SharpAIMinimaxM3TextConfiguration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        self.kvHeads = Array(repeating: configuration.kvHeads, count: configuration.hiddenLayers)
        self.model = SharpAIMinimaxM3TextInner(configuration)
        if !configuration.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                configuration.hiddenSize, configuration.vocabularySize, bias: false)
        }
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: Self.typedCaches(cache))
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    /// One `CacheList(main, indexer)` per layer. Bounded (rotating) caches are not supported,
    /// so `maxKVSize` is ignored.
    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< configuration.hiddenLayers).map { _ -> KVCache in
            CacheList(KVCacheSimple(), KVCacheSimple())
        }
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIMinimaxM3TextSanitizer.remapCheckpoint(weights, configuration: configuration)
    }

    private static func typedCaches(_ cache: [KVCache]?) -> [CacheList]? {
        guard let cache else { return nil }
        let typed = cache.compactMap { $0 as? CacheList }
        precondition(
            typed.count == cache.count,
            "minimax_m3_text expects one CacheList per layer, as produced by newCache")
        return typed
    }
}

extension SharpAIMinimaxM3TextModel: LoRAModel {
    var loraLayers: [Module] {
        model.layers
    }
}

extension SharpAIMinimaxM3TextModel {
    /// The template runs thinking in "adaptive" mode with no off switch: the model opens
    /// `<mm:think>` itself when it chooses to reason. Plain tokens, not special ones.
    var reasoningConfig: ReasoningConfig? {
        ReasoningConfig(
            startDelimiter: "<mm:think>", endDelimiter: "</mm:think>", promptStrategy: .alwaysOn)
    }
}
