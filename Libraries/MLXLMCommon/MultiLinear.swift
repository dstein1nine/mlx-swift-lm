// Copyright © 2026 Apple Inc.

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/mla.py

import Foundation
import MLX
import MLXNN

/// `numHeads` independent linear maps applied with one batched matmul.
///
/// `weight` is `[numHeads, outputDims, inputDims]`. With `transpose: true` (the
/// default) an input `[..., numHeads, L, inputDims]` becomes
/// `[..., numHeads, L, outputDims]`. With `transpose: false` the contraction runs
/// over `outputDims` instead, so `[..., 1 | numHeads, L, outputDims]` becomes
/// `[..., numHeads, L, inputDims]`.
///
/// Absorbed multi-head latent attention (DeepSeek V3.2, GLM-5.3) stores the
/// per-head halves of `kv_b_proj` this way and reuses the same weight on both
/// sides: `embed_q` lifts the query into the latent space during decode and, with
/// `transpose: false`, expands cached latents back into per-head keys during
/// prefill.
open class MultiLinear: Module, Quantizable {
    @ModuleInfo(key: "weight") public var weight: MLXArray

    public let inputDims: Int
    public let outputDims: Int
    public let numHeads: Int

    public init(inputDims: Int, outputDims: Int, numHeads: Int) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numHeads = numHeads

        let scale = sqrt(1.0 / Float(inputDims))
        self._weight.wrappedValue = MLXRandom.uniform(
            -scale ..< scale, [numHeads, outputDims, inputDims])

        super.init()
    }

    /// Initializer for subclasses that supply the weight array directly.
    public init(inputDims: Int, outputDims: Int, numHeads: Int, weight: MLXArray) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numHeads = numHeads
        self._weight.wrappedValue = weight
        super.init()
    }

    open func callAsFunction(_ x: MLXArray, transpose: Bool = true) -> MLXArray {
        if transpose {
            return matmul(x, weight.swappedAxes(-1, -2))
        } else {
            return matmul(x, weight)
        }
    }

    public func toQuantized(groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode) -> Module {
        QuantizedMultiLinear(self, groupSize: groupSize, bits: bits, mode: mode)
    }
}

/// Quantized ``MultiLinear``. `quantizedMatmul` broadcasts the leading head axis
/// of the packed weight against the input, so both `transpose` modes keep the
/// semantics of the float version.
open class QuantizedMultiLinear: MultiLinear, Quantized {
    @ModuleInfo(key: "scales") public var scales: MLXArray
    @ModuleInfo(key: "biases") public var biases: MLXArray?

    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    public init(
        _ other: MultiLinear, groupSize: Int = 64, bits: Int = 4,
        mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        let (quantizedWeight, scales, biases) = MLX.quantized(
            other.weight, groupSize: groupSize, bits: bits, mode: mode)
        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases

        super.init(
            inputDims: other.inputDims, outputDims: other.outputDims, numHeads: other.numHeads,
            weight: quantizedWeight)

        self.freeze()
    }

    override open func callAsFunction(_ x: MLXArray, transpose: Bool = true) -> MLXArray {
        quantizedMatmul(
            x, weight, scales: scales, biases: biases, transpose: transpose,
            groupSize: groupSize, bits: bits, mode: mode)
    }
}
