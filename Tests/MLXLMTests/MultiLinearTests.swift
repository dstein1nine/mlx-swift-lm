import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@Suite("MultiLinear")
struct MultiLinearTests {

    @Test("transpose true and false contract over the expected axes")
    func testShapes() {
        let layer = MultiLinear(inputDims: 16, outputDims: 32, numHeads: 4)

        let x = MLXArray.ones([1, 4, 3, 16])
        #expect(layer(x).shape == [1, 4, 3, 32])

        let latent = MLXArray.ones([1, 1, 5, 32])
        #expect(layer(latent, transpose: false).shape == [1, 4, 5, 16])
    }

    @Test("quantized forward matches the dequantized float forward in both modes")
    func testQuantizedMatchesFloat() {
        let layer = MultiLinear(inputDims: 64, outputDims: 128, numHeads: 4)
        let quantizedLayer = QuantizedMultiLinear(layer, groupSize: 32, bits: 8)
        let reference = MultiLinear(
            inputDims: 64, outputDims: 128, numHeads: 4,
            weight: dequantized(
                quantizedLayer.weight, scales: quantizedLayer.scales,
                biases: quantizedLayer.biases, groupSize: 32, bits: 8))

        let x = MLXRandom.normal([1, 4, 3, 64])
        let a = quantizedLayer(x)
        let b = reference(x)
        eval(a, b)
        #expect(a.shape == [1, 4, 3, 128])
        #expect((a - b).abs().max().item(Float.self) < 1e-2)

        let latent = MLXRandom.normal([1, 1, 5, 128])
        let c = quantizedLayer(latent, transpose: false)
        let d = reference(latent, transpose: false)
        eval(c, d)
        #expect(c.shape == [1, 4, 5, 64])
        #expect((c - d).abs().max().item(Float.self) < 1e-2)
    }

    @Test("toQuantized yields a QuantizedMultiLinear with the loader's weight keys")
    func testToQuantized() {
        let layer = MultiLinear(inputDims: 64, outputDims: 32, numHeads: 2)
        let quantizedLayer = layer.toQuantized(groupSize: 64, bits: 4, mode: .affine)
        #expect(quantizedLayer is QuantizedMultiLinear)

        let keys = Set(quantizedLayer.parameters().flattened().map(\.0))
        #expect(keys == ["weight", "scales", "biases"])
        #expect((quantizedLayer as? QuantizedMultiLinear)?.weight.shape == [2, 32, 8])
    }
}
