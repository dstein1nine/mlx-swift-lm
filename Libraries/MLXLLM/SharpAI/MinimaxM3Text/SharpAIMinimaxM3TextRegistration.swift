// SharpAI fork overlay: minimax_m3_text creator. Not part of upstream mlx-swift-lm.
//
// Same closure shape as LLMTypeRegistry: decode, validate, construct. LLMModelFactory is not
// edited; the overlay consumer merges these creators. Construction builds
// SharpAIMinimaxM3TextModel: per-head qk norm, dense and sparse-MoE layers, and the block indexer.
// Checked against mlx-lm logits on a truncated MiniMax-M3.1 checkpoint (SharpAIMinimaxM3GoldenTests).

import Foundation
import MLXLMCommon

public enum SharpAIMinimaxM3TextOverlay {

    /// Creators keyed by `model_type`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "minimax_m3_text": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIMinimaxM3TextConfiguration.self, from: data)
            try configuration.validateForConstruction()
            return SharpAIMinimaxM3TextModel(configuration)
        },
        // The language model of minimax_m3_vl; the vision tower is not loaded.
        "minimax_m3_vl": { data in
            let configuration = try JSONDecoder.json5().decode(
                VLWrapper.self, from: data
            ).textConfig
            try configuration.validateForConstruction()
            return SharpAIMinimaxM3TextModel(configuration)
        },
    ]

    private struct VLWrapper: Decodable {
        let textConfig: SharpAIMinimaxM3TextConfiguration
        enum CodingKeys: String, CodingKey { case textConfig = "text_config" }
    }
}
