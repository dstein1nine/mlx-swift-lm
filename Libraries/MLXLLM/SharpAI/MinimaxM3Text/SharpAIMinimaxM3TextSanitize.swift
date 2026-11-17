// SharpAI fork overlay: minimax_m3_text checkpoint key remapping. Not part of upstream mlx-swift-lm.
//
// The module tree uses mlx-lm's names, so an MLX checkpoint needs only:
//   - the `language_model.` wrapper prefix of minimax_m3_vl stripped;
//   - vision, projector, MTP and rotary tensors, and a tied `lm_head.weight`, dropped;
//   - per-expert `block_sparse_moe.experts.<i>.{w1,w2,w3}` stacked into `switch_mlp`.
// HF checkpoints with FP8 `weight_scale_inv` tensors are not handled; convert them with mlx-lm.

import Foundation
import MLX

public enum SharpAIMinimaxM3TextSanitizer {

    static let droppedPrefixes = ["vision_tower.", "multi_modal_projector.", "patch_merge_mlp."]

    /// Maps a checkpoint onto this model's module keys. See the file header.
    public static func remapCheckpoint(
        _ weights: [String: MLXArray], configuration: SharpAIMinimaxM3TextConfiguration
    ) -> [String: MLXArray] {
        var out = [String: MLXArray]()
        out.reserveCapacity(weights.count)
        // stacked key -> expert index -> tensor
        var expertParts = [String: [Int: MLXArray]]()

        for (key, value) in weights {
            if droppedPrefixes.contains(where: key.hasPrefix) { continue }
            let name =
                key.hasPrefix("language_model.")
                ? String(key.dropFirst("language_model.".count)) : key
            if name.hasPrefix("mtp") || name.contains(".mtp.") || name.contains("rotary_emb") {
                continue
            }
            if let layer = decoderLayerIndex(name), layer >= configuration.hiddenLayers {
                continue
            }
            if configuration.tieWordEmbeddings, name == "lm_head.weight" { continue }
            if let parsed = expertKey(name) {
                expertParts[parsed.stackedKey, default: [:]][parsed.index] = value
            } else {
                out[name] = value
            }
        }
        for (key, byIndex) in expertParts {
            out[key] = stacked(byIndex.keys.sorted().map { byIndex[$0]! })
        }
        return out
    }

    /// Layer index for `model.layers.<n>.…` keys, nil otherwise.
    private static func decoderLayerIndex(_ key: String) -> Int? {
        let prefix = "model.layers."
        guard key.hasPrefix(prefix) else { return nil }
        return Int(key.dropFirst(prefix.count).prefix { $0 != "." })
    }

    /// `….block_sparse_moe.experts.<i>.<w1|w2|w3>.<part>` -> (`….switch_mlp.<proj>.<part>`, i).
    private static func expertKey(_ key: String) -> (stackedKey: String, index: Int)? {
        guard let range = key.range(of: ".block_sparse_moe.experts.") else { return nil }
        let parts = key[range.upperBound...].split(separator: ".")
        guard parts.count == 3, let index = Int(parts[0]) else { return nil }
        let proj: String
        switch parts[1] {
        case "w1", "gate_proj": proj = "gate_proj"
        case "w3", "up_proj": proj = "up_proj"
        case "w2", "down_proj": proj = "down_proj"
        default: return nil
        }
        let owner = key[..<range.lowerBound]
        return ("\(owner).block_sparse_moe.switch_mlp.\(proj).\(parts[2])", index)
    }
}
