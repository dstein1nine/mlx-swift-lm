import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

/// Kimi K2.x MLX checkpoints: `kimi_k25` wraps a DeepSeek-V3 text model inside a VLM
/// shell (`text_config`, `language_model.` weight prefix) and current conversions ship
/// attention in the absorbed `embed_q` / `unembed_out` layout. They load through the
/// V3.2 model with no indexer, which is plain V3 attention.
@Suite("Kimi K2 / kimi_k25 via DeepseekV32")
struct KimiK2Tests {

    private func kimiStyleConfigData(numHiddenLayers: Int = 3) -> Data {
        let json = """
            {
                "architectures": ["KimiK25ForConditionalGeneration"],
                "model_type": "kimi_k25",
                "auto_map": {},
                "text_config": {
                    "model_type": "kimi_k2",
                    "vocab_size": 128,
                    "hidden_size": 64,
                    "intermediate_size": 128,
                    "moe_intermediate_size": 32,
                    "num_hidden_layers": \(numHiddenLayers),
                    "num_attention_heads": 4,
                    "num_key_value_heads": 4,
                    "n_shared_experts": 1,
                    "n_routed_experts": 4,
                    "routed_scaling_factor": 2.827,
                    "kv_lora_rank": 16,
                    "q_lora_rank": 32,
                    "qk_rope_head_dim": 8,
                    "v_head_dim": 16,
                    "qk_nope_head_dim": 16,
                    "norm_topk_prob": true,
                    "n_group": 1,
                    "topk_group": 1,
                    "num_experts_per_tok": 2,
                    "moe_layer_freq": 1,
                    "first_k_dense_replace": 1,
                    "max_position_embeddings": 262144,
                    "rms_norm_eps": 1e-05,
                    "rope_theta": 50000.0,
                    "rope_scaling": {
                        "beta_fast": 32.0, "beta_slow": 1.0, "factor": 64.0,
                        "mscale": 1.0, "mscale_all_dim": 1.0,
                        "original_max_position_embeddings": 4096, "type": "yarn"
                    },
                    "attention_bias": false,
                    "scoring_func": "sigmoid",
                    "topk_method": "noaux_tc",
                    "num_nextn_predict_layers": 0
                }
            }
            """
        return json.data(using: .utf8)!
    }

    @Test("kimi_k25 config decodes out of text_config with every layer dense")
    func testWrappedConfigDecodes() throws {
        let cfg = try JSONDecoder().decode(
            DeepseekV32Configuration.self, from: kimiStyleConfigData())
        #expect(cfg.hiddenSize == 64)
        #expect(cfg.numHiddenLayers == 3)
        #expect(cfg.ropeTheta == 50000)
        #expect(cfg.ropeScaling?["type"] == .string("yarn"))
        #expect(cfg.indexNHeads == 0)
        #expect(cfg.indexerTypes == ["shared", "shared", "shared"])
    }

    @Test("kimi_k2 and kimi_k25 map to DeepseekV32Model through the factory registry")
    func testRegistryEntries() async throws {
        for modelType in ["kimi_k2", "kimi_k25"] {
            let model = try await LLMTypeRegistry.shared.createModel(
                configuration: kimiStyleConfigData(), modelType: modelType)
            #expect(model is DeepseekV32Model, "\(modelType)")
        }
    }

    @Test("No indexer: single-cache layers, finite prefill and decode with a real cache")
    func testDenseForwardWithRealCache() throws {
        let cfg = try JSONDecoder().decode(
            DeepseekV32Configuration.self, from: kimiStyleConfigData())
        let model = DeepseekV32Model(cfg)
        for layer in model.model.layers {
            #expect(layer.selfAttn.indexer == nil)
        }

        let cache = model.newCache(parameters: nil)
        #expect(cache.count == 3)
        for c in cache {
            #expect((c as? CacheList)?.count == 1)
        }

        let prompt = MLXArray(0 ..< 6).reshaped(1, 6)
        let prefill = model(prompt, cache: cache)
        eval(prefill)
        #expect(prefill.shape == [1, 6, 128])
        #expect(!prefill.sum().item(Float.self).isNaN)

        let next = MLXArray([7]).reshaped(1, 1)
        let step = model(next, cache: cache)
        eval(step)
        #expect(step.shape == [1, 1, 128])
        #expect(!step.sum().item(Float.self).isNaN)
        #expect((cache[0] as? CacheList)?[0].offset == 7)
    }

    @Test("sanitize strips the language_model. wrapper prefix and keeps absorbed attention")
    func testSanitizeStripsWrapperPrefix() throws {
        let cfg = try JSONDecoder().decode(
            DeepseekV32Configuration.self, from: kimiStyleConfigData(numHiddenLayers: 1))
        let model = DeepseekV32Model(cfg)
        let weights: [String: MLXArray] = [
            "language_model.model.embed_tokens.weight": MLXArray.zeros([128, 64]),
            "language_model.lm_head.weight": MLXArray.zeros([128, 64]),
            "language_model.model.layers.0.self_attn.embed_q.weight": MLXArray.zeros([4, 16, 16]),
            "language_model.model.layers.0.self_attn.unembed_out.weight": MLXArray.zeros([
                4, 16, 16,
            ]),
        ]
        let out = model.sanitize(weights: weights)
        #expect(out["model.embed_tokens.weight"] != nil)
        #expect(out["lm_head.weight"] != nil)
        #expect(out["model.layers.0.self_attn.embed_q.weight"] != nil)
        #expect(out["model.layers.0.self_attn.unembed_out.weight"] != nil)
        #expect(out.keys.allSatisfy { !$0.hasPrefix("language_model.") })
    }

    /// The shipped K2.6 config, when present: 61 dense-attention layers, 384 experts.
    @Test("real Kimi K2.6 config decodes through the kimi_k25 registry entry")
    func testRealKimiK26Config() async throws {
        let path = NSString(
            string: "~/dev/stream/models/Kimi-K2.6-DQ3/config.json"
        ).expandingTildeInPath
        guard FileManager.default.fileExists(atPath: path) else { return }
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let cfg = try JSONDecoder().decode(DeepseekV32Configuration.self, from: data)
        #expect(cfg.numHiddenLayers == 61)
        #expect(cfg.nRoutedExperts == 384)
        #expect(cfg.numExpertsPerTok == 8)
        #expect(cfg.firstKDenseReplace == 1)
        #expect(cfg.indexerTypes.count == 61)
        #expect(cfg.indexerTypes.allSatisfy { $0 == "shared" })
        #expect(cfg.ropeScaling?["type"] == .string("yarn"))
    }
}
