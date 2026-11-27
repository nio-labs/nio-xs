//! # Model Downloader & Artifact Lifecycle
//!
//! Manages on-demand downloading of local quantized model weights
//! into `~/.nioxs/models/` with streaming progress and SHA-256 integrity verification.

use futures_util::StreamExt;
use nioxs_contracts::{
    MODEL_BELUGA, MODEL_BLUE, MODEL_CHEETAH, MODEL_FOX, MODEL_MANTA, MODEL_NANO, MODEL_ORCA_MINI,
    MODEL_ORCA_NANO, MODEL_RABBIT, MODEL_SKATE, MODEL_TIGER, MODEL_VIPER,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelArtifactEntry {
    pub model_id: String,
    pub brand_name: String,
    pub base_model: String,
    pub filename: String,
    pub download_url: String,
    pub sha256: String,
    pub size_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<ModelTokenizerArtifact>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelTokenizerArtifact {
    pub filename: String,
    pub download_url: String,
    pub sha256: String,
    pub size_bytes: u64,
}

pub fn default_manifest() -> Vec<ModelArtifactEntry> {
    vec![
        ModelArtifactEntry {
            model_id: MODEL_VIPER.into(),
            brand_name: "Viper (Tactical Terminal Striker)".into(),
            base_model: "Viper-100M".into(),
            filename: "viper.Q4_0.gguf".into(),
            download_url: "https://github.com/nio-labs/nio-xs-models/releases/download/v0.1.0/viper.Q4_0.gguf".into(),
            sha256: "58633aa691b80840dc85586cf2cb8616deff480c64ddf406bf848db749ecc3d4".into(),
            size_bytes: 64_684_160,
            tokenizer: Some(ModelTokenizerArtifact {
                filename: "viper-tokenizer.json".into(),
                download_url: "https://github.com/nio-labs/nio-xs-models/releases/download/v0.1.0/viper-tokenizer.json".into(),
                sha256: "a136e7acd3d47fe1a2572bf0fe7af79528e4492bf15f1ecea24e2216cf78c690".into(),
                size_bytes: 3_173_909,
            }),
        },
        ModelArtifactEntry {
            model_id: MODEL_SKATE.into(),
            brand_name: "Skate (Ambient Desk Companion)".into(),
            base_model: "SmolLM2-135M (30L, v-µLM Companion)".into(),
            filename: "skate.vmicrolm-vector.gguf".into(),
            download_url: "https://github.com/nio-labs/nio-xs-models/releases/download/v0.2.6/skate.vmicrolm-vector.gguf".into(),
            sha256: "a00921dca072e640bb1f9730a95e3d88ed843cba3dd0a5648a6fd5e3e9e3d86b".into(),
            size_bytes: 86_957_792,
            tokenizer: Some(ModelTokenizerArtifact {
                filename: "skate-tokenizer.json".into(),
                download_url: "https://github.com/nio-labs/nio-xs-models/releases/download/v0.2.6/skate-tokenizer.json".into(),
                sha256: "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c".into(),
                size_bytes: 2_104_556,
            }),
        },
        ModelArtifactEntry {
            model_id: MODEL_MANTA.into(),
            brand_name: "Manta (Systems Guru & DAG Architect)".into(),
            base_model: "Qwen/Qwen2.5-0.5B-Instruct (24L, v-µLM Companion)".into(),
            filename: "manta.vmicrolm-vector.gguf".into(),
            download_url: "https://github.com/nio-labs/nio-xs-models/releases/download/v0.2.7/manta.vmicrolm-vector.gguf".into(),
            sha256: "fa981bab81c7cb32a7072dfd193f1e03f988eb800d7e6773a03b69c861d452a5".into(),
            size_bytes: 333_360_320,
            tokenizer: Some(ModelTokenizerArtifact {
                filename: "manta-tokenizer.json".into(),
                download_url: "https://github.com/nio-labs/nio-xs-models/releases/download/v0.2.7/manta-tokenizer.json".into(),
                sha256: "ade71ec5118be2cdd8aef3a7dcd417c51fa7e76b3e164adb339f446b42e97bc8".into(),
                size_bytes: 14_569_356,
            }),
        },
        ModelArtifactEntry {
            model_id: MODEL_BELUGA.into(),
            brand_name: "Beluga (Sovereign Lead Architect)".into(),
            base_model: "Qwen/Qwen2.5-1.5B-Instruct (28L, v-µLM Companion)".into(),
            filename: "beluga.vmicrolm-vector.gguf".into(),
            download_url: "https://github.com/nio-labs/nio-xs-models/releases/download/v0.3.0/beluga.vmicrolm-vector.gguf".into(),
            sha256: "d2d4642d36b907c8658301208e8cd4ab3c6d72051a37a780ebc35a56ff83d5ba".into(),
            size_bytes: 639_362_560,
            tokenizer: Some(ModelTokenizerArtifact {
                filename: "beluga-tokenizer.json".into(),
                download_url: "https://github.com/nio-labs/nio-xs-models/releases/download/v0.3.0/beluga-tokenizer.json".into(),
                sha256: "cf535eea1ed9374c2bc3436625216c18fc703fc37aae8e019b3e85cc658e0395".into(),
                size_bytes: 11_422_174,
            }),
        },
        ModelArtifactEntry {
            model_id: MODEL_BLUE.into(),
            brand_name: "Blue (Sovereign Reflex Lead)".into(),
            base_model: "microsoft/Phi-3.5-mini-instruct (32L, vMicroLM Reflex)".into(),
            filename: "blue.vmicrolm-reflex.gguf".into(),
            download_url: "https://github.com/nio-labs/nio-xs-models/releases/download/v0.3.1/blue.vmicrolm-reflex.gguf".into(),
            sha256: "6b673f5a2ed71eced7b6cc947beccf3cf3f669eb25d6e6233c48e47e11e9bb32".into(),
            size_bytes: 1_499_437_216,
            tokenizer: Some(ModelTokenizerArtifact {
                filename: "blue-tokenizer.json".into(),
                download_url: "https://github.com/nio-labs/nio-xs-models/releases/download/v0.3.1/blue-tokenizer.json".into(),
                sha256: "8fcc6f6a32f0bd5ccdac48a0522dcd1cc575e5daf591f09d489e07f0d9035b2e".into(),
                size_bytes: 3_911_372,
            }),
        },
        ModelArtifactEntry {
            model_id: MODEL_CHEETAH.into(),
            brand_name: "Cheetah (Sovereign Sub-20ms Code Sprinter)".into(),
            base_model: "Qwen/Qwen2.5-Coder-1.5B-Instruct (28L, Dense)".into(),
            filename: "cheetah.1.5b-q4km.gguf".into(),
            download_url: "https://github.com/nio-labs/nio-xs-models/releases/download/v0.4.0/cheetah.1.5b-q4km.gguf".into(),
            sha256: "cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046".into(),
            size_bytes: 1_117_320_768,
            tokenizer: Some(ModelTokenizerArtifact {
                filename: "cheetah-tokenizer.json".into(),
                download_url: "https://github.com/nio-labs/nio-xs-models/releases/download/v0.4.0/cheetah-tokenizer.json".into(),
                sha256: "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539".into(),
                size_bytes: 7_031_645,
            }),
        },
        ModelArtifactEntry {
            model_id: MODEL_TIGER.into(),
            brand_name: "Tiger (Sovereign Vibe Coder & Daily Workhorse)".into(),
            base_model: "Qwen/Qwen2.5-Coder-3B-Instruct (36L, Dense)".into(),
            filename: "tiger.3b-q4km.gguf".into(),
            download_url: "https://github.com/nio-labs/nio-xs-models/releases/download/v0.4.1/tiger.3b-q4km.gguf".into(),
            sha256: "724fb256bec1ff062b2f65e4569e871ad2e95ab2a3989723d1769c54294730b7".into(),
            size_bytes: 2_104_932_800,
            tokenizer: Some(ModelTokenizerArtifact {
                filename: "tiger-tokenizer.json".into(),
                download_url: "https://github.com/nio-labs/nio-xs-models/releases/download/v0.4.1/tiger-tokenizer.json".into(),
                sha256: "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539".into(),
                size_bytes: 7_031_645,
            }),
        },
        ModelArtifactEntry {
            model_id: MODEL_FOX.into(),
            brand_name: "Fox (Sovereign Articulate Mentor & Pair Programmer)".into(),
            base_model: "Qwen/Qwen2.5-Coder-3B-Instruct (36L, Dense)".into(),
            filename: "fox.3b-q4km.gguf".into(),
            download_url: "https://github.com/nio-labs/nio-xs-models/releases/download/v0.4.1/tiger.3b-q4km.gguf".into(),
            sha256: "724fb256bec1ff062b2f65e4569e871ad2e95ab2a3989723d1769c54294730b7".into(),
            size_bytes: 2_104_932_800,
            tokenizer: Some(ModelTokenizerArtifact {
                filename: "fox-tokenizer.json".into(),
                download_url: "https://github.com/nio-labs/nio-xs-models/releases/download/v0.4.1/tiger-tokenizer.json".into(),
                sha256: "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539".into(),
                size_bytes: 7_031_645,
            }),
        },
        ModelArtifactEntry {
            model_id: MODEL_RABBIT.into(),
            brand_name: "Rabbit (Sovereign Deep Thinker & Root-Cause Detective)".into(),
            base_model: "Qwen/Qwen2.5-Coder-3B-Instruct (36L, Dense)".into(),
            filename: "rabbit.3b-q4km.gguf".into(),
            download_url: "https://github.com/nio-labs/nio-xs-models/releases/download/v0.4.1/tiger.3b-q4km.gguf".into(),
            sha256: "724fb256bec1ff062b2f65e4569e871ad2e95ab2a3989723d1769c54294730b7".into(),
            size_bytes: 2_104_932_800,
            tokenizer: Some(ModelTokenizerArtifact {
                filename: "rabbit-tokenizer.json".into(),
                download_url: "https://github.com/nio-labs/nio-xs-models/releases/download/v0.4.1/tiger-tokenizer.json".into(),
                sha256: "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539".into(),
                size_bytes: 7_031_645,
            }),
        },
        orca_entry(
            MODEL_ORCA_NANO,
            "Orca Nano Lead (0.5B Sovereign Router & Lead)",
            "Qwen/Qwen2.5-Coder-0.5B-Instruct (24L, Q4_K_M)",
            "orca-nano-lead-0.5b-q4_k_m.gguf",
            "0af05b68e6dec88ec02af0868888878fbe871e04d4c4910c15d6649ea40775ec",
            397_807_616,
        ),
        orca_entry(
            "orca-nano-architect",
            "Orca Nano Architect (0.5B)",
            "Qwen/Qwen2.5-Coder-0.5B-Instruct (24L, Q4_K_M)",
            "orca-nano-architect-0.5b-q4_k_m.gguf",
            "941f626b522ff4ac7bdefc2e1810273e57024ed4f69842d28c4bbf51da3471ef",
            397_807_616,
        ),
        orca_entry(
            "orca-nano-reasoner",
            "Orca Nano Reasoner (0.5B)",
            "Qwen/Qwen2.5-Coder-0.5B-Instruct (24L, Q4_K_M)",
            "orca-nano-reasoner-0.5b-q4_k_m.gguf",
            "acdc5717058a7fc453f690978d26f84b03955857b553356b878a31c971947649",
            397_807_616,
        ),
        orca_entry(
            "orca-nano-algo-reasoner",
            "Orca Nano Algo Reasoner (0.5B)",
            "Qwen/Qwen2.5-Coder-0.5B-Instruct (24L, Q4_K_M)",
            "orca-nano-algo-reasoner-0.5b-q4_k_m.gguf",
            "7c29cbe6eaa5a1d40afd387381e0331356e76c8e4a45593b0b2846f2469a3e65",
            397_807_616,
        ),
        orca_entry(
            "orca-nano-repair-reasoner",
            "Orca Nano Repair Reasoner (0.5B)",
            "Qwen/Qwen2.5-Coder-0.5B-Instruct (24L, Q4_K_M)",
            "orca-nano-repair-reasoner-0.5b-q4_k_m.gguf",
            "ad35632ebe47ffa17d3d9875558bb56d411103e5909b990450a8e60e4b244345",
            397_807_616,
        ),
        orca_entry(
            "orca-nano-security-reasoner",
            "Orca Nano Security Reasoner (0.5B)",
            "Qwen/Qwen2.5-Coder-0.5B-Instruct (24L, Q4_K_M)",
            "orca-nano-security-reasoner-0.5b-q4_k_m.gguf",
            "6e449dae1bfe800e6e36e1fb110aeec545c2e98f863d18504696c7413d399023",
            397_807_616,
        ),
        orca_entry(
            "orca-nano-test-reasoner",
            "Orca Nano Test Reasoner (0.5B)",
            "Qwen/Qwen2.5-Coder-0.5B-Instruct (24L, Q4_K_M)",
            "orca-nano-test-reasoner-0.5b-q4_k_m.gguf",
            "be1d9f44db02697226a46dd3f9a5664f9b8ad1061a2a1250c4a33cc07175ff82",
            397_807_616,
        ),
        orca_entry(
            "orca-nano-pod-0",
            "Orca Nano Pod 0 (0.5B)",
            "Qwen/Qwen2.5-Coder-0.5B-Instruct (24L, Q4_K_M)",
            "orca-nano-pod-0-0.5b-q4_k_m.gguf",
            "3042d45fe16e1814a75d2dcfc746e9f658aafd506cbe033fb7c740d423d98efd",
            397_807_616,
        ),
        orca_entry(
            "orca-nano-pod-1",
            "Orca Nano Pod 1 (0.5B)",
            "Qwen/Qwen2.5-Coder-0.5B-Instruct (24L, Q4_K_M)",
            "orca-nano-pod-1-0.5b-q4_k_m.gguf",
            "91d3f4f36e5a64e93cc2699e79efa2aa0a937d0cabb2d958ea5fb18dc5bd2fbe",
            397_807_616,
        ),
        orca_entry(
            "orca-nano-pod-2",
            "Orca Nano Pod 2 (0.5B)",
            "Qwen/Qwen2.5-Coder-0.5B-Instruct (24L, Q4_K_M)",
            "orca-nano-pod-2-0.5b-q4_k_m.gguf",
            "42b7b273c9a2243b3c0f8d32fd51c18fd97454becb6427a462498fa2f396a3f1",
            397_807_616,
        ),
        orca_entry(
            "orca-nano-pod-3",
            "Orca Nano Pod 3 (0.5B)",
            "Qwen/Qwen2.5-Coder-0.5B-Instruct (24L, Q4_K_M)",
            "orca-nano-pod-3-0.5b-q4_k_m.gguf",
            "95551f5e6e4ec10748e0ce948375befa3c92076c8aa96f36cd82d88bbffb5abc",
            397_807_616,
        ),
        orca_entry(
            "orca-nano-pod-4",
            "Orca Nano Pod 4 (0.5B)",
            "Qwen/Qwen2.5-Coder-0.5B-Instruct (24L, Q4_K_M)",
            "orca-nano-pod-4-0.5b-q4_k_m.gguf",
            "671316a2ebad327055edee84330fa915bec3197eb6b1ee26089a44f7e5f4bcb3",
            397_807_616,
        ),
        orca_entry(
            "orca-nano-pod-5",
            "Orca Nano Pod 5 (0.5B)",
            "Qwen/Qwen2.5-Coder-0.5B-Instruct (24L, Q4_K_M)",
            "orca-nano-pod-5-0.5b-q4_k_m.gguf",
            "b12a5499f24b8f1e3b1a425eae0fe7f6b2dfaff8e2d0fee2b8cb027a16ab6054",
            397_807_616,
        ),
        orca_entry(
            MODEL_ORCA_MINI,
            "Orca Mini Lead (1.5B Sovereign Router & Lead)",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct (28L, Q4_K_M)",
            "orca-lead-1.5b-q4_k_m.gguf",
            "6aa0cb0f99b32d359eaae3e90d4b01222a9430717d160f3f01ae48f981b07135",
            986_048_192,
        ),
        orca_entry(
            "orca-mini-architect",
            "Orca Mini Architect (1.5B)",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct (28L, Q4_K_M)",
            "orca-architect-1.5b-q4_k_m.gguf",
            "3ef190a8a113ab4f2d3ddc2a3492d3a296e2636fd804dd24ac5c843911e7e584",
            986_048_192,
        ),
        orca_entry(
            "orca-mini-reasoner",
            "Orca Mini Reasoner (1.5B)",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct (28L, Q4_K_M)",
            "orca-reasoner-1.5b-q4_k_m.gguf",
            "ed5e9ef7915eb40b17422ab8af544a34e95030a42f852e50aec2fc2055b41574",
            986_048_192,
        ),
        orca_entry(
            "orca-mini-algo-reasoner",
            "Orca Mini Algo Reasoner (1.5B)",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct (28L, Q4_K_M)",
            "orca-algo-reasoner-1-5b-q4_k_m.gguf",
            "d8bfce8d934b1b23f681772d606938242a34830d6257e1e7229447abdbf1747a",
            986_048_192,
        ),
        orca_entry(
            "orca-mini-repair-reasoner",
            "Orca Mini Repair Reasoner (1.5B)",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct (28L, Q4_K_M)",
            "orca-repair-reasoner-1-5b-q4_k_m.gguf",
            "e79687b4f825cfcdffa27d2725a2c72168eaef6abf50aa47df0a792e06aebe9e",
            986_048_192,
        ),
        orca_entry(
            "orca-mini-security-reasoner",
            "Orca Mini Security Reasoner (1.5B)",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct (28L, Q4_K_M)",
            "orca-security-reasoner-1-5b-q4_k_m.gguf",
            "674936bc473b27f31f97beb7eb18e26f205e074f7d80166e6149d21a15ccc914",
            986_048_192,
        ),
        orca_entry(
            "orca-mini-test-reasoner",
            "Orca Mini Test Reasoner (1.5B)",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct (28L, Q4_K_M)",
            "orca-test-reasoner-1-5b-q4_k_m.gguf",
            "2939f8df5d44380b4bfed55a45bd1dcd20b05260517b2bec88549d0404e7f32f",
            986_048_192,
        ),
        orca_entry(
            "orca-mini-pod-0",
            "Orca Mini Pod 0 (1.5B)",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct (28L, Q4_K_M)",
            "orca-pod-0-1-5b-q4_k_m.gguf",
            "de53a71f8d67ba916665edf6384cbd074d40b52311fbda8a0a46431af519bdd5",
            986_048_192,
        ),
        orca_entry(
            "orca-mini-pod-1",
            "Orca Mini Pod 1 (1.5B)",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct (28L, Q4_K_M)",
            "orca-pod-1-1-5b-q4_k_m.gguf",
            "dceb9734ce024364c667c1e9cd714832ec7eb00bd683cfb3f1bc880bd7b9153d",
            986_048_192,
        ),
        orca_entry(
            "orca-mini-pod-2",
            "Orca Mini Pod 2 (1.5B)",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct (28L, Q4_K_M)",
            "orca-pod-2-1-5b-q4_k_m.gguf",
            "66285737714a395fe79bcaaeb0858038916978d46a208479a4eab74041f244d6",
            986_048_192,
        ),
        orca_entry(
            "orca-mini-pod-3",
            "Orca Mini Pod 3 (1.5B)",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct (28L, Q4_K_M)",
            "orca-pod-3-1-5b-q4_k_m.gguf",
            "e9895784ad3c0d6a050a18f51ad9399ba2d03a1cbd69048df71cbc23a53364e3",
            986_048_192,
        ),
        orca_entry(
            "orca-mini-pod-4",
            "Orca Mini Pod 4 (1.5B)",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct (28L, Q4_K_M)",
            "orca-pod-4-1-5b-q4_k_m.gguf",
            "c414a78056add9efc64506019545903778aaf1499ba20de5aa873d827e065281",
            986_048_192,
        ),
        orca_entry(
            "orca-mini-pod-5",
            "Orca Mini Pod 5 (1.5B)",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct (28L, Q4_K_M)",
            "orca-pod-5-1-5b-q4_k_m.gguf",
            "2b4e4280bf3ff4046cbcc936f80358842a53c871d818071d81fde63e43dccee0",
            986_048_192,
        ),
        orca_entry_tag(
            "v0.6.1",
            "health-nutrition-fitness",
            "Orca Health & Fitness Pod (0.5B, Wave 1)",
            "Qwen/Qwen2.5-0.5B-Instruct (24L, Q4_K_M)",
            "orca-health-nutrition-fitness-0.5b-q4_k_m.gguf",
            "dfe3f132675b2ffe64ef4e51e781a05268cce2067d323123f29c2b6d7d4cd55c",
            397_807_616,
        ),
        orca_entry_tag(
            "v0.6.1",
            "health-general-info",
            "Orca Health General Info Pod (0.5B, Wave 1)",
            "Qwen/Qwen2.5-0.5B-Instruct (24L, Q4_K_M)",
            "orca-health-general-info-0.5b-q4_k_m.gguf",
            "22466678b640a9eb4e6718d9a54c70e63b427be6051cb7c0fb8a42c95f058614",
            397_807_616,
        ),
        orca_entry_tag(
            "v0.6.1",
            "everyday-cooking-travel",
            "Orca Everyday Cooking & Travel Pod (0.5B, Wave 1)",
            "Qwen/Qwen2.5-0.5B-Instruct (24L, Q4_K_M)",
            "orca-everyday-cooking-travel-0.5b-q4_k_m.gguf",
            "f1c03868b06ccf5f2449356542505a07a49dc5f81f058476d8669717824a0bcf",
            397_807_616,
        ),
        orca_entry_tag(
            "v0.6.1",
            "everyday-diy-life",
            "Orca Everyday DIY & Life Pod (0.5B, Wave 1)",
            "Qwen/Qwen2.5-0.5B-Instruct (24L, Q4_K_M)",
            "orca-everyday-diy-life-0.5b-q4_k_m.gguf",
            "92a86aff9c7ef588fc401e6a33daa69ce5d31dbe880646e96f73728dc4e04d4b",
            397_807_616,
        ),
        orca_entry_tag(
            "v0.6.3",
            "math-arithmetic-algebra",
            "Orca math-arithmetic-algebra (0.5B, Wave 2)",
            "Qwen/Qwen2.5-0.5B-Instruct (24L, Q4_K_M)",
            "orca-math-arithmetic-algebra-0.5b-q4_k_m.gguf",
            "f5da1347cd678a1f35e6d9a6bc7dc93586d4515cc436d19e6d87bf5010aaefac",
            397_807_616,
        ),
        orca_entry_tag(
            "v0.6.3",
            "math-geometry-trig",
            "Orca math-geometry-trig (0.5B, Wave 2)",
            "Qwen/Qwen2.5-0.5B-Instruct (24L, Q4_K_M)",
            "orca-math-geometry-trig-0.5b-q4_k_m.gguf",
            "22935663e48a5263dde3c26248d638874011d7abdb49b2387b8accf08bb378df",
            397_807_616,
        ),
        orca_entry_tag(
            "v0.6.3",
            "math-probability-stats",
            "Orca math-probability-stats (0.5B, Wave 2)",
            "Qwen/Qwen2.5-0.5B-Instruct (24L, Q4_K_M)",
            "orca-math-probability-stats-0.5b-q4_k_m.gguf",
            "cd691a4a32f70e9c7d20dea48b265a4f79027fed0d1e2af11c49f378f12601c6",
            397_807_616,
        ),
        orca_entry_tag(
            "v0.6.3",
            "math-calculus-linear-algebra",
            "Orca math-calculus-linear-algebra (1.5B, Wave 2)",
            "Qwen/Qwen2.5-1.5B-Instruct (28L, Q4_K_M)",
            "orca-math-calculus-linear-algebra-1.5b-q4_k_m.gguf",
            "6b0bdbc35a307df6912646fc9ac8613ef97a72bba659313bbc474d3d2849d31a",
            986_048_192,
        ),
        orca_entry_tag(
            "v0.6.3",
            "math-olympiad-proofs",
            "Orca math-olympiad-proofs (1.5B, Wave 2)",
            "Qwen/Qwen2.5-1.5B-Instruct (28L, Q4_K_M)",
            "orca-math-olympiad-proofs-1.5b-q4_k_m.gguf",
            "070017a69b83ed800395452f2df100ba1891929129843a98e8c9538391fecb7c",
            986_048_192,
        ),
        orca_entry_tag(
            "v0.6.3",
            "history-ancient-medieval",
            "Orca history-ancient-medieval (0.5B, Wave 2)",
            "Qwen/Qwen2.5-0.5B-Instruct (24L, Q4_K_M)",
            "orca-history-ancient-medieval-0.5b-q4_k_m.gguf",
            "4810418ca51fa10253b4de7ff7c8254e95fdfa2211fe569ad731cd98857c0ea1",
            397_807_616,
        ),
        orca_entry_tag(
            "v0.6.3",
            "history-modern-world",
            "Orca history-modern-world (0.5B, Wave 2)",
            "Qwen/Qwen2.5-0.5B-Instruct (24L, Q4_K_M)",
            "orca-history-modern-world-0.5b-q4_k_m.gguf",
            "721c61390e7a601d6730d75057cd68dee90b0270cf054c7fa99b45a3c61ad381",
            397_807_616,
        ),
        orca_entry_tag(
            "v0.6.3",
            "history-tech-science",
            "Orca history-tech-science (0.5B, Wave 2)",
            "Qwen/Qwen2.5-0.5B-Instruct (24L, Q4_K_M)",
            "orca-history-tech-science-0.5b-q4_k_m.gguf",
            "fe2af2859483ac2046ada6723d8fb6a790053df016e34e6a9b5fcf0e73f88b85",
            397_807_616,
        ),
        orca_entry_tag(
            "v0.6.3",
            "history-geopolitics-synthesis",
            "Orca history-geopolitics-synthesis (1.5B, Wave 2)",
            "Qwen/Qwen2.5-1.5B-Instruct (28L, Q4_K_M)",
            "orca-history-geopolitics-synthesis-1.5b-q4_k_m.gguf",
            "713a9bc6335f59a0d1d843f564921e0212b478c1dfc38bc4edef4bc8ded22f1a",
            986_048_192,
        ),
        orca_entry_tag(
            "v0.6.3",
            "gk-world-geography-culture",
            "Orca gk-world-geography-culture (0.5B, Wave 2)",
            "Qwen/Qwen2.5-0.5B-Instruct (24L, Q4_K_M)",
            "orca-gk-world-geography-culture-0.5b-q4_k_m.gguf",
            "99184ab32af6f8f39776bf5b4c232e2fe3d7b8174095aa80b2c998a3d77b359d",
            397_807_616,
        ),
        orca_entry_tag(
            "v0.6.3",
            "gk-everyday-science-facts",
            "Orca gk-everyday-science-facts (0.5B, Wave 2)",
            "Qwen/Qwen2.5-0.5B-Instruct (24L, Q4_K_M)",
            "orca-gk-everyday-science-facts-0.5b-q4_k_m.gguf",
            "0d99c05c063940d442e9ecdfd6297a891408c42cf1621feaddeec791003244b8",
            397_807_616,
        ),
        orca_entry_tag(
            "v0.6.3",
            "gk-civics-economics-basics",
            "Orca gk-civics-economics-basics (0.5B, Wave 2)",
            "Qwen/Qwen2.5-0.5B-Instruct (24L, Q4_K_M)",
            "orca-gk-civics-economics-basics-0.5b-q4_k_m.gguf",
            "5d3ce6111519767e3f418cd0b76eba0c17bf62e59d9db264eab308acc6592b8a",
            397_807_616,
        ),
        orca_entry_tag(
            "v0.6.3",
            "gk-logic-critical-thinking",
            "Orca gk-logic-critical-thinking (1.5B, Wave 2)",
            "Qwen/Qwen2.5-1.5B-Instruct (28L, Q4_K_M)",
            "orca-gk-logic-critical-thinking-1.5b-q4_k_m.gguf",
            "b42d0d9b602523bc9affbe15474c6eef1444f54e6afe94eeff802e5582f89dbe",
            986_048_192,
        ),
        orca_entry_tag(
            "v0.6.3",
            "gk-deep-explainer",
            "Orca gk-deep-explainer (1.5B, Wave 2)",
            "Qwen/Qwen2.5-1.5B-Instruct (28L, Q4_K_M)",
            "orca-gk-deep-explainer-1.5b-q4_k_m.gguf",
            "8e427831582193b18d8bfd3a721677ff9e8a390013c041b736f6c9420fee327c",
            986_048_192,
        ),
        orca_entry_tag(
            "v0.6.3",
            "tutor-k12-math-science",
            "Orca tutor-k12-math-science (0.5B, Wave 2)",
            "Qwen/Qwen2.5-0.5B-Instruct (24L, Q4_K_M)",
            "orca-tutor-k12-math-science-0.5b-q4_k_m.gguf",
            "e8f0582608355f38545fb1609605f05f4829dc76a67e68f55c3e6d5490f12819",
            397_807_616,
        ),
        orca_entry_tag(
            "v0.6.3",
            "tutor-reading-essay-coach",
            "Orca tutor-reading-essay-coach (0.5B, Wave 2)",
            "Qwen/Qwen2.5-0.5B-Instruct (24L, Q4_K_M)",
            "orca-tutor-reading-essay-coach-0.5b-q4_k_m.gguf",
            "8d27cf2356cc66b10187d523ba35f09231e3c6397ca0b22bcea3474808812fb5",
            397_807_616,
        ),
        orca_entry_tag(
            "v0.6.3",
            "tutor-language-grammar",
            "Orca tutor-language-grammar (0.5B, Wave 2)",
            "Qwen/Qwen2.5-0.5B-Instruct (24L, Q4_K_M)",
            "orca-tutor-language-grammar-0.5b-q4_k_m.gguf",
            "587aae8cb9cb05302f677436d4c5460ca8063aacb1cbc08edea87713f3cf8b07",
            397_807_616,
        ),
        orca_entry_tag(
            "v0.6.3",
            "tutor-flashcard-quiz-gen",
            "Orca tutor-flashcard-quiz-gen (0.5B, Wave 2)",
            "Qwen/Qwen2.5-0.5B-Instruct (24L, Q4_K_M)",
            "orca-tutor-flashcard-quiz-gen-0.5b-q4_k_m.gguf",
            "f6e5478cfbaccadf0771d913199ef85599576a4bcd79f75ba6d52d4374823ee5",
            397_807_616,
        ),
        orca_entry_tag(
            "v0.6.3",
            "tutor-coding-mentor",
            "Orca tutor-coding-mentor (1.5B, Wave 2)",
            "Qwen/Qwen2.5-1.5B-Instruct (28L, Q4_K_M)",
            "orca-tutor-coding-mentor-1.5b-q4_k_m.gguf",
            "1ecd4b937e768df0ef84a36cd900c7515b04a75fa32cf40d22460f87ae1890b1",
            986_048_192,
        ),
        orca_entry_tag(
            "v0.6.3",
            "tutor-university-stem-solver",
            "Orca tutor-university-stem-solver (1.5B, Wave 2)",
            "Qwen/Qwen2.5-1.5B-Instruct (28L, Q4_K_M)",
            "orca-tutor-university-stem-solver-1.5b-q4_k_m.gguf",
            "3098bdf0dfab2dc3dc7e63813c1968862f55d3f41c08390d07995d92b25f68b0",
            986_048_192,
        ),
        orca_entry_tag(
            "v0.6.4",
            "meerkat-scout",
            "Meerkat Scout (0.5B, vMicroLM-Growth)",
            "Qwen/Qwen2.5-0.5B-Instruct (24L, Q4_K_M)",
            "meerkat-scout-0.5b-q4_k_m.gguf",
            "4f3586f3ac88f3e7551760771b752d190159115985afb0472fbf2c38f3f01dbb",
            397_807_584,
        ),
        orca_entry_tag(
            "v0.6.4",
            "meerkat-builder",
            "Meerkat Builder (1.5B, vMicroLM-Growth)",
            "Qwen/Qwen2.5-1.5B-Instruct (28L, Q4_K_M)",
            "meerkat-builder-1.5b-q4_k_m.gguf",
            "f25f0b614e8bde36275b7085bc29cd4007f35c9684b1b077c7cbbc7f9882430b",
            986_048_160,
        ),
        orca_entry_tag(
            "v0.6.6",
            "tiny-meerkat-reflex",
            "Tiny Meerkat Reflex (0.5B IQ3_XXS, vMicroLM-Growth)",
            "Qwen/Qwen2.5-Coder-0.5B-Instruct (24L, IQ3_XXS)",
            "tiny-meerkat-reflex-0.5b-iq3_xxs.gguf",
            "d72fe1d8ba6b2f95125da7fe4b21f04cdfe79d98958c702acf6e21090ad2fdb3",
            333_704_352,
        ),
        tiny_entry(
            "tiny-meerkat-scout",
            "Tiny Meerkat Scout (135M)",
            "baby-meerkat-scout-135m-q4_k_m.gguf",
            "62c2d06910774604c8d115075e1e967906a6872ee072810989a44b66e992552e",
            105_454_016,
        ),
        tiny_entry(
            "tiny-meerkat-builder",
            "Tiny Meerkat Builder (135M)",
            "baby-meerkat-builder-135m-q4_k_m.gguf",
            "fcec7637d4b4a55e4c1c6bfd58ab914f0263eb11b1112542214dc85852caf519",
            105_454_016,
        ),
        tiny_entry("tiny-orca-lead", "Tiny Orca lead (135M)", "baby-orca-lead-135m-q4_k_m.gguf", "4808d5955f331e677ca2d0726ec52294a1f6d414d1dd0f6b9f6d6942f9c7d0fd", 105_454_016),
        tiny_entry("tiny-orca-web-scout", "Tiny Orca web-scout (135M)", "baby-orca-web-scout-135m-q4_k_m.gguf", "7b92ac6f35173c93e9add042a1532bfae112d2c92c74b64efd06a86ae38bcc39", 105_454_016),
        tiny_entry("tiny-orca-guard", "Tiny Orca guard (135M)", "baby-orca-guard-135m-q4_k_m.gguf", "92c0369dbdf80e6372de2cb05d27b3ed6261c98c29396b6c263ee60ee32b2d24", 105_454_016),
        tiny_entry("tiny-orca-reasoning-think", "Tiny Orca reasoning-think (135M)", "baby-orca-reasoning-think-135m-q4_k_m.gguf", "ad07e500776c9d27df73ff07ca1cd720ea4b864b45ac4dbb6ec747ddc8a92db1", 105_454_016),
        tiny_entry("tiny-orca-python", "Tiny Orca python (135M)", "baby-orca-python-135m-q4_k_m.gguf", "4e2fe3350ce50f87658de7bc1e0093b24c019d3c0067ee6f3c92bdefe40789e6", 105_454_016),
        tiny_entry("tiny-orca-rust", "Tiny Orca rust (135M)", "baby-orca-rust-135m-q4_k_m.gguf", "9a556b039c61998389bb5c28d2a794a299e49005de7b7053b0662d8e58eb8c89", 105_454_016),
        tiny_entry("tiny-orca-typescript", "Tiny Orca typescript (135M)", "baby-orca-typescript-135m-q4_k_m.gguf", "e223328db275b7140f52dd9e14f46de7804f07e415ecc2bcecde4d545ad7a97e", 105_454_016),
        tiny_entry("tiny-orca-go", "Tiny Orca go (135M)", "baby-orca-go-135m-q4_k_m.gguf", "1bc75f490678b3c96ea156cfba311f240b0f71f149c9cb3034a9cd163a1ade29", 105_454_016),
        tiny_entry("tiny-orca-cpp", "Tiny Orca cpp (135M)", "baby-orca-cpp-135m-q4_k_m.gguf", "0c68309643529b748720f3c9c0cbfdaef8ab8902bf7316ed235c7a7f2800bc29", 105_454_016),
        tiny_entry("tiny-orca-sql", "Tiny Orca sql (135M)", "baby-orca-sql-135m-q4_k_m.gguf", "c343887ee67919940eb35296762f07f09b0217f0519c3c45d7d1a32508daaca8", 105_454_016),
        tiny_entry("tiny-orca-bash-cli", "Tiny Orca bash-cli (135M)", "baby-orca-bash-cli-135m-q4_k_m.gguf", "21211175cbb180a4679543d4f2d18d23ed437d47087343380b9596ddc9a98d1c", 105_454_016),
        tiny_entry("tiny-orca-regex-cli", "Tiny Orca regex-cli (135M)", "baby-orca-regex-cli-135m-q4_k_m.gguf", "b6ea4627199abdba193a2825b668857b0f305b3761950c72fa87efc12d673ea8", 105_454_016),
        tiny_entry("tiny-orca-testing", "Tiny Orca testing (135M)", "baby-orca-testing-135m-q4_k_m.gguf", "11fce5b3150eb2245c0256a60fa34e08fcf3900a065e2aa12fc9e074c310f5a5", 105_454_016),
        tiny_entry("tiny-orca-security", "Tiny Orca security (135M)", "baby-orca-security-135m-q4_k_m.gguf", "1e4d83d31fecadb1c221e059f8b2b8da4e8de6a25dbe03a9bceae295c5f0bb79", 105_454_016),
        tiny_entry("tiny-orca-systems", "Tiny Orca systems (135M)", "baby-orca-systems-135m-q4_k_m.gguf", "13c374e2a59c1a040c161b715b7aecc5e014be24eb01cd42d8da404b2dcebce1", 105_454_016),
        tiny_entry("tiny-orca-git-devops", "Tiny Orca git-devops (135M)", "baby-orca-git-devops-135m-q4_k_m.gguf", "03fce30aa81e294e582163688035c46d2e0b8db902c05b78b53dfeb8eb745efa", 105_454_016),
        tiny_entry("tiny-orca-web-api", "Tiny Orca web-api (135M)", "baby-orca-web-api-135m-q4_k_m.gguf", "e90ef7b500a9603daeb44b75b59065bdd0807a233e6dc7a2af73775f9481c03d", 105_454_016),
        tiny_entry("tiny-orca-data-json", "Tiny Orca data-json (135M)", "baby-orca-data-json-135m-q4_k_m.gguf", "c8ff3c67606e66396f976bc1281d5ff83e4587dbe3fda6f305b9a0fefd29d53b", 105_454_016),
        tiny_entry("tiny-orca-architecture-diagrams", "Tiny Orca architecture-diagrams (135M)", "baby-orca-architecture-diagrams-135m-q4_k_m.gguf", "e3e8fc420ccdad60890923a81ef5362994f2e80b7b184d2569a5d0e3ffb99941", 105_454_048),
        tiny_entry("tiny-orca-debugging", "Tiny Orca debugging (135M)", "baby-orca-debugging-135m-q4_k_m.gguf", "78420d5bcaa4d8801fb7efcad280bd3654ec72630416c41034beb00f2f63388a", 105_454_016),
        tiny_entry("tiny-orca-math-logic", "Tiny Orca math-logic (135M)", "baby-orca-math-logic-135m-q4_k_m.gguf", "e3eebb341f979b5e435824f5349bcd0ef4fda1527a0a4c553d534d7974646954", 105_454_016),
        tiny_entry("tiny-orca-docs-summary", "Tiny Orca docs-summary (135M)", "baby-orca-docs-summary-135m-q4_k_m.gguf", "c27a460bce0a4576babe522bd25fb26329e93975e9fafb9f262ad3b47780d4e5", 105_454_016),
        tiny_entry("tiny-orca-coding-expertise", "Tiny Orca coding-expertise (135M)", "baby-orca-coding-expertise-135m-q4_k_m.gguf", "2956b9a8483c25ee7c3144963ea7324a62291db7e6611bd71f0777a6b13dc1bf", 105_454_016),
    ]
}

fn tiny_entry(
    model_id: &str,
    brand_name: &str,
    filename: &str,
    sha256: &str,
    size_bytes: u64,
) -> ModelArtifactEntry {
    ModelArtifactEntry {
        model_id: model_id.into(),
        brand_name: brand_name.into(),
        base_model: "HuggingFaceTB/SmolLM2-135M-Instruct (30L, Q4_K_M)".into(),
        filename: filename.into(),
        download_url: format!(
            "https://github.com/nio-labs/nio-xs-models/releases/download/v0.6.4/{filename}"
        ),
        sha256: sha256.into(),
        size_bytes,
        tokenizer: Some(ModelTokenizerArtifact {
            filename: "baby-orca-tokenizer.json".into(),
            download_url: "https://github.com/nio-labs/nio-xs-models/releases/download/v0.6.4/baby-orca-tokenizer.json".into(),
            sha256: "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c".into(),
            size_bytes: 2_104_556,
        }),
    }
}

fn orca_entry(
    model_id: &str,
    brand_name: &str,
    base_model: &str,
    filename: &str,
    sha256: &str,
    size_bytes: u64,
) -> ModelArtifactEntry {
    orca_entry_tag(
        "v0.6.0", model_id, brand_name, base_model, filename, sha256, size_bytes,
    )
}

fn orca_entry_tag(
    release_tag: &str,
    model_id: &str,
    brand_name: &str,
    base_model: &str,
    filename: &str,
    sha256: &str,
    size_bytes: u64,
) -> ModelArtifactEntry {
    ModelArtifactEntry {
        model_id: model_id.into(),
        brand_name: brand_name.into(),
        base_model: base_model.into(),
        filename: filename.into(),
        download_url: format!(
            "https://github.com/nio-labs/nio-xs-models/releases/download/{release_tag}/{filename}"
        ),
        sha256: sha256.into(),
        size_bytes,
        tokenizer: Some(ModelTokenizerArtifact {
            filename: "orca-tokenizer.json".into(),
            download_url: "https://github.com/nio-labs/nio-xs-models/releases/download/v0.6.0/orca-tokenizer.json".into(),
            sha256: "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539".into(),
            size_bytes: 7_031_645,
        }),
    }
}

pub fn get_manifest_entry(model_id: &str) -> Option<ModelArtifactEntry> {
    let clean = model_id.strip_prefix("nio-").unwrap_or(model_id);
    default_manifest()
        .into_iter()
        .find(|e| e.model_id == clean || e.model_id == model_id)
}

/// Resolves the local models cache directory (`~/.nioxs/models/` by default).
pub fn resolve_models_dir() -> PathBuf {
    if let Ok(dir) = env::var("NIOXS_MODELS_DIR") {
        let p = PathBuf::from(dir);
        if !p.as_os_str().is_empty() {
            return p;
        }
    }
    if let Ok(home) = env::var("HOME") {
        PathBuf::from(home).join(".nioxs").join("models")
    } else {
        PathBuf::from(".nioxs").join("models")
    }
}

pub trait ProgressListener: Send + Sync {
    fn on_start(&self, brand_name: &str, total_bytes: u64);
    fn on_chunk(&self, downloaded_bytes: u64, total_bytes: u64);
    fn on_verify(&self, filename: &str);
    fn on_complete(&self, path: &Path);
}

pub struct SilentProgress;
impl ProgressListener for SilentProgress {
    fn on_start(&self, _: &str, _: u64) {}
    fn on_chunk(&self, _: u64, _: u64) {}
    fn on_verify(&self, _: &str) {}
    fn on_complete(&self, _: &Path) {}
}

pub struct ConsoleProgress;
impl ProgressListener for ConsoleProgress {
    fn on_start(&self, brand_name: &str, total_bytes: u64) {
        let mb = total_bytes as f64 / (1024.0 * 1024.0);
        println!("[NioXS] 📦 {brand_name} is required for this task ({mb:.1} MB).");
    }
    fn on_chunk(&self, downloaded_bytes: u64, total_bytes: u64) {
        let pct = if total_bytes > 0 {
            (downloaded_bytes as f64 / total_bytes as f64 * 100.0) as u32
        } else {
            0
        };
        let d_mb = downloaded_bytes as f64 / (1024.0 * 1024.0);
        let t_mb = total_bytes as f64 / (1024.0 * 1024.0);
        print!("\rDownloading [{pct:3}%] {d_mb:.1} MB / {t_mb:.1} MB");
        let _ = std::io::stdout().flush();
    }
    fn on_verify(&self, filename: &str) {
        println!("\n[NioXS] 🔒 Verifying SHA-256 digest for {filename}...");
    }
    fn on_complete(&self, path: &Path) {
        println!("[NioXS] ✔ Verified and ready: {}", path.display());
    }
}

/// Verifies whether an existing file matches the expected SHA-256 checksum.
pub fn verify_file_sha256(path: &Path, expected_sha256: &str) -> Result<bool, String> {
    if !path.exists() {
        return Ok(false);
    }
    let mut file =
        File::open(path).map_err(|e| format!("failed to open file {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let bytes_read = file
            .read(&mut buffer)
            .map_err(|e| format!("read error: {e}"))?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }
    let computed: String = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    Ok(computed.eq_ignore_ascii_case(expected_sha256))
}

/// Checks if a model's weights exist and are ready on disk.
pub fn is_model_ready(model_id: &str, custom_dir: Option<&Path>) -> bool {
    if model_id == MODEL_NANO {
        return true; // Embedded in-process
    }
    let Some(entry) = get_manifest_entry(model_id) else {
        return false;
    };
    let dir = match custom_dir {
        Some(d) => d.to_path_buf(),
        None => resolve_models_dir(),
    };
    let target = dir.join(&entry.filename);
    let model_ready = target.exists()
        && target.metadata().map(|m| m.len() > 0).unwrap_or(false)
        && verify_file_sha256(&target, &entry.sha256).unwrap_or(false);
    let tokenizer_ready = entry.tokenizer.as_ref().is_none_or(|tokenizer| {
        let tokenizer_path = dir.join(&tokenizer.filename);
        tokenizer_path.exists()
            && verify_file_sha256(&tokenizer_path, &tokenizer.sha256).unwrap_or(false)
    });
    model_ready && tokenizer_ready
}

async fn ensure_tokenizer_asset(
    client: &reqwest::Client,
    tokenizer: &ModelTokenizerArtifact,
    target_dir: &Path,
) -> Result<(), String> {
    fs::create_dir_all(target_dir)
        .map_err(|e| format!("failed to create models dir {}: {e}", target_dir.display()))?;
    let final_path = target_dir.join(&tokenizer.filename);
    if verify_file_sha256(&final_path, &tokenizer.sha256).unwrap_or(false) {
        return Ok(());
    }

    let response = client
        .get(&tokenizer.download_url)
        .send()
        .await
        .map_err(|e| format!("failed to fetch tokenizer {}: {e}", tokenizer.filename))?;
    if !response.status().is_success() {
        return Err(format!(
            "tokenizer download failed with HTTP status {} for {}",
            response.status(),
            tokenizer.filename
        ));
    }
    let bytes = response
        .bytes()
        .await
        .map_err(|e| format!("failed reading tokenizer {}: {e}", tokenizer.filename))?;
    if bytes.len() as u64 != tokenizer.size_bytes {
        return Err(format!(
            "size mismatch for {}: expected {} bytes, received {}",
            tokenizer.filename,
            tokenizer.size_bytes,
            bytes.len()
        ));
    }
    let computed_sha256 = format!("{:x}", Sha256::digest(&bytes));
    if !computed_sha256.eq_ignore_ascii_case(&tokenizer.sha256) {
        return Err(format!(
            "SHA-256 checksum mismatch for {}: expected {}, computed {}",
            tokenizer.filename, tokenizer.sha256, computed_sha256
        ));
    }

    let tmp_path = target_dir.join(format!("{}.download.tmp", tokenizer.filename));
    fs::write(&tmp_path, &bytes)
        .map_err(|e| format!("failed writing tokenizer {}: {e}", tmp_path.display()))?;
    if final_path.exists() {
        fs::remove_file(&final_path)
            .map_err(|e| format!("failed replacing tokenizer {}: {e}", final_path.display()))?;
    }
    fs::rename(&tmp_path, &final_path)
        .map_err(|e| format!("failed moving tokenizer into place: {e}"))?;
    Ok(())
}

fn ensure_tokenizer_asset_sync(
    entry: &ModelArtifactEntry,
    target_dir: &Path,
) -> Result<(), String> {
    let Some(tokenizer) = entry.tokenizer.as_ref() else {
        return Ok(());
    };
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .tcp_keepalive(Duration::from_secs(15))
        .build()
        .map_err(|e| format!("failed to build HTTP client: {e}"))?;
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        tokio::task::block_in_place(|| {
            handle.block_on(ensure_tokenizer_asset(&client, tokenizer, target_dir))
        })
    } else {
        tokio::runtime::Runtime::new()
            .map_err(|e| format!("failed to create runtime: {e}"))?
            .block_on(ensure_tokenizer_asset(&client, tokenizer, target_dir))
    }
}

/// Downloads model weights from URL with streaming progress, computes SHA-256,
/// and renames to the destination path only after validation succeeds.
pub async fn download_and_verify_model<P: ProgressListener>(
    client: &reqwest::Client,
    entry: &ModelArtifactEntry,
    target_dir: &Path,
    listener: &P,
) -> Result<PathBuf, String> {
    fs::create_dir_all(target_dir)
        .map_err(|e| format!("failed to create models dir {}: {e}", target_dir.display()))?;

    let final_path = target_dir.join(&entry.filename);
    if entry.filename == "fox.3b-q4km.gguf" && !final_path.exists() {
        let tiger_path = target_dir.join("tiger.3b-q4km.gguf");
        if tiger_path.exists() && verify_file_sha256(&tiger_path, &entry.sha256).unwrap_or(false) {
            let _ = fs::copy(&tiger_path, &final_path);
            let tiger_tok = target_dir.join("tiger-tokenizer.json");
            let fox_tok = target_dir.join("fox-tokenizer.json");
            if tiger_tok.exists() && !fox_tok.exists() {
                let _ = fs::copy(&tiger_tok, &fox_tok);
            }
        }
    }
    if final_path.exists() {
        if verify_file_sha256(&final_path, &entry.sha256).unwrap_or(false) {
            if let Some(tokenizer) = entry.tokenizer.as_ref() {
                ensure_tokenizer_asset(client, tokenizer, target_dir).await?;
            }
            listener.on_complete(&final_path);
            return Ok(final_path);
        } else {
            // Corrupt file exists, remove it before re-downloading
            let _ = fs::remove_file(&final_path);
        }
    }

    let tmp_path = target_dir.join(format!("{}.download.tmp", entry.filename));
    listener.on_start(&entry.brand_name, entry.size_bytes);

    // GitHub Releases enforce a 2.0 GiB per-asset limit; models > 2 GB published on
    // GitHub Releases are distributed as 1.9 GB parts (`.part-00`, `.part-01`, ...)
    // and streamed seamlessly into the single destination GGUF file.
    const GITHUB_PART_BYTES: u64 = 1_900_000_000;
    let part_urls: Vec<String> =
        if entry.download_url.contains("github.com/") && entry.size_bytes > 2_000_000_000 {
            let num_parts = entry.size_bytes.div_ceil(GITHUB_PART_BYTES);
            (0..num_parts)
                .map(|i| format!("{}.part-{i:02}", entry.download_url))
                .collect()
        } else {
            vec![entry.download_url.clone()]
        };

    let mut downloaded: u64 = 0;
    let mut hasher = Sha256::new();

    let mut file = if tmp_path.exists() {
        let meta = fs::metadata(&tmp_path)
            .map_err(|e| format!("failed reading tmp file metadata: {e}"))?;
        let existing_len = meta.len();
        if existing_len > 0 && existing_len < entry.size_bytes {
            // Read existing bytes to resume SHA-256 state and append
            let mut existing_file = File::open(&tmp_path)
                .map_err(|e| format!("failed reading tmp file {}: {e}", tmp_path.display()))?;
            let mut buf = vec![0u8; 1024 * 1024];
            let mut read_total: u64 = 0;
            while read_total < existing_len {
                let to_read = ((existing_len - read_total) as usize).min(buf.len());
                let n = existing_file
                    .read(&mut buf[..to_read])
                    .map_err(|e| format!("failed reading tmp file chunk: {e}"))?;
                if n == 0 {
                    break;
                }
                hasher.update(&buf[..n]);
                read_total += n as u64;
            }
            downloaded = read_total;
            listener.on_chunk(downloaded, entry.size_bytes);
            fs::OpenOptions::new()
                .write(true)
                .append(true)
                .open(&tmp_path)
                .map_err(|e| format!("failed to open tmp file for append: {e}"))?
        } else {
            let _ = fs::remove_file(&tmp_path);
            File::create(&tmp_path)
                .map_err(|e| format!("failed to create temp file {}: {e}", tmp_path.display()))?
        }
    } else {
        File::create(&tmp_path)
            .map_err(|e| format!("failed to create temp file {}: {e}", tmp_path.display()))?
    };

    for (part_idx, part_url) in part_urls.iter().enumerate() {
        let part_offset_start: u64 = if part_urls.len() == 1 {
            0
        } else {
            part_idx as u64 * GITHUB_PART_BYTES
        };

        let part_expected_len: u64 = if part_urls.len() == 1 {
            entry.size_bytes
        } else if part_idx + 1 == part_urls.len() {
            entry.size_bytes.saturating_sub(part_offset_start)
        } else {
            GITHUB_PART_BYTES
        };

        let part_offset_end = part_offset_start + part_expected_len;

        // Skip parts already present in resumed temp file
        if downloaded >= part_offset_end {
            continue;
        }

        let mut part_downloaded = if downloaded > part_offset_start {
            downloaded - part_offset_start
        } else {
            0
        };

        let mut retries = 0;
        const MAX_RETRIES: usize = 15;

        while part_downloaded < part_expected_len {
            let req = if part_downloaded > 0 {
                client
                    .get(part_url)
                    .header("Range", format!("bytes={part_downloaded}-"))
            } else {
                client.get(part_url)
            };

            let response = match req.send().await {
                Ok(resp)
                    if resp.status().is_success()
                        || resp.status() == reqwest::StatusCode::PARTIAL_CONTENT =>
                {
                    resp
                }
                Ok(resp) => {
                    retries += 1;
                    if retries > MAX_RETRIES {
                        let _ = fs::remove_file(&tmp_path);
                        return Err(format!(
                            "download failed with HTTP status {} on part {part_idx}",
                            resp.status()
                        ));
                    }
                    tokio::time::sleep(Duration::from_millis(1000 * (1 << retries.min(4)))).await;
                    continue;
                }
                Err(e) => {
                    retries += 1;
                    if retries > MAX_RETRIES {
                        let _ = fs::remove_file(&tmp_path);
                        return Err(format!("failed connecting to {part_url}: {e}"));
                    }
                    tokio::time::sleep(Duration::from_millis(1000 * (1 << retries.min(4)))).await;
                    continue;
                }
            };

            let is_full_response_to_range =
                response.status() == reqwest::StatusCode::OK && part_downloaded > 0;
            if is_full_response_to_range {
                // Server ignored Range header, rewind part and file
                downloaded -= part_downloaded;
                part_downloaded = 0;
                let _ = file.set_len(downloaded);
                hasher = Sha256::new();
                if let Ok(mut rf) = File::open(&tmp_path) {
                    let mut b = vec![0u8; 1024 * 1024];
                    let mut cur = 0;
                    while cur < downloaded {
                        let to_r = ((downloaded - cur) as usize).min(b.len());
                        if let Ok(n) = rf.read(&mut b[..to_r]) {
                            if n == 0 {
                                break;
                            }
                            hasher.update(&b[..n]);
                            cur += n as u64;
                        } else {
                            break;
                        }
                    }
                }
            }

            let mut stream = response.bytes_stream();
            let mut stream_interrupted = false;

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        if let Err(e) = file.write_all(&chunk) {
                            let _ = fs::remove_file(&tmp_path);
                            return Err(format!("failed to write chunk: {e}"));
                        }
                        hasher.update(&chunk);
                        let chunk_len = chunk.len() as u64;
                        part_downloaded += chunk_len;
                        downloaded += chunk_len;
                        listener.on_chunk(downloaded, entry.size_bytes);
                    }
                    Err(_) => {
                        stream_interrupted = true;
                        break;
                    }
                }
            }

            if stream_interrupted {
                retries += 1;
                if retries > MAX_RETRIES {
                    let _ = fs::remove_file(&tmp_path);
                    return Err(format!(
                        "stream interrupted and exceeded {MAX_RETRIES} retries on part {part_idx}"
                    ));
                }
                let _ = file.flush();
                tokio::time::sleep(Duration::from_millis(1000 * (1 << retries.min(4)))).await;
            }
        }
    }

    file.flush()
        .map_err(|e| format!("failed to flush downloaded file: {e}"))?;
    drop(file);

    listener.on_verify(&entry.filename);
    let computed_hash: String = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();

    if !computed_hash.eq_ignore_ascii_case(&entry.sha256) {
        let _ = fs::remove_file(&tmp_path);
        return Err(format!(
            "SHA-256 checksum mismatch for {}: expected {}, computed {}",
            entry.filename, entry.sha256, computed_hash
        ));
    }

    fs::rename(&tmp_path, &final_path).map_err(|e| {
        format!(
            "failed to move temp file {} to {}: {e}",
            tmp_path.display(),
            final_path.display()
        )
    })?;

    if let Some(tokenizer) = entry.tokenizer.as_ref() {
        ensure_tokenizer_asset(client, tokenizer, target_dir).await?;
    }

    // Preserve legacy Viper sidecar synchronization until its checksum is added
    // to the manifest alongside the other companion tokenizer assets.
    if entry.filename.contains("viper") {
        let tok_path = target_dir.join("nio-viper-tokenizer.json");
        let tok_url = entry
            .download_url
            .replace("nio-viper.Q4_0.gguf", "nio-viper-tokenizer.json");
        if !tok_path.is_file() {
            if let Ok(resp) = client.get(&tok_url).send().await {
                if resp.status().is_success() {
                    if let Ok(bytes) = resp.bytes().await {
                        let _ = fs::write(&tok_path, bytes);
                    }
                }
            }
        }
    }

    listener.on_complete(&final_path);
    Ok(final_path)
}

/// Synchronous entrypoint to ensure model is present on disk.
pub fn ensure_model_ready(model_id: &str, custom_dir: Option<&Path>) -> Result<PathBuf, String> {
    if model_id == MODEL_NANO {
        return Ok(PathBuf::from("embedded://nio-nano"));
    }
    let entry =
        get_manifest_entry(model_id).ok_or_else(|| format!("unknown model id '{model_id}'"))?;
    if entry.download_url.is_empty() || entry.sha256.len() != 64 {
        return Err(format!(
            "model '{}' has no published, verifiable artifact yet",
            entry.model_id
        ));
    }

    let dir = match custom_dir {
        Some(d) => d.to_path_buf(),
        None => resolve_models_dir(),
    };

    let final_path = dir.join(&entry.filename);
    if final_path.exists() && verify_file_sha256(&final_path, &entry.sha256).unwrap_or(false) {
        ensure_tokenizer_asset_sync(&entry, &dir)?;
        return Ok(final_path);
    }

    // Also check NIOXS_MODELS_DIR environment variable
    if let Ok(dir_env) = env::var("NIOXS_MODELS_DIR") {
        let p = PathBuf::from(dir_env).join(&entry.filename);
        if p.exists() && verify_file_sha256(&p, &entry.sha256).unwrap_or(false) {
            if let Some(parent) = p.parent() {
                ensure_tokenizer_asset_sync(&entry, parent)?;
            }
            return Ok(p);
        }
    }

    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .tcp_keepalive(Duration::from_secs(15))
        .build()
        .map_err(|e| format!("failed to build HTTP client: {e}"))?;

    let download_res = if let Ok(handle) = tokio::runtime::Handle::try_current() {
        tokio::task::block_in_place(|| {
            handle.block_on(download_and_verify_model(
                &client,
                &entry,
                &dir,
                &ConsoleProgress,
            ))
        })
    } else {
        let rt =
            tokio::runtime::Runtime::new().map_err(|e| format!("failed to create runtime: {e}"))?;
        rt.block_on(download_and_verify_model(
            &client,
            &entry,
            &dir,
            &ConsoleProgress,
        ))
    };

    download_res
}

/// Interactive or automated model probe and download.
/// If model weights are present, returns the path immediately.
/// If missing:
///   - In an interactive terminal session: prompts the developer `[Y/n]` to download via `nioxs pull`.
///   - If `auto_yes` is true (or `--yes` / `-y`): automatically downloads.
///   - In non-interactive mode without confirmation: returns an actionable error.
pub fn prompt_and_pull_if_needed(model_id: &str, auto_yes: bool) -> Result<PathBuf, String> {
    if model_id == MODEL_NANO {
        return Ok(PathBuf::from("embedded://nio-nano"));
    }
    let entry =
        get_manifest_entry(model_id).ok_or_else(|| format!("unknown model id '{model_id}'"))?;
    if entry.download_url.is_empty() || entry.sha256.len() != 64 {
        return Err(format!(
            "model '{}' has no published, verifiable artifact yet",
            entry.model_id
        ));
    }

    let dir = resolve_models_dir();
    let final_path = dir.join(&entry.filename);
    if final_path.exists() && verify_file_sha256(&final_path, &entry.sha256).unwrap_or(false) {
        ensure_tokenizer_asset_sync(&entry, &dir)?;
        return Ok(final_path);
    }

    // Also check NIOXS_MODELS_DIR environment variable
    if let Ok(dir_env) = env::var("NIOXS_MODELS_DIR") {
        let p = PathBuf::from(dir_env).join(&entry.filename);
        if p.exists() && verify_file_sha256(&p, &entry.sha256).unwrap_or(false) {
            if let Some(parent) = p.parent() {
                ensure_tokenizer_asset_sync(&entry, parent)?;
            }
            return Ok(p);
        }
    }

    let size_mb = entry.size_bytes as f64 / (1024.0 * 1024.0);

    if !auto_yes {
        use std::io::IsTerminal;
        if std::io::stdin().is_terminal() {
            eprintln!();
            eprintln!("  ┌─────────────────────────────────────────────────────────────┐");
            eprintln!(
                "  │ 📦 Model '{}' is not downloaded locally ({size_mb:.1} MB).",
                entry.model_id
            );
            eprintln!("  │    • Role     : {}", entry.brand_name);
            eprintln!("  │    • Base     : {}", entry.base_model);
            eprintln!("  │    • Target   : {}", final_path.display());
            eprintln!("  │                                                             │");
            eprint!("  │ Would you like to download it now via 'nioxs pull'? [Y/n] ");
            let _ = std::io::stderr().flush();

            let mut input = String::new();
            if std::io::stdin().read_line(&mut input).is_err() {
                return Err("Failed to read user confirmation".into());
            }
            let trimmed = input.trim().to_ascii_lowercase();
            if !trimmed.is_empty() && trimmed != "y" && trimmed != "yes" {
                eprintln!("  │ Download cancelled by user.                                 │");
                eprintln!("  └─────────────────────────────────────────────────────────────┘\n");
                return Err(format!(
                    "model '{}' download cancelled by user",
                    entry.model_id
                ));
            }
            eprintln!("  └─────────────────────────────────────────────────────────────┘\n");
        } else {
            return Err(format!(
                "model '{}' not found in {}. Run 'nioxs pull {}' or pass '--auto-pull' / '-y'.",
                entry.model_id,
                dir.display(),
                entry.model_id
            ));
        }
    }

    ensure_model_ready(model_id, Some(&dir))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::net::TcpListener;
    use std::thread;

    // Pins every manifest entry to the real upstream GGUF artifact it claims to
    // mirror. The checksum/size/url are the LFS `sha256`/`size` the Hugging Face
    // API reports for those exact files (`.vendor/`-free, fetched from
    // `https://huggingface.co/api/models/<repo>?blobs=true` on 2026-10-13), so a
    // fabricated or drifted value fails loudly here instead of silently making
    // every real download die in SHA-256 verification.
    //
    // Two entries were pinned to the wrong quantization's digest (helm carried
    // Q4_K_S's while pointing at the Q4_K_M URL, code carried a digest that
    // belongs to no published file). The `list_unused_manifest_pins` guard below
    // is what would have caught the copy-paste: these values must match
    // `default_manifest()` exactly, not merely look plausible.
    #[test]
    fn test_manifest_pins_real_upstream_artifacts() {
        let expected: [(&str, &str, u64, &str); 11] = [
            (
                MODEL_VIPER,
                "58633aa691b80840dc85586cf2cb8616deff480c64ddf406bf848db749ecc3d4",
                64_684_160,
                "https://github.com/nio-labs/nio-xs-models/releases/download/v0.1.0/viper.Q4_0.gguf",
            ),
            (
                MODEL_SKATE,
                "a00921dca072e640bb1f9730a95e3d88ed843cba3dd0a5648a6fd5e3e9e3d86b",
                86_957_792,
                "https://github.com/nio-labs/nio-xs-models/releases/download/v0.2.6/skate.vmicrolm-vector.gguf",
            ),
            (
                MODEL_MANTA,
                "fa981bab81c7cb32a7072dfd193f1e03f988eb800d7e6773a03b69c861d452a5",
                333_360_320,
                "https://github.com/nio-labs/nio-xs-models/releases/download/v0.2.7/manta.vmicrolm-vector.gguf",
            ),
            (
                MODEL_BELUGA,
                "d2d4642d36b907c8658301208e8cd4ab3c6d72051a37a780ebc35a56ff83d5ba",
                639_362_560,
                "https://github.com/nio-labs/nio-xs-models/releases/download/v0.3.0/beluga.vmicrolm-vector.gguf",
            ),
            (
                MODEL_BLUE,
                "6b673f5a2ed71eced7b6cc947beccf3cf3f669eb25d6e6233c48e47e11e9bb32",
                1_499_437_216,
                "https://github.com/nio-labs/nio-xs-models/releases/download/v0.3.1/blue.vmicrolm-reflex.gguf",
            ),
            (
                MODEL_CHEETAH,
                "cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046",
                1_117_320_768,
                "https://github.com/nio-labs/nio-xs-models/releases/download/v0.4.0/cheetah.1.5b-q4km.gguf",
            ),
            (
                MODEL_TIGER,
                "724fb256bec1ff062b2f65e4569e871ad2e95ab2a3989723d1769c54294730b7",
                2_104_932_800,
                "https://github.com/nio-labs/nio-xs-models/releases/download/v0.4.1/tiger.3b-q4km.gguf",
            ),
            (
                MODEL_FOX,
                "724fb256bec1ff062b2f65e4569e871ad2e95ab2a3989723d1769c54294730b7",
                2_104_932_800,
                "https://github.com/nio-labs/nio-xs-models/releases/download/v0.4.1/tiger.3b-q4km.gguf",
            ),
            (
                MODEL_RABBIT,
                "724fb256bec1ff062b2f65e4569e871ad2e95ab2a3989723d1769c54294730b7",
                2_104_932_800,
                "https://github.com/nio-labs/nio-xs-models/releases/download/v0.4.1/tiger.3b-q4km.gguf",
            ),
            (
                MODEL_ORCA_NANO,
                "0af05b68e6dec88ec02af0868888878fbe871e04d4c4910c15d6649ea40775ec",
                397_807_616,
                "https://github.com/nio-labs/nio-xs-models/releases/download/v0.6.0/orca-nano-lead-0.5b-q4_k_m.gguf",
            ),
            (
                MODEL_ORCA_MINI,
                "6aa0cb0f99b32d359eaae3e90d4b01222a9430717d160f3f01ae48f981b07135",
                986_048_192,
                "https://github.com/nio-labs/nio-xs-models/releases/download/v0.6.0/orca-lead-1.5b-q4_k_m.gguf",
            ),
        ];
        let manifest = default_manifest();
        assert_eq!(manifest.len(), expected.len() + 76);
        for (id, sha256, size_bytes, url) in expected {
            let entry =
                get_manifest_entry(id).unwrap_or_else(|| panic!("missing manifest entry for {id}"));
            assert_eq!(entry.sha256, sha256, "sha256 for {id}");
            assert_eq!(entry.size_bytes, size_bytes, "size for {id}");
            assert_eq!(entry.download_url, url, "download_url for {id}");
        }
        let manta = get_manifest_entry(MODEL_MANTA).unwrap();
        assert_eq!(
            manta.tokenizer,
            Some(ModelTokenizerArtifact {
                filename: "manta-tokenizer.json".into(),
                download_url: "https://github.com/nio-labs/nio-xs-models/releases/download/v0.2.7/manta-tokenizer.json".into(),
                sha256: "ade71ec5118be2cdd8aef3a7dcd417c51fa7e76b3e164adb339f446b42e97bc8".into(),
                size_bytes: 14_569_356,
            })
        );
        let beluga = get_manifest_entry(MODEL_BELUGA).unwrap();
        assert_eq!(
            beluga.tokenizer,
            Some(ModelTokenizerArtifact {
                filename: "beluga-tokenizer.json".into(),
                download_url: "https://github.com/nio-labs/nio-xs-models/releases/download/v0.3.0/beluga-tokenizer.json".into(),
                sha256: "cf535eea1ed9374c2bc3436625216c18fc703fc37aae8e019b3e85cc658e0395".into(),
                size_bytes: 11_422_174,
            })
        );
        let blue = get_manifest_entry(MODEL_BLUE).unwrap();
        assert_eq!(
            blue.tokenizer,
            Some(ModelTokenizerArtifact {
                filename: "blue-tokenizer.json".into(),
                download_url: "https://github.com/nio-labs/nio-xs-models/releases/download/v0.3.1/blue-tokenizer.json".into(),
                sha256: "8fcc6f6a32f0bd5ccdac48a0522dcd1cc575e5daf591f09d489e07f0d9035b2e".into(),
                size_bytes: 3_911_372,
            })
        );
        let cheetah = get_manifest_entry(MODEL_CHEETAH).unwrap();
        assert_eq!(
            cheetah.tokenizer,
            Some(ModelTokenizerArtifact {
                filename: "cheetah-tokenizer.json".into(),
                download_url: "https://github.com/nio-labs/nio-xs-models/releases/download/v0.4.0/cheetah-tokenizer.json".into(),
                sha256: "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539".into(),
                size_bytes: 7_031_645,
            })
        );
        let tiger = get_manifest_entry(MODEL_TIGER).unwrap();
        assert_eq!(
            tiger.tokenizer,
            Some(ModelTokenizerArtifact {
                filename: "tiger-tokenizer.json".into(),
                download_url: "https://github.com/nio-labs/nio-xs-models/releases/download/v0.4.1/tiger-tokenizer.json".into(),
                sha256: "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539".into(),
                size_bytes: 7_031_645,
            })
        );
        let fox = get_manifest_entry(MODEL_FOX).unwrap();
        assert_eq!(
            fox.tokenizer,
            Some(ModelTokenizerArtifact {
                filename: "fox-tokenizer.json".into(),
                download_url: "https://github.com/nio-labs/nio-xs-models/releases/download/v0.4.1/tiger-tokenizer.json".into(),
                sha256: "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539".into(),
                size_bytes: 7_031_645,
            })
        );
        let rabbit = get_manifest_entry(MODEL_RABBIT).unwrap();
        assert_eq!(
            rabbit.tokenizer,
            Some(ModelTokenizerArtifact {
                filename: "rabbit-tokenizer.json".into(),
                download_url: "https://github.com/nio-labs/nio-xs-models/releases/download/v0.4.1/tiger-tokenizer.json".into(),
                sha256: "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539".into(),
                size_bytes: 7_031_645,
            })
        );
    }

    #[test]
    fn test_manifest_contains_all_models() {
        let manifest = default_manifest();
        assert_eq!(manifest.len(), 87);
        for id in [
            MODEL_VIPER,
            MODEL_SKATE,
            MODEL_MANTA,
            MODEL_BELUGA,
            MODEL_BLUE,
            MODEL_CHEETAH,
            MODEL_TIGER,
            MODEL_FOX,
            MODEL_RABBIT,
            MODEL_ORCA_NANO,
            MODEL_ORCA_MINI,
            "health-nutrition-fitness",
            "health-general-info",
            "everyday-cooking-travel",
            "everyday-diy-life",
        ] {
            assert!(manifest.iter().any(|e| e.model_id == id));
        }
    }

    #[test]
    fn test_is_model_ready_embedded_nano() {
        assert!(is_model_ready(MODEL_NANO, None));
    }

    #[test]
    fn test_verify_file_sha256_success_and_failure() {
        let temp_dir = env::temp_dir().join(format!("nioxs_test_{}", std::process::id()));
        fs::create_dir_all(&temp_dir).unwrap();
        let file_path = temp_dir.join("test_file.bin");

        let content = b"hello nio models";
        fs::write(&file_path, content).unwrap();

        let mut hasher = Sha256::new();
        hasher.update(content);
        let expected: String = hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();

        assert_eq!(verify_file_sha256(&file_path, &expected).unwrap(), true);
        assert_eq!(
            verify_file_sha256(&file_path, "00000000000000000000000000000000").unwrap(),
            false
        );

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[tokio::test]
    async fn test_download_and_verify_mock_server() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let payload = b"GGUF_MOCK_MODEL_DATA";
        let mut hasher = Sha256::new();
        hasher.update(payload);
        let expected_sha256: String = hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();

        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut chunk = [0u8; 1024];
            let _ = stream.read(&mut chunk).unwrap();
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                payload.len()
            );
            stream.write_all(resp.as_bytes()).unwrap();
            stream.write_all(payload).unwrap();
        });

        let temp_dir = env::temp_dir().join(format!("nioxs_mock_{}", std::process::id()));
        let entry = ModelArtifactEntry {
            model_id: "test-model".into(),
            brand_name: "Test Model".into(),
            base_model: "Mock-1B".into(),
            filename: "mock_model.gguf".into(),
            download_url: format!("http://{addr}/model.gguf"),
            sha256: expected_sha256.clone(),
            size_bytes: payload.len() as u64,
            tokenizer: None,
        };

        let client = reqwest::Client::new();
        let downloaded = download_and_verify_model(&client, &entry, &temp_dir, &SilentProgress)
            .await
            .unwrap();

        server.join().unwrap();
        assert!(downloaded.exists());
        assert_eq!(
            verify_file_sha256(&downloaded, &expected_sha256).unwrap(),
            true
        );

        let _ = fs::remove_dir_all(&temp_dir);
    }

    /// Records the callback sequence so the provider's progress reporting has a
    /// contract to build on: `on_start` first, `on_chunk` per stream chunk,
    /// `on_verify` before the digest is compared, `on_complete` last.
    #[derive(Default)]
    struct RecordingProgress {
        events: std::sync::Mutex<Vec<String>>,
    }

    impl RecordingProgress {
        fn push(&self, event: String) {
            self.events.lock().unwrap().push(event);
        }
        fn events(&self) -> Vec<String> {
            self.events.lock().unwrap().clone()
        }
    }

    impl ProgressListener for RecordingProgress {
        fn on_start(&self, brand_name: &str, total_bytes: u64) {
            self.push(format!("start:{brand_name}:{total_bytes}"));
        }
        fn on_chunk(&self, downloaded_bytes: u64, total_bytes: u64) {
            self.push(format!("chunk:{downloaded_bytes}/{total_bytes}"));
        }
        fn on_verify(&self, filename: &str) {
            self.push(format!("verify:{filename}"));
        }
        fn on_complete(&self, _path: &Path) {
            self.push("complete".into());
        }
    }

    #[tokio::test]
    async fn test_progress_listener_reports_start_chunks_verify_and_complete() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let payload = b"GGUF_PROGRESS_MOCK_DATA";
        let mut hasher = Sha256::new();
        hasher.update(payload);
        let expected_sha256: String = hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();

        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut chunk = [0u8; 1024];
            let _ = stream.read(&mut chunk).unwrap();
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                payload.len()
            );
            stream.write_all(resp.as_bytes()).unwrap();
            stream.write_all(payload).unwrap();
        });

        let temp_dir = env::temp_dir().join(format!("nioxs_progress_{}", std::process::id()));
        let entry = ModelArtifactEntry {
            model_id: "test-progress".into(),
            brand_name: "Progress Model".into(),
            base_model: "Mock-1B".into(),
            filename: "progress.gguf".into(),
            download_url: format!("http://{addr}/model.gguf"),
            sha256: expected_sha256,
            size_bytes: payload.len() as u64,
            tokenizer: None,
        };

        let client = reqwest::Client::new();
        let progress = RecordingProgress::default();
        download_and_verify_model(&client, &entry, &temp_dir, &progress)
            .await
            .unwrap();
        server.join().unwrap();

        let events = progress.events();
        let size = payload.len();
        let expected_start = format!("start:Progress Model:{size}");
        assert_eq!(
            events.first().map(String::as_str),
            Some(expected_start.as_str()),
            "on_start announces the total before any bytes move: {events:?}"
        );
        assert_eq!(events.last().map(String::as_str), Some("complete"));
        assert_eq!(
            events.iter().filter(|e| e.starts_with("verify:")).count(),
            1,
            "verification is announced exactly once: {events:?}"
        );
        assert!(
            events.iter().any(|e| e == &format!("chunk:{size}/{size}")),
            "final chunk must report full progress: {events:?}"
        );
        let verify_at = events.iter().position(|e| e.starts_with("verify:"));
        let complete_at = events.iter().position(|e| e == "complete");
        assert!(
            verify_at < complete_at,
            "verify precedes complete: {events:?}"
        );

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[tokio::test]
    async fn test_download_checksum_mismatch_cleans_up() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let payload = b"CORRUPTED_DATA";

        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut chunk = [0u8; 1024];
            let _ = stream.read(&mut chunk).unwrap();
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                payload.len()
            );
            stream.write_all(resp.as_bytes()).unwrap();
            stream.write_all(payload).unwrap();
        });

        let temp_dir = env::temp_dir().join(format!("nioxs_corrupt_{}", std::process::id()));
        let entry = ModelArtifactEntry {
            model_id: "test-corrupt".into(),
            brand_name: "Corrupt Model".into(),
            base_model: "Mock-1B".into(),
            filename: "corrupt.gguf".into(),
            download_url: format!("http://{addr}/model.gguf"),
            sha256: "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff".into(),
            size_bytes: payload.len() as u64,
            tokenizer: None,
        };

        let client = reqwest::Client::new();
        let err = download_and_verify_model(&client, &entry, &temp_dir, &SilentProgress)
            .await
            .unwrap_err();

        server.join().unwrap();
        assert!(err.contains("checksum mismatch"));
        assert!(!temp_dir.join("corrupt.gguf").exists());
        assert!(!temp_dir.join("corrupt.gguf.download.tmp").exists());

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
