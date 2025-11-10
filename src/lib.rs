use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

mod delay;
pub use delay::Delay;

mod rechunker;
pub use rechunker::Rechunker;

mod sampler;
pub use sampler::Sampler;

mod vad;
pub use vad::VAD;

mod specter;
pub use specter::{Melspectogram, SPECTOGRAM_SAMPLES, Specter};

mod tee;
pub use tee::Tee;

mod embedder;
pub use embedder::{Embedder, Embedding, NUM_SPECTOGRAMS};

mod runner;
pub use runner::{NUM_EMBEDDINGS, NamedModel, Runner};

mod matcher;
pub use matcher::Matcher;

/// A fixed-size buffer of contiguous audio samples.
#[derive(Debug, Clone, PartialEq)]
pub struct Chunk<const S: usize> {
    pub id: u64,
    pub samples: [f32; S],
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub path: String,
    pub scale: Option<f32>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct MatchStageConfig {
    pub model: String,
    pub activation_threshold: Option<f32>,
    pub timeout_ms: Option<usize>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct MatchConfig {
    pub chain: Vec<MatchStageConfig>,
    pub action: String,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Config {
    pub models: BTreeMap<String, ModelConfig>,
    pub matchers: BTreeMap<String, MatchConfig>,
}
