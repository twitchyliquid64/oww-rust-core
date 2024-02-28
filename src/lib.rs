use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

mod oww;
pub use oww::OWW;

mod sampler;
pub use sampler::{SampleBuffer, Sampler, SAMPLES_PER_BUFFER};

mod specter;
pub use specter::{Melspectogram, Specter};

mod embedder;
pub use embedder::{Embedder, Embedding, NUM_SPECTOGRAMS};

mod runner;
pub use runner::{NamedModel, Runner, NUM_EMBEDDINGS};

mod matcher;
pub use matcher::Matcher;

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
