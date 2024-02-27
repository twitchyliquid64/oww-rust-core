use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::Embedding;
use circular_buffer::CircularBuffer;
use tract_onnx::prelude::*;

#[derive(Clone, Debug)]
pub struct ModelFilters {
    pub scale: f32,        // how much to scale model_val
    pub clamp: (f32, f32), // bounds for output value
}

impl Default for ModelFilters {
    fn default() -> Self {
        Self {
            scale: 1.0,
            clamp: (0.0, 1.0),
        }
    }
}

impl ModelFilters {
    fn apply(&self, model: f32) -> f32 {
        (model * self.scale).min(self.clamp.1).max(self.clamp.0)
    }
}

pub struct NamedModel {
    name: String,
    model: TypedRunnableModel<TypedModel>,
    filters: ModelFilters,
}

impl NamedModel {
    pub fn new<S: Into<String>>(
        name: S,
        path: S,
    ) -> Result<Self, tract_onnx::tract_core::anyhow::Error> {
        let model = tract_onnx::onnx()
            // load the model
            .model_for_path(path.into())?
            .into_optimized()?
            .into_runnable()?;

        let name = name.into();
        let filters = ModelFilters::default();
        Ok(Self {
            name,
            model,
            filters,
        })
    }

    fn apply(&mut self, model_val: f32) -> f32 {
        self.filters.apply(model_val)
    }
}

pub const NUM_EMBEDDINGS: usize = 16;

/// Runner computes watch-word activations over embeddings.
pub struct Runner {
    models: Arc<Mutex<Vec<NamedModel>>>,
    recv: Option<Receiver<Vec<(String, f32)>>>,
    shutdown: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Runner {
    pub fn start(
        embeddings: Receiver<Embedding>,
    ) -> Result<Self, tract_onnx::tract_core::anyhow::Error> {
        let models = Arc::new(Mutex::new(vec![]));
        let (send, recv) = sync_channel(1);
        let shutdown = Arc::new(AtomicBool::new(false));

        let models2 = models.clone();
        let shutdown2 = shutdown.clone();
        let thread = Some(thread::spawn(move || {
            Runner::mainloop(send, models2, shutdown2, embeddings);
        }));

        let out = Self {
            models,
            shutdown,
            thread,
            recv: Some(recv),
        };

        Ok(out)
    }

    pub fn take_receiver(&mut self) -> Option<Receiver<Vec<(String, f32)>>> {
        self.recv.take()
    }

    pub fn add_model(&mut self, model: NamedModel) {
        let models: &mut Vec<NamedModel> = &mut self.models.lock().unwrap();
        models.retain(|m| m.name != model.name);
        models.push(model);
    }

    fn mainloop(
        tx: SyncSender<Vec<(String, f32)>>,
        models: Arc<Mutex<Vec<NamedModel>>>,
        shutdown: Arc<AtomicBool>,
        embs: Receiver<Embedding>,
    ) {
        let mut embeddings = CircularBuffer::<NUM_EMBEDDINGS, Embedding>::new();

        loop {
            if shutdown.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }
            match embs.recv() {
                Ok(s) => embeddings.push_back(s),
                Err(_e) => return,
            };
            if shutdown.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }

            // Don't compute activations unless we have a full set of input (16 embeddings)
            if !embeddings.is_full() {
                continue;
            }
            // Build a tensor that will be the input to the feature model, which is [1, 16, 96].
            let feature_input: Tensor =
                tract_ndarray::Array::<f32, tract_ndarray::Dim<[usize; 1]>>::from_iter(
                    embeddings
                        .iter()
                        .map(|spect| spect.iter())
                        .flatten()
                        .copied(),
                )
                .into_shape((1, NUM_EMBEDDINGS, 96))
                .unwrap()
                .into();
            let results = models
                .lock()
                .unwrap()
                .iter_mut()
                .map(|m| {
                    (
                        m.name.clone(),
                        m.apply(
                            m.model
                                .run(tvec!(feature_input.clone().into()))
                                .unwrap()
                                .remove(0)
                                .as_slice()
                                .unwrap()[0],
                        ),
                    )
                })
                .collect();

            if shutdown.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }

            if let Err(e) = tx.send(results) {
                println!("failed send, model thread shutting down! {:?}", e);
                return;
            }
        }
    }
}

impl Drop for Runner {
    fn drop(&mut self) {
        self.shutdown
            .store(true, std::sync::atomic::Ordering::SeqCst);
        if let Some(hnd) = self.thread.take() {
            hnd.join().ok();
        }
    }
}
