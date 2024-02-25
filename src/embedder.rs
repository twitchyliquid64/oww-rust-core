use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::Arc;
use std::thread;

use crate::Melspectogram;
use circular_buffer::CircularBuffer;
use tract_onnx::prelude::*;

pub const NUM_SPECTOGRAMS: usize = 76;

#[derive(Clone, Debug)]
pub struct Embedding([f32; 96]);

// derive(Default) doesnt work on arrays > 32, grrrr
impl Default for Embedding {
    fn default() -> Self {
        Self([0f32; 96])
    }
}

impl Embedding {
    pub fn iter(&self) -> core::slice::Iter<'_, f32> {
        self.0.iter()
    }
}

/// Embedder collects chunks of melspectograms and outputs embeddings.
pub struct Embedder {
    recv: Option<Receiver<Embedding>>,
    shutdown: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Embedder {
    pub fn start(
        spectos: Receiver<Vec<Melspectogram>>,
    ) -> Result<Self, tract_onnx::tract_core::anyhow::Error> {
        let emb_model = tract_onnx::onnx()
            // load the model
            .model_for_path("embedding_model.onnx")?
            .with_input_fact(0, f32::fact([1, 76, 32, 1]).into())
            .unwrap()
            .into_optimized()?
            .into_runnable()?;

        let (send, recv) = sync_channel(1);
        let shutdown = Arc::new(AtomicBool::new(false));

        let shutdown2 = shutdown.clone();
        let thread = Some(thread::spawn(move || {
            Embedder::mainloop(send, shutdown2, spectos, emb_model);
        }));

        let out = Self {
            shutdown,
            thread,
            recv: Some(recv),
        };

        Ok(out)
    }

    pub fn take_receiver(&mut self) -> Option<Receiver<Embedding>> {
        self.recv.take()
    }

    fn mainloop(
        tx: SyncSender<Embedding>,
        shutdown: Arc<AtomicBool>,
        spectos: Receiver<Vec<Melspectogram>>,
        emb_model: TypedRunnableModel<TypedModel>,
    ) {
        let mut spectograms = CircularBuffer::<NUM_SPECTOGRAMS, Melspectogram>::new();

        loop {
            if shutdown.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }
            match spectos.recv() {
                Ok(s) => s.into_iter().for_each(|s| spectograms.push_back(s)),
                Err(_e) => return,
            };
            if shutdown.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }

            // Don't compute the embeddings unless we have a full set of input (76 spectograms)
            // for the model
            if !spectograms.is_full() {
                continue;
            }

            // Build a tensor that will be the input to the embedding model, which is [?, 76, 32, 1].
            // I presume that means [batch_size=1, num_melspectograms=76, num_spect_bins=32, ?].
            let embedding_input: Tensor =
                tract_ndarray::Array::<f32, tract_ndarray::Dim<[usize; 1]>>::from_iter(
                    spectograms
                        .iter()
                        .map(|spect| spect.iter())
                        .flatten()
                        .copied(),
                )
                .into_shape((1, NUM_SPECTOGRAMS, 32, 1))
                .unwrap()
                .into();
            // println!("model: {:?}", embedding_model.model());

            // Compute the embedding for this chunk of spectograms.
            let out = emb_model
                .run(tvec!(embedding_input.into()))
                .unwrap()
                .remove(0);
            let mut embedding = Embedding::default();
            embedding.0.clone_from_slice(out.as_slice::<f32>().unwrap());

            if shutdown.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }

            if let Err(e) = tx.send(embedding) {
                println!("failed send, embedding thread shutting down! {:?}", e);
                return;
            }
        }
    }
}

impl Drop for Embedder {
    fn drop(&mut self) {
        self.shutdown
            .store(true, std::sync::atomic::Ordering::SeqCst);
        if let Some(hnd) = self.thread.take() {
            hnd.join().ok();
        }
    }
}
