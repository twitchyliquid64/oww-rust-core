use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::Arc;
use std::thread;

use crate::{SampleBuffer, SAMPLES_PER_BUFFER};
use circular_buffer::CircularBuffer;
use tract_onnx::prelude::*;

#[derive(Default, Clone, Debug)]
pub struct Melspectogram([f32; 32]);

impl Melspectogram {
    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, f32> {
        self.0.iter_mut()
    }

    pub fn iter(&self) -> core::slice::Iter<'_, f32> {
        self.0.iter()
    }
}

/// Specter collects chunks of samples and outputs its melspectogram.
pub struct Specter {
    recv: Option<Receiver<Vec<Melspectogram>>>,
    shutdown: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Specter {
    pub fn start(
        samples: Receiver<SampleBuffer>,
    ) -> Result<Self, tract_onnx::tract_core::anyhow::Error> {
        let spec_model = tract_onnx::onnx()
            // load the model
            .model_for_path("melspectrogram.onnx")?
            .into_optimized()?
            .into_runnable()?;

        let (send, recv) = sync_channel(1);
        let shutdown = Arc::new(AtomicBool::new(false));

        let shutdown2 = shutdown.clone();
        let thread = Some(thread::spawn(move || {
            Specter::mainloop(send, shutdown2, samples, spec_model);
        }));

        let out = Self {
            shutdown,
            thread,
            recv: Some(recv),
        };

        Ok(out)
    }

    pub fn take_receiver(&mut self) -> Option<Receiver<Vec<Melspectogram>>> {
        self.recv.take()
    }

    fn mainloop(
        tx: SyncSender<Vec<Melspectogram>>,
        shutdown: Arc<AtomicBool>,
        samples: Receiver<SampleBuffer>,
        spec_model: TypedRunnableModel<TypedModel>,
    ) {
        // Compute the co-efficients to apply the hamming window.
        let co_effs: Vec<_> = apodize::hamming_iter(SAMPLES_PER_BUFFER)
            .map(|x| x as f32)
            .collect();

        // We track three buffers: the one we last computed, the one we are computing now,
        // and the one we will compute next. We use the last and next to overlap with
        // the one we are currently computing now.
        let mut buffers = CircularBuffer::<3, SampleBuffer>::new();

        loop {
            if shutdown.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }
            buffers.push_back(match samples.recv() {
                Ok(s) => s,
                Err(_e) => return,
            });
            if !buffers.is_full() {
                continue;
            }

            if shutdown.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }

            // Overlap by 50% with the buffer before and after: achieving what
            // the literature would call a hamming window with 50% overlap.
            let mut s = buffers.get(1).unwrap().clone();
            s.iter_mut()
                .zip(
                    (0..SAMPLES_PER_BUFFER / 2).map(|_| None).chain(
                        buffers
                            .get(2)
                            .unwrap()
                            .iter()
                            .take(SAMPLES_PER_BUFFER / 2)
                            .map(|s| Some(s)),
                    ),
                )
                .zip(
                    buffers
                        .get(0)
                        .unwrap()
                        .iter()
                        .skip(SAMPLES_PER_BUFFER / 2)
                        .map(|s| Some(s))
                        .chain((0..SAMPLES_PER_BUFFER / 2).map(|_| None)),
                )
                .enumerate()
                .for_each(|(i, ((s, before), after))| {
                    *s = *s
                        + 0.32 * co_effs[i] * before.unwrap_or(&0.)
                        + 0.25 * co_effs[SAMPLES_PER_BUFFER - i - 1] * after.unwrap_or(&0.);
                });

            let samples: Tensor = s.into();

            // run the spectogram on the input
            let out = spec_model.run(tvec!(samples.into())).unwrap().remove(0);

            // so the spectogram output is [1, 1, 5, 32] but we only care about each 32-float sequence,
            // each of which represents a spectogram. Lets iterate in those chunks and add it to our buffer.
            let mut spects: Vec<Melspectogram> = Vec::with_capacity(5);
            spects.extend(out.as_slice::<f32>().unwrap().chunks(32).map(|chunk| {
                let mut out = Melspectogram::default();
                chunk
                    .into_iter()
                    .zip(out.iter_mut())
                    .for_each(|(input, output)| {
                        // Don't h8 this is what openWakeWords does! https://github.com/dscripka/openWakeWord/blob/main/openwakeword/utils.py#L180
                        // ¯\_(ツ)_/¯  ¯\_(ツ)_/¯  ¯\_(ツ)_/¯  ¯\_(ツ)_/¯
                        *output = *input / 10.0 + 2.0;
                    });
                out
            }));
            if shutdown.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }

            if let Err(e) = tx.send(spects) {
                println!("failed send, specter thread shutting down! {:?}", e);
                return;
            }
        }
    }
}

impl Drop for Specter {
    fn drop(&mut self) {
        self.shutdown
            .store(true, std::sync::atomic::Ordering::SeqCst);
        if let Some(hnd) = self.thread.take() {
            hnd.join().ok();
        }
    }
}
