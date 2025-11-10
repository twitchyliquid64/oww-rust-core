use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread;

use earshot::{VoiceActivityDetector, VoiceActivityProfile};

use crate::Chunk;

/// VAD collects chunks of samples and computes a probability that voice is present.
pub struct VAD {
    recv: Option<Receiver<bool>>,
    shutdown: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl VAD {
    pub fn start(samples: Receiver<Chunk<480>>) -> Result<Self, anyhow::Error> {
        // let mut sr = i64::fact(&[1]);
        // sr.konst = Some(rctensor0(16000));

        // let spec_model = tract_onnx::onnx()
        //     // load the model
        //     .model_for_path("silero_vad_16k_op15.onnx")?
        //     .with_input_names(["input", "state", "sr"])?
        //     .with_output_names(["output", "stateN"])?
        //     .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 512)))?
        //     .with_input_fact(
        //         1,
        //         InferenceFact::dt_shape(f32::datum_type(), tvec!(2, 1, 128)),
        //     )?
        //     .with_input_fact(2, sr.into())?;
        // let spec_model = spec_model.into_optimized()?.into_runnable()?;

        let (send, recv) = sync_channel(1);
        let shutdown = Arc::new(AtomicBool::new(false));

        let shutdown2 = shutdown.clone();
        let thread = Some(thread::spawn(move || {
            VAD::mainloop(
                send,
                shutdown2,
                samples,
                VoiceActivityDetector::new(VoiceActivityProfile::VERY_AGGRESSIVE),
            );
        }));

        let out = Self {
            shutdown,
            thread,
            recv: Some(recv),
        };

        Ok(out)
    }

    pub fn take_receiver(&mut self) -> Option<Receiver<bool>> {
        self.recv.take()
    }

    fn mainloop(
        tx: SyncSender<bool>,
        shutdown: Arc<AtomicBool>,
        samples: Receiver<Chunk<480>>,
        mut vad_model: VoiceActivityDetector,
    ) {
        let mut tensor_data: Vec<i16> = Vec::with_capacity(480);

        loop {
            if shutdown.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }
            let new_samples = match samples.recv() {
                Ok(s) => s,
                Err(_e) => return,
            };
            if shutdown.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }

            tensor_data.clear();
            tensor_data.extend(new_samples.samples.iter().map(|s| {
                (*s * (i16::MAX as f32))
                    .max(i16::MIN as f32)
                    .min(i16::MAX as f32) as i16
            }));

            let out = vad_model.predict_16khz(&tensor_data).unwrap();
            if shutdown.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }

            if let Err(e) = tx.try_send(out) {
                if matches!(e, TrySendError::Disconnected(_)) {
                    println!("failed send, VAD thread shutting down! {:?}", e);
                    return;
                }
            }
        }
    }
}

impl Drop for VAD {
    fn drop(&mut self) {
        self.shutdown
            .store(true, std::sync::atomic::Ordering::SeqCst);
        if let Some(hnd) = self.thread.take() {
            hnd.join().ok();
        }
    }
}
