use std::process::{Child, Command, Stdio};

use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread;

use tract_onnx::prelude::*;

pub const SAMPLES_PER_BUFFER: usize = 1280;

pub type SampleBuffer =
    tract_ndarray::ArrayBase<tract_ndarray::OwnedRepr<f32>, tract_ndarray::Dim<[usize; 2]>>;

/// Sampler collects audio samples and outputs fixed-size chunks of samples. Chunks
/// are windowed using the hamming function.
pub struct Sampler {
    child: Child,
    recv: Option<Receiver<SampleBuffer>>,
    shutdown: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Sampler {
    pub fn start(preamp: Option<f32>) -> Result<Self, std::io::Error> {
        let mut child = Command::new("arecord")
            .arg("-r")
            .arg("16000")
            .arg("-f")
            .arg("S16_LE")
            .stdout(Stdio::piped())
            .spawn()?;

        let (send, recv) = channel();
        let shutdown = Arc::new(AtomicBool::new(false));

        let shutdown2 = shutdown.clone();
        let stdout = child.stdout.take().unwrap();
        let thread = Some(thread::spawn(move || {
            Sampler::mainloop(preamp.unwrap_or(0.1), send, shutdown2, stdout);
        }));

        let out = Self {
            child,
            recv: Some(recv),
            shutdown,
            thread,
        };

        Ok(out)
    }

    pub fn take_receiver(&mut self) -> Option<Receiver<SampleBuffer>> {
        self.recv.take()
    }

    fn mainloop(
        scale: f32,
        tx: Sender<SampleBuffer>,
        shutdown: Arc<AtomicBool>,
        mut stdout: std::process::ChildStdout,
    ) {
        loop {
            let buff: SampleBuffer =
                tract_ndarray::Array2::from_shape_fn((1, SAMPLES_PER_BUFFER), |(_, _i)| {
                    if shutdown.load(std::sync::atomic::Ordering::Relaxed) {
                        return 0.0;
                    }

                    use std::io::Read;
                    let mut buffer = [0u8; std::mem::size_of::<u16>()];
                    stdout.read_exact(&mut buffer).unwrap();

                    let sample = i16::from_le_bytes(buffer);
                    (sample as f32) * scale
                });

            if shutdown.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }

            if let Err(e) = tx.send(buff) {
                println!("dropping sample buffer! {:?}", e);
            }
        }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        self.shutdown
            .store(true, std::sync::atomic::Ordering::SeqCst);
        self.child.kill().ok();
        if let Some(hnd) = self.thread.take() {
            hnd.join().ok();
        }
    }
}
