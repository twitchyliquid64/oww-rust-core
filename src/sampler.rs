use std::process::{Child, Command, Stdio};

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread;

use crate::Chunk;

pub const SAMPLE_RATE: usize = 16000;

/// Sampler collects audio samples and outputs fixed-size chunks of samples. Chunks
/// are windowed using the hamming function.
pub struct Sampler<const S: usize> {
    child: Child,
    recv: Option<Receiver<Chunk<S>>>,
    shutdown: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl<const S: usize> Sampler<S> {
    pub fn start(preamp: Option<f32>) -> Result<Self, std::io::Error> {
        let mut child = Command::new("arecord")
            .arg("-r")
            .arg(SAMPLE_RATE.to_string())
            .arg("-c")
            .arg("1")
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

    pub fn take_receiver(&mut self) -> Option<Receiver<Chunk<S>>> {
        self.recv.take()
    }

    fn mainloop(
        scale: f32,
        tx: Sender<Chunk<S>>,
        shutdown: Arc<AtomicBool>,
        mut stdout: std::process::ChildStdout,
    ) {
        let mut chunk_id = 0;
        loop {
            let samples: Vec<f32> = (0..S)
                .map(|_| {
                    if shutdown.load(std::sync::atomic::Ordering::Relaxed) {
                        return 0.0;
                    }

                    use std::io::Read;
                    let mut buffer = [0u8; std::mem::size_of::<u16>()];
                    stdout.read_exact(&mut buffer).unwrap();

                    let sample = i16::from_le_bytes(buffer);
                    (sample as f32) * scale / (i16::MAX as f32)
                })
                .collect();

            let chunk = Chunk {
                id: chunk_id,
                samples: samples.as_slice().try_into().unwrap(),
            };
            chunk_id += 1;

            if shutdown.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }

            if let Err(e) = tx.send(chunk) {
                println!("dropping sample buffer! {:?}", e);
            }
        }
    }
}

impl<const S: usize> Drop for Sampler<S> {
    fn drop(&mut self) {
        self.shutdown
            .store(true, std::sync::atomic::Ordering::SeqCst);
        self.child.kill().ok();
        if let Some(hnd) = self.thread.take() {
            hnd.join().ok();
        }
    }
}
