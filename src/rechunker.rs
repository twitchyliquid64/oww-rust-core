use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::Arc;
use std::thread;

use std::collections::VecDeque;

use crate::Chunk;

/// Specter collects chunks of samples and outputs its melspectogram.
pub struct Rechunker<const I: usize, const O: usize> {
    recv: Option<Receiver<Chunk<O>>>,
    shutdown: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl<const I: usize, const O: usize> Rechunker<I, O> {
    pub fn start(samples: Receiver<Chunk<I>>) -> Result<Self, anyhow::Error> {
        let (send, recv) = sync_channel(1);
        let shutdown = Arc::new(AtomicBool::new(false));

        let shutdown2 = shutdown.clone();
        let thread = Some(thread::spawn(move || {
            Rechunker::mainloop(send, shutdown2, samples);
        }));

        let out = Self {
            shutdown,
            thread,
            recv: Some(recv),
        };

        Ok(out)
    }

    pub fn take_receiver(&mut self) -> Option<Receiver<Chunk<O>>> {
        self.recv.take()
    }

    fn mainloop(tx: SyncSender<Chunk<O>>, shutdown: Arc<AtomicBool>, samples: Receiver<Chunk<I>>) {
        let mut buffer = VecDeque::with_capacity(I.max(O) * 2);
        let mut next_id = 0u64;

        loop {
            if shutdown.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }
            buffer.extend(
                match samples.recv() {
                    Ok(s) => s,
                    Err(_e) => return,
                }
                .samples,
            );

            if shutdown.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }
            // Emit output chunks while we have enough samples buffered
            while buffer.len() >= O {
                let mut output_samples = [0f32; O];
                for s in output_samples.iter_mut().take(O) {
                    *s = buffer.pop_front().unwrap();
                }

                let output_chunk = Chunk {
                    id: next_id,
                    samples: output_samples,
                };
                next_id += 1;

                if shutdown.load(std::sync::atomic::Ordering::SeqCst) {
                    return;
                }

                if let Err(e) = tx.send(output_chunk) {
                    println!("failed send, rechunker thread shutting down! {:?}", e);
                    return;
                }
            }
        }
    }
}

impl<const I: usize, const O: usize> Drop for Rechunker<I, O> {
    fn drop(&mut self) {
        self.shutdown
            .store(true, std::sync::atomic::Ordering::SeqCst);
        if let Some(hnd) = self.thread.take() {
            hnd.join().ok();
        }
    }
}
