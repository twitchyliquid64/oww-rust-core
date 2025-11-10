use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::thread;

use crate::Chunk;

/// Delay delays chunks by a certain amount.
pub struct Delay<const S: usize, const D: usize> {
    recv: Option<Receiver<Chunk<S>>>,
    shutdown: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl<const S: usize, const D: usize> Delay<S, D> {
    pub fn start(samples: Receiver<Chunk<S>>) -> Result<Self, anyhow::Error> {
        let (send, recv) = sync_channel(1);
        let shutdown = Arc::new(AtomicBool::new(false));

        let shutdown2 = shutdown.clone();
        let thread = Some(thread::spawn(move || {
            Delay::<S, D>::mainloop(send, shutdown2, samples);
        }));

        let out = Self {
            shutdown,
            thread,
            recv: Some(recv),
        };

        Ok(out)
    }

    pub fn take_receiver(&mut self) -> Option<Receiver<Chunk<S>>> {
        self.recv.take()
    }

    fn mainloop(tx: SyncSender<Chunk<S>>, shutdown: Arc<AtomicBool>, samples: Receiver<Chunk<S>>) {
        let mut queue = VecDeque::with_capacity(D);

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

            queue.push_back(new_samples);

            if queue.len() >= D {
                let out = queue.pop_front().unwrap();
                if let Err(e) = tx.try_send(out) {
                    if matches!(e, TrySendError::Disconnected(_)) {
                        println!("failed send, delay thread shutting down! {:?}", e);
                        return;
                    }
                }
            }
        }
    }
}

impl<const S: usize, const D: usize> Drop for Delay<S, D> {
    fn drop(&mut self) {
        self.shutdown
            .store(true, std::sync::atomic::Ordering::SeqCst);
        if let Some(hnd) = self.thread.take() {
            hnd.join().ok();
        }
    }
}
