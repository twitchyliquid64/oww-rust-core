use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::Arc;
use std::thread;

use crate::Chunk;

/// Tee takes chunks and writes them to N different streams.
pub struct Tee<const S: usize, const N: usize> {
    recv: [Option<Receiver<Chunk<S>>>; N],
    shutdown: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl<const S: usize, const N: usize> Tee<S, N> {
    pub fn start(samples: Receiver<Chunk<S>>) -> Result<Self, anyhow::Error> {
        let mut recvs: [Option<Receiver<Chunk<S>>>; N] = [const { None }; N];
        let mut sends = [const { None }; N];
        for i in 0..N {
            let (send, recv) = sync_channel(1);
            sends[i] = Some(send);
            recvs[i] = Some(recv);
        }
        let shutdown = Arc::new(AtomicBool::new(false));

        let shutdown2 = shutdown.clone();
        let thread = Some(thread::spawn(move || {
            Tee::mainloop(sends, shutdown2, samples);
        }));

        let out = Self {
            shutdown,
            thread,
            recv: recvs,
        };

        Ok(out)
    }

    pub fn take_receiver(&mut self, idx: usize) -> Option<Receiver<Chunk<S>>> {
        self.recv[idx].take()
    }

    fn mainloop(
        mut txs: [Option<SyncSender<Chunk<S>>>; N],
        shutdown: Arc<AtomicBool>,
        samples: Receiver<Chunk<S>>,
    ) {
        loop {
            if shutdown.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }
            let chunk = match samples.recv() {
                Ok(s) => s,
                Err(_e) => return,
            };

            if shutdown.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }
            for tx in txs.iter_mut() {
                if let Err(e) = tx.as_mut().unwrap().send(chunk.clone()) {
                    println!("failed send, tee thread shutting down! {:?}", e);
                    return;
                }
            }
        }
    }
}

impl<const S: usize, const N: usize> Drop for Tee<S, N> {
    fn drop(&mut self) {
        self.shutdown
            .store(true, std::sync::atomic::Ordering::SeqCst);
        if let Some(hnd) = self.thread.take() {
            hnd.join().ok();
        }
    }
}
