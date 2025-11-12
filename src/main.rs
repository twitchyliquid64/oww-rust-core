use clap::Parser;
use std::env::temp_dir;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::mpsc::RecvTimeoutError;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use oww_rust_core::*;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// scale samples by this amount
    #[arg(short, long)]
    preamp: Option<f32>,
    /// input microphone to listen on
    #[arg(short, long)]
    device: Option<String>,

    /// yaml-formatted config file
    config_file: String,
}

fn main() {
    let args = Args::parse();
    let reader = BufReader::new(File::open(args.config_file).expect("failed opening config file"));
    let config: Config = serde_yaml::from_reader(reader).unwrap();

    // Sample from microphone in 640-sample chunks, split into two streams
    let mut sampler = Sampler::<640>::start(args.preamp, args.device)
        .expect("failed to start listening for samples");
    let mut tee = Tee::<640, 3>::start(sampler.take_receiver().unwrap()).unwrap();

    // VAD pipeline: rechunk to 480-sample chunks, run through VAD, record timestamp of last activity
    let mut vad_rechunker = Rechunker::<640, 480>::start(tee.take_receiver(0).unwrap()).unwrap();
    let mut vad = VAD::start(vad_rechunker.take_receiver().unwrap()).unwrap();
    let vad_recv = vad.take_receiver().unwrap();
    let last_activity = Arc::new(AtomicU64::new(500));
    let la = last_activity.clone();
    thread::spawn(move || {
        loop {
            if vad_recv.recv().unwrap() {
                la.store(
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    std::sync::atomic::Ordering::SeqCst,
                );
            }
        }
    });

    // Record pipeline
    let mut record_rechunker =
        Rechunker::<640, 4000>::start(tee.take_receiver(1).unwrap()).unwrap();
    let mut record_delay =
        Delay::<4000, 4>::start(record_rechunker.take_receiver().unwrap()).unwrap();

    // Wakeword pipeline
    let mut specter_rechunker =
        Rechunker::<640, SPECTOGRAM_SAMPLES>::start(tee.take_receiver(2).unwrap()).unwrap();
    let mut specter = Specter::start(specter_rechunker.take_receiver().unwrap())
        .expect("failed to start processing samples");
    let mut embedder = Embedder::start(specter.take_receiver().unwrap(), 4)
        .expect("failed to start processing spectos");

    let mut runner = Runner::start(embedder.take_receiver().unwrap())
        .expect("failed to start processing embeddings");

    for (name, params) in config.models {
        let m = NamedModel::new(name, params.path, params.scale.unwrap_or(1.)).unwrap();
        runner.add_model(m);
    }

    let mut matcher = Matcher::new();
    for (name, params) in config.matchers {
        matcher.add_rule(name, params);
    }

    let rec = record_delay.take_receiver().unwrap();
    let recv = runner.take_receiver().unwrap();
    let mut recording: Option<Vec<f32>> = None;
    loop {
        match recv.recv_timeout(Duration::from_millis(1)) {
            Ok(results) => {
                // Not recording, lets see if recording is triggered.
                if let Some(wakeword) = &config.utterance.wakeword
                    && recording.is_none()
                {
                    if let Some((_, score)) = results.iter().find(|(name, _score)| name == wakeword)
                        && *score > 0.6
                    {
                        // Recording has been triggered
                        recording = Some(Vec::with_capacity(16_000 * 32));
                    } else {
                        // Drop recording samples.
                        rec.try_iter().for_each(|_| ());
                    }
                }
                matcher.eval(results);
            }
            Err(e) => {
                if e == RecvTimeoutError::Disconnected {
                    panic!("{:?}", e);
                }
            }
        }

        match &mut recording {
            Some(r) => {
                // Recording is in progress, lets:
                //  - Add fresh samples from the recording pipe
                //  - See if VAD has been inactive for long enough to terminate
                if last_activity.fetch_add(0, std::sync::atomic::Ordering::SeqCst)
                    < SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs()
                        - 2
                {
                    let fname = format!(
                        "{}/utterance_{}.wav",
                        temp_dir().as_os_str().to_str().unwrap(),
                        current_time_string()
                    );
                    wavers::write(&fname, r, 16_000, 1).unwrap();
                    recording = None;

                    // Run the utterance command if any.
                    if let Some(cmd) = &config.utterance.exec {
                        let mut cmd = shlex::Shlex::new(cmd);
                        use std::process::Command;

                        let mut c = Command::new(cmd.next().unwrap());
                        let cmd = c
                            .current_dir(std::env::current_dir().unwrap())
                            .args([fname].into_iter().chain(cmd));
                        println!("spawning: {:?}", &cmd);
                        println!("result: {:?}", cmd.spawn());
                    }

                    // Drop recording samples.
                    rec.try_iter().for_each(|_| ());
                } else {
                    while let Ok(chunk) = rec.try_recv() {
                        r.extend_from_slice(&chunk.samples);
                    }
                }
            }
            None => {}
        }
    }
}

fn current_time_string() -> String {
    use chrono::{Datelike, Timelike};
    let now = chrono::Local::now();
    format!(
        "{:04}{:02}{:02}{:02}{:02}{:02}",
        now.year(),
        now.month(),
        now.day(),
        now.hour(),
        now.minute(),
        now.second()
    )
}
