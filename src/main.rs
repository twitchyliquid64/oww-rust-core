use clap::Parser;
use std::fs::File;
use std::io::BufReader;

use oww_rust_core::*;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// scale samples by this amount
    #[arg(short, long)]
    preamp: Option<f32>,

    /// how many spectograms to buffer before computing the embedding
    #[arg(short, long, default_value_t = 4)]
    step_interval: usize,

    /// yaml-formatted config file
    config_file: String,
}

fn main() {
    let args = Args::parse();
    let reader = BufReader::new(File::open(args.config_file).expect("failed opening config file"));
    let config: Config = serde_yaml::from_reader(reader).unwrap();

    let mut sampler = Sampler::start(args.preamp).expect("failed to start listening for samples");
    let mut specter = Specter::start(sampler.take_receiver().unwrap())
        .expect("failed to start processing samples");
    let mut embedder = Embedder::start(specter.take_receiver().unwrap(), args.step_interval)
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

    let recv = runner.take_receiver().unwrap();
    loop {
        let results = recv.recv().unwrap();
        matcher.eval(results);
    }
}
