use oww_rust_core::*;

fn main() {
    let mut sampler = Sampler::start().expect("failed to start listening for samples");
    let mut specter = Specter::start(sampler.take_receiver().unwrap())
        .expect("failed to start processing samples");
    let mut embedder = Embedder::start(specter.take_receiver().unwrap())
        .expect("failed to start processing spectos");

    let mut runner = Runner::start(embedder.take_receiver().unwrap())
        .expect("failed to start processing embeddings");

    runner.add_model(NamedModel::new("rasppy", "models/hey_rhasspy_v0.1.onnx").unwrap());
    runner.add_model(NamedModel::new("HA", "models/Home_assistant.onnx").unwrap());
    runner.add_model(NamedModel::new("freeze", "models/freeze_all_motor_function.onnx").unwrap());
    runner.add_model(NamedModel::new("dnc", "models/die_no_core.onnx").unwrap());
    runner.add_model(NamedModel::new("shutdown", "models/shut_down.onnx").unwrap());

    let recv = runner.take_receiver().unwrap();
    loop {
        let results = recv.recv().unwrap();
        println!("got results! {:?}", results);
    }
}
