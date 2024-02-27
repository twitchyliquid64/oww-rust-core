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
    // runner.add_model(NamedModel::new("freeze", "models/freeze_all_motor_function.onnx").unwrap());
    // runner.add_model(NamedModel::new("dnc", "models/die_no_core.onnx").unwrap());
    // runner.add_model(NamedModel::new("shutdown", "models/shut_down.onnx").unwrap());
    // runner.add_model(NamedModel::new("preset", "models/pre_set.onnx").unwrap());
    runner.add_model(NamedModel::new("turn off", "models/turn_off.onnx").unwrap());
    runner.add_model(NamedModel::new("lumos", "models/lume_moss.onnx").unwrap());
    // runner.add_model(NamedModel::new("living room", "models/living_room.onnx").unwrap());
    // runner.add_model(NamedModel::new("front room", "models/front_room.onnx").unwrap());
    // runner.add_model(NamedModel::new("bed room", "models/bed_room.onnx").unwrap());
    // runner.add_model(NamedModel::new("daytime", "models/daytime.onnx").unwrap());
    // runner.add_model(NamedModel::new("nighttime", "models/nighttime.onnx").unwrap());

    let recv = runner.take_receiver().unwrap();
    loop {
        let results = recv.recv().unwrap();
        println!("{:?}", results);
    }
}
