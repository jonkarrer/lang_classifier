mod data_prep;
mod inference;
mod model;
mod training;

use crate::data_prep::ClassifiedDataset;
use burn::{
    backend::{
        wgpu::{AutoGraphicsApi, WgpuDevice},
        Autodiff, Wgpu,
    },
    nn::transformer::TransformerEncoderConfig,
    optim::{decay::WeightDecayConfig, AdamConfig},
};
use data_prep::PatentRecord;
use inference::gather_test_set;
use training::ExperimentConfig;

fn main() {
    start_inference_experiment();
}

fn start_training_experiment() {
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutoDiffBackend = Autodiff<MyBackend>;

    let devices = vec![WgpuDevice::default()];

    let config = ExperimentConfig::new(
        TransformerEncoderConfig::new(256, 1024, 8, 4)
            .with_norm_first(true)
            .with_quiet_softmax(true),
        AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5))),
        1000,
        5,
        128,
    );

    // training::train::<MyAutoDiffBackend>(
    //     devices,
    //     ClassifiedDataset::training_set().unwrap(),
    //     ClassifiedDataset::test_set().unwrap(),
    //     config,
    //     "/tmp/text-classification-patent",
    // )
    // .unwrap();
}

fn start_inference_experiment() {
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutoDiffBackend = Autodiff<MyBackend>;

    inference::infer::<MyAutoDiffBackend>(
        WgpuDevice::default(),
        "/Users/jkarrer/devjon/rust/lang_classifier/text-classification-patent",
        gather_test_set().unwrap(),
    );
}
