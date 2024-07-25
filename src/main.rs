mod data_prep;
mod inference;
mod model;
mod training;

use crate::data_prep::gather::ClassifiedDataset;
use burn::{
    data::dataset::Dataset,
    nn::transformer::TransformerEncoderConfig,
    optim::{decay::WeightDecayConfig, AdamConfig},
    tensor::backend::AutodiffBackend,
};
use training::ExperimentConfig;

type ElemType = f32;

fn main() {
    wgpu::run();
}

pub fn launch<B: AutodiffBackend>(devices: Vec<B::Device>) {
    let config = ExperimentConfig::new(
        TransformerEncoderConfig::new(256, 1024, 8, 4)
            .with_norm_first(true)
            .with_quiet_softmax(true),
        AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5))),
        1000,
        5,
        128,
    );

    training::train::<B>(
        devices,
        ClassifiedDataset::training_set().unwrap(),
        ClassifiedDataset::test_set().unwrap(),
        config,
        "/tmp/text-classification-ag-news",
    )
    .unwrap();
}

mod wgpu {
    use crate::{launch, ElemType};
    use burn::backend::{
        wgpu::{Wgpu, WgpuDevice},
        Autodiff,
    };

    pub fn run() {
        launch::<Autodiff<Wgpu>>(vec![WgpuDevice::default()]);
    }
}
