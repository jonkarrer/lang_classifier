use std::sync::Arc;

use burn::{
    nn::transformer::{TransformerEncoder, TransformerEncoderConfig},
    optim::AdamConfig,
    tensor::backend::AutodiffBackend,
};

use crate::data_prep::{
    batch::ClassificationBatcher, gather::ClassifiedDataset, tokenizer::BertCaseTokenizer,
};

pub struct ExperimentConfig {
    pub transformer: TransformerEncoderConfig,
    pub optimizer: AdamConfig,
    pub batch_size: usize,
    pub num_epochs: usize,
    pub max_seq_length: usize,
}

pub fn train<B: AutodiffBackend>(
    devices: Vec<B::Device>,
    training_set: ClassifiedDataset,
    test_set: ClassifiedDataset,
    config: ExperimentConfig,
    artifact_dir: &str,
) {
    let tokenizer = Arc::new(BertCaseTokenizer::default());

    let batcher_train = ClassificationBatcher::<B>::new(
        tokenizer.clone(),
        devices[0].clone(),
        config.max_seq_length,
    );

    let batcher_test = ClassificationBatcher::<B::InnerBackend>::new(
        tokenizer.clone(),
        devices[0].clone(),
        config.max_seq_length,
    );

    todo!()
}
