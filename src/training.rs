use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::transform::SamplerDataset},
    lr_scheduler::noam::NoamLrSchedulerConfig,
    nn::transformer::TransformerEncoderConfig,
    optim::AdamConfig,
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, CudaMetric, LearningRateMetric, LossMetric},
        LearnerBuilder,
    },
};
use std::sync::Arc;

use crate::{
    data_prep::{BertCaseTokenizer, ClassificationBatcher, ClassifiedDataset},
    model::{Model, ModelBuilder},
};

#[derive(Config)]
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
) -> anyhow::Result<()> {
    // Initialize the tokenizer
    let tokenizer = Arc::new(BertCaseTokenizer::default());

    // Initialize the batchers
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

    // Initialize the model
    let model: Model<B> = ModelBuilder::new(
        config.transformer.clone(),
        ClassifiedDataset::num_classes(),
        tokenizer.vocab_size(),
        config.max_seq_length,
    )
    .build(&devices[0]);

    // Initialize the dataloaders
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .num_workers(1)
        .build(SamplerDataset::new(training_set, 30_000));

    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .num_workers(1)
        .build(SamplerDataset::new(test_set, 5_000));

    // Initialize the optimizer
    let optimizer = config.optimizer.init();

    // Initialize learning rate
    let lr = NoamLrSchedulerConfig::new(1e-2)
        .with_warmup_steps(1000)
        .with_model_size(config.transformer.d_model)
        .init();

    // Initialize learner
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train(CudaMetric::new())
        .metric_valid(CudaMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(devices)
        .num_epochs(config.num_epochs)
        .summary()
        .build(model, optimizer, lr);

    // Train the model
    let model_trained = learner.fit(dataloader_train, dataloader_test);

    // Save the configuration and the trained model
    config.save(format!("{artifact_dir}/config.json"))?;
    CompactRecorder::new().record(
        model_trained.into_record(),
        format!("{artifact_dir}/model").into(),
    )?;

    Ok(())
}
