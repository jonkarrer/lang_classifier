use anyhow::Ok;
use burn::{
    data::dataloader::batcher::Batcher,
    prelude::*,
    record::{CompactRecorder, Recorder},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{
    data_prep::{BertCaseTokenizer, ClassificationBatcher, ClassifiedDataset},
    model::ModelBuilder,
    training::ExperimentConfig,
};

// Represents a single row in the csv dataset
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TestPatentRecord {
    // Unique patent ID
    pub id: String,

    // The first phrase of the patent
    pub anchor: String,

    // The second phrase of the patent
    pub target: String,

    // CPC classification which indicates the subject within which the similarity is scored
    pub context: String,
}

pub fn gather_test_set() -> anyhow::Result<Vec<String>> {
    let path = std::path::Path::new("dataset/test.csv");
    let mut reader = csv::ReaderBuilder::new().from_path(path).unwrap();

    let rows = reader.deserialize();
    let mut test_set = Vec::new();

    for r in rows {
        let record: TestPatentRecord = r.unwrap();
        let text = format!(
            "TEXT1: {}; TEXT2: {}; ANC1: {};",
            record.context, record.target, record.anchor
        );

        test_set.push(text);
    }

    Ok(test_set)
}

// Define inference function
pub fn infer<B: Backend>(
    device: B::Device, // Device on which to perform computation (e.g., CPU or CUDA device)
    artifact_dir: &str, // Directory containing model and config files
    samples: Vec<String>, // Text samples for inference
) {
    // Load experiment configuration
    let config = ExperimentConfig::load(format!("{artifact_dir}/config.json").as_str())
        .expect("Config file present");

    // Initialize tokenizer
    let tokenizer = Arc::new(BertCaseTokenizer::default());

    // Get number of classes from dataset
    let n_classes = ClassifiedDataset::num_classes();

    // Initialize batcher for batching samples
    let batcher = Arc::new(ClassificationBatcher::<B>::new(
        tokenizer.clone(),
        device.clone(),
        config.max_seq_length,
    ));

    // Load pre-trained model weights
    println!("Loading weights ...");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model weights");

    // Create model using loaded weights
    println!("Creating model ...");
    let model = ModelBuilder::new(
        config.transformer,
        n_classes,
        tokenizer.vocab_size(),
        config.max_seq_length,
    )
    .build(&device)
    .load_record(record); // Initialize model with loaded weights

    // Run inference on the given text samples
    println!("Running inference ...");
    let item = batcher.batch(samples.clone()); // Batch samples using the batcher
    let predictions = model.infer(item); // Get model predictions

    // Print out predictions for each sample
    for (i, text) in samples.into_iter().enumerate() {
        #[allow(clippy::single_range_in_vec_init)]
        let prediction = predictions.clone().slice([i..i + 1]); // Get prediction for current sample
        let logits = prediction.to_data(); // Convert prediction tensor to data
        let class_index = prediction.argmax(1).into_data().convert().value.as_slice()[0]; // Get class index with the highest value
        let class = ClassifiedDataset::class_name(class_index); // Get class name

        // Print sample text, predicted logits and predicted class
        println!(
            "\n=== Item {i} ===\n- Text: {text}\n- Logits: {logits}\n- Prediction: \
             {class}\n================"
        );
    }
}
