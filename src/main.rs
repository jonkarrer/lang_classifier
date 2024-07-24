use std::sync::Arc;

use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::{Dataset, InMemDataset},
    },
    nn::attention::generate_padding_mask,
    prelude::Backend,
    tensor::{Bool, Data, ElementConversion, Int, Tensor},
};

use derive_new::new;
use serde::{Deserialize, Serialize};

/*
*
* Dataset
*
* This is a dataset for text classification.
*/

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PatentRecord {
    // Unique patent ID
    pub id: String,

    // The first phrase of the patent
    pub anchor: String,

    // The second phrase of the patent
    pub target: String,

    // CPC classification which indicates the subject within which the similarity is scored
    pub context: String,

    // The similarity score
    pub score: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone, new)]
pub struct ClassifiedPatentInput {
    pub text: String, // The text for classification
    pub label: f32,   // The label of the text (classification category)
}

pub struct PatentDataset {
    pub dataset: InMemDataset<ClassifiedPatentInput>,
}

impl PatentDataset {
    pub fn prep_training_set() -> Result<Self, std::io::Error> {
        let path = std::path::Path::new("dataset/train.csv");
        let mut reader = csv::ReaderBuilder::new().from_path(path)?;

        let rows = reader.deserialize();
        let mut classified_data = Vec::new();

        for r in rows {
            let record: PatentRecord = r?;
            let text = format!(
                "TEXT1: {}; TEXT2: {}; ANC1: {};",
                record.context, record.target, record.anchor
            );

            classified_data.push(ClassifiedPatentInput {
                text,
                label: record.score,
            })
        }

        let dataset = InMemDataset::new(classified_data);
        Ok(Self { dataset })
    }

    pub fn prep_test_set() -> Result<Self, std::io::Error> {
        let path = std::path::Path::new("dataset/test.csv");
        let reader = csv::ReaderBuilder::new();

        let dataset = InMemDataset::from_csv(path, &reader)?;
        Ok(Self { dataset })
    }

    // Returns the number of unique classes in the dataset
    pub fn num_classes() -> usize {
        5
    }

    // Returns the name of the class given its label
    pub fn class_name(label: f32) -> String {
        match label {
            0.0 => "Unrelated",
            0.25 => "Somewhat Related",
            0.5 => "Different Meaning Synonym",
            0.75 => "Close Synonym",
            1.0 => "Very Close Match",
            _ => panic!("Invalid label"),
        }
        .to_string()
    }
}

impl Dataset<ClassifiedPatentInput> for PatentDataset {
    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn get(&self, index: usize) -> Option<ClassifiedPatentInput> {
        self.dataset.get(index)
    }
}

/*
*
* Tokenizer
*
* Tokenizing the text
*/

struct CustomTokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl Default for CustomTokenizer {
    fn default() -> Self {
        Self {
            tokenizer: tokenizers::Tokenizer::from_pretrained("bert-base-cased", None).unwrap(),
        }
    }
}

// Implementation of the Tokenizer trait for BertCasedTokenizer.
impl CustomTokenizer {
    // Convert a text string into a sequence of tokens using the BERT cased tokenization strategy.
    fn encode(&self, value: &str) -> Vec<usize> {
        let tokens = self.tokenizer.encode(value, true).unwrap();
        tokens.get_ids().iter().map(|t| *t as usize).collect()
    }

    // Converts a sequence of tokens back into a text string.
    fn decode(&self, tokens: &[usize]) -> String {
        let tokens = tokens.iter().map(|t| *t as u32).collect::<Vec<u32>>();
        self.tokenizer.decode(&tokens, false).unwrap()
    }

    // Gets the size of the BERT cased tokenizer's vocabulary.
    fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    // Gets the token used for padding sequences to a consistent length.
    fn pad_token(&self) -> usize {
        self.tokenizer.token_to_id("[PAD]").unwrap() as usize
    }
}

/*
*
* Batcher
*
* Batch the data before training
*/

#[derive(Clone, new)]
pub struct ClassificationBatcher<B: Backend> {
    tokenizer: Arc<CustomTokenizer>, // Tokenizer for converting text to token IDs
    device: B::Device, // Device on which to perform computation (e.g., CPU or CUDA device)
    max_seq_length: usize, // Maximum sequence length for tokenized text
}

pub struct TrainingBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,    // Tokenized text
    pub labels: Tensor<B, 1, Int>,    // Labels of the text
    pub mask_pad: Tensor<B, 2, Bool>, // Padding mask for the tokenized text
}

pub struct InferenceBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,    // Tokenized text
    pub mask_pad: Tensor<B, 2, Bool>, // Padding mask for the tokenized text
}

impl<B: Backend> Batcher<ClassifiedPatentInput, TrainingBatch<B>> for ClassificationBatcher<B> {
    fn batch(&self, items: Vec<ClassifiedPatentInput>) -> TrainingBatch<B> {
        let mut tokens = Vec::new();
        let mut labels = Vec::new();

        for item in items {
            tokens.push(self.tokenizer.encode(&item.text));
            labels.push(Tensor::from_data(
                Data::from([(item.label as i64).elem::<B::IntElem>()]),
                &self.device,
            ));
        }

        // Generate padding mask for tokenized text
        let mask = generate_padding_mask(
            self.tokenizer.pad_token(),
            tokens,
            Some(self.max_seq_length),
            &self.device,
        );

        // Create and return training batch
        TrainingBatch {
            tokens: mask.tensor,
            labels: Tensor::cat(labels, 0),
            mask_pad: mask.mask,
        }
    }
}

impl<B: Backend> Batcher<String, InferenceBatch<B>> for ClassificationBatcher<B> {
    fn batch(&self, items: Vec<String>) -> InferenceBatch<B> {
        let mut tokens = Vec::with_capacity(items.len());

        // Tokenize each string
        for item in items {
            tokens.push(self.tokenizer.encode(&item));
        }

        // Generate padding mask for tokenized text
        let mask = generate_padding_mask(
            self.tokenizer.pad_token(),
            tokens,
            Some(self.max_seq_length),
            &B::Device::default(),
        );

        // Create and return inference batch
        InferenceBatch {
            tokens: mask.tensor.to_device(&self.device),
            mask_pad: mask.mask.to_device(&self.device),
        }
    }
}

fn main() {
    let dataset = PatentDataset::prep_training_set().unwrap().get(0).unwrap();
    dbg!(dataset);
}
