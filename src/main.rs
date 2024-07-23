use burn::data::dataset::{Dataset, InMemDataset};
use burn::{data::dataloader::batcher::Batcher, nn::attention::generate_padding_mask, prelude::*};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokenizers::Tokenizer;
/*
*
* Dataset
*
* This is a dataset for text classification.
*/

// Define a struct for text classification items
#[derive(Clone, Debug)]
pub struct TextClassificationItem {
    pub text: String, // The text for classification
    pub label: f32,   // The label of the text (classification category)
}

impl TextClassificationItem {
    pub fn new(text: String, label: f32) -> Self {
        Self { text, label }
    }
}

// Trait for text classification datasets
pub trait TextClassificationDataset: Dataset<TextClassificationItem> {
    fn num_classes() -> usize; // Returns the number of unique classes in the dataset
    fn class_name(label: f32) -> String; // Returns the name of the class given its label
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PatentRecord {
    // Unique patent ID
    // #[serde(rename = "ID")]
    pub id: String,

    // The first phrase of the patent
    // #[serde(rename = "ANCHOR")]
    pub anchor: String,

    // The second phrase of the patent
    // #[serde(rename = "TARGET")]
    pub target: String,

    // CPC classification which indicates the subject within which the similarity is scored
    // #[serde(rename = "CONTEXT")]
    pub context: String,

    // The similarity score
    // #[serde(rename = "SCORE")]
    pub score: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TokenizedPatentRecord {
    pub text: String, // The text for classification
    pub label: f32,   // The label of the text (classification category)
}

pub struct PatentDataset {
    pub dataset: InMemDataset<TokenizedPatentRecord>,
}

impl PatentDataset {
    pub fn train() -> Result<Self, std::io::Error> {
        let path = std::path::Path::new("dataset/train.csv");
        let reader = csv::ReaderBuilder::new();

        let dataset = InMemDataset::from_csv(path, &reader)?;
        Ok(Self { dataset })
    }

    pub fn test() -> Result<Self, std::io::Error> {
        let path = std::path::Path::new("dataset/test.csv");
        let reader = csv::ReaderBuilder::new();

        let dataset = InMemDataset::from_csv(path, &reader)?;
        Ok(Self { dataset })
    }
}

impl Dataset<TextClassificationItem> for PatentDataset {
    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn get(&self, index: usize) -> Option<TextClassificationItem> {
        self.dataset
            .get(index)
            .map(|item| TextClassificationItem::new(item.text, item.label))
    }
}

impl TextClassificationDataset for PatentDataset {
    fn num_classes() -> usize {
        5
    }

    fn class_name(label: f32) -> String {
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

/*
*
* Tokenizer
*
* Tokenizing the text
*/

pub trait PatentTokenizer: Send + Sync {
    /// Converts a text string into a sequence of tokens.
    fn encode(&self, value: &str) -> Vec<usize>;

    /// Converts a sequence of tokens back into a text string.
    fn decode(&self, tokens: &[usize]) -> String;

    /// Gets the size of the tokenizer's vocabulary.
    fn vocab_size(&self) -> usize;

    /// Gets the token used for padding sequences to a consistent length.
    fn pad_token(&self) -> usize;

    /// Gets the string representation of the padding token.
    /// The default implementation uses `decode` on the padding token.
    fn pad_token_value(&self) -> String {
        self.decode(&[self.pad_token()])
    }
}

// Struct represents a specific tokenizer, in this case a BERT cased tokenization strategy.
pub struct BertCasedTokenizer {
    // The underlying tokenizer from the `tokenizers` library.
    tokenizer: tokenizers::Tokenizer,
}

// Default implementation for creating a new BertCasedTokenizer.
// This uses a pretrained BERT cased tokenizer model.
impl Default for BertCasedTokenizer {
    fn default() -> Self {
        Self {
            tokenizer: tokenizers::Tokenizer::from_pretrained("bert-base-cased", None).unwrap(),
        }
    }
}

// Implementation of the Tokenizer trait for BertCasedTokenizer.
impl PatentTokenizer for BertCasedTokenizer {
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
* Batching the data
*/

// Struct for batching text classification items
#[derive(Clone, derive_new::new)]
pub struct TextClassificationBatcher<B: Backend> {
    tokenizer: Arc<dyn PatentTokenizer>, // Tokenizer for converting text to token IDs
    device: B::Device, // Device on which to perform computation (e.g., CPU or CUDA device)
    max_seq_length: usize, // Maximum sequence length for tokenized text
}

#[derive(Debug, Clone, derive_new::new)]
pub struct TextClassificationTrainingBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,    // Tokenized text
    pub labels: Tensor<B, 1, Int>,    // Labels of the text
    pub mask_pad: Tensor<B, 2, Bool>, // Padding mask for the tokenized text
}

#[derive(Debug, Clone, derive_new::new)]
pub struct TextClassificationInferenceBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,    // Tokenized text
    pub mask_pad: Tensor<B, 2, Bool>, // Padding mask for the tokenized text
}

/// Implement Batcher trait for TextClassificationBatcher struct for training
impl<B: Backend> Batcher<TextClassificationItem, TextClassificationTrainingBatch<B>>
    for TextClassificationBatcher<B>
{
    /// Batches a vector of text classification items into a training batch
    fn batch(&self, items: Vec<TextClassificationItem>) -> TextClassificationTrainingBatch<B> {
        let mut tokens_list = Vec::with_capacity(items.len());
        let mut labels_list = Vec::with_capacity(items.len());

        // Tokenize text and create label tensor for each item
        for item in items {
            tokens_list.push(self.tokenizer.encode(&item.text));
            labels_list.push(Tensor::from_data(
                Data::from([(item.label as i64).elem::<B::IntElem>()]),
                &self.device,
            ));
        }

        // Generate padding mask for tokenized text
        let mask = generate_padding_mask(
            self.tokenizer.pad_token(),
            tokens_list,
            Some(self.max_seq_length),
            &self.device,
        );

        // Create and return training batch
        TextClassificationTrainingBatch {
            tokens: mask.tensor,
            labels: Tensor::cat(labels_list, 0),
            mask_pad: mask.mask,
        }
    }
}

/// Implement Batcher trait for TextClassificationBatcher struct for inference
impl<B: Backend> Batcher<String, TextClassificationInferenceBatch<B>>
    for TextClassificationBatcher<B>
{
    /// Batches a vector of strings into an inference batch
    fn batch(&self, items: Vec<String>) -> TextClassificationInferenceBatch<B> {
        let mut tokens_list: Vec<Vec<usize>> = Vec::with_capacity(items.len());

        // Tokenize each string
        for item in items {
            tokens_list.push(self.tokenizer.encode(&item));
        }

        // Generate padding mask for tokenized text
        let mask = generate_padding_mask(
            self.tokenizer.pad_token(),
            tokens_list,
            Some(self.max_seq_length),
            &B::Device::default(),
        );

        // Create and return inference batch
        TextClassificationInferenceBatch {
            tokens: mask.tensor.to_device(&self.device),
            mask_pad: mask.mask.to_device(&self.device),
        }
    }
}

fn main() {
    // let dataset = PatentDataset::new().unwrap();
    // dbg!(dataset.get(0));
}
