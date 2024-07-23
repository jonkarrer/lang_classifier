use burn::data::dataset::{Dataset, InMemDataset};
use serde::{Deserialize, Serialize};

/*
*
* Dataset
*
* This is a dataset for text classification.
*/

// Trait for text classification datasets

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
pub struct ClassifiedPatentEntry {
    pub text: String, // The text for classification
    pub label: f32,   // The label of the text (classification category)
}

pub trait ClassificationDataset: Dataset<ClassifiedPatentEntry> {
    fn num_classes() -> usize; // Returns the number of unique classes in the dataset
    fn class_name(label: f32) -> String; // Returns the name of the class given its label
}

pub struct PatentDataset {
    pub dataset: InMemDataset<ClassifiedPatentEntry>,
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

impl Dataset<ClassifiedPatentEntry> for PatentDataset {
    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn get(&self, index: usize) -> Option<ClassifiedPatentEntry> {
        self.dataset.get(index)
    }
}

impl ClassificationDataset for PatentDataset {
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

fn main() {
    // let dataset = PatentDataset::new().unwrap();
    // dbg!(dataset.get(0));
}
