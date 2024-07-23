use burn::data::dataset::{Dataset, InMemDataset};
use serde::{Deserialize, Serialize};

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

fn main() {
    // let dataset = PatentDataset::new().unwrap();
    // dbg!(dataset.get(0));
}
