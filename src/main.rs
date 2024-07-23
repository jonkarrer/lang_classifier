use burn::data::dataset::{Dataset, InMemDataset};
use serde::{Deserialize, Serialize};

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

pub struct PatentDataset {
    pub dataset: InMemDataset<PatentRecord>,
}

impl PatentDataset {
    pub fn new() -> Result<Self, std::io::Error> {
        // Fetch csv data from filesystem
        let path = std::path::Path::new("dataset/train.csv");
        let reader = csv::ReaderBuilder::new();

        let dataset = InMemDataset::from_csv(path, &reader)?;

        Ok(Self { dataset })
    }
}

impl Dataset<PatentRecord> for PatentDataset {
    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn get(&self, index: usize) -> Option<PatentRecord> {
        self.dataset.get(index)
    }
}

fn main() {
    let dataset = PatentDataset::new().unwrap();
    dbg!(dataset.get(1));
}
