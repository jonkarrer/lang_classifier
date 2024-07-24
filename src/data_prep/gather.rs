use burn::data::dataset::{Dataset, InMemDataset};
use derive_new::new;
use serde::{Deserialize, Serialize};

// Represents a single row in the csv dataset
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

// This is the data that will be fed into the model after tokenization and batching
#[derive(Serialize, Deserialize, Debug, Clone, new)]
pub struct ClassifiedText {
    pub text: String, // The text for classification
    pub label: f32,   // The label of the text (classification category)
}

// Represents the entire dataset in memory, could also be a sqlite database
pub struct ClassifiedDataset {
    pub dataset: InMemDataset<ClassifiedText>,
}

impl ClassifiedDataset {
    // Gather the training set of data
    pub fn training_set() -> Result<Self, std::io::Error> {
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

            classified_data.push(ClassifiedText {
                text,
                label: record.score,
            })
        }

        let dataset = InMemDataset::new(classified_data);
        Ok(Self { dataset })
    }

    pub fn test_set() -> Result<Self, std::io::Error> {
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

impl Dataset<ClassifiedText> for ClassifiedDataset {
    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn get(&self, index: usize) -> Option<ClassifiedText> {
        self.dataset.get(index)
    }
}
