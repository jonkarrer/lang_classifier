# NLP Classifier Introduction

U.S Patent classifier challenge

[Textbook](https://colab.research.google.com/github/fastai/fastbook/blob/master/10_nlp.ipynb)
[Notebook](https://www.kaggle.com/code/jhoward/getting-started-with-nlp-for-absolute-beginners)

By following these steps, you'll have a robust process for converting raw house features into a structured data object suitable for feeding into machine learning models, ensuring consistency and accuracy in your predictions.

## Overview

We want to train a machine learning model to predict the score for a patent entry. This score is based on the similarity between the phrases in the patent.

### Data

[US Patent Classifier](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/data)

### Model

We will roll our own model for this task.

### Main Libraries

[Burn](https://github.com/tracel-ai/burn/tree/main)
[Tokenizers](https://github.com/huggingface/tokenizers)
[Csv](https://crates.io/crates/csv)

### General Steps

1. Prep data: Convert the dataset into a format that is suitable for training. In this case it is already in CSV form.

2. Tokenization: Convert the text into a list of words (or characters, or substrings, depending on the granularity of your model)

3. Numericalization: Make a list of all of the unique words that appear (the vocab), and convert each word into a number, by looking up its index in the vocab

4. Language model data loader creation: Creating a dependent variable that is offset from the independent variable by one token. Shuffle the training data in such a way that the dependent and independent variables maintain their structure as required.

5. Training:: Train the model on the training data.

## Data Prep

First we need to gather and split the data into training and validation sets. Using the Csv library, we can get each row and tokenize them all in one step. The score is our label for each row.

```rust
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
```

Now we just need to load this data into memory or a sqlite database. We will use the Burn crates `InMemDataset` for this.

```rust
    let dataset = InMemDataset::new(classified_data);
```

Burn puts inference as a pillar in their core values, so as we build the training pipeline, we will also build the inference pipeline. The first step here is assigning labels to human friendly strings.

```rust
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
```

## Batching

Batching is the process of creating batches of data that can be fed into the model. This helps the GPU to process the data in a more efficient manner. During the batching, we will tokenize the text into a list of tokens. We also generate padding masks for each batch. Padding masks are used to pad the tokens to the same length.

We will set the exact size of each batch later in the pipeline, but this is how we can define what a single batch consists of.

```rust
    let mut tokens = Vec::new();
    let mut labels = Vec::new();

    for item in items {
        tokens.push(self.tokenizer.encode(&item.text)); // Tokenize each string
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

    TrainingBatch {
        tokens: mask.tensor,
        labels: Tensor::cat(labels, 0),
        mask_pad: mask.mask,
    }
```

### Model Making

#### Define our model

We could fine tune a pertained model, like bert, but we will roll our own model for fun.

1. Define our neural net architecture: Transformers are a popular architecture for NLP.

2. Create our embedding tokens and embedding position layers: These are used to give context and positional information to the model.

3. Create our output layer: The linear layer that will classify our data.

```rust
pub fn build<B: Backend>(&self, device: &B::Device) -> Model<B> {
    let transformer = self.transformer.init(device);
    let embedding_token =
        EmbeddingConfig::new(self.vocab_size, self.transformer.d_model).init(device);
    let embedding_pos =
        EmbeddingConfig::new(self.max_seq_length, self.transformer.d_model).init(device);
    let output = LinearConfig::new(self.transformer.d_model, self.n_classes).init(device);

    Model {
        transformer,
        embedding_token,
        embedding_pos,
        output,
        n_classes: self.n_classes,
        max_seq_length: self.max_seq_length,
    }
}
```

#### Define our model forward pass
