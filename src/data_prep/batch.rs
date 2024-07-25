use derive_new::new;
use std::sync::Arc;

use burn::{
    data::dataloader::batcher::Batcher,
    nn::attention::generate_padding_mask,
    prelude::Backend,
    tensor::{Bool, Data, ElementConversion, Int, Tensor},
};

use super::{gather::ClassifiedText, tokenizer::BertCaseTokenizer};

#[derive(Clone, new)]
pub struct ClassificationBatcher<B: Backend> {
    tokenizer: Arc<BertCaseTokenizer>, // Tokenizer for converting text to token IDs
    device: B::Device, // Device on which to perform computation (e.g., CPU or CUDA device)
    max_seq_length: usize, // Maximum sequence length for tokenized text
}

#[derive(Clone, Debug)]
pub struct TrainingBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,    // Tokenized text
    pub labels: Tensor<B, 1, Int>,    // Labels of the text
    pub mask_pad: Tensor<B, 2, Bool>, // Padding mask for the tokenized text
}

#[derive(Clone, Debug)]
pub struct InferenceBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,    // Tokenized text
    pub mask_pad: Tensor<B, 2, Bool>, // Padding mask for the tokenized text
}

impl<B: Backend> Batcher<ClassifiedText, TrainingBatch<B>> for ClassificationBatcher<B> {
    fn batch(&self, items: Vec<ClassifiedText>) -> TrainingBatch<B> {
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

        dbg!(&mask.tensor);
        dbg!(&mask.mask);

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
