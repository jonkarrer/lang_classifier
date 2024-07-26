use burn::{
    config::Config,
    module::Module,
    nn::{
        loss::CrossEntropyLossConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        Embedding, EmbeddingConfig, Linear, LinearConfig,
    },
    prelude::Backend,
    tensor::{activation::softmax, backend::AutodiffBackend, Tensor},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::data_prep::{InferenceBatch, TrainingBatch};

#[derive(Config)]
pub struct ModelBuilder {
    transformer: TransformerEncoderConfig,
    n_classes: usize,
    vocab_size: usize,
    max_seq_length: usize,
}

// Define the model structure
#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    transformer: TransformerEncoder<B>,
    embedding_token: Embedding<B>,
    embedding_pos: Embedding<B>,
    output: Linear<B>,
    n_classes: usize,
    max_seq_length: usize,
}

impl ModelBuilder {
    pub fn build<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let output = LinearConfig::new(self.transformer.d_model, self.n_classes).init(device);
        let transformer = self.transformer.init(device);
        let embedding_token =
            EmbeddingConfig::new(self.vocab_size, self.transformer.d_model).init(device);
        let embedding_pos =
            EmbeddingConfig::new(self.max_seq_length, self.transformer.d_model).init(device);

        Model {
            transformer,
            embedding_token,
            embedding_pos,
            output,
            n_classes: self.n_classes,
            max_seq_length: self.max_seq_length,
        }
    }
}

impl<B: Backend> Model<B> {
    // Forward pass for training
    pub fn forward(&self, item: TrainingBatch<B>) -> ClassificationOutput<B> {
        // Get batch and sequence length and device
        let [batch_size, seq_length] = item.tokens.dims();
        let device = &self.embedding_token.devices()[0];

        // Move tensors to device
        let tokens = item.tokens.to_device(device);
        let labels = item.labels.to_device(device);
        let mask_pad = item.mask_pad.to_device(device);

        // Calculate token and position embeddings, then combine them
        let index_positions = Tensor::arange(0..seq_length as i64, device)
            .reshape([1, seq_length])
            .repeat(0, batch_size);
        let embedding_positions = self.embedding_pos.forward(index_positions);
        let embedding_tokens = self.embedding_token.forward(tokens);
        let embedding = (embedding_positions + embedding_tokens) / 2;

        // Perform transformer encoding, calculate output and loss
        let encoded = self
            .transformer
            .forward(TransformerEncoderInput::new(embedding).mask_pad(mask_pad));
        let output = self.output.forward(encoded);

        let output_classification = output
            .slice([0..batch_size, 0..1])
            .reshape([batch_size, self.n_classes]);

        let loss = CrossEntropyLossConfig::new()
            .init(&output_classification.device())
            .forward(output_classification.clone(), labels.clone());

        ClassificationOutput {
            loss,
            output: output_classification,
            targets: labels,
        }
    }

    // Forward pass for inference
    /// Defines forward pass for inference
    pub fn infer(&self, item: InferenceBatch<B>) -> Tensor<B, 2> {
        // Get batch and sequence length, and the device
        let [batch_size, seq_length] = item.tokens.dims();
        let device = &self.embedding_token.devices()[0];

        // Move tensors to the correct device
        let tokens = item.tokens.to_device(device);
        let mask_pad = item.mask_pad.to_device(device);

        // Calculate token and position embeddings, and combine them
        let index_positions = Tensor::arange(0..seq_length as i64, device)
            .reshape([1, seq_length])
            .repeat(0, batch_size);
        let embedding_positions = self.embedding_pos.forward(index_positions);
        let embedding_tokens = self.embedding_token.forward(tokens);
        let embedding = (embedding_positions + embedding_tokens) / 2;

        // Perform transformer encoding, calculate output and apply softmax for prediction
        let encoded = self
            .transformer
            .forward(TransformerEncoderInput::new(embedding).mask_pad(mask_pad));
        let output = self.output.forward(encoded);
        let output = output
            .slice([0..batch_size, 0..1])
            .reshape([batch_size, self.n_classes]);

        softmax(output, 1)
    }
}

/// Define training step
impl<B: AutodiffBackend> TrainStep<TrainingBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: TrainingBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        // Run forward pass, calculate gradients and return them along with the output
        let item = self.forward(item);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }
}

/// Define validation step
impl<B: Backend> ValidStep<TrainingBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: TrainingBatch<B>) -> ClassificationOutput<B> {
        // Run forward pass and return the output
        self.forward(item)
    }
}
