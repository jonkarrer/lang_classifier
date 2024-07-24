#[derive(Clone)]
pub struct BertCaseTokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl Default for BertCaseTokenizer {
    fn default() -> Self {
        Self {
            tokenizer: tokenizers::Tokenizer::from_pretrained("bert-base-cased", None).unwrap(),
        }
    }
}

// Implementation of the Tokenizer trait for BertCasedTokenizer.
impl BertCaseTokenizer {
    // Convert a text string into a sequence of tokens using the BERT cased tokenization strategy.
    pub fn encode(&self, value: &str) -> Vec<usize> {
        let tokens = self.tokenizer.encode(value, true).unwrap();
        tokens.get_ids().iter().map(|t| *t as usize).collect()
    }

    // Converts a sequence of tokens back into a text string.
    pub fn decode(&self, tokens: &[usize]) -> String {
        let tokens = tokens.iter().map(|t| *t as u32).collect::<Vec<u32>>();
        self.tokenizer.decode(&tokens, false).unwrap()
    }

    // Gets the size of the BERT cased tokenizer's vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    // Gets the token used for padding sequences to a consistent length.
    pub fn pad_token(&self) -> usize {
        self.tokenizer.token_to_id("[PAD]").unwrap() as usize
    }
}
