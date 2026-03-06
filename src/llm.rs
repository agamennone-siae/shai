use anyhow::{Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use serde::{Deserialize, Serialize};
use std::num::NonZeroU32;
use std::path::PathBuf;

use llama_cpp_2::model::AddBos;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;

use crate::rag;

pub struct Engine {
    backend: LlamaBackend,
    model: LlamaModel,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct LlmResponse {
    pub is_command: bool,
    pub text: String,
}

impl Engine {
    pub fn new(model_path: &PathBuf) -> Result<Self> {
        let backend = LlamaBackend::init()?;
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)?;

        Ok(Self { backend, model })
    }

    pub async fn generate_response(&self, prompt: &str) -> Result<LlmResponse> {
        // Build the system prompt indicating JSON output is expected
        // with intent classification.
        let system_prompt = r#"You are shai, an AI CLI reasoning engine.
If the user wants a CLI command to be executed, return a JSON object: 
{ "is_command": true, "text": "<the bash command directly without backticks>" }
If they're just chatting or asking for info, return:
{ "is_command": false, "text": "<the text response>" }

Only return valid JSON, no markdown formatting."#;

        // Try to identify a command from the user request and fetch RAG context
        let context_str = if let Some(cmd) = rag::guess_command(prompt) {
            rag::fetch_documentation(&cmd).unwrap_or_default()
        } else {
            "".to_string()
        };

        let full_prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\nContext: {}\n\nRequest: {}\n<|im_end|>\n<|im_start|>assistant\n",
            system_prompt, context_str, prompt
        );

        // Setting up context and batch
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(4096).unwrap()));

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .context("failed to create context")?;

        let tokens = self
            .model
            .str_to_token(&full_prompt, AddBos::Always)
            .context("failed to tokenize")?;

        let mut batch = LlamaBatch::new(2048, 1);
        let last_index = tokens.len() - 1;
        for (i, token) in (0_i32..).zip(tokens.iter().copied()) {
            let is_last = i as usize == last_index;
            batch.add(token, i, &[0], is_last)?;
        }

        ctx.decode(&mut batch)?;

        let mut generated_text = String::new();
        let mut n_cur = batch.n_tokens();

        // Very basic sampler loop
        let mut n_decoded = 0;
        let max_tokens = 500;
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        // Prevent repetitive output strings by applying a repeat penalty
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::penalties(64, 1.1, 0.0, 0.0),
            LlamaSampler::greedy(),
        ]);
        
        // Feed the prompt tokens into the sampler so it knows what to penalize
        sampler.accept_many(tokens.iter().copied());

        while n_decoded < max_tokens {
            let next_token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(next_token);

            if self.model.is_eog_token(next_token) {
                break;
            }

            let token_str = self
                .model
                .token_to_piece(next_token, &mut decoder, false, None)?;
            generated_text.push_str(&token_str);

            batch.clear();
            batch.add(next_token, n_cur, &[0], true)?;
            n_cur += 1;
            ctx.decode(&mut batch)?;
            n_decoded += 1;
        }

        // Parse resulting JSON loosely
        let mut text = generated_text.trim();
        
        // Strip out reasoning tokens from models like DeepSeek-R1 or Qwen distilled variants
        if let Some(idx) = text.find("</think>") {
            text = text[idx + 8..].trim();
        }

        // Since LLMs sometimes format with markdown like ```json ... ```, strip it
        let clean = text
            .replace("```json", "")
            .replace("```", "")
            .trim()
            .to_string();

        let response: LlmResponse = serde_json::from_str(&clean).unwrap_or_else(|_| LlmResponse {
            is_command: false,
            text: "Failed to parse json. Raw output: ".to_string() + text,
        });

        Ok(response)
    }
}
