use anyhow::Result;
use dialoguer::{Select, theme::ColorfulTheme};
use hf_hub::api::tokio::Api;
use std::fs;
use std::path::PathBuf;

const MODELS_DIR: &str = "models";

pub async fn ensure_model() -> Result<PathBuf> {
    fs::create_dir_all(MODELS_DIR)?;

    // Check for existing .gguf models
    let mut existing_models = Vec::new();
    for entry in fs::read_dir(MODELS_DIR)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
            existing_models.push(path);
        }
    }

    if !existing_models.is_empty() {
        if existing_models.len() == 1 {
            println!(
                "Using local model: {:?}",
                existing_models[0].file_name().unwrap()
            );
            return Ok(existing_models[0].clone());
        } else {
            // Let user select if multiple exist
            let file_names: Vec<String> = existing_models
                .iter()
                .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
                .collect();
            let selection = Select::with_theme(&ColorfulTheme::default())
                .with_prompt("Multiple models found in models/ directory. Select one to use")
                .default(0)
                .items(&file_names)
                .interact()?;
            return Ok(existing_models[selection].clone());
        }
    }

    // No models found, prompt to download Qwen 3.5
    println!("No local .gguf models found in models/ directory.");

    // We will use Qwen3.5 GGUFs quantization from unsloth.
    let available_models = [
        (
            "unsloth/Qwen3.5-0.8B-GGUF",
            "Qwen3.5-0.8B-Q4_K_M.gguf",
        ),
        (
            "unsloth/Qwen3.5-2B-GGUF",
            "Qwen3.5-2B-Q4_K_M.gguf",
        ),
        (
            "unsloth/Qwen3.5-4B-GGUF",
            "Qwen3.5-4B-Q4_K_M.gguf",
        ),
    ];

    let items: Vec<String> = available_models
        .iter()
        .map(|(repo, _)| repo.to_string())
        .collect();

    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select a Qwen3.5 model to download (0.8B is fastest, 2B/4B is smarter)")
        .default(0)
        .items(&items)
        .interact()?;

    let (repo_id, filename) = available_models[selection];
    println!(
        "Downloading {} from Hugging Face Hub... This may take a while.",
        filename
    );

    let api = Api::new()?;
    let repo = api.model(repo_id.to_string());

    // hf-hub will download it to its cache
    let path = repo.get(filename).await?;

    let dest_path = PathBuf::from(MODELS_DIR).join(filename);
    println!("Copying model to {:?}", dest_path);
    fs::copy(&path, &dest_path)?;

    Ok(dest_path)
}
