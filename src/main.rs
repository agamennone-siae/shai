use clap::Parser;

mod executor;
mod llm;
mod rag;
mod setup;
mod shell;

#[derive(Parser, Debug)]
#[command(author, version, about = "AI CLI Assistant running Qwen 3.5 locally", long_about = None)]
struct Args {
    /// The natural language request to execute
    #[arg(trailing_var_arg = true)]
    request: Vec<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Ensure the model is available locally. If not, trigger interactive download.
    let model_path = setup::ensure_model().await?;

    let request = args.request.join(" ");

    // Initialize the LLM engine
    let engine = llm::Engine::new(&model_path)?;

    if request.is_empty() {
        // Run interactive shell
        shell::run_repl(engine).await?;
    } else {
        // One-shot mode
        executor::run_oneshot(&engine, &request).await?;
    }

    Ok(())
}
