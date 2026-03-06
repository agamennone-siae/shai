use anyhow::{Context, Result};
use dialoguer::{Confirm, theme::ColorfulTheme};
use std::process::Stdio;

use crate::llm::Engine;

pub async fn run_oneshot(engine: &Engine, request: &str) -> Result<()> {
    // Determine if it's a direct command or chat. The engine handles this.
    // For now, let's assume one-shot request mostly wants commands.
    let response = engine.generate_response(request).await?;

    // We expect a JSON response from the LLM or a structured enum.
    // For simplicity here, let's say the engine returns the command string if it's a command,
    // or plain text if it's a chat response.
    if response.is_command {
        execute_command_with_confirmation(&response.text)?;
    } else {
        println!("\nAI: {}", response.text);
    }

    Ok(())
}

pub fn execute_command_with_confirmation(command: &str) -> Result<()> {
    println!("\nProposed Command:\n\x1b[1;32m{}\x1b[0m\n", command);

    if Confirm::with_theme(&ColorfulTheme::default())
        .with_prompt("Do you want to execute this command?")
        .default(false)
        .interact()?
    {
        // Execute the command in the shell
        let mut child = std::process::Command::new("sh")
            .arg("-c")
            .arg(command)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("Failed to execute command: {}", command))?;

        child.wait()?;
    } else {
        println!("Command execution cancelled.");
    }

    Ok(())
}
