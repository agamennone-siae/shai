use anyhow::{Context, Result};
use dialoguer::{Confirm, theme::ColorfulTheme};
use std::process::Stdio;

use crate::llm::Engine;

pub async fn run_oneshot(engine: &Engine, request: &str) -> Result<()> {
    let response = engine.generate_response(request).await?;

    if !response.text.is_empty() {
        execute_command_with_confirmation(&response.text)?;
    } else {
        println!("No command generated.");
    }

    Ok(())
}

pub fn execute_command_with_confirmation(command: &str) -> Result<()> {
    println!();
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
