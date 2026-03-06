use anyhow::Result;
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;
use std::process::Command;

use crate::executor;
use crate::llm::Engine;

pub async fn run_repl(engine: Engine) -> Result<()> {
    println!("Welcome to shai REPL! Type your natural language request or a normal command.");
    println!("Type 'exit' or 'quit' to close.");

    let mut rl = DefaultEditor::new()?;

    // Optional history support
    // rl.load_history("history.txt").unwrap_or_default();

    loop {
        let readline = rl.readline("shai> ");
        match readline {
            Ok(line) => {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                rl.add_history_entry(line)?;

                if line == "exit" || line == "quit" {
                    break;
                }

                // Super naive check to see if it's an existing system command.
                let parts: Vec<&str> = line.split_whitespace().collect();
                if is_system_cmd(parts[0]) {
                    // Execute raw command immediately since user knows what they're doing
                    let child = Command::new(parts[0]).args(&parts[1..]).spawn();

                    if let Ok(mut c) = child {
                        let _ = c.wait();
                    } else {
                        println!("Failed to execute {}", parts[0]);
                    }
                } else {
                    // Treat as natural language request to AI
                    if let Err(e) = executor::run_oneshot(&engine, line).await {
                        println!("Error: {}", e);
                    }
                }
            }
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => {
                break;
            }
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        }
    }

    // rl.save_history("history.txt")?;
    Ok(())
}

fn is_system_cmd(cmd: &str) -> bool {
    // Return true if `which <cmd>` passes.
    if let Ok(output) = Command::new("which").arg(cmd).output() {
        output.status.success()
    } else {
        false
    }
}
