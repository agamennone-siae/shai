use anyhow::Result;
use rustyline::highlight::{CmdKind, Highlighter};
use rustyline::hint::Hinter;
use rustyline::history::DefaultHistory;
use rustyline::{Completer, Context, Editor, Helper, Validator};
use rustyline::error::ReadlineError;
use std::borrow::Cow;
use std::process::Command;

use crate::executor;
use crate::llm::Engine;

// ---------------------------------------------------------------------------
// Helper struct — wires rustyline completion, validation, hinting, and
// syntax highlighting together for real-time input classification feedback.
// ---------------------------------------------------------------------------

/// The derive macros forward no-op implementations for Completer, Validator
/// and the **trait** Helper (which just requires both Completer + Hinter +
/// Highlighter + Validator are satisfied).  We provide Highlighter and Hinter
/// ourselves so the interesting behaviour is fully under our control.
#[derive(Completer, Helper, Validator)]
struct ShaiHelper;

// --- Input classification --------------------------------------------------

#[derive(Clone, PartialEq)]
enum InputKind {
    Empty,
    /// The first word is a known shell binary (found via `which`).
    Command,
    /// Everything else is treated as a natural-language AI query.
    AiQuery,
}

fn classify(line: &str) -> InputKind {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return InputKind::Empty;
    }
    let first = trimmed.split_whitespace().next().unwrap_or("");
    if is_system_cmd(first) {
        InputKind::Command
    } else {
        InputKind::AiQuery
    }
}

// --- Highlighter -----------------------------------------------------------
//
// Colors the whole line:  bold-green = shell command, cyan = AI query.
// ANSI sequences have zero display-width so rustyline's column maths is fine.

impl Highlighter for ShaiHelper {
    fn highlight<'l>(&self, line: &'l str, _pos: usize) -> Cow<'l, str> {
        match classify(line) {
            InputKind::Empty => Cow::Borrowed(line),
            InputKind::Command => Cow::Owned(format!("\x1b[1;32m{line}\x1b[0m")),
            InputKind::AiQuery => Cow::Owned(format!("\x1b[36m{line}\x1b[0m")),
        }
    }

    /// Return `true` to force a re-highlight on every keystroke.
    fn highlight_char(&self, _line: &str, _pos: usize, _kind: CmdKind) -> bool {
        true
    }
}

// --- Hinter ----------------------------------------------------------------
//
// A dim suffix appears right after the cursor showing the current mode.
// `String` satisfies rustyline's `Hint` trait via the blanket impl.

impl Hinter for ShaiHelper {
    type Hint = String;

    fn hint(&self, line: &str, _pos: usize, _ctx: &Context<'_>) -> Option<String> {
        match classify(line) {
            InputKind::Empty => None,
            // \x1b[2m = dim grey, \x1b[0m = reset
            InputKind::Command => Some("\x1b[2m  [shell command]\x1b[0m".to_string()),
            InputKind::AiQuery => Some("\x1b[2m  [AI query]\x1b[0m".to_string()),
        }
    }
}

// ---------------------------------------------------------------------------
// REPL entry point
// ---------------------------------------------------------------------------

pub async fn run_repl(engine: Engine) -> Result<()> {
    println!("Welcome to shai REPL! Type your natural language request or a normal command.");
    println!("Type 'exit' or 'quit' to close.");
    println!();
    println!("  \x1b[1;32m■\x1b[0m bold green = shell command  (executed directly)");
    println!("  \x1b[36m■\x1b[0m cyan        = AI query      (sent to the LLM)");
    println!();

    let mut rl: Editor<ShaiHelper, DefaultHistory> = Editor::new()?;
    rl.set_helper(Some(ShaiHelper));

    loop {
        let readline = rl.readline("shai> ");
        match readline {
            Ok(line) => {
                let line = line.trim().to_owned();
                if line.is_empty() {
                    continue;
                }

                rl.add_history_entry(line.as_str())?;

                if line == "exit" || line == "quit" {
                    break;
                }

                let parts: Vec<&str> = line.split_whitespace().collect();
                if is_system_cmd(parts[0]) {
                    // Execute raw command immediately
                    let child = Command::new(parts[0]).args(&parts[1..]).spawn();
                    if let Ok(mut c) = child {
                        let _ = c.wait();
                    } else {
                        println!("Failed to execute {}", parts[0]);
                    }
                } else {
                    // Treat as natural language request to AI
                    if let Err(e) = executor::run_oneshot(&engine, &line).await {
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

    Ok(())
}

fn is_system_cmd(cmd: &str) -> bool {
    if let Ok(output) = Command::new("which").arg(cmd).output() {
        output.status.success()
    } else {
        false
    }
}
