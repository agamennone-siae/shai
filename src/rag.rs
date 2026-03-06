use std::process::Command;

pub fn guess_command(prompt: &str) -> Option<String> {
    // Naive fallback: try to find any existing command in the prompt
    let words: Vec<&str> = prompt.split_whitespace().collect();
    for word in words {
        // Strip punctuation
        let clean: String = word
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '-')
            .collect();
        if !clean.is_empty() && is_executable(&clean) {
            // Hardcode some extremely common words to ignore
            let ignores = ["is", "it", "to", "do", "in", "on", "file", "make"];
            if !ignores.contains(&clean.as_str()) {
                return Some(clean);
            }
        }
    }
    None
}

fn is_executable(cmd: &str) -> bool {
    Command::new("which")
        .arg(cmd)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

pub fn fetch_documentation(cmd: &str) -> Option<String> {
    // Extract man page
    if let Ok(output) = Command::new("sh")
        .arg("-c")
        .arg(format!("man {} 2>/dev/null | col -bx | head -n 100", cmd))
        .output()
    {
        if output.status.success() {
            let info = String::from_utf8_lossy(&output.stdout);
            return Some(format!("man documentation for {}:\n{}", cmd, info.trim()));
        }
    }

    // Fallback extract --help
    if let Ok(output) = Command::new(cmd).arg("--help").output() {
        if output.status.success() {
            let info = String::from_utf8_lossy(&output.stdout);
            let truncated: String = info.chars().take(1500).collect();
            return Some(format!("--help output for {}:\n{}", cmd, truncated.trim()));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guess_command_basic() {
        // Mock environment assuming `ls` and `echo` exist
        assert_eq!(
            guess_command("list files using ls pls"),
            Some("ls".to_string())
        );
        assert_eq!(guess_command("echo hello world"), Some("echo".to_string()));

        // Negative test
        assert_eq!(guess_command("i am just talking to you"), None);
    }
}
