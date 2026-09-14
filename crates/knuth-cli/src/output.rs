use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::str::FromStr;

/// How the CLI prints tool calls and their results.
///
/// `default` is the current full dump. `concise` matches the collapsed
/// Codex / Claude Code tool rows: a short invocation and a truncated body.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputStyle {
    #[default]
    Default,
    Concise,
}

impl OutputStyle {
    pub fn as_str(self) -> &'static str {
        match self {
            OutputStyle::Default => "default",
            OutputStyle::Concise => "concise",
        }
    }
}

impl FromStr for OutputStyle {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "default" => Ok(OutputStyle::Default),
            "concise" => Ok(OutputStyle::Concise),
            other => Err(format!(
                "unknown output style '{other}'; expected default or concise"
            )),
        }
    }
}

const CONCISE_ARG_CHARS: usize = 80;
const CONCISE_PREVIEW_LINES: usize = 5;
const CONCISE_LINE_CHARS: usize = 120;

/// Spinner label while a tool is running.
pub fn format_tool_running(
    style: OutputStyle,
    tool_name: &str,
    arguments: &Map<String, Value>,
) -> String {
    match style {
        OutputStyle::Default => format!(
            "Exec {tool_name}({})",
            serde_json::to_string(arguments).unwrap_or_default()
        ),
        OutputStyle::Concise => compact_invocation(tool_name, arguments),
    }
}

/// Final printed block after a tool finishes.
pub fn format_tool_finished(
    style: OutputStyle,
    tool_name: &str,
    arguments: &Map<String, Value>,
    result: &str,
) -> String {
    match style {
        OutputStyle::Default => {
            let running = format_tool_running(style, tool_name, arguments);
            format!("* {running}\n* Result:\n{result}\n")
        }
        OutputStyle::Concise => {
            let heading = compact_invocation(tool_name, arguments);
            let body = indent_result(&concise_result_body(tool_name, result));
            format!("* {heading}\n{body}\n")
        }
    }
}

fn compact_invocation(tool_name: &str, arguments: &Map<String, Value>) -> String {
    let preview = compact_arg_preview(tool_name, arguments);
    if preview.is_empty() {
        tool_name.to_string()
    } else {
        format!("{tool_name}({preview})")
    }
}

fn compact_arg_preview(tool_name: &str, arguments: &Map<String, Value>) -> String {
    let preferred = match tool_name {
        "bash" => string_arg(arguments, "command"),
        "read_file" | "write_file" | "edit_file" => string_arg(arguments, "path"),
        "python_exec" => string_arg(arguments, "code"),
        _ => None,
    };
    if let Some(text) = preferred {
        return heading_preview(text);
    }
    // Known tools: never fall back to dumping the whole argument object
    // (write_file's `content` can be an entire file).
    if matches!(
        tool_name,
        "bash" | "read_file" | "write_file" | "edit_file" | "python_exec"
    ) {
        return String::new();
    }
    heading_preview(&serde_json::to_string(arguments).unwrap_or_default())
}

fn string_arg<'a>(arguments: &'a Map<String, Value>, name: &str) -> Option<&'a str> {
    arguments.get(name).and_then(Value::as_str)
}

fn heading_preview(text: &str) -> String {
    let line = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("");
    truncate_chars(line, CONCISE_ARG_CHARS)
}

fn concise_result_body(tool_name: &str, result: &str) -> String {
    match tool_name {
        "read_file" => summarize_read_file(result)
            .unwrap_or_else(|| preview_lines(result, CONCISE_PREVIEW_LINES)),
        "bash" | "python_exec" => summarize_exec(result),
        _ => preview_lines(result, CONCISE_PREVIEW_LINES),
    }
}

fn summarize_read_file(result: &str) -> Option<String> {
    let header = result.lines().next()?;
    let rest = header.strip_prefix("File(")?;
    let (_path, after) = rest.split_once(") - Lines ")?;
    let (range, total_part) = after.split_once(" of ")?;
    let total = total_part.strip_suffix(" total:")?;
    Some(format!("Read lines {range} of {total}"))
}

fn summarize_exec(result: &str) -> String {
    let Some((status, stdout, stderr)) = split_exec_output(result) else {
        return preview_lines(result, CONCISE_PREVIEW_LINES);
    };
    let success = exec_succeeded(&status);
    let stdout = stdout.trim_end_matches('\n');
    let stderr = stderr.trim_end_matches('\n');
    if success {
        if stdout.is_empty() && stderr.is_empty() {
            return "ok".to_string();
        }
        let body = if !stdout.is_empty() { stdout } else { stderr };
        return preview_lines(body, CONCISE_PREVIEW_LINES);
    }
    let body = if !stderr.is_empty() {
        stderr
    } else if !stdout.is_empty() {
        stdout
    } else {
        return status;
    };
    format!("{status}\n{}", preview_lines(body, CONCISE_PREVIEW_LINES))
}

fn exec_succeeded(status: &str) -> bool {
    status.contains("exit status: 0.")
        || status.contains("exit status: 0\n")
        || status.ends_with("exit status: 0")
        || status.contains("exit code: 0.")
}

fn split_exec_output(result: &str) -> Option<(String, &str, &str)> {
    const STDOUT: &str = "\nstdout:\n";
    const STDERR: &str = "\nstderr:\n";
    let stdout_at = result.find(STDOUT)?;
    let stderr_at = result[stdout_at + STDOUT.len()..].rfind(STDERR)?;
    let stderr_at = stdout_at + STDOUT.len() + stderr_at;
    let status = result[..stdout_at].to_string();
    let stdout = &result[stdout_at + STDOUT.len()..stderr_at];
    let stderr = &result[stderr_at + STDERR.len()..];
    Some((status, stdout, stderr))
}

fn preview_lines(text: &str, max_lines: usize) -> String {
    if text.is_empty() {
        return String::new();
    }
    let lines: Vec<&str> = text.lines().collect();
    let end = lines
        .iter()
        .rposition(|line| !line.is_empty())
        .map(|index| index + 1)
        .unwrap_or(0);
    let lines = &lines[..end];
    if lines.is_empty() {
        return String::new();
    }
    let clipped: Vec<String> = lines
        .iter()
        .take(max_lines)
        .map(|line| truncate_chars(line, CONCISE_LINE_CHARS))
        .collect();
    if lines.len() <= max_lines {
        return clipped.join("\n");
    }
    format!(
        "{}\n… +{} lines",
        clipped.join("\n"),
        lines.len() - max_lines
    )
}

fn indent_result(body: &str) -> String {
    let mut lines = body.lines();
    let Some(first) = lines.next() else {
        return "  ⎿ (empty)".to_string();
    };
    let mut out = format!("  ⎿ {first}");
    for line in lines {
        out.push_str("\n    ");
        out.push_str(line);
    }
    out
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars();
    let truncated: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{truncated}…")
    } else {
        truncated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn args(value: Value) -> Map<String, Value> {
        value.as_object().unwrap().clone()
    }

    #[test]
    fn parses_style_names() {
        assert_eq!(
            "default".parse::<OutputStyle>().unwrap(),
            OutputStyle::Default
        );
        assert_eq!(
            "Concise".parse::<OutputStyle>().unwrap(),
            OutputStyle::Concise
        );
        assert!("verbose".parse::<OutputStyle>().is_err());
    }

    #[test]
    fn default_dumps_exec_line_and_full_result() {
        let arguments = args(json!({"command": "ls"}));
        let result = "Command exited with exit status: 0.\nstdout:\nok\nstderr:\n";
        let rendered = format_tool_finished(OutputStyle::Default, "bash", &arguments, result);
        assert_eq!(
            rendered,
            "* Exec bash({\"command\":\"ls\"})\n* Result:\nCommand exited with exit status: 0.\nstdout:\nok\nstderr:\n\n"
        );
    }

    #[test]
    fn default_running_label_includes_json_arguments() {
        let arguments = args(json!({"path": "notes.txt", "content": "hello"}));
        assert_eq!(
            format_tool_running(OutputStyle::Default, "write_file", &arguments),
            "Exec write_file({\"path\":\"notes.txt\",\"content\":\"hello\"})"
        );
    }

    #[test]
    fn concise_bash_shows_stdout_preview() {
        let arguments = args(json!({"command": "ls"}));
        let result =
            "Command exited with exit status: 0.\nstdout:\nCargo.toml\nREADME.md\nstderr:\n";
        let rendered = format_tool_finished(OutputStyle::Concise, "bash", &arguments, result);
        assert_eq!(rendered, "* bash(ls)\n  ⎿ Cargo.toml\n    README.md\n");
    }

    #[test]
    fn concise_bash_empty_output_is_ok() {
        let arguments = args(json!({"command": "true"}));
        let result = "Command exited with exit status: 0.\nstdout:\n\nstderr:\n";
        let rendered = format_tool_finished(OutputStyle::Concise, "bash", &arguments, result);
        assert_eq!(rendered, "* bash(true)\n  ⎿ ok\n");
    }

    #[test]
    fn concise_truncates_long_output() {
        let arguments = args(json!({"command": "seq 10"}));
        let stdout = (1..=10)
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let result = format!("Command exited with exit status: 0.\nstdout:\n{stdout}\nstderr:\n");
        let rendered = format_tool_finished(OutputStyle::Concise, "bash", &arguments, &result);
        assert_eq!(
            rendered,
            "* bash(seq 10)\n  ⎿ 1\n    2\n    3\n    4\n    5\n    … +5 lines\n"
        );
    }

    #[test]
    fn concise_bash_error_shows_status_and_stderr() {
        let arguments = args(json!({"command": "ls /nope"}));
        let result =
            "Command exited with exit status: 1.\nstdout:\n\nstderr:\nls: /nope: No such file\n";
        let rendered = format_tool_finished(OutputStyle::Concise, "bash", &arguments, result);
        assert_eq!(
            rendered,
            "* bash(ls /nope)\n  ⎿ Command exited with exit status: 1.\n    ls: /nope: No such file\n"
        );
    }

    #[test]
    fn concise_does_not_treat_exit_10_as_success() {
        let arguments = args(json!({"command": "exit 10"}));
        let result = "Command exited with exit status: 10.\nstdout:\n\nstderr:\nbad\n";
        let rendered = format_tool_finished(OutputStyle::Concise, "bash", &arguments, result);
        assert!(rendered.contains("exit status: 10"), "rendered={rendered}");
        assert!(rendered.contains("bad"), "rendered={rendered}");
        assert!(!rendered.contains("  ⎿ ok\n"), "rendered={rendered}");
    }

    #[test]
    fn concise_read_file_summarizes_range_without_body() {
        let arguments = args(json!({"path": "src/main.rs"}));
        let result =
            "File(src/main.rs) - Lines 1-200 of 400 total:\n   1: fn main() {}\n   2: {}\n";
        let rendered = format_tool_finished(OutputStyle::Concise, "read_file", &arguments, result);
        assert_eq!(
            rendered,
            "* read_file(src/main.rs)\n  ⎿ Read lines 1-200 of 400\n"
        );
    }

    #[test]
    fn concise_write_file_hides_content_argument() {
        let arguments = args(json!({"path": "notes.txt", "content": "very long content"}));
        let rendered = format_tool_finished(
            OutputStyle::Concise,
            "write_file",
            &arguments,
            "Wrote notes.txt",
        );
        assert_eq!(rendered, "* write_file(notes.txt)\n  ⎿ Wrote notes.txt\n");
        assert!(!rendered.contains("very long content"));
    }

    #[test]
    fn concise_truncates_long_command_heading() {
        let command = "a".repeat(100);
        let arguments = args(json!({"command": command}));
        let heading = format_tool_running(OutputStyle::Concise, "bash", &arguments);
        assert!(heading.starts_with("bash(aaaa"));
        assert!(heading.contains('…'));
        assert!(heading.ends_with(')'));
        assert_eq!(
            heading.chars().count(),
            "bash(".chars().count() + 80 + "…)".chars().count()
        );
    }

    #[test]
    fn concise_uses_first_line_of_multiline_command() {
        let arguments = args(json!({"command": "echo one\necho two"}));
        let heading = format_tool_running(OutputStyle::Concise, "bash", &arguments);
        assert_eq!(heading, "bash(echo one)");
    }

    #[test]
    fn concise_unknown_tool_falls_back_to_truncated_json() {
        let arguments = args(json!({"query": "hello"}));
        let heading = format_tool_running(OutputStyle::Concise, "search", &arguments);
        assert_eq!(heading, "search({\"query\":\"hello\"})");
    }
}
