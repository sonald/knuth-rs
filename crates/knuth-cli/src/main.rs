use anyhow::{Context, Result};
use dotenvy::dotenv;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use reedline::{DefaultPrompt, FileBackedHistory, Reedline, Signal};
use std::{
    collections::HashMap,
    io::{self, Write},
    path::PathBuf,
    time::Duration,
};

use futures::StreamExt;
use knuth_agent::harness::{AgentConfig, AgentSession};
use knuth_core::{AgentEvent, AgentSubscription, LiveEvent, SessionEvent};

mod config;
use config::UserSettings;

use clap::{Parser, Subcommand};
use crossterm::style::Stylize;
use serde_json::Value;
use tracing::{debug, info};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(name = "knuth")]
struct Args {
    #[arg(short('m'), long)]
    model: Option<String>,

    #[arg(short('c'), long, value_name = "FILE")]
    config: Option<PathBuf>,

    #[arg(long)]
    print_config: bool,

    #[command(subcommand)]
    commands: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Chat {
        #[arg(short('p'), long)]
        input: Option<String>,
        #[arg(short('i'), long)]
        images: Option<Vec<String>>,
    },

    Sessions {},
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_ansi(true)
        .with_ansi_sanitization(false)
        .init();

    let args = Args::parse();
    let model = args.model;
    let config = args.config;

    match args.commands {
        Commands::Chat { input, images } => match input {
            Some(input) => {
                let user_settings = UserSettings::load(model.as_deref(), config.as_deref())?;
                if args.print_config {
                    print_effective_config(&user_settings);
                }
                oneshot(input, images.unwrap_or_default(), user_settings).await?;
            }
            None => {
                let user_settings = UserSettings::load(model.as_deref(), config.as_deref())?;
                if args.print_config {
                    print_effective_config(&user_settings);
                }
                chat_loop(user_settings).await?;
            }
        },
        Commands::Sessions {} => {
            list_sessions().await?;
        }
    }
    Ok(())
}

async fn list_sessions() -> Result<()> {
    println!("Sessions: empty");
    Ok(())
}

const SYSTEM_PROMPT: &str = "You are a helpful assistant.";

async fn build_session(user_settings: &UserSettings) -> Result<(AgentSession, AgentSubscription)> {
    let mut session = AgentSession::build(
        "test".to_string(),
        "test".to_string(),
        AgentConfig {
            model: user_settings.model.clone(),
            options: user_settings.options.clone(),
        },
    )
    .await;

    let subscription = session.subscribe(None).await?;
    session.set_system_prompt(SYSTEM_PROMPT.to_string()).await?;
    Ok((session, subscription))
}

struct CliRenderer {
    progress: MultiProgress,
    thinking: Option<ProgressBar>,
    tools: HashMap<String, ProgressBar>,
}

impl CliRenderer {
    fn new() -> Self {
        Self {
            progress: MultiProgress::new(),
            thinking: None,
            tools: HashMap::new(),
        }
    }

    fn spinner(&self, message: String) -> ProgressBar {
        let spinner = self.progress.add(ProgressBar::new_spinner());
        spinner.set_style(
            ProgressStyle::with_template("{spinner:.cyan} {msg}")
                .expect("hard-coded spinner template is valid")
                .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
        );
        spinner.set_message(message);
        spinner.enable_steady_tick(Duration::from_millis(80));
        spinner
    }

    fn print(&self, text: impl std::fmt::Display) {
        self.progress.suspend(|| {
            print!("{text}");
            let _ = io::stdout().flush();
        });
    }

    fn render(&mut self, event: &SessionEvent) {
        match event {
            SessionEvent::Live(event) => self.render_live(event),
            SessionEvent::Durable(stored) => self.render_durable(&stored.event),
        }
    }

    /// Streaming output: everything the user watches arrive token by token.
    fn render_live(&mut self, event: &LiveEvent) {
        match event {
            LiveEvent::AssistantMessageTextDelta { delta, .. } => {
                self.print(delta.as_str().green());
            }
            LiveEvent::AssistantMessageTextCompleted { .. } => {
                self.print('\n');
            }
            LiveEvent::AssistantMessageThinkingStarted { .. }
            | LiveEvent::AssistantMessageThinkingDelta { .. } => {
                if self.thinking.is_none() {
                    self.thinking = Some(self.spinner("Thinking".to_string()));
                }
            }
            LiveEvent::AssistantMessageThinkingCompleted { content, .. } => {
                if let Some(spinner) = self.thinking.take() {
                    spinner.finish_and_clear();
                }
                self.print(format!("* Thinking:\n{content}\n").blue());
            }
            LiveEvent::ToolExecutionStarted {
                tool_call_id,
                tool_name,
                arguments,
                ..
            } => {
                let message = format!(
                    "Exec {}({})",
                    tool_name,
                    serde_json::to_string(arguments).unwrap_or_default()
                );
                self.tools
                    .insert(tool_call_id.clone(), self.spinner(message));
            }
            LiveEvent::ToolExecutionEnded {
                tool_call_id,
                result,
                ..
            } => {
                if let Some(spinner) = self.tools.remove(tool_call_id) {
                    let message = spinner.message().to_string();
                    spinner.finish_and_clear();
                    self.print(format!("* {message}\n").cyan());
                }
                self.print(format!("* Result:\n{result}\n").cyan());
            }
            LiveEvent::AssistantMessageTextStarted { .. }
            | LiveEvent::ToolExecutionUpdated { .. } => {
                debug!("Live: {}", format!("{event}").dark_yellow());
            }
        }
    }

    /// Committed history: only errors are surfaced; the rest is trace output,
    /// since the live stream already showed the user what happened.
    fn render_durable(&mut self, event: &AgentEvent) {
        match event {
            AgentEvent::ErrorOccurred { message, .. } => {
                self.progress
                    .suspend(|| eprintln!("{}", message.as_str().red()));
            }
            _ => {
                debug!("Ev: {}", format!("{event}").dark_yellow());
            }
        }
    }
}

impl Drop for CliRenderer {
    fn drop(&mut self) {
        if let Some(spinner) = self.thinking.take() {
            spinner.finish_and_clear();
        }
        for (_, spinner) in self.tools.drain() {
            spinner.finish_and_clear();
        }
    }
}

/// Renders events until the current turn ends. The first Ctrl+C cancels the
/// turn (it finishes with a `Cancelled` step and ends normally); a second
/// Ctrl+C force-quits in case the turn cannot end (e.g. a wedged connection).
async fn run_turn(session: &mut AgentSession, subscription: &mut AgentSubscription) -> Result<()> {
    let mut cancel_requested = false;
    let mut renderer = CliRenderer::new();
    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                if cancel_requested {
                    eprintln!("force quit");
                    std::process::exit(130);
                }
                cancel_requested = true;
                info!("Cancelling current turn...");
                session.cancel_current_turn().await?;
            }
            maybe_event = subscription.next() => {
                let Some(event) = maybe_event else {
                    anyhow::bail!("event stream closed unexpectedly");
                };
                renderer.render(&event);
                if matches!(
                    event.as_durable().map(|stored| &stored.event),
                    Some(AgentEvent::AgentTurnEnded { .. })
                ) {
                    println!();
                    return Ok(());
                }
            }
        }
    }
}

async fn oneshot(input: String, images: Vec<String>, user_settings: UserSettings) -> Result<()> {
    let (mut session, mut subscription) = build_session(&user_settings).await?;

    session.submit_input(input, images).await?;
    run_turn(&mut session, &mut subscription).await?;

    session.close().await?;
    Ok(())
}

async fn chat_loop(user_settings: UserSettings) -> Result<()> {
    let (mut session, mut subscription) = build_session(&user_settings).await?;

    let history = Box::new(
        FileBackedHistory::with_file(1000, config::default_history_file()?)
            .context("failed to initialize command history")?,
    );
    let mut reedline = Reedline::create().with_history(history);
    let prompt = DefaultPrompt::default();

    loop {
        let sig = reedline.read_line(&prompt);
        match sig {
            Ok(Signal::Success(line)) => {
                if line.trim().is_empty() {
                    continue;
                }
                session.submit_input(line, vec![]).await?;
                run_turn(&mut session, &mut subscription).await?;
            }
            Ok(Signal::CtrlC) | Ok(Signal::CtrlD) => {
                session.close().await?;
                break;
            }
            x => {
                println!("Unknown signal: {:?}", x);
            }
        }
    }

    Ok(())
}

fn print_effective_config(settings: &UserSettings) {
    let model = &settings.model;
    let options = &settings.options;

    eprintln!("{}", "Effective config".bold());
    eprintln!("  model: {}", model.id);
    eprintln!("  name: {}", model.name);
    eprintln!("  provider: {}", model.provider.0);
    eprintln!("  api: {}", model.api.0);
    eprintln!("  base_url: {}", empty_dash(&model.base_url));
    eprintln!("  context_window: {}", model.context_window);
    eprintln!("  model_max_tokens: {}", model.max_tokens);
    eprintln!("  model_reasoning: {}", model.reasoning);
    eprintln!("  input: {:?}", model.input);
    eprintln!("  options:");
    eprintln!("    max_tokens: {}", opt(options.max_tokens));
    eprintln!("    temperature: {}", opt(options.temperature));
    eprintln!("    cache_retention: {}", opt(options.cache_retention));
    eprintln!(
        "    api_key: {}",
        redacted_presence(options.api_key.as_deref())
    );
    eprintln!("    headers: {}", jsonish(&options.headers));
    eprintln!("    provider_extras: {}", jsonish(&options.provider_extras));
}

fn opt<T: std::fmt::Debug>(value: Option<T>) -> String {
    value
        .map(|value| format!("{value:?}"))
        .unwrap_or_else(|| "-".to_string())
}

fn empty_dash(value: &str) -> &str {
    if value.is_empty() { "-" } else { value }
}

fn redacted_presence(value: Option<&str>) -> String {
    match value {
        Some(value) if !value.is_empty() => format!("<set, {} chars>", value.len()),
        _ => "-".to_string(),
    }
}

fn jsonish<T: serde::Serialize>(value: &T) -> String {
    let mut value = serde_json::to_value(value).unwrap_or(Value::Null);
    redact_json_value(&mut value);
    match value {
        Value::Null => "-".to_string(),
        Value::Object(ref map) if map.is_empty() => "-".to_string(),
        _ => serde_json::to_string_pretty(&value).unwrap_or_else(|_| format!("{value:?}")),
    }
}

fn redact_json_value(value: &mut Value) {
    match value {
        Value::Object(map) => {
            for (key, value) in map {
                if key.to_ascii_lowercase().contains("key")
                    || key.to_ascii_lowercase().contains("token")
                    || key.to_ascii_lowercase().contains("secret")
                {
                    *value = Value::String("<redacted>".to_string());
                } else {
                    redact_json_value(value);
                }
            }
        }
        Value::Array(values) => values.iter_mut().for_each(redact_json_value),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use knuth_core::ids::StepId;
    use std::path::Path;

    #[test]
    fn parses_config_path() {
        let args = Args::parse_from(["knuth", "--config", "/tmp/knuth.yaml", "chat"]);

        assert_eq!(args.config.as_deref(), Some(Path::new("/tmp/knuth.yaml")));
    }

    #[test]
    fn parses_print_config() {
        let args = Args::parse_from(["knuth", "--print-config", "chat"]);

        assert!(args.print_config);
    }

    #[test]
    fn redacts_secret_like_json_keys() {
        let mut value = serde_json::json!({
            "api_key": "secret",
            "nested": { "access_token": "token", "safe": "shown" }
        });

        redact_json_value(&mut value);

        assert_eq!(value["api_key"], "<redacted>");
        assert_eq!(value["nested"]["access_token"], "<redacted>");
        assert_eq!(value["nested"]["safe"], "shown");
    }

    #[test]
    fn renderer_tracks_thinking_and_tool_lifetimes() {
        let mut renderer = CliRenderer::new();
        let step_id = StepId::new();

        renderer.render_live(&LiveEvent::AssistantMessageThinkingStarted {
            step_id,
            content_index: 0,
        });
        assert!(renderer.thinking.is_some());

        renderer.render_live(&LiveEvent::AssistantMessageThinkingCompleted {
            step_id,
            content_index: 0,
            content: "done".to_string(),
        });
        assert!(renderer.thinking.is_none());

        renderer.render_live(&LiveEvent::ToolExecutionStarted {
            step_id,
            tool_call_id: "call-1".to_string(),
            tool_name: "bash".to_string(),
            arguments: serde_json::Map::new(),
        });
        assert!(renderer.tools.contains_key("call-1"));

        renderer.render_live(&LiveEvent::ToolExecutionEnded {
            step_id,
            tool_call_id: "call-1".to_string(),
            tool_name: "bash".to_string(),
            result: "ok".to_string(),
        });
        assert!(renderer.tools.is_empty());
    }
}
