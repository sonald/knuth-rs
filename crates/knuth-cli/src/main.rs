use anyhow::Result;
use dotenvy::dotenv;
use reedline::{DefaultPrompt, Reedline, Signal};
use std::path::PathBuf;

use futures::StreamExt;
use knuth_agent::harness::{AgentConfig, AgentSession};
use knuth_core::AgentEvent;

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

async fn oneshot(input: String, images: Vec<String>, user_settings: UserSettings) -> Result<()> {
    let system_prompt = "You are a helpful assistant.".to_string();

    let mut session = AgentSession::build(
        "test".to_string(),
        "test".to_string(),
        AgentConfig {
            model: user_settings.model.clone(),
            options: user_settings.options.clone(),
        },
    )
    .await;

    let quit_token = tokio_util::sync::CancellationToken::new();
    let quit_token_clone = quit_token.clone();

    let mut subscription = session.subscribe(None).await?;

    tokio::spawn(async move {
        while let Some(event) = subscription.next().await {
            match event.event {
                AgentEvent::AssistantMessageTextDelta { delta, .. } => {
                    print!("{}", delta.green());
                }
                AgentEvent::AssistantMessageThinkingDelta { delta, .. } => {
                    print!("{}", delta.blue());
                }

                AgentEvent::ErrorOccurred { message, .. } => {
                    eprintln!("{}", message.red());
                }

                AgentEvent::AgentTurnEnded { .. } => {
                    quit_token_clone.cancel();
                }

                AgentEvent::ToolExecutionStarted {
                    tool_name,
                    arguments,
                    ..
                } => {
                    println!(
                        "{}",
                        format!(
                            "* ToolRequest {}({})",
                            tool_name,
                            serde_json::to_string(&arguments).unwrap()
                        )
                        .cyan()
                    );
                }
                AgentEvent::ToolExecutionEnded { result, .. } => {
                    println!("{}", format!("* Result: {}", result).cyan());
                }
                _ => {
                    let msg = format!("{}", event);
                    debug!("Ev: {}", msg.dark_yellow());
                }
            }
        }

        debug!("Session ended");
    });

    session.set_system_prompt(system_prompt).await?;
    session.submit_input(input, images).await?;

    loop {
        tokio::select! {
            _ = quit_token.cancelled() => {
                info!("Quitting...");
                session.close().await?;
                break;
            }
        }
    }

    Ok(())
}

async fn chat_loop(user_settings: UserSettings) -> Result<()> {
    let system_prompt = "You are a helpful assistant.".to_string();

    let mut session = AgentSession::build(
        "test".to_string(),
        "test".to_string(),
        AgentConfig {
            model: user_settings.model.clone(),
            options: user_settings.options.clone(),
        },
    )
    .await;

    let (turn_ended_tx, mut turn_ended) = tokio::sync::mpsc::channel(2);

    let mut subscription = session.subscribe(None).await?;
    tokio::spawn(async move {
        while let Some(event) = subscription.next().await {
            match event.event {
                AgentEvent::AssistantMessageTextDelta { delta, .. } => {
                    print!("{}", delta.green());
                }
                AgentEvent::AssistantMessageThinkingDelta { delta, .. } => {
                    print!("{}", delta.blue());
                }

                AgentEvent::ErrorOccurred { message, .. } => {
                    eprintln!("{}", message.red());
                }

                AgentEvent::AgentTurnEnded { .. } => {
                    turn_ended_tx.send(()).await.unwrap();
                }
                AgentEvent::ToolExecutionStarted {
                    tool_name,
                    arguments,
                    ..
                } => {
                    println!(
                        "{}",
                        format!(
                            "* Exec {}({})",
                            tool_name,
                            serde_json::to_string(&arguments).unwrap()
                        )
                        .cyan()
                    );
                }
                AgentEvent::ToolExecutionEnded { result, .. } => {
                    println!("{}", format!("* Result:\n{}", result).cyan());
                }
                _ => {
                    let msg = format!("{}", event);
                    debug!("{}", msg.dark_yellow());
                }
            }
        }
    });

    session.set_system_prompt(system_prompt).await?;

    let mut reedline = Reedline::create();
    let prompt = DefaultPrompt::default();

    loop {
        let sig = reedline.read_line(&prompt);
        match sig {
            Ok(Signal::Success(line)) => {
                session.submit_input(line, vec![]).await?;
                loop {
                    tokio::select! {
                        _ = turn_ended.recv() => {
                            break;
                        }
                    }
                }
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
}
