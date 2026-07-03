use anyhow::Result;
use dotenvy::dotenv;
use reedline::{DefaultPrompt, Reedline, Signal};

use knuth_agent::harness::{AgentSession, AgentConfig};
use knuth_core::AgentEvent;
use futures::StreamExt;

mod config;
use config::UserSettings;


use clap::{Parser, Subcommand};
use tracing::debug;
use crossterm::style::Stylize;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(name = "knuth")]
struct Args {
    #[arg(short('m'), long)]
    model: Option<String>,

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

    Sessions {
    }
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

    match args.commands {
        Commands::Chat { input, images } => {
            match input {
                Some(input) => {
                    oneshot(input, images.unwrap_or_default()).await?;
                }
                None => {
                    chat_loop().await?;
                }
            }
        }
        Commands::Sessions { } => {
            list_sessions().await?;
        }
    }
    Ok(())
}

async fn list_sessions() -> Result<()> {
    println!("Sessions: empty");
    Ok(())
}

async fn oneshot(input: String, images: Vec<String>) -> Result<()> {
    let user_settings = UserSettings::load()?;
    let  system_prompt = "You are a helpful assistant.".to_string();

    let mut session = AgentSession::new("test".to_string(),
     "test".to_string(),
      AgentConfig { model: user_settings.model.clone(), options: user_settings.options.clone() });


    let quit_token = tokio_util::sync::CancellationToken::new();
    let quit_token_clone = quit_token.clone();

    let mut subscription = session.subscribe().await?;
    tokio::spawn(async move {
        while let Some(event) = subscription.next().await {
            match event {
                AgentEvent::AssistantMessageTextDelta { delta, .. } => {
                    print!("{}", delta.green());
                }
                AgentEvent::AssistantMessageThinkingDelta { delta, .. } => {
                    print!("{}", delta.blue());
                }

                AgentEvent::AgentTurnEnded { .. } => {
                    quit_token_clone.cancel();
                }
                _ => {
                    let msg = format!("{}", event);
                    debug!("{}", msg.dark_yellow());
                }
            }
        }
    });

    session.set_system_prompt(system_prompt).await?;
    session.submit_input(input, images).await?;
    loop {
        tokio::select! {
            _ = quit_token.cancelled() => {
                println!("Quitting...");
                session.close().await?;
                break;
            }
        }
    }

    Ok(())
}


async fn chat_loop() -> Result<()> {
    let user_settings = UserSettings::load()?;
    let  system_prompt = "You are a helpful assistant.".to_string();

    let mut session = AgentSession::new("test".to_string(),
     "test".to_string(),
      AgentConfig { model: user_settings.model.clone(), options: user_settings.options.clone() });

    let (turn_ended_tx, mut turn_ended) = tokio::sync::mpsc::channel(2);

    let mut subscription = session.subscribe().await?;
    tokio::spawn(async move {
        while let Some(event) = subscription.next().await {
            match event {
                AgentEvent::AssistantMessageTextDelta { delta, .. } => {
                    print!("{}", delta.green());
                }
                AgentEvent::AssistantMessageThinkingDelta { delta, .. } => {
                    print!("{}", delta.blue());
                }

                AgentEvent::AgentTurnEnded { .. } => {
                    turn_ended_tx.send(()).await.unwrap();
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
