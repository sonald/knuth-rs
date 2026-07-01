use anyhow::Result;
use dotenvy::dotenv;


use knuth_agent::harness::{AgentSession, AgentConfig};
use knuth_core::AgentEvent;
use futures::StreamExt;

mod config;
use config::UserSettings;


use clap::{Parser, Subcommand};

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
        input: String,
    },

    Sessions {
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    let args = Args::parse();

    match args.commands {
        Commands::Chat { input } => {
            chat(input).await?;
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

async fn chat(input: String) -> Result<()> {
    let user_settings = UserSettings::load()?;
    let  system_prompt = "You are a helpful assistant. ".to_string();

    let mut session = AgentSession::new("test".to_string(),
     "test".to_string(),
      system_prompt,
      AgentConfig { model: user_settings.model.clone(), options: user_settings.options.clone() });

    session.submit_input(input).await?;

    let mut subscription = session.subscribe().await?;

    while let Some(event) = subscription.next().await {
        match event {
            AgentEvent::AssistantMessageTextDelta { delta, .. } => {
                print!("{}", delta);
            }
            AgentEvent::AssistantMessageThinkingDelta { delta, .. } => {
                print!("{}", delta);
            }
            _ => {
                println!("{}", event);
            }
        }
    }

    Ok(())
}
