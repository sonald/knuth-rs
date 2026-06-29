use anyhow::Result;
use dotenv::dotenv;

use ai::{
    Api, Context, KnownApi, Message, Model, ModelCost, StreamOptions, UserContent, UserMessage,
    UserRole, models::get_model, stream, types::Provider,
};
use futures::StreamExt;
use std::env;

// pub trait Event<'de>: Debug + Serialize + Deserialize<'de> {}

// pub trait EventStore<E: Event> {
//     async fn append(&self, event: &E) -> Result<()>;
// }

struct UserSettings {
    model: Model,
    options: StreamOptions,
}

impl UserSettings {
    fn new(model: Model, options: StreamOptions) -> Self {
        Self { model, options }
    }

    fn load() -> Result<Self> {
        let model = Self::load_model_from_env()?;
        let options = StreamOptions {
            max_tokens: Some(1024),
            api_key: Some(env::var("KNUTH_API_KEY")?),
            ..Default::default()
        };
        Ok(Self { model, options })
    }

    fn load_model_from_env() -> Result<Model> {
        let model = env::var("KNUTH_MODEL")?;
        let base_url = env::var("KNUTH_BASE_URL")?;

        let model = if model.contains('/') {
            let parts: Vec<&str> = model.split('/').collect();
            let provider = parts[0];
            let model = parts[1];
            get_model(&Provider::from(provider), model)
        } else {
            //this is custom model
            let model = Model {
                id: model.to_string(),
                name: model,
                api: Api::known(KnownApi::OpenAICompletions),
                // api: Api::known(KnownApi::OpenAIResponses),
                provider: Provider::from("custom"),
                base_url: base_url,
                reasoning: true,
                thinking_level_map: None,
                input: vec![ai::InputModality::Text],
                cost: ModelCost::default(),
                context_window: 1_000_000,
                max_tokens: 10_000,
                headers: None,
                compat: None,
            };
            // register_custom_model(model.clone());
            Some(model)
        };

        model.ok_or_else(|| anyhow::anyhow!("Model not found"))
    }
}

fn event_variant_name(event: &ai::AssistantMessageEvent) -> &'static str {
    use ai::AssistantMessageEvent::*;
    match event {
        Start { .. } => "Start",
        TextStart { .. } => "TextStart",
        TextDelta { .. } => "TextDelta",
        TextEnd { .. } => "TextEnd",
        ThinkingStart { .. } => "ThinkingStart",
        ThinkingDelta { .. } => "ThinkingDelta",
        ThinkingEnd { .. } => "ThinkingEnd",
        ToolCallStart { .. } => "ToolCallStart",
        ToolCallDelta { .. } => "ToolCallDelta",
        ToolCallEnd { .. } => "ToolCallEnd",
        Done { .. } => "Done",
        Error { .. } => "Error",
    }
}

async fn run_loop(config: &UserSettings, context: &Context) {
    let mut stream = stream(&config.model, context, Some(&config.options));

    while let Some(event) = stream.next().await {
        match event {
            ai::AssistantMessageEvent::TextDelta { delta, .. } => {
                print!("{}", delta);
            }
            ai::AssistantMessageEvent::ThinkingDelta { delta, .. } => {
                print!("....{}", delta);
            }
            ai::AssistantMessageEvent::Done { message, .. } => {
                println!("\n--- done ---");
                println!(
                    "usage: in={} out={}",
                    message.usage.input, message.usage.output
                );
                break;
            }
            ai::AssistantMessageEvent::Error { error, reason } => {
                eprintln!("error: {:?}", error.error_message);
                eprintln!("reason: {:?}", reason);
                break;
            }
            _ => println!("Event: {}", event_variant_name(&event)),
        }
    }
}

struct Session {}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    let config = UserSettings::load()?;
    let context = Context {
        system_prompt: Some("You are a terse assistant. ".into()),
        messages: vec![Message::User(UserMessage {
            role: UserRole::User,
            content: UserContent::Text("what is the capital of France?".into()),
            timestamp: 0,
        })],
        tools: None,
    };

    run_loop(&config, &context).await;

    Ok(())
}
