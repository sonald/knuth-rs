# AI Crate 完整修复实施计划

> **给执行 agent：** 实施本计划时必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`。逐项按复选框推进，不要跳过红绿测试。

**目标：** 修复已经确认真实存在的 `crates/ai` provider 正确性问题，并为每个修复留下可回归的测试覆盖。

**边界：** 继续沿用现有的 provider-per-wire-protocol 结构。只在已有重复逻辑处增加小型内部 helper：终止事件判断、请求 header 合并、usage cost 计算、测试 mock 支撑。除非某个任务证明现有结构无法承载行为，否则不增加新的公开 provider 抽象或公开领域类型。

**技术栈：** Rust 2024、`tokio`、`reqwest`、`serde_json`、现有 `ai` provider 模块、现有本地 TCP mock-server 风格测试。

## 全局约束

- 遵守仓库要求：`如无必要，勿增实体`。
- 每个行为修复都先写失败的回归测试。
- 保留现有公开结构：`Model`、`StreamOptions`、`Usage`、`AssistantMessageEvent`、`ApiProvider`。
- 各 Cargo feature 必须能独立工作，`all-providers` 也必须通过。
- `pi-mono` 只作为行为参考，不作为搬运未使用 surface 的理由。
- 不新增 mock、CRC、AWS、Google auth、coverage、HTTP server 依赖。
- 完成标准：每个确认问题都有具名回归测试，并出现在文末覆盖矩阵里。

## 文件清单

- 新增：`crates/ai/tests/support/mod.rs`
  - 本地 HTTP/SSE mock server、可指定响应状态的请求捕获、panic-safe env guard、provider model 构造器。
  - Bedrock AWS eventstream frame builder。
- 新增：`crates/ai/tests/provider_terminal_events.rs`
  - Anthropic、OpenAI completions、Mistral、Google、Vertex、Bedrock、OpenAI Responses 的 EOF/终止事件与 usage 测试。
- 新增：`crates/ai/tests/provider_request_metadata.rs`
  - `model.headers`、`options.headers`、Cloudflare URL、Codex `max_tokens`、Bedrock SigV4、Vertex ADC/错误脱敏的请求捕获测试。
- 修改：`crates/ai/src/utils/headers.rs`
  - 保留 `merge_headers`，增加内部 helper：先合并 `model.headers`，再让 `options.headers` 覆盖。
- 修改：`crates/ai/src/models.rs`
  - 增加 `calculate_usage_cost(model: &Model, usage: &mut Usage)`。
- 修改：`crates/ai/src/providers/anthropic.rs`
  - EOF 终止检查、`model.headers`、cost 计算、`response_model`。
- 修改：`crates/ai/src/providers/openai_completions.rs`
  - EOF 终止检查、`model.headers`、Cloudflare base URL、cache-aware usage、cost 计算。
- 修改：`crates/ai/src/providers/mistral.rs`
  - EOF 终止检查、并行 tool-call 累积、`model.headers`、usage/cost。
- 修改：`crates/ai/src/providers/google.rs`
  - EOF 终止检查、`model.headers`、cost 计算。
- 修改：`crates/ai/src/providers/amazon_bedrock.rs`
  - EOF 终止检查、`stream_simple` reasoning 透传、`model.headers`、cost 计算、SigV4 fallback。
- 修改：`crates/ai/src/sigv4.rs`
  - 固定 canonical request/signature 向量，并确保 signer 生成值覆盖调用方同名 headers。
- 修改：`crates/ai/src/providers/openai_responses.rs`
  - cost 计算、cache-aware usage total、done event 按 `output_index`/`item_id` 路由、复用 EOF helper。
- 修改：`crates/ai/src/providers/openai_codex_responses.rs`
  - `max_output_tokens`、encrypted reasoning replay、`model.headers`、cost 计算。
- 修改：`crates/ai/src/providers/google_vertex.rs`
  - `model.headers`、复用 Google consumer 后的 cost 计算、通过现有 `vertex_adc` 做 ADC fallback。
- 修改：`crates/ai/src/providers/register_builtins.rs`
  - 使用 register-if-absent 恢复缺失 built-ins，保留同 API custom provider，并串行化 lifecycle。
- 修改：`crates/ai/src/api_registry.rs`
  - 增加内部 register-if-absent seam；不改公开 API。
- 修改：`crates/ai/src/stream.rs`
  - 在拿到 `RegisteredHandle` 前，复用 lifecycle-protected ensure-and-get lookup。
- 修改：`crates/ai/src/utils/retry.rs`
  - 去掉会 panic 的防御分支，收窄 retryable request error。
- 修改：`crates/ai/src/utils/sse.rs`
  - `data:` 后只去掉一个前导空格，不吞掉所有空格。
- 修改：`crates/ai/README.md`
  - 代码和测试完成后更新 provider 状态表。

---

## Task 1：建立 Provider 级 HTTP/Stream 回归测试支撑

**文件：**
- 新增：`crates/ai/tests/support/mod.rs`

**产物：**
- `serve_once(body, content_type) -> String`
- `serve_capture_once(body, content_type) -> (String, oneshot::Receiver<CapturedRequest>)`
- `aws_eventstream_frame(event_type, payload) -> Vec<u8>`
- provider model builder。

- [x] **Step 1：新增 shared mock server**

```rust
use std::sync::OnceLock;

use ai::{Api, KnownApi, Model, ModelCost, Provider};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

#[derive(Debug, Clone)]
pub struct CapturedRequest {
    pub request: String,
}

pub fn env_lock() -> &'static tokio::sync::Mutex<()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

pub async fn serve_once(body: &'static [u8], content_type: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut buf = [0u8; 8192];
        let _ = socket.read(&mut buf).await;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        socket.write_all(response.as_bytes()).await.unwrap();
        socket.write_all(body).await.unwrap();
        socket.flush().await.unwrap();
    });
    format!("http://{addr}")
}

pub async fn serve_capture_once(
    body: &'static [u8],
    content_type: &'static str,
) -> (String, oneshot::Receiver<CapturedRequest>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut buf = [0u8; 16384];
        let n = socket.read(&mut buf).await.unwrap_or(0);
        let _ = tx.send(CapturedRequest {
            request: String::from_utf8_lossy(&buf[..n]).to_string(),
        });
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        socket.write_all(response.as_bytes()).await.unwrap();
        socket.write_all(body).await.unwrap();
        socket.flush().await.unwrap();
    });
    (format!("http://{addr}"), rx)
}

pub fn aws_eventstream_frame(event_type: &str, payload: &[u8]) -> Vec<u8> {
    let name = b":event-type";
    let mut headers = Vec::new();
    headers.push(name.len() as u8);
    headers.extend_from_slice(name);
    headers.push(7);
    headers.extend_from_slice(&(event_type.len() as u16).to_be_bytes());
    headers.extend_from_slice(event_type.as_bytes());

    let total_len = 12 + headers.len() + payload.len() + 4;
    let mut out = Vec::new();
    out.extend_from_slice(&(total_len as u32).to_be_bytes());
    out.extend_from_slice(&(headers.len() as u32).to_be_bytes());
    out.extend_from_slice(&0u32.to_be_bytes());
    out.extend_from_slice(&headers);
    out.extend_from_slice(payload);
    out.extend_from_slice(&0u32.to_be_bytes());
    out
}

pub fn model(api: KnownApi, provider: &str, id: &str, base_url: String) -> Model {
    Model {
        id: id.into(),
        name: id.into(),
        api: Api::known(api),
        provider: Provider::from(provider),
        base_url,
        reasoning: false,
        thinking_level_map: None,
        input: vec![],
        cost: ModelCost {
            input: 1.0,
            output: 2.0,
            cache_read: 0.25,
            cache_write: 1.25,
        },
        context_window: 128_000,
        max_tokens: 4096,
        headers: None,
        compat: None,
    }
}
```

- [x] **Step 2：支撑代码编译检查**

运行：

```bash
cargo test -p ai --features all-providers --no-run
```

预期：PASS。新的 support module 在被 integration test 引入前，不应改变现有测试。

---

## Task 2：EOF 必须是 Error，除非收到 Provider 原生终止事件

**文件：**
- 新增：`crates/ai/tests/provider_terminal_events.rs`
- 修改：`crates/ai/src/providers/anthropic.rs`
- 修改：`crates/ai/src/providers/openai_completions.rs`
- 修改：`crates/ai/src/providers/mistral.rs`
- 修改：`crates/ai/src/providers/google.rs`
- 修改：`crates/ai/src/providers/amazon_bedrock.rs`

**行为：**
- 网络 EOF 不再自动转成 `Done`。
- 只有 provider 原生终止事件才可以产生 `Done`。
- EOF 前未见终止事件时，产生 `AssistantMessageEvent::Error`，`stop_reason = StopReason::Error`。

- [x] **Step 1：新增失败测试**

在 `provider_terminal_events.rs` 中新增这些具名测试：

```rust
anthropic_eof_before_message_stop_is_error
openai_completions_eof_before_done_or_finish_reason_is_error
mistral_eof_before_done_or_finish_reason_is_error
google_eof_before_finish_reason_is_error
bedrock_eof_before_message_stop_is_error
openai_responses_eof_before_terminal_event_is_error
```

断言统一为：
- 至少收到一个 `AssistantMessageEvent::Error`。
- 不允许收到 `AssistantMessageEvent::Done`。
- `error.error_message` 包含 `stream ended before terminal event`。

- [x] **Step 2：确认失败**

运行：

```bash
cargo test -p ai --features anthropic,openai-completions,mistral,google,amazon-bedrock,openai-responses --test provider_terminal_events
```

预期：新测试失败；当前 provider 会在截断 EOF 后发 `Done` 或静默完成。

- [x] **Step 3：在各 provider stream loop 中记录终止事件**

以 Anthropic 为例，保留现有 `handle_sse`，只在 caller loop 增加终止标记：

```rust
let mut saw_terminal = false;
let stream_error_message = "anthropic stream ended before terminal event";
loop {
    let item = match abort_utils::next_or_abort(&mut sse, options.abort.as_ref()).await {
        AbortableNext::Item(item) => item,
        AbortableNext::Eof => break,
        AbortableNext::Aborted => {
            abort_utils::push_aborted(&mut sender, &model);
            return;
        }
    };

    match item {
        Err(e) => {
            push_error(&mut sender, &model, format!("sse: {e}"));
            return;
        }
        Ok(ev) => {
            if !handle_sse(&ev, &mut partial, &mut tool_arg_buffers, &mut sender) {
                saw_terminal = true;
                break;
            }
        }
    }
}

if saw_terminal {
    return;
}

partial.stop_reason = StopReason::Error;
partial.error_message = Some(stream_error_message.into());
sender.push(AssistantMessageEvent::Error {
    reason: ErrorReason::Error,
    error: partial,
});
```

各 provider 的终止条件：
- Anthropic：`message_stop` 或 `error`。
- OpenAI completions：收到带 `finish_reason` 的 chunk 后，必须再收到 `[DONE]`；若 endpoint 省略 `[DONE]`，EOF 后只允许在已有 terminal `finish_reason` 时 finalize 一次。
- Mistral：同 OpenAI completions，同时保留 Task 3 的 tool-call buffer。
- Google：candidate 带 `finishReason`。
- Bedrock Converse：`messageStop` 或 exception frame。
- OpenAI Responses：已有 EOF Error 行为；保留并纳入测试矩阵。

- [x] **Step 4：终止事件测试通过**

运行：

```bash
cargo test -p ai --features anthropic,openai-completions,mistral,google,amazon-bedrock,openai-responses --test provider_terminal_events
```

预期：PASS。

- [x] **Step 5：全 provider 回归**

运行：

```bash
cargo test -p ai --features all-providers
```

预期：PASS。

---

## Task 3：Mistral 并行 Tool Calls 不能合并

**文件：**
- 修改：`crates/ai/src/providers/mistral.rs`
- 新增或复用：`crates/ai/tests/provider_terminal_events.rs`

**行为：**
- 同一个 chunk 里多个 `tool_calls` 必须按 index/id 分开累积参数。
- 不允许把两个 tool call 的 JSON 参数拼到同一个 buffer。
- 入站 tool-call id 继续遵守 Mistral 的 9 位字母数字 normalization 契约。

- [x] **Step 1：新增失败测试**

测试名：

```rust
mistral_parallel_tool_calls_do_not_merge_arguments
```

mock SSE 内容包含两个并行 tool call：

```json
[
  {
    "index": 0,
    "id": "alpha1234",
    "function": { "name": "a", "arguments": "{\"x\":1}" }
  },
  {
    "index": 1,
    "id": "bravo5678",
    "function": { "name": "b", "arguments": "{\"y\":2}" }
  }
]
```

断言：
- 最终 message 有两个 `ContentBlock::ToolCall`。
- 第一个 id/name/arguments 为 `alpha1234`/`a`/`{"x":1}`。
- 第二个 id/name/arguments 为 `bravo5678`/`b`/`{"y":2}`。
- 另以跨 chunk 事件测试覆盖非法或超长 id 被规范化为 9 位字母数字，且每个 call 的 Start/Delta/End 始终使用自己的 content index。

- [x] **Step 2：确认失败**

运行：

```bash
cargo test -p ai --features mistral --test provider_terminal_events mistral_parallel_tool_calls_do_not_merge_arguments
```

预期：FAIL，当前实现会把参数错误合并。

- [x] **Step 3：按 provider index/id 建立 buffer**

在 `mistral.rs` 中使用现有 collection 类型即可，不新增公开类型：

```rust
let mut tool_call_positions: HashMap<u64, usize> = HashMap::new();
let mut tool_arg_buffers: HashMap<u64, String> = HashMap::new();

for tc in tool_calls {
    let index = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
    let pos = *tool_call_positions.entry(index).or_insert_with(|| {
        let p = partial.content.len();
        partial.content.push(ContentBlock::ToolCall(ToolCall {
            id: normalize_tool_call_id(tc.get("id").and_then(|v| v.as_str()).unwrap_or("")),
            name: tc.pointer("/function/name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            arguments: Map::new(),
            thought_signature: None,
        }));
        p
    });

    if let Some(args) = tc.pointer("/function/arguments").and_then(|v| v.as_str()) {
        tool_arg_buffers.entry(index).or_default().push_str(args);
        sender.push(AssistantMessageEvent::ToolCallDelta {
            content_index: pos,
            delta: args.to_string(),
            partial: partial.clone(),
        });
    }
}
```

- [x] **Step 4：解析各自 buffer**

在 terminal finish 时逐个 index 解析：

```rust
for (index, raw) in &tool_arg_buffers {
    let Some(pos) = tool_call_positions.get(index).copied() else {
        continue;
    };
    let Ok(Value::Object(args)) = crate::utils::json_parse::parse_partial_json(raw) else {
        continue;
    };
    if let Some(ContentBlock::ToolCall(tc)) = partial.content.get_mut(pos) {
        tc.arguments = args;
    }
}
```

- [x] **Step 5：测试通过**

运行：

```bash
cargo test -p ai --features mistral --test provider_terminal_events mistral_parallel_tool_calls_do_not_merge_arguments
```

预期：PASS。

---

## Task 4：`model.headers` 和 Cloudflare Base URL 必须进真实请求

**文件：**
- 新增：`crates/ai/tests/provider_request_metadata.rs`
- 修改：`crates/ai/src/utils/headers.rs`
- 修改：所有发 HTTP 请求的 provider。
- 修改：`crates/ai/src/providers/openai_completions.rs`
- 修改：`crates/ai/src/providers/anthropic.rs`
- 修改：`crates/ai/src/providers/openai_responses.rs`

**行为：**
- 请求 header 顺序：provider 默认 header < `model.headers` < `options.headers`。
- HTTP header 名大小写不敏感；`options.headers` 同名 key 覆盖 `model.headers`。
- 自定义 header 通过 `RequestBuilder::headers(HeaderMap)` 替换 provider 默认值，不能通过会追加值的 `RequestBuilder::header()` 应用。
- 非法 `HeaderName` / `HeaderValue` 通过公开 stream 产生 `Error`，不能 panic。
- Cloudflare base URL 中的 `{CLOUDFLARE_ACCOUNT_ID}` / `{CLOUDFLARE_GATEWAY_ID}` 必须在 OpenAI Completions、Anthropic Messages、OpenAI Responses 三条真实请求路径构造 URL 前解析。

- [x] **Step 1：新增 header 合并失败测试**

测试名：

```rust
openai_completions_merges_model_headers_then_options_headers
model_headers_override_provider_defaults
options_headers_override_provider_defaults
options_headers_override_model_headers_case_insensitively
invalid_custom_header_emits_error
```

断言捕获请求包含：
- `x-model-header: model`
- `x-options-header: options`
- `x-shared-header: options`
- 不包含 `x-shared-header: model`

- [x] **Step 2：新增 Cloudflare URL 失败测试**

```rust
#[tokio::test]
async fn cloudflare_placeholders_are_resolved_before_request() {
    let _guard = support::env_lock().lock().await;
    unsafe {
        std::env::set_var("CLOUDFLARE_ACCOUNT_ID", "acct123");
    }

    let body = br#"data: {"id":"cf_1","choices":[{"finish_reason":"stop","delta":{}}]}

data: [DONE]

"#;
    let (base, rx) = support::serve_capture_once(body, "text/event-stream").await;
    let mut model = support::model(
        KnownApi::OpenAICompletions,
        "cloudflare-workers-ai",
        "@cf/meta/llama",
        format!("{base}/accounts/{{CLOUDFLARE_ACCOUNT_ID}}/ai/v1"),
    );
    let opts = StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    };
    let mut s = stream(&model, &ctx(), Some(&opts));
    while s.next().await.is_some() {}
    let captured = rx.await.unwrap().request;
    assert!(captured.starts_with("POST /accounts/acct123/ai/v1/chat/completions "));

    unsafe {
        std::env::remove_var("CLOUDFLARE_ACCOUNT_ID");
    }
}
```

同时增加并使用能正常产生 provider 原生终止事件的 SSE body：

```rust
cloudflare_anthropic_placeholders_are_resolved_before_request
cloudflare_responses_placeholders_are_resolved_before_request
```

三条 API 路径分别断言：

- OpenAI Completions：解析 placeholder 后再追加 `/chat/completions`。
- Anthropic Messages：解析 placeholder 后再追加 `/v1/messages`。
- OpenAI Responses：解析 placeholder 后再追加 `/v1/responses`。

保留 `cloudflare_missing_placeholder_emits_named_error`，断言缺失环境变量名出现在 `Error` event 中。

- [x] **Step 3：确认失败**

运行：

```bash
cargo test -p ai --features openai-completions,anthropic,openai-responses,cloudflare --test provider_request_metadata
```

预期：header 测试失败，因为追加语义不能替换 provider 默认值，也不能保证大小写不敏感覆盖；Anthropic / Responses Cloudflare 测试失败，因为 placeholder 没被替换。

- [x] **Step 4：增加最小内部 helper**

在 `utils/headers.rs` 中增加内部 helper，构造大小写不敏感的 `HeaderMap`：

```rust
pub(crate) fn merged_model_and_option_headers(
    model_headers: Option<&HashMap<String, String>>,
    option_headers: Option<&HashMap<String, String>>,
) -> Result<HeaderMap, String> {
    let mut merged = HeaderMap::new();
    for headers in [model_headers, option_headers].into_iter().flatten() {
        for (name, value) in headers {
            let name = HeaderName::from_bytes(name.as_bytes())?;
            let value = HeaderValue::from_str(value)?;
            merged.insert(name, value);
        }
    }
    Ok(merged)
}
```

伪代码中的 `?` 代表将两类 parse error 转成内部错误文本；实现不得新增公开错误类型。各 provider 在构造默认 header 后调用 `RequestBuilder::headers(merged)`，利用 reqwest `replace_headers` 语义替换同名默认值。解析失败时发 `Error` event 并返回。不要增加 request builder 抽象。

- [x] **Step 5：Cloudflare 复用现有 resolver**

在 `openai_completions.rs`、`anthropic.rs`、`openai_responses.rs` 计算 base URL 时，对 Cloudflare provider 调用现有 Cloudflare resolver。若 env 缺失，返回 `Error` event，错误信息包含缺失变量名。

- [x] **Step 6：测试通过**

运行：

```bash
cargo test -p ai --features openai-completions,anthropic,openai-responses,cloudflare --test provider_request_metadata
cargo test -p ai --features openai-completions,anthropic,openai-responses,cloudflare --lib utils::headers::tests
cargo test -p ai --features all-providers
```

预期：PASS。

---

## Task 5：Usage 语义和 Cost 计算

**文件：**
- 修改：`crates/ai/src/models.rs`
- 修改：所有填充 `Usage` 的 provider。

**行为：**
- `Usage.cost` 按 `Model.cost` 计算。
- cache read/write 不再重复计入 normal input token。
- `total_tokens` 语义统一：`input + output + cache_read + cache_write`，其中 `input` 是非缓存输入。
- `bedrock_anthropic::Converter` 必须显式接收 `Model` 价格上下文；不得保留无价格的 `Default/new()` 构造路径。
- Mistral 的 `prompt_tokens` 减去 `prompt_tokens_details.cached_tokens`；Google 的非缓存 input 额外包含 `toolUsePromptTokenCount`。
- OpenAI Responses 的 terminal `response.usage` 是整次 snapshot；cache read/write 都从 `input_tokens` 中扣除，重复 snapshot 不累加。
- Bedrock-Anthropic 的 `message_start.message.usage` 与 `message_delta.usage` 复用同一局部更新语义：只替换实际出现的字段，Done/Error 均按显式 Model 结算 cost。

- [x] **Step 1：新增 unit tests**

测试名：

```rust
calculate_usage_cost_uses_per_million_prices
usage_subtracts_cached_tokens_from_openai_prompt_tokens
responses_usage_subtracts_cached_tokens_from_input_tokens
```

cost 计算公式：

```rust
usage.cost = (usage.input as f64 * model.cost.input
    + usage.output as f64 * model.cost.output
    + usage.cache_read as f64 * model.cost.cache_read
    + usage.cache_write as f64 * model.cost.cache_write)
    / 1_000_000.0;
```

- [x] **Step 2：确认失败**

运行：

```bash
cargo test -p ai --features all-providers calculate_usage_cost_uses_per_million_prices usage_subtracts_cached_tokens_from_openai_prompt_tokens responses_usage_subtracts_cached_tokens_from_input_tokens
```

预期：FAIL，当前 `Usage.cost` 为 0，cached token 计数重复。

- [x] **Step 3：增加 helper 并在 provider 终止前调用**

在 `models.rs` 中增加：

```rust
pub fn calculate_usage_cost(model: &Model, usage: &mut Usage) {
    usage.cost = (usage.input as f64 * model.cost.input
        + usage.output as f64 * model.cost.output
        + usage.cache_read as f64 * model.cost.cache_read
        + usage.cache_write as f64 * model.cost.cache_write)
        / 1_000_000.0;
}
```

每个 provider 在发 `Done` 或 terminal `Error` 前调用一次。不要新建 cost service。

- [x] **Step 4：修正 cached token 计数**

OpenAI completions：
- `prompt_tokens` 包含 cached tokens。
- `prompt_tokens_details.cached_tokens` 计入 `cache_read`。
- `usage.input = prompt_tokens - cached_tokens`，使用 saturating subtraction。

OpenAI Responses：
- `input_tokens` 包含 cached tokens。
- `input_tokens_details.cached_tokens` 计入 `cache_read`。
- `usage.input = input_tokens - cached_tokens`，使用 saturating subtraction。

- [x] **Step 5：测试通过**

运行：

```bash
cargo test -p ai --features all-providers calculate_usage_cost_uses_per_million_prices usage_subtracts_cached_tokens_from_openai_prompt_tokens responses_usage_subtracts_cached_tokens_from_input_tokens
cargo test -p ai --features all-providers
```

预期：PASS。

---

## Task 6：Codex Responses 请求行为补齐

**文件：**
- 修改：`crates/ai/src/providers/openai_codex_responses.rs`
- 新增或复用：`crates/ai/tests/provider_request_metadata.rs`

**行为：**
- `StreamOptions.max_tokens` 写入请求体 `max_output_tokens`。
- encrypted reasoning item 从上一轮 assistant message replay 到请求体。
- `model.headers` 参与请求。
- usage/cost 与 OpenAI Responses 一致。

- [x] **Step 1：新增请求捕获测试**

测试名：

```rust
codex_request_includes_max_output_tokens
codex_replays_encrypted_reasoning_items
```

断言：
- 请求 JSON 包含 `"max_output_tokens": 1234`。
- replay 的 encrypted reasoning item 原样出现在 `input` 中。

- [x] **Step 2：确认失败**

运行：

```bash
cargo test -p ai --features openai-codex-responses --test provider_request_metadata codex_request_includes_max_output_tokens codex_replays_encrypted_reasoning_items
```

预期：FAIL。

- [x] **Step 3：补 request body**

在构造 Codex Responses body 时：

```rust
if let Some(max_tokens) = options.max_tokens {
    body["max_output_tokens"] = json!(max_tokens);
}
```

replay encrypted reasoning 时只复用已有 `ContentBlock::Thinking`/provider extras 已保存的信息，不新增公开 message 类型。若现有结构无法表达，先在测试中证明缺口，再加最小内部序列化 helper。

- [x] **Step 4：测试通过**

运行：

```bash
cargo test -p ai --features openai-codex-responses --test provider_request_metadata codex_request_includes_max_output_tokens codex_replays_encrypted_reasoning_items
cargo test -p ai --features all-providers
```

预期：PASS。

---

## Task 7：OpenAI Responses Done Event 必须按 Block Identity 路由

**文件：**
- 修改：`crates/ai/src/providers/openai_responses.rs`

**行为：**
- `response.output_text.done`、`response.reasoning_summary_text.done` 等 done event 不能按“最后一个 block”路由。
- 必须按 `output_index` / `content_index` / `item_id` 找到对应 block。

- [x] **Step 1：新增失败测试**

测试名：

```rust
text_done_routes_by_output_index_not_last_block
```

构造 event 顺序：
1. text block start/index 0。
2. reasoning block start/index 1。
3. text done 指向 index 0。

断言 text done 结束的是 text block，不是最后的 reasoning block。

- [x] **Step 2：确认失败**

运行：

```bash
cargo test -p ai --features openai-responses text_done_routes_by_output_index_not_last_block
```

预期：FAIL。

- [x] **Step 3：复用已有 block map**

检查 `openai_responses.rs` 中已有的 index/item tracking。若已有 map，只补 done event lookup。若没有，用最小内部 map：

```rust
HashMap<(u64, u64), usize>
HashMap<String, usize>
```

不要引入新的公开 block identity 类型。

- [x] **Step 4：测试通过**

运行：

```bash
cargo test -p ai --features openai-responses text_done_routes_by_output_index_not_last_block
cargo test -p ai --features all-providers
```

预期：PASS。

---

## Task 8：Registry、Retry、SSE 的边界修复

**文件：**
- 修改：`crates/ai/src/api_registry.rs`
- 修改：`crates/ai/src/providers/register_builtins.rs`
- 修改：`crates/ai/src/stream.rs`
- 修改：`crates/ai/src/utils/retry.rs`
- 修改：`crates/ai/src/utils/sse.rs`

**行为：**
- 每次 ensure 都补齐缺失 built-ins，不覆盖同 API 的 custom provider。
- `clear_api_providers()` / unregister 与 stream lookup 不会在 ensure 和 handle 获取之间制造 missing-provider。
- retry 工具不再在 request builder error 上 panic。
- SSE `data:` 只移除规范允许的一个前导空格。

- [x] **Step 1：新增 Registry、Retry、SSE 回归测试**

测试名：

```rust
stream_re_registers_builtins_after_clear
ensure_restores_missing_builtins_when_registry_contains_custom_provider
ensure_preserves_custom_override_for_builtin_api
clear_racing_with_stream_lookup_does_not_return_missing_provider_error
retryable_reqwest_errors_exclude_request_builder_errors
data_field_removes_only_one_leading_space
multiple_events_are_emitted_fifo
line_and_data_split_across_chunks_are_reassembled
eof_without_blank_line_flushes_current_event
```

**必须覆盖的回归矩阵：**

| Boundary | Test |
| --- | --- |
| clear 后 built-in stream | `stream_re_registers_builtins_after_clear` |
| request builder error | `retryable_reqwest_errors_exclude_request_builder_errors` |
| 单个 data 前导空格 | `data_field_removes_only_one_leading_space` |
| custom registry 中恢复 built-ins | `ensure_restores_missing_builtins_when_registry_contains_custom_provider` |
| custom 覆盖 built-in API | `ensure_preserves_custom_override_for_builtin_api` |
| clear 与 stream lookup | `clear_racing_with_stream_lookup_does_not_return_missing_provider_error` |
| 多 event FIFO | `multiple_events_are_emitted_fifo` |
| line/data 跨 chunk | `line_and_data_split_across_chunks_are_reassembled` |
| EOF flush 当前 event | `eof_without_blank_line_flushes_current_event` |

全局 registry mutation 测试共用 test-only lock，且以 drop guard 恢复 built-ins。并发测试在
`ensure_and_get` 的 lifecycle lock 内、`register_enabled()` 后且 lookup 前暂停；clear 线程只
能在释放 hook 后完成。不得使用 sleep 或概率轮询。

SSE 覆盖：

```text
data:  two-leading-spaces
多个 event FIFO
line/data 跨多个 Bytes chunk
上游 EOF 无空行时 flush 当前 event
```

预期 `data` 保留一个空格，且保留已有 CRLF、empty data、多 data 行拼接语义。

- [x] **Step 2：确认失败**

运行：

```bash
cargo test -p ai --features all-providers ensure_restores_missing_builtins_when_registry_contains_custom_provider
cargo test -p ai --features all-providers ensure_preserves_custom_override_for_builtin_api
cargo test -p ai --features all-providers clear_racing_with_stream_lookup_does_not_return_missing_provider_error
```

预期：Registry 回归测试在旧实现上 FAIL。SSE FIFO/chunk/EOF 测试可以为 GREEN；如实记录，不为了
制造 RED 改坏 parser。

- [x] **Step 3：Registry 使用缺失注册与生命周期锁**

在 `api_registry.rs` 中增加 crate-private register-if-absent seam。每个 built-in 注册使用此 seam：

```rust
entries.entry(api).or_insert_with(|| RegisteredProvider { provider, source_id });
```

不增加公开 registry 类型或 API；删除不再需要的 `is_empty` / 初始化一次语义。`ensure()` 每次都在
crate-private lifecycle lock 内执行 built-in register-if-absent。`clear_api_providers()`、
`unregister_api_providers()` 也使用同一锁。`stream` / `stream_simple` 通过内部
`ensure_and_get` 在同一临界区完成 ensure 和 `RegisteredHandle` clone，随后立即释放锁，不覆盖网络
stream 生命周期。

- [x] **Step 4：Retry 去掉 panic 分支**

把 request builder error 归类为 non-retryable，直接返回原错误。不要新增 retry policy 类型。

- [x] **Step 5：SSE trim 修正为一个空格**

把 `trim_start()` 类逻辑改为：

```rust
let data = raw.strip_prefix(' ').unwrap_or(raw);
```

- [x] **Step 6：测试通过**

运行：

```bash
cargo test -p ai --features all-providers providers::register_builtins::tests
cargo test -p ai --features all-providers api_registry::tests
cargo test -p ai --features all-providers stream::tests
cargo test -p ai --features all-providers utils::retry::tests
cargo test -p ai --features all-providers utils::sse::tests
cargo test -p ai --features all-providers
```

预期：PASS。

---

## Task 9：Bedrock SigV4 和 Vertex ADC 接入已有基础设施

**文件：**
- 修改：`crates/ai/src/providers/amazon_bedrock.rs`
- 修改：`crates/ai/src/providers/google_vertex.rs`
- 修改：`crates/ai/src/bedrock_provider.rs`
- 修改：`crates/ai/src/sigv4.rs`
- 修改：`crates/ai/src/vertex_adc.rs`
- 修改：`crates/ai/tests/support/mod.rs`
- 修改：`crates/ai/tests/provider_request_metadata.rs`
- 修改：`crates/ai/tests/provider_terminal_events.rs`

**真实 seam：**
- Bedrock 继续使用 `amazon_bedrock::run()` 构造的 `/model/{id}/converse-stream` 请求；只对该 URL 与最终发送的 JSON bytes 调用现有 `sigv4::sign`。不调用固定为 `/invoke-with-response-stream` 的 `bedrock_provider::invoke_stream`。
- SigV4 从最终 `HeaderMap` 选择稳定 headers：存在的 `content-type`、`host` 与全部 `x-amz-*` 必签，`accept` 可不签；调用方不能覆盖 signer 生成的 host/date/payload hash/session token。
- Bedrock simple reasoning 只识别当前内置且有文档协议的 Claude 与 Nova 2 Lite 模型族；未知模型在发起网络请求前 fail closed。
- Vertex 继续使用 `model.base_url`/区域 host 构造最终模型 URL，并复用 Google provider 的原生 Gemini SSE consumer。内置 `{location}` endpoint 对 regional/global 分别解析；ADC 复用 `vertex_adc` 已有 service-account JSON、RS256 JWT 和 token exchange，并继承 stream abort/timeout。
- 没有新增公开认证类型、认证服务、生产 test hook 或依赖。

**认证与请求矩阵：**

| 路径 | 优先级/行为 | 真实请求测试 |
|---|---|---|
| Bedrock bearer | `options.api_key > AWS_BEARER_TOKEN_BEDROCK > SigV4 env credentials`，bearer 分支不附加 SigV4 headers | `bedrock_prefers_bearer_token_but_accepts_sigv4_creds` |
| Bedrock SigV4 | 对最终 path/body/headers 签名；非 greedy `modelId` label 的 `%3A`/`%2F` wire path 不被 canonical URI double-encode；固定向量覆盖 canonical request hash、SignedHeaders 与 signature，捕获测试覆盖最终发送 headers | `sigv4::tests::signs_post_with_payload_and_session_token`、`bedrock_prefers_bearer_token_but_accepts_sigv4_creds`、`bedrock_sigv4_uses_inference_profile_arn_region` |
| Bedrock reasoning | Claude 4.6/4.7 使用 adaptive `thinking`/`output_config`，旧 Claude 使用严格小于 `maxTokens` 的 fixed budget，Nova 2 Lite 使用 `reasoningConfig`；未知模型 fail closed | `bedrock_simple_reasoning_*`、`bedrock_stream_simple_rejects_unknown_claude_protocol` |
| Bedrock signing region | ARN 与标准 runtime endpoint 优先于 env/default；二者冲突时本地报错 | `bedrock_sigv4_uses_inference_profile_arn_region`、`bedrock_signing_region_*` |
| Bedrock 缺认证 | 发具名 Error，连接本地 listener 前返回 | `bedrock_missing_auth_emits_named_error_without_network_request` |
| Vertex token | `options.api_key > GOOGLE_VERTEX_ACCESS_TOKEN > ADC` | `vertex_auth_priority_is_options_then_env_then_adc`、`vertex_can_select_adc_when_access_token_absent` |
| Vertex project | `GOOGLE_VERTEX_PROJECT > service-account project_id` | `vertex_explicit_project_overrides_service_account_project`、`vertex_can_select_adc_when_access_token_absent` |
| Vertex ADC | 本地 token URI 捕获 `grant_type`/JWT assertion，再以返回 token 请求本地 Vertex URL，并断言最终 Gemini JSON body | `vertex_can_select_adc_when_access_token_absent` |
| Vertex endpoint | 内置 `{location}` endpoint 同时覆盖 regional 与 global URL | `builtin_model_url_resolves_regional_and_global_location` |
| Vertex ADC control | token exchange 的 send/body 阶段继承 `abort` 与 `timeout_ms` | `vertex_adc_token_exchange_honors_abort`、`vertex_adc_token_exchange_honors_timeout_ms` |
| Vertex terminal usage | 复用 Gemini terminal event，规范化 input/output/cache/total 并精确结算各 cost 分量 | `vertex_wrapper_done_usage_has_nonzero_cost` |

- [x] **Step 1：三条必需测试先 RED**

分别运行三条具名测试；确认失败原因是缺少 SigV4 provider 接线、缺少 `translate_simple_options`、缺少 ADC provider 接线，而不是测试编译或 fixture 假失败。

- `bedrock_prefers_bearer_token_but_accepts_sigv4_creds`：RED 为 provider 返回 “SigV4 signing not yet implemented”。
- `bedrock_stream_simple_reasoning_sets_additional_model_fields`：RED 为 `translate_simple_options` 不存在。
- `vertex_can_select_adc_when_access_token_absent`：RED 为 provider 返回 “ADC/JWT flow not yet implemented”。

- [x] **Step 2：Bedrock SigV4 与 simple reasoning GREEN**

- bearer 存在时不读取/使用 AWS key；无 bearer 时从 `BedrockCreds::from_env()` 取 access/secret/session/region。
- 先把 `build_request_body()` 序列化为 bytes，再用相同 bytes 签名并发送；测试对捕获 body 重算 SHA-256，并用捕获的 URL/date/body/session token 重新签名后精确比较 authorization。
- `translate_simple_options()` 保留 base options，并按真实模型 ID 写入私有 `bedrock_additional_model_request_fields`：adaptive Claude 使用 `thinking`/`output_config`，旧 Claude 使用 fixed-budget `thinking`，Nova 2 Lite 使用 `reasoningConfig`。
- fixed-budget Claude 保留显式 `maxTokens`，budget 最低 1024 且严格小于 `maxTokens`；无法形成合法组合时本地报错。
- `None` 不发送 reasoning；tools、headers、abort 仍保留。

- [x] **Step 3：Vertex ADC、认证优先级与 project fallback GREEN**

- ADC 分支只加载一次 service account，并用现有 JWT exchange 获取 token。
- 显式 env project 优先；缺失时使用同一 service account 的 `project_id`。
- ADC token server 与最终 Vertex model server 都使用本地捕获，不访问真实 Google endpoint。
- 最终请求断言 bearer、project/location/model URL、model/options headers、Gemini JSON body 与原生 Gemini Done。
- 内置 `{location}` base URL 对 regional location 替换区域，对 `global` 使用无区域 host；自定义 base URL 仍保留。
- ADC token exchange 使用与最终请求相同的 `abort`，并用 `timeout_ms` 覆盖原 15 秒默认超时。

- [x] **Step 4：公开错误与脱敏 GREEN**

以下错误全部通过 public `AssistantMessageEvent::Error` 验证，并断言消息不包含 private key、JWT assertion、access token 或 token endpoint response body：

- `vertex_adc_file_error_is_public_and_redacted`
- `vertex_adc_json_error_is_public_and_redacted`
- `vertex_adc_jwt_error_is_public_and_redacted`
- `vertex_adc_token_http_error_is_public_and_redacted`
- `vertex_adc_token_response_error_is_public_and_redacted`

- [x] **Step 5：环境测试隔离与 Task 5 usage 遗留 GREEN**

- integration tests 共用 `support::env_lock()`。
- 所有环境变量修改通过 panic-safe `EnvVarGuard` 恢复；移除旧的手工恢复 env 单测。
- `vertex_wrapper_done_usage_has_nonzero_cost` 断言 normalized usage、`total_tokens` 及 input/output/cache/total 的精确 cost。

- [x] **Step 6：五项独立审查修复 RED/GREEN**

- tests-only detached `2cc9bda` 上，最终 SignedHeaders 断言 RED；当前实现固定向量与捕获请求均 GREEN。
- `maxTokens=4096` 时旧草稿错误放大为 12288，ARN `us-west-2` 仍使用 env `eu-central-1`；修复后 reasoning budget/上限和 ARN/endpoint/env 决策均 GREEN。
- tests-only detached `2cc9bda` 上，Vertex 内置 URL helper 不存在，ADC abort/timeout 都超过 500ms 门限；修复后 regional/global URL 与两个控制流测试均 GREEN。
- 删除重复 unit assertions、单次调用的 Nova-high helper，以及 hanging capture server 中重复的请求读取代码；没有新增依赖、文件、公开 auth 类型或生产 test hook。

- [x] **Step 7：最终验证**

运行：

```bash
cargo test -p ai --features amazon-bedrock,google-vertex --test provider_request_metadata
cargo test -p ai --features amazon-bedrock,google-vertex --lib providers::amazon_bedrock::tests
cargo test -p ai --features amazon-bedrock,google-vertex --test provider_terminal_events
cargo test -p ai --features amazon-bedrock,google-vertex --lib sigv4
cargo test -p ai --features amazon-bedrock,google-vertex --lib vertex_adc
cargo test -p ai --no-default-features --features amazon-bedrock --lib --test provider_request_metadata --test provider_terminal_events
cargo test -p ai --no-default-features --features google-vertex --lib --test provider_request_metadata --test provider_terminal_events
cargo test -p ai --features amazon-bedrock,google-vertex
cargo test -p ai --features all-providers
cargo fmt -p ai -- --check
rustfmt --edition 2024 --check crates/ai/src/bedrock_provider.rs crates/ai/src/providers/amazon_bedrock.rs crates/ai/src/providers/google_vertex.rs crates/ai/src/vertex_adc.rs crates/ai/tests/provider_request_metadata.rs crates/ai/tests/provider_terminal_events.rs crates/ai/tests/support/mod.rs
git diff --check
```

结果：全部 PASS。独立 Bedrock 为 76 个 lib、9 个 request metadata、1 个 terminal test；独立 Vertex 为 75+10+1。组合 feature 共 132 tests，`all-providers` 共 197 tests，全部 0 failed；最小 feature 仅显示其他模块已有的 cfg/unused warning，`all-providers` 无 warning。

- [x] **Step 8：第二轮复审窄修复 RED/GREEN**

- RED：公开 Bedrock ARN 捕获测试先看到 raw `:` 与 `/`；切换为结构化 URL 后，wire path 已为 `%3A`/`%2F`，但按实际 request-target 重算的 signature 仍不一致，确认 signer 将既有 `%XX` 再编码为 `%25XX`。
- RED：SageMaker endpoint ARN 未覆盖 env region，公开 stream 的 authorization 仍使用 `eu-central-1`；SageMaker ARN 与标准 `eu-central-1` endpoint 冲突时继续尝试 HTTP，而非本地拒绝。
- GREEN：`Url::path_segments_mut()` 隔离 model ID 中的 `/`；`percent_encoding` 只补 `:` 编码，`Url::set_path()` 保留既有 escapes；发送与签名复用同一个 `Url`。SigV4 canonical URI 保留并大写合法 `%XX`，现有固定向量改为 ARN encoded path，防止 double-encode 回归。
- GREEN：现有 ARN region 解析同时接受 `bedrock` 与 `sagemaker` service，不新增类型；公开测试覆盖 SageMaker region 推导与 endpoint 冲突。
- ADC response-body abort/timeout 新测试使用 `200`、较长 `Content-Length` 和挂起 body；两条在生产基线直接 GREEN，证明 `response_text_or_abort` 与 reqwest client timeout 已覆盖 body 阶段，因此不改 `vertex_adc.rs`。
- 删除 integration test 中重复实现的 HMAC/SigV4 辅助，复用生产 signer；两个 hanging mock 复用同一私有 listener 实现。删除 Vertex regional/global 已完成后的过时 TODO。

- [x] **Step 9：第二轮复审最终验证**

运行原 Step 7 全部命令，并确认新增 ARN/SageMaker、ADC response-body 测试、独立 feature、组合 feature、`all-providers`、fmt 与 diff check 全部通过。

结果：全部 PASS。独立 Bedrock 为 76 个 lib、11 个 request metadata、1 个 terminal test；独立 Vertex 为 75+12+1。组合 feature 共 136 tests，`all-providers` 共 201 tests，全部 0 failed；`all-providers` 无 warning，最小 feature 仅显示既有 cfg/unused warning。`cargo fmt -p ai -- --check`、显式 `rustfmt --check` 与 `git diff --check` 均 PASS。

---

## Task 10：README 和最终验证

**文件：**
- 修改：`crates/ai/README.md`

**行为：**
- README 状态表与实现一致。
- 留下完整本地验证记录。

- [x] **Step 1：更新 README 状态**

更新这些行，措辞以最终代码为准：

```markdown
| Cloudflare | working through OpenAI/Anthropic-compatible providers; base-url placeholders resolved from `CLOUDFLARE_ACCOUNT_ID` / `CLOUDFLARE_GATEWAY_ID` |
| Cost accounting | working for streamed usage across built-in providers; `Usage.cost` populated from `Model.cost` |
| Bedrock | Converse Stream with Bearer token or SigV4 env credentials |
| Vertex | access token or service-account ADC via `GOOGLE_APPLICATION_CREDENTIALS` |
| Cross-provider transform_messages | implemented helper; providers use local converters, global handoff remains caller-owned |
```

- [x] **Step 2：完整 ai 验证**

运行：

```bash
cargo test -p ai --features all-providers
```

预期：PASS。

- [x] **Step 3：agent consumer 验证**

运行：

```bash
cargo test -p knuth-agent
```

预期：PASS。

- [x] **Step 4：workspace 编译门禁**

运行：

```bash
cargo test --workspace --no-run
```

预期：PASS。

- [x] **Step 5：最终状态检查**

运行：

```bash
git status --short
```

预期：只出现本计划涉及的文件。

**结果：**

- `cargo test -p ai --features all-providers`：PASS（201 tests）。
- `cargo test -p knuth-agent`：PASS（3 tests）。验证过程中发现基线测试仍等待已不再发送的 `TurnMessage::Finished`；已改为验证真实的 `ModelStepStarted` / `ModelStepEnded` actor contract，并通过独立复审。
- `cargo test --workspace --no-run`：PASS。
- `cargo fmt -p ai -- --check` 与 `git diff --check`：PASS。
- `cargo fmt --all -- --check`：仍被本分支基线已有的 `knuth-agent`、`knuth-cli`、`knuth-core` 格式差异阻断；本计划没有扩大到无关生产文件的格式化。
- tracked worktree：clean；`.superpowers/sdd` 仅保留 ignored 执行记录。

---

## 必须覆盖的回归矩阵

| 问题 | 必须测试 |
|---|---|
| Anthropic 截断流被当作 Done | `anthropic_eof_before_message_stop_is_error` |
| OpenAI completions 截断流被当作 Done | `openai_completions_eof_before_done_or_finish_reason_is_error` |
| Mistral 截断流被当作 Done | `mistral_eof_before_done_or_finish_reason_is_error` |
| Google 截断流被当作 Done | `google_eof_before_finish_reason_is_error` |
| Bedrock 截断流被当作 Done | `bedrock_eof_before_message_stop_is_error` |
| OpenAI Responses 保持 EOF error | `openai_responses_eof_before_terminal_event_is_error` |
| Mistral 并行 tool calls 被合并 | `mistral_parallel_tool_calls_do_not_merge_arguments` |
| Mistral 并行 tool-call 事件错用 content index 或丢失 id normalization | `mistral_parallel_tool_call_events_preserve_content_indices` |
| `model.headers` 被忽略或不能替换 provider 默认值 | `openai_completions_merges_model_headers_then_options_headers`、`model_headers_override_provider_defaults`、`options_headers_override_provider_defaults`，以及 header helper unit coverage |
| header 大小写覆盖或非法值处理错误 | `options_headers_override_model_headers_case_insensitively`、`invalid_custom_header_emits_error` |
| Cloudflare Completions placeholders 未解析 | `cloudflare_placeholders_are_resolved_before_request`、`cloudflare_missing_placeholder_emits_named_error` |
| Cloudflare Anthropic placeholders 未解析 | `cloudflare_anthropic_placeholders_are_resolved_before_request` |
| Cloudflare Responses placeholders 未解析 | `cloudflare_responses_placeholders_are_resolved_before_request` |
| `Usage.cost` 总是 0 | `calculate_usage_cost_uses_per_million_prices`，provider terminal 测试断言 usage 有价格时 cost 非 0 |
| OpenAI cached tokens 重复计入 input | `usage_subtracts_cached_tokens_from_openai_prompt_tokens` |
| Responses cached tokens 重复计入 input | `responses_usage_subtracts_cached_tokens_from_input_tokens` |
| Mistral cached prompt tokens 重复计入 input | `mistral_usage_subtracts_cached_prompt_tokens` |
| Google tool-use prompt tokens 未计入 input | `google_usage_includes_tool_use_prompt_tokens` |
| Responses cache-write tokens 重复计入 input | `responses_usage_subtracts_cache_write_tokens` |
| Responses terminal usage snapshot 被重复累加 | `responses_repeated_usage_snapshot_does_not_accumulate` |
| Google `total_tokens` 信任 provider 原值而偏离统一语义 | `usage_total_tokens_uses_normalized_components` |
| provider 终态未结算 usage cost | `openai_completions_done_usage_has_nonzero_cost` |
| Responses 共享 consumer 终态未结算 usage cost | `openai_responses_done_usage_has_nonzero_cost` |
| Azure Responses wrapper 真实入口未覆盖 terminal usage/cost | `azure_openai_responses_wrapper_done_usage_has_nonzero_cost` |
| Faux replay 未规范化 total 或未按调用模型价格重算 cost | `replayed_usage_normalizes_total_tokens_and_cost` |
| Bedrock-Anthropic 丢失 message_start usage 或 delta 缺失字段覆盖已有值 | `bedrock_anthropic_message_start_usage_combines_with_delta_output_and_prices_done` |
| Bedrock-Anthropic terminal Error 未结算 cost | `bedrock_anthropic_error_terminal_calculates_usage_cost` |
| Codex `max_tokens` 被忽略或缺省时误发送 | `codex_request_includes_max_output_tokens`、`codex_request_omits_max_output_tokens_by_default` |
| Codex encrypted reasoning 没 replay | `codex_replays_encrypted_reasoning_items` |
| Codex Responses wrapper 真实入口未覆盖 terminal usage/cost | `codex_responses_wrapper_done_usage_has_nonzero_cost` |
| Responses done event 按最后 block 路由 | `text_done_routes_by_output_index_not_last_block`、`reasoning_done_routes_by_output_index_not_last_same_type_block` |
| Responses text part 未建立 lifecycle、tuple 路由混淆或 done 未回写终态 | `content_part_added_then_text_done_without_delta_preserves_lifecycle`、`text_done_updates_partial_and_terminal_content`、`interleaved_text_parts_route_start_delta_end_by_tuple_identity` |
| Responses `output_item.done` 重复发出 block End 或快照不一致 | `reasoning_output_item_done_emits_one_thinking_end_with_matching_partial`、`function_call_output_item_done_emits_one_tool_call_end_with_matching_partial` |
| Responses 无 identity 的 reasoning delta 创建孤立 block | `identityless_reasoning_delta_does_not_create_orphan_block` |
| `clear_api_providers()` 后 built-ins 失效 | `stream_re_registers_builtins_after_clear` |
| Registry 非空但 built-ins 缺失 | `ensure_restores_missing_builtins_when_registry_contains_custom_provider` |
| custom same-API override | `ensure_preserves_custom_override_for_builtin_api` |
| clear/stream ensure+lookup race | `clear_racing_with_stream_lookup_does_not_return_missing_provider_error` |
| retry 防御分支可能 panic | `retryable_reqwest_errors_exclude_request_builder_errors` |
| SSE 吞掉过多前导空格 | `data_field_removes_only_one_leading_space` |
| SSE multiple event FIFO | `multiple_events_are_emitted_fifo` |
| SSE line/data chunk reassembly | `line_and_data_split_across_chunks_are_reassembled` |
| SSE EOF flush | `eof_without_blank_line_flushes_current_event` |
| Bedrock simple stream 丢 reasoning 或 base options | `bedrock_stream_simple_reasoning_sets_additional_model_fields`、`bedrock_stream_simple_preserves_base_options_and_omits_absent_reasoning` |
| Bedrock bearer/SigV4 优先级错误，或 converse-stream authorization 未绑定实际 path/body/date/session token | `bedrock_prefers_bearer_token_but_accepts_sigv4_creds` |
| Bedrock 缺认证仍发网络请求 | `bedrock_missing_auth_emits_named_error_without_network_request` |
| Vertex token/ADC 优先级及本地 ADC exchange 未接入最终 Vertex bearer 请求 | `vertex_auth_priority_is_options_then_env_then_adc`、`vertex_can_select_adc_when_access_token_absent` |
| Vertex project 未回退或显式 env 未优先 | `vertex_can_select_adc_when_access_token_absent`、`vertex_explicit_project_overrides_service_account_project` |
| Vertex 最终 URL、headers 或 Gemini JSON body 错误 | `vertex_can_select_adc_when_access_token_absent` |
| Vertex ADC 错误缺 public Error 上下文或泄露凭据 | `vertex_adc_file_error_is_public_and_redacted`、`vertex_adc_json_error_is_public_and_redacted`、`vertex_adc_jwt_error_is_public_and_redacted`、`vertex_adc_token_http_error_is_public_and_redacted`、`vertex_adc_token_response_error_is_public_and_redacted` |
| Vertex wrapper terminal usage 或 input/output/cache/total cost 错误 | `vertex_wrapper_done_usage_has_nonzero_cost` |
| Task 9 环境变量测试并发污染或 panic 后不恢复 | `provider_request_metadata` 与 `vertex_wrapper_done_usage_has_nonzero_cost` 共用 `support::env_lock()` + `EnvVarGuard` |
| README provider 状态过期 | README diff 加最终验证命令 |

## 自检

- 覆盖性：审查中确认的真实问题都落到了任务和覆盖矩阵。
- YAGNI：没有新增公开 provider 抽象；新增内容限于内部 helper 和防回归测试。
- 可执行性：每个任务都有失败测试、修复位置、通过命令。
- Rust 2024：涉及环境变量 mutation 的测试使用 `unsafe`，并用测试锁避免并发污染。
- 完成标准：`cargo test -p ai --features all-providers`、`cargo test -p knuth-agent`、`cargo test --workspace --no-run` 全部通过后才算完成。


---

## Final multi-axis review：Batch 1-3 与 follow-up

### 最终实际改动补充（不替换 Task 1-10 原文件清单）

- crates/ai/src/bedrock_anthropic.rs、crates/ai/src/providers/cloudflare.rs、crates/ai/src/providers/faux.rs、crates/ai/src/providers/google_shared.rs、crates/ai/tests/support_http.rs 纳入最终实际改动范围。
- Task 1-10 原有条目保持不删除；本次仅更新本文件。Batch 3 执行报告保持 ignored，不进入 tracked commit。

### Batch 记录

- [x] **Batch 1**：dc7e6a1、2693b1a。provider infrastructure、credential safety、Cloudflare/retry/SSE 与 Bedrock SigV4 request wiring。
- [x] **Batch 2**：bcee056、49b3e87。Bedrock/Google/Mistral provider fidelity、reasoning/tool streaming 与 request replay。
- [x] **Batch 3**：91245e5。Responses identity/lifecycle、usage normalization、README、strict clippy；service_tier 仅作为请求 metadata，usage cost 始终按静态 Model.cost catalog 计算。
- [x] **Batch 3 follow-up**：c783bc8。generic response.content_part.done 在 incomplete/cancelled terminal 前完成 text/refusal lifecycle；Codex encrypted reasoning replay 拒绝缺失或空 id。

### 新增回归类别

- generic content-part completion、incomplete/cancelled terminal 前 lifecycle、Codex wrapper 公开入口；mixed encrypted-reasoning replay 过滤。
- 具名测试：openai_responses_generic_content_part_done_ends_before_terminal、codex_responses_wrapper_reuses_generic_content_part_done_lifecycle、codex_replays_only_well_formed_encrypted_reasoning_in_mixed_history。

### 当前实测回归矩阵

| 命令 | 当前结果 |
|---|---|
| OpenAI feature | 101/1/4 PASS。 |
| Codex feature | 103/4/3 PASS。 |
| Azure feature | 105/0/1 PASS。 |
| all-providers | 227 PASS。 |
| knuth-agent | 3 PASS。 |
| workspace、clippy、fmt、diff | PASS。 |

### 完成复核

- [x] Task 1-10 标题、全部实施步骤、RED/GREEN 命令、原验收矩阵与已完成状态均从 49b3e87 完整恢复。
- [x] 新增具名测试以 rg 在当前 crates/ai 中核对。
