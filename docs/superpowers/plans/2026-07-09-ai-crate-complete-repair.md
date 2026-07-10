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
  - 本地 HTTP/SSE mock server、请求捕获、provider model 构造器。
  - Bedrock AWS eventstream frame builder。
- 新增：`crates/ai/tests/provider_terminal_events.rs`
  - Anthropic、OpenAI completions、Mistral、Google、Bedrock、OpenAI Responses 的 EOF/终止事件测试。
- 新增：`crates/ai/tests/provider_request_metadata.rs`
  - `model.headers`、`options.headers`、Cloudflare URL、Codex `max_tokens`、Vertex ADC seam 的请求捕获测试。
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
- 修改：`crates/ai/src/providers/openai_responses.rs`
  - cost 计算、cache-aware usage total、done event 按 `output_index`/`item_id` 路由、复用 EOF helper。
- 修改：`crates/ai/src/providers/openai_codex_responses.rs`
  - `max_output_tokens`、encrypted reasoning replay、`model.headers`、cost 计算。
- 修改：`crates/ai/src/providers/google_vertex.rs`
  - `model.headers`、复用 Google consumer 后的 cost 计算、通过现有 `vertex_adc` 做 ADC fallback。
- 修改：`crates/ai/src/providers/register_builtins.rs`
  - 修复 `clear_api_providers()` 后 built-ins 不再注册的问题。
- 修改：`crates/ai/src/api_registry.rs`
  - 增加内部 `is_empty()` 查询；不改公开 API。
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

- [ ] **Step 1：新增 shared mock server**

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

- [ ] **Step 2：支撑代码编译检查**

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

- [ ] **Step 1：新增失败测试**

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

- [ ] **Step 2：确认失败**

运行：

```bash
cargo test -p ai --features anthropic,openai-completions,mistral,google,amazon-bedrock,openai-responses --test provider_terminal_events
```

预期：新测试失败；当前 provider 会在截断 EOF 后发 `Done` 或静默完成。

- [ ] **Step 3：在各 provider stream loop 中记录终止事件**

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

- [ ] **Step 4：终止事件测试通过**

运行：

```bash
cargo test -p ai --features anthropic,openai-completions,mistral,google,amazon-bedrock,openai-responses --test provider_terminal_events
```

预期：PASS。

- [ ] **Step 5：全 provider 回归**

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

- [ ] **Step 1：新增失败测试**

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

- [ ] **Step 2：确认失败**

运行：

```bash
cargo test -p ai --features mistral --test provider_terminal_events mistral_parallel_tool_calls_do_not_merge_arguments
```

预期：FAIL，当前实现会把参数错误合并。

- [ ] **Step 3：按 provider index/id 建立 buffer**

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

- [ ] **Step 4：解析各自 buffer**

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

- [ ] **Step 5：测试通过**

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

- [ ] **Step 1：新增 header 合并失败测试**

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

- [ ] **Step 2：新增 Cloudflare URL 失败测试**

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

- [ ] **Step 3：确认失败**

运行：

```bash
cargo test -p ai --features openai-completions,anthropic,openai-responses,cloudflare --test provider_request_metadata
```

预期：header 测试失败，因为追加语义不能替换 provider 默认值，也不能保证大小写不敏感覆盖；Anthropic / Responses Cloudflare 测试失败，因为 placeholder 没被替换。

- [ ] **Step 4：增加最小内部 helper**

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

- [ ] **Step 5：Cloudflare 复用现有 resolver**

在 `openai_completions.rs`、`anthropic.rs`、`openai_responses.rs` 计算 base URL 时，对 Cloudflare provider 调用现有 Cloudflare resolver。若 env 缺失，返回 `Error` event，错误信息包含缺失变量名。

- [ ] **Step 6：测试通过**

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

- [ ] **Step 1：新增 unit tests**

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

- [ ] **Step 2：确认失败**

运行：

```bash
cargo test -p ai --features all-providers calculate_usage_cost_uses_per_million_prices usage_subtracts_cached_tokens_from_openai_prompt_tokens responses_usage_subtracts_cached_tokens_from_input_tokens
```

预期：FAIL，当前 `Usage.cost` 为 0，cached token 计数重复。

- [ ] **Step 3：增加 helper 并在 provider 终止前调用**

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

- [ ] **Step 4：修正 cached token 计数**

OpenAI completions：
- `prompt_tokens` 包含 cached tokens。
- `prompt_tokens_details.cached_tokens` 计入 `cache_read`。
- `usage.input = prompt_tokens - cached_tokens`，使用 saturating subtraction。

OpenAI Responses：
- `input_tokens` 包含 cached tokens。
- `input_tokens_details.cached_tokens` 计入 `cache_read`。
- `usage.input = input_tokens - cached_tokens`，使用 saturating subtraction。

- [ ] **Step 5：测试通过**

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

- [ ] **Step 1：新增请求捕获测试**

测试名：

```rust
codex_request_includes_max_output_tokens
codex_replays_encrypted_reasoning_items
```

断言：
- 请求 JSON 包含 `"max_output_tokens": 1234`。
- replay 的 encrypted reasoning item 原样出现在 `input` 中。

- [ ] **Step 2：确认失败**

运行：

```bash
cargo test -p ai --features openai-codex-responses --test provider_request_metadata codex_request_includes_max_output_tokens codex_replays_encrypted_reasoning_items
```

预期：FAIL。

- [ ] **Step 3：补 request body**

在构造 Codex Responses body 时：

```rust
if let Some(max_tokens) = options.max_tokens {
    body["max_output_tokens"] = json!(max_tokens);
}
```

replay encrypted reasoning 时只复用已有 `ContentBlock::Thinking`/provider extras 已保存的信息，不新增公开 message 类型。若现有结构无法表达，先在测试中证明缺口，再加最小内部序列化 helper。

- [ ] **Step 4：测试通过**

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

- [ ] **Step 1：新增失败测试**

测试名：

```rust
text_done_routes_by_output_index_not_last_block
```

构造 event 顺序：
1. text block start/index 0。
2. reasoning block start/index 1。
3. text done 指向 index 0。

断言 text done 结束的是 text block，不是最后的 reasoning block。

- [ ] **Step 2：确认失败**

运行：

```bash
cargo test -p ai --features openai-responses text_done_routes_by_output_index_not_last_block
```

预期：FAIL。

- [ ] **Step 3：复用已有 block map**

检查 `openai_responses.rs` 中已有的 index/item tracking。若已有 map，只补 done event lookup。若没有，用最小内部 map：

```rust
HashMap<(u64, u64), usize>
HashMap<String, usize>
```

不要引入新的公开 block identity 类型。

- [ ] **Step 4：测试通过**

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
- 修改：`crates/ai/src/utils/retry.rs`
- 修改：`crates/ai/src/utils/sse.rs`

**行为：**
- `clear_api_providers()` 后，stream 入口能重新注册 built-ins。
- retry 工具不再在 request builder error 上 panic。
- SSE `data:` 只移除规范允许的一个前导空格。

- [ ] **Step 1：新增失败测试**

测试名：

```rust
stream_re_registers_builtins_after_clear
retryable_reqwest_errors_exclude_request_builder_errors
data_field_removes_only_one_leading_space
```

SSE case：

```text
data:  two-leading-spaces
```

预期 data 是 `" two-leading-spaces"`。

- [ ] **Step 2：确认失败**

运行：

```bash
cargo test -p ai --features all-providers stream_re_registers_builtins_after_clear retryable_reqwest_errors_exclude_request_builder_errors data_field_removes_only_one_leading_space
```

预期：FAIL。

- [ ] **Step 3：Registry 只加内部查询**

在 `api_registry.rs` 中增加：

```rust
pub(crate) fn is_empty() -> bool {
    registry().lock().unwrap().is_empty()
}
```

在 built-in ensure 逻辑里：
- `OnceLock` 只表示“曾经初始化过”不够。
- 如果 registry 被 clear 后为空，重新注册 built-ins。

- [ ] **Step 4：Retry 去掉 panic 分支**

把 request builder error 归类为 non-retryable，直接返回原错误。不要新增 retry policy 类型。

- [ ] **Step 5：SSE trim 修正为一个空格**

把 `trim_start()` 类逻辑改为：

```rust
let data = raw.strip_prefix(' ').unwrap_or(raw);
```

- [ ] **Step 6：测试通过**

运行：

```bash
cargo test -p ai --features all-providers stream_re_registers_builtins_after_clear retryable_reqwest_errors_exclude_request_builder_errors data_field_removes_only_one_leading_space
cargo test -p ai --features all-providers
```

预期：PASS。

---

## Task 9：Bedrock SigV4 和 Vertex ADC 接入已有基础设施

**文件：**
- 修改：`crates/ai/src/providers/amazon_bedrock.rs`
- 修改：`crates/ai/src/providers/google_vertex.rs`
- 修改：`crates/ai/src/bedrock_provider.rs`
- 修改：`crates/ai/src/vertex_adc.rs`

**行为：**
- Bedrock 支持 bearer token 或 AWS SigV4 env credentials。
- Vertex 支持 `GOOGLE_VERTEX_ACCESS_TOKEN` 或 service-account ADC exchange。
- 不新增 AWS/Google 依赖。

- [ ] **Step 1：新增 Bedrock SigV4 seam 测试**

```rust
fn env_test_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

#[test]
fn bedrock_prefers_bearer_token_but_accepts_sigv4_creds() {
    let _guard = env_test_lock().lock().unwrap();
    unsafe {
        std::env::remove_var("AWS_BEARER_TOKEN_BEDROCK");
        std::env::set_var("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE");
        std::env::set_var("AWS_SECRET_ACCESS_KEY", "secret");
        std::env::set_var("AWS_REGION", "us-east-1");
    }
    let creds = crate::bedrock_provider::BedrockCreds::from_env();
    assert!(creds.is_some());
    unsafe {
        std::env::remove_var("AWS_ACCESS_KEY_ID");
        std::env::remove_var("AWS_SECRET_ACCESS_KEY");
        std::env::remove_var("AWS_REGION");
    }
}
```

- [ ] **Step 2：新增 Bedrock simple reasoning 测试**

```rust
#[test]
fn bedrock_stream_simple_reasoning_sets_additional_model_fields() {
    let opts = SimpleStreamOptions {
        reasoning: Some(ThinkingLevel::Medium),
        ..Default::default()
    };
    let translated = translate_simple_options(Some(&opts));
    let body = build_request_body(&ctx(), &translated);
    assert_eq!(
        body["additionalModelRequestFields"]["reasoning"]["type"],
        "enabled"
    );
}
```

从 `stream_simple` 中抽出私有 `translate_simple_options()`；返回 `StreamOptions`，并把 simple reasoning 映射到 `build_request_body()` 会序列化的 Bedrock request-body 路径。

- [ ] **Step 3：新增 Vertex ADC fallback 测试**

```rust
#[test]
fn vertex_can_select_adc_when_access_token_absent() {
    let _guard = env_test_lock().lock().unwrap();
    unsafe {
        std::env::remove_var("GOOGLE_VERTEX_ACCESS_TOKEN");
        std::env::set_var("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/service-account.json");
    }
    assert!(vertex_auth_source_for_tests().is_adc());
    unsafe {
        std::env::remove_var("GOOGLE_APPLICATION_CREDENTIALS");
    }
}
```

`vertex_auth_source_for_tests()` 只放在 test module 内，不新增公开 API。

- [ ] **Step 4：确认失败**

运行：

```bash
cargo test -p ai --features amazon-bedrock,google-vertex bedrock_prefers_bearer_token_but_accepts_sigv4_creds bedrock_stream_simple_reasoning_sets_additional_model_fields vertex_can_select_adc_when_access_token_absent
```

预期：Bedrock low-level creds seam 可能已通过，但 provider integration 仍会报 “SigV4 not yet implemented”；Vertex 只认 access token，因此失败。

- [ ] **Step 5：Bedrock SigV4 fallback**

保留 bearer token 路径。缺 bearer token 时：

```rust
let creds = match crate::bedrock_provider::BedrockCreds::from_env() {
    Some(creds) => creds,
    None => {
        push_error(&mut sender, &model, "Bedrock auth missing: set AWS_BEARER_TOKEN_BEDROCK or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY".into());
        return;
    }
};
```

使用现有 `crate::sigv4::sign` 签名 Converse Stream URL 和 JSON body。不要改走 `bedrock_provider::invoke_stream`，因为它针对 `/invoke-with-response-stream`，而当前 provider 使用 `/converse-stream`。

- [ ] **Step 6：Vertex ADC fallback**

token 解析顺序：

```rust
let token = if let Some(t) = options.api_key.clone().filter(|t| !t.is_empty()) {
    t
} else if let Some(t) = std::env::var("GOOGLE_VERTEX_ACCESS_TOKEN").ok().filter(|t| !t.is_empty()) {
    t
} else {
    match crate::vertex_adc::fetch_access_token(None).await {
        Ok(t) => t.token,
        Err(e) => {
            push_error(&mut sender, &model, format!("Vertex ADC auth failed: {e}"));
            return;
        }
    }
};
```

如果 service account 带 `project_id` 且 `GOOGLE_VERTEX_PROJECT` 未设置，使用该 project id。

- [ ] **Step 7：测试通过**

运行：

```bash
cargo test -p ai --features amazon-bedrock,google-vertex bedrock_prefers_bearer_token_but_accepts_sigv4_creds bedrock_stream_simple_reasoning_sets_additional_model_fields vertex_can_select_adc_when_access_token_absent
cargo test -p ai --features all-providers
```

预期：PASS。

---

## Task 10：README 和最终验证

**文件：**
- 修改：`crates/ai/README.md`

**行为：**
- README 状态表与实现一致。
- 留下完整本地验证记录。

- [ ] **Step 1：更新 README 状态**

更新这些行，措辞以最终代码为准：

```markdown
| Cloudflare | working through OpenAI/Anthropic-compatible providers; base-url placeholders resolved from `CLOUDFLARE_ACCOUNT_ID` / `CLOUDFLARE_GATEWAY_ID` |
| Cost accounting | working for streamed usage across built-in providers; `Usage.cost` populated from `Model.cost` |
| Bedrock | Converse Stream with Bearer token or SigV4 env credentials |
| Vertex | access token or service-account ADC via `GOOGLE_APPLICATION_CREDENTIALS` |
| Cross-provider transform_messages | implemented helper; providers use local converters, global handoff remains caller-owned |
```

- [ ] **Step 2：完整 ai 验证**

运行：

```bash
cargo test -p ai --features all-providers
```

预期：PASS。

- [ ] **Step 3：agent consumer 验证**

运行：

```bash
cargo test -p knuth-agent
```

预期：PASS。

- [ ] **Step 4：workspace 编译门禁**

运行：

```bash
cargo test --workspace --no-run
```

预期：PASS。

- [ ] **Step 5：最终状态检查**

运行：

```bash
git status --short
```

预期：只出现本计划涉及的文件。

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
| Codex `max_tokens` 被忽略 | `codex_request_includes_max_output_tokens` |
| Codex encrypted reasoning 没 replay | `codex_replays_encrypted_reasoning_items` |
| Responses done event 按最后 block 路由 | `text_done_routes_by_output_index_not_last_block` |
| `clear_api_providers()` 后 built-ins 失效 | `stream_re_registers_builtins_after_clear` |
| retry 防御分支可能 panic | `retryable_reqwest_errors_exclude_request_builder_errors` |
| SSE 吞掉过多前导空格 | `data_field_removes_only_one_leading_space` |
| Bedrock simple stream 丢 reasoning | `bedrock_stream_simple_reasoning_sets_additional_model_fields` |
| Bedrock SigV4 有基础设施但未接入 | `bedrock_prefers_bearer_token_but_accepts_sigv4_creds`，本地 mock 可验证 `authorization: AWS4-HMAC-SHA256` 时增加 integration 断言 |
| Vertex ADC 有基础设施但未接入 | `vertex_can_select_adc_when_access_token_absent` |
| README provider 状态过期 | README diff 加最终验证命令 |

## 自检

- 覆盖性：审查中确认的真实问题都落到了任务和覆盖矩阵。
- YAGNI：没有新增公开 provider 抽象；新增内容限于内部 helper 和防回归测试。
- 可执行性：每个任务都有失败测试、修复位置、通过命令。
- Rust 2024：涉及环境变量 mutation 的测试使用 `unsafe`，并用测试锁避免并发污染。
- 完成标准：`cargo test -p ai --features all-providers`、`cargo test -p knuth-agent`、`cargo test --workspace --no-run` 全部通过后才算完成。
