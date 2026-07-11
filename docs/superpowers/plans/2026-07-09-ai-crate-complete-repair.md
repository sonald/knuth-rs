# AI Crate 完整修复与最终复审计划

**目标：** 在不增加公开 provider/领域类型的前提下，修复已确认的 `crates/ai`
provider 正确性问题，并以真实 provider 入口测试证明每项行为闭环。

**历史参考：** 下列 Task 1-10 结构恢复自
`49b3e87:docs/superpowers/plans/2026-07-09-ai-crate-complete-repair.md`。所有完成状态保留
已实施的 Batch 1-3 工作；后续复审发现作为追加记录，而非替换原计划。

## 全局约束

- 遵守仓库要求：如无必要，勿增实体。
- 每个生产行为修复先有失败的回归测试。
- 保持公开 `Model`、`StreamOptions`、`Usage`、`AssistantMessageEvent`、`ApiProvider`
  契约不变。
- provider 特有逻辑留在现有模块，复用既有测试支撑；不新增依赖或公开抽象。
- 同时运行独立 provider feature 和 `all-providers`。

## 本次窄修复实际文件清单

本次 Batch 3 follow-up 实际只修改以下文件：

| 文件 | 实际改动 |
|---|---|
| `crates/ai/src/providers/openai_responses.rs` | 消费通用 `response.content_part.done`；只完成一次 text/refusal reconciliation；将 `response.cancelled` 映射到既有 aborted 终态。 |
| `crates/ai/src/providers/openai_codex_responses.rs` | encrypted reasoning replay 要求非空 reasoning item id。 |
| `crates/ai/tests/provider_terminal_events.rs` | 为 OpenAI/Codex 共享 consumer 增加 generic part completion 位于 incomplete/cancelled 之前的公开测试。 |
| `crates/ai/tests/provider_request_metadata.rs` | Codex mixed replay fixture 补充缺失/空 reasoning id。 |
| `docs/superpowers/plans/2026-07-09-ai-crate-complete-repair.md` | 恢复 Task 1-10、复审可追溯记录和实测回归矩阵。 |
| `.superpowers/sdd/final-review-batch-3.md` | ignored 的 Batch 3 RED/GREEN 执行记录。 |

## Task 1：建立 Provider 级 HTTP/Stream 回归测试支撑

- [x] 在 `crates/ai/tests/support/mod.rs` 建立共享本地 HTTP/SSE mock 支撑。
- [x] 增加请求捕获、环境隔离、provider model 构造和 Bedrock eventstream frame，且不增加依赖。
- [x] 以这些支撑覆盖公开 provider 路径，而不是新增 provider 私有 fake。

可追溯证据：`support::serve_once`、`support::serve_capture_once`、`support::env_lock()`
支撑 request metadata 与 terminal event 测试。

## Task 2：EOF 必须是 Error，除非收到 Provider 原生终止事件

- [x] Anthropic、OpenAI Completions、Mistral、Google、Bedrock、Responses 的截断流都产生 Error，
  不再静默完成。
- [x] 每个 provider 仅将原生终止事件作为正常 Done 边界。
- [x] 在 `provider_terminal_events` 覆盖 EOF-before-terminal。

可追溯证据：`*_eof_before_*_is_error` 和 Responses EOF 公开 fixture。

## Task 3：Mistral 并行 Tool Calls 不能合并

- [x] 按 tool-call identity/index 累积 Mistral arguments，不再将并行 call 合并进一个 JSON buffer。
- [x] 保持规范化 tool id 及每个 call 自己的 Start/Delta/End content index。
- [x] 覆盖同 chunk 与跨 chunk 的并行 call。

可追溯证据：`mistral_parallel_tool_calls_do_not_merge_arguments`、
`mistral_parallel_tool_call_events_preserve_content_indices`。

## Task 4：`model.headers` 和 Cloudflare Base URL 必须进入真实请求

- [x] 按 provider default、`model.headers`、`options.headers` 的顺序合并，并以大小写无关规则覆盖。
- [x] 非法 custom header 经公开 Error event 返回。
- [x] 所有兼容 provider 在构建 URL 前解析 Cloudflare account/gateway placeholder。

可追溯证据：Completions、Anthropic、Responses 的 request capture Cloudflare 覆盖。

## Task 5：Usage 语义和 Cost 计算

- [x] 先规范化 input/output/cache-read/cache-write token，再计算 cost。
- [x] 防止重复 terminal snapshot 与 cached token 的重复计数。
- [x] provider terminal message 使用调用时静态 `Model.cost` 目录价；适用的 Error terminal 同样结算。

可追溯证据：`calculate_usage_cost_uses_per_million_prices`、Responses/Vertex/Bedrock
terminal usage 测试及 normalized total-token 回归测试。

## Task 6：Codex Responses 请求行为补齐

- [x] 只在提供时发送 `max_output_tokens`，并保持 Codex request metadata。
- [x] encrypted reasoning replay 仅接受可解析为 reasoning item、具有非空 string `id` 和 string
  `encrypted_content` 的既有签名契约。
- [x] 保持既有 string-content 契约：空 encrypted string 仍是合法 string；malformed、缺失或
  非 string content 不回放。
- [x] 删除无效 thinking 时保持 text/tool/user 的相对顺序。

可追溯证据：`codex_request_includes_max_output_tokens`、
`codex_request_omits_max_output_tokens_by_default`、`codex_replays_encrypted_reasoning_items`、
`codex_replays_only_well_formed_encrypted_reasoning_in_mixed_history`。

## Task 7：OpenAI Responses Done Event 必须按 Block Identity 路由

- [x] text/reasoning 完成按既有 output/item/content identity 路由，而非按最后一个同类 block。
- [x] 用 `TextSignatureV1` 保存 message id/phase，供 store:false replay 保持
  reasoning/message/function-call 原始顺序。
- [x] 支持 `output_text`、`refusal` 和通用 `content_part.done` 完成。
- [x] 用最终 `text`/`refusal` 回写 block，且只发出一次 `TextEnd`。
- [x] `response.incomplete` 为 Length Done；`response.cancelled` 为既有 Aborted Error，
  且二者之前的 content lifecycle 必须已经完整。

可追溯证据：`text_done_routes_by_output_index_not_last_block`、
`interleaved_text_parts_route_start_delta_end_by_tuple_identity`、
`openai_responses_refusal_uses_complete_text_lifecycle`、
`openai_responses_generic_content_part_done_ends_before_terminal` 及等价 Codex wrapper 测试。

## Task 8：Registry、Retry、SSE 的边界修复

- [x] registry clear 后恢复缺失 built-in，同时不覆盖 custom same-API provider。
- [x] 关闭 ensure-and-get lifecycle race。
- [x] 收窄 retryable request error，移除会 panic 的防御分支。
- [x] 保留 SSE data 空格、FIFO、chunk reassembly 与 EOF event flush 语义。

可追溯证据：registry lifecycle/race 和 SSE parser 回归套件。

## Task 9：Bedrock SigV4 和 Vertex ADC 接入已有基础设施

- [x] 以最终 wire URL/body/headers 签名 Bedrock Converse request，并从支持的 ARN/endpoint
  安全推导 region。
- [x] 保持 bearer/SigV4 认证选择、reasoning request translation 和缺认证本地 fail-closed 行为。
- [x] 将 Vertex access-token/ADC 优先级、project fallback、regional/global URL、脱敏错误、
  abort/timeout 接入既有 request 基础设施。

可追溯证据：Bedrock request capture/fixed signer vector、reasoning 测试和 Vertex ADC
public stream 测试。

## Task 10：README 和最终验证

- [x] README provider 状态和静态 cost/service-tier 契约与实现一致。
- [x] 完成 provider、consumer、workspace、格式、lint 和 diff 检查。
- [x] ignored SDD 执行记录不进入 tracked commit。

## 最终复审：Batch 1-3

| Batch | 发现与提交修复 | 主要回归证据 |
|---|---|---|
| Batch 1 | provider 基础设施、EOF/Error、headers/Cloudflare、usage、registry、retry/SSE。 | provider terminal/request 套件和 `all-providers`。 |
| Batch 2 | Google/Vertex/Bedrock signature、terminal reason、Mistral reasoning/tool streaming fidelity。 | provider round trip、terminal-reason matrix、独立 feature。 |
| Batch 3 | Responses refusal lifecycle、store:false identity/order、Codex replay filter、converter compatibility、严格 clippy。提交 `91245e5`（`fix(ai): close Responses API review`）。 | Responses public consumer、request replay、converter、严格 clippy。 |
| Batch 3 follow-up | generic `response.content_part.done` 在 incomplete/cancelled 前完成 output text/refusal；Codex 拒绝缺失/空 encrypted-reasoning id。 | 新增 OpenAI/Codex 共享 public lifecycle 测试和 expanded mixed replay fixture。 |

### Batch 3 Follow-up：RED 到 GREEN

- RED：generic `response.content_part.done` 没有建立或结束 Text lifecycle。OpenAI public fixture
  的 final text 为 `None`，terminal event 可能掩盖缺失的 `TextEnd`。
- GREEN：共享 consumer 按 part type 和既有 output/content identity 建立或路由 Text block，
  以最终 `text`/`refusal` reconciliation，并抑制重复 `TextEnd`。OpenAI 和 Codex wrapper 都断言
  incomplete/cancelled 时的 `TextStart -> TextEnd -> terminal` 顺序。
- RED：Codex mixed history replay 会接纳缺失 id 或 `id: ""` 的 `reasoning` signature。
- GREEN：replay 必须有非空 string id；mixed fixture 证明只剩一个合法 encrypted reasoning item，
  text/tool/user 顺序不变。

## 实测回归矩阵

下列独立 feature 计数均由实际命令取得，而非从复审基线
`101/1/3`、`103/4/2`、`105/0/1` 手算：

| 命令 | 当前结果 |
|---|---|
| `cargo test -q -p ai --no-default-features --features openai-responses --lib --test provider_request_metadata --test provider_terminal_events` | 101 lib、1 request metadata、4 terminal，全部 PASS。 |
| `cargo test -q -p ai --no-default-features --features openai-codex-responses --lib --test provider_request_metadata --test provider_terminal_events` | 103 lib、4 request metadata、3 terminal，全部 PASS。 |
| `cargo test -q -p ai --no-default-features --features azure-openai-responses --lib --test provider_request_metadata --test provider_terminal_events` | 105 lib、0 request metadata、1 terminal，全部 PASS。 |
| `cargo test -p ai --features all-providers` | 227 tests PASS：153 lib、7 Anthropic E2E、4 catalog、41 request metadata、21 terminal、1 support。 |
| `cargo test -p knuth-agent` | 3 tests PASS。 |
| `cargo test --workspace --no-run` | PASS。 |
| `cargo clippy -p ai --features all-providers --all-targets -- -D warnings` | PASS；移除了新增测试中的 `clone_on_copy`。 |
| `cargo fmt -p ai -- --check` 和 `git diff --check` | 最终格式化后 PASS。 |

## 完成标准

- Task 1-10 保持完整、连续和完成状态，不收缩为 Task 1-8。
- 每项 Batch 3 follow-up 都有 RED 证据、窄修复及 OpenAI/Codex public coverage。
- `all-providers`、三个独立 feature、`knuth-agent`、workspace no-run、严格 clippy、fmt、diff
  均通过。
- tracked diff 仅含本次窄修复文件清单中的 5 个版本控制文件；ignored Batch 3 报告不进入提交。
