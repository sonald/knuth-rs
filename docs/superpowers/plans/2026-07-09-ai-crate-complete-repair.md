# AI Crate 完整修复与最终审查计划

**目标：** 在不新增公开领域类型、不回滚既有历史的前提下，完成 `crates/ai`
最终审查的三个修复 batch，并以公开 provider 流、独立 feature、consumer 和 workspace
门禁证明行为闭环。

**Batch 3 基线：** `49b3e875f801a292253a8548b42847de03a95688`

**实现原则：**

- 如无必要，勿增实体；新增类型或概念必须先通过 YAGNI 检查。
- 行为修复先写失败测试并保留 RED 证据。
- Responses refusal 继续使用现有 `ContentBlock::Text` 和 Text lifecycle。
- Responses message 身份复用现有 `TextContent.text_signature` 与 `TextSignatureV1`。
- Codex encrypted reasoning 复用现有 `ThinkingContent.thinking_signature`。
- 不新增依赖，不引入公开 provider/auth/pricing 类型。
- 只修改 `crates/ai`、本正式计划和 ignored Batch 3 执行记录。

## 真实文件清单

| 文件 | Batch 3 实际职责 |
|---|---|
| `crates/ai/src/providers/openai_responses.rs` | refusal lifecycle、message `id/phase` 签名、output-item reconciliation、有序 store:false replay |
| `crates/ai/src/bedrock_anthropic.rs` | 恢复 `Default`/无参 `new()`，新增 `with_model(Model)` cost 路径，明确 utility 边界 |
| `crates/ai/src/models.rs` | 将 `calculate_usage_cost` 收紧为 `pub(crate)` |
| `crates/ai/src/providers/anthropic.rs` | 修复严格 clippy 的 `unnecessary_unwrap` |
| `crates/ai/src/sigv4.rs` | 修复严格 clippy 的两个 `redundant_guard` |
| `crates/ai/tests/provider_terminal_events.rs` | OpenAI 与 Codex wrapper 的公开 refusal lifecycle |
| `crates/ai/tests/provider_request_metadata.rs` | Responses 公开两轮回放、Codex malformed/mixed replay、clippy 小修 |
| `crates/ai/README.md` | 静态目录价、service tier、Bedrock converter utility 契约 |
| `docs/superpowers/plans/2026-07-09-ai-crate-complete-repair.md` | 唯一 tracked 正式计划、Task 1-8、三 batch 矩阵和最终结果 |
| `.superpowers/sdd/final-review-batch-3.md` | ignored RED/GREEN 执行记录，不进入 commit |

`openai_codex_responses.rs`、Azure wrapper、`stream.rs` 和 feature 定义经审计无需生产改动：
Codex/Azure 已复用 `consume_responses_sse`，Codex 现有过滤器已满足 encrypted item 边界；
本批只补公开 fixture 与共享实现。

## Task 1：Responses refusal lifecycle

- [x] 使用 `ContentBlock::Text` 表达 `refusal`，不新增 refusal block。
- [x] 支持 `response.content_part.added` 的 `refusal` part。
- [x] 支持 `response.refusal.delta` 与 `response.refusal.done`。
- [x] 用 `response.output_item.done` 补全 message 最终内容并避免重复 `TextEnd`。
- [x] 保持 `TextStart -> TextDelta -> TextEnd -> Done` 的 partial/terminal 内容一致。
- [x] 真实入口覆盖 OpenAI provider 和 Codex wrapper。

具名测试：

- `refusal_lifecycle_uses_text_events_and_reconciles_output_item_done`
- `message_output_item_done_finalizes_unfinished_refusal`
- `content_part_added_ignores_non_text_and_non_refusal_parts`
- `openai_responses_refusal_uses_complete_text_lifecycle`
- `codex_responses_wrapper_reuses_refusal_text_lifecycle`

## Task 2：store:false output-item 有序回放

- [x] message output item 的 `id/phase` 编码为现有 `TextSignatureV1` JSON。
- [x] replay 时解析 JSON 签名，并兼容旧的纯字符串 message id。
- [x] 每个 Text block 单独序列化为 `type=message`、`status=completed` output item。
- [x] 按原 `ContentBlock` 顺序直接输出 reasoning/message/function_call。
- [x] 不再聚合所有 Text，也不再把全部 function call 后移。
- [x] 有合法 reasoning signature 时，即使 compat 不要求合成 reasoning，也原样回放。
- [x] 公开两轮 fixture 覆盖 `reasoning -> message(commentary) -> function_call -> message(final_answer) -> user`。

具名测试：

- `responses_replay_preserves_interleaved_output_items_and_message_identity`
- `responses_store_false_round_trip_preserves_interleaved_output_items`
- 既有 `reasoning_item_done_is_captured_and_replayed`
- 既有 `tool_call_id_preserves_response_item_id_for_replay`

## Task 3：Codex encrypted reasoning 防御性 fixture

- [x] malformed JSON 不回放。
- [x] 非 `reasoning` type 不回放。
- [x] 缺失或非字符串 `encrypted_content` 不回放。
- [x] 无签名普通 thinking 不回放。
- [x] 仅合法 encrypted reasoning item 原样回放。
- [x] text、tool、user 和合法 reasoning 的相对顺序保持。

具名测试：

- `codex_replays_encrypted_reasoning_items`
- `codex_replays_only_well_formed_encrypted_reasoning_in_mixed_history`

审计结论：现有 `convert_codex_messages` 过滤逻辑已满足边界，因此生产 Codex wrapper
无需修改；Task 2 的共享 serializer 修复了 mixed history 的顺序问题。

## Task 4：Usage helper 可见性

- [x] `calculate_usage_cost` 从 `pub` 收紧为 `pub(crate)`。
- [x] 既有模型 unit test 保留。
- [x] provider terminal usage/cost 公开测试保留。

具名测试：

- `calculate_usage_cost_uses_per_million_prices`
- `openai_responses_done_usage_has_nonzero_cost`
- `codex_responses_wrapper_done_usage_has_nonzero_cost`
- `azure_openai_responses_wrapper_done_usage_has_nonzero_cost`
- `bedrock_anthropic_message_start_usage_combines_with_delta_output_and_prices_done`

## Task 5：Bedrock-Anthropic Converter 源码兼容

- [x] 恢复 `impl Default for Converter`。
- [x] 恢复无参 `Converter::new()`。
- [x] 带价格上下文改为 `Converter::with_model(Model)`。
- [x] 仅 with-model 路径结算 `Model.cost`；默认 utility 不伪造价格。
- [x] 文档明确该 converter 不是 `register_builtins` 的真实 provider。

具名测试：

- `converter_preserves_default_and_no_arg_new_constructors`
- `bedrock_anthropic_message_start_usage_combines_with_delta_output_and_prices_done`
- `bedrock_anthropic_error_terminal_calculates_usage_cost`

## Task 6：严格 clippy 与最小 feature cfg

- [x] Anthropic `unnecessary_unwrap`。
- [x] SigV4 两个 `redundant_guard`。
- [x] 后续暴露的测试 `clone_on_copy`。
- [x] 新增 Codex-only refusal fixture 不再使 EOF helper/`StopReason` 成为 unused。
- [x] `cargo clippy -p ai --features all-providers --all-targets -- -D warnings` 通过。

最小 feature 仍可能显示 `openai_compat_url` 等本批前已存在的 dead-code warning；本批不为
隐藏旧 warning 扩大 cfg 或加 allow。新增测试代码本身不制造新的 unused warning。

## Task 7：README 与 service_tier 契约

- [x] README 标明 Responses 支持 refusal 与 output-item replay。
- [x] README 标明 `Usage.cost` 静态按 `Model.cost` 的每百万 token 目录价计算。
- [x] `service_tier` 只作为 provider 请求字段透传。
- [x] 不支持 service-tier-specific 动态计价，不应用 multiplier，不修改模型目录价。
- [x] README 标明 `bedrock_anthropic::Converter` 是 utility，不是内置 provider。

### service_tier 固定契约

`StreamOptions.provider_extras["service_tier"]` 可以进入 OpenAI Responses 请求 body，
但当前响应 usage 没有足够且稳定的分层价格信息。`calculate_usage_cost` 因而始终只读
调用时 `Model.cost`，不会根据请求 tier 或响应 tier 动态调整。支持动态 tier 计价需要
独立价格来源、版本和回归契约；本计划明确不发明该模型。

## Task 8：正式计划、最终验证与单 commit

- [x] 唯一 tracked 正式计划统一为 Task 1-8。
- [x] 写入真实文件、具名测试、RED/GREEN 和当前检查结果。
- [x] 写入 Final Review Batch 1/2/3 回归矩阵。
- [x] 完成 OpenAI/Codex/Azure 独立 feature 验证。
- [x] 完成 `knuth-agent`、workspace no-run、fmt、diff 和最终 clippy。
- [x] 五轴自审后修复所有 Critical/Important finding。
- [x] 提交一个 Batch 3 commit：本提交 `fix(ai): close Responses API review`。

## RED 记录

| RED | 观察结果 | 根因 |
|---|---|---|
| refusal lifecycle | `starts.len()` 实际为 0，期望 1 | consumer 只接受 `output_text`，忽略 refusal 事件和 message done |
| interleaved replay | 对 output item `type` 取值时得到 `None` 并 panic | Text 被聚合成无 `type/id/phase` 的 assistant message，reasoning 被丢弃，tool call 被后移 |
| Converter 构造器 | E0061: `new` 缺少 `&Model`；E0599: 无 `default` | cost 改动破坏旧公开构造 API |
| unknown content part | `partial.content.is_empty()` 失败 | refusal 分支错误地把未知 part 当成空 Text |
| 严格 clippy 基线 | Anthropic 1 项、SigV4 2 项 | 分支遗留 lint |
| 严格 clippy 后续 | `KnownApi::clone()` 为 `clone_on_copy` | 前三项修复后 clippy 才继续暴露测试 lint |

所有行为 RED 均先确认失败原因，再做最小生产修复；对应具名测试当前均 GREEN。

## 最终审查三 Batch 回归矩阵

| Batch | 基线/提交 | 契约 | 主要回归证据 | 当前状态 |
|---|---|---|---|---|
| Batch 1 基础设施与安全 | `43b54f1..2693b1a` | SigV4 canonical URI、Cloudflare placeholder 白名单、Retry-After overflow、SSE EOF、凭据 Debug 脱敏、ADC 公开 API、Faux lock | SigV4 固定向量、Cloudflare/retry/SSE unit、Bedrock request capture、Vertex ADC public stream、strict clippy | `all-providers` 回归通过 |
| Batch 2 Provider fidelity | `bcee056`、`49b3e87` | Google/Vertex signature、Bedrock reasoning signature、terminal fail-closed、Mistral mixed reasoning/text 与 replay | Google/Vertex/Bedrock/Mistral public round trip、terminal reason matrix、独立 provider feature | `all-providers` 回归通过 |
| Batch 3 Responses/API/计划闭环 | 基线 `49b3e87`，本提交 `fix(ai): close Responses API review` | refusal lifecycle、store:false identity/order、Codex malformed replay、usage visibility、Converter compatibility、clippy、static service-tier cost contract、正式计划 | 本计划 Task 1-8 具名测试、OpenAI/Codex/Azure 独立 feature、agent/workspace/clippy/fmt/diff | 完成 |

## 当前检查结果

| 命令/检查 | 结果 |
|---|---|
| 两个最小 Responses RED | 按预期失败，修复后 PASS |
| Converter 构造器编译 RED | 按预期 E0061/E0599，修复后 6 个 converter tests PASS |
| unknown content part RED | 按预期失败，收紧后 PASS |
| OpenAI refusal 真实入口 | PASS |
| Codex refusal wrapper 真实入口 | PASS |
| Responses 公开两轮 store:false round trip | PASS |
| Codex malformed/mixed replay | PASS |
| `cargo test -p ai --features all-providers` | PASS，225 tests，0 failed（153 unit、7 Anthropic E2E、4 catalog、41 request metadata、19 terminal、1 support） |
| `cargo clippy -p ai --features all-providers --all-targets -- -D warnings` | PASS |
| OpenAI 独立 feature | PASS（100 lib、1 request metadata、3 terminal） |
| Codex 独立 feature | PASS（102 lib、4 request metadata、2 terminal） |
| Azure 独立 feature | PASS（104 lib、1 terminal） |
| `cargo test -p knuth-agent` | PASS，3 tests，0 failed |
| `cargo test --workspace --no-run` | PASS |
| `cargo fmt -p ai -- --check` / `git diff --check` | PASS |
| Batch 3 commit | PASS，本提交 `fix(ai): close Responses API review` |

## 完成标准

- Task 1-8 全部勾选。
- 所有具名、独立 feature、`all-providers`、consumer 和 workspace 门禁通过。
- 严格 ai clippy、ai fmt、diff check 通过。
- tracked diff 只包含本计划真实文件清单。
- 一个 Batch 3 commit，ignored 报告不进入 commit。
