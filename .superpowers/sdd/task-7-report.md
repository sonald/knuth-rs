# Task 7 Report: OpenAI Responses Done Event Routing

## RED / GREEN

- RED: `text_done_routes_by_output_index_not_last_block` failed before the fix because a text done event emitted no `TextEnd` after a later reasoning block.
- RED: `reasoning_done_routes_by_output_index_not_last_same_type_block` failed before the fix because the done payload for output index 0 overwrote the last reasoning block.
- GREEN: both regressions pass after identity-based lookup.

## Event Routing Matrix

| Event | Identity used | Result |
|---|---|---|
| `response.output_text.delta` / `.done` | `(output_index, content_index)` | New private map records and finds the text block. |
| `response.reasoning_summary_text.delta` / `.done` | `item_id`, then `output_index` | Reuses existing item/output maps. |
| `response.function_call_arguments.delta` / `.done` | `item_id`, then `output_index` | Reuses existing maps; existing `tool_argument_deltas_route_by_item_id` remains coverage. |
| `response.output_item.done` | `item.id` / `output_index` | Reuses existing maps; no last-block fallback. |
| `response.content_part.done` | N/A | No existing handler or emitted block lifecycle to route; parser intentionally unchanged. |

## Validation

- `cargo test -p ai --features openai-responses text_done_routes_by_output_index_not_last_block` (RED): failed as expected, no `TextEnd`.
- `cargo test -p ai --features openai-responses reasoning_done_routes_by_output_index_not_last_same_type_block` (RED): failed as expected, last reasoning block changed.
- `cargo test -p ai --features openai-responses 'done_routes_by_output_index_not_last'`: 2 passed.
- `cargo test -p ai --features openai-responses providers::openai_responses::tests::`: 16 passed.
- `cargo test -p ai --features openai-responses`: 93 unit, 15 integration, 0 doctest passed.
- `cargo test -p ai --features all-providers`: 123 unit, 36 integration, 0 doctest passed.
- `cargo fmt -p ai -- --check`: passed.
- `git diff --check`: passed.

The single `dead_code` warning from `crates/ai/src/utils/openai_compat_url.rs` was present during RED and the affected feature run, and no new warning was introduced by this task.

## Files

- `crates/ai/src/providers/openai_responses.rs`
- `docs/superpowers/plans/2026-07-09-ai-crate-complete-repair.md`
- `.superpowers/sdd/task-7-report.md`

## Self Review

- No public `BlockIdentity` or other public type added; the only new state is private `HashMap<(usize, usize), usize>`.
- Task 5 usage/cost and Task 6 encrypted replay paths were left unchanged.
- Task 8 and other providers/integration fixtures were not modified.
