use codex_api::ResponseEvent;
use codex_protocol::models::{ContentItem, ReasoningItemContent, ResponseItem};
use codex_protocol::protocol::TokenUsage;
use genai::chat::ToolChunk;
use genai::chat::{ChatStreamEvent, StopReason, StreamChunk, StreamEnd};

use crate::types::PendingAssistantMessage;

/// Converts a rust-genai `ChatStreamEvent` into zero or more Codex `ResponseEvent`s.
///
/// Because a single genai `End` event can fan out into multiple Codex events
/// (e.g., `OutputItemDone` + `Completed`), this returns a `Vec`.
pub fn chat_event_to_response_event(
    event: ChatStreamEvent,
    pending: &mut PendingAssistantMessage,
) -> Vec<ResponseEvent> {
    match event {
        ChatStreamEvent::Start => {
            vec![ResponseEvent::Created]
        }
        ChatStreamEvent::Chunk(StreamChunk { content }) => {
            let mut events = Vec::new();
            ensure_message_item_added(pending, &mut events);
            pending.text_buffer.push_str(&content);
            events.push(ResponseEvent::OutputTextDelta(content));
            events
        }
        ChatStreamEvent::ReasoningChunk(StreamChunk { content }) => {
            let mut events = Vec::new();
            ensure_message_item_added(pending, &mut events);
            let idx = pending.reasoning_content_index;
            pending.reasoning_content_index += 1;
            pending.reasoning_buffer.push_str(&content);
            events.push(ResponseEvent::ReasoningContentDelta {
                delta: content,
                content_index: idx as i64,
            });
            events
        }
        ChatStreamEvent::ThoughtSignatureChunk(StreamChunk { content }) => {
            let mut events = Vec::new();
            ensure_message_item_added(pending, &mut events);
            let idx = pending.reasoning_content_index;
            pending.reasoning_content_index += 1;
            pending.thought_signatures.push(content.clone());
            events.push(ResponseEvent::ReasoningContentDelta {
                delta: content,
                content_index: idx as i64,
            });
            events
        }
        ChatStreamEvent::ToolCallChunk(ToolChunk { tool_call }) => {
            let call_id = tool_call.call_id.clone();
            let fn_name = tool_call.fn_name.clone();
            // genai stores fn_arguments as Value::String(raw) in streaming chunks.
            // Use as_str() to get the raw content — to_string() would JSON-encode it (with quotes).
            let accumulated = tool_call
                .fn_arguments
                .as_str()
                .map(|s| s.to_string())
                .unwrap_or_else(|| tool_call.fn_arguments.to_string());
            let delta = if let Some(existing) = pending.tool_calls.get(&call_id) {
                let prev = existing.arguments_buffer.as_str();
                if accumulated.len() > prev.len() && accumulated.starts_with(prev) {
                    accumulated[prev.len()..].to_string()
                } else {
                    accumulated.clone()
                }
            } else {
                accumulated.clone()
            };

            let mut events = Vec::new();

            // Emit OutputItemAdded before the first delta so turn.rs creates a diff consumer.
            if !pending.tool_items_added.contains_key(&call_id) {
                pending.tool_items_added.insert(call_id.clone(), true);
                events.push(ResponseEvent::OutputItemAdded(
                    ResponseItem::CustomToolCall {
                        id: Some(call_id.clone()),
                        status: None,
                        call_id: call_id.clone(),
                        name: fn_name.clone(),
                        input: String::new(),
                    },
                ));
            }

            pending.tool_calls.insert(
                call_id.clone(),
                crate::types::PendingToolCall {
                    id: call_id.clone(),
                    name: fn_name,
                    arguments_buffer: accumulated,
                },
            );

            if !delta.is_empty() {
                events.push(ResponseEvent::ToolCallInputDelta {
                    item_id: call_id.clone(),
                    call_id: Some(call_id),
                    delta,
                });
            }
            events
        }
        ChatStreamEvent::End(end) => handle_stream_end(end, pending),
    }
}

fn handle_stream_end(end: StreamEnd, pending: &mut PendingAssistantMessage) -> Vec<ResponseEvent> {
    let mut events: Vec<ResponseEvent> = Vec::new();

    pending.finish_reason = end.captured_stop_reason.as_ref().map(|r| r.to_string());

    pending.token_usage = end.captured_usage.as_ref().map(|u| TokenUsage {
        input_tokens: u.prompt_tokens.unwrap_or(0) as i64,
        cached_input_tokens: u
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens)
            .unwrap_or(0) as i64,
        output_tokens: u.completion_tokens.unwrap_or(0) as i64,
        reasoning_output_tokens: u
            .completion_tokens_details
            .as_ref()
            .and_then(|d| d.reasoning_tokens)
            .unwrap_or(0) as i64,
        total_tokens: u.total_tokens.unwrap_or(0) as i64,
    });

    pending.response_id = end.captured_response_id.clone();

    // 1. Emit OutputItemDone for the assistant message if there was text or reasoning content
    if !pending.text_buffer.is_empty() || !pending.reasoning_buffer.is_empty() {
        let mut content: Vec<ContentItem> = Vec::new();
        if !pending.text_buffer.is_empty() {
            content.push(ContentItem::OutputText {
                text: std::mem::take(&mut pending.text_buffer),
            });
        }
        let message_item = ResponseItem::Message {
            id: pending.text_item_id.take(),
            role: "assistant".into(),
            content,
            phase: None,
        };
        events.push(ResponseEvent::OutputItemDone(message_item));
    }

    // Emit Reasoning item so reasoning content is stored for echo-back
    // to providers that require it in subsequent requests (e.g. DeepSeek).
    if !pending.reasoning_buffer.is_empty() {
        let reasoning_text = std::mem::take(&mut pending.reasoning_buffer);
        let reasoning_id = format!("rsn_{}", pending.reasoning_content_index);
        let reasoning_item = ResponseItem::Reasoning {
            id: reasoning_id.clone(),
            summary: vec![],
            content: Some(vec![ReasoningItemContent::ReasoningText {
                text: reasoning_text.clone(),
            }]),
            encrypted_content: Some(reasoning_text),
        };
        events.push(ResponseEvent::OutputItemAdded(reasoning_item.clone()));
        events.push(ResponseEvent::OutputItemDone(reasoning_item));
    }

    // 2. Emit OutputItemDone for each tool call as FunctionCall.
    //    FunctionCall maps to ToolPayload::Function in codex's dispatch,
    //    which exec_command and other tools require.
    for (_, pending_tc) in std::mem::take(&mut pending.tool_calls) {
        let tc_item = ResponseItem::FunctionCall {
            id: Some(pending_tc.id.clone()),
            name: pending_tc.name,
            namespace: None,
            arguments: pending_tc.arguments_buffer,
            call_id: pending_tc.id,
        };
        events.push(ResponseEvent::OutputItemDone(tc_item));
    }

    // 3. Emit Completed
    //    Match on the StopReason variant directly rather than the Display string.
    let end_turn = end
        .captured_stop_reason
        .as_ref()
        .map(|reason| !matches!(reason, StopReason::ToolCall(_)));

    events.push(ResponseEvent::Completed {
        response_id: pending.response_id.clone().unwrap_or_default(),
        token_usage: pending.token_usage.take(),
        end_turn,
    });

    events
}

/// Ensures `OutputItemAdded(Message)` is emitted once before any text or reasoning deltas.
/// turn.rs requires an active item set up via OutputItemAdded before it can process deltas.
fn ensure_message_item_added(
    pending: &mut PendingAssistantMessage,
    events: &mut Vec<ResponseEvent>,
) {
    if !pending.text_item_added {
        pending.text_item_added = true;
        let item_id = format!("txt_{}", pending.text_buffer.len());
        pending.text_item_id = Some(item_id.clone());
        events.push(ResponseEvent::OutputItemAdded(ResponseItem::Message {
            id: Some(item_id),
            role: "assistant".into(),
            content: vec![],
            phase: None,
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use genai::chat::ToolCall;

    #[test]
    fn test_start_event() {
        let mut pending = PendingAssistantMessage::new();
        let events = chat_event_to_response_event(ChatStreamEvent::Start, &mut pending);
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], ResponseEvent::Created));
    }

    #[test]
    fn test_chunk_emits_output_item_added_then_delta() {
        let mut pending = PendingAssistantMessage::new();
        let events = chat_event_to_response_event(
            ChatStreamEvent::Chunk(StreamChunk {
                content: "Hello".into(),
            }),
            &mut pending,
        );
        // First chunk emits OutputItemAdded(Message) + OutputTextDelta
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], ResponseEvent::OutputItemAdded(
            ResponseItem::Message { role, .. }
        ) if role == "assistant"));
        assert!(matches!(&events[1], ResponseEvent::OutputTextDelta(t) if t == "Hello"));
        assert_eq!(pending.text_buffer, "Hello");
        assert!(pending.text_item_added);
    }

    #[test]
    fn test_multiple_chunks_only_one_added() {
        let mut pending = PendingAssistantMessage::new();
        let events1 = chat_event_to_response_event(
            ChatStreamEvent::Chunk(StreamChunk {
                content: "Hello ".into(),
            }),
            &mut pending,
        );
        assert_eq!(events1.len(), 2); // OutputItemAdded + OutputTextDelta

        let events2 = chat_event_to_response_event(
            ChatStreamEvent::Chunk(StreamChunk {
                content: "world".into(),
            }),
            &mut pending,
        );
        assert_eq!(events2.len(), 1); // Only OutputTextDelta, no duplicate OutputItemAdded
        assert_eq!(pending.text_buffer, "Hello world");
    }

    #[test]
    fn test_end_with_stop() {
        let mut pending = PendingAssistantMessage::new();
        pending.text_buffer = "Done".into();

        let end = StreamEnd {
            captured_usage: None,
            captured_stop_reason: Some(StopReason::Completed("stop".into())),
            captured_content: None,
            captured_reasoning_content: None,
            captured_response_id: Some("resp_1".into()),
        };

        let events = chat_event_to_response_event(ChatStreamEvent::End(end), &mut pending);
        // Should emit OutputItemDone(Message) + Completed
        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], ResponseEvent::OutputItemDone(..)));
        assert!(matches!(
            &events[1],
            ResponseEvent::Completed {
                end_turn: Some(true),
                ..
            }
        ));
    }

    #[test]
    fn test_end_with_tool_calls() {
        let mut pending = PendingAssistantMessage::new();

        let end = StreamEnd {
            captured_usage: None,
            captured_stop_reason: Some(StopReason::ToolCall("tool_calls".into())),
            captured_content: None,
            captured_reasoning_content: None,
            captured_response_id: Some("resp_2".into()),
        };

        let events = chat_event_to_response_event(ChatStreamEvent::End(end), &mut pending);
        assert!(events.iter().any(|e| matches!(
            e,
            ResponseEvent::Completed {
                end_turn: Some(false),
                ..
            }
        )));
    }

    #[test]
    fn test_reasoning_chunk_increments_content_index() {
        let mut pending = PendingAssistantMessage::new();
        let events1 = chat_event_to_response_event(
            ChatStreamEvent::ReasoningChunk(StreamChunk {
                content: "first".into(),
            }),
            &mut pending,
        );
        // First reasoning chunk emits OutputItemAdded(Message) + ReasoningContentDelta
        assert_eq!(events1.len(), 2);
        assert!(matches!(&events1[0], ResponseEvent::OutputItemAdded(..)));
        assert!(matches!(&events1[1], ResponseEvent::ReasoningContentDelta {
            delta, content_index: 0
        } if delta == "first"));

        let events2 = chat_event_to_response_event(
            ChatStreamEvent::ReasoningChunk(StreamChunk {
                content: "second".into(),
            }),
            &mut pending,
        );
        assert_eq!(events2.len(), 1);
        assert!(matches!(&events2[0], ResponseEvent::ReasoningContentDelta {
            delta, content_index: 1
        } if delta == "second"));

        assert_eq!(pending.reasoning_buffer, "firstsecond");
    }

    #[test]
    fn test_reasoning_does_not_duplicate_output_item_added_when_text_arrives_first() {
        let mut pending = PendingAssistantMessage::new();
        // Text arrives first → OutputItemAdded emitted
        chat_event_to_response_event(
            ChatStreamEvent::Chunk(StreamChunk {
                content: "Hello ".into(),
            }),
            &mut pending,
        );
        assert!(pending.text_item_added);
        // Reasoning arrives later → should NOT emit another OutputItemAdded
        let events = chat_event_to_response_event(
            ChatStreamEvent::ReasoningChunk(StreamChunk {
                content: "thinking...".into(),
            }),
            &mut pending,
        );
        assert_eq!(events.len(), 1, "should not emit duplicate OutputItemAdded");
        assert!(matches!(
            &events[0],
            ResponseEvent::ReasoningContentDelta { .. }
        ));
    }

    #[test]
    fn test_tool_call_first_chunk() {
        let mut pending = PendingAssistantMessage::new();
        let events = chat_event_to_response_event(
            ChatStreamEvent::ToolCallChunk(ToolChunk {
                tool_call: ToolCall {
                    call_id: "call_1".into(),
                    fn_name: "get_weather".into(),
                    fn_arguments: serde_json::Value::String(r#"{"city":"SF"}"#.into()),
                    thought_signatures: None,
                },
            }),
            &mut pending,
        );
        // First chunk: OutputItemAdded(CustomToolCall) + ToolCallInputDelta
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], ResponseEvent::OutputItemAdded(
            ResponseItem::CustomToolCall { call_id, name, .. }
        ) if call_id == "call_1" && name == "get_weather"));
        assert!(matches!(&events[1], ResponseEvent::ToolCallInputDelta {
            call_id, ..
        } if call_id.as_deref() == Some("call_1")));
    }

    #[test]
    fn test_tool_call_subsequent_chunk_computes_delta() {
        let mut pending = PendingAssistantMessage::new();
        // First chunk: partial arguments (simulating streaming accumulation)
        let partial = serde_json::Value::String(r#"{"city":"#.into());
        chat_event_to_response_event(
            ChatStreamEvent::ToolCallChunk(ToolChunk {
                tool_call: ToolCall {
                    call_id: "call_1".into(),
                    fn_name: "get_weather".into(),
                    fn_arguments: partial,
                    thought_signatures: None,
                },
            }),
            &mut pending,
        );
        // Second chunk: accumulated to full arguments
        let full = serde_json::Value::String(r#"{"city":"SF"}"#.into());
        let events = chat_event_to_response_event(
            ChatStreamEvent::ToolCallChunk(ToolChunk {
                tool_call: ToolCall {
                    call_id: "call_1".into(),
                    fn_name: "get_weather".into(),
                    fn_arguments: full,
                    thought_signatures: None,
                },
            }),
            &mut pending,
        );
        // Second chunk: only ToolCallInputDelta (OutputItemAdded already emitted)
        assert_eq!(events.len(), 1);
        // Delta should be only the new part: "SF"}"
        assert!(
            matches!(&events[0], ResponseEvent::ToolCallInputDelta { delta, .. } if delta == "\"SF\"}")
        );
    }
}
