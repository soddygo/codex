use codex_protocol::protocol::TokenUsage;
use std::collections::BTreeMap;

/// Accumulates streaming partials into a complete assistant message plus final metadata.
pub(crate) struct PendingAssistantMessage {
    /// Aggregated text from `OutputTextDelta` events.
    pub text_buffer: String,
    /// Aggregated reasoning from `ReasoningContentDelta` events.
    pub reasoning_buffer: String,
    /// Tool calls keyed by `call_id`.
    pub tool_calls: BTreeMap<String, PendingToolCall>,
    /// Normalized finish reason from the stream end event.
    pub finish_reason: Option<String>,
    /// Token usage captured at stream end.
    pub token_usage: Option<TokenUsage>,
    /// Response ID captured at stream end.
    pub response_id: Option<String>,
    /// Thought signatures collected during streaming.
    pub thought_signatures: Vec<String>,

    /// Whether `OutputItemAdded(Message)` has been emitted for the text item.
    pub text_item_added: bool,
    /// Synthetic ID for the text message item (consistent between OutputItemAdded and OutputItemDone).
    pub text_item_id: Option<String>,
    /// Per-tool-call flag: whether `OutputItemAdded` was emitted (keyed by call_id, parallel to `tool_calls`).
    pub tool_items_added: BTreeMap<String, bool>,
    /// Incrementing counter for `ReasoningContentDelta.content_index`.
    pub reasoning_content_index: u64,
}

impl PendingAssistantMessage {
    pub fn new() -> Self {
        Self {
            text_buffer: String::new(),
            reasoning_buffer: String::new(),
            tool_calls: BTreeMap::new(),
            finish_reason: None,
            token_usage: None,
            response_id: None,
            thought_signatures: Vec::new(),
            text_item_added: false,
            text_item_id: None,
            tool_items_added: BTreeMap::new(),
            reasoning_content_index: 0,
        }
    }
}

/// A tool call being incrementally built from streaming deltas.
pub(crate) struct PendingToolCall {
    pub id: String,
    pub name: String,
    /// Accumulated `fn_arguments` value (used to compute actual delta for the next chunk).
    pub arguments_buffer: String,
}
