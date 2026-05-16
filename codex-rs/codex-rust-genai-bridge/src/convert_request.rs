use codex_api::ResponsesApiRequest;
use codex_protocol::models::{ContentItem, FunctionCallOutputBody, ResponseItem};
use genai::chat::{
    ChatMessage, ChatRequest, ChatRole, ContentPart, MessageContent, Tool, ToolCall, ToolResponse,
};
use serde_json::Value;

/// Converts a Codex `ResponsesApiRequest` into a rust-genai `ChatRequest`.
///
/// Returns `None` if the input contains no convertible messages (e.g. only
/// internal/local items).
pub fn responses_request_to_chat_request(request: &ResponsesApiRequest) -> Option<ChatRequest> {
    let system = if request.instructions.is_empty() {
        None
    } else {
        Some(request.instructions.clone())
    };

    let messages = convert_response_items(&request.input);

    if messages.is_empty() && system.is_none() {
        return None;
    }

    let mut chat_req = ChatRequest::new(messages);
    if let Some(sys) = system {
        chat_req = chat_req.with_system(sys);
    }

    let tools = parse_tools(&request.tools);
    if !tools.is_empty() {
        chat_req = chat_req.with_tools(tools);
    }

    if request.store {
        chat_req = chat_req.with_store(true);
    }

    Some(chat_req)
}

fn convert_response_items(items: &[ResponseItem]) -> Vec<ChatMessage> {
    let mut messages: Vec<ChatMessage> = Vec::new();
    let mut pending_thought_signatures: Vec<String> = Vec::new();

    for item in items {
        match item {
            ResponseItem::Message { role, content, .. } => {
                let role = map_role(role);
                let parts = convert_content_items(content);
                let msg = ChatMessage::new(role, MessageContent::from_parts(parts));
                messages.push(msg);
            }
            ResponseItem::Reasoning {
                encrypted_content, ..
            } => {
                if let Some(ec) = encrypted_content {
                    pending_thought_signatures.push(ec.clone());
                }
            }
            ResponseItem::FunctionCall {
                name,
                arguments,
                call_id,
                ..
            } => {
                let fn_arguments: Value =
                    serde_json::from_str(arguments).unwrap_or(Value::String(arguments.clone()));
                let mut tool_call = ToolCall {
                    call_id: call_id.clone(),
                    fn_name: name.clone(),
                    fn_arguments,
                    thought_signatures: None,
                };
                // Attach pending thought signatures to the first tool call
                if !pending_thought_signatures.is_empty() {
                    tool_call.thought_signatures =
                        Some(std::mem::take(&mut pending_thought_signatures));
                }
                messages.push(ChatMessage::assistant(MessageContent::from(
                    ContentPart::ToolCall(tool_call),
                )));
            }
            ResponseItem::CustomToolCall {
                name,
                input,
                call_id,
                ..
            } => {
                let fn_arguments: Value =
                    serde_json::from_str(input).unwrap_or(Value::String(input.clone()));
                let tool_call = ToolCall {
                    call_id: call_id.clone(),
                    fn_name: name.clone(),
                    fn_arguments,
                    thought_signatures: None,
                };
                messages.push(ChatMessage::assistant(MessageContent::from(
                    ContentPart::ToolCall(tool_call),
                )));
            }
            ResponseItem::FunctionCallOutput { call_id, output } => {
                let content = match &output.body {
                    FunctionCallOutputBody::Text(text) => text.clone(),
                    FunctionCallOutputBody::ContentItems(items) => {
                        serde_json::to_string(items).unwrap_or_default()
                    }
                };
                messages.push(ChatMessage::tool(MessageContent::from(
                    ContentPart::ToolResponse(ToolResponse::new(call_id, content)),
                )));
            }
            ResponseItem::CustomToolCallOutput {
                call_id, output, ..
            } => {
                let content = match &output.body {
                    FunctionCallOutputBody::Text(text) => text.clone(),
                    FunctionCallOutputBody::ContentItems(items) => {
                        serde_json::to_string(items).unwrap_or_default()
                    }
                };
                messages.push(ChatMessage::tool(MessageContent::from(
                    ContentPart::ToolResponse(ToolResponse::new(call_id, content)),
                )));
            }
            // Internal/local events — not sent to the model.
            ResponseItem::LocalShellCall { .. }
            | ResponseItem::ToolSearchCall { .. }
            | ResponseItem::ToolSearchOutput { .. }
            | ResponseItem::WebSearchCall { .. }
            | ResponseItem::ImageGenerationCall { .. }
            | ResponseItem::Compaction { .. }
            | ResponseItem::ContextCompaction { .. }
            | ResponseItem::CompactionTrigger => {
                // Skip — these items are internal to Codex.
            }
            ResponseItem::Other => {
                tracing::warn!("Skipping unknown ResponseItem::Other in request conversion");
            }
        }
    }

    messages
}

fn map_role(codex_role: &str) -> ChatRole {
    match codex_role {
        "user" => ChatRole::User,
        "assistant" => ChatRole::Assistant,
        "developer" | "system" => ChatRole::System,
        "tool" => ChatRole::Tool,
        other => {
            tracing::warn!(
                role = %other,
                "Unknown response item role, defaulting to User"
            );
            ChatRole::User
        }
    }
}

fn convert_content_items(items: &[ContentItem]) -> Vec<ContentPart> {
    items
        .iter()
        .filter_map(|item| match item {
            ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                Some(ContentPart::Text(text.clone()))
            }
            ContentItem::InputImage { image_url, .. } => {
                // Infer content type from URL extension or default to image/png
                let content_type = if image_url.ends_with(".jpg") || image_url.ends_with(".jpeg") {
                    "image/jpeg"
                } else if image_url.ends_with(".webp") {
                    "image/webp"
                } else if image_url.ends_with(".gif") {
                    "image/gif"
                } else {
                    "image/png"
                };
                Some(ContentPart::Binary(genai::chat::Binary::from_url(
                    content_type,
                    image_url,
                    None,
                )))
            }
        })
        .collect()
}

fn parse_tools(tools: &[Value]) -> Vec<Tool> {
    tools
        .iter()
        .filter_map(|v| {
            let name = v.get("name")?.as_str()?;
            let description = v
                .get("description")
                .and_then(|d| d.as_str())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string());
            let schema = v.get("parameters").or(v.get("input_schema")).cloned();
            Some(Tool {
                name: name.to_string().into(),
                description,
                schema,
                strict: v.get("strict").and_then(|s| s.as_bool()),
                config: None,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_protocol::models::ContentItem;

    #[test]
    fn test_single_user_message() {
        let request = ResponsesApiRequest {
            model: "gpt-5".into(),
            instructions: String::new(),
            input: vec![ResponseItem::Message {
                id: None,
                role: "user".into(),
                content: vec![ContentItem::InputText {
                    text: "Hello".into(),
                }],
                phase: None,
            }],
            tools: vec![],
            tool_choice: "auto".into(),
            parallel_tool_calls: true,
            reasoning: None,
            store: false,
            stream: true,
            include: vec![],
            service_tier: None,
            prompt_cache_key: None,
            text: None,
            client_metadata: None,
        };

        let result = responses_request_to_chat_request(&request).unwrap();
        assert_eq!(result.messages.len(), 1);
        let msg = &result.messages[0];
        assert_eq!(msg.role, ChatRole::User);
        assert_eq!(msg.content.parts().len(), 1);
    }

    #[test]
    fn test_system_instructions() {
        let request = ResponsesApiRequest {
            model: "gpt-5".into(),
            instructions: "You are helpful.".into(),
            input: vec![ResponseItem::Message {
                id: None,
                role: "user".into(),
                content: vec![ContentItem::InputText { text: "Hi".into() }],
                phase: None,
            }],
            tools: vec![],
            tool_choice: "auto".into(),
            parallel_tool_calls: true,
            reasoning: None,
            store: false,
            stream: true,
            include: vec![],
            service_tier: None,
            prompt_cache_key: None,
            text: None,
            client_metadata: None,
        };

        let result = responses_request_to_chat_request(&request).unwrap();
        assert_eq!(result.system, Some("You are helpful.".into()));
    }

    #[test]
    fn test_function_call_and_output() {
        let request = ResponsesApiRequest {
            model: "gpt-5".into(),
            instructions: String::new(),
            input: vec![
                ResponseItem::FunctionCall {
                    id: None,
                    name: "get_weather".into(),
                    namespace: None,
                    arguments: r#"{"city":"SF"}"#.into(),
                    call_id: "call_1".into(),
                },
                ResponseItem::FunctionCallOutput {
                    call_id: "call_1".into(),
                    output: codex_protocol::models::FunctionCallOutputPayload {
                        body: FunctionCallOutputBody::Text("Sunny".into()),
                        success: Some(true),
                    },
                },
            ],
            tools: vec![],
            tool_choice: "auto".into(),
            parallel_tool_calls: true,
            reasoning: None,
            store: false,
            stream: true,
            include: vec![],
            service_tier: None,
            prompt_cache_key: None,
            text: None,
            client_metadata: None,
        };

        let result = responses_request_to_chat_request(&request).unwrap();
        assert_eq!(result.messages.len(), 2);
        assert_eq!(result.messages[0].role, ChatRole::Assistant);
        assert_eq!(result.messages[1].role, ChatRole::Tool);
    }

    #[test]
    fn test_empty_input() {
        let request = ResponsesApiRequest {
            model: "gpt-5".into(),
            instructions: String::new(),
            input: vec![],
            tools: vec![],
            tool_choice: "auto".into(),
            parallel_tool_calls: true,
            reasoning: None,
            store: false,
            stream: true,
            include: vec![],
            service_tier: None,
            prompt_cache_key: None,
            text: None,
            client_metadata: None,
        };

        assert!(responses_request_to_chat_request(&request).is_none());
    }
}
