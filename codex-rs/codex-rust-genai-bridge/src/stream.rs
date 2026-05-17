use codex_api::ApiError;
use codex_api::Provider;
use codex_api::ResponseStream;
use codex_api::ResponsesApiRequest;
use codex_api::SharedAuthProvider;
use codex_api::TransportError;
use futures::StreamExt;
use genai::adapter::AdapterKind;
use genai::chat::ChatOptions;
use genai::chat::ChatResponseFormat;
use genai::chat::JsonSpec;
use genai::chat::Verbosity;
use http::HeaderMap;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing;

use crate::convert_request::responses_request_to_chat_request;
use crate::convert_response::chat_event_to_response_event;
use crate::resolver::build_extra_headers;
use crate::resolver::build_genai_client;
use crate::types::PendingAssistantMessage;

const RESPONSE_STREAM_CHANNEL_CAPACITY: usize = 256;

/// The main entry point: converts a Codex `ResponsesApiRequest` into a genai
/// `ChatRequest`, streams via rust-genai, and converts each `ChatStreamEvent`
/// back into a Codex `ResponseEvent` stream.
pub async fn stream_via_genai(
    request: &ResponsesApiRequest,
    api_provider: &Provider,
    api_auth: &SharedAuthProvider,
    extra_headers: HeaderMap,
    adapter_kind: AdapterKind,
    idle_timeout: Duration,
) -> Result<ResponseStream, ApiError> {
    let chat_request = match responses_request_to_chat_request(request) {
        Some(req) => req,
        None => {
            return Err(ApiError::InvalidRequest {
                message: "No convertible messages in request".into(),
            });
        }
    };

    let model = request.model.clone();
    let chat_options = build_chat_options(request, api_provider, api_auth, extra_headers);

    let genai_client = build_genai_client(api_provider, api_auth, adapter_kind);

    let msg_count = chat_request.messages.len();
    let system_present = chat_request.system.is_some();
    tracing::info!(
        model = %model,
        adapter = ?adapter_kind,
        message_count = msg_count,
        has_system_prompt = system_present,
        has_reasoning_effort = chat_options.reasoning_effort.is_some(),
        "Dispatching chat stream via genai"
    );

    // Debug: log each message's role and whether it carries reasoning_content
    for (i, msg) in chat_request.messages.iter().enumerate() {
        let has_reasoning = msg
            .content
            .parts()
            .iter()
            .any(|p| matches!(p, genai::chat::ContentPart::ReasoningContent(_)));
        let part_types: Vec<&str> = msg
            .content
            .parts()
            .iter()
            .map(|p| match p {
                genai::chat::ContentPart::Text(_) => "text",
                genai::chat::ContentPart::ReasoningContent(_) => "reasoning",
                genai::chat::ContentPart::ToolCall(_) => "tool_call",
                genai::chat::ContentPart::ToolResponse(_) => "tool_response",
                genai::chat::ContentPart::ThoughtSignature(_) => "thought_sig",
                genai::chat::ContentPart::Binary(_) => "binary",
                genai::chat::ContentPart::Custom(_) => "custom",
            })
            .collect();
        tracing::debug!(
            msg_index = i,
            role = ?msg.role,
            has_reasoning_content = has_reasoning,
            part_types = ?part_types,
            "Chat message detail"
        );
    }

    let chat_stream_response = genai_client
        .exec_chat_stream(&model, chat_request, Some(&chat_options))
        .await
        .map_err(|e| {
            tracing::error!(
                model = %model,
                error = %e,
                "genai stream failed"
            );
            ApiError::Transport(TransportError::Network(format!("genai stream error: {e}")))
        })?;

    let mut chat_stream = chat_stream_response.stream;

    let (tx, rx) = mpsc::channel(RESPONSE_STREAM_CHANNEL_CAPACITY);

    tokio::spawn(async move {
        let mut pending = PendingAssistantMessage::new();

        loop {
            match tokio::time::timeout(idle_timeout, chat_stream.next()).await {
                Ok(Some(Ok(event))) => {
                    let events = chat_event_to_response_event(event, &mut pending);
                    for ev in events {
                        if tx.send(Ok(ev)).await.is_err() {
                            return;
                        }
                    }
                }
                Ok(Some(Err(e))) => {
                    let _ = tx
                        .send(Err(ApiError::Transport(TransportError::Network(format!(
                            "genai stream error: {e}"
                        )))))
                        .await;
                    return;
                }
                Ok(None) => return,
                Err(_elapsed) => {
                    let _ = tx
                        .send(Err(ApiError::Transport(TransportError::Timeout)))
                        .await;
                    return;
                }
            }
        }
    });

    Ok(ResponseStream {
        rx_event: rx,
        upstream_request_id: None,
    })
}

fn build_chat_options(
    request: &ResponsesApiRequest,
    api_provider: &Provider,
    api_auth: &SharedAuthProvider,
    extra_headers: HeaderMap,
) -> ChatOptions {
    let mut options = ChatOptions::default();

    options.capture_usage = Some(true);
    options.capture_content = Some(true);
    options.capture_reasoning_content = Some(true);
    options.capture_tool_calls = Some(true);

    if let Some(ref reasoning) = request.reasoning {
        if let Some(effort) = &reasoning.effort {
            use genai::chat::ReasoningEffort;
            let re: ReasoningEffort = match effort {
                codex_protocol::openai_models::ReasoningEffort::None => ReasoningEffort::None,
                codex_protocol::openai_models::ReasoningEffort::Minimal => ReasoningEffort::Minimal,
                codex_protocol::openai_models::ReasoningEffort::Low => ReasoningEffort::Low,
                codex_protocol::openai_models::ReasoningEffort::Medium => ReasoningEffort::Medium,
                codex_protocol::openai_models::ReasoningEffort::High => ReasoningEffort::High,
                codex_protocol::openai_models::ReasoningEffort::XHigh => {
                    tracing::warn!("XHigh reasoning effort maps to High in genai");
                    ReasoningEffort::High
                }
            };
            options.reasoning_effort = Some(re);
        }
    }

    if let Some(ref tier) = request.service_tier {
        use genai::chat::ServiceTier;
        options.service_tier = Some(match tier.as_str() {
            "flex" => ServiceTier::Flex,
            "auto" => ServiceTier::Auto,
            "default" => ServiceTier::Default,
            other => {
                tracing::warn!(service_tier = %other, "Unknown service tier");
                ServiceTier::Default
            }
        });
    }

    if let Some(ref text) = request.text {
        if let Some(ref verbosity) = text.verbosity {
            options.verbosity = Some(match verbosity {
                codex_api::OpenAiVerbosity::Low => Verbosity::Low,
                codex_api::OpenAiVerbosity::Medium => Verbosity::Medium,
                codex_api::OpenAiVerbosity::High => Verbosity::High,
            });
        }
        if let Some(ref fmt) = text.format {
            options.response_format = Some(ChatResponseFormat::JsonSpec(JsonSpec {
                name: fmt.name.clone(),
                description: None,
                schema: fmt.schema.clone(),
            }));
        }
    }

    options.prompt_cache_key = request.prompt_cache_key.clone();

    // Merge all headers for this request
    options.extra_headers = Some(build_extra_headers(api_provider, api_auth, &extra_headers));

    options
}
