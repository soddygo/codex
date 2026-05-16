use codex_api::ApiError;
use codex_api::Provider;
use codex_api::ResponseStream;
use codex_api::ResponsesApiRequest;
use codex_api::SharedAuthProvider;
use codex_api::TransportError;
use futures::StreamExt;
use genai::adapter::AdapterKind;
use genai::chat::{ChatOptions, ChatResponseFormat, JsonSpec, Verbosity};
use http::HeaderMap;
use std::time::Duration;
use tokio::sync::mpsc;

use crate::convert_request::responses_request_to_chat_request;
use crate::convert_response::chat_event_to_response_event;
use crate::resolver::{build_extra_headers, build_genai_client};
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

    let chat_stream_response = genai_client
        .exec_chat_stream(&model, chat_request, Some(&chat_options))
        .await
        .map_err(|e| {
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
