use codex_api::Provider;
use codex_api::SharedAuthProvider;
use genai::ServiceTarget;
use genai::adapter::AdapterKind;
use genai::resolver::{AuthData, Endpoint};
use http::HeaderMap;

/// Builds a rust-genai `Client` with Codex's auth and endpoint routing.
///
/// - Auth: extracts Bearer token from Codex's `SharedAuthProvider`.
/// - Endpoint: overrides the adapter default with Codex's configured `base_url`.
/// - Custom headers are passed per-request via `ChatOptions.extra_headers`.
pub fn build_genai_client(
    api_provider: &Provider,
    api_auth: &SharedAuthProvider,
    adapter_kind: AdapterKind,
) -> genai::Client {
    let base_url = api_provider.base_url.clone();
    let auth_headers = api_auth.to_auth_headers();

    genai::Client::builder()
        .with_adapter_kind(adapter_kind)
        .with_auth_resolver_fn(move |_model_iden| {
            let token = auth_headers
                .get(http::header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok())
                .map(|v| v.to_string())
                .or_else(|| {
                    auth_headers
                        .get("api-key")
                        .and_then(|v| v.to_str().ok())
                        .map(|v| format!("Bearer {v}"))
                });

            Ok(token.map(AuthData::Key))
        })
        .with_service_target_resolver_fn(move |mut target: ServiceTarget| {
            target.endpoint = Endpoint::from_owned(base_url.clone());
            Ok(target)
        })
        .build()
}

/// Builds genai `Headers` from Codex provider, auth, and extra headers.
pub fn build_extra_headers(
    api_provider: &Provider,
    api_auth: &SharedAuthProvider,
    extra_headers: &HeaderMap,
) -> genai::Headers {
    let mut headers: Vec<(String, String)> = Vec::new();

    // Provider-level headers first
    for (key, value) in api_provider.headers.iter() {
        if let Ok(v) = value.to_str() {
            headers.push((key.as_str().to_string(), v.to_string()));
        }
    }

    // Auth headers override
    let auth_headers = api_auth.to_auth_headers();
    for (key, value) in auth_headers.iter() {
        if let Ok(v) = value.to_str() {
            headers.push((key.as_str().to_string(), v.to_string()));
        }
    }

    // Extra headers take top precedence
    for (key, value) in extra_headers.iter() {
        if let Ok(v) = value.to_str() {
            headers.push((key.as_str().to_string(), v.to_string()));
        }
    }

    headers.into()
}
