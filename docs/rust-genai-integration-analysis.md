# rust-genai 集成到 Codex 深度分析报告

## 1. rust-genai 核心架构

### 1.1 整体调用链

```
┌─────────────────────────────────────────────────────────┐
│                Client::exec_chat / exec_chat_stream       │
│  model: "gpt-5-codex" | chat_req: ChatRequest            │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              ClientConfig::resolve_model_spec()           │
│  ModelSpec → ModelMapper → AuthResolver →                │
│  default_endpoint → ServiceTargetResolver → ServiceTarget│
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              AdapterDispatcher (静态分发)                  │
│  match adapter_kind {                                     │
│    OpenAI     → OpenAIAdapter::to_web_request_data()      │
│    OpenAIResp → OpenAIRespAdapter::to_web_request_data()  │
│    Anthropic  → AnthropicAdapter::to_web_request_data()   │
│    Gemini     → GeminiAdapter::to_web_request_data()      │
│    Groq       → GroqAdapter                               │
│    DeepSeek   → DeepSeekAdapter                           │
│    Ollama     → OllamaAdapter                             │
│    Vertex     → VertexAdapter                             │
│    Bedrock    → BedrockApiAdapter                         │
│    ... (27+ adapters)                                     │
│  }                                                        │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌──────────────────┐    ┌──────────────────────────┐
│  非流式路径       │    │  流式路径 (SSE)            │
│  do_post →        │    │  EventSourceStream →      │
│  to_chat_response │    │  AdapterStreamer →        │
│  → ChatResponse   │    │  InterStreamEvent →       │
│                   │    │  ChatStream →             │
│                   │    │  ChatStreamEvent           │
└──────────────────┘    └──────────────────────────┘
```

### 1.2 请求/响应类型层级

```
ChatRequest                ChatResponse               ChatStreamEvent
├── system: Option<String> ├── content: MessageContent ├── Start
├── messages: Vec<ChatMsg> ├── reasoning_content       ├── Chunk(StreamChunk)
├── tools: Vec<Tool>       ├── model_iden              ├── ReasoningChunk
├── previous_response_id   ├── stop_reason             ├── ThoughtSignatureChunk
└── store: Option<bool>    ├── usage: Usage            ├── ToolCallChunk(ToolChunk)
                           ├── response_id             └── End(StreamEnd)
                           └── captured_raw_body           ├── captured_usage
                                                           ├── captured_stop_reason
ChatMessage                 MessageContent                ├── captured_content
├── role: ChatRole          └── parts: Vec<ContentPart>   ├── captured_reasoning_content
│   ├── System                  ├── Text(String)          └── captured_response_id
│   ├── User                    ├── Binary(Binary)
│   ├── Assistant               ├── ToolCall(ToolCall)
│   └── Tool                    ├── ToolResponse
└── content: MessageContent     ├── ThoughtSignature
                                ├── ReasoningContent
                                └── Custom(CustomPart)

ToolCall                     Tool                        ChatOptions (部分)
├── call_id: String          ├── name: ToolName          ├── temperature
├── fn_name: String          ├── description             ├── max_tokens
├── fn_arguments: Value      ├── schema: Option<Value>   ├── top_p
└── thought_signatures       ├── strict: Option<bool>    ├── stop_sequences
                             └── config: Option<ToolCfg> ├── capture_usage
                                                         ├── capture_content
                                                         ├── capture_reasoning_content
                                                         ├── capture_tool_calls
                                                         ├── response_format
                                                         ├── reasoning_effort
                                                         ├── verbosity
                                                         ├── seed
                                                         ├── extra_headers
                                                         └── extra_body
```

### 1.3 Resolver 扩展机制

rust-genai 提供三个核心扩展点，允许完全自定义服务路由：

```
ModelMapper           →  修改/重定向模型名称
    ↓
AuthResolver          →  提供认证数据 (API key / Bearer token)
    ↓
default_endpoint      →  基于 AdapterKind 获取默认 endpoint
    ↓
ServiceTargetResolver →  最终修改 url / headers / auth (完全控制)
```

### 1.4 中间流事件层 (InterStreamEvent)

适配器内部的流式处理使用 `InterStreamEvent` 作为中间层：

```
AdapterStreamer (OpenAIStreamer / OpenAIRespStreamer / ...)
    │
    ▼  InterStreamEvent
    ├── Start
    ├── Chunk(String)               ← SSE delta content
    ├── ReasoningChunk(String)
    ├── ThoughtSignatureChunk(String)
    ├── ToolCallChunk(ToolCall)
    └── End(InterStreamEnd)         ← 汇总 captured 数据
            ├── captured_usage
            ├── captured_stop_reason
            ├── captured_text_content
            ├── captured_reasoning_content
            ├── captured_tool_calls
            ├── captured_thought_signatures
            └── captured_response_id
```

## 2. Codex 当前架构

### 2.1 调用链

```
┌──────────────────────────────────────────┐
│           core::ModelClient              │
│  管理 session/turn/auth/provider 选择      │
└────────────────┬─────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────┐
│         codex-api::ResponsesClient        │
│  stream_request(ResponsesApiRequest, ...) │
│  → stream(body, headers, compression)     │
└────────────────┬─────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────┐
│   EndpointSession::stream_with()          │
│   → reqwest HTTP POST /v1/responses       │
│   → spawn_response_stream()               │
└────────────────┬─────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────┐
│   process_sse() → process_responses_event()│
│   SSE 解析 → ResponseEvent                │
│   (Created, OutputItemDone,               │
│    OutputTextDelta, Completed, ...)       │
└──────────────────────────────────────────┘
```

### 2.2 WireApi 现状

```rust
// codex-rs/model-provider-info/src/lib.rs
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum WireApi {
    #[default]
    Responses,  // ← 唯一的变体
}

// Chat 变体在反序列化时返回错误:
// "`wire_api = \"chat\"` is no longer supported."
```

### 2.3 核心协议类型

```rust
// ResponsesApiRequest - 发送给 /v1/responses
pub struct ResponsesApiRequest {
    pub model: String,
    pub instructions: String,
    pub input: Vec<ResponseItem>,  // ← 核心数据模型
    pub tools: Vec<Value>,
    pub tool_choice: String,
    pub parallel_tool_calls: bool,
    pub reasoning: Option<Reasoning>,
    pub store: bool,
    pub stream: bool,
    pub include: Vec<String>,
    pub service_tier: Option<String>,
    pub prompt_cache_key: Option<String>,
    pub text: Option<TextControls>,
    pub client_metadata: Option<HashMap<String, String>>,
}

// ResponseEvent - 流式事件
pub enum ResponseEvent {
    Created,
    OutputItemDone(ResponseItem),
    OutputItemAdded(ResponseItem),
    ServerModel(String),
    ModelVerifications(Vec<ModelVerification>),
    ServerReasoningIncluded(bool),
    Completed { response_id, token_usage, end_turn },
    OutputTextDelta(String),
    ToolCallInputDelta { item_id, call_id, delta },
    ReasoningSummaryDelta { delta, summary_index },
    ReasoningContentDelta { delta, content_index },
    ReasoningSummaryPartAdded { summary_index },
    RateLimits(RateLimitSnapshot),
    ModelsEtag(String),
}
```

## 3. 类型映射对照表

### 3.1 请求方向: Codex → rust-genai

| Codex 字段 | rust-genai 字段 | 转换方式 |
|---|---|---|
| `instructions: String` | `ChatRequest.system: Option<String>` | 直接映射 |
| `input: Vec<ResponseItem>` | `ChatRequest.messages: Vec<ChatMessage>` | **需转换器** |
| `tools: Vec<Value>` | `ChatRequest.tools: Option<Vec<Tool>>` | 需转换器 |
| `model: String` | `model: impl Into<ModelSpec>` | 直接映射 |
| `reasoning.effort` | `ChatOptions.reasoning_effort` | 直接映射 |
| `reasoning.summary` | `ChatOptions.capture_reasoning_content` | 语义对应 |
| `text.verbosity` | `ChatOptions.verbosity` | 直接映射 |
| `text.format` | `ChatOptions.response_format` | 需转换器 |
| `store` | `ChatRequest.store` | 直接映射 |
| `parallel_tool_calls` | ❌ rust-genai 无对应字段 | 丢失 |
| `tool_choice: String` | ❌ rust-genai 无对应字段 | 丢失/通过 extra_body |
| `include: Vec<String>` | 内部处理 | 自动设置 |
| `client_metadata` | ❌ rust-genai 无对应字段 | 丢失/通过 extra_headers |
| `service_tier` | `ChatOptions.service_tier` | 直接映射 |
| `prompt_cache_key` | `ChatOptions.prompt_cache_key` | 直接映射 |

### 3.2 流式事件方向: rust-genai → Codex

| rust-genai ChatStreamEvent | Codex ResponseEvent | 转换方式 |
|---|---|---|
| `Start` | `Created` | 直接映射 |
| `Chunk(StreamChunk)` | `OutputTextDelta(String)` | 直接映射 |
| `ReasoningChunk(StreamChunk)` | `ReasoningContentDelta { delta, content_index }` | content_index 用自增计数 |
| `ToolCallChunk(ToolChunk)` | `ToolCallInputDelta { item_id, call_id, delta }` | 需组装 |
| `End(StreamEnd)` | `Completed { response_id, token_usage, end_turn }` | 字段映射 |
| - | `OutputItemDone(ResponseItem)` | ToolCall 完成时需要额外发送 |
| - | `ServerModel(String)` | rust-genai SSE 不暴露此 header |
| - | `RateLimits(RateLimitSnapshot)` | rust-genai SSE 不暴露 |
| - | `ModelsEtag(String)` | rust-genai SSE 不暴露 |

### 3.3 ResponseItem ↔ ChatMessage 转换 (核心难点)

```
ResponseItem::Message { role, content }
  ──→ ChatMessage { role: ChatRole, content: MessageContent }

    content: [{"type": "output_text", "text": "Hello"}, {"type": "input_image", "image_url": "..."}]
      ──→ MessageContent::from_parts([
             ContentPart::Text("Hello"),
             ContentPart::Binary(Binary::from_url("..."))
           ])

ResponseItem::FunctionCall { call_id, name, arguments }
  ──→ ChatMessage::assistant(MessageContent::from(ToolCall { call_id, fn_name, fn_arguments }))

ResponseItem::FunctionCallOutput { call_id, output }
  ──→ ChatMessage::tool(MessageContent::from(ToolResponse { call_id, content: output }))

ResponseItem::Reasoning { encrypted_content }
  ──→ ChatMessage::assistant(ContentPart::ThoughtSignature(encrypted_content))

ResponseItem::WebSearchCall { ... }
  ──→ ChatMessage::tool(...) 或忽略 (取决于 Codex 处理方式)
```

## 4. 三个可行方案

### 方案 A: 完全替换 (激进)

**思路:** 用 rust-genai 完全替换 Codex 的 ResponsesClient、SSE 处理、请求构建。

```
Codex Core → rust-genai Client → OpenAIRespAdapter/OpenAIAdapter → HTTP
```

| 维度 | 评估 |
|---|---|
| 代码改动量 | ~1500行删除 + ~500行新增 |
| 风险 | **高** - ResponseItem 是 Codex 协议层核心类型 |
| 多Provider | ✅ 完整 |
| Chat API | ✅ |
| 维护成本 | 低 (单一实现) |

**主要风险:** `ResponseItem` 和 `ResponseEvent` 被 `compaction`、`memory`、`realtime`、`app-server` 等 20+ 模块使用，大面积替换风险极高。

---

### 方案 B: Adapter 层 (推荐) ⭐

**思路:** 在 Codex 现有抽象下面引入 rust-genai，添加双向转换器，保留 `ResponseItem` / `ResponseEvent` 作为上层接口。

```
Codex Core
  │ ResponseItem[], ResponseEvent (保持不变!)
  ▼
┌─────────────────────────────────────┐
│  ResponseItem → ChatRequest 转换器   │  ← 新增 ~300行
│  ChatStreamEvent → ResponseEvent 转换器│  ← 新增 ~200行
└──────────────┬──────────────────────┘
               │ ChatRequest, ChatStreamEvent
               ▼
┌──────────────────────────────────────┐
│  rust-genai Client (多Provider)       │  ← 引入
│  + 自定义 ServiceTargetResolver       │
│  + 自定义 AuthResolver               │
└──────────────────────────────────────┘
```

**新增文件:**
- `codex-api/src/genai_bridge.rs` — 双向类型转换器 (~300行)
- `codex-api/src/genai_client.rs` — 封装 rust-genai Client (~200行)

**修改文件:**
- `codex-rs/core/src/client.rs` — ModelClient 集成 (~80行改动)
- `codex-rs/model-provider-info/src/lib.rs` — WireApi 加回 Chat (~10行)
- `codex-api/Cargo.toml` — 添加 genai 依赖

| 维度 | 评估 |
|---|---|
| 代码改动量 | ~200行删除 + ~800行新增 |
| 风险 | **中** - 上层零改动，仅底层替换 |
| 多Provider | ✅ 完整 |
| Chat API | ✅ |
| 维护成本 | 中 (需维护转换层) |

---

### 方案 C: 仅 Chat API (保守)

**思路:** rust-genai 只用于 `WireApi::Chat` 路径，现有 Responses API 实现不变。

| 维度 | 评估 |
|---|---|
| 代码改动量 | ~20行删除 + ~200行新增 |
| 风险 | **低** |
| 多Provider | 仅 Chat API 享受 |
| Chat API | ✅ |
| 维护成本 | **高** (两套实现长期并存) |

---

## 5. 方案对比总结

| 维度 | A (替换) | B (Adapter层) | C (仅Chat) |
|---|---|---|---|
| 风险 | 高 | 中 | 低 |
| Chat API 支持 | ✅ | ✅ | ✅ |
| Responses 统一 | ✅ | ✅ (渐进) | ❌ |
| 多Provider | ✅ | ✅ | 仅Chat |
| 消费者改动 | 大量 | 零 | 零 |
| 维护成本 | 低 | 中 | 高 |
| 实施周期 | 3-4周 | 2-3周 | 1周 |

## 6. 推荐: 方案 B + 分阶段实施

### Phase 1: Chat API 回归 (1-2天)

- 新增 `WireApi::Chat` variant
- 使用 rust-genai `OpenAI` adapter 调用 `/v1/chat/completions`
- 新增 `ResponseItem[]` ↔ `ChatRequest` 转换器
- 改动范围: ~300行

### Phase 2: Responses API 适配 (1周)

- 新增 `GenaiResponsesClient`，使用 rust-genai `OpenAIResp` adapter
- 复用 Phase 1 的转换器
- 与现有 `ResponsesClient` 通过 feature flag 并存
- 改动范围: ~500行

### Phase 3: 多 Provider (3-5天)

- 自定义 `ServiceTargetResolver` 映射 Codex provider 配置
- 支持 Anthropic、Gemini、Ollama、DeepSeek 等
- 改动范围: ~200行

## 7. 关键风险与缓解

| 风险 | 缓解措施 |
|---|---|
| `parallel_tool_calls` 不兼容 | 通过 `extra_body` 注入 Responses API 特有字段 |
| `ServerModel` / `RateLimits` 事件丢失 | 扩展 `ChatOptions.extra_headers` 或直接从 HTTP response 解析 |
| 转换器性能开销 | 转换器是零拷贝引用映射，开销可忽略 |
| rust-genai 版本不兼容 | 固定版本，或 fork 到 codex 仓库 |
| `ResponseItem` 变体不被 rust-genai 原生支持 | 通过 `ContentPart::Custom` 透传或忽略 |
