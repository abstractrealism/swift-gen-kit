import Foundation
import OSLog
import Anthropic
import JSONSchema
import SharedKit

private let logger = Logger(subsystem: "AnthropicService", category: "GenKit")

public actor AnthropicService {

    private let client: Anthropic.Client

    public init(session: URLSession? = nil, host: URL? = nil, apiKey: String) {
        self.client = .init(session: session, host: host, apiKey: apiKey)
    }

    private func makeRequest(model: Model, messages: [Message], tools: [Tool] = [], toolChoice: Tool? = nil) -> Anthropic.ChatRequest {
        .init(
            model: model.id,
            messages: encode(messages: messages),
            max_tokens: model.maxOutput ?? 8192,
            system: encodeSystemPrompt(from: messages),
            tool_choice: encode(toolChoice),
            tools: encode(tools)
        )
    }
}

// MARK: - Services

extension AnthropicService: ChatService {

    public func completion(_ request: ChatServiceRequest) async throws -> Message {
        var req = makeRequest(model: request.model, messages: request.messages, tools: request.tools)
        req.temperature = request.options["temperature"]?.doubleValue
        let result = try await client.chatCompletions(req)
        return decode(result)
    }

    public func completionStream(_ request: ChatServiceRequest, update: (Message) async throws -> Void) async throws {
        var req = makeRequest(model: request.model, messages: request.messages, tools: request.tools)
        req.temperature = request.options["temperature"]?.doubleValue
        req.stream = true

        var message = Message(role: .assistant)
        for try await result in try client.chatCompletionsStream(req) {
            patchMessage(&message, with: result)
            try await update(message)
        }
    }
}

extension AnthropicService: ModelService {

    public func models() async throws -> [Model] {
        let result = try await client.models()
        return result.data.map { decode($0) }
    }
}

// MARK: - Encoders

extension AnthropicService {

    private func encode(messages: [Message]) -> [Anthropic.ChatRequest.Message] {
        return messages
            .filter { $0.role != .system }
            .map { message in
                Anthropic.ChatRequest.Message(
                    role: encode(message.role),
                    content: message.contents?.compactMap { encode($0) } ??
                            message.toolCalls?.compactMap { encode($0) } ??
                            [encode(message)].compactMap { $0 }
                )
            }
    }

    private func encodeSystemPrompt(from messages: [Message]) -> [Anthropic.ChatRequest.Message.Content]? {
        guard let message = messages.first, message.role == .system else { return nil }
        guard case .text(let text) = message.contents?.first else { return nil }
        return [.init(type: .text, text: text)]
    }

    private func encode(_ role: Message.Role) -> Anthropic.ChatRequest.Message.Role {
        switch role {
        case .system, .user, .tool:
            return .user
        case .assistant:
            return .assistant
        }
    }

    private func encode(_ content: Message.Content) -> Anthropic.ChatRequest.Message.Content? {
        switch content {
        case .text(let text):
            return .init(type: .text, text: text)
        case .image(let image):
            if let data = try? Data(contentsOf: image.url) {
                return .init(
                    type: .image,
                    source: .init(
                        type: .base64,
                        media_type: .init(rawValue: image.format.rawValue) ?? .png,
                        data: data
                    )
                )
            } else {
                return nil
            }
        case .audio:
            return nil
        case .json(let json):
            return .init(type: .text, text: json.object)
        case .file(let file):
            guard
                let data = try? Data(contentsOf: .documentsDirectory.appending(path: file.path)),
                let content = String(data: data, encoding: .utf8) else { return nil }
            return .init(type: .text, text: """
                ```\(file.mimetype.preferredFilenameExtension ?? "txt") \(file.path)
                \(content)
                ```
                """)
        }
    }

    private func encode(_ toolCall: ToolCall) -> Anthropic.ChatRequest.Message.Content? {
        guard let data = toolCall.function?.arguments.data(using: .utf8) else { return nil }
        guard let input = try? JSONDecoder().decode([String: JSONValue].self, from: data) else { return nil }
        return .init(
            type: .tool_use,
            id: toolCall.id,
            name: toolCall.function?.name,
            input: input
        )
    }

    private func encode(_ message: Message) -> Anthropic.ChatRequest.Message.Content? {
        guard message.role == .tool else { return nil }
        return .init(
            type: .tool_result,
            tool_use_id: message.toolCallID,
            content: message.contents?.compactMap { encode($0) }
        )
    }

    private func encode(_ tools: [Tool]) -> [Anthropic.ChatRequest.Tool] {
        tools.compactMap { tool in
            if let function = tool.function {
                .init(
                    name: function.name,
                    description: function.description,
                    input_schema: function.parameters
                )
            } else {
                nil
            }
        }
    }

    private func encode(_ toolChoice: Tool?) -> Anthropic.ChatRequest.ToolChoice? {
        guard let toolChoice else {
            return .init(type: .auto)
        }
        return .init(
            type: .tool,
            name: toolChoice.function?.name
        )
    }
}

// MARK: - Decoders

extension AnthropicService {

    private func decode(_ resp: Anthropic.ChatResponse) -> Message {
        Message(
            role: decode(resp.role) ?? .assistant,
            contents: resp.content?.compactMap { decode($0) } ?? [],
            toolCalls: resp.content?.compactMap { decodeToolCall($0) },
            finishReason: decode(resp.stop_reason)
        )
    }

    private func decode(_ role: Anthropic.ChatResponse.Role?) -> Message.Role? {
        guard let role else { return nil }
        switch role {
        case .assistant:
            return .assistant
        case .user:
            return .user
        }
    }

    private func decode(_ content: Anthropic.ChatResponse.Content) -> Message.Content? {
        switch content.type {
        case .text, .text_delta:
            if let text = content.text {
                return .text(text)
            } else {
                return nil
            }
        case .thinking:
            // Folded into the content as a `<think>` block, the same shape the
            // Ollama service emits, so a caller doesn't have to know which
            // kind of model produced the reply.
            guard let thinking = content.thinking, !thinking.isEmpty else { return nil }
            return .text("<think>\n" + thinking + "\n</think>")
        case .tool_use, .input_json_delta, .none, .thinking_delta, .redacted_thinking, .signature_delta:
            return nil
        }
    }

    private func decode(_ reason: Anthropic.ChatResponse.StopReason?) -> Message.FinishReason? {
        guard let reason else { return nil }
        switch reason {
        case .end_turn:
            return .stop
        case .max_tokens:
            return .length
        case .stop_sequence:
            return .stop
        case .tool_use:
            return .toolCalls
        }
    }

    private func decodeToolCall(_ content: Anthropic.ChatResponse.Content) -> ToolCall? {
        switch content.type {
        case .tool_use:
            let data = try? JSONEncoder().encode(content.input)
            let arguments = (data != nil) ? String(data: data!, encoding: .utf8)! : ""
            return .init(
                id: content.id ?? .id,
                type: "function",
                function: .init(
                    name: content.name ?? "",
                    arguments: arguments
                ),
                custom: nil
            )
        case .text, .text_delta, .input_json_delta, .none, .thinking, .thinking_delta, .redacted_thinking,
             .signature_delta:
            return nil
        }
    }

    private func decode(_ model: Anthropic.Model) -> Model {
        Model(
            id: model.id,
            name: model.display_name,
            owner: "anthropic"
        )
    }

    /// Closes an open `<think>` block, if the message is in one.
    ///
    /// Left alone otherwise, and idempotent: a reply that reasons and then
    /// stops without answering keeps an open tag, which readers already handle
    /// — an unterminated block runs to the end of the text, which is exactly
    /// what an interrupted reply is.
    private func closeThinking(_ message: inout Message) {
        guard case .text(let existing) = message.contents?.last,
              existing.hasPrefix("<think>"),
              !existing.contains("</think>")
        else {
            return
        }
        message.contents![message.contents!.count-1] = .text(existing + "\n</think>")
    }

    private func patchMessage(_ message: inout Message, with resp: ChatResponseStream) {
        switch resp.type {
        case .ping, .message_delta, .content_block_stop:
            return
        case .message_start:
            if let message_resp = resp.message {
                message.id = (message_resp.id != nil) ? message_resp.id! : message.id
                message.finishReason = decode(message_resp.stop_reason)
            }
        case .message_stop:
            message.finishReason = (message.toolCalls != nil) ? .toolCalls : .stop
        case .content_block_start:
            if let contentBlock = resp.content_block {
                switch contentBlock.type {
                case .thinking:
                    // Deliberately nothing. Where thinking is on by default the
                    // display is `omitted`, and an omitted block still opens
                    // and closes — it just carries a signature and no
                    // `thinking_delta` at all. Starting a `<think>` block here
                    // would leave an empty one in every single answer. The
                    // first delta with something in it starts it instead.
                    break
                case .text:
                    // Reasoning comes first and in its own block, so the
                    // answer beginning is where it ends. Appended rather than
                    // assigned, or the answer would replace the reasoning it
                    // was arrived at by.
                    closeThinking(&message)
                    if let text = contentBlock.text {
                        if message.contents?.isEmpty ?? true {
                            message.contents = [.text(text)]
                        } else {
                            message.contents?.append(.text(text))
                        }
                    }
                case .tool_use:
                    var toolCall = ToolCall(
                        type: "function",
                        function: .init(name: contentBlock.name ?? "", arguments: ""),
                        custom: nil
                    )
                    toolCall.id = contentBlock.id ?? toolCall.id
                    if message.toolCalls == nil {
                        message.toolCalls = []
                    }
                    message.toolCalls?.append(toolCall)
                default:
                    break
                }
            }
        case .content_block_delta:
            if let delta = resp.delta {
                switch delta.type {
                case .thinking_delta:
                    // Opened here, by the first reasoning to actually arrive,
                    // and left open until the answer's block closes it.
                    guard let thinking = delta.thinking, !thinking.isEmpty else { break }
                    if case .text(let existing) = message.contents?.last,
                       existing.hasPrefix("<think>"), !existing.contains("</think>") {
                        message.contents![message.contents!.count-1] = .text(existing + thinking)
                    } else {
                        message.contents = (message.contents ?? []) + [.text("<think>\n" + thinking)]
                    }
                case .text_delta:
                    if case .text(let existing) = message.contents?.last {
                        let patched = GenKit.patch(string: existing, with: delta.text) ?? existing
                        message.contents![message.contents!.count-1] = .text(patched)
                    } else if let text = delta.text {
                        message.contents?.append(.text(text))
                    }
                case .input_json_delta:
                    if var existing = message.toolCalls?.last {
                        if let function = existing.function {
                            existing.function!.arguments = GenKit.patch(string: function.arguments, with: delta.partial_json) ?? ""
                        }
                        message.toolCalls![message.toolCalls!.count-1] = existing
                    }
                default:
                    break
                }
            }
        }
        message.modified = .now
    }
}
