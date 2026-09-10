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

    private func makeRequest(
        model: Model,
        messages: [Message],
        tools: [Tool] = [],
        toolChoice: Tool? = nil,
        options: [String: Value] = [:]
    ) -> Anthropic.ChatRequest {
        var req = Anthropic.ChatRequest(
            model: model.id,
            messages: encode(messages: messages),
            // Bounds the reasoning and the reply together, so a figure chosen
            // when replies were the only output can now be spent thinking and
            // leave the answer truncated. The caller's setting wins; failing
            // that the model's own ceiling; failing that a figure with room in
            // it for both.
            max_tokens: options["max_tokens"]?.intValue ?? model.maxOutput ?? 16384,
            system: encodeSystemPrompt(from: messages),
            tool_choice: encode(toolChoice),
            tools: encode(tools)
        )

        // Reasoning, and how much work to put in overall. Two settings rather
        // than one: `thinking` decides whether the model reasons at all, and
        // `effort` how deeply — including how deeply it reasons when it does.
        //
        // `summarized` because the default is `omitted`, under which the
        // blocks still arrive carrying a signature and no text. A caller that
        // means to show reasoning has to say so.
        switch effort(from: options) {
        case .none:
            break
        case .off:
            req.thinking = .disabled
        case .level(let level):
            req.thinking = .adaptive(display: "summarized")
            req.output_config = .init(effort: level)
        }

        return req
    }

    /// The temperature to send, if any.
    ///
    /// Newer models reject a non-default `temperature`, `top_p` or `top_k`
    /// with a 400 on **every** request, whether or not thinking is involved —
    /// so passing one through unconditionally turns the first model that says
    /// so into a service that can't answer at all. Since nothing has needed to
    /// set one here, the safe reading of a value we can't check per model is
    /// to drop it and say so, rather than to fail the send.
    private func temperature(from options: [String: Value], model: Model) -> Double? {
        guard let temperature = options["temperature"]?.doubleValue else { return nil }
        logger.warning(
            "\(model.id, privacy: .public): dropping temperature \(temperature, privacy: .public) — models from Claude Sonnet 5 and Opus 4.7 onwards reject any non-default sampling value"
        )
        return nil
    }

    /// What the caller asked for, in this provider's terms.
    private enum Effort {
        case none
        case off
        case level(String)
    }

    /// Maps the caller's effort onto what this API takes.
    ///
    /// `brief` and `full` are the steered pair local models use, where the
    /// grading is done by instruction rather than by the request — here there
    /// is a real control, so they land on the levels that mean the same thing.
    private func effort(from options: [String: Value]) -> Effort {
        guard let raw = options["effort"]?.stringValue else { return .none }
        switch raw {
        case "off":
            return .off
        case "brief":
            return .level("low")
        case "full":
            // The API's own default, and the docs are explicit that passing it
            // is identical to passing nothing. Said out loud anyway, so the
            // request describes what was asked for.
            return .level("high")
        case "low", "medium", "high", "xhigh", "max":
            return .level(raw)
        default:
            logger.warning("ignoring unrecognised effort \(raw, privacy: .public)")
            return .none
        }
    }
}

// MARK: - Services

extension AnthropicService: ChatService {

    public func completion(_ request: ChatServiceRequest) async throws -> Message {
        var req = makeRequest(
            model: request.model,
            messages: request.messages,
            tools: request.tools,
            options: request.options
        )
        req.temperature = temperature(from: request.options, model: request.model)
        let result = try await client.chatCompletions(req)
        return decode(result)
    }

    public func completionStream(_ request: ChatServiceRequest, update: (Message) async throws -> Void) async throws {
        var req = makeRequest(
            model: request.model,
            messages: request.messages,
            tools: request.tools,
            options: request.options
        )
        req.temperature = temperature(from: request.options, model: request.model)
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
