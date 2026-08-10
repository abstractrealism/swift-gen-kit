import Foundation
import OSLog
import SharedKit
import Ollama

private let logger = Logger(subsystem: "OllamaService", category: "GenKit")

public actor OllamaService {
    
    private let client: Ollama.Client

    public init(session: URLSession? = nil, host: URL? = nil) {
        self.client = .init(session: session, host: host)
    }
}

// MARK: - Services

extension OllamaService: ChatService {
    
    public func completion(_ request: ChatServiceRequest) async throws -> Message {
        let req = ChatRequest(
            model: request.model.id,
            messages: encode(messages: request.messages),
            tools: encode(request.tools),
            stream: false,
            think: thinking(from: request.options)
        )
        let resp = try await client.chatCompletions(req)
        return decode(resp)
    }
    
    public func completionStream(_ request: ChatServiceRequest, update: (Message) async throws -> Void) async throws {
        let req = ChatRequest(
            model: request.model.id,
            messages: encode(messages: request.messages),
            tools: encode(request.tools),
            stream: true,
            think: thinking(from: request.options)
        )
        var message = Message(role: .assistant)
        for try await resp in try client.chatCompletionsStream(req) {
            patchMessage(&message, with: resp)
            try await update(message)
            
            // The connection hangs if we don't explicitly return when the stream has stopped.
            if message.finishReason == .stop {
                return
            }
        }
    }
}

extension OllamaService: EmbeddingService {
    
    public func embeddings(_ request: EmbeddingServiceRequest) async throws -> [Double] {
        let req = EmbeddingsRequest(model: request.model.id, input: request.input)
        let result = try await client.embeddings(req)
        return result.embedding
    }
}

extension OllamaService: ModelService {
    
    public func models() async throws -> [Model] {
        let resp = try await client.models()
        return resp.models.map { decode($0) }
    }
}

// MARK: - Encoders

extension OllamaService {

    private func encode(messages: [Message]) -> [Ollama.Message] {
        messages.map { message in
            Ollama.Message(
                role: encode(message.role),
                content: encode(message.contents),
                images: encode(message.contents)
            )
        }
    }

    private func encode(_ role: Message.Role) -> Ollama.Message.Role {
        switch role {
        case .system: .system
        case .user: .user
        case .assistant: .assistant
        case .tool: .tool
        }
    }

    private func encode(_ contents: [Message.Content]?) -> String {
        contents?.compactMap {
            switch $0 {
            case .text(let text):
                return text
            case .json(let json):
                return """
                ```json
                \(json.object)
                ```
                """
            case .file(let file):
                guard
                    let data = try? Data(contentsOf: .documentsDirectory.appending(path: file.path)),
                    let content = String(data: data, encoding: .utf8) else { return nil }
                return """
                ```\(file.mimetype.preferredFilenameExtension ?? "txt") \(file.path)
                \(content)
                ``` 
                """
            default: 
                return nil
            }
        }.joined() ?? ""
    }

    private func encode(_ contents: [Message.Content]?) -> [Data]? {
        let images = contents?.compactMap {
            switch $0 {
            case .image(let image):
                return try? Data(contentsOf: image.url)
            default:
                return nil
            }
        }
        return images?.isEmpty == true ? nil : images
    }

    private func encode(_ tools: [Tool]) -> [Ollama.Tool] {
        tools.compactMap { tool in
            // Does not support custom tool functions
            if let function = tool.function {
                Ollama.Tool(
                    type: "function",
                    function: .init(
                        name: function.name,
                        description: function.description,
                        parameters: function.parameters
                    )
                )
            } else {
                nil
            }
        }
    }
}

// MARK: - Decoders

extension OllamaService {

    private func decode(_ resp: Ollama.ChatResponse) -> Message {
        Message(
            role: decode(resp.message?.role) ?? .assistant,
            content: compose(
                thinking: resp.message?.thinking ?? "",
                answer: resp.message?.content ?? ""
            ),
            toolCalls: resp.message?.tool_calls?.map { decode($0) },
            finishReason: decode(resp.done),
            metadata: usage(from: resp)
        )
    }

    private func decode(_ role: Ollama.Message.Role?) -> Message.Role? {
        guard let role else { return nil }
        switch role {
        case .system: return .system
        case .user: return .user
        case .assistant: return .assistant
        case .tool: return .tool
        }
    }

    private func decode(_ toolCall: Ollama.ToolCall) -> ToolCall {
        .init(
            type: "function",
            function: .init(
                name: toolCall.function.name,
                arguments: toolCall.function.arguments
            ),
            custom: nil
        )
    }

    private func decode(_ done: Bool?) -> Message.FinishReason? {
        guard let done, done else { return nil }
        return .stop
    }

    private func decode(_ model: Ollama.ModelResponse) -> Model {
        .init(
            id: model.model,
            family: model.details?.family,
            name: model.name,
            owner: "ollama",
            contextWindow: nil,
            maxOutput: nil,
            trainingCutoff: nil
        )
    }

    /// Only ever returns false or nil. A model that can't reason rejects a
    /// request that asks it to ("<model> does not support thinking"), so the
    /// flag is sent only to turn reasoning off; leaving it out lets each model
    /// keep its own default.
    private func thinking(from options: [String: Value]) -> Bool? {
        guard case .bool(false) = options["think"] else { return nil }
        return false
    }

    /// Folds reasoning into the content as a `<think>` block.
    ///
    /// Ollama returns reasoning in its own field, but plenty of models inline
    /// the tag in the answer instead, and readers already have to handle that.
    /// Emitting one shape means a caller doesn't need to know which kind of
    /// model produced the reply. The block is left open until the answer
    /// starts, so a reader streaming the reply can tell reasoning from answer
    /// as it arrives.
    private func compose(thinking: String, answer: String) -> String {
        guard !thinking.isEmpty else { return answer }
        let closing = answer.isEmpty ? "" : "\n</think>\n"
        return "<think>\n" + thinking + closing + answer
    }

    /// Ollama reports token counts and timings on the final chunk of a
    /// response. Carry them on the message so a caller can show what a
    /// generation cost, under provider-neutral names since other services
    /// report the same things.
    private func usage(from resp: Ollama.ChatResponse) -> [String: Value] {
        var usage: [String: Value] = [:]
        if let input = resp.prompt_eval_count {
            usage["inputTokens"] = .int(input)
        }
        if let output = resp.eval_count {
            usage["outputTokens"] = .int(output)
        }
        if let nanos = resp.eval_duration, nanos > 0 {
            usage["outputSeconds"] = .double(Double(nanos) / 1_000_000_000)
        }
        return usage
    }

    private func patchMessage(_ message: inout Message, with resp: Ollama.ChatResponse) {

        // Reasoning and answer stream as separate deltas, so accumulate each
        // and rebuild the combined content from both every time.
        if resp.message?.thinking != nil || message.metadata["thinking"] != nil {
            let thinking = (message.metadata["thinking"]?.stringValue ?? "") + (resp.message?.thinking ?? "")
            let answer = (message.metadata["answer"]?.stringValue ?? "") + (resp.message?.content ?? "")

            message.metadata["thinking"] = .string(thinking)
            message.metadata["answer"] = .string(answer)
            message.contents = [.text(compose(thinking: thinking, answer: answer))]
        }

        // Patch message content
        else if case .text(let text) = message.contents?.last {
            if let patched = GenKit.patch(string: text, with: resp.message?.content) {
                message.contents = [.text(patched)]
            }
        } else if let text = resp.message?.content {
            message.contents = [.text(text)]
        }

        // Patch message tool calls
        if let toolCalls = resp.message?.tool_calls {
            if message.toolCalls == nil {
                message.toolCalls = []
            }
            message.toolCalls! += toolCalls.map { decode($0) }
        }

        message.metadata.merge(usage(from: resp)) { _, new in new }
        message.finishReason = decode(resp.done)
        message.modified = .now
    }
}
