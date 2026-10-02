import Foundation
import OSLog
import Mistral

private let logger = Logger(subsystem: "MistralService", category: "GenKit")

public actor MistralService {
    
    private let client: Mistral.Client

    public init(session: URLSession? = nil, host: URL? = nil, apiKey: String) {
        self.client = .init(session: session, host: host, apiKey: apiKey)
    }
}

// MARK: - Services

extension MistralService: ChatService {
    
    public func completion(_ request: ChatServiceRequest) async throws -> Message {
        var req = ChatRequest(
            model: request.model.id,
            messages: encode(messages: request.messages),
            tools: encode(request.tools),
            tool_choice: encode(request.toolChoice)
        )
        req.temperature = request.options["temperature"]?.doubleValue

        logger.info("→ mistral completion | model: \(req.model, privacy: .public) | messages: \(req.messages.count, privacy: .public) | tools: \(req.tools?.count ?? 0, privacy: .public)")

        let resp: Mistral.ChatResponse
        do {
            resp = try await client.chatCompletions(req)
        } catch {
            logger.error("← mistral completion failed: \(error, privacy: .public)")
            throw error
        }
        guard let message = decode(resp) else {
            throw ChatServiceError.responseError("Missing response choice")
        }
        return message
    }
    
    public func completionStream(_ request: ChatServiceRequest, update: (Message) async throws -> Void) async throws {
        var req = ChatRequest(
            model: request.model.id,
            messages: encode(messages: request.messages),
            tools: encode(request.tools),
            tool_choice: encode(request.toolChoice)
        )
        req.temperature = request.options["temperature"]?.doubleValue
        req.stream = true
        
        logger.info("→ mistral stream | model: \(req.model, privacy: .public) | messages: \(req.messages.count, privacy: .public) | tools: \(req.tools?.count ?? 0, privacy: .public)")

        // Counted so a failure says how far it got. A stream that dies on its
        // first chunk and one that dies near the end are different problems,
        // and the decode failure that prompted this reached the screen with
        // nothing in the log to say where it happened.
        var chunks = 0
        var message = Message(role: .assistant)
        do {
            for try await resp in try client.chatCompletionsStream(req) {
                chunks += 1
                patchMessage(&message, with: resp)
                try await update(message)
            }
        } catch {
            logger.error("← mistral stream failed after \(chunks, privacy: .public) chunks: \(error, privacy: .public)")
            throw error
        }
        logger.info("← mistral stream finished | \(chunks, privacy: .public) chunks")
    }
}

extension MistralService: EmbeddingService {
    
    public func embeddings(_ request: EmbeddingServiceRequest) async throws -> [Double] {
        let req = EmbeddingsRequest(model: request.model.id, input: [request.input])
        let result = try await client.embeddings(req)
        return result.data.first?.embedding ?? []
    }
}

extension MistralService: ModelService {
    
    public func models() async throws -> [Model] {
        decodeModels(try await client.models())
    }

    /// Everything `models()` does except the request, so the alias folding
    /// and the capability reading can be driven from a captured response.
    nonisolated func decodeModels(_ response: Mistral.ModelsResponse) -> [Model] {
        deduplicated(response.data).map { decode($0) }
    }

    /// One entry per model, where Mistral lists one per alias.
    ///
    /// Measured 1 Oct 2026 against `api.mistral.ai`: **53 entries, 20 distinct
    /// names**, and within every group sharing a name the context length and
    /// the capabilities were identical — only `id` differed. So
    /// `codestral-2508`, `codestral-latest`, `mistral-code-latest` and
    /// `mistral-code-fim-latest` are four ways to address one model, and
    /// `mistral-medium-latest` had nine, including `magistral-medium-latest`
    /// and `mistral-vibe-cli-latest`. Settings showed all of them, under
    /// identical labels, because the list displays `name`.
    ///
    /// Which one to keep is the provider's answer rather than a rule invented
    /// here: in every one of those groups exactly one entry had `id == name`,
    /// `name` being Mistral's own canonical label. That does mean the choice
    /// between a pinned id and a floating `-latest` follows Mistral's
    /// inconsistency — `mistral-large-2512` is canonical where
    /// `mistral-medium-latest` is — which is the right way round: reporting
    /// what the service says beats deciding on its behalf.
    ///
    /// Entries with no name are kept as they are, having nothing to group by.
    nonisolated private func deduplicated(_ models: [Mistral.ModelsResponse.Model]) -> [Mistral.ModelsResponse.Model] {
        var byName: [String: Mistral.ModelsResponse.Model] = [:]
        var order: [String] = []
        var unnamed: [Mistral.ModelsResponse.Model] = []

        for model in models {
            guard let name = model.name else {
                unnamed.append(model)
                continue
            }
            if byName[name] == nil {
                order.append(name)
                byName[name] = model
            } else if model.id == name {
                byName[name] = model
            }
        }
        return order.compactMap { byName[$0] } + unnamed
    }
}

// MARK: - Encoders

extension MistralService {
    
    private func encode(messages: [Message]) -> [Mistral.ChatRequest.Message] {
        messages.map { message in
            Mistral.ChatRequest.Message(
                content: encode(message.contents),
                tool_calls: encode(message.toolCalls),
                prefix: nil,
                role: encode(message.role)
            )
        }
    }
    
    private func encode(_ role: Message.Role) -> Mistral.ChatRequest.Message.Role {
        switch role {
        case .system: .system
        case .user: .user
        case .assistant: .assistant
        case .tool: .tool
        }
    }
    
    private func encode(_ contents: [Message.Content]?) -> [Mistral.ChatRequest.Message.Content]? {
        contents?.compactMap { content in
            switch content {
            case .text(let text):
                return .init(type: .text, text: text)
            case .image(let image):
                guard let data = try? Data(contentsOf: image.url) else { return nil }
                return .init(type: .image_url, image_url: .init(url: "data:image/\(image.format);base64,\(data.base64EncodedString())"))
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
            default:
                return nil
            }
        }
    }
    
    private func encode(_ toolCalls: [ToolCall]?) -> [Mistral.ChatRequest.Message.ToolCall]? {
        toolCalls?.compactMap { toolCall in
            if let function = toolCall.function {
                Mistral.ChatRequest.Message.ToolCall(
                    id: toolCall.id,
                    function: .init(
                        name: function.name,
                        arguments: function.arguments
                    )
                )
            } else {
                nil
            }
        }
    }

    private func encode(_ tools: [Tool]) -> [Mistral.ChatRequest.Tool] {
        tools.compactMap { tool in
            if let function = tool.function {
                Mistral.ChatRequest.Tool(
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
    
    private func encode(_ toolChoice: Tool?) -> Mistral.ChatRequest.ToolChoice? {
        guard let name = toolChoice?.function?.name else { return nil }
        return .tool(.init(function: .init(name: name)))
    }
}

// MARK: - Decoders

extension MistralService {

    private func decode(_ resp: Mistral.ChatResponse) -> Message? {
        guard let choice = resp.choices.first else { return nil }
        var message = Message(
            role: .assistant,
            content: compose(
                thinking: choice.message.content?.thinking ?? "",
                answer: choice.message.content?.text ?? ""
            ),
            toolCalls: choice.message.tool_calls?.map { decode($0) },
            finishReason: decode(choice.finish_reason)
        )
        // The counts arrive here in one piece rather than on a final chunk,
        // but they are the same figures under the same names.
        record(
            inputTokens: resp.usage?.prompt_tokens,
            outputTokens: resp.usage?.completion_tokens,
            model: resp.model,
            into: &message
        )
        return message
    }
    
    private func decode(_ toolCall: Mistral.ChatResponse.Choice.Message.ToolCall) -> ToolCall {
        ToolCall(
            id: toolCall.id ?? "",
            type: "function",
            function: .init(
                name: toolCall.function.name,
                arguments: toolCall.function.arguments
            ),
            custom: nil
        )
    }
    
    private func decode(_ reason: Mistral.ChatResponse.Choice.FinishReason?) -> Message.FinishReason? {
        guard let reason else { return nil }
        switch reason {
        case .stop:
            return .stop
        case .length, .model_length:
            return .length
        case .error:
            return .error
        case .tool_calls:
            return .toolCalls
        }
    }
    
    /// Folds reasoning into the content as a `<think>` block.
    ///
    /// The shape Ollama's and OpenAI's reasoning already take, so a reader
    /// needn't know which provider a reply came from. Left open until the
    /// answer starts, so a reader streaming the reply can tell reasoning from
    /// answer as it arrives.
    nonisolated private func compose(thinking: String, answer: String) -> String {
        guard !thinking.isEmpty else { return answer }
        let closing = answer.isEmpty ? "" : "\n</think>\n"
        return "<think>\n" + thinking + closing + answer
    }

    /// What the reply cost, under the provider-neutral names the other
    /// providers use.
    ///
    /// Nothing recorded this, so a Mistral reply arrived with no counts and
    /// no model against it and the signature line showed nothing — the same
    /// gap the OpenAI and Anthropic providers had. Mistral has been reporting
    /// all of it the whole time; `swift-mistral` decodes `prompt_tokens`,
    /// `completion_tokens` and `total_tokens`, and nothing here read them.
    ///
    /// Each field written only when it arrives, because this runs on every
    /// chunk of a stream and the totals come on one of them.
    ///
    /// No reasoning split: Mistral reports no such figure, so the caller's
    /// estimate from the stream stands where a model has been reasoning.
    /// Takes the figures rather than either response's `Usage`, which are two
    /// different types describing the same three numbers — and whose
    /// memberwise initialisers aren't public, so one could not be built to
    /// stand in for the other.
    nonisolated private func record(inputTokens: Int?, outputTokens: Int?, model: String?, into message: inout Message) {
        // Which model answered, on every chunk, and worth keeping per message
        // because a conversation can change model between messages.
        if let model, !model.isEmpty {
            message.metadata["model"] = .string(model)
        }
        // Only what arrived. Writing zeros where a response carried no usage
        // would put "0 tokens" on screen, which reads as a measurement rather
        // than as the absence of one.
        if let inputTokens {
            message.metadata["inputTokens"] = .int(inputTokens)
        }
        if let outputTokens {
            message.metadata["outputTokens"] = .int(outputTokens)
        }
    }

    nonisolated private func decode(_ model: Mistral.ModelsResponse.Model) -> Model {
        Model(
            id: model.id,
            name: model.name,
            owner: model.owned_by ?? "mistral",
            contextWindow: model.max_context_length,
            // Mistral reports no output limit, and this used to report the
            // context window as one — so Longest Reply offered a 262,144-token
            // ceiling on a model whose answers are nothing like that. Nil says
            // "not published", which is true, and the clamp that reads this
            // simply doesn't apply.
            maxOutput: nil,
            trainingCutoff: nil,
            capabilities: decode(model.capabilities)
        )
    }

    /// What the service says each model is for.
    ///
    /// Mistral reports this per model and nothing read it, so its embedding,
    /// OCR, transcription and speech models all appeared in the chat picker —
    /// seven of the twenty, once the aliases above are folded together.
    ///
    /// `reasoning` is true of five of the twenty — `mistral-medium-latest`,
    /// `mistral-small-2603`, `glm-5-2`, `zai-glm-5-3` and
    /// `labs-leanstral-1-5-1` — and could not be read until `swift-mistral`
    /// decoded it, because guessing from the name is worse than useless here:
    /// `magistral-medium-latest` is an alias of `mistral-medium-latest`, so
    /// the model that sounds like the reasoning one and the model that is it
    /// are the same thing under two names.
    ///
    /// `fine_tuning` has no counterpart here and is dropped, as are the OCR
    /// and audio keys — a model is kept out of the chat list by what it
    /// *lacks*, `completion_chat`, rather than by what else it can do.
    nonisolated private func decode(_ capabilities: Mistral.ModelsResponse.Model.Capabilities?) -> Set<Model.Capability>? {
        guard let capabilities else { return nil }
        var found: Set<Model.Capability> = []
        if capabilities.completion_chat { found.insert(.completion) }
        if capabilities.function_calling { found.insert(.tools) }
        if capabilities.vision { found.insert(.vision) }
        if capabilities.completion_fim { found.insert(.insert) }
        // Optional upstream, because a key absent from one model's response
        // would otherwise fail the decode for every model in it.
        if capabilities.reasoning == true { found.insert(.thinking) }
        return found
    }
    
    nonisolated private func decode(_ reason: Mistral.ChatStreamResponse.Choice.FinishReason?) -> Message.FinishReason? {
        guard let reason else { return nil }
        switch reason {
        case .stop:
            return .stop
        case .length, .model_length:
            return .length
        case .error:
            return .error
        case .tool_calls:
            return .toolCalls
        }
    }
    
    nonisolated private func decode(_ toolCall: Mistral.ChatStreamResponse.Choice.Message.ToolCall) -> ToolCall {
        ToolCall(
            id: toolCall.id ?? "",
            type: "function",
            function: .init(
                name: toolCall.function.name,
                arguments: toolCall.function.arguments
            ),
            custom: nil
        )
    }
    
    /// Internal rather than private so a test can drive it with a chunk,
    /// which is how the OpenAI provider's counts are covered. The interesting
    /// case has no choices at all and cannot be reached any other way.
    nonisolated func patchMessage(_ message: inout Message, with resp: Mistral.ChatStreamResponse) {
        // Before the guard below, deliberately. A stream reports its totals on
        // a chunk that carries no choices at all, so read after that guard
        // they were never read — which is how every hosted service but Ollama
        // came to answer with no counts against it, and how the same mistake
        // was found in the OpenAI provider.
        record(
            inputTokens: resp.usage?.prompt_tokens,
            outputTokens: resp.usage?.completion_tokens,
            model: resp.model,
            into: &message
        )

        // If there is no choice, return existing message
        guard let choice = resp.choices.first else {
            return
        }

        // Reasoning and answer arrive as separate blocks on the models that
        // reason, so each is accumulated and the pair rebuilt every time —
        // the same thing the Ollama provider does with its own two fields.
        // Accumulated under their own keys rather than patched into the
        // composed text, because the `<think>` wrapper has to be rebuilt as
        // the answer starts and can't be appended to.
        if let delta = choice.delta.content, delta.thinking != nil || message.metadata["thinking"] != nil {
            let thinking = (message.metadata["thinking"]?.stringValue ?? "") + (delta.thinking ?? "")
            let answer = (message.metadata["answer"]?.stringValue ?? "") + (delta.text ?? "")

            message.metadata["thinking"] = .string(thinking)
            message.metadata["answer"] = .string(answer)
            message.contents = [.text(compose(thinking: thinking, answer: answer))]

        // Everything else: a plain string, patched as before.
        } else if case .text(let text) = message.contents?.last, let delta = choice.delta.content?.text {
            if let patched = GenKit.patch(string: text, with: delta) {
                message.contents = [.text(patched)]
            }
        } else if let delta = choice.delta.content?.text {
            message.contents = [.text(delta)]
        }

        // Patch message tool calls
        if let toolCalls = choice.delta.tool_calls {
            if message.toolCalls == nil {
                message.toolCalls = []
            }
            message.toolCalls! += toolCalls.map { decode($0) }
        }

        message.finishReason = decode(choice.finish_reason)
        message.modified = .now
    }
}
