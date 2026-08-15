import Foundation
import OSLog
import SharedKit
import JSONSchema
import Ollama

private let logger = Logger(subsystem: "OllamaService", category: "GenKit")

public actor OllamaService {

    private let client: Ollama.Client

    /// What each model supports, as the server reported it.
    ///
    /// Held for the life of the service: it costs one extra round trip the
    /// first time a model is used, and the answer only changes if the model
    /// itself is replaced, which means restarting the app anyway.
    private var detailsByModel: [String: ModelDetails] = [:]

    /// What one `/api/show` call is worth keeping. Held together because they
    /// arrive together — asking twice for two fields of the same response
    /// would double the round trips this is trying to avoid.
    struct ModelDetails: Sendable {
        var capabilities: Set<Model.Capability>?
        var contextLength: Int?
    }

    public init(session: URLSession? = nil, host: URL? = nil) {
        self.client = .init(session: session, host: host)
    }
}

// MARK: - Services

extension OllamaService: ChatService {
    
    public func completion(_ request: ChatServiceRequest) async throws -> Message {
        let req = await makeChatRequest(request, stream: false)
        let resp = try await client.chatCompletions(req)
        return decode(resp)
    }

    public func completionStream(_ request: ChatServiceRequest, update: (Message) async throws -> Void) async throws {
        let req = await makeChatRequest(request, stream: true)
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
        var models = resp.models.map { decode($0) }

        // Capabilities come from a separate endpoint, one call per model.
        // Worth doing here rather than on demand: it reads metadata only —
        // against a local server nothing is loaded into memory and a call
        // returns in around 13ms — so carrying the answers on the models lets
        // the app ask what one supports without a round trip at the point of
        // use.
        //
        // Together rather than one after another. Locally the difference is
        // nothing, but this is one round trip per model, and against a server
        // reached over a phone connection or a VPN that becomes the whole
        // cost: sequentially it grows with the number of models installed,
        // concurrently it stays roughly the latency of one.
        let found = await withTaskGroup(of: (String, ModelDetails).self) { group in
            for model in models {
                group.addTask { (model.id, await self.details(for: model.id)) }
            }
            var found: [String: ModelDetails] = [:]
            for await (id, details) in group {
                found[id] = details
            }
            return found
        }
        // What a model was actually loaded with, which is not what it says it
        // can do. A model built for 262,144 tokens is routinely loaded with a
        // fraction of that, so reporting the built-in maximum would overstate
        // the room available — by 8x on a model measured here. One call for
        // every loaded model, and only models currently in memory appear.
        let running = (try? await client.modelsRunning().models) ?? []
        let loadedContext = Dictionary(
            running.compactMap { model in model.context_length.map { (model.model, $0) } },
            uniquingKeysWith: { first, _ in first }
        )

        for index in models.indices {
            let id = models[index].id
            guard let details = found[id] else { continue }
            models[index].capabilities = details.capabilities
            // Kept apart deliberately. The ceiling is what the model was built
            // for and is what a caller offering to change the setting has to
            // bound itself by; the loaded figure is what a request is measured
            // against right now. Collapsing them into one number means
            // whichever question you ask second gets the wrong answer.
            models[index].contextWindow = details.contextLength
            models[index].loadedContextWindow = loadedContext[id]
        }
        return models
    }
}

// MARK: - Capabilities

extension OllamaService {

    /// Builds a chat request carrying only what this model can actually accept.
    ///
    /// Ollama treats an unsupported capability as a hard error rather than an
    /// option to ignore: a request with tools attached fails outright against a
    /// model that has none ("<model> does not support tools"), and so does
    /// `think: true` without reasoning support. Since a conversation keeps its
    /// tools and a service its reasoning setting, switching to a smaller model
    /// used to break every send until those were found and changed by hand.
    /// Asking the server what the model can do turns that into nothing.
    private func makeChatRequest(_ request: ChatServiceRequest, stream: Bool) async -> ChatRequest {
        let capabilities = await capabilities(for: request.model.id)

        // No answer — an older server, or the lookup failed — means send what
        // was asked for, exactly as before this check existed. Better to let a
        // request fail the way it used to than to quietly strip tools from a
        // model that would have handled them.
        func supports(_ capability: Model.Capability) -> Bool {
            capabilities?.contains(capability) ?? true
        }

        var tools = encode(request.tools)
        if !tools.isEmpty, !supports(.tools) {
            logger.warning("\(request.model.id, privacy: .public) does not support tools; sending the request without \(tools.count, privacy: .public) of them")
            tools = []
        }

        // `think: false` happens to be tolerated by models that can't reason,
        // but there is nothing for it to switch off, so drop the flag rather
        // than relying on that.
        var think = thinking(from: request.options)
        if think != nil, !supports(.thinking) {
            think = nil
        }

        return ChatRequest(
            model: request.model.id,
            messages: encode(messages: request.messages),
            // Omitted rather than sent empty. An empty array is accepted, but
            // "no tools" is the absence of the key.
            tools: tools.isEmpty ? nil : tools,
            options: runtimeOptions(from: request.options),
            stream: stream,
            think: think
        )
    }

    /// What the server says this model supports, or nil when it won't say.
    private func capabilities(for modelID: String) async -> Set<Model.Capability>? {
        await details(for: modelID).capabilities
    }

    /// What the server reports about a model, remembered per model.
    ///
    /// A failed lookup returns empty rather than throwing, and isn't cached:
    /// nothing here is worth failing a request over, and a server that was
    /// briefly unreachable should be asked again rather than remembered as
    /// having said nothing.
    private func details(for modelID: String) async -> ModelDetails {
        if let known = detailsByModel[modelID] {
            return known
        }
        do {
            let resp = try await client.model(.init(name: modelID))
            let details = ModelDetails(
                capabilities: resp.capabilities.map { Set($0.compactMap(Model.Capability.init(rawValue:))) },
                contextLength: resp.contextLength
            )
            detailsByModel[modelID] = details
            return details
        } catch {
            logger.warning("Could not read details for \(modelID, privacy: .public): \(error, privacy: .public)")
            return ModelDetails()
        }
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

    /// Request options Ollama takes under `options`, which until now were
    /// being dropped on the floor — a caller could ask for a context length
    /// and nothing carried it as far as the request.
    ///
    /// Named individually rather than passed through wholesale. `think` is a
    /// field of its own and would be meaningless here, and forwarding
    /// everything else unseen would mean a typo in a key travelling all the
    /// way to the server to be quietly ignored.
    private func runtimeOptions(from options: [String: Value]) -> [String: JSONValue]? {
        var runtime: [String: JSONValue] = [:]

        // How much context to load the model with. Sent only when somebody
        // asked: leaving it out is what lets the server apply its own default,
        // and naming a different value from the one already loaded costs a
        // reload.
        if let contextLength = options["num_ctx"]?.intValue {
            runtime["num_ctx"] = .int(contextLength)
        }
        return runtime.isEmpty ? nil : runtime
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

        // Which model actually answered, reported on every chunk. Recorded
        // per message because the answer can change within one conversation —
        // the model can be switched between messages, and a conversation
        // whose model has since been removed falls back to another. Without
        // this there is nothing anywhere saying what wrote a given reply.
        usage["model"] = .string(resp.model)

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
