import Foundation
import OSLog
import OpenAI
import SharedKit

private let logger = Logger(subsystem: "OpenAIService", category: "GenKit")

public actor OpenAIService {

    /// Which OpenAI-shaped endpoint this is talking to.
    ///
    /// One client serves several services that speak the same protocol and
    /// disagree at the edges: which reasoning levels a model takes, which
    /// name the token ceiling goes by, where reasoning comes back. The host
    /// could say, but a proxy or a mirror would then be misread, so the
    /// service that made this says outright.
    public enum Dialect: String, Sendable {
        case openAI, deepSeek, grok, groq, openRouter
    }

    private var client: OpenAI.Client
    private let session: URLSession
    private let dialect: Dialect

    public init(session: URLSession? = nil, host: URL? = nil, apiKey: String, dialect: Dialect = .openAI) {
        self.client = .init(session: session, host: host, apiKey: apiKey)
        self.session = session ?? URLSession(configuration: .default)
        self.dialect = dialect
    }
}

// MARK: - Services

extension OpenAIService: ChatService {

    public func completion(_ request: ChatServiceRequest) async throws -> Message {
            let req = makeRequest(request, streaming: false)
            log(req, streaming: false)
            let resp: OpenAI.ChatResponse
            do {
                resp = try await client.chatCompletions(req)
            } catch {
                logger.error("← \(self.hostName, privacy: .public) completion failed: \(error, privacy: .public)")
                throw error
            }
            logger.info(
                "← \(self.hostName, privacy: .public) completion finished | \(self.describe(resp), privacy: .public)"
            )
            guard let message = decode(resp) else {
                throw ChatServiceError.responseError("Missing response choice")
            }
            return message
        }

        public func completionStream(_ request: ChatServiceRequest, update: (Message) async throws -> Void) async throws {
            let req = makeRequest(request, streaming: true)
            log(req, streaming: true)

            var message = Message(role: .assistant)
            var seen: [String: Int] = [:]
            do {
                for try await resp in try client.chatCompletionsStream(req) {
                    tally(resp, into: &seen)
                    patchMessage(&message, with: resp)
                    try await update(message)
                }
            } catch {
                logger.error(
                    "← \(self.hostName, privacy: .public) stream failed after \(self.describe(seen), privacy: .public): \(error, privacy: .public)"
                )
                throw error
            }
            logger.info("← \(self.hostName, privacy: .public) stream finished | \(self.describe(seen), privacy: .public)")
        }
}

// MARK: - Requests

extension OpenAIService {

    private func makeRequest(_ request: ChatServiceRequest, streaming: Bool) -> OpenAI.ChatRequest {
        var req = OpenAI.ChatRequest(
            messages: encode(messages: request.messages),
            model: request.model.id,
            stream: streaming ? true : nil,
            temperature: request.options["temperature"]?.doubleValue,
            tools: encode(request.tools),
            tool_choice: encode(request.toolChoice),
            verbosity: request.options["verbosity"]?.stringValue.flatMap(OpenAI.ChatRequest.Verbosity.init)
        )
        req.reasoning_effort = reasoningEffort(from: request.options, model: request.model)

        // The ceiling goes by two names. OpenAI deprecated `max_tokens` for
        // `max_completion_tokens` and its reasoning models reject the old one;
        // DeepSeek documents `max_tokens` and nothing else.
        if let ceiling = request.options["max_tokens"]?.intValue {
            switch dialect {
            case .deepSeek:
                req.max_tokens = ceiling
            case .openAI, .grok, .groq, .openRouter:
                req.max_completion_tokens = ceiling
            }
        }
        return req
    }

    /// What `reasoning_effort` to send, if anything.
    ///
    /// The caller's `effort` is provider-neutral — see the levels below — and
    /// what a model accepts is the model's own business: the API refuses what
    /// it doesn't take rather than ignoring it, and a model that doesn't
    /// reason refuses the parameter altogether. So this asks the table first,
    /// and asks for the nearest thing where the exact level isn't offered.
    nonisolated func reasoningEffort(from options: [String: Value], model: Model) -> OpenAI.ChatRequest.ReasoningEffort? {
        // A caller speaking the API's own terms is taken at its word.
        if let raw = options["reasoning_effort"]?.stringValue {
            return .init(rawValue: raw)
        }
        guard let raw = options["effort"]?.stringValue else { return nil }

        guard let levels = Reasoning.supported(byModel: model.id, dialect: dialect).levels else {
            // Either the model takes no such setting — in which case Off is
            // already true of it — or nothing is known, and a guess here is a
            // 400 if wrong where leaving it alone is merely the default.
            if raw != "off" {
                logger.info("\(model.id, privacy: .public) takes no known reasoning setting; leaving it to the model")
            }
            return nil
        }

        let wanted: String
        switch raw {
        case "off":
            wanted = "none"
        case "brief":
            // The steered pair local models use. Here the request grades it.
            wanted = "low"
        case "full":
            // Whatever the model does unprompted, which is what sending
            // nothing asks for.
            return nil
        case "low", "medium", "high", "xhigh", "max":
            wanted = raw
        default:
            logger.warning("ignoring unrecognised effort \(raw, privacy: .public)")
            return nil
        }

        if levels.contains(wanted) {
            return .init(rawValue: wanted)
        }

        // Not offered, so the nearest that is: `xhigh` on a model that stops
        // at `high` gets `high`; `none` on a model that can't stop gets the
        // least it will do. Ties go to the lower level.
        let order = Reasoning.order
        guard let target = order.firstIndex(of: wanted) else { return nil }
        let nearest = levels
            .compactMap { level in order.firstIndex(of: level).map { (level, abs($0 - target)) } }
            .min { $0.1 < $1.1 }?.0
        guard let nearest else { return nil }

        if wanted == "none" {
            logger.info(
                "\(model.id, privacy: .public) can't be told not to reason; asking for the least, \(nearest, privacy: .public)"
            )
        } else {
            logger.info(
                "\(model.id, privacy: .public) doesn't take \(wanted, privacy: .public); asking for \(nearest, privacy: .public)"
            )
        }
        return .init(rawValue: nearest)
    }

    /// Whether a model can be told not to reason, where that's known.
    ///
    /// True where it takes `none`, or `minimal` — which is as near to none as
    /// gpt-5 gets, and nearer than any level a menu offers. False where the
    /// least it takes is `low`. Nil where it takes no such setting at all,
    /// since a model that never reasons has nothing to switch off, or where
    /// nothing is known — only a fact is worth acting on.
    public nonisolated static func canDisableThinking(modelID: String, dialect: Dialect) -> Bool? {
        guard let levels = Reasoning.supported(byModel: modelID, dialect: dialect).levels else { return nil }
        return levels.contains("none") || levels.contains("minimal")
    }

    /// What a model accepts for `reasoning_effort`.
    ///
    /// There is no endpoint that says. The API refuses a value a model doesn't
    /// take, with a message naming the ones it does, and that is where every
    /// row here came from:
    ///
    ///     gpt-5       minimal, low, medium, high
    ///     gpt-5.6     none, low, medium, high, xhigh
    ///     gpt-4o      refuses the parameter altogether
    ///     DeepSeek    none, minimal, low, medium, high, xhigh, max
    ///
    /// DeepSeek's is the endpoint's own vocabulary, checked before any model
    /// is; whether every one of its models honours every level is not yet
    /// measured. The o-series row is from the documentation rather than a
    /// refusal, and `o1-mini` and `o1-preview` are documented as taking none.
    /// Grok, Groq and OpenRouter are unmeasured and get nothing, which is the
    /// safe failure: a level not sent is the model's default, where a level
    /// refused is no answer at all.
    struct Reasoning {
        /// Ascending. Nil where the model takes no such setting.
        let levels: [String]?

        /// Every level any endpoint has named, least to most.
        static let order = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]

        static func supported(byModel id: String, dialect: Dialect) -> Reasoning {
            switch dialect {
            case .deepSeek:
                return Reasoning(levels: order)
            case .grok, .groq, .openRouter:
                return Reasoning(levels: nil)
            case .openAI:
                break
            }

            if id.hasPrefix("o1-mini") || id.hasPrefix("o1-preview") {
                return Reasoning(levels: nil)
            }
            if let first = id.first, first == "o", id.dropFirst().first?.isNumber == true {
                return Reasoning(levels: ["low", "medium", "high"])
            }

            // The generation decides, and it's in the id: 5 for gpt-5-mini,
            // 5.6 for gpt-5.6-sol, 4.1 for gpt-4.1, 4 for gpt-4o. `none` and
            // `xhigh` arrived with 5.1; before that the least was `minimal`.
            // Anything newer than what's been measured is assumed to take the
            // newest vocabulary, an unknown id being likelier new than old.
            guard let generation = gptGeneration(of: id) else {
                return Reasoning(levels: nil)
            }
            if generation >= 5.1 {
                return Reasoning(levels: ["none", "low", "medium", "high", "xhigh"])
            }
            if generation >= 5 {
                return Reasoning(levels: ["minimal", "low", "medium", "high"])
            }
            return Reasoning(levels: nil)
        }

        /// The number after `gpt-`, read as far as it stays a number.
        private static func gptGeneration(of id: String) -> Double? {
            guard id.hasPrefix("gpt-") else { return nil }
            let digits = id.dropFirst(4).prefix { $0.isNumber || $0 == "." }
            return Double(digits)
        }
    }
}

// MARK: - Logging

extension OpenAIService {

    /// Who this is talking to. One client serves OpenAI, DeepSeek, Grok, Groq
    /// and OpenRouter alike, so a line that only said "openai" would name the
    /// wrong service four times out of five.
    private var hostName: String {
        client.host.host() ?? client.host.absoluteString
    }

    /// What was asked for, in the terms the service was asked in.
    ///
    /// Counts and settings only, never the messages: this is for telling why a
    /// request was refused, and a conversation's contents don't answer that.
    private func log(_ req: OpenAI.ChatRequest, streaming: Bool) {
        logger.info("""
            → \(self.hostName, privacy: .public) \(streaming ? "stream" : "completion", privacy: .public) \
            | model: \(req.model, privacy: .public) \
            | reasoning_effort: \(req.reasoning_effort?.rawValue ?? "unset", privacy: .public) \
            | verbosity: \(req.verbosity?.rawValue ?? "unset", privacy: .public) \
            | max_completion_tokens: \(req.max_completion_tokens.map { "\($0)" } ?? "unset", privacy: .public) \
            | max_tokens: \(req.max_tokens.map { "\($0)" } ?? "unset", privacy: .public) \
            | temperature: \(req.temperature.map { "\($0)" } ?? "unset", privacy: .public) \
            | tools: \(req.tools?.count ?? 0, privacy: .public) \
            | messages: \(req.messages.count, privacy: .public)
            """)
    }

    /// What a stream's chunk carried, by kind, so a failure can say how far
    /// the reply had got and what it had received by then. The first
    /// `keyNotFound` from DeepSeek arrived on the last chunk of a complete
    /// answer, and nothing said so.
    private func tally(_ resp: OpenAI.ChatStreamResponse, into seen: inout [String: Int]) {
        seen["chunks", default: 0] += 1
        for choice in resp.choices {
            if choice.delta.content != nil { seen["content", default: 0] += 1 }
            if choice.delta.tool_calls != nil { seen["tool_calls", default: 0] += 1 }
            if choice.delta.refusal != nil { seen["refusal", default: 0] += 1 }
            if let reason = choice.finish_reason { seen["finish:\(reason)", default: 0] += 1 }
        }
        if resp.usage != nil { seen["usage", default: 0] += 1 }
    }

    private func describe(_ seen: [String: Int]) -> String {
        guard !seen.isEmpty else { return "no chunks" }
        return seen
            .sorted { $0.key < $1.key }
            .map { "\($0.key)×\($0.value)" }
            .joined(separator: " ")
    }

    private func describe(_ resp: OpenAI.ChatResponse) -> String {
        var parts = ["choices×\(resp.choices.count)"]
        if let reason = resp.choices.first?.finish_reason { parts.append("finish:\(reason)") }
        if let usage = resp.usage {
            parts.append("tokens: \(usage.prompt_tokens ?? 0) in, \(usage.completion_tokens ?? 0) out")
            if let reasoning = usage.completion_tokens_details?.reasoning_tokens {
                parts.append("reasoning: \(reasoning)")
            }
        }
        return parts.joined(separator: " ")
    }
}

extension OpenAIService: EmbeddingService {

    public func embeddings(_ request: EmbeddingServiceRequest) async throws -> [Double] {
        let req = OpenAI.EmbeddingsRequest(
            input: request.input,
            model: request.model.id
        )
        let result = try await client.embeddings(req)
        return result.data.first?.embedding ?? []
    }
}

extension OpenAIService: ModelService {

    public func models() async throws -> [Model] {
        let result = try await client.models()
        return result.data.map { decode($0) }
    }
}

extension OpenAIService: ImageService {

    public func imagine(_ request: ImagineServiceRequest) async throws -> [Data] {
        let req = OpenAI.ImageRequest(
            prompt: request.prompt,
            model: request.model.id,
            n: request.n,
            size: .size_1024x1024
        )
        let result = try await client.imagesGenerations(req)

        // HACK: Wait for a second for the images to be available on OpenAI's CDN. Without this the URLs in the
        // result may fail.
        let seconds = 1
        try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))

        return try await withThrowingTaskGroup(of: Data?.self) { group in
            for image in result.data {
                group.addTask {
                    guard let imageURL = image.url, let url = URL(string: imageURL) else { return nil }
                    let (data, _) = try await self.session.data(from: url)
                    return data
                }
            }
            var downloadedImages: [Data] = []
            for try await data in group {
                if let data {
                    downloadedImages.append(data)
                }
            }
            return downloadedImages
        }
    }
}

extension OpenAIService: TranscriptionService {

    public func transcribe(_ request: TranscriptionServiceRequest) async throws -> String {
        let req = OpenAI.TranscriptionRequest(
            file: request.file,
            model: request.model.id,
            language: request.language,
            prompt: request.prompt,
            response_format: (request.responseFormat != nil) ? .init(rawValue: request.responseFormat!) : nil,
            temperature: request.temperature
        )
        let result = try await client.transcriptions(req)
        return result.text
    }
}

extension OpenAIService: SpeechService {

    public func voices() async throws -> [Voice] {
        return ["alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"].map {
            Voice(id: $0, name: nil)
        }
    }

    public func voiceClone(_ request: SpeechVoiceCloneRequest) async throws -> String {
        throw ServiceError.notImplemented
    }

    public func speak(_ request: SpeechServiceRequest) async throws -> Data {
        let req = OpenAI.SpeechRequest(
            model: request.model.id,
            input: request.input,
            voice: .init(rawValue: request.voice) ?? .alloy,
            response_format: (request.responseFormat != nil) ? .init(rawValue: request.responseFormat!) : nil,
            speed: request.speed
        )
        return try await client.speech(req)
    }
}

// MARK: - Encoders

extension OpenAIService {

    private func encode(messages: [Message]) -> [OpenAI.ChatRequest.Message] {
        messages.map { message in
            OpenAI.ChatRequest.Message(
                content: encode(message.contents, role: message.role),
                role: encode(message.role),
                name: message.name,
                tool_calls: encode(message.toolCalls),
                tool_call_id: message.toolCallID
            )
        }
    }

    private func encode(_ contents: [Message.Content]?, role: Message.Role) -> [OpenAI.ChatRequest.Message.Content]? {
        contents?.map {
            switch $0 {
            case .text(let text):
                return .init(type: "text", text: text)
            case .image(let image):
                guard role == .user else {
                    return nil
                }
                guard let data = try? Data(contentsOf: image.url) else {
                    return nil
                }
                return .init(type: "image_url", image_url: .init(url: "data:\(image.format.rawValue);base64,\(data.base64EncodedString())"))
            case .audio(let audio):
                guard role == .user else {
                    return nil
                }
                guard let data = try? Data(contentsOf: audio.url) else {
                    return nil
                }
                return .init(type: "input_audio", input_audio: .init(data: data.base64EncodedString(), format: audio.format.rawValue))
            case .json(let json):
                return .init(type: "text", text: json.object)
            case .file(let file):
                guard
                    let data = try? Data(contentsOf: .documentsDirectory.appending(path: file.path)),
                    let content = String(data: data, encoding: .utf8) else { return nil }
                return .init(type: "text", text: """
                    ```\(file.mimetype.preferredFilenameExtension ?? "txt") \(file.path)
                    \(content)
                    ```
                    """)
            }
        }.compactMap({$0})
    }

    private func encode(_ role: Message.Role) -> OpenAI.ChatRequest.Message.Role {
        switch role {
        case .system: .system
        case .assistant: .assistant
        case .user: .user
        case .tool: .tool
        }
    }

    private func encode(_ toolCalls: [ToolCall]?) -> [OpenAI.ChatRequest.Message.ToolCall]? {
        toolCalls?.map { toolCall in
            OpenAI.ChatRequest.Message.ToolCall(
                id: toolCall.id,
                type: toolCall.type,
                function: encode(toolCall.function),
                custom: encode(toolCall.custom)
            )
        }
    }

    private func encode(_ functionCall: ToolCall.Function?) -> OpenAI.ChatRequest.Message.ToolCall.Function? {
        guard let functionCall else { return nil }
        return .init(name: functionCall.name, arguments: functionCall.arguments)
    }

    private func encode(_ customCall: ToolCall.Custom?) -> OpenAI.ChatRequest.Message.ToolCall.Custom? {
        guard let customCall else { return nil }
        return .init(name: customCall.name, input: customCall.input)
    }

    private func encode(_ tools: [Tool]) -> [OpenAI.ChatRequest.Tool]? {
        guard !tools.isEmpty else { return nil }
        return tools.map { tool in
            OpenAI.ChatRequest.Tool(
                type: tool.type.rawValue,
                function: encode(tool.function),
                custom: encode(tool.custom)
            )
        }
    }

    private func encode(_ function: Tool.Function?) -> OpenAI.ChatRequest.Tool.Function? {
        guard let function else { return nil }
        return .init(
            name: function.name,
            description: function.description,
            parameters: function.parameters
        )
    }

    private func encode(_ custom: Tool.Custom?) -> OpenAI.ChatRequest.Tool.Custom? {
        guard let custom else { return nil }
        return .init(
            name: custom.name,
            description: custom.description,
            format: (custom.format != nil) ? .init(
                type: custom.format!.type,
                grammar: (custom.format!.grammar != nil) ? .init(
                    definition: custom.format!.grammar!.definition,
                    syntax: custom.format!.grammar!.syntax
                ) : nil
            ) : nil
        )
    }

    private func encode(_ toolChoice: Tool?) -> OpenAI.ChatRequest.ToolChoice? {
        if let function = toolChoice?.function {
            return .tool(.init(function: function.name))
        }
        if let custom = toolChoice?.custom {
            return .tool(.init(custom: custom.name))
        }
        return nil
    }

    private func encode(responseFormat: String?) -> OpenAI.TranslationRequest.ResponseFormat? {
        guard let responseFormat else { return nil }
        return .init(rawValue: responseFormat)
    }

    // Speech

    private func encode(responseFormat: String?) throws -> OpenAI.SpeechRequest.ResponseFormat? {
        guard let responseFormat else { return nil }
        switch responseFormat {
        case "mp3":
            return .mp3
        case "opus":
            return .opus
        case "aac":
            return .aac
        case "flac":
            return .flac
        default:
            throw ServiceError.unsupportedResponseFormat
        }
    }
}

// MARK: - Decoders

// MARK: - Decoders

extension OpenAIService {

    private func decode(_ resp: OpenAI.ChatResponse) -> Message? {
        guard let choice = resp.choices.first else { return nil }
        let reasoning = choice.message.reasoning_content ?? ""
        let text = reasoning.isEmpty
            ? choice.message.content
            : compose(thinking: reasoning, answer: choice.message.content ?? "")
        return Message(
            role: decode(choice.message.role),
            contents: [decode(text)].compactMap { $0 },
            toolCalls: choice.message.tool_calls?.map { decode($0) },
            finishReason: decode(choice.finish_reason)
        )
    }
    
    private func decode(_ role: String) -> Message.Role {
        switch role {
        case "system":
            return .system
        case "user":
            return .user
        case "assistant":
            return .assistant
        case "tool":
            return .tool
        default:
            return .assistant
        }
    }
    
    private func decode(_ content: String?) -> Message.Content? {
        guard let content else { return nil }
        return .text(content)
    }
    
    nonisolated private func decode(_ reason: String?) -> Message.FinishReason? {
        switch reason {
        case "stop":
            return .stop
        case "length":
            return .length
        case "tool_calls":
            return .toolCalls
        case "content_filter":
            return .contentFilter
        default:
            return nil
        }
    }
    
    nonisolated private func decode(_ resp: OpenAI.ChatResponse.Choice.Message.ToolCall) -> ToolCall {
        ToolCall(
            index: resp.index,
            id: resp.id ?? "",
            type: resp.type ?? "function",
            function: (resp.function != nil) ? .init(
                name: resp.function!.name ?? "",
                arguments: resp.function!.arguments
            ) : nil,
            custom: (resp.custom != nil) ? .init(
                name: resp.custom!.name ?? "",
                input: resp.custom!.input
            ) : nil
        )
    }
    
    private func decode(_ model: OpenAI.Model) -> Model {
        Model(
            id: .init(model.id),
            name: model.id,
            owner: model.owned_by ?? "openai"
        )
    }
    
    /// Folds reasoning into the content as a `<think>` block.
    ///
    /// The shape Ollama's reasoning already takes, so a reader needn't know
    /// which provider a reply came from; see `OllamaService.compose`. Left
    /// open until the answer starts, so a reader streaming the reply can tell
    /// reasoning from answer as it arrives.
    nonisolated private func compose(thinking: String, answer: String) -> String {
        guard !thinking.isEmpty else { return answer }
        let closing = answer.isEmpty ? "" : "\n</think>\n"
        return "<think>\n" + thinking + closing + answer
    }

    nonisolated func patchMessage(_ message: inout Message, with resp: OpenAI.ChatStreamResponse) {
        guard let choice = resp.choices.first else { return }

        // Where the endpoint returns reasoning in the open, it and the answer
        // stream as separate deltas — DeepSeek sends `reasoning_content` a few
        // characters at a time with `content` null, then the reverse. So each
        // is accumulated and the combined content rebuilt from both every
        // time, as the Ollama provider does.
        if choice.delta.reasoning_content != nil || message.metadata["thinking"] != nil {
            let thinking = (message.metadata["thinking"]?.stringValue ?? "") + (choice.delta.reasoning_content ?? "")
            let answer = (message.metadata["answer"]?.stringValue ?? "") + (choice.delta.content ?? "")

            message.metadata["thinking"] = .string(thinking)
            message.metadata["answer"] = .string(answer)
            message.contents = [.text(compose(thinking: thinking, answer: answer))]
        } else if case .text(let text) = message.contents?.last {
            if let patched = GenKit.patch(string: text, with: choice.delta.content) {
                message.contents = [.text(patched)]
            }
        } else if let text = choice.delta.content {
            message.contents = [.text(text)]
        }
        message.finishReason = decode(choice.finish_reason)
        message.modified = .now

        // Add new tool calls and patch the last tool call being streamed in
        if let toolCalls = choice.delta.tool_calls {
            if message.toolCalls == nil {
                message.toolCalls = []
            }
            for toolCall in toolCalls {
                if var existing = message.toolCalls?.first(where: { $0.index == toolCall.index }) {
                    if let function = existing.function {
                        existing.function!.arguments = GenKit.patch(string: function.arguments, with: toolCall.function?.arguments) ?? ""
                    }
                    if let custom = existing.custom {
                        existing.custom!.input = GenKit.patch(string: custom.input, with: toolCall.custom?.input) ?? ""
                    }
                    message.toolCalls![existing.index!] = existing
                } else {
                    message.toolCalls?.append(decode(toolCall))
                }
            }
        }
    }
}
