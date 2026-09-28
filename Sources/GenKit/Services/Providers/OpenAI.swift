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

    func makeRequest(_ request: ChatServiceRequest, streaming: Bool) -> OpenAI.ChatRequest {
        // Where the service listed what the model can do, send only that.
        // Unknown is not incapable: a model that reported nothing gets the
        // request exactly as asked for, since every endpoint but Groq reports
        // nothing and narrowing on a silence would strip tools from models
        // that handle them perfectly well. Same rule as the Ollama provider.
        func supports(_ capability: Model.Capability) -> Bool {
            request.model.capabilities?.contains(capability) ?? true
        }

        var tools = encode(request.tools)
        if let offered = tools, !offered.isEmpty, !supports(.tools) {
            logger.warning(
                "\(request.model.id, privacy: .public) does not support tools; sending the request without \(offered.count, privacy: .public) of them"
            )
            tools = nil
        }

        var req = OpenAI.ChatRequest(
            messages: encode(messages: request.messages),
            model: request.model.id,
            stream: streaming ? true : nil,
            temperature: request.options["temperature"]?.doubleValue,
            tools: tools,
            tool_choice: tools == nil ? nil : encode(request.toolChoice),
            verbosity: request.options["verbosity"]?.stringValue.flatMap(OpenAI.ChatRequest.Verbosity.init)
        )
        // A stream reports what it cost only if asked. Groq sends usage
        // regardless, OpenAI and DeepSeek send none without this, and every
        // dialect here documents the option.
        if streaming {
            req.stream_options = .init(include_usage: true)
        }
        // A model the service says doesn't reason has nothing to grade, and
        // the endpoint refuses the parameter rather than ignoring it.
        if supports(.thinking) {
            switch reasoning(from: request.options, model: request.model) {
            case .nothing:
                break
            case .effort(let level):
                req.reasoning_effort = level
            case .thinkingOff:
                req.thinking = .disabled
            }
        }

        // The ceiling goes by two names. OpenAI deprecated `max_tokens` for
        // `max_completion_tokens` and its reasoning models reject the old one;
        // DeepSeek documents `max_tokens` and nothing else.
        if var ceiling = request.options["max_tokens"]?.intValue {
            // Not above what the model allows. The setting is one figure for
            // a whole service, and a service's models don't agree on what
            // they'll write — Groq allows gpt-oss-120b 65,536 tokens and
            // qwen3.8-27b 16,384 — so a figure chosen with one in mind is a
            // refused request on the other. Asking for less than was set is
            // the lesser wrong: the alternative is no answer at all.
            if let allowed = request.model.maxOutput, ceiling > allowed {
                logger.info(
                    "\(request.model.id, privacy: .public) allows \(allowed, privacy: .public) tokens a reply; asking for that rather than \(ceiling, privacy: .public)"
                )
                ceiling = allowed
            }
            switch dialect {
            case .deepSeek:
                req.max_tokens = ceiling
            case .openAI, .grok, .groq, .openRouter:
                req.max_completion_tokens = ceiling
            }
        }
        return req
    }

    /// What to send about reasoning, if anything.
    enum ReasoningSetting: Equatable {
        case nothing
        case effort(OpenAI.ChatRequest.ReasoningEffort)
        /// DeepSeek's switch: it has no level that means none.
        case thinkingOff
    }

    /// What to ask for, in this endpoint's terms.
    ///
    /// The caller's `effort` is provider-neutral — see the levels below — and
    /// what a model accepts is the model's own business: the API refuses what
    /// it doesn't take rather than ignoring it, and a model that doesn't
    /// reason refuses the parameter altogether. So this asks the table first,
    /// and asks for the nearest thing where the exact level isn't offered.
    nonisolated func reasoning(from options: [String: Value], model: Model) -> ReasoningSetting {
        // A caller speaking the API's own terms is taken at its word.
        if let raw = options["reasoning_effort"]?.stringValue {
            return OpenAI.ChatRequest.ReasoningEffort(rawValue: raw).map { .effort($0) } ?? .nothing
        }
        guard let raw = options["effort"]?.stringValue else { return .nothing }

        let supported = Reasoning.supported(by: model, dialect: dialect)
        guard let levels = supported.levels else {
            // Either the model takes no such setting — in which case Off is
            // already true of it — or nothing is known, and a guess here is a
            // 400 if wrong where leaving it alone is merely the default.
            if raw != "off" {
                logger.info("\(model.id, privacy: .public) takes no known reasoning setting; leaving it to the model")
            }
            return .nothing
        }

        var wanted: String
        switch raw {
        case "off":
            // Where there's a switch, throw it: a level that means none is
            // the exception, not the rule.
            if supported.offSwitch { return .thinkingOff }
            wanted = "none"
        case "brief":
            // The steered pair local models use. Here the request grades it.
            wanted = "low"
        case "full":
            // Whatever the model does unprompted, which is what sending
            // nothing asks for.
            return .nothing
        case "low", "medium", "high", "xhigh", "max":
            wanted = raw
        default:
            logger.warning("ignoring unrecognised effort \(raw, privacy: .public)")
            return .nothing
        }

        // An endpoint that maps some levels onto others is followed rather
        // than second-guessed — DeepSeek documents medium as high.
        if let alias = supported.aliases[wanted] {
            wanted = alias
        }

        if levels.contains(wanted), let level = OpenAI.ChatRequest.ReasoningEffort(rawValue: wanted) {
            return .effort(level)
        }

        // Not offered, so the nearest that is: `xhigh` on a model that stops
        // at `high` gets `high`; `none` on a model that can't stop gets the
        // least it will do. Ties go to the lower level.
        //
        // Never the other way round. Asking for some reasoning and getting
        // none is not "nearest", it's the opposite of what was asked — and
        // a model whose only level is `none` (Groq's qwen3 family) reasons
        // by default, so leaving it alone is what a graded request means.
        let order = Reasoning.order
        guard let target = order.firstIndex(of: wanted) else { return .nothing }
        let candidates = wanted == "none" ? levels : levels.filter { $0 != "none" }
        let nearest = candidates
            .compactMap { level in order.firstIndex(of: level).map { (level, abs($0 - target)) } }
            .min { $0.1 < $1.1 }?.0
        guard let nearest, let level = OpenAI.ChatRequest.ReasoningEffort(rawValue: nearest) else {
            logger.info("\(model.id, privacy: .public) has no level for \(wanted, privacy: .public); leaving it to the model")
            return .nothing
        }

        if wanted == "none" {
            logger.info(
                "\(model.id, privacy: .public) can't be told not to reason; asking for the least, \(nearest, privacy: .public)"
            )
        } else {
            logger.info(
                "\(model.id, privacy: .public) doesn't take \(wanted, privacy: .public); asking for \(nearest, privacy: .public)"
            )
        }
        return .effort(level)
    }

    /// Whether a model can be told not to reason, where that's known.
    ///
    /// True where it takes `none`, or `minimal` — which is as near to none as
    /// gpt-5 gets, and nearer than any level a menu offers. False where the
    /// least it takes is `low`. Nil where it takes no such setting at all,
    /// since a model that never reasons has nothing to switch off, or where
    /// nothing is known — only a fact is worth acting on.
    public nonisolated static func canDisableThinking(modelID: String, dialect: Dialect) -> Bool? {
        let supported = Reasoning.supported(byModel: modelID, dialect: dialect)
        guard let levels = supported.levels else { return nil }
        return supported.offSwitch || levels.contains("none") || levels.contains("minimal")
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
    ///
    /// DeepSeek is from its Thinking Mode guide rather than a refusal. Its
    /// endpoint accepts all seven names but has three levels — low, high and
    /// max — and documents the rest as aliases: minimal is low, medium and
    /// xhigh are high. It is switched off with its own `thinking` parameter,
    /// not a level; thinking is on by default at high. The o-series row is
    /// from the documentation too, and `o1-mini` and `o1-preview` are
    /// documented as taking none.
    ///
    /// Groq is from its API reference, which is per model family:
    ///
    ///     openai/gpt-oss-*   low, medium, high (medium by default; no none)
    ///     qwen/qwen3.8-*     none, low, medium, high (none by default;
    ///                        high is the model's own xhigh)
    ///     other qwen3        none, or the model's default
    ///
    /// and "values outside a model's supported set are rejected with a 400".
    /// Anything else there — Llama, MiniMax — is unmeasured. Grok and
    /// OpenRouter are unmeasured throughout and get nothing, which is the
    /// safe failure: a level not sent is the model's default, where a level
    /// refused is no answer at all.
    struct Reasoning {
        /// Ascending. Nil where the model takes no such setting.
        let levels: [String]?

        /// Levels the endpoint accepts and reads as another, as documented.
        var aliases: [String: String] = [:]

        /// Whether thinking is turned off with a switch of its own rather
        /// than a level.
        var offSwitch = false

        /// Every level any endpoint has named, least to most.
        static let order = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]

        /// What this model takes, preferring what the service said over what
        /// this file remembers.
        ///
        /// DeepSeek reports its levels on every model — low, high and max,
        /// with high by default — which is what the row below records after
        /// a reading of its Thinking Mode guide. A reported answer doesn't
        /// go stale, so it wins; the table answers for the endpoints that
        /// report nothing, and for models stored before this was read.
        ///
        /// The aliases and the off switch stay with the table either way:
        /// the list says which levels a model takes, not that `medium` is
        /// another word for `high`, nor that thinking is turned off with a
        /// parameter of its own rather than a level.
        static func supported(by model: Model, dialect: Dialect) -> Reasoning {
            let remembered = supported(byModel: model.id, dialect: dialect)
            guard let reported = model.reasoning?.levels, !reported.isEmpty else {
                return remembered
            }
            return Reasoning(
                levels: reported,
                aliases: remembered.aliases,
                offSwitch: remembered.offSwitch
            )
        }

        static func supported(byModel id: String, dialect: Dialect) -> Reasoning {
            switch dialect {
            case .deepSeek:
                return Reasoning(
                    levels: ["low", "high", "max"],
                    aliases: ["minimal": "low", "medium": "high", "xhigh": "high"],
                    offSwitch: true
                )
            case .groq:
                // Ids carry the lab as a prefix: openai/gpt-oss-120b,
                // qwen/qwen3.8-27b. The family is what's after it.
                let family = id.split(separator: "/").last.map(String.init) ?? id
                if family.hasPrefix("gpt-oss") {
                    return Reasoning(levels: ["low", "medium", "high"])
                }
                if family.hasPrefix("qwen3.8") {
                    return Reasoning(levels: ["none", "low", "medium", "high"])
                }
                if family.hasPrefix("qwen3") {
                    // A switch, not a scale: none, or leave it to the model.
                    return Reasoning(levels: ["none"])
                }
                return Reasoning(levels: nil)
            case .grok, .openRouter:
                return Reasoning(levels: nil)
            case .openAI:
                break
            }

            if id.hasPrefix("o1-mini") || id.hasPrefix("o1-preview") {
                return Reasoning(levels: nil)
            }

            // The search models — gpt-4o-search-preview, gpt-5-search-api —
            // are chat models with search built in and no reasoning setting:
            // gpt-5-search-api refuses the parameter outright, "Unrecognized
            // request argument", whatever its generation number says.
            if id.split(separator: "-").contains("search") {
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
            | thinking: \(req.thinking?.type ?? "unset", privacy: .public) \
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
            // Counted under the name each arrived by, so the line says which
            // endpoint's convention was seen — not just that reasoning was.
            if choice.delta.reasoning_content != nil { seen["reasoning_content", default: 0] += 1 }
            if choice.delta.reasoning != nil { seen["reasoning", default: 0] += 1 }
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
            if let cached = usage.prompt_cache_hit_tokens ?? usage.prompt_tokens_details?.cached_tokens, cached > 0 {
                parts.append("cached: \(cached)")
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

    /// The encoder, reachable by a test. Not otherwise part of the API.
    nonisolated func encodeForTest(messages: [Message]) -> [OpenAI.ChatRequest.Message] {
        encode(messages: messages)
    }

    nonisolated private func encode(messages: [Message]) -> [OpenAI.ChatRequest.Message] {
        messages.map { message in
            var encoded = OpenAI.ChatRequest.Message(
                content: encode(message.contents, role: message.role),
                role: encode(message.role),
                name: message.name,
                tool_calls: encode(message.toolCalls),
                tool_call_id: message.toolCallID
            )

            // A reply that reasoned in the open keeps its reasoning and its
            // answer apart in metadata (see `patchMessage`), and DeepSeek
            // wants them back that way: the reasoning as `reasoning_content`
            // — required on every assistant message whenever tools are sent,
            // a 400 without it — and the answer alone as content, rather than
            // the folded `<think>` block it never wrote. Sent to no one else:
            // OpenAI refuses the field on input.
            if dialect == .deepSeek, message.role == .assistant,
               let thinking = message.metadata["thinking"]?.stringValue {
                encoded.reasoning_content = thinking
                if let answer = message.metadata["answer"]?.stringValue {
                    // Kept even when empty — a round that only called tools
                    // has no answer, and `content: ""` is what its own guide
                    // sends back for that.
                    encoded.content = [.init(type: "text", text: answer)]
                }
            }
            return encoded
        }
    }

    nonisolated private func encode(_ contents: [Message.Content]?, role: Message.Role) -> [OpenAI.ChatRequest.Message.Content]? {
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

    nonisolated private func encode(_ role: Message.Role) -> OpenAI.ChatRequest.Message.Role {
        switch role {
        case .system: .system
        case .assistant: .assistant
        case .user: .user
        case .tool: .tool
        }
    }

    nonisolated private func encode(_ toolCalls: [ToolCall]?) -> [OpenAI.ChatRequest.Message.ToolCall]? {
        toolCalls?.map { toolCall in
            OpenAI.ChatRequest.Message.ToolCall(
                id: toolCall.id,
                type: toolCall.type,
                function: encode(toolCall.function),
                custom: encode(toolCall.custom)
            )
        }
    }

    nonisolated private func encode(_ functionCall: ToolCall.Function?) -> OpenAI.ChatRequest.Message.ToolCall.Function? {
        guard let functionCall else { return nil }
        return .init(name: functionCall.name, arguments: functionCall.arguments)
    }

    nonisolated private func encode(_ customCall: ToolCall.Custom?) -> OpenAI.ChatRequest.Message.ToolCall.Custom? {
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
        // Two names for one thing, never both: DeepSeek's and Groq's.
        let reasoning = choice.message.reasoning_content ?? choice.message.reasoning ?? ""
        let text = reasoning.isEmpty
            ? choice.message.content
            : compose(thinking: reasoning, answer: choice.message.content ?? "")
        var message = Message(
            role: decode(choice.message.role),
            contents: [decode(text)].compactMap { $0 },
            toolCalls: choice.message.tool_calls?.map { decode($0) },
            finishReason: decode(choice.finish_reason)
        )
        record(usage: resp.usage, model: resp.model, into: &message)
        return message
    }

    /// What the reply cost, carried on the message under the same
    /// provider-neutral names the Ollama provider uses, so a caller can show
    /// it without knowing which service answered.
    ///
    /// Nothing recorded this before, so every hosted service's reply arrived
    /// with no counts at all — and an app showing them showed nothing, on
    /// every provider but Ollama.
    ///
    /// Each field is written only when it arrives. A stream reports usage on
    /// one chunk out of thousands, and this runs on all of them.
    /// Whether `reasoning_tokens` is additional to `completion_tokens` rather
    /// than counted inside it.
    ///
    /// OpenAI, Groq and DeepSeek count reasoning within the completion, and
    /// this assumed everyone did. Grok does not. Measured 28 Sept 2026: a
    /// reply reporting 306 completion tokens against 1,396 reasoning ones,
    /// which cannot be a share of them — the app subtracted one from the
    /// other and displayed an answer of −1,090 tokens.
    ///
    /// Decided from the service's own total rather than from a table of who
    /// does what. `total_tokens` either accounts for the reasoning on top of
    /// the completion or it doesn't, and that is this question put to the
    /// only party that knows — the same reason the model capabilities are
    /// now read from each provider's list instead of being written down here
    /// and going quietly out of date.
    ///
    /// Where there is no total, or it agrees with neither sum, the sizes
    /// decide: reasoning larger than the completion cannot be inside it.
    /// That leaves one case undetectable — a service reporting them apart
    /// where the reasoning happens to be the smaller — which is why the
    /// total is asked first and the comparison is only the fallback.
    nonisolated private func reasoningIsExtra(_ usage: OpenAI.Usage) -> Bool {
        guard let reasoning = usage.completion_tokens_details?.reasoning_tokens, reasoning > 0,
              let completion = usage.completion_tokens
        else {
            return false
        }

        if let total = usage.total_tokens, let prompt = usage.prompt_tokens {
            if total == prompt + completion + reasoning { return true }
            if total == prompt + completion { return false }
        }
        return reasoning > completion
    }

    nonisolated private func record(usage: OpenAI.Usage?, model: String?, into message: inout Message) {
        // Which model actually answered. Reported on every chunk, and worth
        // keeping per message: a conversation can change model between
        // messages, so tokens and a rate mean nothing without it.
        if let model, !model.isEmpty {
            message.metadata["model"] = .string(model)
        }
        guard let usage else { return }

        if let input = usage.prompt_tokens {
            message.metadata["inputTokens"] = .int(input)
        }
        if let output = usage.completion_tokens {
            // Everything generated, reasoning included. Usually that is
            // `completion_tokens` unchanged, reasoning being a share of it.
            // Where a service reports the two apart they have to be added, or
            // the reply carries more reasoning than output and a caller
            // splitting one from the other gets a negative answer.
            let reasoning = usage.completion_tokens_details?.reasoning_tokens ?? 0
            message.metadata["outputTokens"] =
                .int(reasoningIsExtra(usage) ? output + reasoning : output)
        }
        // Generation alone, where timing the request from outside would also
        // count the wait in the queue. Groq reports it; nothing else does, so
        // a rate is shown where it can be worked out and left off elsewhere.
        if let seconds = usage.completion_time, seconds > 0 {
            message.metadata["outputSeconds"] = .double(seconds)
        }
        // Reasoning as a share of the output, `outputTokens` above having
        // been made to include it either way.
        //
        // Under its own name, not `thinkingTokens`: that one is a caller's
        // estimate for services that report no such figure, apportioned from
        // where the reasoning ended in the stream. A counted number and a
        // guess shouldn't share a key, or whichever is written second wins.
        if let reasoning = usage.completion_tokens_details?.reasoning_tokens, reasoning > 0 {
            message.metadata["reasoningTokens"] = .int(reasoning)
        }
        // How much of the prompt was already cached. DeepSeek caches without
        // being asked and reports it under a name of its own; OpenAI reports
        // the same figure in the details. Nothing here asks for caching —
        // there is nothing to ask — so this is purely what happened.
        let cached = usage.prompt_cache_hit_tokens ?? usage.prompt_tokens_details?.cached_tokens
        if let cached, cached > 0 {
            message.metadata["cachedTokens"] = .int(cached)
        }
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
    
    func decodeForTesting(_ model: OpenAI.Model) -> Model { decode(model) }

    private func decode(_ model: OpenAI.Model) -> Model {
        Model(
            id: .init(model.id),
            // The id where there's no name, since something has to be shown
            // and the id is what every caller showed before names existed.
            name: model.name ?? model.id,
            owner: model.owned_by ?? "openai",
            contextWindow: model.context_window,
            maxOutput: model.maxOutput,
            capabilities: Self.capabilities(of: model),
            reasoning: Self.reasoning(of: model),
            retiresOn: Self.retirement(of: model)
        )
    }

    /// How a listed model wants to be asked to reason, where it says.
    ///
    /// DeepSeek reports the levels outright, which is what this provider
    /// had hardcoded from its Thinking Mode guide. Groq reports only that a
    /// model reasons, so its levels stay in the table.
    nonisolated static func reasoning(of model: OpenAI.Model) -> Model.Reasoning? {
        guard let effort = model.effort,
              effort.supported_levels?.isEmpty == false || effort.default_level != nil
        else {
            return nil
        }
        return .init(levels: effort.supported_levels, defaultLevel: effort.default_level)
    }

    /// The day the service stops serving it. A day, with no time and no
    /// zone, so it's read as one.
    nonisolated static func retirement(of model: OpenAI.Model) -> Date? {
        guard let raw = model.shutdown_date, !raw.isEmpty else { return nil }
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC") ?? .gmt
        let parts = raw.split(separator: "-").compactMap { Int($0) }
        guard parts.count == 3 else { return nil }
        return calendar.date(from: DateComponents(year: parts[0], month: parts[1], day: parts[2]))
    }

    /// What a listed model can do, where the list says.
    ///
    /// Nil where it doesn't, which is every endpoint but Groq — and nil
    /// means unknown rather than incapable, so nothing narrows on a silence.
    ///
    /// The modality vocabulary is wider than text or not. Whisper reports
    /// `["audio"]` in and `["transcription"]` out; a speech model text in
    /// and `["audio"]` out; a vision model `["text", "image"]` in. Both ends
    /// have to be text for a model to hold a conversation, and a model
    /// failing that is not a lesser chat model but a model for another
    /// endpoint entirely.
    nonisolated static func capabilities(of model: OpenAI.Model) -> Set<Model.Capability>? {
        guard model.input_modalities != nil
            || model.output_modalities != nil
            || model.supported_features != nil
            || model.effort != nil
        else {
            return nil
        }
        let inputs = Set(model.input_modalities ?? [])
        let outputs = Set(model.output_modalities ?? [])
        let features = Set(model.supported_features ?? [])

        var capabilities: Set<Model.Capability> = []
        if inputs.contains("text"), outputs.contains("text") {
            capabilities.insert(.completion)
        }
        if inputs.contains("image") {
            capabilities.insert(.vision)
        }
        if features.contains("tools") {
            capabilities.insert(.tools)
        }
        // Named in one place or the other: Groq puts it among the features,
        // DeepSeek reports the levels instead and mentions it nowhere else.
        // Missing it means the request is stripped of a setting the model
        // does take, which is how a reasoning model ends up not reasoning.
        if features.contains("reasoning") || model.effort != nil {
            capabilities.insert(.thinking)
        }
        return capabilities
    }
    
    /// Folds reasoning into the content as a `<think>` block.
    ///
    /// The shape Ollama's reasoning already takes, so a reader needn't know
    /// which provider a reply came from; see `OllamaService.compose`. Left
    /// open until the answer starts, so a reader streaming the reply can tell
    /// reasoning from answer as it arrives.
    nonisolated private func compose(thinking: String, answer: String) -> String {
        guard !thinking.isEmpty else { return answer }
        let answer = strippingLeadingCloseTag(answer)
        let closing = answer.isEmpty ? "" : "\n</think>\n"
        return "<think>\n" + thinking + closing + answer
    }

    nonisolated func patchMessage(_ message: inout Message, with resp: OpenAI.ChatStreamResponse) {
        // Before the guard below, because the chunk carrying usage may carry
        // nothing else: OpenAI sends it as a final chunk with `choices` empty,
        // so reading it after that guard read it never.
        record(usage: resp.usage, model: resp.model, into: &message)

        guard let choice = resp.choices.first else { return }

        // Where the endpoint returns reasoning in the open, it and the answer
        // stream as separate deltas — DeepSeek sends `reasoning_content` a few
        // characters at a time with `content` null, then the reverse. So each
        // is accumulated and the combined content rebuilt from both every
        // time, as the Ollama provider does.
        let reasoningDelta = choice.delta.reasoning_content ?? choice.delta.reasoning
        if reasoningDelta != nil || message.metadata["thinking"] != nil {
            let thinking = (message.metadata["thinking"]?.stringValue ?? "") + (reasoningDelta ?? "")
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
