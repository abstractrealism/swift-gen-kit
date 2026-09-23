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

        applyReasoning(to: &req, options: options, model: model)
        return req
    }

    /// Reasoning, and how much work to put in overall. Two settings rather
    /// than one: `thinking` decides whether the model reasons at all, and
    /// `effort` how deeply — including how deeply it reasons when it does.
    ///
    /// What each model accepts differs, and it refuses what it doesn't rather
    /// than ignoring it, so this asks first. See `Thinking`.
    private func applyReasoning(to req: inout Anthropic.ChatRequest, options: [String: Value], model: Model) {
        let supported = Thinking.supported(by: model)

        switch effort(from: options) {
        case .none:
            break

        case .off:
            if supported.canDisable {
                req.thinking = .disabled
            } else {
                // Some models reason on every request and reject being told
                // not to. Asking for the least is as close as this gets, and
                // closer than failing the send.
                logger.info(
                    "\(model.id, privacy: .public) always reasons; asking for the least rather than none"
                )
                apply(effort: "low", to: &req, supported: supported, model: model)
            }

        case .level(let level):
            apply(effort: level, to: &req, supported: supported, model: model)
        }
    }

    /// Asks for reasoning at a level, in whichever of the two shapes this
    /// model takes.
    private func apply(
        effort level: String,
        to req: inout Anthropic.ChatRequest,
        supported: Thinking,
        model: Model
    ) {
        let level = nearestLevel(to: level, of: model)
        if supported.adaptive {
            // `summarized` because the default is `omitted`, under which the
            // blocks still arrive carrying a signature and no text. A caller
            // that means to show reasoning has to say so.
            req.thinking = .adaptive(display: "summarized")
        } else if supported.extended {
            // The older control: a token budget rather than a level. Its
            // blocks come back with their text, so there's no display to ask
            // for. Half the reply's ceiling at most, since that ceiling covers
            // the reasoning and the answer together and an answer needs room.
            let budget = min(Thinking.budget(for: level), max(1024, req.max_tokens / 2))
            req.thinking = .init(type: "enabled", budget_tokens: budget)
        } else {
            logger.info(
                "\(model.id, privacy: .public) takes no thinking setting; leaving it to the model"
            )
            return
        }

        // Not every model that reasons takes a level for it — the
        // extended-thinking generation mostly doesn't.
        if supported.effort {
            req.output_config = .init(effort: level)
        }
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

    func makeRequestForTesting(model: Model, options: [String: Value]) -> Anthropic.ChatRequest {
        var req = Anthropic.ChatRequest(model: model.id, messages: [], max_tokens: 16384)
        applyReasoning(to: &req, options: options, model: model)
        return req
    }

    /// The level to ask for, given what this model says it takes.
    ///
    /// Only where it says. Models differ by more than which mode they
    /// take — Sonnet 4.6 takes low, medium, high and max but not xhigh,
    /// while Opus 5.5 takes all five — and a level a model doesn't have is
    /// refused rather than rounded, so asking for the nearest it does have
    /// is the difference between a slightly shallower answer and none.
    ///
    /// Ties go to the lower level, the levels being in ascending order.
    private func nearestLevel(to level: String, of model: Model) -> String {
        guard let levels = model.reasoning?.levels, !levels.isEmpty, !levels.contains(level) else {
            return level
        }
        let order = ["low", "medium", "high", "xhigh", "max"]
        guard let target = order.firstIndex(of: level) else { return level }

        let nearest = levels
            .compactMap { candidate in order.firstIndex(of: candidate).map { (candidate, abs($0 - target)) } }
            .min { $0.1 < $1.1 }?.0
        guard let nearest else { return level }

        logger.info(
            "\(model.id, privacy: .public) doesn't take \(level, privacy: .public); asking for \(nearest, privacy: .public)"
        )
        return nearest
    }

    /// What a model will accept in the way of reasoning.
    ///
    /// The API rejects what a model doesn't take rather than ignoring it —
    /// `adaptive` to an extended-thinking model is a 400, and so is `enabled`
    /// to an adaptive one — so this has to be right per model.
    ///
    /// The model list answers it, which was not known when the table below
    /// was written: every entry reports its thinking types and effort levels.
    /// So the order is reported, then remembered, then assumed. The table is
    /// what answers for a service that reported nothing — a model loaded
    /// before any of this existed, say — and it no longer needs adding to as
    /// models arrive, since a listed model describes itself.
    ///
    /// Matched on the id containing a pattern, since ids carry a date
    /// (`claude-haiku-4-5-20251001`), and in order, because the more specific
    /// pattern has to win: `claude-fable-5` is a prefix of
    /// `claude-fable-5-1`.
    struct Thinking {
        /// Takes `thinking: {type: "adaptive"}` and decides for itself.
        var adaptive = false

        /// Takes `thinking: {type: "enabled", budget_tokens: n}`.
        var extended = false

        /// Takes `thinking: {type: "disabled"}`. False where a model reasons
        /// on every request and refuses being told not to.
        var canDisable = true

        /// Takes `output_config.effort`.
        var effort = true

        /// Everything an unrecognised model is assumed to take.
        ///
        /// Adaptive, because every model from Claude Sonnet 4.6 onwards is,
        /// and a model this doesn't know about is far likelier to be newer
        /// than older. Wrong for an old model, and it says so out loud when
        /// the request comes back refused.
        static let assumed = Thinking(adaptive: true)

        static func supported(by model: Model) -> Thinking {
            reported(by: model) ?? known(byModel: model.id) ?? .assumed
        }

        /// What the service itself said about this model, where it said
        /// anything.
        ///
        /// The list reports, per model, which `thinking.type` values it takes
        /// and which effort levels — the two things the table below exists to
        /// remember, and the two that change with every model that ships. A
        /// reported answer is always better than a remembered one, so it
        /// wins; the table stays for services and accounts that report
        /// nothing, and for the one thing the list doesn't say.
        ///
        /// `canDisable` is that thing. Nothing in the response distinguishes
        /// a model that refuses `thinking: {type: "disabled"}` from one that
        /// accepts it, so it comes from the table where the table knows and
        /// is assumed true otherwise — the same assumption as before, and
        /// wrong the same way, which is caught by the refusal.
        static func reported(by model: Model) -> Thinking? {
            guard let reasoning = model.reasoning else { return nil }
            let modes = Set(reasoning.modes ?? [])
            guard !modes.isEmpty || reasoning.levels != nil else { return nil }

            return Thinking(
                adaptive: modes.contains("adaptive"),
                extended: modes.contains("enabled"),
                canDisable: known(byModel: model.id)?.canDisable ?? true,
                effort: !(reasoning.levels ?? []).isEmpty
            )
        }

        /// What's actually recorded for a model, or nil where it isn't one
        /// this knows. Distinct from `supported(byModel:)` so a caller can
        /// tell a fact from an assumption.
        static func known(byModel id: String) -> Thinking? {
            let table: [(String, Thinking)] = [
                // Reason always, and reject being turned off.
                ("claude-fable-5-1", .init(adaptive: true, canDisable: false)),
                ("claude-mythos-5-1", .init(adaptive: true, canDisable: false)),
                ("claude-fable-5", .init(adaptive: true, canDisable: false)),
                ("claude-mythos-preview", .init(adaptive: true, extended: true, canDisable: false)),
                ("claude-mythos-5", .init(adaptive: true, canDisable: false)),

                // Adaptive only.
                ("claude-opus-5", .init(adaptive: true)),
                ("claude-opus-4-8", .init(adaptive: true)),
                ("claude-opus-4-7", .init(adaptive: true)),
                ("claude-sonnet-5", .init(adaptive: true)),

                // Both, while the older one is still accepted.
                ("claude-opus-4-6", .init(adaptive: true, extended: true)),
                ("claude-sonnet-4-6", .init(adaptive: true, extended: true)),

                // A budget and nothing else. Of these only Opus 4.5 takes a
                // level as well.
                ("claude-opus-4-5", .init(extended: true)),
                ("claude-haiku-4-5", .init(extended: true, effort: false)),
                ("claude-sonnet-4-5", .init(extended: true, effort: false)),
            ]

            for (pattern, supported) in table where id.contains(pattern) {
                return supported
            }
            return nil
        }

        /// A thinking budget for a level, for the models that want one rather
        /// than a name. Anthropic's floor is 1,024.
        static func budget(for level: String) -> Int {
            switch level {
            case "low": 2048
            case "medium": 4096
            case "high": 8192
            case "xhigh": 16384
            default: 32768
            }
        }
    }

    /// Whether this model will accept being told not to reason.
    ///
    /// Nil where the model isn't one the table knows, which a caller should
    /// read as "probably" rather than as a promise: the request degrades to
    /// the least reasoning if the guess turns out wrong, so there's nothing to
    /// warn anybody about in advance.
    ///
    /// `nonisolated` because it reads a table rather than the actor.
    public nonisolated static func canDisableThinking(modelID: String) -> Bool? {
        Thinking.known(byModel: modelID)?.canDisable
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
        log(req, streaming: false)
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
        log(req, streaming: true)

        var message = Message(role: .assistant)
        var seen: [String: Int] = [:]
        do {
            for try await result in try client.chatCompletionsStream(req) {
                seen[result.type.rawValue, default: 0] += 1
                patchMessage(&message, with: result)
                try await update(message)
            }
        } catch {
            logger.error(
                "← anthropic stream failed after \(self.describe(seen), privacy: .public): \(error, privacy: .public)"
            )
            throw error
        }
        logger.info("← anthropic stream finished | \(self.describe(seen), privacy: .public)")
    }

    /// What was asked for, in the terms the service was asked in.
    ///
    /// Counts and settings only, never the messages: this is for telling why a
    /// request was refused, and a conversation's contents don't answer that.
    private func log(_ req: Anthropic.ChatRequest, streaming: Bool) {
        let thinking = req.thinking.map { thinking in
            thinking.display.map { "\(thinking.type)/\($0)" } ?? thinking.type
        } ?? "unset"
        logger.info("""
            → anthropic \(streaming ? "stream" : "completion", privacy: .public) \
            | model: \(req.model, privacy: .public) \
            | thinking: \(thinking, privacy: .public) \
            | effort: \(req.output_config?.effort ?? "unset", privacy: .public) \
            | max_tokens: \(req.max_tokens, privacy: .public) \
            | temperature: \(req.temperature.map { "\($0)" } ?? "unset", privacy: .public) \
            | tools: \(req.tools?.count ?? 0, privacy: .public) \
            | system: \(req.system?.count ?? 0, privacy: .public) \
            | messages: \(req.messages.count, privacy: .public)
            """)
    }

    private func describe(_ events: [String: Int]) -> String {
        guard !events.isEmpty else { return "no events" }
        return events
            .sorted { $0.key < $1.key }
            .map { "\($0.key)×\($0.value)" }
            .joined(separator: " ")
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
        var message = Message(
            role: decode(resp.role) ?? .assistant,
            contents: resp.content?.compactMap { decode($0) } ?? [],
            toolCalls: resp.content?.compactMap { decodeToolCall($0) },
            finishReason: decode(resp.stop_reason)
        )
        record(usage: resp.usage, model: resp.model, into: &message)
        return message
    }

    /// What the reply cost, under the provider-neutral names the Ollama
    /// provider uses. Nothing recorded this before, so an app showing what a
    /// generation cost showed nothing on every service but Ollama.
    ///
    /// Each field is written only when it arrives: a stream reports the
    /// input count on its first event and the output count on a later one,
    /// and this runs on every event in between.
    nonisolated func record(usage: Anthropic.ChatResponse.Usage?, model: String?, into message: inout Message) {
        if let model, !model.isEmpty {
            message.metadata["model"] = .string(model)
        }
        guard let usage else { return }

        // Cached reads and writes are prompt tokens billed differently, not
        // extra ones, but `input_tokens` counts neither — so a cached turn
        // reported a fraction of what it actually sent.
        let input = [usage.input_tokens, usage.cache_read_input_tokens, usage.cache_creation_input_tokens]
            .compactMap { $0 }
        if !input.isEmpty {
            message.metadata["inputTokens"] = .int(input.reduce(0, +))
        }
        if let output = usage.output_tokens {
            message.metadata["outputTokens"] = .int(output)
        }
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

    nonisolated func decode(_ reason: Anthropic.ChatResponse.StopReason?) -> Message.FinishReason? {
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
            owner: "anthropic",
            contextWindow: model.max_input_tokens,
            maxOutput: model.max_tokens,
            capabilities: Self.capabilities(of: model),
            reasoning: Self.reasoning(of: model)
        )
    }

    /// What a listed model can do. Nil where the list didn't say, which is
    /// unknown rather than incapable.
    nonisolated static func capabilities(of model: Anthropic.Model) -> Set<Model.Capability>? {
        guard let reported = model.capabilities else { return nil }

        // Every model here converses, and every one takes tools; neither is
        // reported because neither has ever been in question.
        var capabilities: Set<Model.Capability> = [.completion, .tools]
        if reported.thinking?.supported == true {
            capabilities.insert(.thinking)
        }
        if reported.image_input?.supported == true {
            capabilities.insert(.vision)
        }
        return capabilities
    }

    /// How a listed model wants to be asked to reason.
    ///
    /// The levels are read one by one rather than from `effort.supported`,
    /// because a model that takes none still reports the capability — with
    /// nothing under it. So the levels present are the answer, and an empty
    /// set means a model that reasons on a budget instead.
    nonisolated static func reasoning(of model: Anthropic.Model) -> Model.Reasoning? {
        guard let reported = model.capabilities else { return nil }

        var levels: [String] = []
        if let effort = reported.effort, effort.supported != false {
            if effort.low?.supported == true { levels.append("low") }
            if effort.medium?.supported == true { levels.append("medium") }
            if effort.high?.supported == true { levels.append("high") }
            if effort.xhigh?.supported == true { levels.append("xhigh") }
            if effort.max?.supported == true { levels.append("max") }
        }

        var modes: [String] = []
        if reported.thinking?.types?.adaptive?.supported == true { modes.append("adaptive") }
        if reported.thinking?.types?.enabled?.supported == true { modes.append("enabled") }

        guard !levels.isEmpty || !modes.isEmpty else { return nil }
        return .init(levels: levels.isEmpty ? nil : levels, modes: modes.isEmpty ? nil : modes)
    }

    /// Closes an open `<think>` block, if the message is in one.
    ///
    /// Left alone otherwise, and idempotent: a reply that reasons and then
    /// stops without answering keeps an open tag, which readers already handle
    /// — an unterminated block runs to the end of the text, which is exactly
    /// what an interrupted reply is.
    nonisolated private func closeThinking(_ message: inout Message) {
        guard case .text(let existing) = message.contents?.last,
              existing.hasPrefix("<think>"),
              !existing.contains("</think>")
        else {
            return
        }
        message.contents![message.contents!.count-1] = .text(existing + "\n</think>")
    }

    nonisolated func patchMessage(_ message: inout Message, with resp: ChatResponseStream) {
        // Before the switch, because the events carrying usage are among
        // those it returns from: the opening one reports what the prompt
        // cost, and `message_delta` the running output count.
        record(usage: resp.message?.usage ?? resp.usage, model: resp.message?.model, into: &message)

        switch resp.type {
        case .ping, .content_block_stop:
            return
        case .message_delta:
            // Where a stream says why the model stopped. Read here rather
            // than assumed at message_stop, which called every reply that
            // wasn't a tool call a finished one — including a reply cut off
            // at the token ceiling.
            if let reason = decode(resp.delta?.stop_reason) {
                message.finishReason = reason
            }
        case .message_start:
            if let message_resp = resp.message {
                message.id = (message_resp.id != nil) ? message_resp.id! : message.id
                message.finishReason = decode(message_resp.stop_reason)
            }
        case .message_stop:
            // Only where the event above didn't say — it always does, but a
            // reply with no stop reason at all shouldn't read as unfinished.
            if message.finishReason == nil {
                message.finishReason = (message.toolCalls != nil) ? .toolCalls : .stop
            }
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
