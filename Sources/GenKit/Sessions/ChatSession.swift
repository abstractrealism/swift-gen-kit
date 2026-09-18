import Foundation
import OSLog
import SharedKit

private let logger = Logger(subsystem: "ChatSession", category: "GenKit")

public class ChatSession {
    public static let shared = ChatSession()

    /// How long a round may take to start answering, in seconds.
    ///
    /// Two figures, because the two waits are different in kind. The first
    /// round of a turn is a model reading a prompt it has mostly seen before,
    /// and a minute is generous. A later round is the model reading every
    /// tool result gathered so far — ten web searches' worth, by the fifth
    /// round of one measured turn — and a local model can take several
    /// minutes over that before its first token. One figure covering both
    /// is either too short for the second or far too long for the first.
    ///
    /// The bound is on the wait for the round's *first* message. Once the
    /// model has started, its stream is governed by the session's own
    /// inactivity timeout, as before. Nil means unbounded.
    public struct RoundTimeouts: Sendable {
        public var firstToken: TimeInterval?
        public var afterTools: TimeInterval?

        public init(firstToken: TimeInterval? = nil, afterTools: TimeInterval? = nil) {
            self.firstToken = firstToken
            self.afterTools = afterTools
        }

        public static let none = RoundTimeouts()

        func limit(afterTools: Bool) -> TimeInterval? {
            afterTools ? self.afterTools : firstToken
        }
    }

    public func stream(_ request: ChatSessionRequest, runLoopLimit: Int = 10, maxConcurrentToolCalls: Int = 3, timeouts: RoundTimeouts = .none, countdownFrom: Int = 3) -> AsyncThrowingStream<Message, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let runID = String.id
                    var messages = request.messages
                    var runLoopCount = 0
                    var runShouldContinue = true

                    while runShouldContinue {
                        try Task.checkCancellation()
                        guard runLoopCount < runLoopLimit else {
                            continuation.finish(throwing: ChatSessionError.maxRunLoopLimit)
                            return
                        }

                        // Prepare service request, DO NOT include a tool choice on subsequent runs, this will
                        // cause an expensive infinite loop of tool calls.
                        let req = ChatServiceRequest(
                            model: request.model,
                            messages: request.prepared(messages),
                            tools: request.tools,
                            toolChoice: (runLoopCount > 0) ? nil : request.tool, // FIRST REQUEST ONLY
                            options: request.options
                        )
                        let afterTools = runLoopCount > 0
                        let round = boundedStream(req, service: request.service, limit: timeouts.limit(afterTools: afterTools), afterTools: afterTools)
                        for try await message in round {
                            let message = apply(runID: runID, message: message)
                            messages = apply(message: message, messages: messages)
                            continuation.yield(message)
                        }

                        // Determine if there were any tool calls on the last message, process them by calling their
                        // repsective functions to return tool responses, then decide whether the loop should continue.
                        guard request.toolCallback != nil else {
                            break
                        }
                        let lastMessage = messages.last!
                        guard (lastMessage.toolCalls?.count ?? 0) > 0 else {
                            break
                        }
                        var (toolMessages, shouldContinue) = try await processToolCalls(in: lastMessage, callback: request.toolCallback, maxConcurrent: maxConcurrentToolCalls)
                        runLoopCount += 1
                        logger.info("tool round \(runLoopCount, privacy: .public) of \(runLoopLimit, privacy: .public): \(lastMessage.toolCalls?.count ?? 0, privacy: .public) calls answered, \(runLoopLimit - runLoopCount, privacy: .public) replies left")
                        countDown(in: &toolMessages, repliesLeft: runLoopLimit - runLoopCount, from: countdownFrom)
                        for message in toolMessages {
                            let message = apply(runID: runID, message: message)
                            messages = apply(message: message, messages: messages)
                            continuation.yield(message)
                        }
                        runShouldContinue = shouldContinue
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            // The work above runs in an unstructured task, so cancelling the
            // consumer does not reach it. Without this the run loop keeps
            // going after the consumer has stopped listening: the service
            // request stays open, and any pending tool-call round still fires.
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    public func completion(_ request: ChatSessionRequest, runLoopLimit: Int = 10, maxConcurrentToolCalls: Int = 3, timeouts: RoundTimeouts = .none, countdownFrom: Int = 3) async throws -> ChatSessionResponse {
        let runID = String.id

        var messages = request.messages
        var response = ChatSessionResponse(messages: [])

        var runLoopCount = 0
        var runShouldContinue = true

        while runShouldContinue {
            guard runLoopCount < runLoopLimit else {
                throw ChatSessionError.maxRunLoopLimit
            }

            // Prepare service request, DO NOT include a tool choice on subsequent runs, this will
            // cause an expensive infinite loop of tool calls.
            let req = ChatServiceRequest(
                model: request.model,
                messages: request.prepared(messages),
                tools: request.tools,
                toolChoice: (runLoopCount > 0) ? nil : request.tool, // FIRST REQUEST ONLY
                options: request.options
            )
            let afterTools = runLoopCount > 0
            var message = try await boundedCompletion(req, service: request.service, limit: timeouts.limit(afterTools: afterTools), afterTools: afterTools)
            message = apply(runID: runID, message: message)

            response.messages = apply(message: message, messages: response.messages)
            messages = apply(message: message, messages: messages)

            // Determine if there were any tool calls on the last message, process them by calling their
            // repsective functions to return tool responses, then decide whether the loop should continue.
            guard request.toolCallback != nil else {
                break
            }
            guard (message.toolCalls?.count ?? 0) > 0 else {
                break
            }
            var (toolMessages, shouldContinue) = try await processToolCalls(in: message, callback: request.toolCallback, maxConcurrent: maxConcurrentToolCalls)
            logger.info("tool round \(runLoopCount + 1, privacy: .public) of \(runLoopLimit, privacy: .public): \(message.toolCalls?.count ?? 0, privacy: .public) calls answered, \(runLoopLimit - runLoopCount - 1, privacy: .public) replies left")
            countDown(in: &toolMessages, repliesLeft: runLoopLimit - (runLoopCount + 1), from: countdownFrom)
            for message in toolMessages {
                let message = apply(runID: runID, message: message)
                response.messages = apply(message: message, messages: response.messages)
                messages = apply(message: message, messages: messages)
            }
            runShouldContinue = shouldContinue
            runLoopCount += 1
        }

        return response
    }

    /// Tells the model, in its tool results, how many replies it has left.
    ///
    /// The run-loop limit is a hard stop: a model still calling tools when
    /// it's reached gets an error, not an answer, and a hard question can
    /// take a model that many rounds — one measured turn found its answer
    /// by round eight and was still verifying at ten. Told the count, it
    /// can conclude instead. Said once the remaining replies are down to
    /// `countdownFrom` (three: a warning, a last chance, and the answer),
    /// on the last tool result of the round so it's the freshest thing the
    /// model reads. Appended to the result rather than sent as a message of
    /// its own, since not every provider allows a message between a tool
    /// result and the reply to it.
    private func countDown(in toolMessages: inout [Message], repliesLeft: Int, from countdownFrom: Int) {
        guard countdownFrom > 0, repliesLeft <= countdownFrom, repliesLeft >= 1, !toolMessages.isEmpty else { return }
        let notice: String
        if repliesLeft == 1 {
            notice = "Note: this is the last set of tool results for this turn. Reply now with your answer, using what you have — any further tool calls will not be answered."
        } else {
            let rounds = repliesLeft - 1
            notice = "Note: \(repliesLeft) replies remain in this turn, including your final answer. You may call tools \(rounds) more \(rounds == 1 ? "time" : "times"), then you must answer with what you have."
        }
        logger.info("countdown: told the model \(repliesLeft, privacy: .public) \(repliesLeft == 1 ? "reply remains" : "replies remain", privacy: .public)")
        let index = toolMessages.count - 1
        var last = toolMessages[index]
        if let contents = last.contents, let textIndex = contents.lastIndex(where: { if case .text = $0 { true } else { false } }),
           case .text(let text) = contents[textIndex] {
            last.contents![textIndex] = .text(text + "\n\n" + notice)
        } else {
            last.contents = (last.contents ?? []) + [.text(notice)]
        }
        toolMessages[index] = last
    }

    /// One round's stream, given up on if nothing has arrived within `limit`.
    ///
    /// The service's stream is consumed in a task of its own and relayed;
    /// a watchdog sleeps for the limit and, if the first message hasn't come
    /// by then, fails the relay with a timeout and cancels the request. Once
    /// something has arrived the watchdog does nothing — the bound is on
    /// starting, not on finishing — and the round runs on to its end.
    private func boundedStream(_ req: ChatServiceRequest, service: ChatService, limit: TimeInterval?, afterTools: Bool) -> AsyncThrowingStream<Message, Error> {
        AsyncThrowingStream { continuation in
            let started = Started()
            let producer = Task {
                do {
                    try await service.completionStream(req) { message in
                        started.mark()
                        continuation.yield(message)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            let watchdog: Task<Void, Never>? = limit.map { limit in
                Task {
                    try? await Task.sleep(for: .seconds(limit))
                    guard !Task.isCancelled, !started.hasStarted else { return }
                    continuation.finish(throwing: ChatSessionError.timedOut(after: limit, afterTools: afterTools))
                    producer.cancel()
                }
            }
            continuation.onTermination = { _ in
                producer.cancel()
                watchdog?.cancel()
            }
        }
    }

    /// The non-streaming counterpart: the whole reply must arrive within
    /// the limit, there being no first message to wait for.
    private func boundedCompletion(_ req: ChatServiceRequest, service: ChatService, limit: TimeInterval?, afterTools: Bool) async throws -> Message {
        guard let limit else { return try await service.completion(req) }
        return try await withThrowingTaskGroup(of: Message?.self) { group in
            group.addTask { try await service.completion(req) }
            group.addTask {
                try await Task.sleep(for: .seconds(limit))
                return nil
            }
            defer { group.cancelAll() }
            guard let first = try await group.next(), let message = first else {
                throw ChatSessionError.timedOut(after: limit, afterTools: afterTools)
            }
            return message
        }
    }

    /// Whether a round's first message has arrived. A flag rather than an
    /// actor, since it's set from a service's update closure and read from
    /// a watchdog, neither of which wants to await.
    private final class Started: @unchecked Sendable {
        private let lock = NSLock()
        private var flag = false
        var hasStarted: Bool { lock.withLock { flag } }
        func mark() { lock.withLock { flag = true } }
    }

    func processToolCalls(in message: Message, callback: ChatSessionRequest.ToolCallback?, maxConcurrent: Int) async throws -> ([Message], Bool) {
        guard let callback else { return ([], false) }
        guard let toolCalls = message.toolCalls else { return ([], false) }
        let runID = message.runID

        // Parallelize tool calls.
        var responses: [ToolCallResponse] = []
        await withTaskGroup(of: ToolCallResponse.self) { group in
            for toolCall in toolCalls {
                group.addTask {
                    do {
                        return try await callback(toolCall)
                    } catch {
                        let message = Message(
                            role: .tool,
                            content: "Unknown tool.",
                            toolCallID: toolCall.id,
                            name: toolCall.function?.name ?? toolCall.custom?.name,
                            metadata: ["label": .string("Unknown tool")]
                        )
                        return .init(messages: [message], shouldContinue: false)
                    }
                }
            }
            for await response in group {
                responses.append(response)
            }
        }

        // Flatten messages from task responses and annotate each message with a Run identifier.
        let messages = responses
            .flatMap { $0.messages }
            .map {
                var message = $0
                message.runID = runID
                return message
            }

        // If any task response suggests the Run should stop, stop it.
        let shouldContinue = !responses.contains(where: { $0.shouldContinue == false })

        return (messages, shouldContinue)
    }

    func apply(message: Message, messages: [Message]) -> [Message] {
        var messages = messages
        if let index = messages.firstIndex(where: { $0.id == message.id }) {
            messages[index] = message
            return messages
        } else {
            messages.append(message)
            return messages
        }
    }

    func apply(runID: String?, message: Message) -> Message {
        var message = message
        message.runID = runID
        return message
    }
}

// MARK: - Types

public struct ChatSessionRequest {
    public typealias ToolCallback = @Sendable (ToolCall) async throws -> ToolCallResponse

    /// Last word on what a round's request carries, given everything the
    /// turn has accumulated so far — the history handed in plus every reply
    /// and tool result since. What it returns is what's sent; what the
    /// session keeps and yields is untouched.
    ///
    /// The session builds each round from the whole turn, and a turn that
    /// searches grows by thousands of tokens a round: every search's
    /// results, every reply's reasoning, all re-read by the model before it
    /// can say a word. Which of that is still worth sending is the caller's
    /// call — it knows the provider and the setting — so it's a hook here
    /// rather than a policy.
    public typealias HistoryPreparer = @Sendable ([Message]) -> [Message]

    public let service: ChatService
    public let model: Model
    public let toolCallback: ToolCallback?

    public private(set) var system: String? = nil
    public private(set) var history: [Message] = []
    public private(set) var tools: [Tool] = []
    public private(set) var tool: Tool? = nil
    public private(set) var context: [String: Value] = [:]
    public private(set) var options: [String: Value] = [:]
    public private(set) var prepareHistory: HistoryPreparer? = nil

    public init(service: ChatService, model: Model, toolCallback: ToolCallback? = nil) {
        self.service = service
        self.model = model
        self.toolCallback = toolCallback
    }

    public mutating func with(system: String?) {
        self.system = system
    }

    public mutating func with(history: [Message]) {
        self.history = history
    }

    public mutating func with(tools: [Tool]) {
        self.tools = tools
    }

    public mutating func with(tool: Tool?) {
        if let tool {
            self.tool = tool
            self.tools.append(tool)
        } else {
            self.tool = nil
        }
    }

    public mutating func with(context: [String: Value]) {
        self.context = context
    }

    public mutating func with(option key: String, value: Value) {
        options[key] = value
    }

    public mutating func with(prepareHistory: HistoryPreparer?) {
        self.prepareHistory = prepareHistory
    }

    /// The messages to send this round: the turn so far, prepared.
    ///
    /// The system message is set aside first and put back after, so the
    /// preparer sees exactly the conversation — what a caller thinks of as
    /// history — and can't lose the prompt by accident.
    func prepared(_ messages: [Message]) -> [Message] {
        guard let prepareHistory else { return messages }
        let system = messages.prefix { $0.role == .system }
        let rest = messages.dropFirst(system.count)
        return Array(system) + prepareHistory(Array(rest))
    }

    var messages: [Message] {
        var messages: [Message] = []

        // Apply user context
        var systemContext = ""
        if let memories = context["MEMORIES"] {
            systemContext = """
            The following is context about the current user:
            <user_context>
            \(memories)
            </user_context>
            """
        }

        // Apply system prompt
        if let system {
            let prompt = [system, systemContext].joined(separator: "\n\n")
            messages.append(.init(role: .system, content: prompt))
        }

        // Apply history
        messages += history

        return messages
    }
}

public struct ChatSessionResponse: Sendable {
    public var messages: [Message]

    public func extractTool<T: Codable>(name: String, type: T.Type) throws -> T {
        guard let message = messages.last else {
            throw ChatSessionError.missingMessage
        }
        guard let toolCalls = message.toolCalls else {
            throw ChatSessionError.missingToolCalls
        }
        guard let toolCall = toolCalls.first(where: { $0.function?.name == name || $0.custom?.name == name }) else {
            throw ChatSessionError.missingToolCall
        }
        guard let data = toolCall.function?.arguments.data(using: .utf8) ?? toolCall.custom?.input.data(using: .utf8) else {
            throw ChatSessionError.unknown
        }
        return try JSONDecoder().decode(type, from: data)
    }
}

/// Public, so a caller can tell these apart from a service's own failures.
/// Thrown across the module boundary but declared internal, they reached an
/// app only as a string — "maxRunLoopLimit" — which is neither matchable nor
/// something a person can act on.
public enum ChatSessionError: Error, CustomStringConvertible {
    case missingMessage
    case missingToolCalls
    case missingToolCall
    case missingTool
    /// The run loop went round `runLoopLimit` times — a tool call answered
    /// and the model calling again, every time — without the model ever
    /// finishing an answer. A ceiling on rounds of tool calls, not on tokens.
    case maxRunLoopLimit
    /// A round didn't start answering within the time allowed — see
    /// `RoundTimeouts`. Says which wait it was, since the two are sized
    /// differently and mean different things to whoever reads it.
    case timedOut(after: TimeInterval, afterTools: Bool)
    case unknown

    public var description: String {
        switch self {
        case .missingMessage:
            "The response carried no message."
        case .missingToolCalls:
            "The message carried no tool calls to process."
        case .missingToolCall:
            "A tool call was expected and not found."
        case .missingTool:
            "The model called a tool this session doesn't have."
        case .maxRunLoopLimit:
            "The model kept calling tools without finishing an answer, and the limit on rounds of tool calls was reached."
        case .timedOut(let seconds, let afterTools):
            afterTools
                ? "The model didn't start answering within \(Int(seconds)) seconds of receiving the tool results."
                : "The model didn't start answering within \(Int(seconds)) seconds."
        case .unknown:
            "An unknown chat session error occurred."
        }
    }
}
