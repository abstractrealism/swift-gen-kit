import XCTest
@testable import GenKit

/// A round of a turn has a bound on how long it may take to *start*: one
/// figure for the first round, another for a round that follows tool results.
/// These drive the session with a service that answers on a schedule.
final class ChatSessionTimeoutTests: XCTestCase {

    /// Answers each round after a delay: a tool call for every round but the
    /// last, then a plain reply. Delays are per round, in seconds.
    private struct ScheduledService: ChatService {
        let delays: [TimeInterval]
        let toolRounds: Int

        func completion(_ request: ChatServiceRequest) async throws -> Message {
            let round = request.messages.filter { $0.role == .tool }.count
            try await Task.sleep(for: .seconds(delays[min(round, delays.count - 1)]))
            return reply(round: round)
        }

        func completionStream(_ request: ChatServiceRequest, update: (Message) async throws -> Void) async throws {
            let round = request.messages.filter { $0.role == .tool }.count
            try await Task.sleep(for: .seconds(delays[min(round, delays.count - 1)]))
            try await update(reply(round: round))
        }

        private func reply(round: Int) -> Message {
            if round < toolRounds {
                let call = ToolCall(type: "function", function: .init(name: "lookup", arguments: "{}"), custom: nil)
                return Message(role: .assistant, contents: [], toolCalls: [call])
            }
            return Message(role: .assistant, contents: [.text("done after \(round) tool rounds")])
        }
    }

    private static let model = Model(id: "scheduled", owner: "test")

    private static let answersTool: ChatSessionRequest.ToolCallback = { call in
        .init(messages: [Message(role: .tool, contents: [.text("found")], toolCallID: call.id, name: "lookup")], shouldContinue: true)
    }

    private func run(_ service: ScheduledService, timeouts: ChatSession.RoundTimeouts, tools: Bool = false) async throws -> [Message] {
        let request = ChatSessionRequest(service: service, model: Self.model, toolCallback: tools ? Self.answersTool : nil)
        var received: [Message] = []
        for try await message in ChatSession.shared.stream(request, timeouts: timeouts) {
            received.append(message)
        }
        return received
    }

    func testAFirstRoundThatStartsInTimeCompletes() async throws {
        let messages = try await run(ScheduledService(delays: [0.2], toolRounds: 0), timeouts: .init(firstToken: 2))
        XCTAssertEqual(messages.last?.content, "done after 0 tool rounds")
    }

    func testAFirstRoundThatDoesNotStartInTimeFails() async {
        do {
            _ = try await run(ScheduledService(delays: [1.5], toolRounds: 0), timeouts: .init(firstToken: 0.3))
            XCTFail("expected a timeout")
        } catch let error as ChatSessionError {
            guard case .timedOut(let seconds, let afterTools) = error else { return XCTFail("\(error)") }
            XCTAssertEqual(seconds, 0.3)
            XCTAssertFalse(afterTools)
        } catch {
            XCTFail("\(error)")
        }
    }

    /// The second figure governs rounds after tool results, the first the
    /// first round. Here the first round answers at once and the second
    /// takes a second: too slow for a tight after-tools bound, fine for a
    /// generous one, and the first-token bound never enters into it.
    func testARoundAfterToolsIsBoundByTheSecondFigure() async throws {
        let slowSecondRound = ScheduledService(delays: [0.05, 1.0], toolRounds: 1)

        let messages = try await run(slowSecondRound, timeouts: .init(firstToken: 0.3, afterTools: 3), tools: true)
        XCTAssertEqual(messages.last?.content, "done after 1 tool rounds")

        do {
            _ = try await run(slowSecondRound, timeouts: .init(firstToken: 3, afterTools: 0.3), tools: true)
            XCTFail("expected a timeout after tools")
        } catch let error as ChatSessionError {
            guard case .timedOut(_, let afterTools) = error else { return XCTFail("\(error)") }
            XCTAssertTrue(afterTools)
        }
    }

    func testNoBoundMeansNoTimeout() async throws {
        let messages = try await run(ScheduledService(delays: [0.6], toolRounds: 0), timeouts: .none)
        XCTAssertEqual(messages.count, 1)
    }

    func testTheNonStreamingPathIsBoundTheSameWay() async {
        let request = ChatSessionRequest(service: ScheduledService(delays: [1.5], toolRounds: 0), model: Self.model)
        do {
            _ = try await ChatSession.shared.completion(request, timeouts: .init(firstToken: 0.3))
            XCTFail("expected a timeout")
        } catch let error as ChatSessionError {
            guard case .timedOut = error else { return XCTFail("\(error)") }
        } catch {
            XCTFail("\(error)")
        }
    }

    func testTheErrorSaysWhichWait() {
        XCTAssertEqual("\(ChatSessionError.timedOut(after: 60, afterTools: false))", "The model didn't start answering within 60 seconds.")
        XCTAssertEqual("\(ChatSessionError.timedOut(after: 300, afterTools: true))", "The model didn't start answering within 300 seconds of receiving the tool results.")
    }
}
