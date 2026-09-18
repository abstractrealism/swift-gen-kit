import XCTest
@testable import GenKit

/// The run-loop limit is a hard stop, so as it nears, the model is told in
/// its tool results how many replies it has left. These drive the session
/// with a model that keeps calling tools until told it's on its last reply.
final class ChatSessionCountdownTests: XCTestCase {

    /// Calls a tool every round until the latest tool result says it's the
    /// last one, then answers. A model that heeds the notice.
    private struct HeedingService: ChatService {
        func completion(_ request: ChatServiceRequest) async throws -> Message {
            reply(to: request)
        }

        func completionStream(_ request: ChatServiceRequest, update: (Message) async throws -> Void) async throws {
            try await update(reply(to: request))
        }

        private func reply(to request: ChatServiceRequest) -> Message {
            let latestToolResult = request.messages.last(where: { $0.role == .tool })?.content ?? ""
            if latestToolResult.contains("last set of tool results") {
                return Message(role: .assistant, contents: [.text("answered")])
            }
            let call = ToolCall(type: "function", function: .init(name: "lookup", arguments: "{}"), custom: nil)
            return Message(role: .assistant, contents: [], toolCalls: [call])
        }
    }

    private static let model = Model(id: "heeding", owner: "test")

    private static let answersTool: ChatSessionRequest.ToolCallback = { call in
        .init(messages: [Message(role: .tool, contents: [.text("found something")], toolCallID: call.id, name: "lookup")], shouldContinue: true)
    }

    private func run(limit: Int, countdownFrom: Int) async throws -> [Message] {
        let request = ChatSessionRequest(service: HeedingService(), model: Self.model, toolCallback: Self.answersTool)
        var received: [Message] = []
        for try await message in ChatSession.shared.stream(request, runLoopLimit: limit, countdownFrom: countdownFrom) {
            received.append(message)
        }
        return received
    }

    func testTheCountdownIsAppendedToTheLastToolResultOfEachRound() async throws {
        let messages = try await run(limit: 5, countdownFrom: 3)
        let toolResults = messages.filter { $0.role == .tool }.map { $0.content ?? "" }

        // Five replies allowed. After tool round 1, four remain: no notice.
        XCTAssertEqual(toolResults.count, 4)
        XCTAssertFalse(toolResults[0].contains("Note:"))
        // After round 2, three remain: the warning. Then two, then the last.
        XCTAssertTrue(toolResults[1].contains("3 replies remain"), toolResults[1])
        XCTAssertTrue(toolResults[1].contains("call tools 2 more times"), toolResults[1])
        XCTAssertTrue(toolResults[2].contains("2 replies remain"), toolResults[2])
        XCTAssertTrue(toolResults[2].contains("call tools 1 more time,"), toolResults[2])
        XCTAssertTrue(toolResults[3].contains("last set of tool results"), toolResults[3])

        // The original result is still there, ahead of the note.
        XCTAssertTrue(toolResults[3].hasPrefix("found something"))
    }

    func testAModelThatHeedsTheCountdownFinishesWithinTheLimit() async throws {
        let messages = try await run(limit: 5, countdownFrom: 3)
        XCTAssertEqual(messages.last?.role, .assistant)
        XCTAssertEqual(messages.last?.content, "answered")
        XCTAssertEqual(messages.filter { $0.role == .assistant }.count, 5, "every allowed reply was used, and the last one answered")
    }

    func testWithoutTheCountdownTheSameModelHitsTheLimit() async {
        do {
            _ = try await run(limit: 5, countdownFrom: 0)
            XCTFail("expected the run-loop limit")
        } catch let error as ChatSessionError {
            guard case .maxRunLoopLimit = error else { return XCTFail("\(error)") }
        } catch {
            XCTFail("\(error)")
        }
    }

    func testTheNonStreamingPathCountsDownTheSameWay() async throws {
        let request = ChatSessionRequest(service: HeedingService(), model: Self.model, toolCallback: Self.answersTool)
        let response = try await ChatSession.shared.completion(request, runLoopLimit: 5, countdownFrom: 3)
        XCTAssertEqual(response.messages.last?.content, "answered")
        let notices = response.messages.filter { $0.role == .tool && ($0.content ?? "").contains("Note:") }
        XCTAssertEqual(notices.count, 3)
    }
}
