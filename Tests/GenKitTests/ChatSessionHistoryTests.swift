import XCTest
@testable import GenKit

/// A caller can decide what each round actually sends, without touching
/// what the session keeps.
final class ChatSessionHistoryTests: XCTestCase {

    /// Records what it was sent each round; calls a tool once, then answers.
    private final class RecordingService: ChatService, @unchecked Sendable {
        private let lock = NSLock()
        private(set) var sent: [[Message]] = []

        func completion(_ request: ChatServiceRequest) async throws -> Message {
            lock.withLock { sent.append(request.messages) }
            return reply(round: request.messages.filter { $0.role == .tool }.count)
        }

        func completionStream(_ request: ChatServiceRequest, update: (Message) async throws -> Void) async throws {
            lock.withLock { sent.append(request.messages) }
            try await update(reply(round: request.messages.filter { $0.role == .tool }.count))
        }

        private func reply(round: Int) -> Message {
            if round == 0 {
                let call = ToolCall(type: "function", function: .init(name: "lookup", arguments: "{}"), custom: nil)
                return Message(role: .assistant, contents: [.text("<think>\nscratch work\n</think>\n")], toolCalls: [call])
            }
            return Message(role: .assistant, contents: [.text("answer")])
        }
    }

    private static let answersTool: ChatSessionRequest.ToolCallback = { call in
        .init(messages: [Message(role: .tool, contents: [.text("found")], toolCallID: call.id, name: "lookup")], shouldContinue: true)
    }

    func testThePreparerShapesWhatIsSentAndNotWhatIsKept() async throws {
        let service = RecordingService()
        var request = ChatSessionRequest(service: service, model: Model(id: "m", owner: "t"), toolCallback: Self.answersTool)
        request.with(system: "be brief")
        request.with(history: [Message(role: .user, contents: [.text("hi")])])
        request.with(prepareHistory: { messages in
            messages.map { message in
                guard message.role == .assistant else { return message }
                var stripped = message
                stripped.contents = [.text("[reasoning removed]")]
                return stripped
            }
        })

        var kept: [Message] = []
        for try await message in ChatSession.shared.stream(request) {
            kept.append(message)
        }

        // Two rounds were sent. The second carried the first reply, stripped.
        XCTAssertEqual(service.sent.count, 2)
        let secondRound = service.sent[1]
        XCTAssertEqual(secondRound.first?.role, .system, "the system prompt is put back in front")
        XCTAssertTrue(secondRound.first?.content?.hasPrefix("be brief") ?? false)
        let sentReply = secondRound.first { $0.role == .assistant }
        XCTAssertEqual(sentReply?.content, "[reasoning removed]")
        XCTAssertNotNil(sentReply?.toolCalls, "only the content was touched")

        // What the session yielded is the original.
        let keptReply = kept.first { $0.role == .assistant }
        XCTAssertTrue(keptReply?.content?.contains("scratch work") ?? false)
    }

    func testWithoutAPreparerEverythingIsSentAsIs() async throws {
        let service = RecordingService()
        var request = ChatSessionRequest(service: service, model: Model(id: "m", owner: "t"), toolCallback: Self.answersTool)
        request.with(history: [Message(role: .user, contents: [.text("hi")])])
        for try await _ in ChatSession.shared.stream(request) {}
        let sentReply = service.sent[1].first { $0.role == .assistant }
        XCTAssertTrue(sentReply?.content?.contains("scratch work") ?? false)
    }
}
