import XCTest
import Anthropic
import OpenAI
@testable import GenKit

/// Asking Anthropic to cache the part of the request that repeats.
///
/// Every request carries the whole conversation, the endpoint holding none
/// of it — so a turn with eight tool rounds sends the same growing prefix
/// eight times. Two breakpoints: the system prompt, which also covers the
/// tools sent ahead of it and is identical for the life of a conversation,
/// and the end of the conversation so far, which each round extends.
final class AnthropicPromptCachingTests: XCTestCase {

    private let service = AnthropicService(apiKey: "test")

    private func request(_ messages: [Message]) async -> Anthropic.ChatRequest {
        await service.makeRequestForTesting(
            model: Model(id: "claude-sonnet-5", owner: "anthropic"),
            messages: messages
        )
    }

    private static let conversation: [Message] = [
        Message(role: .system, content: "Be brief."),
        Message(role: .user, content: "What is 17 times 23?"),
        Message(role: .assistant, content: "391."),
        Message(role: .user, content: "And 18 times 23?"),
    ]

    func testTheSystemPromptIsMarked() async {
        let sent = await request(Self.conversation)
        XCTAssertEqual(sent.system?.count, 1)
        XCTAssertEqual(sent.system?.first?.cache_control?.type, .ephemeral)
        XCTAssertEqual(sent.system?.first?.text, "Be brief.")
    }

    /// On the last block of the last message, so the cached prefix is
    /// everything up to here and the next round's prefix extends it.
    func testTheEndOfTheConversationIsMarked() async {
        let sent = await request(Self.conversation)
        XCTAssertEqual(sent.messages.count, 3, "the system message isn't one of them")

        let last = try? XCTUnwrap(sent.messages.last)
        XCTAssertEqual(last?.content.last?.cache_control?.type, .ephemeral)
        XCTAssertEqual(last?.content.last?.text, "And 18 times 23?")
    }

    /// One breakpoint, not one per message: each marks a prefix, and marking
    /// every message would spend all four of them on the same conversation.
    func testEarlierMessagesAreNotMarked() async {
        let sent = await request(Self.conversation)
        let earlier = sent.messages.dropLast()
        XCTAssertFalse(earlier.isEmpty)
        for message in earlier {
            XCTAssertTrue(
                message.content.allSatisfy { $0.cache_control == nil },
                "only the end of the conversation is a breakpoint"
            )
        }
    }

    /// A turn whose last message is a tool result — which is every round
    /// after the first. `tool_result` takes a marker; a thinking block would
    /// not, and none is sent.
    func testAToolResultCanCarryTheMarker() async {
        var messages = Self.conversation
        messages.append(Message(role: .tool, content: "found something", toolCallID: "t1", name: "lookup"))

        let sent = await request(messages)
        XCTAssertEqual(sent.messages.last?.content.last?.cache_control?.type, .ephemeral)
    }

    func testAConversationWithNoHistoryStillSendsASystemBreakpoint() async {
        let sent = await request([Message(role: .system, content: "Be brief.")])
        XCTAssertEqual(sent.system?.first?.cache_control?.type, .ephemeral)
        XCTAssertTrue(sent.messages.isEmpty)
    }
}

/// What the services report about caching, which is what says whether any
/// of it is working.
final class CacheReportingTests: XCTestCase {

    func testAnthropicReadsAndWritesAreRecordedApart() throws {
        let event = #"""
        {"type":"message_start","message":{"id":"m","type":"message","role":"assistant","model":"claude-sonnet-5","content":[],"stop_reason":null,"usage":{"input_tokens":653,"cache_creation_input_tokens":0,"cache_read_input_tokens":21952,"output_tokens":1}}}
        """#
        let decoded = try JSONDecoder().decode(Anthropic.ChatResponseStream.self, from: Data(event.utf8))

        var message = Message(role: .assistant)
        AnthropicService(apiKey: "test").patchMessage(&message, with: decoded)

        // Apart, because they cost differently: a read is a tenth of an
        // ordinary input token and a write a quarter more.
        XCTAssertEqual(message.metadata["cachedTokens"]?.intValue, 21952)
        XCTAssertNil(message.metadata["cacheWriteTokens"], "nothing was written this round")
        // And counted in the input either way, cached or not.
        XCTAssertEqual(message.metadata["inputTokens"]?.intValue, 653 + 21952)
    }

    func testAnthropicWritesAreRecorded() throws {
        let event = #"""
        {"type":"message_start","message":{"id":"m","type":"message","role":"assistant","model":"claude-sonnet-5","content":[],"stop_reason":null,"usage":{"input_tokens":12,"cache_creation_input_tokens":4096,"cache_read_input_tokens":0,"output_tokens":1}}}
        """#
        let decoded = try JSONDecoder().decode(Anthropic.ChatResponseStream.self, from: Data(event.utf8))

        var message = Message(role: .assistant)
        AnthropicService(apiKey: "test").patchMessage(&message, with: decoded)
        XCTAssertEqual(message.metadata["cacheWriteTokens"]?.intValue, 4096)
        XCTAssertNil(message.metadata["cachedTokens"])
    }

    /// DeepSeek caches without being asked and reports it under its own
    /// name; OpenAI reports the same figure in the details.
    func testDeepSeekCacheHitsAreRecorded() throws {
        let chunk = #"""
        {"id":"x","object":"chat.completion.chunk","created":1,"model":"deepseek-flash","choices":[],"usage":{"prompt_tokens":22605,"completion_tokens":412,"total_tokens":23017,"prompt_cache_hit_tokens":21952,"prompt_cache_miss_tokens":653}}
        """#
        let decoded = try JSONDecoder().decode(OpenAI.ChatStreamResponse.self, from: Data(chunk.utf8))

        var message = Message(role: .assistant)
        OpenAIService(apiKey: "test", dialect: .deepSeek).patchMessage(&message, with: decoded)
        XCTAssertEqual(message.metadata["cachedTokens"]?.intValue, 21952)
        XCTAssertEqual(message.metadata["inputTokens"]?.intValue, 22605)
    }

    func testOpenAICacheHitsAreRecordedFromTheDetails() throws {
        let chunk = #"""
        {"id":"x","object":"chat.completion.chunk","created":1,"model":"gpt-5.6-sol","choices":[],"usage":{"prompt_tokens":2048,"completion_tokens":30,"total_tokens":2078,"prompt_tokens_details":{"cached_tokens":1024}}}
        """#
        let decoded = try JSONDecoder().decode(OpenAI.ChatStreamResponse.self, from: Data(chunk.utf8))

        var message = Message(role: .assistant)
        OpenAIService(apiKey: "test").patchMessage(&message, with: decoded)
        XCTAssertEqual(message.metadata["cachedTokens"]?.intValue, 1024)
    }

    func testNothingIsClaimedWhenNothingIsReported() throws {
        let chunk = #"""
        {"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[],"usage":{"prompt_tokens":20,"completion_tokens":30,"total_tokens":50}}
        """#
        let decoded = try JSONDecoder().decode(OpenAI.ChatStreamResponse.self, from: Data(chunk.utf8))

        var message = Message(role: .assistant)
        OpenAIService(apiKey: "test").patchMessage(&message, with: decoded)
        XCTAssertNil(message.metadata["cachedTokens"])
    }
}

/// Blocks Anthropic won't accept.
///
/// "messages: text content blocks must be non-empty" — a 400 on the whole
/// request, which is how a second prompt to `claude-haiku-4-5` failed with
/// three messages in the history and nothing obviously wrong with any of
/// them.
///
/// Nothing the model produces is empty. These come from stripping: an
/// assistant turn that reasoned and never reached an answer is a `<think>`
/// block and nothing else, and a caller that takes reasoning out of the
/// history before re-sending it — which Heat does by default, that reasoning
/// being working rather than something the assistant said — is left holding
/// an empty string.
final class AnthropicEmptyBlockTests: XCTestCase {

    private let service = AnthropicService(apiKey: "test")

    private func request(_ messages: [Message]) async -> Anthropic.ChatRequest {
        await service.makeRequestForTesting(
            model: Model(id: "claude-haiku-4-5-20251001", owner: "anthropic"),
            messages: messages
        )
    }

    /// The failing shape, as the history arrives after stripping.
    func testAnAssistantTurnStrippedToNothingIsLeftOut() async {
        let sent = await request([
            Message(role: .system, content: "Be brief."),
            Message(role: .user, content: "What is 17 times 23?"),
            Message(role: .assistant, content: ""),
            Message(role: .user, content: "And 18 times 23?"),
        ])

        XCTAssertEqual(sent.messages.count, 2)
        XCTAssertEqual(sent.messages.map(\.role), [.user, .user])
        XCTAssertTrue(
            sent.messages.allSatisfy { $0.content.allSatisfy { ($0.text ?? "x").isEmpty == false } },
            "no empty text block reaches the request"
        )
    }

    /// An empty block among good ones goes, and the message stays.
    func testAnEmptyBlockBesideARealOneIsDropped() async {
        var message = Message(role: .assistant)
        message.contents = [.text(""), .text("391.")]

        let sent = await request([message])
        XCTAssertEqual(sent.messages.count, 1)
        XCTAssertEqual(sent.messages.first?.content.count, 1)
        XCTAssertEqual(sent.messages.first?.content.first?.text, "391.")
    }

    /// And the cache breakpoint lands on what's left, not on what went.
    func testTheBreakpointFollowsTheBlocksThatRemain() async {
        var stripped = Message(role: .assistant)
        stripped.contents = [.text("")]

        let sent = await request([
            Message(role: .user, content: "hi"),
            stripped,
        ])
        XCTAssertEqual(sent.messages.count, 1)
        XCTAssertEqual(sent.messages.first?.content.last?.text, "hi")
        XCTAssertEqual(sent.messages.first?.content.last?.cache_control?.type, .ephemeral)
    }

    /// Whitespace is content. Only an actually empty string is refused, and
    /// guessing at which whitespace the model meant isn't this layer's
    /// business.
    func testWhitespaceIsLeftAlone() async {
        var message = Message(role: .assistant)
        message.contents = [.text("\n")]

        let sent = await request([message])
        XCTAssertEqual(sent.messages.count, 1)
        XCTAssertEqual(sent.messages.first?.content.first?.text, "\n")
    }
}
