import XCTest
import Anthropic
import OpenAI
@testable import GenKit

/// What a reply cost, carried on the message.
///
/// Only the Ollama provider recorded it, so every hosted service answered
/// with no counts at all and an app showing them showed nothing — which is
/// how a reply that stopped at the token ceiling could not be told from one
/// that finished, there being no figure to compare against anything.
final class OpenAIUsageRecordingTests: XCTestCase {

    private let service = OpenAIService(apiKey: "test", dialect: .groq)

    private func chunk(_ json: String) throws -> OpenAI.ChatStreamResponse {
        try JSONDecoder().decode(OpenAI.ChatStreamResponse.self, from: Data(json.utf8))
    }

    /// Verbatim shapes from a Groq `openai/gpt-oss-120b` stream.
    private static let reasoningChunk = #"""
    {"id":"x","object":"chat.completion.chunk","created":1,"model":"openai/gpt-oss-120b","choices":[{"index":0,"delta":{"reasoning":"17","channel":"analysis"},"finish_reason":null}]}
    """#

    private static let finalChunk = #"""
    {"id":"x","object":"chat.completion.chunk","created":1,"model":"openai/gpt-oss-120b","choices":[{"index":0,"delta":{},"finish_reason":"length"}],"usage":{"queue_time":0.046,"prompt_tokens":78,"prompt_time":0.004,"completion_tokens":1862,"completion_time":1.693,"total_tokens":1940,"total_time":1.697}}
    """#

    /// OpenAI's own final chunk has no choices at all — which is why usage
    /// has to be read before the message is patched, not after.
    private static let usageOnlyChunk = #"""
    {"id":"x","object":"chat.completion.chunk","created":1,"model":"gpt-5.6-sol","choices":[],"usage":{"prompt_tokens":20,"completion_tokens":30,"total_tokens":50,"completion_tokens_details":{"reasoning_tokens":12}}}
    """#

    func testTheCountsAndTheModelAreRecordedFromAStream() throws {
        var message = Message(role: .assistant)
        service.patchMessage(&message, with: try chunk(Self.reasoningChunk))
        XCTAssertEqual(message.metadata["model"]?.stringValue, "openai/gpt-oss-120b")
        XCTAssertNil(message.metadata["outputTokens"], "nothing to report until usage arrives")

        service.patchMessage(&message, with: try chunk(Self.finalChunk))
        XCTAssertEqual(message.metadata["inputTokens"]?.intValue, 78)
        XCTAssertEqual(message.metadata["outputTokens"]?.intValue, 1862)
        XCTAssertEqual(message.metadata["outputSeconds"]?.doubleValue, 1.693, "generation alone, not the queue wait")
        XCTAssertEqual(message.finishReason, .length)
    }

    func testAChunkCarryingUsageAndNothingElseIsStillRead() throws {
        var message = Message(role: .assistant)
        service.patchMessage(&message, with: try chunk(Self.usageOnlyChunk))
        XCTAssertEqual(message.metadata["inputTokens"]?.intValue, 20)
        XCTAssertEqual(message.metadata["outputTokens"]?.intValue, 30)
        XCTAssertEqual(message.metadata["reasoningTokens"]?.intValue, 12, "the counted figure, not the caller's estimate")
        XCTAssertNil(message.metadata["thinkingTokens"])
        // No timings outside Groq, so no rate is claimed.
        XCTAssertNil(message.metadata["outputSeconds"])
    }

    /// Grok reports reasoning apart from the completion where everything else
    /// reports it within.
    ///
    /// Measured on 28 Sept 2026 against `api.x.ai`: 306 completion tokens and
    /// 1,396 reasoning ones. Read as a share of the output that made the
    /// answer −1,090 tokens long, which is how it was noticed. The totals are
    /// what settle it — `total_tokens` here accounts for all three figures,
    /// where OpenAI's accounts for two.
    private static let grokChunk = #"""
    {"id":"x","object":"chat.completion.chunk","created":1,"model":"grok-4","choices":[],"usage":{"prompt_tokens":832,"completion_tokens":306,"total_tokens":2534,"completion_tokens_details":{"reasoning_tokens":1396}}}
    """#

    func testGrokReasoningIsAddedToTheOutputRatherThanTakenFromIt() throws {
        var message = Message(role: .assistant)
        service.patchMessage(&message, with: try chunk(Self.grokChunk))

        XCTAssertEqual(message.metadata["inputTokens"]?.intValue, 832)
        XCTAssertEqual(message.metadata["reasoningTokens"]?.intValue, 1396)
        XCTAssertEqual(
            message.metadata["outputTokens"]?.intValue, 1702,
            "the completion plus the reasoning, since the service's total counts both"
        )

        // The thing that was actually wrong on screen: what's left for the
        // answer once the reasoning is taken out.
        let output = try XCTUnwrap(message.metadata["outputTokens"]?.intValue)
        let reasoning = try XCTUnwrap(message.metadata["reasoningTokens"]?.intValue)
        XCTAssertEqual(output - reasoning, 306, "the answer, and not a negative number")
    }

    /// The same shape from OpenAI, where the total accounts for two figures
    /// rather than three, must not be touched.
    func testOpenAIReasoningStaysAShareOfTheOutput() throws {
        var message = Message(role: .assistant)
        service.patchMessage(&message, with: try chunk(Self.usageOnlyChunk))
        XCTAssertEqual(message.metadata["outputTokens"]?.intValue, 30, "unchanged: 20 + 30 == 50")
        XCTAssertEqual(message.metadata["reasoningTokens"]?.intValue, 12)
    }

    /// The sizes outrank the total, because they are arithmetic and it is
    /// convention. Grok's own total was never captured — the figure on screen
    /// was Heat's own sum — so a service that reports reasoning separately
    /// while totalling as though it hadn't must still come out right.
    func testReasoningLargerThanTheCompletionWinsOverAnAgreeableTotal() throws {
        let json = #"""
        {"id":"x","object":"chat.completion.chunk","created":1,"model":"grok-4","choices":[],"usage":{"prompt_tokens":832,"completion_tokens":306,"total_tokens":1138,"completion_tokens_details":{"reasoning_tokens":1396}}}
        """#
        var message = Message(role: .assistant)
        service.patchMessage(&message, with: try chunk(json))
        XCTAssertEqual(
            message.metadata["outputTokens"]?.intValue, 1702,
            "1,396 cannot be part of 306, whatever the total claims"
        )
    }

    /// With no total to go on, the sizes decide — reasoning larger than the
    /// completion cannot be a part of it.
    func testWithoutATotalTheLargerReasoningIsTakenAsExtra() throws {
        let extra = #"""
        {"id":"x","object":"chat.completion.chunk","created":1,"model":"grok-4","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":40,"completion_tokens_details":{"reasoning_tokens":900}}}
        """#
        var message = Message(role: .assistant)
        service.patchMessage(&message, with: try chunk(extra))
        XCTAssertEqual(message.metadata["outputTokens"]?.intValue, 940)

        // And the other way: reasoning that fits inside the completion is
        // left where it is, because it can be a share of it.
        let within = #"""
        {"id":"x","object":"chat.completion.chunk","created":1,"model":"gpt-5.6-sol","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":900,"completion_tokens_details":{"reasoning_tokens":40}}}
        """#
        var other = Message(role: .assistant)
        service.patchMessage(&other, with: try chunk(within))
        XCTAssertEqual(other.metadata["outputTokens"]?.intValue, 900)
    }

    /// A stream reports usage on one chunk out of thousands, and the patch
    /// runs on all of them. What arrived must not be undone by what follows.
    func testALaterChunkDoesNotEraseWhatWasRecorded() throws {
        var message = Message(role: .assistant)
        service.patchMessage(&message, with: try chunk(Self.finalChunk))
        service.patchMessage(&message, with: try chunk(Self.reasoningChunk))
        XCTAssertEqual(message.metadata["outputTokens"]?.intValue, 1862)
    }

    func testAStreamIsAskedForItsUsage() async {
        let request = ChatServiceRequest(model: Model(id: "openai/gpt-oss-120b", owner: "groq"), messages: [])
        let streamed = await service.makeRequest(request, streaming: true)
        XCTAssertEqual(streamed.stream_options?.include_usage, true)
        let completion = await service.makeRequest(request, streaming: false)
        XCTAssertNil(completion.stream_options, "not a stream, so there is nothing to ask for")
    }
}

final class AnthropicUsageRecordingTests: XCTestCase {

    private let service = AnthropicService(apiKey: "test")

    private func event(_ json: String) throws -> Anthropic.ChatResponseStream {
        try JSONDecoder().decode(Anthropic.ChatResponseStream.self, from: Data(json.utf8))
    }

    private static let start = #"""
    {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-5","content":[],"stop_reason":null,"usage":{"input_tokens":8,"cache_creation_input_tokens":0,"cache_read_input_tokens":2048,"output_tokens":1}}}
    """#

    private static let cutOff = #"""
    {"type":"message_delta","delta":{"stop_reason":"max_tokens","stop_sequence":null},"usage":{"output_tokens":4096}}
    """#

    private static let finished = #"""
    {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":14}}
    """#

    private static let stop = #"{"type":"message_stop"}"#

    func testTheInputCountIncludesWhatWasReadFromCache() throws {
        var message = Message(role: .assistant)
        service.patchMessage(&message, with: try event(Self.start))
        // Cached reads are prompt tokens billed differently, not extra ones,
        // and `input_tokens` counts neither: 8 alone understates what was
        // sent by the whole cached prompt.
        XCTAssertEqual(message.metadata["inputTokens"]?.intValue, 2056)
        XCTAssertEqual(message.metadata["model"]?.stringValue, "claude-sonnet-5")
    }

    func testAReplyCutOffAtTheCeilingSaysSo() throws {
        var message = Message(role: .assistant)
        service.patchMessage(&message, with: try event(Self.start))
        service.patchMessage(&message, with: try event(Self.cutOff))
        service.patchMessage(&message, with: try event(Self.stop))
        XCTAssertEqual(message.finishReason, .length, "message_stop used to call this a finished reply")
        XCTAssertEqual(message.metadata["outputTokens"]?.intValue, 4096)
    }

    func testAnOrdinaryEndingStillReadsAsFinished() throws {
        var message = Message(role: .assistant)
        service.patchMessage(&message, with: try event(Self.start))
        service.patchMessage(&message, with: try event(Self.finished))
        service.patchMessage(&message, with: try event(Self.stop))
        XCTAssertEqual(message.finishReason, .stop)
        XCTAssertEqual(message.metadata["outputTokens"]?.intValue, 14)
    }
}

import Mistral

/// Mistral has its own provider rather than going through the OpenAI one
/// with a dialect, so none of the counting added there reached it: a reply
/// arrived with no tokens and no model against it and the signature line
/// showed nothing. Mistral had been reporting all of it the whole time.
final class MistralUsageRecordingTests: XCTestCase {

    private let service = MistralService(apiKey: "test")

    private func chunk(_ json: String) throws -> Mistral.ChatStreamResponse {
        try JSONDecoder().decode(Mistral.ChatStreamResponse.self, from: Data(json.utf8))
    }

    func testTheModelIsRecordedFromTheFirstChunk() throws {
        let json = #"""
        {"id":"x","object":"chat.completion.chunk","model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"Hel"}}]}
        """#
        var message = Message(role: .assistant)
        service.patchMessage(&message, with: try chunk(json))

        XCTAssertEqual(message.metadata["model"]?.stringValue, "mistral-large-latest")
        XCTAssertNil(message.metadata["outputTokens"], "nothing to report until usage arrives")
    }

    /// The case that made this worth testing rather than just writing. The
    /// totals ride on a chunk with an empty choices array, and `patchMessage`
    /// returns early when there is no choice — so anything read after that
    /// guard is never read at all. That is precisely how the OpenAI provider
    /// came to drop its counts, and this provider had the same shape.
    func testACountOnlyChunkWithNoChoicesIsStillRead() throws {
        let json = #"""
        {"id":"x","object":"chat.completion.chunk","model":"mistral-large-latest","choices":[],"usage":{"prompt_tokens":91,"completion_tokens":240,"total_tokens":331}}
        """#
        var message = Message(role: .assistant)
        service.patchMessage(&message, with: try chunk(json))

        XCTAssertEqual(message.metadata["inputTokens"]?.intValue, 91)
        XCTAssertEqual(message.metadata["outputTokens"]?.intValue, 240)
        XCTAssertEqual(message.metadata["model"]?.stringValue, "mistral-large-latest")
    }

    /// A response carrying no usage writes none. Zeros would put "0 tokens"
    /// on screen, which reads as a measurement rather than the absence of one.
    func testNoUsageWritesNoCounts() throws {
        let json = #"""
        {"id":"x","object":"chat.completion.chunk","model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"lo"}}]}
        """#
        var message = Message(role: .assistant)
        service.patchMessage(&message, with: try chunk(json))

        XCTAssertNil(message.metadata["inputTokens"])
        XCTAssertNil(message.metadata["outputTokens"])
    }
}
