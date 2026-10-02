import XCTest
import Mistral
@testable import GenKit

/// Mistral lists one entry per alias, so Settings showed 53 models under 20
/// labels — four identical `codestral-2508` rows, nine identical
/// `mistral-medium-latest` rows — and seven of the twenty were not chat
/// models at all.
///
/// The fixtures are trimmed from a live `/v1/models` response of 1 Oct 2026.
final class MistralModelsTests: XCTestCase {

    private let service = MistralService(apiKey: "test")

    private func models(_ json: String) throws -> [GenKit.Model] {
        let response = try JSONDecoder().decode(Mistral.ModelsResponse.self, from: Data(json.utf8))
        return service.decodeModels(response)
    }

    /// Four ids, one model. Everything but `id` was identical across them.
    private static let codestral = #"""
    {"object":"list","data":[
      {"id":"codestral-2508","name":"codestral-2508","max_context_length":256000,"capabilities":{"completion_chat":true,"completion_fim":true,"function_calling":true,"fine_tuning":false,"vision":false}},
      {"id":"codestral-latest","name":"codestral-2508","max_context_length":256000,"capabilities":{"completion_chat":true,"completion_fim":true,"function_calling":true,"fine_tuning":false,"vision":false}},
      {"id":"mistral-code-latest","name":"codestral-2508","max_context_length":256000,"capabilities":{"completion_chat":true,"completion_fim":true,"function_calling":true,"fine_tuning":false,"vision":false}},
      {"id":"mistral-code-fim-latest","name":"codestral-2508","max_context_length":256000,"capabilities":{"completion_chat":true,"completion_fim":true,"function_calling":true,"fine_tuning":false,"vision":false}}
    ]}
    """#

    func testAliasesCollapseToTheEntryNamedAfterItself() throws {
        let decoded = try models(Self.codestral)
        XCTAssertEqual(decoded.count, 1, "four ids, one model")
        XCTAssertEqual(decoded.first?.id, "codestral-2508", "the one whose id matches its name")
    }

    /// The canonical entry is not always first, so the rule has to be the
    /// name and not the order.
    func testTheCanonicalEntryIsFoundWhereverItSits() throws {
        let json = #"""
        {"object":"list","data":[
          {"id":"mistral-large-latest","name":"mistral-large-2512","max_context_length":262144,"capabilities":{"completion_chat":true,"completion_fim":false,"function_calling":true,"fine_tuning":true,"vision":true}},
          {"id":"mistral-large-2512","name":"mistral-large-2512","max_context_length":262144,"capabilities":{"completion_chat":true,"completion_fim":false,"function_calling":true,"fine_tuning":true,"vision":true}}
        ]}
        """#
        let decoded = try models(json)
        XCTAssertEqual(decoded.map(\.id), ["mistral-large-2512"])
    }

    /// Where Mistral names nothing after itself, keep one rather than all.
    func testAGroupWithNoCanonicalEntryKeepsTheFirst() throws {
        let json = #"""
        {"object":"list","data":[
          {"id":"something-a","name":"shared","max_context_length":1000,"capabilities":{"completion_chat":true,"completion_fim":false,"function_calling":false,"fine_tuning":false,"vision":false}},
          {"id":"something-b","name":"shared","max_context_length":1000,"capabilities":{"completion_chat":true,"completion_fim":false,"function_calling":false,"fine_tuning":false,"vision":false}}
        ]}
        """#
        XCTAssertEqual(try models(json).map(\.id), ["something-a"])
    }

    /// The embedding, OCR, transcription and speech models say they do not
    /// chat, and were being offered as chat models.
    func testWhatIsNotAChatModelSaysSo() throws {
        let json = #"""
        {"object":"list","data":[
          {"id":"mistral-embed","name":"mistral-embed","max_context_length":8192,"capabilities":{"completion_chat":false,"completion_fim":false,"function_calling":false,"fine_tuning":false,"vision":false}},
          {"id":"mistral-ocr-latest","name":"mistral-ocr-latest","max_context_length":16384,"capabilities":{"completion_chat":false,"completion_fim":false,"function_calling":true,"fine_tuning":false,"vision":true}},
          {"id":"voxtral-small-2507","name":"voxtral-small-2507","max_context_length":32768,"capabilities":{"completion_chat":true,"completion_fim":false,"function_calling":true,"fine_tuning":false,"vision":false}}
        ]}
        """#
        let decoded = try models(json)
        let chatting = decoded.filter { $0.capabilities?.contains(.completion) == true }.map(\.id)
        XCTAssertEqual(chatting, ["voxtral-small-2507"], "voxtral-small chats; embed and OCR do not")
        XCTAssertEqual(
            decoded.first(where: { $0.id == "mistral-ocr-latest" })?.capabilities,
            [.tools, .vision],
            "read as sent, rather than guessed from the name"
        )
    }

    /// Magistral became an alias of mistral-medium rather than a model of its
    /// own, so the model that *sounds* like the reasoning one and the model
    /// that *is* it are the same thing under two names. Guessing from the
    /// name is therefore worse than useless, and this is why `reasoning` had
    /// to be decoded upstream before it could be read here.
    func testReasoningIsReadWhereTheServiceReportsIt() throws {
        let json = #"""
        {"object":"list","data":[
          {"id":"mistral-medium-latest","name":"mistral-medium-latest","max_context_length":262144,"capabilities":{"completion_chat":true,"completion_fim":false,"function_calling":true,"fine_tuning":false,"vision":true,"reasoning":true}},
          {"id":"ministral-3b-2512","name":"ministral-3b-2512","max_context_length":131072,"capabilities":{"completion_chat":true,"completion_fim":false,"function_calling":true,"fine_tuning":true,"vision":true,"reasoning":false}}
        ]}
        """#
        let decoded = try models(json)
        XCTAssertEqual(decoded.first(where: { $0.id == "mistral-medium-latest" })?.capabilities?.contains(.thinking), true)
        XCTAssertEqual(decoded.first(where: { $0.id == "ministral-3b-2512" })?.capabilities?.contains(.thinking), false)
    }

    /// The whole reason the new keys are optional upstream. A response with
    /// none of them must still decode — otherwise one model that happens not
    /// to report `reasoning` takes every other model down with it, and the
    /// service list comes back empty rather than merely incomplete.
    func testAResponseWithoutTheNewerKeysStillDecodes() throws {
        let json = #"""
        {"object":"list","data":[
          {"id":"mistral-small-2603","name":"mistral-small-2603","max_context_length":262144,"capabilities":{"completion_chat":true,"completion_fim":false,"function_calling":true,"fine_tuning":false,"vision":true}}
        ]}
        """#
        let decoded = try models(json)
        XCTAssertEqual(decoded.count, 1)
        XCTAssertEqual(decoded.first?.capabilities, [.completion, .tools, .vision], "no thinking claimed where none is reported")
    }

    /// Mistral publishes no output limit, and this used to report the context
    /// window as one — so Longest Reply offered a 262,144-token ceiling.
    func testNoOutputCeilingIsClaimed() throws {
        let decoded = try models(Self.codestral)
        XCTAssertEqual(decoded.first?.contextWindow, 256_000)
        XCTAssertNil(decoded.first?.maxOutput, "not published, so not claimed")
    }
}

/// The reasoning models send content as typed blocks where the rest send a
/// string — and `zai-glm-5-3` sends both in one stream, two chunks of
/// `"content": ""` before the first block arrives. Declared `String?`, that
/// failed the decode on the third chunk and took the whole generation with
/// it, reporting only to the screen.
///
/// Chunks verbatim from a live stream of 2 October 2026.
final class MistralBlockContentTests: XCTestCase {

    private let service = MistralService(apiKey: "test")

    private func chunk(_ json: String) throws -> Mistral.ChatStreamResponse {
        try JSONDecoder().decode(Mistral.ChatStreamResponse.self, from: Data(json.utf8))
    }

    private static let plain = #"""
    {"id":"x","object":"chat.completion.chunk","created":1,"model":"zai-glm-5-3","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}
    """#

    private static let firstThought = #"""
    {"id":"x","object":"chat.completion.chunk","created":1,"model":"zai-glm-5-3","choices":[{"index":0,"delta":{"index":0,"content":[{"type":"thinking","thinking":[{"type":"text","text":"The"}],"closed":true}]},"finish_reason":null,"logprobs":null}]}
    """#

    private static let moreThought = #"""
    {"id":"x","object":"chat.completion.chunk","created":1,"model":"zai-glm-5-3","choices":[{"index":0,"delta":{"index":0,"content":[{"type":"thinking","thinking":[{"type":"text","text":" user is asking for a"}],"closed":true}]},"finish_reason":null,"logprobs":null}]}
    """#

    private static let answer = #"""
    {"id":"x","object":"chat.completion.chunk","created":1,"model":"zai-glm-5-3","choices":[{"index":0,"delta":{"index":0,"content":[{"type":"text","text":"391"}]},"finish_reason":null,"logprobs":null}]}
    """#

    /// The decode itself, which is what actually broke.
    func testAStreamMixingStringAndBlockContentDecodes() throws {
        XCTAssertNoThrow(try chunk(Self.plain))
        XCTAssertNoThrow(try chunk(Self.firstThought))
    }

    /// Reasoning accumulates under its own key and the content is rebuilt
    /// from both halves, as it is for Ollama and OpenAI, so a reader needn't
    /// know which provider answered.
    func testReasoningAccumulatesAndTheAnswerFollowsIt() throws {
        var message = Message(role: .assistant)
        service.patchMessage(&message, with: try chunk(Self.plain))
        service.patchMessage(&message, with: try chunk(Self.firstThought))
        service.patchMessage(&message, with: try chunk(Self.moreThought))

        XCTAssertEqual(message.metadata["thinking"]?.stringValue, "The user is asking for a")
        XCTAssertEqual(message.content, "<think>\nThe user is asking for a", "left open until the answer starts")

        service.patchMessage(&message, with: try chunk(Self.answer))
        XCTAssertEqual(message.metadata["answer"]?.stringValue, "391")
        XCTAssertEqual(message.content, "<think>\nThe user is asking for a\n</think>\n391")
    }

    /// A model that sends a plain string is unaffected — no `<think>` wrapper
    /// appears around an answer that never had reasoning beside it.
    func testPlainStringContentIsUntouched() throws {
        let hello = #"""
        {"id":"x","object":"chat.completion.chunk","created":1,"model":"mistral-medium-latest","choices":[{"index":0,"delta":{"content":"17 × 2"},"finish_reason":null}]}
        """#
        var message = Message(role: .assistant)
        service.patchMessage(&message, with: try chunk(hello))
        XCTAssertEqual(message.content, "17 × 2")
        XCTAssertNil(message.metadata["thinking"])
    }
}
