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

    /// Mistral publishes no output limit, and this used to report the context
    /// window as one — so Longest Reply offered a 262,144-token ceiling.
    func testNoOutputCeilingIsClaimed() throws {
        let decoded = try models(Self.codestral)
        XCTAssertEqual(decoded.first?.contextWindow, 256_000)
        XCTAssertNil(decoded.first?.maxOutput, "not published, so not claimed")
    }
}
