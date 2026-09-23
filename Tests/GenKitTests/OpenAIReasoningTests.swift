import XCTest
import OpenAI
@testable import GenKit

/// The provider-neutral effort a caller sends, against what each model was
/// measured to accept. The rows in `OpenAIService.Reasoning` came from the
/// API's own refusals; these pin what's sent for each of them.
final class OpenAIReasoningTests: XCTestCase {

    /// The `reasoning_effort` that would be sent, or "thinking:disabled"
    /// where the endpoint is switched off instead, or nil for nothing.
    private func sent(_ effort: String, to modelID: String, dialect: OpenAIService.Dialect = .openAI) -> String? {
        let service = OpenAIService(apiKey: "test", dialect: dialect)
        let model = Model(id: modelID, owner: "test")
        switch service.reasoning(from: ["effort": .string(effort)], model: model) {
        case .nothing: return nil
        case .effort(let level): return level.rawValue
        case .thinkingOff: return "thinking:disabled"
        }
    }

    // gpt-5.6: none, low, medium, high, xhigh

    func testNewestOpenAIModelTakesEveryLevelButMax() {
        XCTAssertEqual(sent("off", to: "gpt-5.6-sol"), "none")
        XCTAssertEqual(sent("low", to: "gpt-5.6-sol"), "low")
        XCTAssertEqual(sent("medium", to: "gpt-5.6-sol"), "medium")
        XCTAssertEqual(sent("high", to: "gpt-5.6-sol"), "high")
        XCTAssertEqual(sent("xhigh", to: "gpt-5.6-sol"), "xhigh")
        // Not offered, so the nearest that is.
        XCTAssertEqual(sent("max", to: "gpt-5.6-sol"), "xhigh")
    }

    // gpt-5: minimal, low, medium, high

    func testFirstGPT5StopsAtHighAndCannotBeSwitchedOff() {
        XCTAssertEqual(sent("off", to: "gpt-5"), "minimal")
        XCTAssertEqual(sent("off", to: "gpt-5-mini"), "minimal")
        XCTAssertEqual(sent("xhigh", to: "gpt-5"), "high")
        XCTAssertEqual(sent("max", to: "gpt-5"), "high")
        XCTAssertEqual(sent("low", to: "gpt-5"), "low")
    }

    // gpt-4o: refuses the parameter

    func testModelsThatDoNotReasonAreSentNothing() {
        // gpt-5-search-api refused reasoning_effort outright — "Unrecognized
        // request argument" — whatever its generation number suggests.
        for id in ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-3.5-turbo", "chatgpt-4o-latest", "o1-mini",
                   "gpt-5-search-api", "gpt-4o-search-preview", "gpt-4o-mini-search-preview"] {
            XCTAssertNil(sent("high", to: id), id)
            XCTAssertNil(sent("off", to: id), id)
        }
    }

    func testOSeriesCannotBeSwitchedOffSoOffAsksForTheLeast() {
        XCTAssertEqual(sent("off", to: "o3"), "low")
        XCTAssertEqual(sent("off", to: "o4-mini"), "low")
        XCTAssertEqual(sent("high", to: "o3"), "high")
        XCTAssertEqual(sent("xhigh", to: "o3"), "high")
    }

    // DeepSeek: low, high, max, switched off with its own parameter; the
    // other names documented as aliases.

    func testDeepSeekHasThreeLevelsAndASwitch() {
        XCTAssertEqual(sent("off", to: "deepseek-v4-pro", dialect: .deepSeek), "thinking:disabled")
        XCTAssertEqual(sent("low", to: "deepseek-v4-pro", dialect: .deepSeek), "low")
        XCTAssertEqual(sent("high", to: "deepseek-v4-pro", dialect: .deepSeek), "high")
        XCTAssertEqual(sent("max", to: "deepseek-v4-pro", dialect: .deepSeek), "max")
        // Its own table: medium and xhigh are high.
        XCTAssertEqual(sent("medium", to: "deepseek-flash", dialect: .deepSeek), "high")
        XCTAssertEqual(sent("xhigh", to: "deepseek-flash", dialect: .deepSeek), "high")
        XCTAssertEqual(sent("brief", to: "deepseek-flash", dialect: .deepSeek), "low")
    }

    func testSteeredLevelsTranslate() {
        // Brief is the least worth grading; Full is the model's default,
        // which is what sending nothing asks for.
        XCTAssertEqual(sent("brief", to: "gpt-5.6-sol"), "low")
        XCTAssertNil(sent("full", to: "gpt-5.6-sol"))
    }

    func testUnmeasuredDialectsAreSentNothing() {
        XCTAssertNil(sent("high", to: "grok-4", dialect: .grok))
        XCTAssertNil(sent("high", to: "openai/gpt-oss-120b", dialect: .openRouter))
        // Groq's Llama and MiniMax rows are unmeasured too.
        XCTAssertNil(sent("high", to: "llama-3.3-70b-versatile", dialect: .groq))
        XCTAssertNil(sent("off", to: "minimaxai/minimax-m2.7", dialect: .groq))
    }

    // Groq, from its API reference. Ids carry the lab as a prefix.

    func testGroqGPTOSSHasThreeLevelsAndNoOff() {
        XCTAssertEqual(sent("low", to: "openai/gpt-oss-120b", dialect: .groq), "low")
        XCTAssertEqual(sent("medium", to: "openai/gpt-oss-20b", dialect: .groq), "medium")
        XCTAssertEqual(sent("high", to: "openai/gpt-oss-120b", dialect: .groq), "high")
        XCTAssertEqual(sent("xhigh", to: "openai/gpt-oss-120b", dialect: .groq), "high")
        // Can't be told none, so Off asks for the least.
        XCTAssertEqual(sent("off", to: "openai/gpt-oss-120b", dialect: .groq), "low")
        XCTAssertEqual(sent("brief", to: "openai/gpt-oss-120b", dialect: .groq), "low")
        XCTAssertNil(sent("full", to: "openai/gpt-oss-120b", dialect: .groq))
    }

    func testGroqQwen38TakesNoneAndThreeLevels() {
        XCTAssertEqual(sent("off", to: "qwen/qwen3.8-27b", dialect: .groq), "none")
        XCTAssertEqual(sent("low", to: "qwen/qwen3.8-27b", dialect: .groq), "low")
        XCTAssertEqual(sent("high", to: "qwen/qwen3.8-27b", dialect: .groq), "high")
        XCTAssertEqual(sent("max", to: "qwen/qwen3.8-27b", dialect: .groq), "high")
    }

    /// The older qwen3 models take `none` and nothing else — a switch, not
    /// a scale. Asking for some reasoning must not be rounded down to none.
    func testAGradedRequestIsNeverRoundedDownToNone() {
        XCTAssertEqual(sent("off", to: "qwen/qwen3-32b", dialect: .groq), "none")
        XCTAssertNil(sent("low", to: "qwen/qwen3-32b", dialect: .groq))
        XCTAssertNil(sent("high", to: "qwen/qwen3-32b", dialect: .groq))
        XCTAssertNil(sent("brief", to: "qwen/qwen3-32b", dialect: .groq))
    }

    func testACallerSpeakingTheAPIsOwnTermsIsTakenAtItsWord() {
        let service = OpenAIService(apiKey: "test")
        let model = Model(id: "gpt-4o", owner: "test")
        let setting = service.reasoning(from: ["reasoning_effort": .string("high")], model: model)
        XCTAssertEqual(setting, .effort(.high))
    }

    func testSearchModelsSearchByThemselves() {
        let openAI = Service(kind: .openAI, name: "OpenAI")
        for id in ["gpt-5-search-api", "gpt-4o-search-preview", "gpt-4o-mini-search-preview"] {
            XCTAssertTrue(openAI.hasBuiltInWebSearch(modelID: id), id)
        }
        for id in ["gpt-5", "gpt-4o", "text-search-ada-doc-001"] {
            XCTAssertFalse(openAI.hasBuiltInWebSearch(modelID: id), id)
        }
    }

    func testGenerationIsReadFromTheID() {
        XCTAssertEqual(OpenAIService.Reasoning.supported(byModel: "gpt-5.1-codex-mini", dialect: .openAI).levels?.first, "none")
        XCTAssertEqual(OpenAIService.Reasoning.supported(byModel: "gpt-5-nano", dialect: .openAI).levels?.first, "minimal")
        XCTAssertEqual(OpenAIService.Reasoning.supported(byModel: "gpt-6", dialect: .openAI).levels?.first, "none")
        XCTAssertNil(OpenAIService.Reasoning.supported(byModel: "gpt-4.1-mini", dialect: .openAI).levels)
        XCTAssertNil(OpenAIService.Reasoning.supported(byModel: "gpt-oss-120b", dialect: .openAI).levels)
    }

    func testCanDisableThinking() {
        XCTAssertEqual(OpenAIService.canDisableThinking(modelID: "gpt-5.6-sol", dialect: .openAI), true)
        XCTAssertEqual(OpenAIService.canDisableThinking(modelID: "gpt-5", dialect: .openAI), true)
        XCTAssertEqual(OpenAIService.canDisableThinking(modelID: "o3", dialect: .openAI), false)
        XCTAssertNil(OpenAIService.canDisableThinking(modelID: "gpt-4o", dialect: .openAI))
        XCTAssertEqual(OpenAIService.canDisableThinking(modelID: "deepseek-flash", dialect: .deepSeek), true)
        XCTAssertEqual(OpenAIService.canDisableThinking(modelID: "openai/gpt-oss-120b", dialect: .groq), false)
        XCTAssertEqual(OpenAIService.canDisableThinking(modelID: "qwen/qwen3.8-27b", dialect: .groq), true)
        XCTAssertNil(OpenAIService.canDisableThinking(modelID: "llama-3.3-70b-versatile", dialect: .groq))
    }
}

/// Reasoning arriving in the open, folded into the one shape every reader
/// already handles. The chunks are the shape DeepSeek streams: the role
/// with an empty `reasoning_content`, reasoning a few characters at a time
/// with `content` null, then the answer with `reasoning_content` gone.
final class OpenAIReasoningFoldTests: XCTestCase {

    private func chunk(reasoning: String?, content: String?, role: String? = nil, field: String = "reasoning_content") throws -> OpenAI.ChatStreamResponse {
        var delta: [String: Any] = [:]
        delta["content"] = content ?? NSNull()
        if let reasoning { delta[field] = reasoning }
        if let role { delta["role"] = role }
        let json: [String: Any] = [
            "id": "x", "object": "chat.completion.chunk", "created": 1, "model": "deepseek-flash",
            "choices": [["index": 0, "delta": delta, "finish_reason": NSNull()]],
        ]
        let data = try JSONSerialization.data(withJSONObject: json)
        return try JSONDecoder().decode(OpenAI.ChatStreamResponse.self, from: data)
    }

    private func text(of message: Message) -> String? {
        if case .text(let text) = message.contents?.last { return text }
        return nil
    }

    func testReasoningStreamsIntoAnOpenThinkBlockAndTheAnswerCloses() throws {
        let service = OpenAIService(apiKey: "test", dialect: .deepSeek)
        var message = Message(role: .assistant)

        service.patchMessage(&message, with: try chunk(reasoning: "", content: nil, role: "assistant"))
        XCTAssertEqual(text(of: message), "", "nothing to show until reasoning actually arrives")

        service.patchMessage(&message, with: try chunk(reasoning: "We", content: nil))
        service.patchMessage(&message, with: try chunk(reasoning: " need", content: nil))
        XCTAssertEqual(text(of: message), "<think>\nWe need", "left open while reasoning streams")

        service.patchMessage(&message, with: try chunk(reasoning: nil, content: "391"))
        XCTAssertEqual(text(of: message), "<think>\nWe need\n</think>\n391", "the answer closes it")

        service.patchMessage(&message, with: try chunk(reasoning: nil, content: "."))
        XCTAssertEqual(text(of: message), "<think>\nWe need\n</think>\n391.")
    }

    /// DeepSeek's guide: with tools in the request, every assistant message
    /// carries its reasoning back as `reasoning_content`, and the answer
    /// alone as content — not the `<think>` fold, which the model never wrote.
    /// Groq's gpt-oss models stream the same thing as `reasoning`, and so
    /// does OpenRouter. Folded identically; nothing about the fold is
    /// DeepSeek's but the field name.
    func testReasoningUnderGroqsNameFoldsTheSameWay() throws {
        let service = OpenAIService(apiKey: "test", dialect: .groq)
        var message = Message(role: .assistant)

        service.patchMessage(&message, with: try chunk(reasoning: "17 times", content: nil, role: "assistant", field: "reasoning"))
        service.patchMessage(&message, with: try chunk(reasoning: " 23", content: nil, field: "reasoning"))
        XCTAssertEqual(text(of: message), "<think>\n17 times 23", "left open while reasoning streams")

        service.patchMessage(&message, with: try chunk(reasoning: nil, content: "391"))
        XCTAssertEqual(text(of: message), "<think>\n17 times 23\n</think>\n391")
        XCTAssertEqual(message.metadata["thinking"]?.stringValue, "17 times 23")
        XCTAssertEqual(message.metadata["answer"]?.stringValue, "391")
    }

    /// Groq's Qwen parses `<think>…</think>` out of the model's raw output
    /// into `reasoning` and leaves the closing tag at the front of the
    /// content. Wrapped in a block of our own, it appeared on screen as a
    /// stray `</think>` above the answer.
    func testAClosingTagLeftAtTheFrontOfTheAnswerIsDropped() throws {
        let service = OpenAIService(apiKey: "test", dialect: .groq)
        var message = Message(role: .assistant)

        service.patchMessage(&message, with: try chunk(reasoning: "Let's search.", content: nil, field: "reasoning"))
        service.patchMessage(&message, with: try chunk(reasoning: nil, content: "</think>"))
        service.patchMessage(&message, with: try chunk(reasoning: nil, content: "\n\n<tool_call>"))

        // The block closes once, where we close it, and what the model wrote
        // after it is left exactly as written — a tool call it invented with
        // no tools offered is the model's business, not ours to tidy away.
        XCTAssertEqual(text(of: message), "<think>\nLet's search.\n</think>\n<tool_call>")
    }

    func testAnAnswerThatMerelyMentionsTheTagIsUntouched() throws {
        let service = OpenAIService(apiKey: "test", dialect: .groq)
        var message = Message(role: .assistant)

        service.patchMessage(&message, with: try chunk(reasoning: "hm", content: nil, field: "reasoning"))
        service.patchMessage(&message, with: try chunk(reasoning: nil, content: "Close it with </think> at the end."))
        XCTAssertEqual(text(of: message), "<think>\nhm\n</think>\nClose it with </think> at the end.")
    }

    func testReasoningIsHandedBackToDeepSeekApart() throws {
        let service = OpenAIService(apiKey: "test", dialect: .deepSeek)
        var message = Message(role: .assistant)
        service.patchMessage(&message, with: try chunk(reasoning: "We need", content: nil, role: "assistant"))
        service.patchMessage(&message, with: try chunk(reasoning: nil, content: "391"))
        XCTAssertEqual(text(of: message), "<think>\nWe need\n</think>\n391")

        let encoded = service.encodeForTest(messages: [message])
        XCTAssertEqual(encoded.first?.reasoning_content, "We need")
        XCTAssertEqual(encoded.first?.content?.first?.text, "391")
    }

    func testAToolCallRoundHandsBackAnEmptyAnswer() throws {
        let service = OpenAIService(apiKey: "test", dialect: .deepSeek)
        var message = Message(role: .assistant)
        service.patchMessage(&message, with: try chunk(reasoning: "Search first.", content: nil, role: "assistant"))
        let encoded = service.encodeForTest(messages: [message])
        XCTAssertEqual(encoded.first?.reasoning_content, "Search first.")
        XCTAssertEqual(encoded.first?.content?.first?.text, "")
    }

    func testNothingIsHandedBackToOpenAI() throws {
        let service = OpenAIService(apiKey: "test", dialect: .openAI)
        var message = Message(role: .assistant)
        service.patchMessage(&message, with: try chunk(reasoning: "We need", content: nil, role: "assistant"))
        service.patchMessage(&message, with: try chunk(reasoning: nil, content: "391"))
        let encoded = service.encodeForTest(messages: [message])
        XCTAssertNil(encoded.first?.reasoning_content)
    }

    func testAReplyWithoutReasoningIsUntouched() throws {
        let service = OpenAIService(apiKey: "test")
        var message = Message(role: .assistant)
        service.patchMessage(&message, with: try chunk(reasoning: nil, content: "Hi", role: "assistant"))
        service.patchMessage(&message, with: try chunk(reasoning: nil, content: " there"))
        XCTAssertEqual(text(of: message), "Hi there")
        XCTAssertNil(message.metadata["thinking"])
    }
}
