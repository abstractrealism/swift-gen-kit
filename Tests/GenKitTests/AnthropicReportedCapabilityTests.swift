import XCTest
import Anthropic
@testable import GenKit

/// Anthropic's model list says, per model, which thinking mode and which
/// effort levels it takes — the two things the hand-written table exists to
/// remember, and the two the API refuses rather than ignores.
///
/// Entries verbatim from `GET /v1/models`, 23 Sept 2026.
final class AnthropicReportedCapabilityTests: XCTestCase {

    private func listed(_ json: String) throws -> Anthropic.Model {
        try JSONDecoder().decode(Anthropic.Model.self, from: Data(json.utf8))
    }

    /// Adaptive only, every level, 128k out.
    private static let opus55 = #"""
    {"type":"model","id":"claude-opus-5-5","display_name":"Claude Opus 5.5","created_at":"2026-09-21T16:24:00Z","max_input_tokens":1000000,"max_tokens":128000,"capabilities":{"effort":{"supported":true,"low":{"supported":true},"medium":{"supported":true},"high":{"supported":true},"xhigh":{"supported":true},"max":{"supported":true}},"image_input":{"supported":true},"thinking":{"supported":true,"types":{"enabled":{"supported":false},"adaptive":{"supported":true}}}}}
    """#

    /// Sonnet 4.6: no xhigh. The table said only that it took both modes.
    private static let sonnet46 = #"""
    {"type":"model","id":"claude-sonnet-4-6","display_name":"Claude Sonnet 4.6","created_at":"2026-02-01T00:00:00Z","max_input_tokens":200000,"max_tokens":128000,"capabilities":{"effort":{"supported":true,"low":{"supported":true},"medium":{"supported":true},"high":{"supported":true},"max":{"supported":true}},"image_input":{"supported":true},"thinking":{"supported":true,"types":{"enabled":{"supported":true},"adaptive":{"supported":true}}}}}
    """#

    /// A 4.5: a budget and no levels at all.
    private static let haiku45 = #"""
    {"type":"model","id":"claude-haiku-4-5-20251001","display_name":"Claude Haiku 4.5","created_at":"2025-10-01T00:00:00Z","max_input_tokens":200000,"max_tokens":64000,"capabilities":{"effort":{"supported":false},"image_input":{"supported":true},"thinking":{"supported":true,"types":{"enabled":{"supported":true},"adaptive":{"supported":false}}}}}
    """#

    private func reasoning(_ json: String) throws -> GenKit.Model.Reasoning {
        try XCTUnwrap(AnthropicService.reasoning(of: try listed(json)))
    }

    func testTheNewestReportsAdaptiveAndEveryLevel() throws {
        let reported = try reasoning(Self.opus55)
        XCTAssertEqual(reported.modes, ["adaptive"])
        XCTAssertEqual(reported.levels, ["low", "medium", "high", "xhigh", "max"])
    }

    /// The table never knew this: it recorded that Sonnet 4.6 takes both
    /// modes and said nothing about which levels, so `xhigh` was sent to a
    /// model that doesn't take it.
    func testAModelMissingALevelSaysWhichOnesItHas() throws {
        let reported = try reasoning(Self.sonnet46)
        XCTAssertEqual(reported.levels, ["low", "medium", "high", "max"])
        XCTAssertEqual(reported.modes, ["adaptive", "enabled"])
    }

    func testAModelWithNoLevelsReportsTheModeAlone() throws {
        let reported = try reasoning(Self.haiku45)
        XCTAssertNil(reported.levels, "the capability is reported with nothing under it")
        XCTAssertEqual(reported.modes, ["enabled"])
    }

    func testWhatIsReportedBeatsWhatIsRemembered() throws {
        let model = GenKit.Model(
            id: "claude-haiku-4-5-20251001",
            owner: "anthropic",
            reasoning: try reasoning(Self.haiku45)
        )
        let supported = AnthropicService.Thinking.supported(by: model)
        XCTAssertFalse(supported.adaptive)
        XCTAssertTrue(supported.extended)
        XCTAssertFalse(supported.effort, "no levels, so no level is sent")
    }

    /// A model the table never heard of — and which it answers for anyway,
    /// by accident: it matches on `id.contains`, so `claude-opus-5-5` is
    /// caught by the `claude-opus-5` row. Right here by luck. Reported data
    /// doesn't depend on a model's name resembling an older one's.
    func testAModelTheTableOnlyMatchesByAccidentIsAnsweredProperly() throws {
        XCTAssertNotNil(
            AnthropicService.Thinking.known(byModel: "claude-opus-5-5"),
            "matched by the claude-opus-5 row, there being no row of its own"
        )

        let model = GenKit.Model(id: "claude-opus-5-5", owner: "anthropic", reasoning: try reasoning(Self.opus55))
        let supported = AnthropicService.Thinking.supported(by: model)
        XCTAssertTrue(supported.adaptive)
        XCTAssertFalse(supported.extended, "reported false, and it is a 400 if sent")
        XCTAssertTrue(supported.effort)
    }

    /// The table is what answers for a model stored before any of this
    /// existed — reloading a service's models is what fills it in.
    func testAModelThatReportsNothingFallsBackToTheTable() {
        let model = GenKit.Model(id: "claude-haiku-4-5-20251001", owner: "anthropic")
        let supported = AnthropicService.Thinking.supported(by: model)
        XCTAssertTrue(supported.extended)
        XCTAssertFalse(supported.effort)
    }

    /// Not reported by anything in the response, so it still comes from the
    /// table, and is assumed true where the table doesn't know.
    func testWhetherReasoningCanBeRefusedStillComesFromTheTable() throws {
        let fable = GenKit.Model(id: "claude-fable-5-1", owner: "anthropic", reasoning: try reasoning(Self.opus55))
        XCTAssertFalse(AnthropicService.Thinking.supported(by: fable).canDisable)

        let opus = GenKit.Model(id: "claude-opus-5-5", owner: "anthropic", reasoning: try reasoning(Self.opus55))
        XCTAssertTrue(AnthropicService.Thinking.supported(by: opus).canDisable)
    }

    func testTheLimitsAndTheRestAreCarriedOnto() throws {
        let capabilities = try XCTUnwrap(AnthropicService.capabilities(of: listed(Self.opus55)))
        XCTAssertTrue(capabilities.contains(.completion))
        XCTAssertTrue(capabilities.contains(.tools))
        XCTAssertTrue(capabilities.contains(.thinking))
        XCTAssertTrue(capabilities.contains(.vision))
    }

    func testAListThatSaysNothingAnswersNothing() throws {
        let bare = #"{"type":"model","id":"claude-opus-5","display_name":"Claude Opus 5","created_at":"2026-01-01T00:00:00Z"}"#
        XCTAssertNil(AnthropicService.capabilities(of: try listed(bare)))
        XCTAssertNil(AnthropicService.reasoning(of: try listed(bare)))
    }
}

/// Asking a model for a level it doesn't have.
///
/// The levels differ per model and a level a model lacks is refused rather
/// than rounded, so the nearest it does have is the difference between a
/// slightly shallower answer and no answer.
final class AnthropicNearestLevelTests: XCTestCase {

    private func request(_ reasoning: GenKit.Model.Reasoning?, asking level: String) async -> Anthropic.ChatRequest {
        let service = AnthropicService(apiKey: "test")
        let model = GenKit.Model(id: "claude-sonnet-4-6", owner: "anthropic", reasoning: reasoning)
        return await service.makeRequestForTesting(model: model, options: ["effort": .string(level)])
    }

    private static let sonnet46 = GenKit.Model.Reasoning(
        levels: ["low", "medium", "high", "max"],
        modes: ["adaptive", "enabled"]
    )

    func testALevelTheModelHasIsSentAsAsked() async {
        let sent = await request(Self.sonnet46, asking: "high")
        XCTAssertEqual(sent.output_config?.effort, "high")
    }

    /// Sonnet 4.6 takes low, medium, high and max — and not xhigh, which the
    /// table had no way to know and so sent anyway.
    func testALevelTheModelLacksBecomesTheNearestItHas() async {
        let sent = await request(Self.sonnet46, asking: "xhigh")
        XCTAssertEqual(sent.output_config?.effort, "high", "ties go to the lower level")
    }

    /// A model that reports levels and has none takes no level at all —
    /// which is the 4.5 generation, reasoning on a budget instead.
    func testAModelWithNoLevelsIsSentNone() async {
        let sent = await request(.init(levels: nil, modes: ["enabled"]), asking: "xhigh")
        XCTAssertNil(sent.output_config?.effort)
        XCTAssertEqual(sent.thinking?.type, "enabled", "a budget instead")
    }

    /// A model stored before any of this was read. Nothing is substituted,
    /// because nothing is known to substitute against — exactly as before.
    func testAModelThatReportsNothingIsSentWhatWasAsked() async {
        let sent = await request(nil, asking: "xhigh")
        XCTAssertEqual(sent.output_config?.effort, "xhigh")
    }
}
