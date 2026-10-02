import XCTest
@testable import GenKit

/// Reading a service's own refusal for the levels it named.
///
/// The error bodies here are verbatim. The Mistral one is from a live refusal
/// of 2 October 2026; the OpenAI shape is the one its provider's table was
/// built from by hand.
final class ReasoningProbeTests: XCTestCase {

    /// A Python `repr` leaking through an API. Nothing in the parser depends
    /// on its structure, because that structure can change without notice.
    func testMistralRefusalNamesItsLevels() {
        let body = #"""
        {"object":"error","message":"reasoning_effort low is not supported for this model, supported values: [<ReasoningEffort.high: 'high'>, <ReasoningEffort.none: 'none'>]","type":"invalid_request_invalid_args","param":null,"code":"3051","raw_status_code":400}
        """#
        XCTAssertEqual(ReasoningProbe.levels(in: body), ["none", "high"], "least to most, not the order named")
    }

    func testOpenAIRefusalNamesItsLevels() {
        let body = #"""
        {"error":{"message":"Invalid value: '__genkit_probe__'. Supported values are: 'minimal', 'low', 'medium', and 'high'.","type":"invalid_request_error","param":"reasoning_effort","code":"invalid_value"}}
        """#
        XCTAssertEqual(ReasoningProbe.levels(in: body), ["minimal", "low", "medium", "high"])
    }

    /// The value offered must not come back as a level. It can't match the
    /// filter — underscores aren't letters — which is why it is spelt that
    /// way rather than excluded by name.
    func testTheOfferedValueIsNeverMistakenForALevel() {
        let body = "Invalid value: '\(ReasoningProbe.offered)'. Supported values are: 'low', 'high'."
        XCTAssertEqual(ReasoningProbe.levels(in: body), ["low", "high"])
        XCTAssertFalse(ReasoningProbe.levels(in: body).contains(ReasoningProbe.offered))
    }

    /// An apostrophe in the prose shifts which fragments sit "inside" quotes.
    /// The fragments that result have spaces and punctuation in them, so the
    /// filter drops them rather than reading them as answers.
    func testAnApostropheInTheProseDoesNotProduceNonsense() {
        let body = "This model doesn't take that. Supported values are: 'low', 'high'."
        let found = ReasoningProbe.levels(in: body)
        XCTAssertTrue(found.allSatisfy { $0.allSatisfy(\.isLetter) }, "no fragments of prose: \(found)")
        XCTAssertTrue(found.contains("high"), "and the real answer still survives it")
    }

    /// A level nobody here has heard of still comes back — it sorts last
    /// rather than being dropped, because a list of known levels is the thing
    /// this exists to stop maintaining.
    func testAnUnknownLevelSurvivesAndSortsLast() {
        let body = "Supported values are: 'high', 'ludicrous', 'low'."
        XCTAssertEqual(ReasoningProbe.levels(in: body), ["low", "high", "ludicrous"])
    }

    /// A refusal about something else entirely teaches nothing, and must not
    /// be read as teaching something.
    func testARefusalNamingNoLevelsYieldsNothing() {
        let body = #"{"error":{"message":"You exceeded your current quota.","type":"insufficient_quota"}}"#
        XCTAssertEqual(ReasoningProbe.levels(in: body), [])
    }

    /// Knowable before asking, so a caller can skip a service entirely
    /// rather than discovering it from a `.failed` it can't tell apart from
    /// a service it couldn't reach.
    func testWhetherThereIsAnythingToAskIsKnownUpFront() {
        for kind in [Service.Kind.openAI, .deepseek, .grok, .groq, .mistral, .openRouter] {
            let service = Service(kind: kind, name: "\(kind)", host: "https://example.invalid/v1", token: "t")
            XCTAssertTrue(service.canProbeReasoningLevels, "\(kind) takes reasoning_effort")
        }
        for kind in [Service.Kind.anthropic, .ollama, .elevenLabs, .fal] {
            let service = Service(kind: kind, name: "\(kind)", host: "https://example.invalid/v1", token: "t")
            XCTAssertFalse(service.canProbeReasoningLevels, "\(kind) has no such parameter")
        }
    }

    func testOnlyTheServicesWithSuchAParameterAreAsked() async {
        let anthropic = Service(id: "a", kind: .anthropic, name: "Anthropic", host: "https://api.anthropic.com/v1", token: "t")
        let ollama = Service(id: "o", kind: .ollama, name: "Ollama", host: "http://localhost:11434/v1", token: "t")
        let forAnthropic = await anthropic.probeReasoningLevels(modelID: "claude-opus-5")
        let forOllama = await ollama.probeReasoningLevels(modelID: "qwen3")
        XCTAssertEqual(forAnthropic, .failed)
        XCTAssertEqual(forOllama, .failed)
    }

    /// Nothing to ask with, so nothing is asked — and in particular no
    /// unauthenticated request goes out to be refused for the wrong reason.
    func testAServiceWithNoKeyIsNotAsked() async {
        let service = Service(id: "m", kind: .mistral, name: "Mistral", host: "https://api.mistral.ai/v1", token: "")
        let outcome = await service.probeReasoningLevels(modelID: "mistral-medium-latest")
        XCTAssertEqual(outcome, .failed)
    }
}
