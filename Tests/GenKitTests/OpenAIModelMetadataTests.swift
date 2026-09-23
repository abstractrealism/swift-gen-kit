import XCTest
import OpenAI
@testable import GenKit

/// What a service's own model list says about its models, where it says
/// anything.
///
/// OpenAI's reports an id and an owner, so an app has to guess from the name
/// which of ~130 models converse. Groq's reports modality, features and
/// limits on every entry — undocumented, and worth more than any guess:
/// asking Whisper to chat answers "does not support chat completions", and
/// the list said so all along.
///
/// Entries verbatim from `GET /openai/v1/models`, 21 Sept 2026.
final class OpenAIModelMetadataTests: XCTestCase {

    private func model(_ json: String) throws -> OpenAI.Model {
        try JSONDecoder().decode(OpenAI.Model.self, from: Data(json.utf8))
    }

    private static let whisper = #"""
    {"id":"whisper-large-v3-turbo","object":"model","created":1728413088,"owned_by":"OpenAI","active":true,"context_window":448,"max_completion_tokens":448,"name":"Whisper Large V3 Turbo","input_modalities":["audio"],"output_modalities":["transcription"],"context_length":448,"max_output_length":448}
    """#

    private static let qwen = #"""
    {"id":"qwen/qwen3.8-27b","object":"model","created":1786984846,"owned_by":"Alibaba Cloud","active":true,"context_window":131042,"max_completion_tokens":16384,"name":"Qwen/Qwen3.8-27B","input_modalities":["text","image"],"output_modalities":["text"],"context_length":131042,"max_output_length":16384,"supported_sampling_parameters":["temperature","top_p","stop","seed","max_tokens"],"supported_features":["tools","json_mode","reasoning"]}
    """#

    /// A speech model: text in, audio out. It fails on the output alone,
    /// which is why both ends have to be checked.
    private static let orpheus = #"""
    {"id":"canopylabs/orpheus-v1-english","object":"model","created":1,"owned_by":"Canopy Labs","active":true,"name":"Orpheus English","input_modalities":["text"],"output_modalities":["audio"]}
    """#

    func testATranscriptionModelIsNotAChatModel() throws {
        let capabilities = try XCTUnwrap(OpenAIService.capabilities(of: model(Self.whisper)))
        XCTAssertFalse(capabilities.contains(.completion))
        XCTAssertFalse(capabilities.contains(.tools))
    }

    func testASpeechModelIsNotAChatModelEither() throws {
        let capabilities = try XCTUnwrap(OpenAIService.capabilities(of: model(Self.orpheus)))
        XCTAssertFalse(capabilities.contains(.completion), "text in, but audio out")
    }

    func testAChatModelReportsWhatItCanDo() throws {
        let capabilities = try XCTUnwrap(OpenAIService.capabilities(of: model(Self.qwen)))
        XCTAssertTrue(capabilities.contains(.completion))
        XCTAssertTrue(capabilities.contains(.tools))
        XCTAssertTrue(capabilities.contains(.thinking))
        XCTAssertTrue(capabilities.contains(.vision), "it takes images as well as text")
    }

    /// Unknown, not incapable. Every endpoint but Groq reports none of this,
    /// and narrowing a request on a silence would strip tools from models
    /// that handle them perfectly well.
    func testAListThatSaysNothingAnswersNothing() throws {
        let plain = #"{"id":"gpt-5.6-sol","object":"model","created":1,"owned_by":"system"}"#
        XCTAssertNil(OpenAIService.capabilities(of: try model(plain)))
    }

    func testTheNameAndLimitsAreCarriedOntoTheModel() async throws {
        let service = OpenAIService(apiKey: "test", dialect: .groq)
        let decoded = await service.decodeForTesting(try model(Self.qwen))
        XCTAssertEqual(decoded.name, "Qwen/Qwen3.8-27B")
        XCTAssertEqual(decoded.contextWindow, 131042)
        XCTAssertEqual(decoded.maxOutput, 16384)
    }

    func testAModelWithoutANameKeepsItsID() async throws {
        let plain = #"{"id":"gpt-5.6-sol","object":"model","created":1,"owned_by":"system"}"#
        let service = OpenAIService(apiKey: "test")
        let decoded = await service.decodeForTesting(try model(plain))
        XCTAssertEqual(decoded.name, "gpt-5.6-sol")
        XCTAssertNil(decoded.capabilities)
    }
}

/// Sending only what a model can take, where the service said what that is.
final class OpenAICapabilityAwareRequestTests: XCTestCase {

    private let service = OpenAIService(apiKey: "test", dialect: .groq)

    private static let lookup = Tool(
        function: .init(
            name: "lookup",
            description: "look something up",
            parameters: .object(properties: ["q": .string(description: "what to look up")], required: ["q"])
        )
    )

    private func request(_ capabilities: Set<GenKit.Model.Capability>?) -> ChatServiceRequest {
        var request = ChatServiceRequest(
            // A real id, because which *levels* a model takes is still a
            // table keyed by id — the list says whether a model reasons, not
            // how it wants to be asked.
            model: GenKit.Model(id: "openai/gpt-oss-120b", owner: "groq", capabilities: capabilities),
            messages: [Message(role: .user, content: "hi")]
        )
        request.tools = [Self.lookup]
        request.options["effort"] = .string("high")
        return request
    }

    func testAModelWithoutToolsIsSentNone() async {
        let sent = await service.makeRequest(request([.completion]), streaming: true)
        XCTAssertNil(sent.tools)
        XCTAssertNil(sent.tool_choice)
    }

    func testAModelThatDoesNotReasonIsSentNoLevel() async {
        let sent = await service.makeRequest(request([.completion, .tools]), streaming: true)
        XCTAssertNil(sent.reasoning_effort)
        XCTAssertNotNil(sent.tools)
    }

    func testAModelThatCanDoBothGetsBoth() async {
        let sent = await service.makeRequest(request([.completion, .tools, .thinking]), streaming: true)
        XCTAssertEqual(sent.tools?.count, 1)
        XCTAssertEqual(sent.reasoning_effort?.rawValue, "high")
    }

    func testTheReplyCeilingIsCappedAtWhatTheModelAllows() async {
        var request = request([.completion])
        request.model = GenKit.Model(id: "qwen/qwen3.8-27b", owner: "groq", maxOutput: 16384)
        request.options["max_tokens"] = .int(65536)

        let sent = await service.makeRequest(request, streaming: true)
        XCTAssertEqual(sent.max_completion_tokens, 16384, "one service-wide figure, models that don't agree")
    }

    func testAFigureWithinTheModelsLimitIsSentAsChosen() async {
        var request = request([.completion])
        request.model = GenKit.Model(id: "qwen/qwen3.8-27b", owner: "groq", maxOutput: 16384)
        request.options["max_tokens"] = .int(8192)

        let sent = await service.makeRequest(request, streaming: true)
        XCTAssertEqual(sent.max_completion_tokens, 8192)
    }

    func testAModelThatReportsNoLimitIsSentTheFigureAsChosen() async {
        var request = request([.completion])
        request.options["max_tokens"] = .int(65536)

        let sent = await service.makeRequest(request, streaming: true)
        XCTAssertEqual(sent.max_completion_tokens, 65536)
    }

    /// The services that report nothing, which is all of them but Groq.
    func testAModelReportingNothingIsSentEverythingAsked() async {
        let sent = await service.makeRequest(request(nil), streaming: true)
        XCTAssertEqual(sent.tools?.count, 1)
        XCTAssertEqual(sent.reasoning_effort?.rawValue, "high")
    }
}

/// DeepSeek reports its reasoning levels and its reply ceiling under its own
/// names. Entry verbatim from `GET /v1/models`, 23 Sept 2026.
final class DeepSeekModelMetadataTests: XCTestCase {

    private static let flash = #"""
    {"id":"deepseek-flash","object":"model","owned_by":"deepseek","name":"DeepSeek-V4.1-Flash","context_window":1048576,"max_output_tokens":393216,"input_modalities":["text","image"],"output_modalities":["text"],"effort":{"supported_levels":["low","high","max"],"default_level":"high"}}
    """#

    private func model(_ json: String) throws -> OpenAI.Model {
        try JSONDecoder().decode(OpenAI.Model.self, from: Data(json.utf8))
    }

    func testTheLevelsAndTheCeilingAreCarriedOnto() async throws {
        let service = OpenAIService(apiKey: "test", dialect: .deepSeek)
        let decoded = await service.decodeForTesting(try model(Self.flash))

        XCTAssertEqual(decoded.name, "DeepSeek-V4.1-Flash")
        XCTAssertEqual(decoded.contextWindow, 1_048_576)
        XCTAssertEqual(decoded.maxOutput, 393_216, "reported as max_output_tokens, not max_completion_tokens")
        XCTAssertEqual(decoded.reasoning?.levels, ["low", "high", "max"])
        XCTAssertEqual(decoded.reasoning?.defaultLevel, "high")
        XCTAssertEqual(decoded.capabilities?.contains(.vision), true)
        // Named by the presence of the levels and nowhere else. Missing it
        // stripped the setting from the request, and a reasoning model then
        // didn't reason.
        XCTAssertEqual(decoded.capabilities?.contains(.thinking), true)
    }

    /// The table recorded low/high/max from DeepSeek's Thinking Mode guide.
    /// The service now says so itself, and what it says wins.
    func testWhatIsReportedBeatsWhatIsRemembered() async throws {
        let service = OpenAIService(apiKey: "test", dialect: .deepSeek)
        let decoded = await service.decodeForTesting(try model(Self.flash))

        let supported = OpenAIService.Reasoning.supported(by: decoded, dialect: .deepSeek)
        XCTAssertEqual(supported.levels, ["low", "high", "max"])
        // The parts the list doesn't report stay with the table: that medium
        // is another word for high, and that thinking is switched off with a
        // parameter of its own rather than a level.
        XCTAssertEqual(supported.aliases["medium"], "high")
        XCTAssertTrue(supported.offSwitch)
    }

    func testAModelThatReportsNoLevelsStillUsesTheTable() {
        let model = GenKit.Model(id: "deepseek-flash", owner: "deepseek")
        let supported = OpenAIService.Reasoning.supported(by: model, dialect: .deepSeek)
        XCTAssertEqual(supported.levels, ["low", "high", "max"])
    }

    func testAReportedLevelIsWhatGetsSent() async throws {
        let service = OpenAIService(apiKey: "test", dialect: .deepSeek)
        let decoded = await service.decodeForTesting(try model(Self.flash))

        var request = ChatServiceRequest(model: decoded, messages: [Message(role: .user, content: "hi")])
        request.options["effort"] = .string("medium")

        // Aliased to high by the table, and high is a level it reports.
        let sent = await service.makeRequest(request, streaming: true)
        XCTAssertEqual(sent.reasoning_effort?.rawValue, "high")
        XCTAssertEqual(sent.max_tokens, nil, "no ceiling was asked for")
    }
}

/// The day a service stops serving a model, where it publishes one.
final class RetirementDateTests: XCTestCase {

    private func model(_ json: String) throws -> OpenAI.Model {
        try JSONDecoder().decode(OpenAI.Model.self, from: Data(json.utf8))
    }

    func testADateIsRead() throws {
        let json = #"{"id":"gpt-5-codex","object":"model","created":1,"owned_by":"system","shutdown_date":"2026-07-23"}"#
        let date = try XCTUnwrap(OpenAIService.retirement(of: model(json)))

        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC")!
        XCTAssertEqual(calendar.dateComponents([.year, .month, .day], from: date).year, 2026)
        XCTAssertEqual(calendar.dateComponents([.year, .month, .day], from: date).month, 7)
        XCTAssertEqual(calendar.dateComponents([.year, .month, .day], from: date).day, 23)
    }

    func testNoDateIsNoClaim() throws {
        let explicit = #"{"id":"m","object":"model","created":1,"owned_by":"o","shutdown_date":null}"#
        XCTAssertNil(OpenAIService.retirement(of: try model(explicit)))

        let absent = #"{"id":"m","object":"model","created":1,"owned_by":"o"}"#
        XCTAssertNil(OpenAIService.retirement(of: try model(absent)))
    }

    func testAPastDateHasRetiredAndAFutureOneHasNot() {
        let past = GenKit.Model(id: "a", owner: "o", retiresOn: .now.addingTimeInterval(-86_400 * 2))
        let future = GenKit.Model(id: "b", owner: "o", retiresOn: .now.addingTimeInterval(86_400 * 2))
        let undated = GenKit.Model(id: "c", owner: "o")

        XCTAssertTrue(past.hasRetired)
        XCTAssertFalse(future.hasRetired)
        XCTAssertFalse(undated.hasRetired, "no announced end is not an end")
    }

    /// The day it stops, not the instant — so the day itself is not yet past.
    func testTheDayItselfIsNotPast() {
        let today = GenKit.Model(id: "a", owner: "o", retiresOn: .now)
        XCTAssertFalse(today.hasRetired)
    }
}
