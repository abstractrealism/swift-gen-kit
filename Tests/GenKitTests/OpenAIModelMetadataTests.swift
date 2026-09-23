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
