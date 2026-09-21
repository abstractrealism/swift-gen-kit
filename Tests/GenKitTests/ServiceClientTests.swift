import XCTest
@testable import GenKit

/// A service hands out a client whether or not a preferred model is set.
///
/// The preference is a default for callers that have no model of their own
/// in mind. One that has — a conversation pinned to a model — was refused a
/// client for a service it had every right to use, with "missing service",
/// which named the one thing that wasn't. Seen the first time a model was
/// picked from a service whose Chats default had never been chosen.
final class ServiceClientTests: XCTestCase {

    private static let chatKinds: [Service.Kind] = [
        .anthropic, .deepseek, .grok, .groq, .mistral, .ollama, .openAI, .openRouter,
    ]

    private func service(_ kind: Service.Kind) -> Service {
        Service(kind: kind, name: "\(kind)", host: "https://example.invalid/v1", token: "t")
    }

    func testAChatClientNeedsNoPreferredModel() throws {
        for kind in Self.chatKinds {
            let service = service(kind)
            XCTAssertNil(service.preferredChatModel)
            XCTAssertNoThrow(try service.chatService(session: nil), "\(kind)")
        }
    }

    /// The summarization client checked the *chat* preference, so a service
    /// with a summarization model set and no chat model was refused for a job
    /// it had been configured for.
    func testASummarizationClientNeedsNeitherPreference() throws {
        var service = service(.groq)
        service.preferredSummarizationModel = "small"
        XCTAssertNoThrow(try service.summarizationService(session: nil))
    }

    func testTheOtherClientsNeedNoPreferenceEither() throws {
        XCTAssertNoThrow(try service(.openAI).imageService(session: nil))
        XCTAssertNoThrow(try service(.openAI).embeddingService(session: nil))
        XCTAssertNoThrow(try service(.openAI).transcriptionService(session: nil))
        XCTAssertNoThrow(try service(.elevenLabs).speechService(session: nil))
    }

    /// What a service can't do is still refused — and refused in words.
    func testAServiceThatCannotChatSaysSo() {
        for kind in [Service.Kind.elevenLabs, .fal] {
            XCTAssertThrowsError(try service(kind).chatService(session: nil), "\(kind)") { error in
                guard case ServiceError.unsupportedService = error else {
                    return XCTFail("\(kind): \(error)")
                }
                // A thread prints what it's given. A bare case reads as its
                // name, which says nothing anybody can act on.
                XCTAssertFalse("\(error)".contains("unsupportedService"), "\(error)")
                XCTAssertTrue("\(error)".contains("doesn't do that kind of work"), "\(error)")
                XCTAssertEqual(error.localizedDescription, "\(error)")
            }
        }
    }
}
