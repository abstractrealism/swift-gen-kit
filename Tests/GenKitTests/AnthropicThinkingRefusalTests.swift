import XCTest
import Anthropic
@testable import GenKit

/// `canDisable` is the one thing Anthropic's model list doesn't report, so
/// any model the table doesn't name is assumed to accept being told not to
/// think — and a model that reasons unconditionally failed the turn until
/// somebody edited the table.
///
/// It is learned from the refusal now. Deliberately not probed on Load
/// Models the way reasoning levels are: a level list has to fill a picker
/// before anybody chooses, where this is one boolean that only matters when
/// a request goes out, and the first real request teaches it for free.
final class AnthropicThinkingRefusalTests: XCTestCase {

    private let service = AnthropicService(apiKey: "test")

    /// A model that reports it can be told not to think, and isn't in the
    /// table — so the old assumption and the new one agree until it refuses.
    private var model: GenKit.Model {
        .init(
            id: "claude-something-new",
            owner: "anthropic",
            capabilities: [.completion, .thinking],
            reasoning: .init(levels: ["low", "medium", "high"], modes: ["enabled", "adaptive"])
        )
    }

    func testThinkingOffAsksToDisableUntilItIsRefused() async {
        let req = await service.makeRequestForTesting(model: model, options: ["effort": .string("off")])
        XCTAssertNotNil(req.thinking, "off means disabled, which is what it asks for first")
    }

    /// Once refused, the same request asks for the least instead — the
    /// behaviour the table already produced for the models it knew about,
    /// reached without the table knowing.
    func testOnceRefusedItAsksForTheLeastInstead() async {
        let req = await service.makeRequestForTesting(
            model: model,
            options: ["effort": .string("off")],
            refusedDisabling: true
        )
        XCTAssertNotEqual(
            String(describing: req.thinking), String(describing: Anthropic.ChatRequest.Thinking.disabled),
            "it must not ask again for the thing that was refused"
        )
    }

    /// Only a 400 is read as "you asked wrongly". The others are about
    /// something else entirely and must not trigger a retry that would then
    /// record the wrong lesson: 401 and 403 are the key, 429 the rate, 5xx
    /// theirs. Narrowing to 400 is what makes this safe without matching the
    /// message, which is prose and can be reworded at any time.
    func testOnlyABadRequestCountsAsBeingAskedWrongly() throws {
        let url = try XCTUnwrap(URL(string: "https://api.anthropic.com/v1/messages"))
        func failure(_ status: Int) -> Swift.Error {
            let response = HTTPURLResponse(url: url, statusCode: status, httpVersion: nil, headerFields: nil)!
            return Anthropic.Client.Error.responseError(response: response, detail: "nope")
        }

        XCTAssertTrue(service.isBadRequestForTesting(failure(400)))
        for status in [401, 403, 404, 429, 500, 529] {
            XCTAssertFalse(service.isBadRequestForTesting(failure(status)), "\(status) is not about thinking")
        }
        XCTAssertFalse(
            service.isBadRequestForTesting(URLError(.notConnectedToInternet)),
            "a connection that never arrived teaches nothing"
        )
    }
}
