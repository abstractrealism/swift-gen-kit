import Foundation
import OSLog

private let logger = Logger(subsystem: "ReasoningProbe", category: "GenKit")

/// Which reasoning efforts a model accepts, learned by offering one it doesn't.
///
/// No endpoint says. Every service here refuses a value a model doesn't take
/// and names the ones it does in the refusal — and that is where the
/// hand-written tables in the OpenAI and Anthropic providers came from, a
/// person sending a bad value, reading the error and writing the answer down.
/// `OpenAI.swift` says so outright: *"that is where every row here came
/// from."* This asks the same question and reads the same answer, so the
/// table doesn't have to be kept by hand and can't quietly go out of date.
///
/// **A raw request rather than one built with either typed client.** Both
/// `swift-openai` and `swift-anthropic` declare `reasoning_effort` as an enum
/// of the values they know, so neither can express the invalid value this
/// depends on — and widening those enums to allow nonsense, for the sake of a
/// diagnostic, would be the wrong way round. The body is the same shape for
/// every service that speaks the OpenAI dialect, Mistral included, so one
/// request covers all six.
///
/// **It costs nothing.** An invalid parameter is refused while the request is
/// being validated, before it reaches a model, so no tokens are generated or
/// billed and `max_tokens` never comes into it.
public enum ReasoningProbe {

    /// The value offered.
    ///
    /// Deliberately not a plausible level, and deliberately not spellable as
    /// one: no service can accept it, so the request can never reach a model,
    /// and the leading underscores keep it out of `levels(in:)` below without
    /// needing to be excluded by name.
    static let offered = "__genkit_probe__"

    /// What asking produced.
    ///
    /// Three outcomes rather than an optional, because two of the failures
    /// want opposite handling: a service that refused without naming its
    /// levels will do the same tomorrow and shouldn't be asked again, while
    /// one that couldn't be reached should.
    public enum Outcome: Sendable, Equatable {
        /// The levels it named, in the order below.
        case levels([String])
        /// It refused, and named nothing. Asking again would learn nothing.
        case refusedWithoutSaying
        /// The question didn't get through. Worth asking again later.
        case failed
    }

    public static func ask(
        host: URL,
        token: String,
        modelID: String,
        session: URLSession = .shared
    ) async -> Outcome {
        var request = URLRequest(url: host.appending(path: "chat/completions"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        request.httpBody = try? JSONSerialization.data(
            withJSONObject: [
                "model": modelID,
                "messages": [["role": "user", "content": "."]],
                "max_tokens": 1,
                "reasoning_effort": offered,
            ]
        )

        do {
            let (data, response) = try await session.data(for: request)
            let status = (response as? HTTPURLResponse)?.statusCode ?? 0
            let body = String(decoding: data, as: UTF8.self)

            guard status >= 400 else {
                // Accepted, which should not be possible: the value offered is
                // not a level any service has. Something was generated and
                // nothing was learned, so say both.
                logger.warning(
                    "\(modelID, privacy: .public) accepted a nonsense reasoning effort — learned nothing, and it generated"
                )
                return .refusedWithoutSaying
            }

            let found = levels(in: body)
            guard !found.isEmpty else {
                logger.info(
                    "\(modelID, privacy: .public) refused without naming its levels: \(body.prefix(240), privacy: .public)"
                )
                return .refusedWithoutSaying
            }
            logger.info("\(modelID, privacy: .public) takes \(found.joined(separator: ", "), privacy: .public)")
            return .levels(found)

        } catch {
            logger.error("couldn't ask \(modelID, privacy: .public): \(error, privacy: .public)")
            return .failed
        }
    }

    /// The level names quoted in a refusal.
    ///
    /// Two shapes seen, and this matches neither exactly, on purpose:
    ///
    ///     Mistral  supported values: [<ReasoningEffort.high: 'high'>, <ReasoningEffort.none: 'none'>]
    ///     OpenAI   Supported values are: 'minimal', 'low', 'medium', and 'high'.
    ///
    /// The first is a Python `repr` leaking through an API, which can change
    /// without notice, so nothing here depends on its structure. Everything
    /// between single quotes is taken and then filtered to what could be a
    /// level at all: two to twelve lowercase letters. That filter is what
    /// makes the split safe — an apostrophe in the prose would shift which
    /// fragments are "inside" quotes, and the fragments that result have
    /// spaces and punctuation in them, so they fall out here rather than
    /// being mistaken for answers.
    ///
    /// Deliberately **not** checked against a list of known levels. Such a
    /// list is the thing this exists to stop maintaining, and a level added
    /// by a service next month should arrive on its own.
    static func levels(in body: String) -> [String] {
        let quoted = body.components(separatedBy: "'")
            .filter { fragment in
                (2...12).contains(fragment.count)
                    && fragment.allSatisfy { $0.isASCII && $0.isLowercase && $0.isLetter }
            }

        var seen = Set<String>()
        let unique = quoted.filter { seen.insert($0).inserted }
        return unique.sorted { rank(of: $0) < rank(of: $1) }
    }

    /// Where a level sits from least to most, for display.
    ///
    /// Ordering only — it never decides what is kept. A name nobody here has
    /// heard of still comes back, it just sorts after the ones that are
    /// recognised, so a new level appears at the far end of a picker rather
    /// than not at all.
    private static func rank(of level: String) -> Int {
        let known = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
        return known.firstIndex(of: level) ?? known.count
    }
}
