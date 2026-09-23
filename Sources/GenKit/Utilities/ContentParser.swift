import Foundation

public final class ContentParser {
    public static let shared = ContentParser()

    public struct Result: Sendable {
        public var contents: [Content]

        public enum Content: Sendable {
            case text(String)
            case tag(Tag)
        }

        public struct Tag: Sendable {
            public let name: String
            public let content: String?
            public let params: [String: String]
            public let hasClosingTag: Bool

            public init(name: String, content: String? = nil, params: [String : String] = [:], hasClosingTag: Bool = true) {
                self.name = name
                self.content = content
                self.params = params
                self.hasClosingTag = hasClosingTag
            }
        }

        public init(contents: [Content]) {
            self.contents = contents
        }

        public func first(tag name: String) -> Tag? {
            for content in contents {
                if case .tag(let tag) = content {
                    if tag.name == name { return tag }
                }
            }
            return nil
        }
    }

    private let tagPattern = #/<(?<name>[^>\s]+)(?<params>\s+[^>]+)?>(?<content>.*?)(?<closing><\/\k<name>>|$)/#
    private let tagParamsPattern = #/(?<name>\w+)="(?<value>[^"]*)"/#

    private init() {}

    public func parse(input: String, tags: [String] = []) throws -> Result {
        let pattern = tagPattern.dotMatchesNewlines()
        var contents: [Result.Content] = []
        var positionIndex = input.startIndex

        // Append text to contents by checking if the last item in the contents array is already a
        // text element. When it is, the text gets appended rather than creating a new text element.
        let appendToContent: (String) -> Void = { text in
            if case .text(let lastText) = contents.last {
                contents[contents.count - 1] = .text(lastText + text)
            } else {
                contents.append(.text(text))
            }
        }

        // One match at a time from where the last left off, rather than every
        // match up front: how far a match reaches depends on whether its tag
        // is one that was asked for, so where the next search starts isn't
        // known until this one has been looked at.
        while positionIndex < input.endIndex, let match = input[positionIndex...].firstMatch(of: pattern) {
            let range = match.range
            let name = String(match.output.name)

            guard tags.isEmpty || tags.contains(name) else {
                // Not a tag anyone asked for — HTML or JSX in an answer, a
                // generic like `Array<Int>`, an autolink. It's text, and it
                // has to stay in one piece with the text around it: split
                // off, each piece is read as markdown on its own, and a code
                // block that spans the split is cut in two. And what the
                // match took to be the tag's content may hold a tag that *is*
                // wanted — an autolink has no closing tag, so its match ran
                // to the end of the message and hid everything after it — so
                // only the opening `<…>` is passed over, and the search goes
                // on from just after it. The opening tag ends at the first
                // `>`; neither the name nor its attributes can contain one.
                let openingEnd = input[range].firstIndex(of: ">").map { input.index(after: $0) } ?? range.upperBound
                appendToContent(String(input[positionIndex..<openingEnd]))
                positionIndex = openingEnd
                continue
            }

            // Add text before the tag if there's any
            if positionIndex < range.lowerBound {
                appendToContent(String(input[positionIndex..<range.lowerBound]))
            }

            let content = String(match.output.content)
            let params = try parseTagParams(match.output.params)

            // Determine if the tag has a closing element by checking the captured closing group
            let closingTag = String(match.output.closing)
            let hasClosingTag = closingTag.starts(with: "</") && closingTag.contains(name)

            contents.append(.tag(.init(name: name, content: content, params: params, hasClosingTag: hasClosingTag)))
            positionIndex = range.upperBound
        }

        // Add any remaining text after the last tag
        if positionIndex < input.endIndex {
            let text = String(input[positionIndex...])
            appendToContent(text)
        }

        // Filter out any text content with empty strings after stripping whitespace
        contents = contents.filter { content in
            switch content {
            case .text(let text):
                return !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            case .tag:
                return true
            }
        }
        return .init(contents: contents)
    }

    private func parseTagParams(_ input: Substring?) throws -> [String: String] {
        guard let input else { return [:] }
        let matches = input.matches(of: tagParamsPattern)
        var out: [String: String] = [:]
        for match in matches {
            let (_, name, value) = match.output
            out[String(name)] = String(value)
        }
        return out
    }
}

/// A reply's answer as it should be read, given the reasoning is being
/// wrapped separately.
///
/// Where a service parses `<think>…</think>` out of a model's raw output
/// into a field of its own, the closing tag can be left behind at the front
/// of the content. Groq's Qwen does this: the reasoning arrives whole in
/// `reasoning`, and the content it hands back begins `</think>`. Wrapped in
/// a block of our own making, that tag became the first thing on screen
/// after the reasoning — a stray `</think>` above the answer, which reads as
/// the app having mangled something.
///
/// Only ever a tag with nothing before it but space, and only where
/// reasoning is being folded anyway. An answer that genuinely opens with a
/// closing tag it never opened isn't a thing a model writes.
func strippingLeadingCloseTag(_ answer: String) -> String {
    let trimmed = answer.drop { $0.isWhitespace }
    guard trimmed.hasPrefix("</think>") else { return answer }
    return String(trimmed.dropFirst("</think>".count).drop { $0.isWhitespace })
}
