import XCTest
@testable import GenKit

final class ContentParserTests: XCTestCase {

    func testTag() throws {
        let simpleInput = "<tag>content</tag>"
        let simpleResult = try ContentParser.shared.parse(input: simpleInput)
        
        XCTAssertEqual(simpleResult.contents.count, 1)
        XCTAssertEqual(simpleResult.first(tag: "tag")?.content, "content")
    }
    
    func testTagWithAttributes() throws {
        let attributeInput = "<tag name=\"Tag\" id=\"1\">content</tag>"
        let attributeResult = try ContentParser.shared.parse(input: attributeInput)
        
        XCTAssertEqual(attributeResult.contents.count, 1)
        
        let attributeTag = attributeResult.first(tag: "tag")
        
        XCTAssertEqual(attributeTag?.content, "content")
        XCTAssertEqual(attributeTag?.params["name"], "Tag")
        XCTAssertEqual(attributeTag?.params["id"], "1")
    }
    
    func testMultipleTagsAndText() throws {
        let mixedInput = "Text before <foo>Foo</foo> Text between <bar>Bar</bar> Text after"
        let mixedResult = try ContentParser.shared.parse(input: mixedInput)
        
        XCTAssertEqual(mixedResult.contents.count, 5)
        XCTAssertEqual(mixedResult.first(tag: "foo")?.content, "Foo")
        XCTAssertEqual(mixedResult.first(tag: "bar")?.content, "Bar")
    }
    
    func testSpecificTag() throws {
        let specificTagsInput = "<keep>Keep this</keep> <ignore>Ignore this</ignore>"
        let specificTagsResult = try ContentParser.shared.parse(input: specificTagsInput, tags: ["keep"])
        
        XCTAssertEqual(specificTagsResult.contents.count, 2)
        XCTAssertNotNil(specificTagsResult.first(tag: "keep"))
        XCTAssertNil(specificTagsResult.first(tag: "ignore"))
    }
    
    func testNestedTags() throws {
        let nestedInput = "<outer><inner>Nested content</inner></outer>"
        let nestedResult = try ContentParser.shared.parse(input: nestedInput)
        XCTAssertEqual(nestedResult.contents.count, 1)
        XCTAssertEqual(nestedResult.first(tag: "outer")?.content, "<inner>Nested content</inner>")
    }

    func testHasClosingTag() throws {
        let simpleInput = """
            Lorem ipsum
            <foo>content
            Dolor sit amet
            """
        let simpleResult = try ContentParser.shared.parse(input: simpleInput)

        XCTAssertEqual(simpleResult.first(tag: "foo")?.hasClosingTag, false)
    }
}

/// Tags the caller didn't ask for are text, and stay in one piece with the
/// text around them.
///
/// They didn't. Every `<name>…</name>` in a message was matched whether or
/// not it was wanted, and while an unwanted one was put back as text, the
/// text *before* it was started as a segment of its own. A renderer reads
/// each segment as a markdown document, so an answer with HTML in a code
/// block came apart at the HTML: the block was cut where `<p>…</p>` was,
/// the closing fence then *opened* one, and the rest of the answer rendered
/// as code. The message below is the one it was seen on.
final class ContentParserUnwantedTagTests: XCTestCase {

    private static let answer = """
        3. **Build a Weather widget** (`src/components/Weather.jsx`)  
           ```jsx
           import { useEffect, useState } from "react";

           export default function Weather({ city }) {
             const [data, setData] = useState(null);
             if (!data) return <p>Loading…</p>;
             return (
               <div className="p-4 bg-white rounded shadow">
                 <h2 className="font-bold">{data.name}</h2>
                 <p>{Math.round(data.main.temp)}°C – {data.weather[0].description}</p>
               </div>
             );
           }
           ```
        4. **Add other widgets** (GitHub commits, a simple to‑do list using `useState`).

        ### Resources

        - **Vite docs** – fast dev server & build: <https://vitejs.dev>
        - **React 18** – hooks & concurrent features: <https://react.dev>
        """

    private static let heatTags = ["thinking", "think", "artifact", "output", "summary", "image_search_query"]

    func testHTMLInAnAnswerLeavesItInOnePiece() throws {
        let result = try ContentParser.shared.parse(input: Self.answer, tags: Self.heatTags)
        XCTAssertEqual(result.contents.count, 1)
        guard case .text(let text)? = result.contents.first else { return XCTFail("\(result.contents)") }
        XCTAssertEqual(text, Self.answer)
    }

    /// An autolink has no closing tag, so its match reached the end of the
    /// message and hid every tag after it.
    func testAWantedTagAfterAnAutolinkIsStillFound() throws {
        let input = "See <https://example.com> for the docs.\n<artifact name=\"a\">x</artifact>"
        let result = try ContentParser.shared.parse(input: input, tags: ["artifact"])
        XCTAssertEqual(result.contents.count, 2)
        XCTAssertEqual(result.first(tag: "artifact")?.content, "x")
        XCTAssertEqual(result.first(tag: "artifact")?.params["name"], "a")
        guard case .text(let text)? = result.contents.first else { return XCTFail("\(result.contents)") }
        XCTAssertEqual(text, "See <https://example.com> for the docs.\n")
    }

    /// Likewise a wanted tag inside an unwanted one's content.
    func testAWantedTagInsideAnUnwantedOneIsStillFound() throws {
        let result = try ContentParser.shared.parse(input: "<div><think>hm</think>done</div>", tags: ["think"])
        XCTAssertEqual(result.contents.count, 3)
        XCTAssertEqual(result.first(tag: "think")?.content, "hm")
        guard case .text(let tail)? = result.contents.last else { return XCTFail("\(result.contents)") }
        XCTAssertEqual(tail, "done</div>")
    }

    /// The text on either side of a wanted tag is still its own segment.
    func testWantedTagsStillSplitTheText() throws {
        let result = try ContentParser.shared.parse(input: "a <b>1</b> c <think>t</think> d", tags: ["think"])
        XCTAssertEqual(result.contents.count, 3)
        guard case .text(let head)? = result.contents.first else { return XCTFail("\(result.contents)") }
        XCTAssertEqual(head, "a <b>1</b> c ")
        XCTAssertEqual(result.first(tag: "think")?.content, "t")
    }

    /// Mid-stream, a wanted tag that hasn't closed yet still reads as open.
    func testAnUnclosedWantedTagIsStillOpen() throws {
        let result = try ContentParser.shared.parse(input: "<p>x</p>\n<think>so far", tags: ["think"])
        XCTAssertEqual(result.first(tag: "think")?.content, "so far")
        XCTAssertEqual(result.first(tag: "think")?.hasClosingTag, false)
    }
}
