// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("MiniMax M3 tool calls")
struct MiniMaxM3ToolCallTests {

    /// Three calls rendered by MiniMax-M3.1's own `chat_template.jinja` (jinja2 3.1.6).
    static let rendered = #"""
        ]<]minimax[>[<tool_call>
        ]<]minimax[>[<invoke name="write">]<]minimax[>[<filePath>/tmp/a b.swift]<]minimax[>[</filePath>]<]minimax[>[<content>let x: Array<Int> = [1, 2]
        if a < b && c > d { print("hi") }
        ]<]minimax[>[</content>]<]minimax[>[<overwrite>true]<]minimax[>[</overwrite>]<]minimax[>[<mode>420]<]minimax[>[</mode>]<]minimax[>[<ratio>0.5]<]minimax[>[</ratio>]<]minimax[>[</invoke>
        ]<]minimax[>[<invoke name="edit">]<]minimax[>[<edits>]<]minimax[>[<item>]<]minimax[>[<old>foo]<]minimax[>[</old>]<]minimax[>[<new>bar]<]minimax[>[</new>]<]minimax[>[<line>3]<]minimax[>[</line>]<]minimax[>[</item>]<]minimax[>[<item>]<]minimax[>[<old>x]<]minimax[>[</old>]<]minimax[>[<new>y]<]minimax[>[</new>]<]minimax[>[<line>7]<]minimax[>[</line>]<]minimax[>[</item>]<]minimax[>[</edits>]<]minimax[>[<tags>]<]minimax[>[<item>a]<]minimax[>[</item>]<]minimax[>[<item>b]<]minimax[>[</item>]<]minimax[>[</tags>]<]minimax[>[<options>]<]minimax[>[<dry>false]<]minimax[>[</dry>]<]minimax[>[<env>]<]minimax[>[<HOME>/Users/x]<]minimax[>[</HOME>]<]minimax[>[</env>]<]minimax[>[</options>]<]minimax[>[<empty>]<]minimax[>[</empty>]<]minimax[>[</invoke>
        ]<]minimax[>[<invoke name="todoread">]<]minimax[>[</invoke>
        ]<]minimax[>[</tool_call>
        """#

    static func tool(_ name: String, _ properties: [String: any Sendable]) -> [String: any Sendable]
    {
        [
            "type": "function",
            "function": [
                "name": name,
                "parameters": ["type": "object", "properties": properties]
                    as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }

    static let tools: [[String: any Sendable]] = [
        tool(
            "write",
            [
                "filePath": ["type": "string"], "content": ["type": "string"],
                "overwrite": ["type": "boolean"], "mode": ["type": "integer"],
                "ratio": ["type": "number"],
            ]),
        tool(
            "edit",
            [
                "edits": [
                    "type": "array",
                    "items": [
                        "type": "object",
                        "properties": [
                            "old": ["type": "string"], "new": ["type": "string"],
                            "line": ["type": "integer"],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
                "tags": ["type": "array", "items": ["type": "string"]] as [String: any Sendable],
                "options": [
                    "type": "object",
                    "properties": [
                        "dry": ["type": "boolean"],
                        "env": ["type": "object", "additionalProperties": ["type": "string"]]
                            as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
                "empty": ["type": "array", "items": ["type": "string"]] as [String: any Sendable],
            ]),
        tool("todoread", [:]),
    ]

    /// Feeds `text` to a processor in 3-character chunks, like a token stream.
    static func stream(_ text: String, tools: [[String: any Sendable]]?) -> (
        String, [ToolCall]
    ) {
        let processor = ToolCallProcessor(format: .minimaxM3, tools: tools)
        var output = ""
        var index = text.startIndex
        while index < text.endIndex {
            let end = text.index(index, offsetBy: 3, limitedBy: text.endIndex) ?? text.endIndex
            output += processor.processChunk(String(text[index ..< end])) ?? ""
            index = end
        }
        output += processor.processEOS(returnBufferedText: true) ?? ""
        return (output, processor.toolCalls)
    }

    @Test func templateRenderRoundTripsWithSchema() throws {
        let (output, calls) = Self.stream("Let me write it.\n\n" + Self.rendered, tools: Self.tools)
        #expect(output == "Let me write it.\n\n")
        try #require(calls.map(\.function.name) == ["write", "edit", "todoread"])

        #expect(
            calls[0].function.arguments == [
                "filePath": .string("/tmp/a b.swift"),
                "content": .string(
                    "let x: Array<Int> = [1, 2]\nif a < b && c > d { print(\"hi\") }\n"),
                "overwrite": .bool(true), "mode": .int(420), "ratio": .double(0.5),
            ])
        #expect(
            calls[1].function.arguments == [
                "edits": .array([
                    .object(["old": .string("foo"), "new": .string("bar"), "line": .int(3)]),
                    .object(["old": .string("x"), "new": .string("y"), "line": .int(7)]),
                ]),
                "tags": .array([.string("a"), .string("b")]),
                "options": .object([
                    "dry": .bool(false), "env": .object(["HOME": .string("/Users/x")]),
                ]),
                "empty": .array([]),
            ])
        #expect(calls[2].function.arguments.isEmpty)
    }

    @Test func withoutSchemaLeavesAreTextAndItemsAreArrays() throws {
        let calls = MiniMaxM3ToolCallParser().parseFrame(content: Self.rendered, tools: nil)
        try #require(calls.count == 3)
        #expect(calls[0].function.arguments["mode"] == .string("420"))
        #expect(calls[0].function.arguments["overwrite"] == .string("true"))
        #expect(calls[1].function.arguments["tags"] == .array([.string("a"), .string("b")]))
        #expect(
            calls[1].function.arguments["options"]
                == .object(["dry": .string("false"), "env": .object(["HOME": .string("/Users/x")])])
        )
        #expect(calls[1].function.arguments["empty"] == .string(""))
    }

    @Test func objectSchemaKeepsAnItemKeyAsAField() throws {
        let p = MiniMaxM3ToolCallParser.prefix
        let content = "\(p)<invoke name=\"f\">\(p)<x>\(p)<item>1\(p)</item>\(p)</x>\(p)</invoke>"
        let tools = [Self.tool("f", ["x": ["type": "object"]])]
        let call = try #require(MiniMaxM3ToolCallParser().parse(content: content, tools: tools))
        #expect(call.function.arguments["x"] == .object(["item": .string("1")]))
    }

    @Test func toleratesMissingClosingWrapper() throws {
        let p = MiniMaxM3ToolCallParser.prefix
        let content = "\(p)<tool_call>\n\(p)<invoke name=\"f\">\(p)<k>v\(p)</k>\(p)</invoke>\n"
        let calls = MiniMaxM3ToolCallParser().parseEOS(content, tools: nil)
        try #require(calls.count == 1)
        #expect(calls[0].function.arguments["k"] == .string("v"))
    }

    @Test(arguments: [
        // Unclosed invoke (generation cut off mid-call).
        "]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"f\">]<]minimax[>[<k>v",
        // Mismatched closing tag.
        "]<]minimax[>[<invoke name=\"f\">]<]minimax[>[<k>v]<]minimax[>[</j>]<]minimax[>[</invoke>",
        // Text between invokes.
        "]<]minimax[>[<invoke name=\"f\">]<]minimax[>[</invoke>stray]<]minimax[>[<invoke name=\"g\">]<]minimax[>[</invoke>",
        // Missing name.
        "]<]minimax[>[<invoke>]<]minimax[>[</invoke>",
        // Text mixed with child tags.
        "]<]minimax[>[<invoke name=\"f\">]<]minimax[>[<k>a]<]minimax[>[<j>b]<]minimax[>[</j>]<]minimax[>[</k>]<]minimax[>[</invoke>",
    ])
    func rejectsMalformedBlocks(content: String) {
        #expect(MiniMaxM3ToolCallParser().parseFrame(content: content, tools: nil).isEmpty)
    }

    @Test func processorKeepsTextWithoutCalls() {
        let (output, calls) = Self.stream("No tools: a < b and Array<Int>.", tools: Self.tools)
        #expect(output == "No tools: a < b and Array<Int>.")
        #expect(calls.isEmpty)
    }

    /// The template joins prefix and tag at render time and also mentions `tool_call.name`,
    /// so without the MiniMax signature it would match the generic JSON format.
    @Test func formatIsInferred() {
        #expect(
            ToolCallFormat.inferred(
                fromChatTemplate:
                    "{%- set ns = ']<]minimax[>[' -%}{{ ns ~ '<tool_call>' }} tool_call.name"
            ) == .minimaxM3)
        #expect(ToolCallFormat.infer(from: "minimax_m3_vl") == .minimaxM3)
        #expect(ToolCallFormat.infer(from: "minimax_m3_text") == .minimaxM3)
        #expect(ToolCallFormat(toolParserType: "minimax_m3") == .minimaxM3)
        #expect(ToolCallFormat.minimaxM3.rawValue == "minimax_m3")
    }
}
