// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("DeepSeek DSML tool calls")
struct DeepSeekDSMLToolCallTests {

    static let twoCalls = """
        <｜DSML｜ calls>
        <｜DSML｜ invoke name="read">
        <｜DSML｜ parameter name="filePath" string="true">/tmp/a.txt</｜DSML｜ parameter>
        <｜DSML｜ parameter name="limit" string="false">20</｜DSML｜ parameter>
        </｜DSML｜ invoke>
        <｜DSML｜ invoke name="bash">
        <｜DSML｜ parameter name="command" string="true">ls -la "dir with spaces"</｜DSML｜ parameter>
        <｜DSML｜ parameter name="options" string="false">{"timeout": 30, "env": ["A", "B"], "dry": true}</｜DSML｜ parameter>
        </｜DSML｜ invoke>
        </｜DSML｜ calls>
        """

    @Test func parsesEveryInvokeInOneBlock() throws {
        let calls = DeepSeekDSMLToolCallParser().parseFrame(content: Self.twoCalls, tools: nil)
        try #require(calls.count == 2)
        #expect(calls[0].function.name == "read")
        #expect(calls[0].function.arguments["filePath"] == .string("/tmp/a.txt"))
        #expect(calls[0].function.arguments["limit"] == .int(20))
        #expect(calls[1].function.name == "bash")
        #expect(calls[1].function.arguments["command"] == .string(#"ls -la "dir with spaces""#))
        #expect(
            calls[1].function.arguments["options"]
                == .object([
                    "timeout": .int(30), "env": .array([.string("A"), .string("B")]),
                    "dry": .bool(true),
                ]))
    }

    @Test func keepsStringValuesVerbatim() throws {
        let content = """
            <｜DSML｜ calls>
            <｜DSML｜ invoke name="write">
            <｜DSML｜ parameter name="content" string="true">line 1
              line 2 with "quotes" and <tags>
            </｜DSML｜ parameter>
            <｜DSML｜ parameter name="n" string="true">42</｜DSML｜ parameter>
            </｜DSML｜ invoke>
            </｜DSML｜ calls>
            """
        let call = try #require(DeepSeekDSMLToolCallParser().parse(content: content, tools: nil))
        #expect(
            call.function.arguments["content"]
                == .string("line 1\n  line 2 with \"quotes\" and <tags>\n"))
        #expect(call.function.arguments["n"] == .string("42"))
    }

    @Test func invokeWithoutParameters() throws {
        let content =
            "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"todoread\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>"
        let call = try #require(DeepSeekDSMLToolCallParser().parse(content: content, tools: nil))
        #expect(call.function.name == "todoread")
        #expect(call.function.arguments.isEmpty)
    }

    @Test func invalidJSONFallsBackToRawText() throws {
        let content = """
            <｜DSML｜ calls>
            <｜DSML｜ invoke name="f">
            <｜DSML｜ parameter name="x" string="false">not json</｜DSML｜ parameter>
            </｜DSML｜ invoke>
            </｜DSML｜ calls>
            """
        let call = try #require(DeepSeekDSMLToolCallParser().parse(content: content, tools: nil))
        #expect(call.function.arguments["x"] == .string("not json"))
    }

    @Test(arguments: [
        // Text between invokes.
        "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n</｜DSML｜ invoke>\nstray\n</｜DSML｜ calls>",
        // Unclosed parameter.
        "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n<｜DSML｜ parameter name=\"x\" string=\"true\">v\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
        // Missing name.
        "<｜DSML｜ calls>\n<｜DSML｜ invoke>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
    ])
    func rejectsMalformedBlocks(content: String) {
        #expect(DeepSeekDSMLToolCallParser().parseFrame(content: content, tools: nil).isEmpty)
    }

    @Test func processorEmitsAllCallsFromStreamedChunks() throws {
        let processor = ToolCallProcessor(format: .deepseekDSML)
        let text = "Let me look.\n\n" + Self.twoCalls
        var output = ""
        var index = text.startIndex
        while index < text.endIndex {
            let end = text.index(index, offsetBy: 3, limitedBy: text.endIndex) ?? text.endIndex
            output += processor.processChunk(String(text[index ..< end])) ?? ""
            index = end
        }
        output += processor.processEOS(returnBufferedText: true) ?? ""

        #expect(output == "Let me look.\n\n")
        try #require(processor.toolCalls.count == 2)
        #expect(processor.toolCalls.map(\.function.name) == ["read", "bash"])
    }

    @Test func processorKeepsTextWithoutCalls() {
        let processor = ToolCallProcessor(format: .deepseekDSML)
        let output =
            (processor.processChunk("No tools needed: a < b.") ?? "")
            + (processor.processEOS(returnBufferedText: true) ?? "")
        #expect(output == "No tools needed: a < b.")
        #expect(processor.toolCalls.isEmpty)
    }

    @Test func kimiSectionWithSeveralCalls() throws {
        let content =
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.read:0<|tool_call_argument_begin|>{\"filePath\": \"a\"}<|tool_call_end|><|tool_call_begin|>functions.bash:1<|tool_call_argument_begin|>{\"command\": \"ls\"}<|tool_call_end|><|tool_calls_section_end|>"
        let processor = ToolCallProcessor(format: .kimiK2)
        _ = processor.processChunk(content)
        try #require(processor.toolCalls.count == 2)
        #expect(processor.toolCalls[0].function.arguments["filePath"] == .string("a"))
        #expect(processor.toolCalls[1].function.name == "bash")
        #expect(processor.toolCalls.map(\.id) == ["functions.read:0", "functions.bash:1"])
    }

    @Test func formatIsInferred() {
        #expect(
            ToolCallFormat.inferred(fromChatTemplate: "write a \"<｜DSML｜ calls>\" block")
                == .deepseekDSML)
        #expect(ToolCallFormat.infer(from: "deepseek_v41") == .deepseekDSML)
        #expect(ToolCallFormat(toolParserType: "deepseek_v4") == .deepseekDSML)
        #expect(ToolCallFormat.deepseekDSML.rawValue == "deepseek_dsml")
    }
}
