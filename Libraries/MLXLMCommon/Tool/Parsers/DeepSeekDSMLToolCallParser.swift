// Copyright © 2026 Apple Inc.

import Foundation

/// Parser for DeepSeek V4's DSML tool-call format.
///
/// One `calls` block can hold several invokes:
/// ```
/// <｜DSML｜ calls>
/// <｜DSML｜ invoke name="f">
/// <｜DSML｜ parameter name="k" string="true">v</｜DSML｜ parameter>
/// </｜DSML｜ invoke>
/// </｜DSML｜ calls>
/// ```
/// `string="false"` values are JSON. Reference: the model's `chat_template.jinja`.
public struct DeepSeekDSMLToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = "<｜DSML｜ calls>"
    public let endTag: String? = "</｜DSML｜ calls>"

    private static let invokePrefix = "<｜DSML｜ invoke"
    private static let invokeEnd = "</｜DSML｜ invoke>"
    private static let parameterPrefix = "<｜DSML｜ parameter"
    private static let parameterEnd = "</｜DSML｜ parameter>"

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        parseFrame(content: content, tools: tools).first
    }

    public func parseFrame(content: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        var body = content[...]
        if let startTag, let start = body.range(of: startTag) {
            guard body[..<start.lowerBound].allSatisfy(\.isWhitespace) else { return [] }
            body = body[start.upperBound...]
        }
        if let endTag, let end = body.range(of: endTag, options: .backwards) {
            guard body[end.upperBound...].allSatisfy(\.isWhitespace) else { return [] }
            body = body[..<end.lowerBound]
        }

        var calls: [ToolCall] = []
        while let start = body.range(of: Self.invokePrefix) {
            guard body[..<start.lowerBound].allSatisfy(\.isWhitespace),
                let tagEnd = body[start.upperBound...].firstIndex(of: ">"),
                let close = body.range(
                    of: Self.invokeEnd, range: body.index(after: tagEnd) ..< body.endIndex),
                let name = Self.attribute("name", in: body[start.upperBound ..< tagEnd]),
                !name.isEmpty,
                let arguments = Self.arguments(
                    in: body[body.index(after: tagEnd) ..< close.lowerBound])
            else { return [] }
            calls.append(ToolCall(function: .init(name: name, arguments: arguments)))
            body = body[close.upperBound...]
        }
        guard body.allSatisfy(\.isWhitespace) else { return [] }
        return calls
    }

    public func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        parseFrame(content: toolCallBuffer, tools: tools)
    }

    private static func arguments(in text: Substring) -> [String: any Sendable]? {
        var arguments: [String: any Sendable] = [:]
        var remaining = text
        while let start = remaining.range(of: parameterPrefix) {
            guard remaining[..<start.lowerBound].allSatisfy(\.isWhitespace),
                let tagEnd = remaining[start.upperBound...].firstIndex(of: ">"),
                let close = remaining.range(
                    of: parameterEnd,
                    range: remaining.index(after: tagEnd) ..< remaining.endIndex)
            else { return nil }
            let attributes = remaining[start.upperBound ..< tagEnd]
            guard let name = attribute("name", in: attributes), !name.isEmpty else { return nil }
            let raw = String(remaining[remaining.index(after: tagEnd) ..< close.lowerBound])
            arguments[name] = attribute("string", in: attributes) == "false" ? json(raw) : raw
            remaining = remaining[close.upperBound...]
        }
        return remaining.allSatisfy(\.isWhitespace) ? arguments : nil
    }

    /// The value of `key="..."` in a tag's attribute text.
    private static func attribute(_ key: String, in text: Substring) -> String? {
        guard let start = text.range(of: " \(key)=\"") else { return nil }
        guard let end = text[start.upperBound...].firstIndex(of: "\"") else { return nil }
        return String(text[start.upperBound ..< end])
    }

    /// Non-string values are JSON; keep the raw text if the model wrote invalid JSON.
    private static func json(_ raw: String) -> any Sendable {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let data = trimmed.data(using: .utf8),
            let object = try? JSONSerialization.jsonObject(with: data, options: .fragmentsAllowed)
        else { return raw }
        return asSendable(object)
    }
}
