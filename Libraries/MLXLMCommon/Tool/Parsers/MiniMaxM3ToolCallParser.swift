// Copyright © 2026 Apple Inc.

import Foundation

/// Parser for MiniMax M3's tool-call format.
///
/// Every tag carries the `]<]minimax[>[` prefix, and one block can hold several invokes:
/// ```
/// ]<]minimax[>[<tool_call>
/// ]<]minimax[>[<invoke name="f">]<]minimax[>[<k>v]<]minimax[>[</k>]<]minimax[>[</invoke>
/// ]<]minimax[>[</tool_call>
/// ```
/// Objects nest as child tags and arrays as `<item>` children. Leaf values are text; the
/// tool schema decides object vs array here, and `ToolArgumentNormalization` types the leaves.
/// Only prefixed tags are markup, so values may contain `<` and `>`.
/// Reference: the model's `chat_template.jinja` (`to_xml`).
public struct MiniMaxM3ToolCallParser: ToolCallParser, Sendable {
    static let prefix = "]<]minimax[>["
    public let startTag: String? = prefix + "<tool_call>"
    public let endTag: String? = prefix + "</tool_call>"

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        parseFrame(content: content, tools: tools).first
    }

    public func parseFrame(content: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        guard let root = Self.tree(content) else { return [] }
        var invokes = root.children
        if invokes.count == 1, invokes[0].name == "tool_call" {
            guard invokes[0].text.allSatisfy(\.isWhitespace) else { return [] }
            invokes = invokes[0].children
        }
        var calls: [ToolCall] = []
        for invoke in invokes {
            guard invoke.name == "invoke", invoke.text.allSatisfy(\.isWhitespace),
                let name = invoke.attribute("name"), !name.isEmpty
            else { return [] }
            let properties = getParameterConfig(funcName: name, tools: tools)
            var arguments: [String: any Sendable] = [:]
            for parameter in invoke.children {
                guard
                    let value = Self.value(
                        parameter, schema: properties[parameter.name] as? [String: any Sendable])
                else { return [] }
                arguments[parameter.name] = value.sendableValue
            }
            calls.append(ToolCall(function: .init(name: name, arguments: arguments)))
        }
        return calls
    }

    public func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        parseFrame(content: toolCallBuffer, tools: tools)
    }

    /// One prefixed element. `tag` is the text between `<` and `>`, attributes included.
    struct Element {
        var tag: Substring
        var children: [Element] = []
        var text = ""

        var name: String { String(tag.prefix { !$0.isWhitespace }) }

        func attribute(_ key: String) -> String? {
            guard let start = tag.range(of: " \(key)=\"") else { return nil }
            guard let end = tag[start.upperBound...].firstIndex(of: "\"") else { return nil }
            return String(tag[start.upperBound ..< end])
        }
    }

    /// Builds the element tree. Fails on stray text or mismatched tags; an unclosed
    /// `tool_call` wrapper is tolerated (generation can stop right after the last invoke).
    static func tree(_ content: String) -> Element? {
        let pieces = content.components(separatedBy: prefix)
        guard pieces[0].allSatisfy(\.isWhitespace) else { return nil }
        var stack = [Element(tag: "")]
        for piece in pieces.dropFirst() {
            guard piece.first == "<", let close = piece.firstIndex(of: ">") else { return nil }
            let tag = piece[piece.index(after: piece.startIndex) ..< close]
            if tag.first == "/" {
                guard stack.count > 1, stack[stack.count - 1].name == String(tag.dropFirst())
                else { return nil }
                let element = stack.removeLast()
                stack[stack.count - 1].children.append(element)
            } else {
                guard !tag.isEmpty else { return nil }
                stack.append(Element(tag: tag))
            }
            stack[stack.count - 1].text += piece[piece.index(after: close)...]
        }
        if stack.count == 2, stack[1].name == "tool_call" {
            let element = stack.removeLast()
            stack[0].children.append(element)
        }
        guard stack.count == 1, stack[0].text.allSatisfy(\.isWhitespace) else { return nil }
        return stack[0]
    }

    /// A parameter's value. Leaves stay text; children form an object, or an array when
    /// the schema says so or (without one saying object) every child is `<item>`.
    static func value(_ element: Element, schema: [String: any Sendable]?) -> JSONValue? {
        let type = schemaType(schema)
        guard !element.children.isEmpty else {
            if element.text.isEmpty, type == "array" { return .array([]) }
            if element.text.isEmpty, type == "object" { return .object([:]) }
            return .string(element.text)
        }
        guard element.text.allSatisfy(\.isWhitespace) else { return nil }
        let isArray =
            type == "array"
            || (type != "object" && element.children.allSatisfy { $0.name == "item" })
        if isArray {
            let items = schema?["items"] as? [String: any Sendable]
            var array: [JSONValue] = []
            for child in element.children {
                guard let item = value(child, schema: items) else { return nil }
                array.append(item)
            }
            return .array(array)
        }
        let properties = schema?["properties"] as? [String: any Sendable] ?? [:]
        let extra = schema?["additionalProperties"] as? [String: any Sendable]
        var object: [String: JSONValue] = [:]
        for child in element.children {
            guard
                let field = value(
                    child, schema: properties[child.name] as? [String: any Sendable] ?? extra)
            else { return nil }
            object[child.name] = field
        }
        return .object(object)
    }

    /// The schema's single non-null `type`, if it declares one.
    private static func schemaType(_ schema: [String: any Sendable]?) -> String? {
        switch schema?["type"] {
        case let name as String: return name
        case let names as [String]:
            let nonNull = names.filter { $0 != "null" }
            return nonNull.count == 1 ? nonNull[0] : nil
        default: return nil
        }
    }
}
