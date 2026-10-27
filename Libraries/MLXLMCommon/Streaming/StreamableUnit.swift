// StreamableUnit.swift — the unit of weight streaming: byte ranges on disk
// that are fetched together.

import Foundation

/// One contiguous byte range in a weight file.
public struct UnitExtent: Sendable, Hashable {
    public let path: String
    public let fileOffset: UInt64
    public let length: Int

    public init(path: String, fileOffset: UInt64, length: Int) {
        self.path = path
        self.fileOffset = fileOffset
        self.length = length
    }
}

/// One expert of one MoE layer. `layer` is the engine's registration index.
public struct UnitID: Sendable, Hashable {
    public let layer: Int
    public let expert: Int

    public init(layer: Int, expert: Int) {
        self.layer = layer
        self.expert = expert
    }
}

/// Weights that are fetched as one piece. An expert is three extents
/// (gate, up, down), which may live in different shard files.
public struct StreamableUnit: Sendable {
    public let id: UnitID
    public let extents: [UnitExtent]
    public var residencyPriority: Double

    public init(id: UnitID, extents: [UnitExtent], residencyPriority: Double = 0) {
        self.id = id
        self.extents = extents
        self.residencyPriority = residencyPriority
    }

    public var byteLength: Int { extents.reduce(0) { $0 + $1.length } }
}

public struct SafetensorsLayoutError: Error, CustomStringConvertible {
    public let description: String
}

/// Tensor byte ranges read from safetensors headers, parsed once per file.
public final class SafetensorsLayout: @unchecked Sendable {
    public static let shared = SafetensorsLayout()

    public struct Tensor: Sendable {
        /// Absolute file offset of the tensor's first byte.
        public let dataStart: UInt64
        public let byteLength: Int
        public let shape: [Int]

        /// Extent of slab `index` along axis 0. Matches `mlx_fast_pread_into`:
        /// slab size is `byteLength / shape[0]`.
        public func slab(_ index: Int, path: String) -> UnitExtent? {
            guard let count = shape.first, count > 0, index >= 0, index < count else { return nil }
            let length = byteLength / count
            return UnitExtent(
                path: path, fileOffset: dataStart + UInt64(index * length), length: length)
        }
    }

    private let lock = NSLock()
    private var files: [String: [String: Tensor]] = [:]

    public init() {}

    public func tensor(_ name: String, in path: String) throws -> Tensor {
        let tensors = try lock.withLock {
            if let cached = files[path] { return cached }
            let parsed = try Self.parseHeader(path)
            files[path] = parsed
            return parsed
        }
        guard let tensor = tensors[name] else {
            throw SafetensorsLayoutError(description: "tensor \(name) not found in \(path)")
        }
        return tensor
    }

    static func parseHeader(_ path: String) throws -> [String: Tensor] {
        guard let handle = FileHandle(forReadingAtPath: path) else {
            throw SafetensorsLayoutError(description: "cannot open \(path)")
        }
        defer { try? handle.close() }
        guard let lengthBytes = try handle.read(upToCount: 8), lengthBytes.count == 8 else {
            throw SafetensorsLayoutError(description: "cannot read header length of \(path)")
        }
        let headerLength = lengthBytes.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
        guard let header = try handle.read(upToCount: Int(headerLength)),
            header.count == Int(headerLength),
            let json = try JSONSerialization.jsonObject(with: header) as? [String: Any]
        else {
            throw SafetensorsLayoutError(description: "cannot read header JSON of \(path)")
        }
        let dataSection = 8 + headerLength
        var tensors = [String: Tensor]()
        for (name, value) in json where name != "__metadata__" {
            guard let entry = value as? [String: Any],
                let offsets = entry["data_offsets"] as? [Int], offsets.count == 2,
                let shape = entry["shape"] as? [Int]
            else { continue }
            tensors[name] = Tensor(
                dataStart: dataSection + UInt64(offsets[0]),
                byteLength: offsets[1] - offsets[0], shape: shape)
        }
        return tensors
    }
}
