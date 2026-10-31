// StreamEngine.swift — concurrent pread of streamable units into resident
// buffers, with hit/miss and byte metrics.

import Foundation
import MLX

/// One extent to copy into `dst` at byte `dstOffset`.
public struct ReadJob {
    public let dst: MLXArray
    public let dstOffset: Int
    public let extent: UnitExtent

    public init(dst: MLXArray, dstOffset: Int, extent: UnitExtent) {
        self.dst = dst
        self.dstOffset = dstOffset
        self.extent = extent
    }
}

public final class StreamEngine: @unchecked Sendable {
    public static let shared = StreamEngine()

    public enum ReadKind { case prefetch, miss }

    public struct Metrics: Sendable {
        public var hits = 0
        public var misses = 0
        public var prefetchedUnits = 0
        public var bytesRead: UInt64 = 0
        public var readNanos: UInt64 = 0
        public var readBatches = 0
    }

    /// Concurrent preads per batch. 0 means one per job.
    public var queueDepth: Int
    /// Bypass the unified buffer cache (`F_NOCACHE`).
    public var noCache: Bool
    /// Split extents into preads of at most this many bytes, so a layer's few
    /// misses still keep more reads in flight (`MLX_MOE_STREAM_CHUNK_KB`, default 2048:
    /// +2.6% decode on K2.6; 256 KB is slower). 0 means one pread per extent.
    public var chunkBytes: Int

    private let lock = NSLock()
    private var metrics = Metrics()
    private var layerCount = 0

    init(environment: [String: String] = ProcessInfo.processInfo.environment) {
        queueDepth = Int(environment["MLX_MOE_STREAM_QD"] ?? "") ?? 0
        let nc = environment["MLX_MOE_STREAM_NOCACHE"] ?? ""
        noCache = nc == "1" || nc.lowercased() == "true"
        chunkBytes = (Int(environment["MLX_MOE_STREAM_CHUNK_KB"] ?? "") ?? 2048) * 1024
    }

    /// `jobs` with every extent longer than `chunkBytes` split into consecutive pieces.
    func split(_ jobs: [ReadJob]) -> [ReadJob] {
        guard chunkBytes > 0, jobs.contains(where: { $0.extent.length > chunkBytes }) else { return jobs }
        var out: [ReadJob] = []
        for job in jobs {
            var done = 0
            while done < job.extent.length {
                let n = min(chunkBytes, job.extent.length - done)
                out.append(ReadJob(
                    dst: job.dst, dstOffset: job.dstOffset + done,
                    extent: UnitExtent(
                        path: job.extent.path, fileOffset: job.extent.fileOffset + UInt64(done),
                        length: n)))
                done += n
            }
        }
        return out
    }

    /// Next layer index; each streaming MoE layer registers once.
    public func registerLayer() -> Int {
        lock.withLock {
            defer { layerCount += 1 }
            return layerCount
        }
    }

    public func snapshot() -> Metrics { lock.withLock { metrics } }

    public func recordResolution(hits: Int, misses: Int) {
        lock.withLock {
            metrics.hits += hits
            metrics.misses += misses
        }
    }

    /// Read every job, `queueDepth` at a time. Errors go to the active
    /// `SSDStreamingErrorLatch`, as with the other SSD read paths.
    @discardableResult
    public func read(_ jobs: [ReadJob], units: Int, kind: ReadKind) -> SSDStreamingError? {
        guard !jobs.isEmpty else { return nil }
        let jobs = split(jobs)
        let errState = ThreadSafeError()
        let noCache = self.noCache
        let workers = queueDepth > 0 ? min(queueDepth, jobs.count) : jobs.count
        let start = DispatchTime.now().uptimeNanoseconds
        DispatchQueue.concurrentPerform(iterations: workers) { worker in
            for i in stride(from: worker, to: jobs.count, by: workers) {
                errState.catchError {
                    let job = jobs[i]
                    MLXFast.preadRangeInto(
                        job.dst, dstOffset: job.dstOffset, path: job.extent.path,
                        fileOffset: job.extent.fileOffset, length: job.extent.length,
                        noCache: noCache)
                }
            }
        }
        let elapsed = DispatchTime.now().uptimeNanoseconds - start
        let bytes = jobs.reduce(UInt64(0)) { $0 + UInt64($1.extent.length) }
        lock.withLock {
            metrics.bytesRead += bytes
            metrics.readNanos += elapsed
            metrics.readBatches += 1
            if kind == .prefetch { metrics.prefetchedUnits += units }
        }
        return errState.check()
    }
}
