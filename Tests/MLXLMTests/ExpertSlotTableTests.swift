import Foundation
import XCTest

@testable import MLXLMCommon

final class ExpertSlotTableTests: XCTestCase {
    private func makeTable(_ capacity: Int) -> ExpertSlotTable {
        ExpertSlotTable(
            capacity: capacity, policy: LRUResidencyPolicy(), predictor: PreviousTokenPredictor())
    }

    func testFillsEmptySlotsInOrderThenEvictsLeastRecent() {
        let table = makeTable(4)
        table.beginToken()
        XCTAssertEqual(table.resolve([10, 11])?.slots, [0, 1])
        table.beginToken()
        XCTAssertEqual(table.resolve([12, 13])?.slots, [2, 3])
        table.beginToken()
        let r = table.resolve([11, 14])
        XCTAssertEqual(r?.slots, [1, 0], "14 evicts slot 0, the oldest")
        XCTAssertEqual(r?.misses.map(\.expert), [14])
        XCTAssertEqual(table.occupant, [14, 11, 12, 13])
    }

    func testLRUTiesGoToLowestSlot() {
        let table = makeTable(3)
        table.beginToken()
        _ = table.resolve([1, 2, 3])
        table.beginToken()
        XCTAssertEqual(table.resolve([4])?.slots, [0])
    }

    private final class FixedPredictor: UnitPredictor {
        var next: [Int] = []
        func predict() -> [Int] { next }
        func observe(_ experts: [Int]) {}
    }

    func testPreviousTokenPredictionIsAlreadyResident() {
        let table = makeTable(4)
        table.beginToken()
        _ = table.resolve([1, 2])
        table.beginToken()
        XCTAssertTrue(table.claimPredicted().isEmpty)
    }

    func testPredictionClaimsSlotsWithoutRefreshingThem() {
        let predictor = FixedPredictor()
        let table = ExpertSlotTable(capacity: 3, policy: LRUResidencyPolicy(), predictor: predictor)
        table.beginToken()
        _ = table.resolve([1])
        table.beginToken()
        _ = table.resolve([2, 3])
        table.beginToken()
        predictor.next = [2, 9]
        let claimed = table.claimPredicted()
        XCTAssertEqual(claimed.map(\.slot), [0], "9 replaces the oldest slot")
        XCTAssertEqual(claimed.map(\.expert), [9])
        XCTAssertEqual(table.lastUsed[0], 1, "a claim does not count as a use")
        XCTAssertEqual(table.resolve([4])?.slots, [0], "the unused prediction is evicted first")
    }

    func testFailedResolveLeavesTableUnchanged() {
        let table = makeTable(2)
        table.beginToken()
        _ = table.resolve([1, 2])
        let occupant = table.occupant
        let lastUsed = table.lastUsed
        table.beginToken()
        XCTAssertNil(table.resolve([3, 4, 5]))
        XCTAssertEqual(table.occupant, occupant)
        XCTAssertEqual(table.lastUsed, lastUsed)
        XCTAssertEqual(table.resolve([1])?.misses.count, 0, "1 is still resident")
    }

    func testSafetensorsSlabMatchesPerExpertOffsets() throws {
        let tensor = SafetensorsLayout.Tensor(dataStart: 1000, byteLength: 4 * 256, shape: [4, 8, 8])
        let e = try XCTUnwrap(tensor.slab(2, path: "f"))
        XCTAssertEqual(e.fileOffset, 1000 + 2 * 256)
        XCTAssertEqual(e.length, 256)
        XCTAssertNil(tensor.slab(4, path: "f"))
    }

    func testMissNeverEvictsAPrefetchedExpertRoutedLaterInTheList() {
        let table = makeTable(3)
        table.beginToken()
        _ = table.resolve([1, 2, 3])
        // Prefetch 9 into the oldest slot; it keeps that slot's old lastUsed.
        XCTAssertEqual(table.claim([9]).map(\.slot), [0])
        table.beginToken()
        // 7 misses before 9 is reached; it must not take 9's slot.
        let r = table.resolve([7, 9])
        XCTAssertEqual(r?.misses.map(\.expert), [7])
        XCTAssertEqual(r?.slots[1], 0, "9 stays a hit in its prefetched slot")
    }

    func testClaimSkipsResidentExperts() {
        let table = makeTable(4)
        table.beginToken()
        _ = table.resolve([1, 2])
        let claimed = table.claim([2, 5])
        XCTAssertEqual(claimed.map(\.expert), [5])
        XCTAssertEqual(table.occupant.compactMap { $0 }.sorted(), [1, 2, 5])
    }

    func testReleasedSlotsBecomeMisses() {
        let table = makeTable(2)
        table.beginToken()
        let claimed = table.claim([4])
        table.release(claimed.map(\.slot))
        table.beginToken()
        XCTAssertEqual(table.resolve([4])?.misses.map(\.expert), [4])
    }
}
