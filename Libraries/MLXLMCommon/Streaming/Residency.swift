// Residency.swift — which experts stay in a layer's resident slots, and which
// experts to fetch ahead of routing.

import Foundation

/// Picks the slot to fill next in a layer's slot table.
public protocol ResidencyPolicy: AnyObject {
    /// The slot to (re)use, or nil when every slot is in `claimed`.
    func victim(in table: ExpertSlotTable, excluding claimed: Set<Int>) -> Int?
}

/// First empty slot, else the least recently used one. Ties go to the lowest slot.
public final class LRUResidencyPolicy: ResidencyPolicy {
    public init() {}

    public func victim(in table: ExpertSlotTable, excluding claimed: Set<Int>) -> Int? {
        for s in 0 ..< table.capacity where !claimed.contains(s) && table.occupant[s] == nil {
            return s
        }
        var best: Int?
        var bestTick = Int.max
        for s in 0 ..< table.capacity where !claimed.contains(s) && table.lastUsed[s] < bestTick {
            bestTick = table.lastUsed[s]
            best = s
        }
        return best
    }
}

/// Predicts the experts a layer will route to before its router runs.
public protocol UnitPredictor: AnyObject {
    func predict() -> [Int]
    func observe(_ experts: [Int])
}

/// Predicts that the next token routes to the same experts as the last one.
public final class PreviousTokenPredictor: UnitPredictor {
    private var last: [Int] = []

    public init() {}

    public func predict() -> [Int] { last }
    public func observe(_ experts: [Int]) { last = experts }
}

/// Slot bookkeeping for one MoE layer: which expert occupies each resident slot.
public final class ExpertSlotTable {
    public let capacity: Int
    public private(set) var occupant: [Int?]
    public private(set) var lastUsed: [Int]
    /// Token counter for this layer, advanced once per decode call.
    public private(set) var tick = 0

    let policy: ResidencyPolicy
    let predictor: UnitPredictor

    public init(capacity: Int, policy: ResidencyPolicy, predictor: UnitPredictor) {
        self.capacity = capacity
        self.occupant = Array(repeating: nil, count: capacity)
        self.lastUsed = Array(repeating: 0, count: capacity)
        self.policy = policy
        self.predictor = predictor
    }

    public func beginToken() { tick += 1 }

    private func residentSlots() -> [Int: Int] {
        var map = [Int: Int]()
        for (slot, expert) in occupant.enumerated() {
            if let expert { map[expert] = slot }
        }
        return map
    }

    /// Claim slots for predicted experts that are not resident. Claimed slots
    /// keep their `lastUsed`, so an unused prediction is the first to go.
    public func claimPredicted() -> [(slot: Int, expert: Int)] {
        claim(predictor.predict())
    }

    /// Claim slots for `experts` that are not resident, for a read ahead of routing.
    public func claim(_ experts: [Int]) -> [(slot: Int, expert: Int)] {
        var resident = residentSlots()
        var claimed = Set<Int>()
        var targets: [(slot: Int, expert: Int)] = []
        for expert in experts where resident[expert] == nil {
            guard let slot = policy.victim(in: self, excluding: claimed) else { break }
            if let old = occupant[slot] { resident.removeValue(forKey: old) }
            occupant[slot] = expert
            resident[expert] = slot
            claimed.insert(slot)
            targets.append((slot, expert))
        }
        return targets
    }

    /// Empty `slots`, e.g. after a failed read left their contents undefined.
    public func release(_ slots: [Int]) {
        for s in slots { occupant[s] = nil }
    }

    /// Map each routed expert to a slot. Returns nil, with the table unchanged,
    /// when the slots run out; otherwise the slot per expert and the misses that
    /// must be read.
    /// `reserved` slots are never chosen as victims (e.g. a read is still landing in them).
    public func resolve(
        _ experts: [Int], reserved: Set<Int> = []
    ) -> (slots: [Int], misses: [(slot: Int, expert: Int)])? {
        let savedOccupant = occupant
        let savedLastUsed = lastUsed
        var resident = residentSlots()
        var slots = [Int](repeating: -1, count: experts.count)
        var misses: [(slot: Int, expert: Int)] = []
        // Hits first, so a miss can never evict an expert routed later in the list
        // (prefetched slots keep an old lastUsed and would otherwise go first).
        var claimed = reserved
        for (n, expert) in experts.enumerated() {
            if let slot = resident[expert], !claimed.contains(slot) {
                slots[n] = slot
                claimed.insert(slot)
                lastUsed[slot] = tick
            }
        }
        for (n, expert) in experts.enumerated() where slots[n] < 0 {
            guard let slot = policy.victim(in: self, excluding: claimed) else {
                // A partial update would leave slots naming experts that were never read.
                occupant = savedOccupant
                lastUsed = savedLastUsed
                return nil
            }
            if let old = occupant[slot] { resident.removeValue(forKey: old) }
            occupant[slot] = expert
            resident[expert] = slot
            lastUsed[slot] = tick
            claimed.insert(slot)
            slots[n] = slot
            misses.append((slot, expert))
        }
        predictor.observe(experts)
        return (slots, misses)
    }
}
