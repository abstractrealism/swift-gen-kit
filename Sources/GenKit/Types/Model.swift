import Foundation
import SharedKit

public struct Model: Identifiable, Codable, Hashable, Sendable {
    public var id: String
    public var family: String?
    public var name: String?
    public var owner: String
    public var contextWindow: Int?
    public var maxOutput: Int?
    public var trainingCutoff: Date?

    /// What this model can do beyond plain completion, where the service says.
    ///
    /// `nil` means unknown, not incapable — most services don't report this,
    /// so a caller deciding whether to offer a feature should treat nil as
    /// "go ahead" rather than hide it.
    public var capabilities: Set<Capability>?

    public enum Capability: String, Codable, Hashable, Sendable, CaseIterable {
        case completion
        case tools
        case thinking
        case vision
        case insert
    }

    public init(id: String, family: String? = nil, name: String? = nil, owner: String, contextWindow: Int? = nil,
                maxOutput: Int? = nil, trainingCutoff: Date? = nil, capabilities: Set<Capability>? = nil) {
        self.id = id
        self.family = family
        self.name = name
        self.owner = owner
        self.contextWindow = contextWindow
        self.maxOutput = maxOutput
        self.trainingCutoff = trainingCutoff
        self.capabilities = capabilities
    }

    /// Whether this model supports something, when nothing is known counting
    /// as yes. See `capabilities`.
    public func supports(_ capability: Capability) -> Bool {
        capabilities?.contains(capability) ?? true
    }
}
