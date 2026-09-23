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

    /// The context this model is currently loaded with, where that's knowable
    /// and different from what it was built for.
    ///
    /// A local server decides at load time how much of a model's context to
    /// give it, and routinely gives far less than the maximum — 32,768 against
    /// a built-in 262,144, in one measured case. `contextWindow` is the
    /// ceiling; this is what a request is actually measured against, so it's
    /// the one to show someone filling it up.
    ///
    /// Nil when the model isn't loaded, or the service doesn't say.
    public var loadedContextWindow: Int?

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

    /// How this model wants to be asked to reason, where the service says.
    ///
    /// `capabilities` answers whether a model reasons at all; this answers
    /// how to ask, which is the part that differs and the part a request is
    /// refused over. Both halves are named in the provider's own vocabulary
    /// rather than translated, because an unrecognised value here should be
    /// carried through and left unused rather than quietly mapped onto
    /// something it isn't.
    ///
    /// Nil where the service doesn't report it, which is most of them.
    public struct Reasoning: Codable, Hashable, Sendable {
        /// The effort levels it accepts — Anthropic reports low through max
        /// and differs by model, DeepSeek reports low, high and max.
        public var levels: [String]?

        /// The level it uses when none is asked for, where that's reported.
        public var defaultLevel: String?

        /// The shapes the request may take. Anthropic's `adaptive` and
        /// `enabled` are refused rather than ignored — each is a 400 on a
        /// model that wants the other — which is the whole reason this is
        /// worth carrying.
        public var modes: [String]?

        public init(levels: [String]? = nil, defaultLevel: String? = nil, modes: [String]? = nil) {
            self.levels = levels
            self.defaultLevel = defaultLevel
            self.modes = modes
        }
    }

    public var reasoning: Reasoning?

    public init(id: String, family: String? = nil, name: String? = nil, owner: String, contextWindow: Int? = nil,
                maxOutput: Int? = nil, trainingCutoff: Date? = nil, capabilities: Set<Capability>? = nil,
                loadedContextWindow: Int? = nil, reasoning: Reasoning? = nil) {
        self.id = id
        self.family = family
        self.name = name
        self.owner = owner
        self.contextWindow = contextWindow
        self.maxOutput = maxOutput
        self.trainingCutoff = trainingCutoff
        self.capabilities = capabilities
        self.loadedContextWindow = loadedContextWindow
        self.reasoning = reasoning
    }

    /// Whether this model supports something, when nothing is known counting
    /// as yes. See `capabilities`.
    public func supports(_ capability: Capability) -> Bool {
        capabilities?.contains(capability) ?? true
    }
}
