// A convenient structure used to manage service credentials, available models and preferred models to use within
// an application or implementation.

import Foundation

public struct Service: Identifiable, Codable, Hashable, Sendable {
    public var id: String
    public var kind: Kind
    public var name: String
    public var host: String
    public var token: String
    public var models: [Model]
    public var status: Status

    public var preferredChatModel: String?
    public var preferredImageModel: String?
    public var preferredEmbeddingModel: String?
    public var preferredTranscriptionModel: String?
    public var preferredSpeechModel: String?
    public var preferredSummarizationModel: String?

    public enum Kind: String, Codable, Identifiable, CaseIterable, Hashable, Sendable {
        case anthropic
        case deepseek
        case elevenLabs
        case fal
        case grok
        case groq
        case mistral
        case ollama
        case openAI
        case openRouter

        public var id: String { rawValue }
    }

    public enum Status: String, Codable, Hashable, Sendable {
        case ready
        case missingHost
        case missingToken
        case unknown
    }

    public init(id: String = UUID().uuidString, kind: Kind, name: String, host: String = "", token: String = "",
                models: [Model] = [], preferredChatModel: String? = nil, preferredImageModel: String? = nil,
                preferredEmbeddingModel: String? = nil, preferredTranscriptionModel: String? = nil,
                preferredSpeechModel: String? = nil, preferredSummarizationModel: String? = nil) {
        self.id = id
        self.kind = kind
        self.name = name
        self.host = host
        self.token = token
        self.models = models
        self.status = .unknown

        self.preferredChatModel = preferredChatModel
        self.preferredImageModel = preferredImageModel
        self.preferredEmbeddingModel = preferredEmbeddingModel
        self.preferredTranscriptionModel = preferredTranscriptionModel
        self.preferredSpeechModel = preferredSpeechModel
        self.preferredSummarizationModel = preferredSummarizationModel
    }

    var hostURL: URL? {
        URL(string: host)
    }
}

extension Service {

    public func anthropic(session: URLSession?) -> AnthropicService {
        AnthropicService(session: session, host: hostURL, apiKey: token)
    }

    public func deepseek(session: URLSession?) -> OpenAIService {
        OpenAIService(session: session, host: hostURL, apiKey: token)
    }

    public func elevenLabs(session: URLSession?) -> ElevenLabsService {
        ElevenLabsService(session: session, apiKey: token)
    }

    public func fal(session: URLSession?) -> FalService {
        FalService(session: session, host: hostURL, apiKey: token)
    }

    public func grok(session: URLSession?) -> OpenAIService {
        OpenAIService(session: session, host: hostURL, apiKey: token)
    }

    public func groq(session: URLSession?) -> OpenAIService {
        OpenAIService(session: session, host: hostURL, apiKey: token)
    }

    public func mistral(session: URLSession?) -> MistralService {
        MistralService(session: session, host: hostURL, apiKey: token)
    }

    public func ollama(session: URLSession?) -> OllamaService {
        OllamaService(session: session, host: hostURL)
    }

    public func openAI(session: URLSession?) -> OpenAIService {
        OpenAIService(session: session, host: hostURL, apiKey: token)
    }

    public func openRouter(session: URLSession?) -> OpenAIService {
        OpenAIService(session: session, host: hostURL, apiKey: token)
    }

    /// Whether a model looks like one worth offering for a conversation.
    ///
    /// A guess from the name, and only ever a guess: OpenAI's model list
    /// reports an id and an owner and nothing about what a model does, so
    /// there is no capability to consult. It keeps everything it has ever
    /// served — around 130 entries, most of them video, image, audio,
    /// embedding or moderation models, or completion models from before chat
    /// existed — and a picker listing all of them is a picker nobody can use.
    ///
    /// Matched on hyphen-separated components rather than substrings, so
    /// `text-ada-001` is caught without `ada` matching something that merely
    /// contains those letters.
    ///
    /// Services that list only what's installed, or only a dozen models,
    /// don't have this problem and everything they offer counts.
    public func isLikelyChatModel(modelID: String) -> Bool {
        switch kind {
        case .openAI:
            let notChat: Set<String> = [
                // Other modalities.
                "sora", "dall", "tts", "whisper", "transcribe", "audio",
                "realtime", "speech", "image", "video",
                // Not conversation. Note "search" is deliberately absent:
                // the search-preview models are chat models that look things
                // up, and the legacy `text-search-…` ones are caught by their
                // base model's name anyway.
                "embedding", "embeddings", "moderation", "similarity",
                // Completion models from before chat, and harnesses that
                // aren't a chat endpoint.
                "davinci", "babbage", "ada", "curie", "instruct",
                "computer", "codex", "edit",
            ]
            let components = modelID.lowercased().split(separator: "-").map(String.init)
            guard Set(components).isDisjoint(with: notChat) else { return false }

            // Pinned snapshots. Most of what OpenAI keeps is dated copies of
            // models that also exist undated — `gpt-4o-2024-05-13` beside
            // `gpt-4o`, `gpt-4-0613` beside `gpt-4` — and they are the bulk of
            // the list rather than a handful. All chat models, so the check
            // above keeps every one of them, which left a picker no shorter
            // than before.
            //
            // The undated name is the one to offer: it follows the current
            // snapshot, which is what somebody picking a model from a menu
            // means. Anyone wanting a version pinned can say so.
            let isSnapshot = components.contains { component in
                // A date and nothing else: 0613, 2024, 20251001. Not `4`,
                // not `3.5`, not `32k`, not `16k`.
                (component.count == 4 || component.count == 6 || component.count == 8)
                    && component.allSatisfy(\.isNumber)
            }
            return !isSnapshot
        default:
            return true
        }
    }

    /// Whether a model of this service can be told not to reason.
    ///
    /// Nil where nothing is known — an unrecognised model, or a service that
    /// doesn't distinguish. Only `false` is a fact worth acting on.
    public func canDisableThinking(modelID: String) -> Bool? {
        switch kind {
        case .anthropic:
            AnthropicService.canDisableThinking(modelID: modelID)
        default:
            nil
        }
    }

    public func modelService(session: URLSession?) -> ModelService {
        switch kind {
        case .anthropic:
            anthropic(session: session)
        case .deepseek:
            deepseek(session: session)
        case .elevenLabs:
            elevenLabs(session: session)
        case .fal:
            fal(session: session)
        case .grok:
            grok(session: session)
        case .groq:
            groq(session: session)
        case .mistral:
            mistral(session: session)
        case .ollama:
            ollama(session: session)
        case .openAI:
            openAI(session: session)
        case .openRouter:
            openRouter(session: session)
        }
    }

    public func chatService(session: URLSession?) throws -> ChatService {
        guard preferredChatModel != nil else {
            throw ServiceError.missingService
        }
        switch kind {
        case .anthropic:
            return anthropic(session: session)
        case .deepseek:
            return deepseek(session: session)
        case .grok:
            return grok(session: session)
        case .groq:
            return groq(session: session)
        case .mistral:
            return mistral(session: session)
        case .ollama:
            return ollama(session: session)
        case .openAI:
            return openAI(session: session)
        case .openRouter:
            return openRouter(session: session)
        case .elevenLabs, .fal:
            throw ServiceError.unsupportedService
        }
    }

    public func imageService(session: URLSession?) throws -> ImageService {
        guard preferredImageModel != nil else {
            throw ServiceError.missingService
        }
        switch kind {
        case .fal:
            return fal(session: session)
        case .openAI:
            return openAI(session: session)
        case .anthropic, .deepseek, .elevenLabs, .grok, .groq, .mistral, .ollama, .openRouter:
            throw ServiceError.unsupportedService
        }
    }

    public func embeddingService(session: URLSession?) throws -> EmbeddingService {
        guard preferredEmbeddingModel != nil else {
            throw ServiceError.missingService
        }
        switch kind {
        case .groq:
            return groq(session: session)
        case .grok:
            return grok(session: session)
        case .mistral:
            return mistral(session: session)
        case .ollama:
            return ollama(session: session)
        case .openAI:
            return openAI(session: session)
        case .anthropic, .deepseek, .elevenLabs, .fal, .openRouter:
            throw ServiceError.unsupportedService
        }
    }

    public func transcriptionService(session: URLSession?) throws -> TranscriptionService {
        guard preferredTranscriptionModel != nil else {
            throw ServiceError.missingService
        }
        switch kind {
        case .groq:
            return groq(session: session)
        case .openAI:
            return openAI(session: session)
        case .anthropic, .deepseek, .elevenLabs, .fal, .grok, .mistral, .ollama, .openRouter:
            throw ServiceError.unsupportedService
        }
    }

    public func speechService(session: URLSession?) throws -> SpeechService {
        guard preferredSpeechModel != nil else {
            throw ServiceError.missingService
        }
        switch kind {
        case .elevenLabs:
            return elevenLabs(session: session)
        case .fal:
            return fal(session: session)
        case .groq:
            return groq(session: session)
        case .openAI:
            return openAI(session: session)
        case .anthropic, .deepseek, .grok, .mistral, .ollama, .openRouter:
            throw ServiceError.unsupportedService
        }
    }

    public func summarizationService(session: URLSession?) throws -> ChatService {
        guard preferredChatModel != nil else {
            throw ServiceError.missingService
        }
        switch kind {
        case .anthropic:
            return anthropic(session: session)
        case .deepseek:
            return deepseek(session: session)
        case .grok:
            return grok(session: session)
        case .groq:
            return groq(session: session)
        case .mistral:
            return mistral(session: session)
        case .ollama:
            return ollama(session: session)
        case .openAI:
            return openAI(session: session)
        case .openRouter:
            return openRouter(session: session)
        case .elevenLabs, .fal:
            throw ServiceError.unsupportedService
        }
    }
}

extension Service {

    public var supportsChats: Bool {
        preferredChatModel != nil
    }

    public var supportsImages: Bool {
        preferredImageModel != nil
    }

    public var supportsEmbeddings: Bool {
        preferredEmbeddingModel != nil
    }

    public var supportsTranscriptions: Bool {
        preferredTranscriptionModel != nil
    }

    public var supportsSpeech: Bool {
        preferredSpeechModel != nil
    }

    public var supportsSummarization: Bool {
        preferredSummarizationModel != nil
    }
}

extension Service {

    public mutating func applyPreferredModels(_ service: Service) {
        self.preferredChatModel = service.preferredChatModel
        self.preferredImageModel = service.preferredImageModel
        self.preferredEmbeddingModel = service.preferredEmbeddingModel
        self.preferredTranscriptionModel = service.preferredTranscriptionModel
        self.preferredSpeechModel = service.preferredSpeechModel
        self.preferredSummarizationModel = service.preferredSummarizationModel
    }
}
