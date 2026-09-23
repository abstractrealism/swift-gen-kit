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
        OpenAIService(session: session, host: hostURL, apiKey: token, dialect: .deepSeek)
    }

    public func elevenLabs(session: URLSession?) -> ElevenLabsService {
        ElevenLabsService(session: session, apiKey: token)
    }

    public func fal(session: URLSession?) -> FalService {
        FalService(session: session, host: hostURL, apiKey: token)
    }

    public func grok(session: URLSession?) -> OpenAIService {
        OpenAIService(session: session, host: hostURL, apiKey: token, dialect: .grok)
    }

    public func groq(session: URLSession?) -> OpenAIService {
        OpenAIService(session: session, host: hostURL, apiKey: token, dialect: .groq)
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
        OpenAIService(session: session, host: hostURL, apiKey: token, dialect: .openRouter)
    }

    /// Whether this model reaches the web by itself, with nothing asked of it.
    ///
    /// Server-side search is normally opt-in: a provider offers it as a tool
    /// and it happens only when that tool is sent. These models are the
    /// exception — searching is part of what the model *is*, not something
    /// switched on, so it happens whether or not a caller offers any tools and
    /// can't be switched off.
    ///
    /// Worth saying out loud, because an app with its own search tool
    /// otherwise presents a switch that appears to govern something it has no
    /// say in.
    public func hasBuiltInWebSearch(modelID: String) -> Bool {
        switch kind {
        case .openAI:
            // A `search` component on a GPT model: the preview pair, and
            // gpt-5-search-api after them. Not the legacy `text-search-…`
            // embeddings, which aren't chat models at all.
            let id = modelID.lowercased()
            return id.hasPrefix("gpt-") && id.split(separator: "-").contains("search")
        default:
            return false
        }
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
        // A guess only where there's nothing better. Where the service said
        // what a model is for, that settles it — a transcription model is
        // not a chat model whose name happens to look wrong, and a chat
        // model with an odd name is still a chat model.
        if let capabilities = models.first(where: { $0.id == modelID })?.capabilities {
            return capabilities.contains(.completion)
        }
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
        case .openAI:
            OpenAIService.canDisableThinking(modelID: modelID, dialect: .openAI)
        case .deepseek:
            OpenAIService.canDisableThinking(modelID: modelID, dialect: .deepSeek)
        case .groq:
            OpenAIService.canDisableThinking(modelID: modelID, dialect: .groq)
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

    // A client doesn't depend on a preferred model, so none of these ask for
    // one. They used to — each began by refusing unless its preference was
    // set — which made the preference a condition of using the service at
    // all, rather than the default a caller falls back on. A caller that has
    // chosen a model itself, as a conversation pinned to one does, was
    // refused a client for a service it had every right to use, and refused
    // with "missing service", the one thing that wasn't missing. The model is
    // the caller's to supply in each request; whether a service does this
    // kind of work at all is what the switches below answer.

    public func chatService(session: URLSession?) throws -> ChatService {
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
