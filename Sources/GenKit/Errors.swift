import Foundation

enum ServiceError: Error {
    case missingService
    case missingServiceHost
    case missingServiceToken
    case missingServiceHostAndToken
    case missingImageData
    case missingUserMessage
    case unsupportedResponseFormat
    case unsupportedService
    case notImplemented
}

// Said in words, because these reach people. Thrown out of public
// functions from an internal type, so a caller can neither match the cases
// nor do better than print what it was given — and printed, a bare case
// reads as `unsupportedService`, which names nothing anybody can act on.
// `CustomStringConvertible` covers the caller that interpolates the error
// and `LocalizedError` the one that asks for its `localizedDescription`,
// which otherwise reads "The operation couldn't be completed".
extension ServiceError: LocalizedError, CustomStringConvertible {

    var errorDescription: String? { description }

    var description: String {
        switch self {
        case .missingService:
            "No service is set up for this."
        case .missingServiceHost:
            "The service has no address."
        case .missingServiceToken:
            "The service has no API key."
        case .missingServiceHostAndToken:
            "The service has no address and no API key."
        case .missingImageData:
            "The service returned an image with no data in it."
        case .missingUserMessage:
            "There is no message to send."
        case .unsupportedResponseFormat:
            "The service doesn't offer that response format."
        case .unsupportedService:
            "This service doesn't do that kind of work — it has no models for it."
        case .notImplemented:
            "Not implemented yet."
        }
    }
}
