import Foundation
import Observation

// MARK: - MLXModel

struct MLXModel: Identifiable, Codable, Hashable {
    let id: String
    let name: String
    let author: String
    let downloads: Int
    let likes: Int
    let lastModified: Date
    let size: Int64?
    let collections: [String]?
    let tags: [String]?

    var displayName: String {
        name.replacingOccurrences(of: author + "/", with: "")
    }

    var sizeString: String {
        guard let size else { return "Unknown" }
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter.string(fromByteCount: size)
    }

    var collectionNames: [String] {
        collections?.compactMap { collection in
            // Extract clean collection name from slug
            let slug = collection.components(separatedBy: "/").last ?? collection
            return slug.replacingOccurrences(of: "-", with: " ")
                .capitalized
        } ?? []
    }

    var isTextGeneration: Bool {
        guard let tags else { return true } // Default to true if no tags

        // Include if it has text-generation tag
        if tags.contains("text-generation") {
            return true
        }

        // Exclude specific non-text-generation tags
        let excludedTags = [
            "automatic-speech-recognition",
            "text-to-speech",
            "audio",
            "whisper",
            "feature-extraction",
            "sentence-similarity",
            "image-text-to-text",
            "vision",
            "multimodal"
        ]

        return !tags.contains(where: { excludedTags.contains($0) })
    }
}

// MARK: - HuggingFaceResponse

struct HuggingFaceResponse: Codable {
    struct Sibling: Codable {
        let rfilename: String
        let size: Int64?
    }

    let id: String
    let modelId: String?
    let author: String
    let downloads: Int
    let likes: Int
    let lastModified: String
    let siblings: [Sibling]?
    let tags: [String]?
}

// MARK: - HuggingFaceCollection

struct HuggingFaceCollection: Codable {
    let slug: String
    let title: String
    let description: String?
    let modelIds: [String]

    var displayName: String {
        title.isEmpty ? slug.split(separator: "/").last.map { String($0).replacingOccurrences(of: "-", with: " ").capitalized } ?? slug : title
    }
}

// MARK: - ModelStatus

enum ModelStatus: Equatable {
    case available
    case downloading(progress: Double)
    case installed
    case error(String)
}

// MARK: - DownloadedModel

@Observable class DownloadedModel: Identifiable, Hashable {
    // MARK: Lifecycle

    init(
        id: String,
        model: MLXModel,
        localPath: URL,
        status: ModelStatus = .installed,
        totalSize: Int64? = nil
    ) {
        self.id = id
        self.model = model
        self.localPath = localPath
        self.status = status
        self.totalSize = totalSize
    }

    // MARK: Internal

    let id: String
    let model: MLXModel
    var localPath: URL
    var status: ModelStatus
    var totalSize: Int64?

    var sizeString: String {
        guard let totalSize else { return "Unknown" }
        return Self.sizeFormatter.string(fromByteCount: totalSize)
    }

    // MARK: Private

    private static let sizeFormatter: ByteCountFormatter = {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter
    }()

    static func == (lhs: DownloadedModel, rhs: DownloadedModel) -> Bool {
        lhs.id == rhs.id
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}

// MARK: - ChatMessage

struct ChatMessage: Identifiable {
    enum Role {
        case user
        case assistant
        case system
    }

    let id = UUID()
    let role: Role
    let content: String
    let timestamp = Date()
}

// MARK: - BenchmarkMetrics

struct BenchmarkMetrics {
    var promptTokens: Int = 0
    var completionTokens: Int = 0
    var tokensPerSecond: Double = 0
    var timeToFirstToken: Double = 0
    var totalTime: Double = 0
    var memoryUsage: UInt64 = 0

    var promptTime: Double {
        guard promptTokens > 0, tokensPerSecond > 0 else { return 0 }
        return Double(promptTokens) / tokensPerSecond
    }

    var millisecondsPerToken: Double {
        guard completionTokens > 0, totalTime > timeToFirstToken else { return 0 }
        let generationTime = totalTime - timeToFirstToken
        return (generationTime * 1000.0) / Double(completionTokens)
    }
}
