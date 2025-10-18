import Foundation
import Hub
import os

// MARK: - CollectionResponse

struct CollectionResponse: Codable {
    struct CollectionItem: Codable {
        let id: String
        let type: String
    }

    let slug: String
    let title: String
    let description: String?
    let items: [CollectionItem]?
}

// MARK: - HuggingFaceService

@MainActor class HuggingFaceService {
    // MARK: Lifecycle

    init(session: URLSession = .shared) {
        self.session = session
    }

    // MARK: Internal

    static let shared = HuggingFaceService()

    func fetchCollections(forceReload: Bool = false) async throws -> [HuggingFaceCollection] {
        if !forceReload, !cachedCollections.isEmpty {
            return cachedCollections
        }

        var allCollections: [HuggingFaceCollection] = []
        var offset = 0

        repeat {
            let url = try makeURL(
                path: "collections",
                queryItems: [
                    URLQueryItem(name: "owner", value: mlxCommunity),
                    URLQueryItem(name: "limit", value: "\(collectionPageSize)"),
                    URLQueryItem(name: "offset", value: "\(offset)")
                ]
            )

            AppLogger.network.debug("Fetching collections at offset \(offset)")

            let (data, _) = try await session.data(from: url)
            let responses = try decoder.decode([CollectionResponse].self, from: data)

            guard !responses.isEmpty else {
                break
            }

            let collections = responses.map { response in
                HuggingFaceCollection(
                    slug: response.slug,
                    title: response.title,
                    description: response.description,
                    modelIds: response.items?
                        .filter { $0.type == "model" }
                        .map(\.id) ?? []
                )
            }

            allCollections.append(contentsOf: collections)
            offset += collectionPageSize

            AppLogger.network.debug("Loaded \(collections.count) collections (total: \(allCollections.count))")
        } while allCollections.count % collectionPageSize == 0

        cachedCollections = allCollections
        return allCollections
    }

    func fetchModels(collection: String? = nil) async throws -> [MLXModel] {
        var allModels: [MLXModel] = []
        var seenModelIds = Set<String>()
        var nextCursor: String?
        var pageCount = 0

        let collectionModelIds = try await collectionModelIdentifiers(for: collection)

        repeat {
            pageCount += 1
            var queryItems = [
                URLQueryItem(name: "author", value: mlxCommunity),
                URLQueryItem(name: "sort", value: "downloads"),
                URLQueryItem(name: "direction", value: "-1"),
                URLQueryItem(name: "limit", value: "\(modelPageSize)"),
                URLQueryItem(name: "full", value: "true")
            ]

            if let cursor = nextCursor {
                queryItems.append(URLQueryItem(name: "cursor", value: cursor))
            }

            let url = try makeURL(path: "models", queryItems: queryItems)
            AppLogger.network.debug("Fetching models page \(pageCount) (cursor: \(nextCursor != nil ? "yes" : "no"))")

            let (data, response) = try await session.data(from: url)

            guard let httpResponse = response as? HTTPURLResponse else {
                throw APIError.invalidResponse
            }

            AppLogger.network.debug("Received status code \(httpResponse.statusCode, privacy: .public)")

            nextCursor = parseNextCursor(from: httpResponse)

            let responses: [HuggingFaceResponse]
            do {
                responses = try decoder.decode([HuggingFaceResponse].self, from: data)
                AppLogger.network.debug("Decoded \(responses.count) models on page \(pageCount)")
            } catch {
                AppLogger.network.error("Failed decoding models: \(error.localizedDescription, privacy: .public)")
                throw error
            }

            guard !responses.isEmpty else {
                break
            }

            let models = responses.compactMap { response -> MLXModel? in
                let modelId = response.modelId ?? response.id

                guard !seenModelIds.contains(modelId) else {
                    return nil
                }

                guard let lastModified = isoDateFormatter.date(from: response.lastModified) else {
                    AppLogger.network.error("Invalid date for model \(modelId, privacy: .public)")
                    return nil
                }

                seenModelIds.insert(modelId)

                return MLXModel(
                    id: modelId,
                    name: modelId,
                    author: response.author,
                    downloads: response.downloads,
                    likes: response.likes,
                    lastModified: lastModified,
                    size: nil,
                    collections: nil,
                    tags: response.tags
                )
            }

            let textGenerationModels = models.filter(\.isTextGeneration)
            allModels.append(contentsOf: textGenerationModels)

            AppLogger.network.debug("Accumulated \(allModels.count) text generation models so far")

        } while nextCursor != nil && pageCount < maxModelPages

        if pageCount >= maxModelPages, nextCursor != nil {
            AppLogger.network.error("Hit page limit while fetching models")
        }

        if let modelIds = collectionModelIds, !modelIds.isEmpty {
            allModels = allModels.filter { modelIds.contains($0.id) }
        }

        AppLogger.network.debug("Returning \(allModels.count) models")
        return allModels
    }

    // MARK: Private

    private enum APIError: Error {
        case invalidURL
        case invalidResponse
    }

    private let session: URLSession
    private let baseURL = URL(string: "https://huggingface.co/api")!
    private let mlxCommunity = "mlx-community"
    private let decoder = JSONDecoder()
    private let isoDateFormatter: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()

    private let collectionPageSize = 100
    private let modelPageSize = 500
    private let maxModelPages = 10

    private var cachedCollections: [HuggingFaceCollection] = []

    private func makeURL(path: String, queryItems: [URLQueryItem]) throws -> URL {
        var components = URLComponents(url: baseURL.appendingPathComponent(path), resolvingAgainstBaseURL: false)
        components?.queryItems = queryItems
        guard let url = components?.url else {
            throw APIError.invalidURL
        }
        return url
    }

    private func parseNextCursor(from response: HTTPURLResponse) -> String? {
        guard let linkHeader = response.value(forHTTPHeaderField: "link") else {
            return nil
        }

        return linkHeader
            .components(separatedBy: ",")
            .compactMap { component -> String? in
                let parts = component.components(separatedBy: ";")
                guard
                    let urlPart = parts.first?.trimmingCharacters(in: .whitespacesAndNewlines),
                    let relPart = parts.dropFirst().first(where: { $0.contains("rel=\"next\"") }),
                    relPart.contains("rel=\"next\"")
                else {
                    return nil
                }

                guard let urlStart = urlPart.firstIndex(of: "<"),
                      let urlEnd = urlPart.firstIndex(of: ">") else {
                    return nil
                }

                let urlString = urlPart[urlStart...urlEnd].trimmingCharacters(in: CharacterSet(charactersIn: "<>"))

                guard let components = URLComponents(string: String(urlString)) else {
                    return nil
                }

                return components.queryItems?.first(where: { $0.name == "cursor" })?.value
            }
            .first
    }

    private func collectionModelIdentifiers(for name: String?) async throws -> Set<String>? {
        guard let name else { return nil }

        let collections = try await fetchCollections()
        guard let collection = collections.first(where: { $0.displayName == name || $0.slug == name }) else {
            AppLogger.network.error("No collection found for \(name, privacy: .public)")
            return nil
        }

        return Set(collection.modelIds)
    }
}
